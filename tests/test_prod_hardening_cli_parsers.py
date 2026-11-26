"""Production hardening tests for parsers, pricing, pytest plugin, CLI, and CI workflows."""

from __future__ import annotations

import base64
import json
import logging
import math
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from rhinotape import Cassette, ModelPrice, TapeMode, tape
from rhinotape.cli import _cassette_stats, build_parser
from rhinotape.cli import main as cli_main
from rhinotape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    Usage,
)
from rhinotape.parsers import _detect_provider, parse_llm_interaction
from rhinotape.pricing import PRICING_FILE_ENV, estimate_cost_usd, load_pricing_file
from rhinotape.pytest_plugin import (
    _REPORT_KEY,
    _VIOLATIONS_KEY,
    _validate_tape_marker_kwargs,
    _write_bytes_atomic,
    pytest_sessionfinish,
)

pytest_plugins = ["pytester"]


# --- 1. parsers.py ---------------------------------------------------------


def test_parsers_malformed_json_payloads_never_crash() -> None:
    malformed_cases: list[tuple[str, Any, dict[str, Any]]] = [
        # OpenAI Chat Completions with non-dict choices, messages, tool_calls, usage, model
        (
            "https://api.openai.com/v1/chat/completions",
            {"model": ["not-a-str"]},
            {
                "model": {"bad": 1},
                "usage": {"prompt_tokens": "bad", "completion_tokens": True, "input_tokens": None},
                "choices": [
                    "not-a-dict",
                    None,
                    {"message": "not-a-dict"},
                    {
                        "message": {
                            "content": [123, {"text": 456}, {"text": "ok"}],
                            "tool_calls": [
                                "bad-tc",
                                {"id": ["bad"], "function": "not-a-dict"},
                                {"id": 99, "function": {"name": ["bad"], "arguments": "{}"}},
                            ],
                        }
                    },
                ],
                "output": ["bad-item", {"type": "reasoning", "summary": ["bad", {"type": "text"}]}],
            },
        ),
        # OpenAI with usage as string/list
        (
            "https://api.openai.com/v1/chat/completions",
            {},
            {"usage": "invalid", "choices": "invalid", "output": 123},
        ),
        # Anthropic JSON malformed
        (
            "https://api.anthropic.com/v1/messages",
            {"model": 123},
            {
                "type": "message",
                "model": ["bad"],
                "usage": {
                    "input_tokens": "NaN",
                    "output_tokens": float("inf"),
                    "cache_read_input_tokens": True,
                    "cache_creation_input_tokens": {"x": 1},
                },
                "content": ["bad", {"type": "tool_use", "name": 123, "input": "x"}],
            },
        ),
        # Gemini JSON malformed
        (
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            {},
            {
                "modelVersion": 42,
                "usageMetadata": {
                    "promptTokenCount": "nope",
                    "candidatesTokenCount": True,
                    "thoughtsTokenCount": {"bad": 1},
                },
                "candidates": [
                    "bad-cand",
                    {"content": "bad-content"},
                    {"content": {"parts": ["bad-part", {"functionCall": "bad"}]}},
                ],
            },
        ),
        # Ollama JSON malformed
        (
            "http://localhost:11434/api/chat",
            {},
            {
                "model": {"bad": True},
                "prompt_eval_count": "not-int",
                "eval_count": False,
                "message": {
                    "content": 123,
                    "thinking": 456,
                    "tool_calls": ["bad", {"function": 1}],
                },
            },
        ),
        # Bedrock JSON malformed
        (
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/claude/converse",
            {"modelId": 999},
            {
                "usage": {"inputTokens": "x", "outputTokens": None},
                "output": {"message": {"content": ["bad", {"toolUse": "bad"}]}},
            },
        ),
        # Cohere JSON malformed
        (
            "https://api.cohere.com/v2/chat",
            {},
            {
                "meta": {"tokens": "bad", "billed_units": {"input_tokens": "nan"}},
                "message": {"content": "bad", "tool_calls": ["bad", {"function": "bad"}]},
                "tool_calls": ["bad", {"name": 123}],
            },
        ),
    ]

    for url, req, resp in malformed_cases:
        summary = parse_llm_interaction(url, req, resp)
        assert isinstance(summary, LLMCallSummary)
        assert isinstance(summary.usage, Usage)
        assert math.isfinite(summary.estimated_cost_usd)


def test_parsers_malformed_sse_and_ndjson_never_crash() -> None:
    # Malformed OpenAI SSE
    oai = parse_llm_interaction(
        url="https://api.openai.com/v1/chat/completions",
        request_body={},
        response_body=None,
        stream_chunks=(
            'data: {"choices": ["bad", {"delta": "bad"}, '
            '{"delta": {"tool_calls": ["bad", {"index": "not-int", "function": "bad"}]}}]}\n\n',
            'data: {"usage": {"prompt_tokens": "bad", "completion_tokens": true}}\n\n',
            "data: [DONE]\n\n",
        ),
        content_type="text/event-stream",
    )
    assert oai.provider == "openai"
    assert oai.is_stream is True

    # Malformed Anthropic SSE
    anth = parse_llm_interaction(
        url="https://api.anthropic.com/v1/messages",
        request_body={},
        response_body=None,
        stream_chunks=(
            'event: message_start\ndata: {"type":"message_start","message":"bad"}\n\n',
            'event: content_block_start\ndata: {"type":"content_block_start","index":"bad",'
            '"content_block":"bad"}\n\n',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":null,'
            '"delta":"bad"}\n\n',
            'event: message_delta\ndata: {"type":"message_delta","usage":"bad"}\n\n',
        ),
        content_type="text/event-stream",
    )
    assert anth.provider == "anthropic"

    # Malformed Gemini SSE
    gem = parse_llm_interaction(
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent",
        request_body={},
        response_body=None,
        stream_chunks=(
            'data: {"usageMetadata":{"promptTokenCount":"x","thoughtsTokenCount":true},'
            '"candidates":["bad",{"content":"bad"}]}\n\n',
        ),
        content_type="text/event-stream",
    )
    assert gem.provider == "gemini"

    # Malformed Cohere SSE
    coh = parse_llm_interaction(
        url="https://api.cohere.com/v2/chat",
        request_body={},
        response_body=None,
        stream_chunks=(
            'event: content-delta\ndata: {"type":"content-delta","delta":"bad"}\n\n',
            'event: tool-call-start\ndata: {"type":"tool-call-start","index":"x",'
            '"delta":{"message":{"tool_calls":"bad"}}}\n\n',
            'event: message-end\ndata: {"type":"message-end","delta":{"usage":"bad"}}\n\n',
        ),
        content_type="text/event-stream",
    )
    assert coh.provider == "cohere"

    # Malformed Ollama NDJSON
    oll = parse_llm_interaction(
        url="http://localhost:11434/api/chat",
        request_body={},
        response_body=None,
        stream_chunks=(
            '{"model":123,"prompt_eval_count":"x","eval_count":true,'
            '"message":{"content":1,"thinking":2,"tool_calls":["bad"]}}\n',
        ),
        content_type="application/x-ndjson",
    )
    assert oll.provider == "ollama"


def test_openai_responses_reasoning_json_and_sse_and_tool_calls() -> None:
    # JSON reasoning item with summary_text and text
    json_summary = parse_llm_interaction(
        url="https://api.openai.com/v1/responses",
        request_body={"model": "o3"},
        response_body={
            "model": "o3",
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "Analyze the problem."},
                        {"type": "text", "text": "Verify the math."},
                        {"type": "other", "text": "ignored"},
                    ],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Answer is 42."}],
                },
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 60,
                "output_tokens_details": {"reasoning_tokens": 40},
            },
        },
    )
    assert json_summary.thinking == "Analyze the problem.\nVerify the math."
    assert json_summary.text == "Answer is 42."
    assert json_summary.usage.reasoning_tokens == 40

    # SSE reasoning deltas preserved across response.completed when completed omits reasoning,
    # plus response.output_item.added tracking for function_call_arguments.done without name
    sse_summary = parse_llm_interaction(
        url="https://api.openai.com/v1/responses",
        request_body={"model": "o3"},
        response_body=None,
        stream_chunks=(
            'data: {"type":"response.reasoning_summary_text.delta","delta":"Step 1. "}\n\n',
            'data: {"type":"response.reasoning.delta","delta":"Step 2."}\n\n',
            'data: {"type":"response.output_item.added","item":{"id":"fc_item_1",'
            '"type":"function_call","call_id":"call_weather_1","name":"get_weather"}}\n\n',
            'data: {"type":"response.function_call_arguments.done","item_id":"fc_item_1",'
            '"arguments":"{\\"city\\":\\"Tokyo\\"}"}\n\n',
            'data: {"type":"response.output_text.delta","delta":"Done"}\n\n',
            'data: {"type":"response.completed","response":{"model":"o3","output":[],'
            '"usage":{"input_tokens":50,"output_tokens":20}}}\n\n',
        ),
        content_type="text/event-stream",
    )
    assert sse_summary.is_stream is True
    assert sse_summary.thinking == "Step 1. Step 2."
    assert sse_summary.text == "Done"
    assert len(sse_summary.tool_calls) == 1
    assert sse_summary.tool_calls[0].name == "get_weather"
    assert sse_summary.tool_calls[0].call_id == "call_weather_1"
    assert sse_summary.tool_calls[0].arguments == {"city": "Tokyo"}


def test_gemini_sse_retains_earlier_thoughts_tokens_in_output_tokens() -> None:
    summary = parse_llm_interaction(
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent",
        request_body={},
        response_body=None,
        stream_chunks=(
            'data: {"candidates":[{"content":{"parts":[{"text":"thinking...","thought":true}]}}],'
            '"usageMetadata":{"promptTokenCount":100,"thoughtsTokenCount":75}}\n\n',
            'data: {"candidates":[{"content":{"parts":[{"text":"final answer"}]}}],'
            '"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":20}}\n\n',
        ),
        content_type="text/event-stream",
    )
    assert summary.usage.input_tokens == 100
    assert summary.usage.reasoning_tokens == 75
    assert summary.usage.output_tokens == 95
    assert summary.thinking == "thinking..."
    assert summary.text == "final answer"


def test_anthropic_sse_message_delta_usage_and_content_block_start_initial_values() -> None:
    summary = parse_llm_interaction(
        url="https://api.anthropic.com/v1/messages",
        request_body={"model": "claude-sonnet-4"},
        response_body=None,
        stream_chunks=(
            'event: message_start\ndata: {"type":"message_start","message":'
            '{"model":"claude-sonnet-4","usage":{"input_tokens":10,"output_tokens":1}}}\n\n',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"thinking","thinking":"Initial thought. "}}\n\n',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"thinking_delta","thinking":"More thought."}}\n\n',
            'event: content_block_start\ndata: {"type":"content_block_start","index":1,'
            '"content_block":{"type":"text","text":"Hello "}}\n\n',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,'
            '"delta":{"type":"text_delta","text":"world"}}\n\n',
            'event: content_block_start\ndata: {"type":"content_block_start","index":2,'
            '"content_block":{"type":"tool_use","id":"toolu_init","name":"lookup",'
            '"input":{"key":"val"}}}\n\n',
            'event: message_delta\ndata: {"type":"message_delta","usage":'
            '{"input_tokens":40,"cache_read_input_tokens":100,'
            '"cache_creation_input_tokens":60,"output_tokens":25}}\n\n',
        ),
        content_type="text/event-stream",
    )
    assert summary.thinking == "Initial thought. More thought."
    assert summary.text == "Hello world"
    assert len(summary.tool_calls) == 1
    assert summary.tool_calls[0].name == "lookup"
    assert summary.tool_calls[0].arguments == {"key": "val"}
    assert summary.tool_calls[0].call_id == "toolu_init"
    assert summary.usage.cached_input_tokens == 100
    assert summary.usage.cache_creation_tokens == 60
    assert summary.usage.input_tokens == 200
    assert summary.usage.output_tokens == 25


def test_ollama_json_and_ndjson_extract_thinking() -> None:
    json_res = parse_llm_interaction(
        url="http://localhost:11434/api/chat",
        request_body={"model": "deepseek-r1"},
        response_body={
            "model": "deepseek-r1",
            "message": {
                "role": "assistant",
                "thinking": "Internal chain of thought.",
                "content": "Final reply.",
            },
            "prompt_eval_count": 30,
            "eval_count": 15,
            "done": True,
        },
    )
    assert json_res.provider == "ollama"
    assert json_res.thinking == "Internal chain of thought."
    assert json_res.text == "Final reply."

    ndjson_res = parse_llm_interaction(
        url="http://localhost:11434/api/chat",
        request_body={"model": "qwen3"},
        response_body=None,
        stream_chunks=(
            '{"model":"qwen3","message":{"role":"assistant","thinking":"Part 1. "},"done":false}\n',
            '{"model":"qwen3","message":{"role":"assistant","thinking":"Part 2."},"done":false}\n',
            '{"model":"qwen3","message":{"role":"assistant","content":"Answer"},"prompt_eval_count":20,"eval_count":10,"done":true}\n',
        ),
        content_type="application/x-ndjson",
    )
    assert ndjson_res.provider == "ollama"
    assert ndjson_res.thinking == "Part 1. Part 2."
    assert ndjson_res.text == "Answer"


def test_detect_provider_matches_path_tokens_not_query_strings() -> None:
    assert _detect_provider("https://example.com/login?next=/responses", None) == "unknown"
    assert (
        _detect_provider("https://example.com/auth?redirect=/chat/completions", None) == "unknown"
    )
    assert (
        _detect_provider("https://example.com/callback?target=/model/foo/converse", None)
        == "unknown"
    )
    assert _detect_provider("https://proxy.internal/v1/responses?foo=bar", None) == "openai"
    assert _detect_provider("https://proxy.internal/v1/chat/completions?v=1", None) == "openai"


# --- 2. pricing.py ---------------------------------------------------------


def test_model_price_validation_and_pricing_file_errors_and_env_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # ModelPrice.__post_init__ rejects negative, non-finite, and bool values
    bad_prices: list[tuple[Any, Any, Any, Any]] = [
        (-1.0, 1.0, None, None),
        (1.0, -0.5, None, None),
        (1.0, 1.0, -0.1, None),
        (1.0, 1.0, None, -0.1),
        (float("nan"), 1.0, None, None),
        (1.0, float("inf"), None, None),
        (1.0, 1.0, float("-inf"), None),
        (1.0, 1.0, None, float("nan")),
        (True, 1.0, None, None),
        (1.0, False, None, None),
        (1.0, 1.0, True, None),
        (1.0, 1.0, None, False),
    ]
    for inp, out, cached, write in bad_prices:
        with pytest.raises(ValueError):
            ModelPrice(inp, out, cached_input_per_million=cached, cache_write_per_million=write)

    # Zero prices are allowed
    assert ModelPrice(0.0, 0.0, cached_input_per_million=0.0, cache_write_per_million=0.0)

    # load_pricing_file rejects booleans, negatives, and non-finite numbers
    for idx, payload in enumerate(
        [
            {"m": {"input_per_million": True, "output_per_million": 1.0}},
            {"m": {"input_per_million": 1.0, "output_per_million": False}},
            {
                "m": {
                    "input_per_million": 1.0,
                    "output_per_million": 1.0,
                    "cached_input_per_million": True,
                }
            },
            {"m": {"input_per_million": -1.0, "output_per_million": 1.0}},
            {"m": {"input_per_million": "nan", "output_per_million": 1.0}},
            {"m": {"input_per_million": "inf", "output_per_million": 1.0}},
        ]
    ):
        p = tmp_path / f"bad_price_{idx}.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_pricing_file(p)

    # _load_env_pricing_cached logs warning via logging (not warnings.warn) when file is invalid
    broken = tmp_path / "broken_env_pricing.json"
    broken.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv(PRICING_FILE_ENV, str(broken))
    with caplog.at_level(logging.WARNING, logger="rhinotape.pricing"):
        cost = estimate_cost_usd("gpt-4o-mini", 1_000_000, 0)
    assert cost == pytest.approx(0.15)
    assert any("RHINOTAPE_PRICING_FILE" in rec.message for rec in caplog.records)


# --- 3. pytest_plugin.py ---------------------------------------------------


def test_pytest_plugin_unknown_marker_kwarg_and_forwarded_options(
    pytester: pytest.Pytester,
) -> None:
    # Direct helper check
    with pytest.raises(pytest.UsageError, match="redact_pi"):
        _validate_tape_marker_kwargs({"redact_pi": True})

    # End-to-end pytester check: typo in @pytest.mark.tape fails fast with UsageError
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    pytester.makepyfile(
        test_bad_marker="""
        import pytest

        @pytest.mark.tape(redact_pi=True)
        def test_typo(rhinotape):
            pass
        """
    )
    bad_res = pytester.runpytest()
    assert bad_res.ret == pytest.ExitCode.USAGE_ERROR
    assert "redact_pi" in (bad_res.stdout.str() + bad_res.stderr.str())


def test_pytest_plugin_forwards_all_tape_session_options_and_help(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    help_res = pytester.runpytest("--help")
    help_out = help_res.stdout.str()
    assert "new_episodes" in help_out

    markers_res = pytester.runpytest("--markers")
    markers_out = markers_res.stdout.str()
    for opt in (
        "custom_pricing",
        "extra_sensitive_headers",
        "extra_sensitive_query_params",
        "ignore_request",
        "patch_httpx",
        "store",
        "redact_pii",
        "strict_unused",
    ):
        assert opt in markers_out

    pytester.makepyfile(
        test_forward="""
        import httpx
        import pytest
        from rhinotape import MemoryCassetteStore, ModelPrice

        MEM_STORE = MemoryCassetteStore()

        def _ignore_health(req: httpx.Request) -> bool:
            return req.url.path == "/health"

        @pytest.mark.tape(
            mode="record",
            custom_pricing={"custom-x": ModelPrice(10.0, 20.0)},
            extra_sensitive_headers=("x-internal-secret",),
            extra_sensitive_query_params=("session_sig",),
            ignore_request=_ignore_health,
            patch_httpx=False,
            store=MEM_STORE,
        )
        def test_forwarded(rhinotape):
            assert rhinotape.custom_pricing == {"custom-x": ModelPrice(10.0, 20.0)}
            assert rhinotape.extra_sensitive_headers == ("x-internal-secret",)
            assert rhinotape.extra_sensitive_query_params == ("session_sig",)
            assert rhinotape.ignore_request is _ignore_health
            assert rhinotape.patch_httpx is False
            assert rhinotape.store is MEM_STORE
        """
    )
    ok_res = pytester.runpytest()
    ok_res.assert_outcomes(passed=1)


def test_pytest_plugin_non_finite_cost_budget_and_atomic_write(tmp_path: Path) -> None:
    out_file = tmp_path / "nested" / "atomic.txt"
    _write_bytes_atomic(out_file, b"hello atomic")
    assert out_file.read_bytes() == b"hello atomic"

    # Both JSON and HTML report outputs use _write_bytes_atomic cleanly
    sample_tape = _make_sample_cassette(tmp_path / "report_sample.tape.json")
    json_out = tmp_path / "reports" / "out.json"
    html_out = tmp_path / "reports" / "out.html"
    ok_stash = pytest.Stash()
    ok_stash[_REPORT_KEY] = [
        {
            "nodeid": "test_ok.py::test_x",
            "cassette": str(sample_tape),
            "action": "replayed",
            "calls": 1,
            "input_tokens": 10,
            "cached_input_tokens": 0,
            "output_tokens": 5,
            "total_tokens": 15,
            "estimated_cost_usd": 0.001,
            "latency_ms": 1.0,
        }
    ]
    ok_stash[_VIOLATIONS_KEY] = []
    ok_opts: dict[str, Any] = {
        "--tape-max-tokens": None,
        "--tape-max-cost": 1.0,
        "--tape-json-report": str(json_out),
        "--tape-html-report": str(html_out),
    }
    ok_cfg = SimpleNamespace(
        stash=ok_stash,
        getoption=lambda name, default=None, _opts=ok_opts: _opts.get(name, default),
    )
    ok_session = SimpleNamespace(config=ok_cfg, exitstatus=pytest.ExitCode.OK)
    pytest_sessionfinish(ok_session, 0)  # type: ignore[arg-type]
    assert ok_session.exitstatus == pytest.ExitCode.OK
    assert json_out.is_file()
    assert html_out.is_file()
    assert "pytest rhinotape Report" in html_out.read_text(encoding="utf-8")

    for bad_cost in (float("nan"), float("inf")):
        stash = pytest.Stash()
        stash[_REPORT_KEY] = [
            {
                "nodeid": "test_nan.py::test_x",
                "cassette": "x.tape.json",
                "action": "replayed",
                "calls": 1,
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "output_tokens": 5,
                "total_tokens": 15,
                "estimated_cost_usd": bad_cost,
                "latency_ms": 1.0,
            }
        ]
        stash[_VIOLATIONS_KEY] = []
        opts: dict[str, Any] = {
            "--tape-max-tokens": None,
            "--tape-max-cost": 1.0,
            "--tape-json-report": None,
            "--tape-html-report": None,
        }
        cfg = SimpleNamespace(
            stash=stash,
            getoption=lambda name, default=None, _opts=opts: _opts.get(name, default),
        )
        session = SimpleNamespace(config=cfg, exitstatus=pytest.ExitCode.OK)
        pytest_sessionfinish(session, 0)  # type: ignore[arg-type]
        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
        assert len(stash[_VIOLATIONS_KEY]) == 1
        assert "non-finite" in stash[_VIOLATIONS_KEY][0]


# --- 4. cli.py -------------------------------------------------------------


def _make_sample_cassette(
    path: Path,
    *,
    status_code: int = 200,
    latency_ms: float = 10.0,
    method: str = "POST",
    url: str = "https://api.openai.com/v1/chat/completions",
    fingerprint: str = "fp1234",
    req_body: Any = None,
    req_encoding: str = "json",
    resp_body: Any = None,
    resp_encoding: str = "json",
    resp_headers: dict[str, str] | None = None,
    stream_chunks: tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    inter = Interaction(
        request=RecordedRequest(
            method=method,
            url=url,
            headers={"authorization": "[REDACTED]"},
            body=req_body if req_body is not None else {"model": "gpt-4o-mini"},
            fingerprint=fingerprint,
            body_encoding=req_encoding,
        ),
        response=RecordedResponse(
            status_code=status_code,
            headers=resp_headers
            if resp_headers is not None
            else {"content-type": "application/json"},
            body=resp_body if resp_body is not None else {"choices": []},
            body_encoding=resp_encoding,
            stream_chunks=stream_chunks,
        ),
        latency_ms=latency_ms,
        llm=LLMCallSummary(
            provider="openai",
            model="gpt-4o-mini",
            usage=Usage(input_tokens=100, output_tokens=50),
            estimated_cost_usd=0.001,
            text="ok",
            is_stream=stream_chunks is not None,
        ),
    )
    c = Cassette(path=path, interactions=[inter], metadata=metadata or {})
    c.save()
    return path


def test_cli_verify_runs_cassette_validate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_status = _make_sample_cassette(tmp_path / "bad_status.tape.json", status_code=999)
    assert cli_main(["verify", str(bad_status), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert any("status_code" in e["error"] for e in out["schema_errors"])

    bad_stream = _make_sample_cassette(
        tmp_path / "bad_stream.tape.json", resp_encoding="stream", stream_chunks=None
    )
    assert cli_main(["verify", str(bad_stream)]) == 1
    assert "stream_chunks is None" in capsys.readouterr().err

    bad_fp = _make_sample_cassette(
        tmp_path / "bad_fp.tape.json",
        method="",
        url="",
        fingerprint="",
        latency_ms=-5.0,
        req_encoding="invalid_enc",
    )
    assert cli_main(["verify", str(bad_fp), "--json"]) == 1
    issues = [e["error"] for e in json.loads(capsys.readouterr().out)["schema_errors"]]
    assert any("method" in i for i in issues)
    assert any("url" in i for i in issues)
    assert any("fingerprint" in i for i in issues)
    assert any("latency_ms" in i for i in issues)
    assert any("body_encoding" in i for i in issues)


def test_cli_verify_expanded_secrets_and_base64_bodies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hf_tok = "hf_" + "A" * 30
    groq_tok = "gsk_" + "B" * 30
    pplx_tok = "pplx-" + "C" * 30
    xai_tok = "xai-" + "D" * 30
    pem_hdr = "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."

    # Encode secrets inside base64 request & response bodies (with invalid UTF-8 bytes prefix)
    req_b64 = base64.b64encode(b"\xff\xfe" + f"{hf_tok} {groq_tok}".encode()).decode("ascii")
    resp_b64 = base64.b64encode(b"\x80\x81" + f"{pplx_tok} {xai_tok}\n{pem_hdr}".encode()).decode(
        "ascii"
    )

    cpath = _make_sample_cassette(
        tmp_path / "b64_secrets.tape.json",
        req_body=req_b64,
        req_encoding="base64",
        resp_body=resp_b64,
        resp_encoding="base64",
    )
    assert cli_main(["verify", str(cpath), "--check-secrets", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    found_patterns = {f["pattern"] for f in data["secret_findings"]}
    assert found_patterns == {
        "huggingface_token",
        "groq_key",
        "perplexity_key",
        "xai_key",
        "pem_private_key",
    }
    raw_out = json.dumps(data)
    assert hf_tok not in raw_out
    assert groq_tok not in raw_out
    assert pplx_tok not in raw_out
    assert xai_tok not in raw_out


def test_cli_redact_ci_content_type_metadata_and_url_pattern(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cpath = _make_sample_cassette(
        tmp_path / "redact_all.tape.json",
        url="https://api.openai.com/tenants/secret-org-99/v1/chat/completions",
        resp_headers={"Content-Type": "text/event-stream"},
        resp_body=None,
        resp_encoding="stream",
        stream_chunks=(
            'data: {"model":"gpt-4o-mini",'
            '"choices":[{"delta":{"content":"Email alice@example.com"}}]}\n\n',
            "data: [DONE]\n\n",
        ),
        metadata={"owner": "bob@example.com", "secret_note": "top-secret"},
    )

    # Dry-run reports metadata-only update without mutating disk
    meta_only = _make_sample_cassette(
        tmp_path / "meta_only.tape.json",
        metadata={"contact": "carol@example.com"},
    )
    assert cli_main(["redact", str(meta_only), "--pii", "email", "--dry-run"]) == 0
    assert "WOULD UPDATE" in capsys.readouterr().out
    assert Cassette.load(meta_only).metadata["contact"] == "carol@example.com"

    # Repeatable --url-pattern + --pii + --field + case-insensitive Content-Type SSE reparse
    rc = cli_main(
        [
            "redact",
            str(cpath),
            "--pii",
            "email",
            "--field",
            "secret_note",
            "--url-pattern",
            r"(?<=/tenants/)[^/]+",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    loaded = Cassette.load(cpath)
    assert loaded.metadata["owner"] == "[REDACTED]"
    assert loaded.metadata["secret_note"] == "[REDACTED]"
    assert "secret-org-99" not in loaded.interactions[0].request.url
    assert "/tenants/REDACTED/" in loaded.interactions[0].request.url
    assert "alice@example.com" not in loaded.interactions[0].llm.text
    assert loaded.interactions[0].response.stream_chunks is not None
    assert "alice@example.com" not in "".join(loaded.interactions[0].response.stream_chunks)


def test_cli_numeric_flag_validation_and_corrupt_reporting_in_search_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _make_sample_cassette(tmp_path / "good.tape.json")
    corrupt = tmp_path / "corrupt.tape.json"
    corrupt.write_text("{broken json", encoding="utf-8")

    # Numeric flag validation rejects nan, inf, and negative values
    invalid_invocations = [
        ["summary", str(tmp_path), "--max-cost", "nan"],
        ["summary", str(tmp_path), "--max-cost", "inf"],
        ["summary", str(tmp_path), "--max-cost", "-0.1"],
        ["summary", str(tmp_path), "--max-tokens", "-1"],
        ["diff", str(good), str(good), "--max-cost-increase", "nan"],
        ["diff", str(good), str(good), "--max-cost-increase", "-1"],
        ["diff", str(good), str(good), "--max-cost-increase-pct", "inf"],
        ["diff", str(good), str(good), "--max-cost-increase-pct", "-5"],
        ["search", str(tmp_path), "--min-cost", "nan"],
        ["search", str(tmp_path), "--min-cost", "inf"],
        ["search", str(tmp_path), "--min-cost", "-0.01"],
    ]
    for argv in invalid_invocations:
        assert cli_main(argv) == 1
        assert "Error:" in capsys.readouterr().err

    # search warns on stderr (plain) and includes errors in JSON
    assert cli_main(["search", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "skipping corrupt cassette" in captured.err
    assert "good.tape.json" in captured.out

    assert cli_main(["search", str(tmp_path), "--json"]) == 0
    search_json = json.loads(capsys.readouterr().out)
    assert search_json["count"] == 1
    assert len(search_json["errors"]) == 1
    assert search_json["errors"][0]["path"] == "corrupt.tape.json"

    # list warns on stderr (plain) and includes errors in JSON
    assert cli_main(["list", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "skipping corrupt cassette" in captured.err
    assert "good.tape.json" in captured.out

    assert cli_main(["list", str(tmp_path), "--json"]) == 0
    list_json = json.loads(capsys.readouterr().out)
    assert list_json["count"] == 1
    assert len(list_json["errors"]) == 1
    assert list_json["errors"][0]["path"] == "corrupt.tape.json"


def test_cli_replay_check_custom_matching_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cpath = tmp_path / "custom_match.tape.json"
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            },
        )
    )
    with tape(
        cpath,
        mode=TapeMode.RECORD,
        ignore_fields=("request_id",),
        ignore_query_params=("ts",),
        match_on=("method", "path", "body"),
        match_headers=("x-tenant",),
    ):
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions?ts=12345",
                headers={"x-tenant": "acme"},
                json={"model": "gpt-4o-mini", "request_id": "abc"},
            )

    # Without custom flags, fingerprint mismatches default matching
    assert cli_main(["replay-check", str(cpath)]) == 1
    capsys.readouterr()

    # With matching custom flags, replay-check succeeds cleanly
    assert (
        cli_main(
            [
                "replay-check",
                str(cpath),
                "--ignore-fields",
                "request_id",
                "--ignore-query-params",
                "ts",
                "--match-on",
                "method,path,body",
                "--match-headers",
                "x-tenant",
            ]
        )
        == 0
    )
    assert "OK:" in capsys.readouterr().out

    # Invalid --match-on returns exit code 1
    assert cli_main(["replay-check", str(cpath), "--match-on", "bogus"]) == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_serve_sigterm_and_matching_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cpath = _make_sample_cassette(tmp_path / "serve.tape.json")

    # Parser exposes --ignore-fields and --match-on
    parsed = build_parser().parse_args(
        ["serve", str(cpath), "--ignore-fields", "ts,req_id", "--match-on", "method,path"]
    )
    assert parsed.ignore_fields == ["ts,req_id"]
    assert parsed.match_on == ["method,path"]

    # Invalid --match-on rejected cleanly
    assert cli_main(["serve", str(cpath), "--match-on", "invalid_key"]) == 1
    assert "Error:" in capsys.readouterr().err

    # Verify _cmd_serve forwards ignore_fields and match_on to TapeReplayServer
    import rhinotape.server as server_mod

    captured_kwargs: dict[str, Any] = {}
    real_cls = server_mod.TapeReplayServer

    def _recording_server(*args: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(server_mod, "TapeReplayServer", _recording_server)

    # SIGTERM gracefully shuts down serve and restores previous handler
    if hasattr(signal, "SIGTERM"):
        prev_handler = signal.getsignal(signal.SIGTERM)

        def _send_sigterm() -> None:
            time.sleep(0.15)
            signal.raise_signal(signal.SIGTERM)

        sender = threading.Thread(target=_send_sigterm, daemon=True)
        sender.start()
        rc = cli_main(
            [
                "serve",
                str(cpath),
                "--port",
                "0",
                "--ignore-fields",
                "request_id,trace_id",
                "--match-on",
                "method,path,body",
            ]
        )
        sender.join(timeout=2.0)
        assert rc == 0
        assert signal.getsignal(signal.SIGTERM) is prev_handler
        assert captured_kwargs["ignore_fields"] == ["request_id", "trace_id"]
        assert captured_kwargs["match_on"] == ("method", "path", "body")
        out = capsys.readouterr().out
        assert "Stopping rhinotape replay server." in out


def test_cassette_stats_delegation(tmp_path: Path) -> None:
    cpath = _make_sample_cassette(tmp_path / "stats_eq.tape.json")
    cass = Cassette.load(cpath)
    assert _cassette_stats(cass) == cass.stats()


def test_workflows_hardening_contents() -> None:
    root = Path(__file__).resolve().parent.parent
    ci_text = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ruff check src tests examples" in ci_text
    assert "ruff format --check src tests examples" in ci_text
    assert "MemoryCassetteStore" in ci_text
    assert "TapeSession" in ci_text
    assert "MockTransport" in ci_text

    pub_text = (root / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert "pytest -q -W error" in pub_text
    assert "_version.py" in pub_text
    assert "RELEASE_TAG" in pub_text
