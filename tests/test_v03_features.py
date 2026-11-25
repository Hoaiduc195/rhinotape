"""Comprehensive tests for llm-tape v0.3.0 features."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from llm_tape import (
    Cassette,
    ModelPrice,
    TapeAssertionError,
    UnmatchedRequestError,
    __version__,
    estimate_cost_usd,
    load_pricing_file,
    redact_body,
    redact_text,
    tape,
)
from llm_tape.cli import main as cli_main
from llm_tape.parsers import parse_llm_interaction
from llm_tape.pricing import PRICING_FILE_ENV
from llm_tape.scrubbing import fingerprint_request, validate_match_on

pytest_plugins = ["pytester"]


def test_single_source_version() -> None:
    assert __version__ == "0.3.0"


def test_cached_input_pricing_and_pricing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # OpenAI gpt-4o-mini: input=0.15/1M, cached_input=0.075/1M, output=0.60/1M
    cost_cached = estimate_cost_usd(
        "gpt-4o-mini",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_input_tokens=800_000,
    )
    expected = (200_000 * 0.15 + 800_000 * 0.075 + 100_000 * 0.60) / 1_000_000
    assert cost_cached == pytest.approx(expected)

    # OpenRouter & Bedrock model prefixes strip cleanly
    assert estimate_cost_usd("openai/gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost_usd(
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0", 1_000_000, 0
    ) == pytest.approx(3.00)

    # Load pricing from JSON file + env var
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(
        json.dumps(
            {
                "custom-llm": {
                    "input_per_million": 2.0,
                    "output_per_million": 4.0,
                    "cached_input_per_million": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )
    loaded = load_pricing_file(pricing_file)
    assert loaded["custom-llm"] == ModelPrice(2.0, 4.0, cached_input_per_million=0.5)

    monkeypatch.setenv(PRICING_FILE_ENV, str(pricing_file))
    env_cost = estimate_cost_usd(
        "custom-llm-v1",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=500_000,
    )
    assert env_cost == pytest.approx((500_000 * 2.0 + 500_000 * 0.5) / 1_000_000)

    # Validation errors on malformed pricing file
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="top level"):
        load_pricing_file(bad_file)


def test_body_scrubbing_redact_fields_patterns_and_callback(tmp_path: Path) -> None:
    sample = {
        "user": {"email": "alice@example.com", "ssn": "123-45-6789"},
        "messages": [
            {"role": "user", "content": "My key is sk-secret999 and phone 555-1234"},
        ],
        "tokens": ["t1", "t2"],
    }
    redacted = redact_body(
        sample,
        redact_fields=["user.ssn", "tokens.*"],
        redact_patterns=[r"sk-[a-z0-9]+", re.compile(r"\b[\w.]+@example\.com\b")],
        scrub_body=lambda b: {**b, "scrubbed": True},
    )
    assert redacted["user"]["ssn"] == "[REDACTED]"
    assert redacted["user"]["email"] == "[REDACTED]"
    assert redacted["tokens"] == ["[REDACTED]", "[REDACTED]"]
    assert "sk-secret999" not in redacted["messages"][0]["content"]
    assert redacted["scrubbed"] is True
    # Original object not mutated
    assert sample["user"]["ssn"] == "123-45-6789"
    assert redact_text("token=sk-abc123", [r"sk-[a-z0-9]+"]) == "token=[REDACTED]"

    # End-to-end record + replay with body redaction & SSE pattern redaction
    cassette_path = tmp_path / "redacted.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "Secret sk-resp777"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
        )

    with tape(
        cassette_path,
        mode="once",
        redact_fields=["user_secret"],
        redact_patterns=[r"sk-[a-z0-9]+"],
    ) as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"user_secret": "top-secret", "prompt": "key sk-req111"},
            )
        assert session.total_usage.cached_input_tokens == 80

    raw_text = cassette_path.read_text(encoding="utf-8")
    assert "top-secret" not in raw_text
    assert "sk-req111" not in raw_text
    assert "sk-resp777" not in raw_text

    # Replay matches even with different secret values because both redact to [REDACTED]
    with tape(
        cassette_path,
        mode="replay",
        redact_fields=["user_secret"],
        redact_patterns=[r"sk-[a-z0-9]+"],
    ) as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"user_secret": "other-secret", "prompt": "key sk-req222"},
            )
            assert "[REDACTED]" in resp.json()["choices"][0]["message"]["content"]
        session.assert_call_count(1)


def test_flexible_match_on_and_custom_matcher(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid match_on"):
        validate_match_on(["method", "bogus"])
    with pytest.raises(ValueError, match="non-empty"):
        validate_match_on([])

    # Different hosts, same path + body -> match when match_on=("method", "path", "body")
    fp1 = fingerprint_request(
        "POST",
        "https://prod.openai.com/v1/chat/completions?v=1",
        {"a": 1},
        match_on=("method", "path", "body"),
    )
    fp2 = fingerprint_request(
        "POST",
        "https://staging.openai.com/v1/chat/completions?v=2",
        {"a": 1},
        match_on=("method", "path", "body"),
    )
    assert fp1 == fp2

    # Headers matching
    fp_h1 = fingerprint_request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        {"a": 1},
        headers={"X-Tenant": "t1"},
        match_on=("method", "host", "path", "query", "headers", "body"),
    )
    fp_h2 = fingerprint_request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        {"a": 1},
        headers={"X-Tenant": "t2"},
        match_on=("method", "host", "path", "query", "headers", "body"),
    )
    assert fp_h1 != fp_h2

    cassette_path = tmp_path / "matcher.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    with tape(cassette_path, mode="record"):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"prompt": "v1"})

    # Custom matcher matches purely on URL path regardless of body
    with tape(
        cassette_path,
        mode="replay",
        custom_matcher=lambda rec, inc: rec.method == inc.method and rec.url == inc.url,
    ) as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            r = client.post("https://api.openai.com/v1/chat/completions", json={"prompt": "v2"})
            assert r.json() == {"ok": True}
        session.assert_no_unused_interactions()

    # Custom matcher failure raises UnmatchedRequestError with diff
    with tape(
        cassette_path,
        mode="replay",
        custom_matcher=lambda rec, inc: False,
    ):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(UnmatchedRequestError, match="custom_matcher"):
                client.post("https://api.openai.com/v1/chat/completions", json={"prompt": "v2"})


@pytest.mark.asyncio
async def test_replay_latency_scale_sync_and_async(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="replay_latency_scale"):
        tape(tmp_path / "neg.tape.json", replay_latency_scale=-0.1)

    cassette_path = tmp_path / "latency.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(0.02)
        return httpx.Response(200, json={"ok": True})

    with tape(cassette_path, mode="record"):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.get("https://api.openai.com/v1/models")

    t0 = time.perf_counter()
    with tape(cassette_path, mode="replay", replay_latency_scale=1.0):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.get("https://api.openai.com/v1/models")
    elapsed_sync = time.perf_counter() - t0
    assert elapsed_sync >= 0.015

    t1 = time.perf_counter()
    async with tape(cassette_path, mode="replay", replay_latency_scale=1.0):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await client.get("https://api.openai.com/v1/models")
    elapsed_async = time.perf_counter() - t1
    assert elapsed_async >= 0.015


def test_new_providers_azure_vertex_ollama_and_bedrock(tmp_path: Path) -> None:
    # 1. Azure OpenAI (model extracted from deployment name in URL)
    azure_url = "https://myres.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-06-01"
    azure_resp = {
        "choices": [{"message": {"role": "assistant", "content": "Hello from Azure"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }
    s_azure = parse_llm_interaction(azure_url, {}, azure_resp)
    assert s_azure.provider == "openai"
    assert s_azure.model == "gpt-4o-mini"
    assert s_azure.estimated_cost_usd > 0

    # 2. Vertex AI Gemini with cachedContentTokenCount
    vertex_url = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/"
        "publishers/google/models/gemini-2.5-flash:generateContent"
    )
    vertex_resp = {
        "candidates": [{"content": {"parts": [{"text": "Vertex OK"}]}}],
        "usageMetadata": {
            "promptTokenCount": 1000,
            "cachedContentTokenCount": 800,
            "candidatesTokenCount": 50,
        },
    }
    s_vertex = parse_llm_interaction(vertex_url, {}, vertex_resp)
    assert s_vertex.provider == "gemini"
    assert s_vertex.model == "gemini-2.5-flash"
    assert s_vertex.usage.cached_input_tokens == 800

    # 3. Ollama JSON (/api/chat) + NDJSON streaming
    ollama_resp = {
        "model": "llama3.2",
        "message": {
            "role": "assistant",
            "content": "Checking...",
            "tool_calls": [{"function": {"name": "get_time", "arguments": {"tz": "UTC"}}}],
        },
        "done": True,
        "prompt_eval_count": 40,
        "eval_count": 15,
    }
    s_ollama = parse_llm_interaction("http://localhost:11434/api/chat", {}, ollama_resp)
    assert s_ollama.provider == "ollama"
    assert s_ollama.model == "llama3.2"
    assert s_ollama.usage.total_tokens == 55
    assert s_ollama.tool_calls[0].name == "get_time"
    assert s_ollama.tool_calls[0].arguments == {"tz": "UTC"}

    ndjson_chunks = [
        '{"model":"llama3.2","message":{"content":"Hi "},"done":false}\n',
        '{"model":"llama3.2","message":{"content":"there!"},"done":true,"prompt_eval_count":10,"eval_count":5}\n',
    ]
    s_ollama_stream = parse_llm_interaction(
        "http://localhost:11434/api/chat",
        {},
        None,
        stream_chunks=ndjson_chunks,
        content_type="application/x-ndjson",
    )
    assert s_ollama_stream.provider == "ollama"
    assert s_ollama_stream.is_stream is True
    assert s_ollama_stream.text == "Hi there!"
    assert s_ollama_stream.usage.total_tokens == 15

    # 4. AWS Bedrock Converse API
    bedrock_url = (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/"
        "anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse"
    )
    bedrock_resp = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Using tool."},
                    {"toolUse": {"toolUseId": "tu_1", "name": "search", "input": {"q": "aws"}}},
                ],
            }
        },
        "usage": {"inputTokens": 200, "outputTokens": 40, "cacheReadInputTokens": 100},
    }
    s_bedrock = parse_llm_interaction(bedrock_url, {}, bedrock_resp)
    assert s_bedrock.provider == "bedrock"
    assert s_bedrock.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert s_bedrock.usage.cached_input_tokens == 100
    assert s_bedrock.tool_calls[0].name == "search"
    assert s_bedrock.tool_calls[0].arguments == {"q": "aws"}
    assert s_bedrock.estimated_cost_usd > 0


def test_new_assertions_tool_args_unused_and_latency(tmp_path: Path) -> None:
    cassette_path = tmp_path / "assertions.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "pytest", "limit": 5}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            },
        )

    with tape(cassette_path, mode="record"):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"step": 1})
            client.post("https://api.openai.com/v1/chat/completions", json={"step": 2})

    # Consume only 1 of 2 -> assert_no_unused_interactions fails
    with tape(cassette_path, mode="replay") as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"step": 1})

        session.assert_tool_args("search", {"query": "pytest"})
        session.assert_tool_args("search", {"query": "pytest", "limit": 5}, exact=True)
        session.assert_max_latency_ms(60_000.0)

        with pytest.raises(TapeAssertionError, match="never called"):
            session.assert_tool_args("other_tool", {})
        with pytest.raises(TapeAssertionError, match="did not match"):
            session.assert_tool_args("search", {"query": "pytest"}, exact=True)
        with pytest.raises(TapeAssertionError, match="unused recorded interaction"):
            session.assert_no_unused_interactions()
        with pytest.raises(TapeAssertionError, match="Latency budget exceeded"):
            session.assert_max_latency_ms(-1.0)


def test_cli_redact_and_summary_budget_gates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cassette_path = tmp_path / "secret.tape.json"

    class _SSE(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield (
                b'data: {"model":"gpt-4o-mini",'
                b'"choices":[{"delta":{"content":"Hi sk-live999"}}]}\n\n'
            )
            yield (
                b'data: {"model":"gpt-4o-mini","choices":[],'
                b'"usage":{"prompt_tokens":500,"completion_tokens":100}}\n\n'
            )
            yield b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "x-internal-key": "secret-hdr"},
            stream=_SSE(),
        )

    with tape(cassette_path, mode="record"):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions?custom_tok=abc",
                headers={"X-Internal-Key": "secret-hdr"},
                json={"email": "bob@corp.com", "prompt": "hello"},
            ) as r:
                _ = list(r.iter_bytes())

    # Dry-run reports change without modifying file
    rc = cli_main(
        [
            "redact",
            str(cassette_path),
            "--header",
            "x-internal-key",
            "--query-param",
            "custom_tok",
            "--field",
            "email",
            "--pattern",
            r"sk-[a-z0-9]+",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert "WOULD UPDATE" in capsys.readouterr().out
    assert "bob@corp.com" in cassette_path.read_text(encoding="utf-8")

    # Actual redact updates file cleanly
    rc = cli_main(
        [
            "redact",
            str(tmp_path),
            "--header",
            "x-internal-key",
            "--query-param",
            "custom_tok",
            "--field",
            "email",
            "--pattern",
            r"sk-[a-z0-9]+",
        ]
    )
    assert rc == 0
    assert "REDACTED:" in capsys.readouterr().out
    text = cassette_path.read_text(encoding="utf-8")
    assert "secret-hdr" not in text
    assert "custom_tok=abc" not in text
    assert "bob@corp.com" not in text
    assert "sk-live999" not in text
    assert Cassette.load(cassette_path).interactions[0].request.body["email"] == "[REDACTED]"

    # Budget gate flags on summary
    assert cli_main(["summary", str(tmp_path), "--max-tokens", "1000", "--max-cost", "1.0"]) == 0
    capsys.readouterr()
    assert cli_main(["summary", str(tmp_path), "--max-tokens", "10"]) == 1
    assert "Budget exceeded: total tokens" in capsys.readouterr().err
    assert cli_main(["summary", str(tmp_path), "--max-cost", "0.000001", "--json"]) == 1
    assert "Budget exceeded: total cost" in capsys.readouterr().err

    # Missing path returns 1
    assert cli_main(["redact", str(tmp_path / "nope")]) == 1


def test_pytester_ini_defaults(pytester: pytest.Pytester, tmp_path: Path) -> None:
    tape_dir = tmp_path / "ini_tapes"
    pytester.makeini(
        f"""
        [pytest]
        asyncio_default_fixture_loop_scope = function
        tape_mode = once
        tape_dir = {tape_dir}
        tape_ignore_localhost = true
        tape_ignore_fields =
            metadata.trace_id
        tape_ignore_query_params =
            ts
        """
    )
    pytester.makepyfile(
        test_ini_cfg="""
        import httpx

        def _h(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )

        def test_uses_ini_defaults(llm_tape):
            with httpx.Client(transport=httpx.MockTransport(_h)) as client:
                client.get("http://127.0.0.1:8000/ping")
                client.post(
                    "https://api.openai.com/v1/chat/completions?ts=123",
                    json={"metadata": {"trace_id": "abc"}, "msg": "hi"},
                )
            llm_tape.assert_call_count(1)
        """
    )
    res1 = pytester.runpytest()
    res1.assert_outcomes(passed=1)
    assert len(list(tape_dir.rglob("*.tape.json"))) == 1

    # Replay with different ts & trace_id succeeds using ini defaults
    res2 = pytester.runpytest("--tape-mode=replay")
    res2.assert_outcomes(passed=1)


@pytest.mark.asyncio
async def test_deflate_and_raw_deflate_streams_and_stream_latency_replay(tmp_path: Path) -> None:
    import zlib

    raw_sse = (
        b'data: {"model":"gpt-4o-mini","choices":[{"delta":{"content":"Deflate OK"}}]}\n\n'
        b'data: {"model":"gpt-4o-mini","choices":[],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n'
        b"data: [DONE]\n\n"
    )

    # 1. Zlib-wrapped deflate
    zlib_bytes = zlib.compress(raw_sse)
    # 2. Raw RFC 1951 deflate (no zlib header)
    comp = zlib.compressobj(level=6, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
    raw_deflate_bytes = comp.compress(raw_sse) + comp.flush()

    class _SyncBytes(httpx.SyncByteStream):
        def __init__(self, data: bytes) -> None:
            self._data = data

        def __iter__(self) -> Iterator[bytes]:
            mid = len(self._data) // 2
            yield self._data[:mid]
            yield self._data[mid:]

    for label, payload in [("zlib", zlib_bytes), ("raw", raw_deflate_bytes)]:
        cpath = tmp_path / f"deflate_{label}.tape.json"

        def _make_handler(data: bytes) -> httpx.MockTransport:
            def _h(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream", "content-encoding": "deflate"},
                    stream=_SyncBytes(data),
                )

            return httpx.MockTransport(_h)

        with tape(cpath, mode="record") as session:
            with httpx.Client(transport=_make_handler(payload)) as client:
                with client.stream(
                    "POST", "https://api.openai.com/v1/chat/completions", json={"m": label}
                ) as r:
                    _ = list(r.iter_bytes())
            assert session.interactions[0].llm.text == "Deflate OK"
            assert session.total_tokens == 15

    # Streamed replay with replay_latency_scale > 0 (sync & async)
    stream_path = tmp_path / "stream_delay.tape.json"

    class _TwoChunkSync(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            time.sleep(0.01)
            yield b'data: {"model":"gpt-4o-mini","choices":[{"delta":{"content":"A"}}]}\n\n'
            time.sleep(0.01)
            yield b'data: {"model":"gpt-4o-mini","choices":[{"delta":{"content":"B"}}]}\n\n'

    def _slow_stream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_TwoChunkSync(),
        )

    with tape(stream_path, mode="record"):
        with httpx.Client(transport=httpx.MockTransport(_slow_stream)) as client:
            with client.stream("POST", "https://api.openai.com/v1/chat/completions", json={}) as r:
                _ = list(r.iter_bytes())

    t0 = time.perf_counter()
    with tape(stream_path, mode="replay", replay_latency_scale=1.0) as s_sync:
        with httpx.Client(transport=httpx.MockTransport(_slow_stream)) as client:
            with client.stream("POST", "https://api.openai.com/v1/chat/completions", json={}) as r:
                assert len(list(r.iter_lines())) >= 2
        s_sync.assert_no_unused_interactions()
    assert time.perf_counter() - t0 >= 0.015

    t1 = time.perf_counter()
    async with tape(stream_path, mode="replay", replay_latency_scale=1.0) as s_async:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_slow_stream)) as client:
            async with client.stream(
                "POST", "https://api.openai.com/v1/chat/completions", json={}
            ) as r:
                chunks = [c async for c in r.aiter_text()]
                assert len(chunks) >= 2
        s_async.assert_no_unused_interactions()
    assert time.perf_counter() - t1 >= 0.015


def test_unused_interactions_with_allow_playback_repeats_and_extra_edge_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cassette_path = tmp_path / "repeats.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    with tape(cassette_path, mode="record"):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"x": 1})

    # allow_playback_repeats=True must STILL mark the interaction as consumed
    with tape(cassette_path, mode="replay", allow_playback_repeats=True) as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"x": 1})
            client.post("https://api.openai.com/v1/chat/completions", json={"x": 1})
        session.assert_call_count(2)
        session.assert_no_unused_interactions()

    # Same with custom_matcher + allow_playback_repeats=True
    with tape(
        cassette_path,
        mode="replay",
        allow_playback_repeats=True,
        custom_matcher=lambda rec, inc: True,
    ) as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"x": 2})
            client.post("https://api.openai.com/v1/chat/completions", json={"x": 3})
        session.assert_call_count(2)
        session.assert_no_unused_interactions()

    # Wildcard dict & list index redaction
    obj = {"meta": {"a": 1, "b": 2}, "items": [{"secret": "s1"}, {"secret": "s2"}]}
    out = redact_body(obj, redact_fields=["meta.*", "items.*.secret", "items.0"])
    assert out["meta"] == {"a": "[REDACTED]", "b": "[REDACTED]"}
    assert out["items"][0] == "[REDACTED]"
    assert out["items"][1] == {"secret": "[REDACTED]"}

    # OpenAI Responses API response.completed SSE event + Ollama /api/generate + NDJSON tools
    resp_completed_chunks = [
        'data: {"type":"response.output_text.delta","delta":"Draft "}\n\n',
        (
            'data: {"type":"response.completed","response":{"model":"gpt-4.1-mini",'
            '"output":[{"type":"message","content":[{"type":"output_text",'
            '"text":"Final answer"}]}],'
            '"usage":{"input_tokens":100,"output_tokens":20,'
            '"input_tokens_details":{"cached_tokens":50}}}}\n\n'
        ),
    ]
    s_resp = parse_llm_interaction(
        "https://api.openai.com/v1/responses",
        {},
        None,
        stream_chunks=resp_completed_chunks,
        content_type="text/event-stream",
    )
    assert s_resp.text == "Final answer"
    assert s_resp.usage.cached_input_tokens == 50

    s_gen = parse_llm_interaction(
        "http://localhost:11434/api/generate",
        {"model": "llama3.2"},
        {"response": "Generated text", "done": True, "prompt_eval_count": 12, "eval_count": 8},
    )
    assert s_gen.provider == "ollama"
    assert s_gen.text == "Generated text"

    s_ndjson = parse_llm_interaction(
        "http://localhost:11434/api/chat",
        {"model": "llama3.2"},
        None,
        stream_chunks=[
            "not-json\n",
            '{"message":{"content":"","tool_calls":[{"function":{"name":"ping","arguments":{"host":"1.1.1.1"}}}]}}\n',
            '{"response":"done","done":true,"prompt_eval_count":5,"eval_count":3}\n',
        ],
        content_type="application/x-ndjson",
    )
    assert s_ndjson.tool_calls[0].name == "ping"
    assert s_ndjson.text == "done"

    # Bad pricing entry & broken env pricing file fallback
    bad_entry = tmp_path / "bad_entry.json"
    bad_entry.write_text('{"m": 123}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_pricing_file(bad_entry)
    monkeypatch.setenv(PRICING_FILE_ENV, str(bad_entry))
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
