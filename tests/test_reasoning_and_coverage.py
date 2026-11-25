"""Tests for reasoning tokens (OpenAI/Gemini), o3/o4-mini pricing, and coverage hardening."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
import requests

from llm_tape import TapeMode, Usage, estimate_cost_usd, tape
from llm_tape.cli import main as cli_main
from llm_tape.parsers import parse_llm_interaction
from llm_tape.requests_adapter import (
    _extract_body_bytes,
    _RecordingRequestsRaw,
    _ReplayRequestsRaw,
    _status_reason,
)


def test_reasoning_tokens_openai_and_gemini_and_o3_pricing() -> None:
    # Usage roundtrip with reasoning_tokens
    u = Usage(input_tokens=100, output_tokens=80, cached_input_tokens=20, reasoning_tokens=50)
    d = u.to_dict()
    assert d["reasoning_tokens"] == 50
    assert Usage.from_dict(d) == u

    # Zero reasoning_tokens omitted for backwards-compatible JSON
    u_zero = Usage(input_tokens=10, output_tokens=5)
    assert "reasoning_tokens" not in u_zero.to_dict()

    # OpenAI Chat Completions JSON
    s1 = parse_llm_interaction(
        url="https://api.openai.com/v1/chat/completions",
        request_body={"model": "o3"},
        response_body={
            "model": "o3-2025-04-16",
            "choices": [{"message": {"role": "assistant", "content": "42"}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 200},
                "completion_tokens_details": {"reasoning_tokens": 350},
            },
        },
    )
    assert s1.usage.reasoning_tokens == 350
    assert s1.usage.cached_input_tokens == 200
    assert s1.estimated_cost_usd > 0

    # OpenAI Chat Completions SSE
    s2 = parse_llm_interaction(
        url="https://api.openai.com/v1/chat/completions",
        request_body={"model": "o4-mini"},
        response_body=None,
        stream_chunks=(
            'data: {"model":"o4-mini","choices":[{"delta":{"content":"done"}}]}\n\n',
            'data: {"model":"o4-mini","choices":[],"usage":{"prompt_tokens":100,'
            '"completion_tokens":90,"completion_tokens_details":{"reasoning_tokens":60}}}\n\n',
            "data: [DONE]\n\n",
        ),
        content_type="text/event-stream",
    )
    assert s2.usage.reasoning_tokens == 60
    assert s2.text == "done"

    # Gemini JSON & SSE thoughtsTokenCount -> reasoning_tokens
    g_json = parse_llm_interaction(
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
        request_body={},
        response_body={
            "candidates": [{"content": {"parts": [{"text": "answer"}]}}],
            "usageMetadata": {
                "promptTokenCount": 200,
                "candidatesTokenCount": 40,
                "thoughtsTokenCount": 120,
            },
        },
    )
    assert g_json.usage.reasoning_tokens == 120
    assert g_json.usage.output_tokens == 160

    g_sse = parse_llm_interaction(
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent",
        request_body={},
        response_body=None,
        stream_chunks=(
            'data: {"candidates":[{"content":{"parts":[{"text":"hi"}]}}],'
            '"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":10,'
            '"thoughtsTokenCount":75,"cachedContentTokenCount":30}}\n\n',
        ),
        content_type="text/event-stream",
    )
    assert g_sse.usage.reasoning_tokens == 75
    assert g_sse.usage.cached_input_tokens == 30
    assert g_sse.usage.output_tokens == 85

    # Pricing for o3 & o4-mini
    assert estimate_cost_usd("o3", 1_000_000, 0) == pytest.approx(2.0)
    assert estimate_cost_usd(
        "o4-mini", 1_000_000, 0, cached_input_tokens=1_000_000
    ) == pytest.approx(0.275)


def test_parsers_proxy_url_heuristic_detection() -> None:
    # Custom proxy URLs without provider domain still detect via body / SSE shape
    anth = parse_llm_interaction(
        url="https://proxy.internal/llm",
        request_body={},
        response_body={
            "type": "message",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    assert anth.provider == "anthropic"

    gem = parse_llm_interaction(
        url="https://proxy.internal/llm",
        request_body={},
        response_body={"candidates": [{"content": {"parts": [{"text": "g"}]}}]},
    )
    assert gem.provider == "gemini"

    bed = parse_llm_interaction(
        url="https://proxy.internal/llm",
        request_body={},
        response_body={
            "output": {"message": {"role": "assistant", "content": [{"text": "b"}]}},
            "usage": {"inputTokens": 5, "outputTokens": 2},
        },
    )
    assert bed.provider == "bedrock"

    oll = parse_llm_interaction(
        url="https://proxy.internal/llm",
        request_body={},
        response_body={"model": "llama3", "message": {"content": "o"}, "done": True},
    )
    assert oll.provider == "ollama"

    oai = parse_llm_interaction(
        url="https://proxy.internal/llm",
        request_body={},
        response_body={"choices": [{"message": {"content": "x"}}]},
    )
    assert oai.provider == "openai"

    # SSE proxy detection
    anth_sse = parse_llm_interaction(
        url="https://proxy.internal/stream",
        request_body={},
        response_body=None,
        stream_chunks=('event: message_start\ndata: {"type":"message_start","message":{}}\n\n',),
        content_type="text/event-stream",
    )
    assert anth_sse.provider == "anthropic"


def test_requests_adapter_helpers_and_latency_coverage(tmp_path: Path) -> None:
    class _DummyReq:
        def __init__(self, body: Any) -> None:
            self.body = body

    assert _extract_body_bytes(_DummyReq(None)) == b""
    assert _extract_body_bytes(_DummyReq(bytearray(b"ba"))) == b"ba"
    assert _extract_body_bytes(_DummyReq("str-body")) == b"str-body"
    assert _extract_body_bytes(_DummyReq(["a", b"b"])) == b"ab"
    assert _extract_body_bytes(_DummyReq(12345)) == b"12345"

    assert _status_reason(200) == "OK"
    assert _status_reason(999) == ""

    # _ReplayRequestsRaw partial reads, chunked stream, release_conn, close
    raw = _ReplayRequestsRaw((b"abcdef", b"ghij"))
    assert raw.read(3) == b"abc"
    assert raw.read(5) == b"defgh"
    assert raw.read() == b"ij"
    assert raw.read() == b""
    raw.release_conn()
    raw.close()

    raw2 = _ReplayRequestsRaw((b"0123456789",), chunk_delay_s=0.001)
    parts = list(raw2.stream(amt=4))
    assert parts == [b"0123", b"4567", b"89"]

    # _RecordingRequestsRaw __getattr__, release_conn, close
    rec_raw = _RecordingRequestsRaw(_ReplayRequestsRaw((b"xy",)), is_sse=False)
    assert rec_raw.closed is False
    rec_raw.release_conn()
    rec_raw.close()

    # replay_latency_scale on requests adapter (non-stream)
    tape_path = tmp_path / "req_lat.tape.json"

    class _MockAdapter(requests.adapters.BaseAdapter):
        def send(
            self,
            request: requests.PreparedRequest,
            stream: bool = False,
            timeout: Any = None,
            verify: Any = True,
            cert: Any = None,
            proxies: Any = None,
        ) -> requests.Response:
            del stream, timeout, verify, cert, proxies
            time.sleep(0.02)
            r = requests.Response()
            r.status_code = 200
            r.reason = "OK"
            r.url = str(request.url)
            r.request = request
            r.headers = requests.structures.CaseInsensitiveDict(
                {"Content-Type": "application/json"}
            )
            raw_b = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")
            r.raw = _ReplayRequestsRaw((raw_b,))
            r._content = raw_b
            r._content_consumed = True
            return r

        def close(self) -> None:
            pass

    with tape(tape_path, mode=TapeMode.RECORD):
        s = requests.Session()
        s.mount("https://", _MockAdapter())
        assert (
            s.post("https://api.openai.com/v1/chat/completions", json={"a": 1}).status_code == 200
        )

    t0 = time.perf_counter()
    with tape(tape_path, mode=TapeMode.REPLAY, replay_latency_scale=0.5):
        s = requests.Session()
        s.mount("https://", _MockAdapter())
        assert (
            s.post("https://api.openai.com/v1/chat/completions", json={"a": 1}).status_code == 200
        )
    assert (time.perf_counter() - t0) >= 0.005


def test_cli_edge_error_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "does_not_exist"
    assert cli_main(["verify", str(missing)]) == 1
    assert cli_main(["diff", str(missing), str(tmp_path)]) == 1
    assert cli_main(["diff", str(tmp_path), str(missing)]) == 1
    assert cli_main(["prune", str(missing), "--older-than-days", "1"]) == 1
    capsys.readouterr()

    # Corrupt cassette in verify --json and prune --older-than-days (falls back to mtime)
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    bad = corrupt_dir / "broken.tape.json"
    bad.write_text("{not json", encoding="utf-8")
    assert cli_main(["verify", str(corrupt_dir), "--json"]) == 1
    v_data = json.loads(capsys.readouterr().out)
    assert len(v_data["schema_errors"]) == 1

    # Prune dry-run human output with --older-than-days 0 on corrupt file (uses mtime)
    assert cli_main(["prune", str(corrupt_dir), "--older-than-days", "0"]) == 0
    assert "WOULD DELETE" in capsys.readouterr().out
