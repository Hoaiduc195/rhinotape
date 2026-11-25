"""Performance regression and optimization tests (Batch 15, Waves 88-91)."""

from __future__ import annotations

import time

import httpx

from llm_tape import MemoryCassetteStore, RecordedRequest, TapeMode, ToolCall, tape
from llm_tape.parsers import _detect_provider
from llm_tape.requests_adapter import _ReplayRequestsRaw
from llm_tape.scrubbing import REDACTED, compile_sensitive_headers, scrub_headers
from llm_tape.session import _ReplayByteStream


def test_direct_to_dict_and_compiled_headers() -> None:
    tc = ToolCall(name="search", arguments={"q": "python"}, call_id="c1")
    assert tc.to_dict() == {"name": "search", "arguments": {"q": "python"}, "call_id": "c1"}

    req = RecordedRequest(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        headers={"content-type": "application/json"},
        body={"model": "gpt-4o"},
        fingerprint="abc123",
    )
    assert req.to_dict()["fingerprint"] == "abc123"
    assert req.to_dict()["body_encoding"] == "json"

    compiled = compile_sensitive_headers(["x-internal-*", "x-custom-secret"])
    cleaned = scrub_headers(
        {
            "Authorization": "Bearer tok",
            "X-Internal-Key": "k1",
            "X-Custom-Secret": "k2",
            "Accept": "application/json",
        },
        compiled=compiled,
    )
    assert cleaned["Authorization"] == REDACTED
    assert cleaned["X-Internal-Key"] == REDACTED
    assert cleaned["X-Custom-Secret"] == REDACTED
    assert cleaned["Accept"] == "application/json"


def test_detect_provider_fast_paths() -> None:
    assert _detect_provider("https://api.anthropic.com/v1/messages", {}) == "anthropic"
    assert _detect_provider("https://generativelanguage.googleapis.com/v1/models/x", {}) == "gemini"
    assert (
        _detect_provider("https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse", {})
        == "bedrock"
    )
    assert _detect_provider("http://localhost:11434/api/chat", {}) == "ollama"
    assert _detect_provider("https://api.cohere.com/v2/chat", {}) == "cohere"
    assert _detect_provider("https://api.openai.com/v1/chat/completions", {}) == "openai"


def test_lazy_stream_encoding_and_replay_throughput() -> None:
    chunks = ("data: hello\n\n", "data: world\n\n")
    bs = _ReplayByteStream(chunks)
    assert bs._chunks is chunks
    assert list(bs) == [b"data: hello\n\n", b"data: world\n\n"]

    raw = _ReplayRequestsRaw(chunks)
    assert list(raw.stream()) == [b"data: hello\n\n", b"data: world\n\n"]

    store = MemoryCassetteStore()
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "1",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )
    )
    with tape("bench.tape.json", mode=TapeMode.RECORD, store=store):
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )

    t0 = time.perf_counter()
    with tape(
        "bench.tape.json",
        mode=TapeMode.REPLAY,
        store=store,
        allow_playback_repeats=True,
    ) as session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            for _ in range(200):
                r = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
                )
                assert r.status_code == 200
        assert session.call_count == 200
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0
