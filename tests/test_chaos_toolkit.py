"""Tests for the Replay Chaos Toolkit (fail_nth, fail_first, latency, corrupt, truncate, chain)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from llm_tape import (
    TapeMode,
    chain_replay_hooks,
    corrupt_json_body,
    fail_first,
    fail_nth,
    inject_latency,
    tape,
    truncate_stream,
)


def _record_basic(cassette_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "chatcmpl-chaos",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD):
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )


def test_fail_first_and_fail_nth_with_repeats(tmp_path: Path) -> None:
    cassette_path = tmp_path / "retry.tape.json"
    _record_basic(cassette_path)

    with tape(
        cassette_path,
        mode=TapeMode.REPLAY,
        allow_playback_repeats=True,
        on_replay=fail_first(2, status_code=429, headers={"Retry-After": "1"}),
    ):
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            r0 = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )
            r1 = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )
            r2 = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r0.status_code == 429
            assert r0.headers["Retry-After"] == "1"
            assert r1.status_code == 429
            assert r2.status_code == 200

    with tape(
        cassette_path,
        mode=TapeMode.REPLAY,
        allow_playback_repeats=True,
        on_replay=fail_nth(1, status_code=502, body={"error": "bad gateway"}),
    ):
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            a = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )
            b = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert a.status_code == 200
            assert b.status_code == 502
            assert b.json() == {"error": "bad gateway"}


def test_corrupt_json_inject_latency_and_chain(tmp_path: Path) -> None:
    cassette_path = tmp_path / "corrupt.tape.json"
    _record_basic(cassette_path)

    hook = chain_replay_hooks(
        inject_latency(250.0, nth=0),
        corrupt_json_body(nth=0),
    )
    with tape(cassette_path, mode=TapeMode.REPLAY, on_replay=hook) as session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 200
            with pytest.raises(json.JSONDecodeError):
                resp.json()
        assert session.interactions[0].latency_ms >= 250.0


def test_truncate_stream_and_validation(tmp_path: Path) -> None:
    cassette_path = tmp_path / "stream.tape.json"

    class _MultiChunkStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"c"}}]}\n\n'
            yield b"data: [DONE]\n\n"

    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_MultiChunkStream(),
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD):
        with httpx.Client(transport=transport) as client:
            with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "stream": True},
            ) as resp:
                resp.read()

    with tape(cassette_path, mode=TapeMode.REPLAY, on_replay=truncate_stream(1)):
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "stream": True},
            ) as resp:
                body = resp.read().decode("utf-8")
                assert '"a"' in body
                assert '"b"' not in body
                assert "[DONE]" not in body

    # Failing an SSE interaction normalizes Content-Type to application/json
    with tape(cassette_path, mode=TapeMode.REPLAY, on_replay=fail_nth(0, status_code=429)):
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "stream": True},
            )
            assert resp.status_code == 429
            assert resp.headers["content-type"] == "application/json"
            assert "error" in resp.json()

    with pytest.raises(ValueError):
        fail_nth(-1)
    with pytest.raises(ValueError):
        fail_first(-1)
    with pytest.raises(ValueError):
        inject_latency(-1.0)
    with pytest.raises(ValueError):
        inject_latency(float("nan"))
    with pytest.raises(ValueError):
        corrupt_json_body(nth=-1)
    with pytest.raises(ValueError):
        truncate_stream(-1)
