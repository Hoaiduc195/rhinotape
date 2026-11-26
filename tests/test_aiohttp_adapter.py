"""Tests for optional async `aiohttp` adapter (JSON, SSE streaming, scrubbing, concurrency)."""

from __future__ import annotations

import asyncio
import importlib
import time
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from rhinotape import Cassette, TapeMode, tape
from rhinotape.aiohttp_adapter import (
    _ReplayAiohttpStreamReader,
    acquire_aiohttp_patch,
    release_aiohttp_patch,
)


async def _start_local_server() -> tuple[web.AppRunner, str, dict[str, int]]:
    hits = {"count": 0}

    async def handle_chat(request: web.Request) -> web.Response:
        hits["count"] += 1
        body = await request.json()
        content = body.get("messages", [{}])[0].get("content", "hi")
        return web.json_response(
            {
                "id": "chatcmpl-aio",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": f"echo:{content}"}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
            }
        )

    async def handle_sse(request: web.Request) -> web.StreamResponse:
        hits["count"] += 1
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        chunks = [
            b'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"content":"Hi "}}]}\n\n',
            b'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"content":"aio!"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        for c in chunks:
            await resp.write(c)
        await resp.write_eof()
        return resp

    async def handle_error(request: web.Request) -> web.Response:
        del request
        hits["count"] += 1
        return web.json_response({"error": "rate_limit"}, status=429)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_post("/v1/chat/sse", handle_sse)
    app.router.add_get("/v1/error", handle_error)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}", hits


@pytest.mark.asyncio
async def test_aiohttp_json_record_replay_scrubbing_and_errors(tmp_path: Path) -> None:
    runner, base_url, hits = await _start_local_server()
    orig_req = aiohttp.ClientSession._request
    try:
        tape_path = tmp_path / "aio_json.tape.json"
        async with tape(tape_path, mode=TapeMode.RECORD) as rec:
            async with aiohttp.ClientSession(
                base_url=base_url,
                headers={"Authorization": "Bearer sk-aio-secret-1234567890"},
            ) as client:
                async with client.post(
                    "/v1/chat/completions",
                    params={"api_key": "secret_q"},
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "yo"}]},
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["choices"][0]["message"]["content"] == "echo:yo"
                async with client.get("/v1/error") as err_resp:
                    assert err_resp.status == 429
            rec.assert_call_count(2)
            assert rec.total_tokens == 20

        assert aiohttp.ClientSession._request is orig_req
        assert hits["count"] == 2

        saved = Cassette.load(tape_path)
        assert saved.interactions[0].request.headers["Authorization"] == "[REDACTED]"
        assert "secret_q" not in saved.interactions[0].request.url

        # Replay offline (server hits must NOT increment)
        async with tape(tape_path, mode=TapeMode.REPLAY) as rep:
            async with aiohttp.ClientSession(
                base_url=base_url,
                headers={"Authorization": "Bearer sk-other"},
            ) as client:
                async with client.post(
                    "/v1/chat/completions",
                    params={"api_key": "other_q"},
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "yo"}]},
                ) as resp2:
                    assert resp2.status == 200
                    data2 = await resp2.json()
                    assert data2["choices"][0]["message"]["content"] == "echo:yo"
                async with client.get("/v1/error") as err_resp2:
                    assert err_resp2.status == 429
                    with pytest.raises(aiohttp.ClientResponseError):
                        err_resp2.raise_for_status()
            rep.assert_call_count(2)
            rep.assert_no_unused_interactions()

        assert hits["count"] == 2
        assert aiohttp.ClientSession._request is orig_req
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_aiohttp_sse_stream_record_replay_and_concurrency(tmp_path: Path) -> None:
    runner, base_url, hits = await _start_local_server()
    try:
        sse_tape = tmp_path / "aio_sse.tape.json"
        async with tape(sse_tape, mode=TapeMode.RECORD) as rec:
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    f"{base_url}/v1/chat/sse",
                    json={"model": "gpt-4o-mini", "stream": True},
                ) as resp:
                    lines = [
                        line.decode("utf-8").strip() async for line in resp.content if line.strip()
                    ]
            assert len(lines) == 3
            rec.assert_call_count(1)
            assert rec.interactions[0].llm.text == "Hi aio!"
            assert rec.interactions[0].llm.is_stream is True

        # Replay SSE with latency scale
        t0 = time.perf_counter()
        async with tape(sse_tape, mode=TapeMode.REPLAY, replay_latency_scale=0.1) as rep:
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    f"{base_url}/v1/chat/sse",
                    json={"model": "gpt-4o-mini", "stream": True},
                ) as resp2:
                    lines2 = [
                        line.decode("utf-8").strip() async for line in resp2.content if line.strip()
                    ]
            assert lines2 == lines
            rep.assert_call_count(1)
        assert (time.perf_counter() - t0) >= 0.0
        assert hits["count"] == 1

        # Concurrent isolated async sessions
        async def run_worker(name: str) -> str:
            tpath = tmp_path / f"worker_{name}.tape.json"
            async with tape(tpath, mode=TapeMode.RECORD):
                async with aiohttp.ClientSession() as client:
                    async with client.post(
                        f"{base_url}/v1/chat/completions",
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": name}],
                        },
                    ) as r:
                        d = await r.json()
                        return str(d["choices"][0]["message"]["content"])

        res_a, res_b = await asyncio.gather(run_worker("alpha"), run_worker("beta"))
        assert res_a == "echo:alpha"
        assert res_b == "echo:beta"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_aiohttp_stream_reader_helpers_and_missing_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _ReplayAiohttpStreamReader((b"line1\nli", b"ne2\n"))
    assert reader.at_eof() is False
    assert await reader.read(0) == b""
    assert await reader.readline() == b"line1\n"
    chunks = [c async for c in reader.iter_chunked(3)]
    assert b"".join(chunks) == b"line2\n"
    assert reader.at_eof() is True

    reader2 = _ReplayAiohttpStreamReader((b"abc", b"def"))
    any_chunks = [c async for c in reader2.iter_any()]
    assert any_chunks == [b"abc", b"def"]

    real_import_module = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "aiohttp":
            raise ImportError("No module named 'aiohttp'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert acquire_aiohttp_patch(lambda: None) is False
    release_aiohttp_patch()


@pytest.mark.asyncio
async def test_aiohttp_rv_review_regressions(tmp_path: Path) -> None:
    runner, base_url, _hits = await _start_local_server()
    try:
        # P1 #1 & P2 #3: raise_for_status + RequestInfo formatting + .ok + .json content_type
        err_tape = tmp_path / "err_rfs.tape.json"
        async with tape(err_tape, mode=TapeMode.RECORD):
            async with aiohttp.ClientSession(base_url=base_url) as client:
                async with client.get("/v1/error") as r:
                    assert r.status == 429

        async with tape(err_tape, mode=TapeMode.REPLAY, allow_playback_repeats=True):
            async with aiohttp.ClientSession(base_url=base_url, raise_for_status=True) as client:
                with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                    async with client.get("/v1/error"):
                        pass
                # str(exc) must format cleanly without AttributeError
                assert "429" in str(exc_info.value)

            async with aiohttp.ClientSession(base_url=base_url) as client:
                async with client.get("/v1/error") as r2:
                    assert r2.ok is False
                    with pytest.raises(aiohttp.ContentTypeError):
                        await r2.json(content_type="text/plain")
                    # content_type=None skips check
                    assert (await r2.json(content_type=None))["error"] == "rate_limit"

        # P1 #2: readany(), readchunk(), iter_chunks() record & replay SSE properly
        sse_tape = tmp_path / "sse_chunks.tape.json"
        async with tape(sse_tape, mode=TapeMode.RECORD) as rec:
            async with aiohttp.ClientSession(base_url=base_url) as client:
                async with client.post("/v1/chat/sse", json={"stream": True}) as r:
                    assert r.ok is True
                    got = b"".join([c async for c, _ in r.content.iter_chunks()])
                    assert b"Hi " in got and b"aio!" in got
            rec.assert_call_count(1)
            assert rec.interactions[0].llm.text == "Hi aio!"

        async with tape(sse_tape, mode=TapeMode.REPLAY, allow_playback_repeats=True):
            async with aiohttp.ClientSession(base_url=base_url) as client:
                async with client.post("/v1/chat/sse", json={"stream": True}) as r:
                    assert r.ok is True
                    got2 = b"".join([c async for c, _ in r.content.iter_chunks()])
                    assert got2 == got
                async with client.post("/v1/chat/sse", json={"stream": True}) as r2:
                    first = await r2.content.readany()
                    assert first
    finally:
        await runner.cleanup()
