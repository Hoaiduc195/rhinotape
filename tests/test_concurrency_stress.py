"""Concurrency, thread-safety, nesting, and mixed-client stress tests."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
import httpx
import pytest
import requests

from llm_tape import Cassette, TapeMode, TapeTransport, tape


def test_multithreaded_httpx_and_requests_patch_refcount_isolation(tmp_path: Path) -> None:
    orig_httpx = httpx.Client.send
    orig_requests = requests.Session.send

    shared_tape = tmp_path / "shared.tape.json"
    upstream = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "shared-ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )
    )
    with tape(shared_tape, mode=TapeMode.RECORD, patch_httpx=False) as rec:
        with httpx.Client(transport=TapeTransport(rec, upstream=upstream)) as c:
            c.post("https://api.openai.com/v1/chat/completions", json={"shared": True})

    def worker(idx: int) -> tuple[int, str, str]:
        with tape(
            shared_tape,
            mode=TapeMode.REPLAY,
            allow_playback_repeats=True,
        ) as s:
            r_httpx = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                json={"shared": True},
            )
            r_req = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json={"shared": True},
            )
            s.assert_call_count(2)
            return (
                idx,
                r_httpx.json()["choices"][0]["message"]["content"],
                r_req.json()["choices"][0]["message"]["content"],
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, range(16)))

    assert len(results) == 16
    for _, h_text, r_text in results:
        assert h_text == "shared-ok"
        assert r_text == "shared-ok"

    # Global monkey-patches must be restored cleanly after all threads exit
    assert httpx.Client.send is orig_httpx
    assert requests.Session.send is orig_requests


@pytest.mark.asyncio
async def test_high_concurrency_async_mixed_httpx_and_aiohttp_replay(tmp_path: Path) -> None:
    orig_async_httpx = httpx.AsyncClient.send
    orig_aiohttp = aiohttp.ClientSession._request

    tapes: list[Path] = []
    for idx in range(12):
        tp = tmp_path / f"task_{idx}.tape.json.gz"
        tapes.append(tp)

        def _make_handler(i: int) -> httpx.MockTransport:
            return httpx.MockTransport(
                lambda req, i=i: httpx.Response(  # type: ignore[misc]
                    200,
                    json={
                        "model": "gpt-4o-mini",
                        "choices": [{"message": {"role": "assistant", "content": f"task-{i}"}}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                    },
                )
            )

        with tape(tp, mode=TapeMode.RECORD, patch_httpx=False) as rec:
            with httpx.Client(transport=TapeTransport(rec, upstream=_make_handler(idx))) as c:
                c.post("https://api.openai.com/v1/chat/completions", json={"id": idx})

    async def run_task(idx: int) -> tuple[str, str]:
        async with tape(tapes[idx], mode=TapeMode.REPLAY, allow_playback_repeats=True) as s:
            await asyncio.sleep(0.001 * (idx % 4))
            async with httpx.AsyncClient() as hc:
                hr = await hc.post("https://api.openai.com/v1/chat/completions", json={"id": idx})
            async with aiohttp.ClientSession() as ac:
                async with ac.post(
                    "https://api.openai.com/v1/chat/completions", json={"id": idx}
                ) as ar:
                    ad = await ar.json()
            s.assert_call_count(2)
            return hr.json()["choices"][0]["message"]["content"], ad["choices"][0]["message"][
                "content"
            ]

    outputs = await asyncio.gather(*(run_task(i) for i in range(12)))
    for i, (h_out, a_out) in enumerate(outputs):
        assert h_out == f"task-{i}"
        assert a_out == f"task-{i}"

    assert httpx.AsyncClient.send is orig_async_httpx
    assert aiohttp.ClientSession._request is orig_aiohttp


def test_nested_exception_unwinding_and_cross_client_gzip_sse(tmp_path: Path) -> None:
    outer_tape = tmp_path / "outer.tape.json.gz"
    inner_tape = tmp_path / "inner.tape.json.gz"
    secret = "sk-proj-SPLITSECRET1234567890ABCD"

    full_event = (
        f'data: {{"model":"gpt-4o-mini","choices":[{{"delta":{{"content":"key {secret}"}}}}]}}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    split_pos = full_event.index(secret.encode()) + 15

    class _SplitStream(httpx.SyncByteStream):
        def __iter__(self):  # type: ignore[no-untyped-def]
            yield full_event[:split_pos]
            yield full_event[split_pos:]

    upstream = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_SplitStream(),
        )
    )

    with tape(
        outer_tape,
        mode=TapeMode.RECORD,
        redact_patterns=[r"sk-proj-[A-Za-z0-9]+"],
        patch_httpx=False,
    ) as outer:
        with httpx.Client(transport=TapeTransport(outer, upstream=upstream)) as c:
            with c.stream(
                "POST", "https://api.openai.com/v1/chat/completions", json={"outer": True}
            ) as r:
                r.read()

        with pytest.raises(RuntimeError, match="inner boom"):
            with tape(inner_tape, mode=TapeMode.RECORD, save_on_error=False):
                raise RuntimeError("inner boom")

    assert outer_tape.exists()
    assert not inner_tape.exists()
    saved = Cassette.load(outer_tape)
    assert secret not in saved.interactions[0].llm.text
    assert secret[:15] not in "".join(saved.interactions[0].response.stream_chunks or ())
    assert secret[15:] not in "".join(saved.interactions[0].response.stream_chunks or ())

    # Replay the exact same gzipped SSE cassette via `requests`
    with tape(outer_tape, mode=TapeMode.REPLAY):
        with requests.post(
            "https://api.openai.com/v1/chat/completions",
            json={"outer": True},
            stream=True,
        ) as resp:
            replayed = b"".join(resp.iter_content(chunk_size=64)).decode("utf-8")
    assert "[REDACTED]" in replayed
    assert secret[15:] not in replayed
