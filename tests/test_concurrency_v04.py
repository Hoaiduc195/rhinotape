"""Thread-safety and concurrency stress tests (Wave 102)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from rhinotape import (
    Cassette,
    MemoryCassetteStore,
    TapeMode,
    TapeReplayServer,
    tape,
)


def test_memory_cassette_store_concurrent_rw() -> None:
    store = MemoryCassetteStore(compress=True)

    def _worker(idx: int) -> str:
        key = f"worker_{idx % 4}.tape.json"
        c = Cassette(path=Path(key), interactions=[], metadata={"worker": idx})
        store.save(key, c)
        loaded = store.load(key)
        assert isinstance(loaded.metadata, dict)
        return key

    with ThreadPoolExecutor(max_workers=12) as pool:
        keys = list(pool.map(_worker, range(48)))
    assert len(keys) == 48
    assert len(store.list_keys()) == 4


def test_replay_server_concurrent_clients(tmp_path: Path) -> None:
    cassette_path = tmp_path / "concurrent.tape.json"
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "chatcmpl-c",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD):
        with httpx.Client(transport=transport) as client:
            client.post("https://api.openai.com/v1/chat/completions", json=body)

    with TapeReplayServer(cassette_path, allow_playback_repeats=True) as srv:
        url = f"{srv.url}/v1/chat/completions"

        def _hit(_: int) -> int:
            with httpx.Client() as client:
                r = client.post(url, json=body, timeout=5.0)
                return r.status_code

        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(_hit, range(40)))

        assert all(s == 200 for s in statuses)
        stats = srv.stats()
        assert stats["served_requests"] == 40
        assert stats["unmatched_requests"] == 0


def test_concurrent_isolated_tape_sessions_in_memory() -> None:
    store = MemoryCassetteStore()

    def _run_session(idx: int) -> int:
        key = f"iso_{idx}.tape.json"
        t = httpx.MockTransport(
            lambda r, n=idx: httpx.Response(
                200,
                json={
                    "id": f"c-{n}",
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"role": "assistant", "content": f"reply-{n}"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )
        )
        with tape(key, mode=TapeMode.RECORD, store=store, max_cost_usd=1.0) as sess:
            with httpx.Client(transport=t) as client:
                r = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": str(idx)}],
                    },
                )
                assert r.status_code == 200
            return sess.call_count

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(_run_session, range(16)))
    assert counts == [1] * 16
    assert len(store.list_keys()) == 16
