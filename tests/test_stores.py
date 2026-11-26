"""Tests for in-memory Cassette serialization and pluggable CassetteStore backends."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from rhinotape import (
    Cassette,
    CassetteStore,
    CorruptCassetteError,
    FileCassetteStore,
    MemoryCassetteStore,
    TapeMode,
    TapeNotFoundError,
    tape,
)


def _mock_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "chatcmpl-store",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "stored ok"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            },
        )
    )


def test_cassette_to_from_json_and_bytes(tmp_path: Path) -> None:
    p = tmp_path / "ser.tape.json"
    with tape(p, mode=TapeMode.RECORD):
        with httpx.Client(transport=_mock_transport()) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )

    orig = Cassette.load(p)
    raw_json = orig.to_json()
    from_j = Cassette.from_json(raw_json)
    assert len(from_j.interactions) == 1
    assert from_j.interactions[0].llm.text == "stored ok"

    raw_gz = orig.to_bytes(compress=True)
    from_b = Cassette.from_bytes(raw_gz, path="inmem.tape.json.gz")
    assert from_b.interactions[0].llm.text == "stored ok"

    with pytest.raises(CorruptCassetteError, match="safety size limit"):
        Cassette.from_bytes(orig.to_bytes(compress=False), max_bytes=10)


def test_memory_cassette_store_with_tape_session() -> None:
    store = MemoryCassetteStore(compress=True)
    assert isinstance(store, CassetteStore)
    assert not store.exists("agent/turn1.tape.json")

    with pytest.raises(TapeNotFoundError):
        store.load("agent/turn1.tape.json")

    # Record directly into MemoryCassetteStore (no disk I/O!)
    with tape("agent/turn1.tape.json", mode=TapeMode.ONCE, store=store):
        with httpx.Client(transport=_mock_transport()) as client:
            r = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
            )
            assert r.status_code == 200

    assert store.exists("agent/turn1.tape.json")
    assert store.list_keys() == ["agent/turn1.tape.json"]

    # Replay from MemoryCassetteStore
    with tape("agent/turn1.tape.json", mode=TapeMode.REPLAY, store=store) as session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            r = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
            )
            assert r.status_code == 200
            assert session.call_count == 1

    assert store.delete("agent/turn1.tape.json") is True
    assert store.delete("agent/turn1.tape.json") is False
    store.clear()
    assert store.list_keys() == []

    # MemoryCassetteStore(compress=False) with .gz key must round-trip cleanly
    plain_store = MemoryCassetteStore(compress=False)
    with tape("compressed.tape.json.gz", mode=TapeMode.RECORD, store=plain_store):
        with httpx.Client(transport=_mock_transport()) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "gz"}]},
            )
    assert plain_store.load("compressed.tape.json.gz").interactions[0].llm.text == "stored ok"


def test_file_cassette_store_and_path_traversal_guard(tmp_path: Path) -> None:
    store = FileCassetteStore(tmp_path)
    assert isinstance(store, CassetteStore)

    with tape("sub/demo.tape.json", mode=TapeMode.RECORD, store=store):
        with httpx.Client(transport=_mock_transport()) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]},
            )

    assert store.exists("sub/demo.tape.json")
    assert store.list_keys() == ["sub/demo.tape.json"]
    loaded = store.load("sub/demo.tape.json")
    assert loaded.interactions[0].llm.text == "stored ok"

    with pytest.raises(ValueError, match="escapes root directory"):
        store.exists("../outside.tape.json")
    with pytest.raises(ValueError, match="escapes root directory"):
        store.exists(".")
    with pytest.raises(ValueError, match="must be relative"):
        store.exists("/etc/passwd")

    assert store.delete("sub/demo.tape.json") is True
    assert store.delete("sub/demo.tape.json") is False
