"""Tests for production hardening: contextvars, filters, diffs, atomic save, edge cases."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from rhinotape import (
    Cassette,
    CorruptCassetteError,
    ModelPrice,
    TapeSession,
    TapeTransport,
    UnmatchedRequestError,
    estimate_cost_usd,
    tape,
)
from rhinotape.scrubbing import (
    decode_body_with_encoding,
    encode_body_with_encoding,
    scrub_url,
)


def _ok_resp(label: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-4o-mini",
            "choices": [{"message": {"role": "assistant", "content": label}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def test_nested_sessions_isolated_by_contextvars(tmp_path: Path) -> None:
    outer_path = tmp_path / "outer.tape.json"
    inner_path = tmp_path / "inner.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_resp(request.url.path)

    transport = httpx.MockTransport(handler)

    with tape(outer_path, mode="record") as outer:
        with httpx.Client(transport=transport) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"scope": "outer1"})
            with tape(inner_path, mode="record") as inner:
                client.post("https://api.openai.com/v1/chat/completions", json={"scope": "inner"})
                inner.assert_call_count(1)
            client.post("https://api.openai.com/v1/chat/completions", json={"scope": "outer2"})
        outer.assert_call_count(2)

    assert len(Cassette.load(outer_path).interactions) == 2
    assert len(Cassette.load(inner_path).interactions) == 1


@pytest.mark.asyncio
async def test_concurrent_async_sessions_do_not_clobber_each_other(tmp_path: Path) -> None:
    path_a = tmp_path / "a.tape.json"
    path_b = tmp_path / "b.tape.json"

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.01)
        return _ok_resp("ok")

    transport = httpx.MockTransport(handler)

    async def worker(cassette_path: Path, tag: str) -> None:
        async with tape(cassette_path, mode="record") as session:
            async with httpx.AsyncClient(transport=transport) as client:
                await client.post("https://api.openai.com/v1/chat/completions", json={"tag": tag})
            session.assert_call_count(1)

    await asyncio.gather(worker(path_a, "A"), worker(path_b, "B"))

    assert Cassette.load(path_a).interactions[0].request.body == {"tag": "A"}
    assert Cassette.load(path_b).interactions[0].request.body == {"tag": "B"}


def test_ignore_hosts_localhost_and_custom_predicate(tmp_path: Path) -> None:
    cassette_path = tmp_path / "filtered.tape.json"
    upstream_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        upstream_urls.append(str(request.url))
        return _ok_resp("ok")

    transport = httpx.MockTransport(handler)

    with tape(
        cassette_path,
        mode="record",
        ignore_localhost=True,
        ignore_hosts=["*.internal.svc"],
        ignore_request=lambda req: req.url.path == "/healthz",
    ) as session:
        with httpx.Client(transport=transport) as client:
            client.get("http://127.0.0.1:8000/local")
            client.get("https://db.internal.svc/query")
            client.get("https://api.openai.com/healthz")
            client.post("https://api.openai.com/v1/chat/completions", json={"real": True})
        session.assert_call_count(1)

    assert len(upstream_urls) == 4
    assert len(Cassette.load(cassette_path).interactions) == 1


def test_ignore_fields_and_query_params_for_matching(tmp_path: Path) -> None:
    cassette_path = tmp_path / "ignore_fields.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_resp("matched!")

    with tape(
        cassette_path,
        mode="once",
        ignore_fields=["metadata.trace_id", "messages.*.id"],
        ignore_query_params=["ts"],
    ):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions?ts=111",
                json={
                    "model": "gpt-4o-mini",
                    "metadata": {"trace_id": "trace-1"},
                    "messages": [{"id": "m1", "content": "hi"}],
                },
            )

    with tape(
        cassette_path,
        mode="replay",
        ignore_fields=["metadata.trace_id", "messages.*.id"],
        ignore_query_params=["ts"],
    ) as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions?ts=999",
                json={
                    "model": "gpt-4o-mini",
                    "metadata": {"trace_id": "trace-2"},
                    "messages": [{"id": "m999", "content": "hi"}],
                },
            )
            assert resp.json()["choices"][0]["message"]["content"] == "matched!"
            session.assert_call_count(1)


def test_unmatched_request_shows_diff_and_exhausted_queue_message(tmp_path: Path) -> None:
    cassette_path = tmp_path / "diff.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_resp("ok")

    with tape(cassette_path, mode="record"):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"prompt": "hello"})

    # 1. Mismatching body shows unified diff
    with tape(cassette_path, mode="replay"):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(UnmatchedRequestError) as exc_info:
                client.post("https://api.openai.com/v1/chat/completions", json={"prompt": "bye"})
            msg = str(exc_info.value)
            assert "--- recorded" in msg
            assert "+++ incoming" in msg
            assert '-  "prompt": "hello"' in msg
            assert '+  "prompt": "bye"' in msg

    # 2. Exhausted queue gives dedicated message
    with tape(cassette_path, mode="replay"):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"prompt": "hello"})
            with pytest.raises(UnmatchedRequestError, match="Replay queue exhausted"):
                client.post("https://api.openai.com/v1/chat/completions", json={"prompt": "hello"})

    # 3. allow_playback_repeats=True permits repeating the last recorded interaction
    with tape(cassette_path, mode="replay", allow_playback_repeats=True) as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"prompt": "hello"})
            client.post("https://api.openai.com/v1/chat/completions", json={"prompt": "hello"})
        session.assert_call_count(2)


def test_no_save_on_error_by_default(tmp_path: Path) -> None:
    cassette_path = tmp_path / "abort.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_resp("ok")

    with pytest.raises(RuntimeError, match="boom"):
        with tape(cassette_path, mode="once"):
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                client.post("https://api.openai.com/v1/chat/completions", json={"a": 1})
            raise RuntimeError("boom")

    assert not cassette_path.exists()


def test_binary_and_json_primitive_roundtrip_and_url_userinfo() -> None:
    raw_bin = b"\x89PNG\r\n\x1a\n\xff\xfe"
    val, enc = decode_body_with_encoding(raw_bin)
    assert enc == "base64"
    assert encode_body_with_encoding(val, enc) == raw_bin

    val_null, enc_null = decode_body_with_encoding(b"null")
    assert enc_null == "json" and val_null is None
    assert encode_body_with_encoding(val_null, enc_null) == b"null"

    val_str, enc_str = decode_body_with_encoding(b'"hello"')
    assert enc_str == "json" and val_str == "hello"
    assert encode_body_with_encoding(val_str, enc_str) == b'"hello"'

    scrubbed = scrub_url("https://alice:s3cr3t@api.example.com/v1/chat?key=abc&ts=1")
    assert "alice" not in scrubbed and "s3cr3t" not in scrubbed and "abc" not in scrubbed
    assert "REDACTED@api.example.com" in scrubbed
    # Re-scrubbing (as fingerprint_request does with ignore_query_params) must not crash urlsplit!
    rescrubbed = scrub_url(scrubbed, ignore_query_params=["ts"])
    assert "ts=" not in rescrubbed
    assert "REDACTED@api.example.com" in rescrubbed


def test_pricing_boundary_safety() -> None:
    # "gpt-4" custom entry must NOT match "gpt-4o-unknown"
    cost = estimate_cost_usd(
        "gpt-4x-experimental",
        1_000_000,
        0,
        custom_pricing={"gpt-4": ModelPrice(30.0, 60.0)},
    )
    assert cost == 0.0


def test_corrupt_cassette_and_transport_upstream_ownership(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.tape.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CorruptCassetteError):
        Cassette.load(bad_json)

    future_ver = tmp_path / "future.tape.json"
    future_ver.write_text('{"schema_version": 999, "interactions": []}', encoding="utf-8")
    with pytest.raises(CorruptCassetteError, match="Unsupported schema_version"):
        Cassette.load(future_ver)

    closed = False

    class _TrackingTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return _ok_resp("ok")

        def close(self) -> None:
            nonlocal closed
            closed = True

    upstream = _TrackingTransport()
    session = TapeSession(tmp_path / "own.tape.json", mode="none", patch_httpx=False)
    t = TapeTransport(session, upstream=upstream)
    t.close()
    assert closed is False  # Does not close user-owned upstream

    # Bool schema_version rejected, null metadata accepted, non-dict metadata rejected
    bool_ver = tmp_path / "bool_ver.tape.json"
    bool_ver.write_text('{"schema_version": true, "interactions": []}', encoding="utf-8")
    with pytest.raises(CorruptCassetteError):
        Cassette.load(bool_ver)

    null_meta = tmp_path / "null_meta.tape.json"
    null_meta.write_text(
        '{"schema_version": 2, "metadata": null, "interactions": []}', encoding="utf-8"
    )
    assert Cassette.load(null_meta).metadata == {}

    bad_meta = tmp_path / "bad_meta.tape.json"
    bad_meta.write_text(
        '{"schema_version": 2, "metadata": 123, "interactions": []}', encoding="utf-8"
    )
    with pytest.raises(CorruptCassetteError):
        Cassette.load(bad_meta)


@pytest.mark.asyncio
async def test_use_tape_decorator_sync_and_async_and_reentrancy_guard(tmp_path: Path) -> None:
    from rhinotape import TapeError, use_tape

    sync_path = tmp_path / "dec_sync.tape.json"
    async_path = tmp_path / "dec_async.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_resp("decorated")

    @use_tape(sync_path, mode="record")
    def call_sync() -> str:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            r = client.post("https://api.openai.com/v1/chat/completions", json={"s": 1})
            return str(r.json()["choices"][0]["message"]["content"])

    @use_tape(async_path, mode="record")
    async def call_async() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", json={"a": 1})
            return str(r.json()["choices"][0]["message"]["content"])

    assert call_sync() == "decorated"
    assert await call_async() == "decorated"
    assert Cassette.load(sync_path).interactions[0].request.body == {"s": 1}
    assert Cassette.load(async_path).interactions[0].request.body == {"a": 1}

    # Re-entering the same active TapeSession raises TapeError
    s = TapeSession(tmp_path / "reenter.tape.json", mode="none")
    with s:
        with pytest.raises(TapeError, match="already active"):
            with s:
                pass
