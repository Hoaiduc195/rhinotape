"""Production hardening unit and integration tests for core rhinotape modules."""

from __future__ import annotations

import asyncio
import contextvars
import gzip
import http.client
import http.server
import json
import socket
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import aiohttp
import httpx
import pytest
import requests
from aiohttp import web

import rhinotape.cassette as cassette_mod
import rhinotape.session as session_mod
from rhinotape import (
    Cassette,
    CorruptCassetteError,
    TapeAssertionError,
    TapeError,
    TapeMode,
    TapeReplayServer,
    TapeSession,
    tape,
)
from rhinotape.cassette import cassette_json_schema
from rhinotape.html_report import render_html_report
from rhinotape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    Usage,
)
from rhinotape.requests_adapter import _ReplayRequestsRaw
from rhinotape.scrubbing import DEFAULT_SENSITIVE_QUERY_PARAMS, scrub_url
from rhinotape.session import _StreamContentDecoder

# ===========================================================================
# 1. server.py (TapeReplayServer)
# ===========================================================================


def _sample_server_cassette(path: Path) -> Cassette:
    return Cassette(
        path=path,
        interactions=[
            Interaction(
                request=RecordedRequest(
                    method="POST",
                    url="https://api.openai.com/v1/chat/completions",
                    headers={"content-type": "application/json", "x-tenant": "acme"},
                    body={"model": "gpt-4o-mini", "tags": ["b", "a"]},
                    fingerprint="fp_post",
                ),
                response=RecordedResponse(
                    status_code=200,
                    headers={"content-type": "application/json"},
                    body={"ok": True},
                ),
            ),
            Interaction(
                request=RecordedRequest(
                    method="HEAD",
                    url="https://api.openai.com/v1/models",
                    headers={},
                    body=None,
                    fingerprint="fp_head",
                    body_encoding="empty",
                ),
                response=RecordedResponse(
                    status_code=200,
                    headers={"x-RateLimit": "100"},
                    body=None,
                    body_encoding="empty",
                ),
            ),
            Interaction(
                request=RecordedRequest(
                    method="OPTIONS",
                    url="https://api.openai.com/v1/chat/completions",
                    headers={},
                    body=None,
                    fingerprint="fp_options",
                    body_encoding="empty",
                ),
                response=RecordedResponse(
                    status_code=204,
                    headers={"Allow": "POST, OPTIONS", "Access-Control-Allow-Origin": "*"},
                    body=None,
                    body_encoding="empty",
                ),
            ),
            Interaction(
                request=RecordedRequest(
                    method="GET",
                    url="https://api.openai.com/x",
                    headers={},
                    body=None,
                    fingerprint="fp_get_x",
                    body_encoding="empty",
                ),
                response=RecordedResponse(
                    status_code=200,
                    headers={"content-type": "application/json"},
                    body={"route": "/x"},
                ),
            ),
        ],
    )


def test_server_param_validation(tmp_path: Path) -> None:
    c = _sample_server_cassette(tmp_path / "srv.tape.json")
    with pytest.raises(ValueError, match="max_request_bytes"):
        TapeReplayServer(c, max_request_bytes=-1)
    with pytest.raises(ValueError, match="max_request_bytes"):
        TapeReplayServer(c, max_request_bytes=True)  # type: ignore[arg-type]

    for bad_timeout in (0.0, -5.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="request_timeout"):
            TapeReplayServer(c, request_timeout=bad_timeout)
    with pytest.raises(ValueError, match="request_timeout"):
        TapeReplayServer(c, request_timeout=True)  # type: ignore[arg-type]


def test_server_max_request_bytes_content_length_413(tmp_path: Path) -> None:
    c = _sample_server_cassette(tmp_path / "srv.tape.json")
    with TapeReplayServer(c, max_request_bytes=32) as srv:
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5.0)
        try:
            big_body = json.dumps({"data": "x" * 100}).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=big_body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            assert resp.status == 413
            assert json.loads(resp.read().decode("utf-8")) == {"error": "payload_too_large"}
        finally:
            conn.close()


def test_server_chunked_encoding_valid_413_and_malformed_400(tmp_path: Path) -> None:
    c = _sample_server_cassette(tmp_path / "srv.tape.json")
    body_bytes = json.dumps({"model": "gpt-4o-mini", "tags": ["b", "a"]}).encode("utf-8")
    part1 = body_bytes[:12]
    part2 = body_bytes[12:]

    def _raw_http(port: int, raw_req: bytes) -> tuple[int, dict[str, Any]]:
        with socket.create_connection(("127.0.0.1", port), timeout=5.0) as sock:
            sock.sendall(raw_req)
            chunks: list[bytes] = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
        raw_resp = b"".join(chunks)
        head, _, payload = raw_resp.partition(b"\r\n\r\n")
        status_code = int(head.splitlines()[0].split()[1])
        return status_code, json.loads(payload.decode("utf-8"))

    with TapeReplayServer(c, max_request_bytes=128) as srv:
        # 1. Valid chunked request with chunk extension and trailer
        valid_chunked = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Type: application/json\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            + f"{len(part1):x};ext=1\r\n".encode("ascii")
            + part1
            + b"\r\n"
            + f"{len(part2):X}\r\n".encode("ascii")
            + part2
            + b"\r\n"
            + b"0\r\nX-Trailer: done\r\n\r\n"
        )
        status, data = _raw_http(srv.port, valid_chunked)
        assert status == 200
        assert data == {"ok": True}

        # 2. Chunked request exceeding max_request_bytes -> 413
        too_big = b"x" * 200
        oversized_chunked = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            + f"{len(too_big):x}\r\n".encode("ascii")
            + too_big
            + b"\r\n0\r\n\r\n"
        )
        status413, data413 = _raw_http(srv.port, oversized_chunked)
        assert status413 == 413
        assert data413 == {"error": "payload_too_large"}

        # 3. Malformed chunk size (non-hex and signed hex) -> 400
        for bad_size in (b"ZZZ\r\nabc\r\n0\r\n\r\n", b"+4\r\nabcd\r\n0\r\n\r\n"):
            bad_req = (
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"Connection: close\r\n\r\n" + bad_size
            )
            status400, _ = _raw_http(srv.port, bad_req)
            assert status400 == 400

        # 4. Missing CRLF after chunk payload -> 400
        bad_crlf = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            b"4\r\nabcdXX0\r\n\r\n"
        )
        status400_b, _ = _raw_http(srv.port, bad_crlf)
        assert status400_b == 400


def test_server_request_timeout_prevents_slowloris(tmp_path: Path) -> None:
    c = _sample_server_cassette(tmp_path / "srv.tape.json")
    with TapeReplayServer(c, request_timeout=0.15) as srv:
        t0 = time.perf_counter()
        with socket.create_connection(("127.0.0.1", srv.port), timeout=3.0) as sock:
            # Send partial HTTP request line and stall
            sock.sendall(b"POST /v1/chat/completions ")
            # Server handler should time out and close connection promptly
            data = sock.recv(1024)
            assert data == b""
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.5
        # Server remains healthy for subsequent clients
        r = httpx.get(f"{srv.url}/__tape__/health")
        assert r.status_code == 200


def test_server_head_and_options_handlers(tmp_path: Path) -> None:
    c = _sample_server_cassette(tmp_path / "srv.tape.json")
    with TapeReplayServer(c) as srv:
        with httpx.Client() as client:
            # HEAD health
            rh = client.head(f"{srv.url}/__tape__/health")
            assert rh.status_code == 200
            assert rh.content == b""

            # Matched HEAD
            rm = client.head(f"{srv.url}/v1/models")
            assert rm.status_code == 200
            assert rm.headers["x-ratelimit"] == "100"
            assert rm.content == b""

            # Unmatched HEAD -> 404 with empty body
            ru = client.head(f"{srv.url}/v1/missing")
            assert ru.status_code == 404
            assert ru.content == b""

            # Matched OPTIONS
            ro = client.options(f"{srv.url}/v1/chat/completions")
            assert ro.status_code == 204
            assert ro.headers["allow"] == "POST, OPTIONS"

            # Unmatched OPTIONS -> 404 JSON (not 501 HTML)
            rou = client.options(f"{srv.url}/v1/unknown")
            assert rou.status_code == 404
            assert rou.json()["error"] == "unmatched_request"
            assert rou.json()["method"] == "OPTIONS"


def test_server_protocol_relative_and_weird_paths_safe(tmp_path: Path) -> None:
    c = _sample_server_cassette(tmp_path / "srv.tape.json")
    with TapeReplayServer(c, extra_sensitive_query_params=["my_tok"]) as srv:
        # 1. Protocol-relative path //evil.com/x?key=sec&my_tok=custom MUST NOT match /x
        #    and MUST scrub secrets without returning protocol-relative //evil.com
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5.0)
        try:
            conn.request("GET", "//evil.com/x?key=topsecret&my_tok=tok123&ok=1")
            r1 = conn.getresponse()
            body1 = json.loads(r1.read().decode("utf-8"))
            assert r1.status == 404
            assert "topsecret" not in body1["path"]
            assert "tok123" not in body1["path"]
            assert "REDACTED" in body1["path"]
            assert not body1["path"].startswith("//")
        finally:
            conn.close()

        # 2. @evil.com/x?key=topsecret must not corrupt netloc or slice into REDACTED
        conn2 = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5.0)
        try:
            conn2.request("GET", "@evil.com/x?key=topsecret")
            r2 = conn2.getresponse()
            body2 = json.loads(r2.read().decode("utf-8"))
            assert r2.status == 404
            assert "topsecret" not in body2["path"]
            assert body2["path"].startswith("/@evil.com/x")
        finally:
            conn2.close()


def test_server_matching_and_scrubbing_optional_params(tmp_path: Path) -> None:
    c = Cassette(
        path=tmp_path / "opts.tape.json",
        interactions=[
            Interaction(
                request=RecordedRequest(
                    method="POST",
                    url="https://api.openai.com/v1/chat/completions?custom_sec=[REDACTED]&q=1",
                    headers={
                        "content-type": "application/json",
                        "x-tenant": "acme",
                        "x-custom-secret": "[REDACTED]",
                    },
                    body={"model": "gpt-4o-mini", "tags": ["alpha", "beta"]},
                    fingerprint="fp1",
                ),
                response=RecordedResponse(
                    status_code=200,
                    headers={"content-type": "application/json"},
                    body={"matched": True},
                ),
            )
        ],
    )

    with TapeReplayServer(
        c,
        match_on=("method", "path", "query", "body", "headers"),
        match_headers=("x-tenant", "x-custom-secret"),
        ignore_query_params=("nonce",),
        ignore_json_array_order=("tags",),
        extra_sensitive_headers=("x-custom-secret",),
        extra_sensitive_query_params=("custom_sec",),
    ) as srv:
        with httpx.Client() as client:
            # Different array order, extra ignored query param, live custom secrets -> matches!
            r = client.post(
                f"{srv.url}/v1/chat/completions?q=1&nonce=999&custom_sec=real_q_secret",
                headers={
                    "X-Tenant": "acme",
                    "X-Custom-Secret": "real_h_secret",
                },
                json={"model": "gpt-4o-mini", "tags": ["beta", "alpha"]},
            )
            assert r.status_code == 200
            assert r.json() == {"matched": True}

            # Wrong x-tenant header -> 404
            r_miss = client.post(
                f"{srv.url}/v1/chat/completions?q=1&custom_sec=real_q_secret",
                headers={"X-Tenant": "other", "X-Custom-Secret": "real_h_secret"},
                json={"model": "gpt-4o-mini", "tags": ["beta", "alpha"]},
            )
            assert r_miss.status_code == 404
            assert "real_q_secret" not in r_miss.json()["path"]


# ===========================================================================
# 2. cassette.py & models.py
# ===========================================================================


def test_cassette_load_streams_gzip_bounded_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gz_path = tmp_path / "bomb.tape.json.gz"
    big_payload = json.dumps(
        {"schema_version": 2, "interactions": [], "metadata": {"pad": "A" * 2048}}
    ).encode("utf-8")
    with gzip.open(gz_path, "wb") as gz:
        gz.write(big_payload)

    def _forbid_read_bytes(self: Path) -> bytes:
        raise AssertionError("Path.read_bytes() must not be called in Cassette.load()")

    monkeypatch.setattr(Path, "read_bytes", _forbid_read_bytes)

    # Normal load streams cleanly without Path.read_bytes()
    loaded = Cassette.load(gz_path)
    assert loaded.metadata["pad"] == "A" * 2048

    # Bounded streaming rejects bomb exceeding MAX_CASSETTE_BYTES
    monkeypatch.setattr(cassette_mod, "MAX_CASSETTE_BYTES", 256)
    with pytest.raises(CorruptCassetteError, match="exceeds maximum allowed size"):
        Cassette.load(gz_path)


def test_cassette_schema_and_stats_include_cache_creation_tokens(tmp_path: Path) -> None:
    schema = cassette_json_schema()
    usage_props = schema["properties"]["interactions"]["items"]["properties"]["llm"]["properties"][
        "usage"
    ]["properties"]
    assert usage_props["cache_creation_tokens"] == {"type": "integer", "minimum": 0}

    c = Cassette(
        path=tmp_path / "stats.tape.json",
        interactions=[
            Interaction(
                request=RecordedRequest(
                    "POST", "https://api.anthropic.com/v1/messages", {}, {}, "f1"
                ),
                response=RecordedResponse(200, {}, {}),
                llm=LLMCallSummary(
                    provider="anthropic",
                    model="claude-sonnet-4-20250514",
                    usage=Usage(
                        input_tokens=100,
                        output_tokens=20,
                        cached_input_tokens=50,
                        cache_creation_tokens=30,
                    ),
                ),
            )
        ],
    )
    st = c.stats()
    assert st["cache_creation_tokens"] == 30


@pytest.mark.parametrize("bad_const", ["NaN", "Infinity", "-Infinity"])
def test_cassette_from_json_rejects_non_finite_constants(bad_const: str) -> None:
    raw = (
        '{"schema_version": 2, "interactions": [{'
        '"request": {"method": "POST", "url": "https://x.test/v1", "fingerprint": "fp"}, '
        '"response": {"status_code": 200}, '
        f'"latency_ms": {bad_const}'
        "}]}"
    )
    with pytest.raises(CorruptCassetteError):
        Cassette.from_json(raw)


def test_models_reject_bool_for_int_and_validate_headers() -> None:
    # Usage token fields must reject bools
    for field_name in (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
    ):
        with pytest.raises((TypeError, ValueError)):
            Usage.from_dict({field_name: True})
        with pytest.raises((TypeError, ValueError)):
            Usage.from_dict({field_name: False})

    # RecordedResponse.status_code must reject bools
    with pytest.raises((TypeError, ValueError)):
        RecordedResponse.from_dict({"status_code": True})

    # headers: None -> {}, dict -> ok, non-dict -> TypeError/ValueError -> CorruptCassetteError
    req_null_hdr = RecordedRequest.from_dict(
        {"method": "GET", "url": "https://x.test", "headers": None, "fingerprint": "fp"}
    )
    assert req_null_hdr.headers == {}

    resp_null_hdr = RecordedResponse.from_dict({"status_code": 200, "headers": None})
    assert resp_null_hdr.headers == {}

    for bad_headers in (["ab", "cd"], "not-a-dict", 123, True):
        with pytest.raises((TypeError, ValueError)):
            RecordedRequest.from_dict(
                {
                    "method": "GET",
                    "url": "https://x.test",
                    "headers": bad_headers,
                    "fingerprint": "fp",
                }
            )
        with pytest.raises((TypeError, ValueError)):
            RecordedResponse.from_dict({"status_code": 200, "headers": bad_headers})

    # End-to-end via Cassette.from_json -> CorruptCassetteError
    bad_cassette_json = json.dumps(
        {
            "schema_version": 2,
            "interactions": [
                {
                    "request": {
                        "method": "POST",
                        "url": "https://x.test",
                        "headers": ["ab"],
                        "fingerprint": "fp",
                    },
                    "response": {"status_code": True},
                }
            ],
        }
    )
    with pytest.raises(CorruptCassetteError):
        Cassette.from_json(bad_cassette_json)


# ===========================================================================
# 3. session.py (Concurrency, NaN guards, compiled headers, StreamContentDecoder)
# ===========================================================================


def test_session_concurrent_reentry_locked(tmp_path: Path) -> None:
    sess = TapeSession(tmp_path / "reentry.tape.json", mode=TapeMode.NONE)

    # Verify _prepare() and _finalize() acquire sess._lock for _active transitions
    prepare_done = threading.Event()
    with sess._lock:
        t_prep = threading.Thread(target=lambda: (sess._prepare(), prepare_done.set()))  # type: ignore[func-returns-value]
        t_prep.start()
        assert not prepare_done.wait(timeout=0.05)
    t_prep.join(timeout=2.0)
    assert prepare_done.is_set()
    assert sess._active is True

    finalize_done = threading.Event()
    with sess._lock:
        t_fin = threading.Thread(target=lambda: (sess._finalize(), finalize_done.set()))  # type: ignore[func-returns-value]
        t_fin.start()
        assert not finalize_done.wait(timeout=0.05)
    t_fin.join(timeout=2.0)
    assert finalize_done.is_set()
    assert sess._active is False

    barrier = threading.Barrier(6)
    entered = 0
    rejected = 0
    lock = threading.Lock()

    def _worker() -> None:
        nonlocal entered, rejected
        barrier.wait()
        try:
            with sess:
                with lock:
                    entered += 1
                time.sleep(0.05)
        except TapeError:
            with lock:
                rejected += 1

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_worker) for _ in range(6)]
        for f in futs:
            f.result()

    assert entered == 1
    assert rejected == 5


@pytest.mark.asyncio
async def test_session_inflight_guardrail_race_and_slot_release_httpx(tmp_path: Path) -> None:
    cassette_path = tmp_path / "race_httpx.tape.json"
    upstream_hits = 0
    hits_lock = threading.Lock()

    def _slow_handler(req: httpx.Request) -> httpx.Response:
        nonlocal upstream_hits
        with hits_lock:
            upstream_hits += 1
        time.sleep(0.06)
        return httpx.Response(200, json={"ok": True})

    barrier = threading.Barrier(6)
    ok_count = 0
    tripped_count = 0

    with tape(cassette_path, mode=TapeMode.RECORD, max_interactions=2, save_on_error=True) as sess:
        transport = httpx.MockTransport(_slow_handler)

        def _call(idx: int) -> str:
            barrier.wait()
            with httpx.Client(transport=transport) as client:
                r = client.post("https://api.openai.com/v1/chat/completions", json={"i": idx})
                return str(r.status_code)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = [pool.submit(contextvars.copy_context().run, _call, i) for i in range(6)]
            for f in futs:
                try:
                    if f.result() == "200":
                        ok_count += 1
                except TapeAssertionError:
                    tripped_count += 1

        assert ok_count == 2
        assert tripped_count == 4
        assert upstream_hits == 2
        assert sess._inflight_recordings == 0

    # Verify slot release on:
    # (a) on_record returning None, (b) upstream error, (c) sync & async early stream close
    slot_path = tmp_path / "slot_release.tape.json"
    drop_next = True

    def _on_record(inter: Interaction) -> Interaction | None:
        nonlocal drop_next
        if drop_next:
            drop_next = False
            return None
        return inter

    class _TwoChunkSyncStream(httpx.SyncByteStream):
        def __iter__(self) -> Any:
            yield b"data: chunk1\n\n"
            yield b"data: chunk2\n\n"

    class _TwoChunkAsyncStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> Any:
            yield b"data: achunk1\n\n"
            yield b"data: achunk2\n\n"

    def _mixed_handler(req: httpx.Request) -> httpx.Response:
        kind = json.loads(req.content.decode("utf-8"))["kind"]
        if kind == "boom":
            raise httpx.ConnectError("network down", request=req)
        if kind == "stream":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_TwoChunkSyncStream(),
            )
        return httpx.Response(200, json={"ok": kind})

    async def _async_stream_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_TwoChunkAsyncStream(),
        )

    async with tape(
        slot_path, mode=TapeMode.RECORD, max_interactions=1, on_record=_on_record
    ) as sess2:
        with httpx.Client(transport=httpx.MockTransport(_mixed_handler)) as client:
            # 1. on_record returns None -> slot released, 0 recorded
            client.post("https://api.openai.com/v1/chat/completions", json={"kind": "dropped"})
            assert sess2._inflight_recordings == 0
            assert sess2.call_count == 0

            # 2. Upstream raises ConnectError -> slot released, 0 recorded
            with pytest.raises(httpx.ConnectError):
                client.post("https://api.openai.com/v1/chat/completions", json={"kind": "boom"})
            assert sess2._inflight_recordings == 0

            # 3. Sync stream closed early without EOF -> slot released, 0 recorded
            with client.stream(
                "POST", "https://api.openai.com/v1/chat/completions", json={"kind": "stream"}
            ) as sr:
                it = sr.iter_bytes()
                assert next(it) == b"data: chunk1\n\n"
            assert sess2._inflight_recordings == 0
            assert sess2.call_count == 0

        # 4. Async stream closed early without EOF -> slot released, 0 recorded
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_async_stream_handler)
        ) as aclient:
            async with aclient.stream(
                "POST", "https://api.openai.com/v1/chat/completions", json={"kind": "astream"}
            ) as asr:
                ait = asr.aiter_bytes()
                assert await anext(ait) == b"data: achunk1\n\n"
            assert sess2._inflight_recordings == 0
            assert sess2.call_count == 0

        # 5. Now 1 real interaction succeeds because all previous slots were released!
        with httpx.Client(transport=httpx.MockTransport(_mixed_handler)) as client:
            r_final = client.post("https://api.openai.com/v1/chat/completions", json={"kind": "ok"})
            assert r_final.status_code == 200
            assert sess2.call_count == 1
            assert sess2._inflight_recordings == 0


def test_session_nan_and_nonfinite_assertion_guards(tmp_path: Path) -> None:
    nan_inter = Interaction(
        request=RecordedRequest("POST", "https://api.openai.com/v1/chat/completions", {}, {}, "fp"),
        response=RecordedResponse(200, {}, {}),
        latency_ms=float("nan"),
        llm=LLMCallSummary(provider="openai", model="gpt-4o", estimated_cost_usd=float("nan")),
    )
    c = Cassette(path=tmp_path / "nan.tape.json", interactions=[nan_inter])

    sess = TapeSession(tmp_path / "nan.tape.json", mode=TapeMode.NONE)
    sess._interactions.extend(c.interactions)

    # Invalid threshold arguments -> ValueError
    for bad in (float("nan"), float("inf"), float("-inf"), -0.01):
        with pytest.raises(ValueError):
            sess.assert_cost_under(bad)
        with pytest.raises(ValueError):
            sess.assert_max_cost_per_call(bad)
        with pytest.raises(ValueError):
            sess.assert_max_latency_ms(bad)

    for bad_bool in (True, False):
        with pytest.raises(ValueError):
            sess.assert_cost_under(bad_bool)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            sess.assert_max_cost_per_call(bad_bool)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            sess.assert_max_latency_ms(bad_bool)  # type: ignore[arg-type]

    # Observed NaN cost/latency MUST fail with TapeAssertionError (not silent pass or ValueError)
    with pytest.raises(TapeAssertionError):
        sess.assert_cost_under(10.0)
    with pytest.raises(TapeAssertionError):
        sess.assert_max_cost_per_call(10.0)
    with pytest.raises(TapeAssertionError):
        sess.assert_max_latency_ms(1000.0)

    # Live max_cost_usd circuit-breaker also trips on NaN cost from on_record
    def _inject_nan_cost(inter: Interaction) -> Interaction:
        return Interaction(
            request=inter.request,
            response=inter.response,
            latency_ms=inter.latency_ms,
            llm=LLMCallSummary(estimated_cost_usd=float("nan")),
        )

    with pytest.raises(TapeAssertionError):
        with tape(
            tmp_path / "live_nan.tape.json",
            mode=TapeMode.RECORD,
            max_cost_usd=1.0,
            on_record=_inject_nan_cost,
        ):
            with httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": 1}))
            ) as client:
                client.post("https://api.openai.com/v1/chat/completions", json={"a": 1})


def test_session_response_scrub_headers_uses_precompiled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real_compile = session_mod.compile_sensitive_headers

    def _counting_compile(extra: Any = ()) -> Any:
        nonlocal calls
        calls += 1
        return real_compile(extra)

    monkeypatch.setattr(session_mod, "compile_sensitive_headers", _counting_compile)
    monkeypatch.setattr("rhinotape.scrubbing.compile_sensitive_headers", _counting_compile)

    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            headers={"X-Secret-Resp": "secret-val", "X-Public": "pub"},
            json={"ok": True},
        )
    )
    with tape(
        tmp_path / "precompiled.tape.json",
        mode=TapeMode.RECORD,
        extra_sensitive_headers=["x-secret-*"],
    ) as sess:
        assert calls == 1  # Compiled once in TapeSession.__init__
        with httpx.Client(transport=transport) as client:
            for i in range(3):
                client.post("https://api.openai.com/v1/chat/completions", json={"i": i})
        assert calls == 1  # Never recompiled per request or per response!
        assert sess.interactions[0].response.headers["x-secret-resp"] == "[REDACTED]"
        assert sess.interactions[0].response.headers["x-public"] == "pub"


def test_stream_content_decoder_bomb_multivalue_and_corrupt() -> None:
    # 1. Multi-value Content-Encoding: "gzip, identity" and "deflate, gzip"
    plain = b"data: hello multi-encoding\n\n"
    gz_bytes = gzip.compress(plain)
    dec_gz_id = _StreamContentDecoder("gzip, identity")
    assert dec_gz_id.decompress(gz_bytes) + dec_gz_id.flush() == plain

    deflate_then_gzip = gzip.compress(zlib.compress(plain))
    dec_nested = _StreamContentDecoder("deflate, gzip")
    assert dec_nested.decompress(deflate_then_gzip) + dec_nested.flush() == plain

    # 2. Bomb exceeding max_bytes raises TapeError
    bomb_payload = gzip.compress(b"A" * 4096)
    dec_bomb = _StreamContentDecoder("gzip", max_bytes=256)
    with pytest.raises(TapeError, match="exceeds maximum allowed size"):
        dec_bomb.decompress(bomb_payload)

    # 3. Invalid gzip and invalid deflate handled gracefully without raising zlib.error
    dec_bad_gz = _StreamContentDecoder("gzip")
    out_gz = dec_bad_gz.decompress(b"not actually gzip data") + dec_bad_gz.flush()
    assert out_gz == b"not actually gzip data"

    dec_bad_deflate = _StreamContentDecoder("deflate")
    out_def = dec_bad_deflate.decompress(b"\xff\xfe\xfd\xfc") + dec_bad_deflate.flush()
    assert out_def == b"\xff\xfe\xfd\xfc"


# ===========================================================================
# 4. requests_adapter.py, aiohttp_adapter.py, urllib_adapter.py
# ===========================================================================


def test_requests_and_urllib_crlf_sse_splitting(tmp_path: Path) -> None:
    crlf_sse = (
        b'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"content":"CR"}}]}\r\n\r\n'
        b'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"content":"LF!"}}]}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )

    # 1. requests non-streaming CRLF SSE
    class _ReqAdapter(requests.adapters.BaseAdapter):
        def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
            del kwargs
            resp = requests.Response()
            resp.status_code = 200
            resp.reason = "OK"
            resp.url = str(request.url)
            resp.request = request
            resp.headers = requests.structures.CaseInsensitiveDict(
                {"Content-Type": "text/event-stream"}
            )
            resp.raw = _ReplayRequestsRaw((crlf_sse,))
            resp._content = crlf_sse
            resp._content_consumed = True
            return resp

        def close(self) -> None:
            pass

    req_tape = tmp_path / "req_crlf.tape.json"
    with tape(req_tape, mode=TapeMode.RECORD) as s_req:
        rs = requests.Session()
        rs.mount("https://", _ReqAdapter())
        rs.post("https://api.openai.com/v1/chat/completions", json={"stream": True})
        assert len(s_req.interactions[0].response.stream_chunks or ()) == 3
        assert s_req.interactions[0].llm.text == "CRLF!"

    # 2. urllib CRLF SSE
    class _CRLFHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def do_POST(self) -> None:
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(crlf_sse)))
            self.end_headers()
            self.wfile.write(crlf_sse)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CRLFHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        urllib_tape = tmp_path / "urllib_crlf.tape.json"
        with tape(urllib_tape, mode=TapeMode.RECORD, patch_urllib=True) as s_url:
            ureq = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=b'{"stream":true}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(ureq) as uresp:
                _ = uresp.read()
            assert len(s_url.interactions[0].response.stream_chunks or ()) == 3
            assert s_url.interactions[0].llm.text == "CRLF!"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.asyncio
async def test_adapters_inflight_recording_slots_and_early_close(tmp_path: Path) -> None:
    # --- A. requests: concurrent max_interactions + early stream close ---
    req_hits = 0
    req_lock = threading.Lock()

    class _SlowReqAdapter(requests.adapters.BaseAdapter):
        def send(
            self, request: requests.PreparedRequest, stream: bool = False, **kwargs: Any
        ) -> requests.Response:
            del kwargs
            nonlocal req_hits
            if "fail" in str(request.url):
                raise requests.ConnectionError("boom")
            with req_lock:
                req_hits += 1
            time.sleep(0.05)
            resp = requests.Response()
            resp.status_code = 200
            resp.reason = "OK"
            resp.url = str(request.url)
            resp.request = request
            if stream:
                resp.headers = requests.structures.CaseInsensitiveDict(
                    {"Content-Type": "text/event-stream"}
                )
                resp.raw = _ReplayRequestsRaw((b"data: 1\n\n", b"data: 2\n\n"))
                resp._content = False
                resp._content_consumed = False
            else:
                raw = b'{"ok":true}'
                resp.headers = requests.structures.CaseInsensitiveDict(
                    {"Content-Type": "application/json"}
                )
                resp.raw = _ReplayRequestsRaw((raw,))
                resp._content = raw
                resp._content_consumed = True
            return resp

        def close(self) -> None:
            pass

    req_barrier = threading.Barrier(4)
    with tape(tmp_path / "req_slots.tape.json", mode=TapeMode.RECORD, max_interactions=1) as sr:
        sess = requests.Session()
        sess.mount("https://", _SlowReqAdapter())

        # Connection error releases slot
        with pytest.raises(requests.ConnectionError):
            sess.get("https://api.openai.com/fail")
        assert sr._inflight_recordings == 0

        # Early stream close releases slot
        stream_resp = sess.post(
            "https://api.openai.com/v1/chat/completions", json={"s": 1}, stream=True
        )
        assert next(stream_resp.iter_content(chunk_size=4))
        stream_resp.close()
        assert sr._inflight_recordings == 0

        # Concurrent race: 4 threads, max_interactions=1 -> only 1 admitted!
        req_hits = 0

        def _req_worker(i: int) -> int:
            req_barrier.wait()
            r = sess.post("https://api.openai.com/v1/chat/completions", json={"i": i})
            return int(r.status_code)

        ok_r = 0
        tripped_r = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            for fut in [
                pool.submit(contextvars.copy_context().run, _req_worker, i) for i in range(4)
            ]:
                try:
                    if fut.result() == 200:
                        ok_r += 1
                except TapeAssertionError:
                    tripped_r += 1
        assert ok_r == 1
        assert tripped_r == 3
        assert req_hits == 1
        assert sr._inflight_recordings == 0

    # --- B. aiohttp: concurrent max_interactions + early stream exit ---
    aio_hits = 0

    async def _aio_json(request: web.Request) -> web.Response:
        nonlocal aio_hits
        aio_hits += 1
        await asyncio.sleep(0.04)
        return web.json_response({"ok": True})

    async def _aio_sse(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b"data: chunk1\n\n")
        await asyncio.sleep(0.05)
        try:
            await resp.write(b"data: chunk2\n\n")
            await resp.write_eof()
        except ConnectionResetError:
            pass
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat", _aio_json)
    app.router.add_post("/v1/sse", _aio_sse)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    aio_port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    aio_base = f"http://127.0.0.1:{aio_port}"

    try:
        async with tape(
            tmp_path / "aio_slots.tape.json", mode=TapeMode.RECORD, max_interactions=1
        ) as sa:
            async with aiohttp.ClientSession(base_url=aio_base) as client:
                # Early exit from SSE context manager without EOF releases slot!
                async with client.post("/v1/sse", json={"s": 1}) as sresp:
                    first = await sresp.content.read(6)
                    assert first == b"data: "
                assert sa._inflight_recordings == 0
                assert sa.call_count == 0

                # Concurrent tasks with max_interactions=1 -> only 1 hits handler!
                results = await asyncio.gather(
                    *(client.post("/v1/chat", json={"i": i}) for i in range(4)),
                    return_exceptions=True,
                )
                ok_aio = [r for r in results if not isinstance(r, BaseException)]
                err_aio = [r for r in results if isinstance(r, TapeAssertionError)]
                for r in ok_aio:
                    r.release()
                assert len(ok_aio) == 1
                assert len(err_aio) == 3
                assert aio_hits == 1
                assert sa._inflight_recordings == 0
    finally:
        await runner.cleanup()

    # --- C. urllib: concurrent max_interactions + error slot release ---
    urllib_hits = 0
    urllib_lock = threading.Lock()

    class _SlowUrllibHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def do_POST(self) -> None:
            nonlocal urllib_hits
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with urllib_lock:
                urllib_hits += 1
            time.sleep(0.05)
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    u_httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SlowUrllibHandler)
    u_port = u_httpd.server_address[1]
    u_thread = threading.Thread(target=u_httpd.serve_forever, daemon=True)
    u_thread.start()
    try:
        u_barrier = threading.Barrier(4)
        with tape(
            tmp_path / "urllib_slots.tape.json",
            mode=TapeMode.RECORD,
            patch_urllib=True,
            max_interactions=1,
        ) as su:
            # Connection error to closed port releases slot
            with pytest.raises(urllib.error.URLError):
                urllib.request.urlopen("http://127.0.0.1:1/unreachable", timeout=0.5)
            assert su._inflight_recordings == 0

            def _u_worker(i: int) -> int:
                u_barrier.wait()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{u_port}/v1/chat",
                    data=json.dumps({"i": i}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req) as resp:
                    _ = resp.read()
                    return int(resp.status)

            ok_u = 0
            tripped_u = 0
            with ThreadPoolExecutor(max_workers=4) as pool:
                for fut in [
                    pool.submit(contextvars.copy_context().run, _u_worker, i) for i in range(4)
                ]:
                    try:
                        if fut.result() == 200:
                            ok_u += 1
                    except TapeAssertionError:
                        tripped_u += 1
            assert ok_u == 1
            assert tripped_u == 3
            assert urllib_hits == 1
            assert su._inflight_recordings == 0
    finally:
        u_httpd.shutdown()
        u_httpd.server_close()


# ===========================================================================
# 5. scrubbing.py & html_report.py
# ===========================================================================


def test_scrubbing_default_sensitive_query_params() -> None:
    expected = {
        "secret",
        "client_secret",
        "sig",
        "signature",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
    }
    assert expected.issubset(DEFAULT_SENSITIVE_QUERY_PARAMS)

    raw_url = (
        "https://storage.example.com/bucket/obj?"
        "X-Amz-Credential=AKIA123&X-Amz-Signature=deadbeef&"
        "X-Amz-Security-Token=sts_tok&client_secret=cs123&"
        "secret=s1&sig=s2&signature=s3&keep=ok"
    )
    scrubbed = scrub_url(raw_url)
    for leaked in ("AKIA123", "deadbeef", "sts_tok", "cs123", "=s1", "=s2", "=s3"):
        assert leaked not in scrubbed
    assert "keep=ok" in scrubbed


def test_html_report_latency_guards_badges_and_truncation(tmp_path: Path) -> None:
    huge_text = "T" * 10000
    huge_body = {"payload": "B" * 5000}

    c1 = Cassette(
        path=tmp_path / "alpha.tape.json",
        interactions=[
            Interaction(
                request=RecordedRequest(
                    "POST", "https://api.openai.com/v1/chat", {}, huge_body, "f1"
                ),
                response=RecordedResponse(200, {}, huge_body),
                latency_ms=-25.0,
                llm=LLMCallSummary(provider="openai", model="gpt-4o", text=huge_text),
            ),
            Interaction(
                request=RecordedRequest("POST", "https://api.openai.com/v1/chat", {}, {}, "f2"),
                response=RecordedResponse(200, {}, {}),
                latency_ms=float("nan"),
                llm=LLMCallSummary(provider="openai", model="gpt-4o", text="ok"),
            ),
        ],
    )
    c2 = Cassette(
        path=tmp_path / "beta.tape.json",
        interactions=[
            Interaction(
                request=RecordedRequest(
                    "POST", "https://api.anthropic.com/v1/messages", {}, {}, "f3"
                ),
                response=RecordedResponse(200, {}, {}),
                latency_ms=float("inf"),
                llm=LLMCallSummary(provider="anthropic", model="claude-sonnet-4", text="hi"),
            ),
        ],
    )

    # Single cassette: no cassette filename badge, truncated body, capped data-search, safe widths
    single_html = render_html_report([c1], max_body_chars=500)
    assert "<span class='badge'>alpha.tape.json</span>" not in single_html
    assert "width:-" not in single_html
    assert "width:nan%" not in single_html.lower()
    assert "width:inf%" not in single_html.lower()
    assert "truncated" in single_html
    assert "B" * 1000 not in single_html

    # data-search must be capped (e.g. <= ~2000 chars, not 10,000 't's)
    assert "t" * 3000 not in single_html

    # Multi-cassette: shows cassette filename badges on cards
    multi_html = render_html_report([c1, c2], max_body_chars=500)
    assert "<span class='badge'>alpha.tape.json</span>" in multi_html
    assert "<span class='badge'>beta.tape.json</span>" in multi_html
    assert "width:inf%" not in multi_html.lower()
    assert "width:nan%" not in multi_html.lower()
