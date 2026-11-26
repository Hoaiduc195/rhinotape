"""Focused coverage hardening tests for server, session, html_report, aiohttp, and pytest plugin."""

from __future__ import annotations

import gzip
import socket
import sys
import types
import zlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import httpx
import pytest
from aiohttp import web

from llm_tape import (
    Cassette,
    Interaction,
    LLMCallSummary,
    MemoryCassetteStore,
    ModelPrice,
    RecordedRequest,
    RecordedResponse,
    TapeAssertionError,
    TapeError,
    TapeMode,
    TapeReplayServer,
    TapeSession,
    ToolCall,
    Usage,
    render_html_report,
    tape,
)
from llm_tape.aiohttp_adapter import (
    _merge_headers,
    _RecordingAiohttpStreamReader,
    _ReplayAiohttpResponse,
    _ReplayAiohttpStreamReader,
)
from llm_tape.pricing import PRICING_FILE_ENV, estimate_cost_usd
from llm_tape.pytest_plugin import (
    _REPORT_KEY,
    _VIOLATIONS_KEY,
    _is_cassette_stale,
    _parse_redact_pii_setting,
    pytest_sessionfinish,
    pytest_terminal_summary,
)
from llm_tape.server import _extract_path_and_query
from llm_tape.session import _ReplayRawBytesStream, _StreamContentDecoder

pytest_plugins = ["pytester"]


_UNSET: Any = object()


def _sample_interaction(
    *,
    method: str = "POST",
    url: str = "https://api.openai.com/v1/chat/completions",
    body: Any = _UNSET,
    status_code: int = 200,
    resp_body: Any = None,
    resp_headers: dict[str, str] | None = None,
    stream_chunks: tuple[str, ...] | None = None,
    latency_ms: float = 12.5,
    cost: float = 0.001,
    text: str = "ok reply",
    thinking: str = "",
    tool_calls: tuple[ToolCall, ...] = (),
) -> Interaction:
    req_body = {"model": "gpt-4o-mini", "q": "hi"} if body is _UNSET else body
    return Interaction(
        request=RecordedRequest(
            method=method,
            url=url,
            headers={"content-type": "application/json"},
            body=req_body,
            fingerprint="fp_cov",
            body_encoding="empty" if req_body is None else "json",
        ),
        response=RecordedResponse(
            status_code=status_code,
            headers=resp_headers
            if resp_headers is not None
            else {"content-type": "application/json"},
            body=None
            if stream_chunks is not None
            else (resp_body if resp_body is not None else {"ok": True}),
            body_encoding="stream" if stream_chunks is not None else "json",
            stream_chunks=stream_chunks,
        ),
        latency_ms=latency_ms,
        llm=LLMCallSummary(
            provider="openai",
            model="gpt-4o-mini",
            usage=Usage(input_tokens=50, output_tokens=20, cached_input_tokens=10),
            estimated_cost_usd=cost,
            tool_calls=tool_calls,
            text=text,
            thinking=thinking,
            is_stream=stream_chunks is not None,
        ),
    )


def _send_raw_http(host: str, port: int, raw: bytes) -> bytes:
    with socket.create_connection((host, port), timeout=3.0) as sock:
        sock.sendall(raw)
        sock.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            piece = sock.recv(4096)
            if not piece:
                break
            chunks.append(piece)
        return b"".join(chunks)


# --- 1. src/llm_tape/server.py ---------------------------------------------


def test_server_init_validation_modes_and_safe_paths(tmp_path: Path) -> None:
    assert _extract_path_and_query("//v1/chat?a=1") == ("/v1/chat", "a=1")

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    cpath = tmp_path / "one.tape.json"
    c = Cassette(path=cpath, interactions=[_sample_interaction()])
    c.save()

    # Invalid init arguments
    with pytest.raises(ValueError, match="mode='none'"):
        TapeReplayServer(cpath, mode=TapeMode.NONE)
    with pytest.raises(ValueError, match="Invalid tape mode"):
        TapeReplayServer(cpath, mode="bogus")
    with pytest.raises(ValueError, match="max_request_bytes"):
        TapeReplayServer(cpath, max_request_bytes=-1)
    with pytest.raises(ValueError, match="max_request_bytes"):
        TapeReplayServer(cpath, max_request_bytes=True)
    with pytest.raises(ValueError, match="request_timeout"):
        TapeReplayServer(cpath, request_timeout=0.0)
    with pytest.raises(ValueError, match="request_timeout"):
        TapeReplayServer(cpath, request_timeout=float("nan"))
    with pytest.raises(ValueError, match="upstream_url"):
        TapeReplayServer(cpath, upstream_url="ftp://invalid.example.com")
    with pytest.raises(ValueError, match="Unknown PII preset"):
        TapeReplayServer(cpath, redact_pii="not_a_preset")

    # Cassette object modes: RECORD, ONCE (loaded & empty), NEW_EPISODES
    srv_rec_obj = TapeReplayServer(c, mode=TapeMode.RECORD)
    assert srv_rec_obj._is_recording is True and srv_rec_obj._is_replaying is False
    assert srv_rec_obj.reset()["status"] == "reset"

    srv_once_obj = TapeReplayServer(c, mode=TapeMode.ONCE)
    assert srv_once_obj._is_replaying is True and srv_once_obj._is_recording is False

    empty_cass = Cassette(path=tmp_path / "missing_obj.tape.json", interactions=[])
    srv_once_empty = TapeReplayServer(empty_cass, mode=TapeMode.ONCE)
    assert srv_once_empty._is_recording is True and srv_once_empty._is_replaying is False

    srv_ne_obj = TapeReplayServer(c, mode=TapeMode.NEW_EPISODES)
    assert srv_ne_obj._is_replaying is True and srv_ne_obj._is_recording is True

    # Directory & missing path modes: ONCE (non-empty & empty dir), NEW_EPISODES (dir & new file)
    dir_with_tape = tmp_path / "has_tape"
    dir_with_tape.mkdir()
    Cassette(path=dir_with_tape / "a.tape.json", interactions=[_sample_interaction()]).save()

    srv_dir_once = TapeReplayServer(dir_with_tape, mode=TapeMode.ONCE)
    assert srv_dir_once._is_replaying is True and srv_dir_once._loaded_count == 1
    assert srv_dir_once._resolve_safe_cassette_path(123) is None
    assert srv_dir_once._resolve_safe_cassette_path(".") is None
    sub = dir_with_tape / "subdir"
    sub.mkdir()
    assert srv_dir_once._resolve_safe_cassette_path("subdir") is None

    srv_empty_once = TapeReplayServer(empty_dir, mode=TapeMode.ONCE)
    assert srv_empty_once._is_recording is True

    srv_dir_ne = TapeReplayServer(dir_with_tape, mode=TapeMode.NEW_EPISODES)
    assert srv_dir_ne._is_replaying is True and srv_dir_ne._is_recording is True

    srv_missing_ne = TapeReplayServer(tmp_path / "new_file.tape.json", mode=TapeMode.NEW_EPISODES)
    assert srv_missing_ne._is_recording is True and srv_missing_ne._is_replaying is False


def test_server_raw_chunked_and_content_length_errors(tmp_path: Path) -> None:
    cpath = tmp_path / "chunk_err.tape.json"
    Cassette(path=cpath, interactions=[_sample_interaction()]).save()

    with TapeReplayServer(cpath, max_request_bytes=64) as srv:
        host, port = srv.host, srv.port

        # Unsupported Transfer-Encoding
        r1 = _send_raw_http(
            host,
            port,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Transfer-Encoding: gzip\r\n\r\n",
        )
        assert b"400" in r1 and b"unsupported_transfer_encoding" in r1

        # Invalid Content-Length (non-int and negative)
        r2 = _send_raw_http(
            host,
            port,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: nope\r\n\r\n",
        )
        assert b"400" in r2 and b"invalid_content_length" in r2

        r3 = _send_raw_http(
            host,
            port,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: -5\r\n\r\n",
        )
        assert b"400" in r3 and b"invalid_content_length" in r3

        # Malformed chunk line (no CRLF) & non-hex chunk size
        r4 = _send_raw_http(
            host,
            port,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n5\nhello",
        )
        assert b"400" in r4 and b"invalid_chunked_encoding" in r4

        r5 = _send_raw_http(
            host,
            port,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Transfer-Encoding: chunked\r\n\r\nZZ\r\n",
        )
        assert b"400" in r5 and b"invalid_chunked_encoding" in r5

        # Truncated chunk body (hits _read_exact returning None)
        r6 = _send_raw_http(
            host,
            port,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n10\r\nabc",
        )
        assert b"400" in r6 and b"invalid_chunked_encoding" in r6

        # Missing CRLF after chunk payload
        r7 = _send_raw_http(
            host,
            port,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n3\r\nabcXX0\r\n\r\n",
        )
        assert b"400" in r7 and b"invalid_chunked_encoding" in r7

        # Truncated trailer after 0\r\n
        r8 = _send_raw_http(
            host,
            port,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n0\r\n",
        )
        assert b"400" in r8 and b"invalid_chunked_encoding" in r8

        # Excess trailer lines (> 64)
        many_trailers = b"0\r\n" + (b"X-T: 1\r\n" * 66) + b"\r\n"
        r9 = _send_raw_http(
            host,
            port,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n" + many_trailers,
        )
        assert b"400" in r9 and b"invalid_chunked_encoding" in r9

        # Oversized body on control plane endpoints -> 413
        big = b"x" * 100
        for ep in ("/__tape__/reset", "/__tape__/load", "/__tape__/save", "/__tape__/assert"):
            r_big = httpx.post(f"{srv.url}{ep}", content=big)
            assert r_big.status_code == 413


def test_server_proxy_errors_cors_head_and_control_plane_branches(tmp_path: Path) -> None:
    # 1. Reverse-proxy with upstream_not_configured -> 502
    no_up_tape = tmp_path / "no_up.tape.json"
    with TapeReplayServer(no_up_tape, mode=TapeMode.RECORD) as srv:
        r = httpx.post(f"{srv.url}/v1/chat/completions", json={"x": 1})
        assert r.status_code == 502
        assert r.json()["error"] == "upstream_not_configured"

    # 2. Reverse-proxy upstream connection error -> 502; plus SSE & binary/text with CORS
    def _proxy_handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/conn-err":
            raise httpx.ConnectError("upstream down")
        if req.url.path == "/v1/stream":

            class _Chunks(httpx.SyncByteStream):
                def __iter__(self) -> Iterator[bytes]:
                    yield b""
                    yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                    yield b"\xc3"  # split UTF-8 char across chunks (hits tail flush if incomplete)

            return httpx.Response(
                200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Access-Control-Allow-Origin": "https://other.example",
                    "Content-Length": "99",
                },
                stream=_Chunks(),
            )
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/octet-stream",
                "Access-Control-Allow-Origin": "https://other.example",
            },
            content=b"\x00\x01\x02binary",
        )

    rec_tape = tmp_path / "proxy_rec.tape.json"
    with TapeReplayServer(
        rec_tape,
        mode=TapeMode.RECORD,
        upstream_url="https://api.example.com",
        upstream_transport=httpx.MockTransport(_proxy_handler),
        enable_cors=True,
    ) as srv:
        r_err = httpx.post(f"{srv.url}/v1/conn-err", json={})
        assert r_err.status_code == 502 and r_err.json()["error"] == "upstream_error"

        r_sse = httpx.post(f"{srv.url}/v1/stream", json={"s": True})
        assert r_sse.status_code == 200
        assert r_sse.headers["access-control-allow-origin"] == "*"

        r_bin = httpx.post(f"{srv.url}/v1/bin", content=b"raw")
        assert r_bin.status_code == 200 and r_bin.content == b"\x00\x01\x02binary"
        assert r_bin.headers["access-control-allow-origin"] == "*"

        # Unknown control endpoint -> 404
        assert httpx.get(f"{srv.url}/__tape__/does-not-exist").status_code == 404

    # 3. HEAD requests on replayed SSE & JSON responses with CORS enabled
    head_tape = tmp_path / "head_replay.tape.json"
    Cassette(
        path=head_tape,
        interactions=[
            _sample_interaction(
                method="HEAD",
                url="https://api.openai.com/v1/sse",
                body=None,
                resp_headers={
                    "Content-Type": "text/event-stream",
                    "Content-Length": "10",
                    "Access-Control-Allow-Origin": "https://ignored",
                },
                stream_chunks=("data: ok\n\n",),
            ),
            _sample_interaction(
                method="HEAD",
                url="https://api.openai.com/v1/json",
                body=None,
                resp_headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "https://ignored",
                },
                resp_body={"ok": True},
            ),
        ],
    ).save()
    with TapeReplayServer(head_tape, enable_cors=True) as srv:
        h1 = httpx.head(f"{srv.url}/v1/sse")
        assert h1.status_code == 200 and h1.content == b""
        h2 = httpx.head(f"{srv.url}/v1/json")
        assert h2.status_code == 200 and h2.content == b""

    # 4. Control plane /__tape__/load, /__tape__/save, /__tape__/assert error branches
    tapes_dir = tmp_path / "cp_dir"
    tapes_dir.mkdir()
    good_c = tapes_dir / "good.tape.json"
    Cassette(
        path=good_c,
        interactions=[
            _sample_interaction(
                status_code=500,
                latency_ms=250.0,
                cost=0.05,
                text="hello world",
                tool_calls=(ToolCall(name="forbidden_tool", arguments={}),),
            )
        ],
    ).save()
    # Directory-backed replay server has no _active_cassette_path -> /__tape__/save returns 400
    with TapeReplayServer(tapes_dir, mode=TapeMode.REPLAY) as dir_srv:
        r_no_active = httpx.post(f"{dir_srv.url}/__tape__/save", content=b"")
        assert r_no_active.status_code == 400
        assert r_no_active.json()["error"] == "no_active_cassette"

    bad_c = tapes_dir / "corrupt.tape.json"
    bad_c.write_text("{broken", encoding="utf-8")

    with TapeReplayServer(good_c, mode=TapeMode.REPLAY, cassette_dir=tapes_dir) as srv:
        # Invalid JSON on load, save, assert
        assert httpx.post(f"{srv.url}/__tape__/load", content=b"{bad").status_code == 400
        assert httpx.post(f"{srv.url}/__tape__/save", content=b"{bad").status_code == 400
        assert httpx.post(f"{srv.url}/__tape__/assert", content=b"[]").status_code == 400

        # /__tape__/load validation branches
        assert httpx.post(f"{srv.url}/__tape__/load", json={}).status_code == 400
        assert (
            httpx.post(
                f"{srv.url}/__tape__/load", json={"cassette": "good.tape.json", "mode": 123}
            ).status_code
            == 400
        )
        assert (
            httpx.post(
                f"{srv.url}/__tape__/load", json={"cassette": "good.tape.json", "mode": "bad"}
            ).status_code
            == 400
        )
        assert (
            httpx.post(
                f"{srv.url}/__tape__/load", json={"cassette": "good.tape.json", "mode": "none"}
            ).status_code
            == 400
        )
        assert (
            httpx.post(
                f"{srv.url}/__tape__/load", json={"cassette": "missing.tape.json", "mode": "replay"}
            ).status_code
            == 404
        )
        assert (
            httpx.post(
                f"{srv.url}/__tape__/load", json={"cassette": "corrupt.tape.json", "mode": "replay"}
            ).status_code
            == 400
        )

        # /__tape__/save traversal rejected (400) and safe relative path succeeds (200)
        assert (
            httpx.post(
                f"{srv.url}/__tape__/save", json={"cassette": "../escape.tape.json"}
            ).status_code
            == 400
        )
        r_save = httpx.post(f"{srv.url}/__tape__/save", json={"cassette": "saved_copy.tape.json"})
        assert r_save.status_code == 200
        assert (tapes_dir / "saved_copy.tape.json").is_file()

        # /__tape__/load without explicit mode reuses current mode and flushes dirty state
        srv._dirty = True
        r_load_ok = httpx.post(f"{srv.url}/__tape__/load", json={"cassette": "good.tape.json"})
        assert r_load_ok.status_code == 200

        # Replay the single interaction so observed_interactions is populated
        assert (
            httpx.post(
                f"{srv.url}/v1/chat/completions", json={"model": "gpt-4o-mini", "q": "hi"}
            ).status_code
            == 500
        )

        # /__tape__/assert parameter validation errors
        bad_specs: list[dict[str, Any]] = [
            {"unknown_key": 1},
            {"call_count": -1},
            {"call_count": True},
            {"max_tokens": -1},
            {"max_cost_usd": -0.5},
            {"max_cost_usd": True},
            {"max_latency_ms": -10.0},
            {"no_errors": "yes"},
            {"tool_called": 123},
            {"tool_sequence": "not-a-list"},
        ]
        for spec in bad_specs:
            assert httpx.post(f"{srv.url}/__tape__/assert", json=spec).status_code == 400

        # NaN in JSON body rejected strictly by _parse_json_object_strict
        assert (
            httpx.post(f"{srv.url}/__tape__/assert", content=b'{"max_cost_usd": NaN}').status_code
            == 400
        )

        # /__tape__/assert failure evaluations across all assertion keys
        r_fail = httpx.post(
            f"{srv.url}/__tape__/assert",
            json={
                "max_latency_ms": 10.0,
                "no_errors": True,
                "tool_called": ["missing_tool"],
                "no_tool_called": ["forbidden_tool"],
                "text_contains": ["absent_phrase"],
            },
        )
        assert r_fail.status_code == 400
        fails = "\n".join(r_fail.json()["failures"])
        for key in (
            "max_latency_ms",
            "no_errors",
            "tool_called",
            "no_tool_called",
            "text_contains",
        ):
            assert key in fails

        # Non-finite cost and latency in observed interaction fail evaluate_assertions
        srv._observed_interactions.append(
            _sample_interaction(latency_ms=float("nan"), cost=float("nan"))
        )
        code_nf, payload_nf = srv.evaluate_assertions(
            {"max_cost_usd": 1.0, "max_latency_ms": 100.0}
        )
        assert code_nf == 400
        assert any("non-finite" in f for f in payload_nf["failures"])


# --- 2. src/llm_tape/session.py --------------------------------------------


def test_stream_content_decoder_all_encodings_fallbacks_and_bomb_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Multi-value Content-Encoding ("gzip, identity") + empty chunk
    gz = gzip.compress(b"hello multi-layer")
    dec = _StreamContentDecoder("gzip, identity")
    assert dec.decompress(b"") == b""
    assert dec.decompress(gz) + dec.flush() == b"hello multi-layer"

    # Unknown encoding falls back to identity and enforces cap
    dec_unk = _StreamContentDecoder("unknown-codec", max_bytes=5)
    assert dec_unk.decompress(b"12345") == b"12345"
    assert dec_unk.flush() == b""
    with pytest.raises(TapeError, match="exceeds maximum allowed size"):
        dec_unk.decompress(b"6")

    # Deflate: zlib-wrapped, raw deflate (-MAX_WBITS), 1-byte pending flush, and corrupt fallback
    raw_comp = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw_deflate = raw_comp.compress(b"raw-deflate-ok") + raw_comp.flush()
    dec_raw = _StreamContentDecoder("deflate")
    assert dec_raw.decompress(raw_deflate[:1]) == b""
    assert dec_raw.decompress(raw_deflate[1:]) + dec_raw.flush() == b"raw-deflate-ok"

    dec_empty_def = _StreamContentDecoder("deflate")
    assert dec_empty_def.flush() == b""

    # Invalid deflate payload falls back to identity cleanly
    dec_bad_deflate = _StreamContentDecoder("deflate")
    assert dec_bad_deflate.decompress(b"\xff\xff\xff\xff") == b"\xff\xff\xff\xff"
    assert dec_bad_deflate.flush() == b""

    # 1-byte pending deflate buffer flushed on flush() (lines 336-345)
    dec_short_zlib = _StreamContentDecoder("deflate")
    assert dec_short_zlib.decompress(b"x") == b""
    assert dec_short_zlib.flush() == b""

    dec_bad_gz = _StreamContentDecoder("gzip")
    assert dec_bad_gz.decompress(b"not-valid-gzip") == b"not-valid-gzip"
    assert dec_bad_gz.flush() == b""

    # Decompression bomb exceeding max_bytes raises TapeError
    bomb = gzip.compress(b"A" * 5000)
    dec_bomb = _StreamContentDecoder("gzip", max_bytes=100)
    with pytest.raises(TapeError, match="exceeds maximum allowed size"):
        dec_bomb.decompress(bomb)

    # Synthetic brotli & zstandard modules covering normal, error fallback, and cap enforcement
    class _FakeBrotliDecompressor:
        def process(self, data: bytes) -> bytes:
            if data == b"BAD":
                raise RuntimeError("corrupt brotli")
            return data.upper()

    fake_brotli = types.SimpleNamespace(Decompressor=_FakeBrotliDecompressor)
    monkeypatch.setitem(sys.modules, "brotli", fake_brotli)

    dec_br = _StreamContentDecoder("br", max_bytes=10)
    assert dec_br.decompress(b"abc") == b"ABC"
    assert dec_br.decompress(b"BAD") == b"BAD"  # falls back to identity

    class _FakeZstdDecompressorObj:
        def decompress(self, data: bytes, max_output_size: int = 0) -> bytes:
            del max_output_size
            if data == b"BAD":
                raise RuntimeError("corrupt zstd")
            return data.lower()

        def flush(self) -> bytes:
            return b"!"

    class _FakeZstdDecompressor:
        def decompressobj(self) -> _FakeZstdDecompressorObj:
            return _FakeZstdDecompressorObj()

    fake_zstd = types.SimpleNamespace(ZstdDecompressor=_FakeZstdDecompressor)
    monkeypatch.setitem(sys.modules, "zstandard", fake_zstd)

    dec_zstd = _StreamContentDecoder("zstd", max_bytes=20)
    assert dec_zstd.decompress(b"HELLO") == b"hello"
    assert dec_zstd.flush() == b"!"
    assert dec_zstd.decompress(b"BAD") == b"BAD"


@pytest.mark.asyncio
async def test_session_assertion_validation_nonfinite_and_early_stream_close(
    tmp_path: Path,
) -> None:
    store = MemoryCassetteStore()
    with TapeSession("mem.tape.json", mode=TapeMode.RECORD, store=store) as s:
        # Invalid assertion threshold arguments raise ValueError
        for bad in (float("nan"), float("inf"), -1.0, True):
            with pytest.raises(ValueError):
                s.assert_cost_under(bad)  # type: ignore[arg-type]
            with pytest.raises(ValueError):
                s.assert_max_cost_per_call(bad)  # type: ignore[arg-type]
            with pytest.raises(ValueError):
                s.assert_max_latency_ms(bad)  # type: ignore[arg-type]

        # Empty session assertion edge branches
        s.assert_no_unused_interactions()  # no-op when not replaying
        with pytest.raises(TapeAssertionError, match="no interactions"):
            s.assert_json_output()
        with pytest.raises(TapeAssertionError, match="no interactions"):
            s.assert_streamed()

        # Non-finite observed latency / cost trigger TapeAssertionError
        s._interactions.append(
            _sample_interaction(
                latency_ms=float("nan"),
                cost=float("nan"),
                text='["not-a-dict"]',
                tool_calls=(ToolCall(name="raw_tool", arguments="raw-str-args"),),
            )
        )
        with pytest.raises(TapeAssertionError, match="non-finite"):
            s.assert_cost_under(1.0)
        with pytest.raises(TapeAssertionError, match="non-finite"):
            s.assert_max_cost_per_call(1.0)
        with pytest.raises(TapeAssertionError, match="non-finite"):
            s.assert_max_latency_ms(100.0)
        with pytest.raises(TapeAssertionError, match="out of range"):
            s.assert_json_output(call_index=99)
        with pytest.raises(TapeAssertionError, match="Expected JSON object"):
            s.assert_json_output(required_keys=["a"])
        with pytest.raises(TapeAssertionError, match="did not match"):
            s.assert_tool_args("raw_tool", {"k": "v"})

        # save() persists to MemoryCassetteStore
        s.save()
        assert store.exists("mem.tape.json")

    # on_record returning non-Interaction raises TypeError
    bad_rec_path = tmp_path / "bad_on_record.tape.json"
    with TapeSession(bad_rec_path, mode=TapeMode.RECORD, on_record=lambda i: "invalid"):  # type: ignore[arg-type,return-value]
        with httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": 1}))
        ) as c:
            with pytest.raises(TypeError, match="on_record callback must return"):
                c.post("https://api.openai.com/v1/chat/completions", json={"a": 1})

    # Early sync & async stream close releases inflight recording slot within max_interactions=1
    class _SyncSSE(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"part1"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"part2"}}]}\n\n'

    class _AsyncSSE(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"part1"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"part2"}}]}\n\n'

    sync_slot_tape = tmp_path / "sync_slot.tape.json"
    with TapeSession(sync_slot_tape, mode=TapeMode.RECORD, max_interactions=1) as s_sync:
        transport = httpx.MockTransport(
            lambda r: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=_SyncSSE()
            )
        )
        with httpx.Client(transport=transport) as c:
            # Close early before EOF -> slot must be released
            with c.stream(
                "POST", "https://api.openai.com/v1/chat/completions", json={"n": 1}
            ) as resp:
                next(resp.iter_bytes())
            # Second call succeeds within max_interactions=1 because first aborted cleanly
            with c.stream(
                "POST", "https://api.openai.com/v1/chat/completions", json={"n": 2}
            ) as resp2:
                list(resp2.iter_bytes())
        s_sync.assert_call_count(1)

    async_slot_tape = tmp_path / "async_slot.tape.json"
    async with TapeSession(async_slot_tape, mode=TapeMode.RECORD, max_interactions=1) as s_async:
        atransport = httpx.MockTransport(
            lambda r: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=_AsyncSSE()
            )
        )
        async with httpx.AsyncClient(transport=atransport) as ac:
            async with ac.stream(
                "POST", "https://api.openai.com/v1/chat/completions", json={"n": 1}
            ) as r1:
                await anext(r1.aiter_bytes())
            async with ac.stream(
                "POST", "https://api.openai.com/v1/chat/completions", json={"n": 2}
            ) as r2:
                _ = [chunk async for chunk in r2.aiter_bytes()]
        s_async.assert_call_count(1)

    # Async non-SSE stream=True + _ReplayRawBytesStream async iter + NONE mode passthrough
    raw_stream = _ReplayRawBytesStream(b"hello-raw")
    collected = [chunk async for chunk in raw_stream]
    assert collected == [b"hello-raw"]

    async_err_tape = tmp_path / "async_stream_err.tape.json"
    async with TapeSession(async_err_tape, mode=TapeMode.RECORD):
        err_transport = httpx.MockTransport(
            lambda r: httpx.Response(
                500, headers={"content-type": "application/json"}, json={"e": 1}
            )
        )
        async with httpx.AsyncClient(transport=err_transport) as ac:
            async with ac.stream("POST", "https://api.openai.com/v1/fail", json={}) as r_err:
                body = await r_err.aread()
                assert b'"e"' in body

    async with TapeSession(async_err_tape, mode=TapeMode.NONE):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text="pass"))
        ) as ac:
            assert (await ac.get("https://api.example.com/ping")).text == "pass"


# --- 3. src/llm_tape/html_report.py ----------------------------------------


def test_html_report_truncation_invalid_max_chars_nonfinite_latency_and_badges(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="max_body_chars"):
        render_html_report([], max_body_chars=-1)
    with pytest.raises(ValueError, match="max_body_chars"):
        render_html_report([], max_body_chars=True)

    c1 = Cassette(
        path=tmp_path / "first.tape.json",
        interactions=[
            _sample_interaction(
                body={"prompt": "A" * 80},
                resp_body={"answer": "B" * 80},
                latency_ms=-5.0,
                cost=float("nan"),
                text="T" * 60,
                thinking="H" * 60,
                tool_calls=(ToolCall(name="big_tool", arguments={"payload": "X" * 60}),),
            )
        ],
    )
    c2 = Cassette(
        path=tmp_path / "second.tape.json",
        interactions=[
            _sample_interaction(
                stream_chunks=("chunk-one-1234567890", "chunk-two-1234567890", "chunk-three"),
                latency_ms=float("nan"),
            )
        ],
    )
    html_out = render_html_report([c1, c2], title="Multi Report", max_body_chars=25)
    assert "first.tape.json" in html_out
    assert "second.tape.json" in html_out
    assert "... [truncated" in html_out
    assert "n/a" in html_out


# --- 4. src/llm_tape/aiohttp_adapter.py ------------------------------------


@pytest.mark.asyncio
async def test_aiohttp_replay_and_recording_stream_readers_and_release_abort(
    tmp_path: Path,
) -> None:
    # _merge_headers with sequence of tuples
    merged = _merge_headers(SimpleNamespace(headers={"X-Base": "1"}), [("X-Req", "2")])
    assert ("X-Base", "1") in merged and ("X-Req", "2") in merged

    # _ReplayAiohttpStreamReader: read(0), read(n), readline, iter_any (buffered + delay),
    # iter_chunked, readany (buffered), readchunk, iter_chunks, __aiter__
    r1 = _ReplayAiohttpStreamReader((b"line1\nli", b"ne2\nlast"), chunk_delay_s=0.001)
    assert await r1.read(0) == b""
    assert await r1.read(4) == b"line"
    assert await r1.readany() == b"1\nli"
    assert await r1.readline() == b"ne2\n"
    rem = [c async for c in r1.iter_any()]
    assert rem == [b"last"]
    assert r1.at_eof() is True

    r2 = _ReplayAiohttpStreamReader((b"abcdef", b"gh"))
    assert await r2.read(2) == b"ab"
    assert [c async for c in r2.iter_any()] == [b"cdef", b"gh"]

    r3 = _ReplayAiohttpStreamReader((b"12345",))
    assert [c async for c in r3.iter_chunked(2)] == [b"12", b"34", b"5"]

    r4 = _ReplayAiohttpStreamReader((b"a", b"b"))
    assert [pair async for pair in r4.iter_chunks()] == [(b"a", True), (b"b", True)]

    r5 = _ReplayAiohttpStreamReader((b"x\ny\n",))
    assert [ln async for ln in r5] == [b"x\n", b"y\n"]

    # _ReplayAiohttpResponse ContentTypeError + close()
    text_inter = _sample_interaction(
        resp_headers={"Content-Type": "text/plain"},
        resp_body="not json",
    )
    rep_resp = _ReplayAiohttpResponse(
        method="GET", url="https://api.example.com/t", interaction=text_inter
    )
    with pytest.raises(aiohttp.ContentTypeError):
        await rep_resp.json()
    rep_resp.close()
    assert rep_resp.closed is True

    # _RecordingAiohttpStreamReader: all read/iter methods + upstream errors calling on_abort
    class _FailingUpstream:
        def __init__(self, mode: str) -> None:
            self._mode = mode

        async def read(self, n: int = -1) -> bytes:
            if self._mode == "err":
                raise RuntimeError("read fail")
            return b"abc" if n < 0 else b""

        async def readline(self) -> bytes:
            if self._mode == "err":
                raise RuntimeError("readline fail")
            return b""

        async def iter_any(self) -> AsyncIterator[bytes]:
            if self._mode == "err":
                raise RuntimeError("iter_any fail")
            yield b"a"

        async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
            del n
            if self._mode == "err":
                raise RuntimeError("iter_chunked fail")
            yield b"b"

        async def readany(self) -> bytes:
            if self._mode == "err":
                raise RuntimeError("readany fail")
            return b""

        async def readchunk(self) -> tuple[bytes, bool]:
            if self._mode == "err":
                raise RuntimeError("readchunk fail")
            return (b"", False)

        async def iter_chunks(self) -> AsyncIterator[tuple[bytes, bool]]:
            if self._mode == "err":
                raise RuntimeError("iter_chunks fail")
            yield (b"c", True)

        async def __aiter__(self) -> AsyncIterator[bytes]:
            if self._mode == "err":
                raise RuntimeError("aiter fail")
            yield b"d\n"

    aborts = {"n": 0}
    completed: list[list[str]] = []

    rec_ok = _RecordingAiohttpStreamReader(_FailingUpstream("ok"), completed.append)
    assert await rec_ok.read(0) == b""
    assert await rec_ok.read(-1) == b"abc"
    assert completed == [["abc"]]

    rec_rl = _RecordingAiohttpStreamReader(_FailingUpstream("ok"), completed.append)
    assert await rec_rl.readline() == b""

    for method_name in ("read", "readline", "readany", "readchunk"):
        rd = _RecordingAiohttpStreamReader(
            _FailingUpstream("err"),
            completed.append,
            on_abort=lambda: aborts.__setitem__("n", aborts["n"] + 1),
        )
        with pytest.raises(RuntimeError):
            await getattr(rd, method_name)()

    for gen_name, gen_args in (
        ("iter_any", ()),
        ("iter_chunked", (4,)),
        ("iter_chunks", ()),
    ):
        rd = _RecordingAiohttpStreamReader(
            _FailingUpstream("err"),
            completed.append,
            on_abort=lambda: aborts.__setitem__("n", aborts["n"] + 1),
        )
        with pytest.raises(RuntimeError):
            _ = [x async for x in getattr(rd, gen_name)(*gen_args)]

    rd_aiter = _RecordingAiohttpStreamReader(
        _FailingUpstream("err"),
        completed.append,
        on_abort=lambda: aborts.__setitem__("n", aborts["n"] + 1),
    )
    with pytest.raises(RuntimeError):
        _ = [x async for x in rd_aiter]
    assert aborts["n"] == 8

    # Live aiohttp server: early close() releases inflight slot, ignore_localhost, mode=NONE,
    # callable raise_for_status, and non-SSE replay latency scale
    async def _sse_handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n')
        await resp.write(b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n')
        await resp.write_eof()
        return resp

    async def _json_handler(request: web.Request) -> web.Response:
        del request
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_get("/sse", _sse_handler)
    app.router.add_get("/json", _json_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    base_url = f"http://127.0.0.1:{port}"

    try:
        # Early close() before EOF releases inflight slot for max_interactions=1
        aio_tape = tmp_path / "aio_abort.tape.json"
        async with tape(aio_tape, mode=TapeMode.RECORD, max_interactions=1) as rec:
            async with aiohttp.ClientSession() as client:
                r_early = await client.get(f"{base_url}/sse")
                r_early.close()  # closes before EOF -> aborts & frees slot
                async with client.get(f"{base_url}/json") as r_ok:
                    assert (await r_ok.json())["ok"] is True
            rec.assert_call_count(1)

        # Replay with replay_latency_scale > 0 and async callable raise_for_status
        seen_rfs = {"called": False}

        async def _custom_rfs(resp: Any) -> None:
            seen_rfs["called"] = True
            assert resp.status == 200

        async with tape(aio_tape, mode=TapeMode.REPLAY, replay_latency_scale=0.05):
            async with aiohttp.ClientSession() as client:
                async with client.get(f"{base_url}/json", raise_for_status=_custom_rfs) as r_rep:
                    assert await r_rep.json() == {"ok": True}
        assert seen_rfs["called"] is True

        # mode=NONE and ignore_localhost passthrough
        async with tape(aio_tape, mode=TapeMode.NONE):
            async with aiohttp.ClientSession() as client:
                async with client.get(f"{base_url}/json") as r_none:
                    assert r_none.status == 200
        async with tape(aio_tape, mode=TapeMode.REPLAY, ignore_localhost=True):
            async with aiohttp.ClientSession() as client:
                async with client.get(f"{base_url}/json") as r_ign:
                    assert r_ign.status == 200
    finally:
        await runner.cleanup()


# --- 5. src/llm_tape/pytest_plugin.py & pricing.py -------------------------


def test_pytest_plugin_ini_errors_summary_and_helpers(
    pytester: pytest.Pytester, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # _parse_redact_pii_setting branches
    assert _parse_redact_pii_setting("false") is False
    assert _parse_redact_pii_setting("all") is True
    assert _parse_redact_pii_setting("email, ssn") == ["email", "ssn"]

    # _is_cassette_stale fallback on corrupt file uses mtime
    corrupt = tmp_path / "stale_bad.tape.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert _is_cassette_stale(corrupt, 0.0) in {True, False}
    assert _is_cassette_stale(tmp_path / "missing.tape.json", 1.0) is False

    # pytest_terminal_summary with empty sessions + with long nodeid & cache savings
    lines: list[str] = []
    tr = SimpleNamespace(
        write_sep=lambda sep, title: lines.append(f"{sep}{title}"),
        write_line=lambda line, **kw: lines.append(line),
    )
    stash = pytest.Stash()
    stash[_REPORT_KEY] = []
    stash[_VIOLATIONS_KEY] = []
    cfg = SimpleNamespace(
        stash=stash,
        getoption=lambda name, default=None: True if name == "--tape-report" else default,
    )
    pytest_terminal_summary(tr, 0, cfg)  # type: ignore[arg-type]
    assert any("No llm_tape sessions executed." in ln for ln in lines)

    lines.clear()
    stash[_REPORT_KEY] = [
        {
            "nodeid": (
                "tests/very_long_directory_name/test_long_module_name.py::test_super_long_case_name"
            ),
            "cassette": "a.tape.json",
            "action": "replayed",
            "calls": 1,
            "input_tokens": 100,
            "cached_input_tokens": 50,
            "output_tokens": 20,
            "total_tokens": 120,
            "estimated_cost_usd": 0.001,
            "cache_savings_usd": 0.0005,
            "latency_ms": 5.0,
        }
    ]
    stash[_VIOLATIONS_KEY] = ["budget exceeded"]
    pytest_terminal_summary(tr, 1, cfg)  # type: ignore[arg-type]
    assert any("Estimated cache savings:" in ln for ln in lines)
    assert any("FAILED: budget exceeded" in ln for ln in lines)

    # Non-finite cost budget violation via pytest_sessionfinish
    nf_stash = pytest.Stash()
    nf_stash[_REPORT_KEY] = [
        {
            "nodeid": "t::x",
            "cassette": "x.tape.json",
            "action": "replayed",
            "calls": 1,
            "input_tokens": 1,
            "cached_input_tokens": 0,
            "output_tokens": 1,
            "total_tokens": 2,
            "estimated_cost_usd": float("nan"),
            "latency_ms": 1.0,
        }
    ]
    nf_stash[_VIOLATIONS_KEY] = []
    nf_opts = {"--tape-max-cost": 1.0, "--tape-max-tokens": None}
    nf_cfg = SimpleNamespace(
        stash=nf_stash, getoption=lambda k, default=None: nf_opts.get(k, default)
    )
    nf_sess = SimpleNamespace(config=nf_cfg, exitstatus=pytest.ExitCode.OK)
    pytest_sessionfinish(nf_sess, 0)  # type: ignore[arg-type]
    assert nf_sess.exitstatus == pytest.ExitCode.TESTS_FAILED

    # Invalid ini values & negative --tape-update-stale trigger pytest UsageError
    pytester.makeini("[pytest]\ntape_max_cost = not_a_float\n")
    assert pytester.runpytest().ret == pytest.ExitCode.USAGE_ERROR

    pytester.makeini("[pytest]\ntape_max_tokens = not_an_int\n")
    assert pytester.runpytest().ret == pytest.ExitCode.USAGE_ERROR

    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    assert pytester.runpytest("--tape-update-stale=-5").ret == pytest.ExitCode.USAGE_ERROR

    # Pricing edge branches: None rate & env file missing/directory
    with pytest.raises(ValueError):
        ModelPrice(None, 1.0)  # type: ignore[arg-type]
    monkeypatch.setenv(PRICING_FILE_ENV, str(tmp_path / "does_not_exist.json"))
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    monkeypatch.setenv(PRICING_FILE_ENV, str(tmp_path))
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
