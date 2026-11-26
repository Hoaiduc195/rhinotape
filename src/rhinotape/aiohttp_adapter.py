"""Optional thread-safe, contextvars-isolated async adapter for `aiohttp`.

Gracefully no-ops if `aiohttp` is not installed (zero hard runtime dependency).
"""

from __future__ import annotations

import asyncio
import codecs
import contextvars
import http
import importlib
import json
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from rhinotape.models import Interaction, TapeMode
from rhinotape.scrubbing import encode_body_with_encoding

if TYPE_CHECKING:
    from rhinotape.session import TapeSession

_STRIP_REPLAY_HEADERS = frozenset({"content-encoding", "transfer-encoding", "content-length"})
_SSE_CONTENT_TYPES = ("text/event-stream", "application/x-ndjson", "application/ndjson")

_patch_lock = threading.Lock()
_patch_refcount = 0
_orig_client_request: Any = None
_session_getter: Callable[[], TapeSession | None] | None = None
_in_aiohttp_request: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "rhinotape_in_aiohttp_request", default=False
)


def _status_reason(status_code: int) -> str:
    try:
        return http.HTTPStatus(status_code).phrase
    except ValueError:
        return ""


def _resolve_url(client_session: Any, str_or_url: Any, params: Any) -> str:
    yarl_mod: Any = importlib.import_module("yarl")
    url = yarl_mod.URL(str_or_url)
    base_url = getattr(client_session, "_base_url", None)
    if base_url is not None and not url.is_absolute():
        url = base_url.join(url)
    if params:
        url = url.update_query(params)
    return str(url)


def _merge_headers(client_session: Any, req_headers: Any) -> list[tuple[str, str]]:
    merged: dict[str, tuple[str, str]] = {}
    session_headers = getattr(client_session, "headers", None)
    if session_headers:
        for k, v in session_headers.items():
            merged[str(k).lower()] = (str(k), str(v))
    if req_headers:
        items = req_headers.items() if isinstance(req_headers, Mapping) else req_headers
        for k, v in items:
            merged[str(k).lower()] = (str(k), str(v))
    return list(merged.values())


def _extract_request_body(kwargs: dict[str, Any]) -> bytes:
    if "json" in kwargs and kwargs["json"] is not None:
        return json.dumps(kwargs["json"], ensure_ascii=False).encode("utf-8")
    data = kwargs.get("data")
    if data is None:
        return b""
    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, Mapping):
        return urlencode(data, doseq=True).encode("utf-8")
    return str(data).encode("utf-8")


class _ReplayAiohttpStreamReader:
    """Async stream reader compatible with aiohttp's response.content interface."""

    def __init__(self, chunks: tuple[bytes, ...], *, chunk_delay_s: float = 0.0) -> None:
        self._chunks = chunks
        self._chunk_delay_s = max(0.0, chunk_delay_s)
        self._idx = 0
        self._buf = bytearray()
        self._eof = False

    def at_eof(self) -> bool:
        return self._eof and not self._buf

    async def _fill_next(self) -> bool:
        if self._idx >= len(self._chunks):
            self._eof = True
            return False
        if self._chunk_delay_s > 0.0:
            await asyncio.sleep(self._chunk_delay_s)
        self._buf.extend(self._chunks[self._idx])
        self._idx += 1
        return True

    async def read(self, n: int = -1) -> bytes:
        if n == 0:
            return b""
        if n < 0:
            while await self._fill_next():
                pass
            out = bytes(self._buf)
            self._buf.clear()
            return out
        while len(self._buf) < n:
            if not await self._fill_next():
                break
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    async def readline(self) -> bytes:
        while True:
            nl = self._buf.find(b"\n")
            if nl != -1:
                out = bytes(self._buf[: nl + 1])
                del self._buf[: nl + 1]
                return out
            if not await self._fill_next():
                out = bytes(self._buf)
                self._buf.clear()
                return out

    async def iter_any(self) -> AsyncIterator[bytes]:
        if self._buf:
            out = bytes(self._buf)
            self._buf.clear()
            yield out
        while self._idx < len(self._chunks):
            if self._chunk_delay_s > 0.0:
                await asyncio.sleep(self._chunk_delay_s)
            chunk = self._chunks[self._idx]
            self._idx += 1
            if chunk:
                yield chunk
        self._eof = True

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        while True:
            chunk = await self.read(n)
            if not chunk:
                break
            yield chunk

    async def readany(self) -> bytes:
        if self._buf:
            out = bytes(self._buf)
            self._buf.clear()
            return out
        if not await self._fill_next():
            return b""
        out = bytes(self._buf)
        self._buf.clear()
        return out

    async def readchunk(self) -> tuple[bytes, bool]:
        chunk = await self.readany()
        return (chunk, bool(chunk))

    async def iter_chunks(self) -> AsyncIterator[tuple[bytes, bool]]:
        while True:
            chunk, end_of_chunk = await self.readchunk()
            if not chunk:
                break
            yield (chunk, end_of_chunk)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iter_lines()

    async def _iter_lines(self) -> AsyncIterator[bytes]:
        while True:
            line = await self.readline()
            if not line:
                break
            yield line


class _ReplayAiohttpResponse:
    """Lightweight aiohttp.ClientResponse-compatible object for offline replay."""

    def __init__(
        self,
        *,
        method: str,
        url: str,
        interaction: Interaction,
        chunk_delay_s: float = 0.0,
    ) -> None:
        aiohttp_mod: Any = importlib.import_module("aiohttp")
        multidict_mod: Any = importlib.import_module("multidict")
        yarl_mod: Any = importlib.import_module("yarl")

        self.method = method
        self.url = yarl_mod.URL(url)
        self.status = interaction.response.status_code
        self.reason = _status_reason(self.status)

        replay_headers = {
            k: v
            for k, v in interaction.response.headers.items()
            if k.lower() not in _STRIP_REPLAY_HEADERS
        }
        self.headers = multidict_mod.CIMultiDictProxy(multidict_mod.CIMultiDict(replay_headers))
        self.request_info = aiohttp_mod.RequestInfo(
            url=self.url,
            method=self.method,
            headers=self.headers,
            real_url=self.url,
        )
        raw_ct = self.headers.get("Content-Type", "application/octet-stream")
        self.content_type = raw_ct.split(";", 1)[0].strip().lower()
        self.charset = "utf-8"
        self.closed = False

        if interaction.response.stream_chunks is not None:
            chunks = tuple(c.encode("utf-8") for c in interaction.response.stream_chunks)
        else:
            raw_bytes = encode_body_with_encoding(
                interaction.response.body, interaction.response.body_encoding
            )
            chunks = (raw_bytes,) if raw_bytes else ()

        self.content = _ReplayAiohttpStreamReader(chunks, chunk_delay_s=chunk_delay_s)
        self._body: bytes | None = None

    @property
    def ok(self) -> bool:
        return self.status < 400

    async def read(self) -> bytes:
        if self._body is None:
            self._body = await self.content.read()
        return self._body

    async def text(self, encoding: str | None = None, errors: str = "strict") -> str:
        data = await self.read()
        return data.decode(encoding or self.charset or "utf-8", errors=errors)

    async def json(
        self,
        *,
        encoding: str | None = None,
        loads: Callable[[str], Any] = json.loads,
        content_type: str | None = "application/json",
    ) -> Any:
        if content_type is not None:
            ct_lower = content_type.lower()
            if self.content_type != ct_lower and not (
                ct_lower == "application/json" and self.content_type.endswith("+json")
            ):
                aiohttp_mod: Any = importlib.import_module("aiohttp")
                raise aiohttp_mod.ContentTypeError(
                    self.request_info,
                    history=(),
                    message=f"Attempt to decode JSON with unexpected mimetype: {self.content_type}",
                    headers=self.headers,
                )
        return loads(await self.text(encoding=encoding))

    def raise_for_status(self) -> None:
        if self.status >= 400:
            aiohttp_mod: Any = importlib.import_module("aiohttp")
            raise aiohttp_mod.ClientResponseError(
                request_info=self.request_info,
                history=(),
                status=self.status,
                message=self.reason,
                headers=self.headers,
            )

    def release(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> _ReplayAiohttpResponse:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.release()


class _RecordingAiohttpStreamReader:
    """Wraps a live aiohttp StreamReader to record UTF-8 SSE chunks and commit on EOF."""

    def __init__(
        self,
        upstream: Any,
        on_complete: Callable[[list[str]], None],
        *,
        on_abort: Callable[[], None] | None = None,
    ) -> None:
        self._upstream = upstream
        self._on_complete = on_complete
        self._on_abort = on_abort
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._chunks: list[str] = []
        self._reached_eof = False
        self._settled = False

    def _record_bytes(self, data: bytes) -> None:
        if data:
            text = self._utf8_decoder.decode(data, final=False)
            if text:
                self._chunks.append(text)

    def _finish(self) -> None:
        if not self._reached_eof:
            tail = self._utf8_decoder.decode(b"", final=True)
            if tail:
                self._chunks.append(tail)
            self._reached_eof = True
        if not self._settled:
            self._settled = True
            self._on_complete(self._chunks)

    def _abort(self) -> None:
        if not self._settled:
            self._settled = True
            if self._on_abort is not None:
                self._on_abort()

    async def read(self, n: int = -1) -> bytes:
        if n == 0:
            return b""
        try:
            data: bytes = await self._upstream.read(n)
        except BaseException:
            self._abort()
            raise
        if data:
            self._record_bytes(data)
            if n < 0:
                self._finish()
        else:
            self._finish()
        return data

    async def readline(self) -> bytes:
        try:
            line: bytes = await self._upstream.readline()
        except BaseException:
            self._abort()
            raise
        if line:
            self._record_bytes(line)
        else:
            self._finish()
        return line

    async def iter_any(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._upstream.iter_any():
                self._record_bytes(chunk)
                yield chunk
            self._finish()
        except BaseException:
            self._abort()
            raise

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._upstream.iter_chunked(n):
                self._record_bytes(chunk)
                yield chunk
            self._finish()
        except BaseException:
            self._abort()
            raise

    async def readany(self) -> bytes:
        try:
            data: bytes = await self._upstream.readany()
        except BaseException:
            self._abort()
            raise
        if data:
            self._record_bytes(data)
        else:
            self._finish()
        return data

    async def readchunk(self) -> tuple[bytes, bool]:
        try:
            chunk, end_of_chunk = await self._upstream.readchunk()
        except BaseException:
            self._abort()
            raise
        if chunk:
            self._record_bytes(chunk)
        else:
            self._finish()
        return (chunk, end_of_chunk)

    async def iter_chunks(self) -> AsyncIterator[tuple[bytes, bool]]:
        try:
            async for chunk, end_of_chunk in self._upstream.iter_chunks():
                self._record_bytes(chunk)
                yield (chunk, end_of_chunk)
            self._finish()
        except BaseException:
            self._abort()
            raise

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iter_lines()

    async def _iter_lines(self) -> AsyncIterator[bytes]:
        try:
            async for line in self._upstream:
                self._record_bytes(line)
                yield line
            self._finish()
        except BaseException:
            self._abort()
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upstream, name)


async def _dispatched_aiohttp_request(
    client_session: Any,
    method: str,
    str_or_url: Any,
    **kwargs: Any,
) -> Any:
    aiohttp_mod: Any = importlib.import_module("aiohttp")

    with _patch_lock:
        orig = _orig_client_request
        getter = _session_getter

    if orig is None:
        orig = aiohttp_mod.ClientSession._request
        if orig is _dispatched_aiohttp_request:
            raise RuntimeError("aiohttp.ClientSession._request original reference is unavailable")

    if _in_aiohttp_request.get():
        return await orig(client_session, method, str_or_url, **kwargs)

    session = getter() if getter is not None else None
    if session is None or session.mode is TapeMode.NONE:
        return await orig(client_session, method, str_or_url, **kwargs)

    url = _resolve_url(client_session, str_or_url, kwargs.get("params"))
    if session._should_ignore_url(url):
        return await orig(client_session, method, str_or_url, **kwargs)

    headers_items = _merge_headers(client_session, kwargs.get("headers"))
    body_bytes = _extract_request_body(kwargs)

    recorded_req = session._build_recorded_request_parts(
        method=method.upper(),
        url=url,
        headers=headers_items,
        body_bytes=body_bytes,
    )

    matched, slot_owned = session._match_or_acquire_slot(recorded_req)
    if matched is not None:
        interaction = matched
        delay_s = (
            (interaction.latency_ms * session.replay_latency_scale) / 1000.0
            if session.replay_latency_scale > 0.0 and interaction.latency_ms > 0.0
            else 0.0
        )
        chunk_delay_s = 0.0
        if delay_s > 0.0:
            if interaction.response.stream_chunks:
                chunk_delay_s = delay_s / len(interaction.response.stream_chunks)
            else:
                await asyncio.sleep(delay_s)
        replay_resp = _ReplayAiohttpResponse(
            method=method.upper(),
            url=url,
            interaction=interaction,
            chunk_delay_s=chunk_delay_s,
        )
        rfs = kwargs.get("raise_for_status")
        if rfs is None:
            rfs = getattr(client_session, "_raise_for_status", False)
        if callable(rfs):
            await rfs(replay_resp)
        elif rfs:
            replay_resp.raise_for_status()
        return replay_resp

    slot_for_call = slot_owned
    try:
        start = time.perf_counter()
        token = _in_aiohttp_request.set(True)
        try:
            response = await orig(client_session, method, str_or_url, **kwargs)
        finally:
            _in_aiohttp_request.reset(token)

        content_type = str(response.headers.get("Content-Type", "")).lower()
        is_sse = any(ct in content_type for ct in _SSE_CONTENT_TYPES)

        if is_sse and response.status < 400 and getattr(response, "content", None) is not None:
            resp_headers_items = list(response.headers.items())
            resp_status = int(response.status)

            def on_complete(chunks: list[str]) -> None:
                latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
                session._record_interaction_parts(
                    recorded_req=recorded_req,
                    status_code=resp_status,
                    response_headers=resp_headers_items,
                    response_bytes=None,
                    latency_ms=latency_ms,
                    stream_chunks=chunks,
                    release_slot=slot_for_call,
                )

            reader = _RecordingAiohttpStreamReader(
                response.content,
                on_complete,
                on_abort=session._release_recording_slot if slot_for_call else None,
            )
            response.content = reader

            orig_release = getattr(response, "release", None)
            if callable(orig_release):

                def _wrapped_release() -> Any:
                    try:
                        return orig_release()
                    finally:
                        if reader._reached_eof:
                            reader._finish()
                        else:
                            reader._abort()

                response.release = _wrapped_release

            orig_close = getattr(response, "close", None)
            if callable(orig_close):

                def _wrapped_close() -> None:
                    try:
                        orig_close()
                    finally:
                        if reader._reached_eof:
                            reader._finish()
                        else:
                            reader._abort()

                response.close = _wrapped_close

            slot_owned = False
            return response

        raw_bytes: bytes = await response.read()
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        slot_owned = False
        session._record_interaction_parts(
            recorded_req=recorded_req,
            status_code=int(response.status),
            response_headers=list(response.headers.items()),
            response_bytes=raw_bytes,
            latency_ms=latency_ms,
            release_slot=slot_for_call,
        )
        return response
    finally:
        if slot_owned:
            session._release_recording_slot()


def acquire_aiohttp_patch(get_active_session: Callable[[], TapeSession | None]) -> bool:
    """Install ref-counted aiohttp.ClientSession._request hook if `aiohttp` is installed."""
    global _patch_refcount, _orig_client_request, _session_getter
    try:
        aiohttp_mod: Any = importlib.import_module("aiohttp")
    except ImportError:
        return False

    with _patch_lock:
        if _patch_refcount == 0:
            _orig_client_request = aiohttp_mod.ClientSession._request
            _session_getter = get_active_session
            aiohttp_mod.ClientSession._request = _dispatched_aiohttp_request
        _patch_refcount += 1
    return True


def release_aiohttp_patch() -> None:
    """Release ref-counted aiohttp.ClientSession._request hook."""
    global _patch_refcount
    with _patch_lock:
        if _patch_refcount > 0:
            _patch_refcount -= 1
            if _patch_refcount == 0 and _orig_client_request is not None:
                try:
                    aiohttp_mod: Any = importlib.import_module("aiohttp")
                    aiohttp_mod.ClientSession._request = _orig_client_request
                except ImportError:
                    pass
