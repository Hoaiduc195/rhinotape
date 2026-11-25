"""Optional thread-safe, contextvars-isolated adapter for the `requests` library.

Gracefully no-ops if `requests` is not installed (zero hard runtime dependency).
"""

from __future__ import annotations

import codecs
import contextvars
import http
import importlib
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

from llm_tape.models import Interaction, TapeMode
from llm_tape.scrubbing import encode_body_with_encoding

if TYPE_CHECKING:
    from llm_tape.session import TapeSession

_STRIP_REPLAY_HEADERS = frozenset({"content-encoding", "transfer-encoding", "content-length"})
_SSE_CONTENT_TYPES = ("text/event-stream", "application/x-ndjson", "application/ndjson")

_patch_lock = threading.Lock()
_patch_refcount = 0
_orig_session_send: Any = None
_session_getter: Callable[[], TapeSession | None] | None = None
_in_requests_send: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "llm_tape_in_requests_send", default=False
)


def _extract_body_bytes(prepared_request: Any) -> bytes:
    body = getattr(prepared_request, "body", None)
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    if isinstance(body, (bytearray, memoryview)):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8")
    if hasattr(body, "read"):
        raw = body.read()
        data = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        prepared_request.body = data
        return data
    if isinstance(body, Iterable):
        chunks = [c.encode("utf-8") if isinstance(c, str) else bytes(c) for c in body]
        data = b"".join(chunks)
        prepared_request.body = data
        return data
    return str(body).encode("utf-8")


class _ReplayRequestsRaw:
    """Minimal urllib3-compatible raw stream for replaying recorded bytes/SSE chunks."""

    def __init__(self, chunks: tuple[bytes, ...], *, chunk_delay_s: float = 0.0) -> None:
        self._chunks = chunks
        self._chunk_delay_s = max(0.0, chunk_delay_s)
        self._idx = 0
        self._buf = b""
        self.closed = False

    def stream(self, amt: int | None = 65536, decode_content: bool = True) -> Iterator[bytes]:
        del decode_content
        for chunk in self._chunks:
            if self.closed:
                break
            if self._chunk_delay_s > 0.0:
                time.sleep(self._chunk_delay_s)
            if amt is None or amt <= 0 or len(chunk) <= amt:
                if chunk:
                    yield chunk
            else:
                for offset in range(0, len(chunk), amt):
                    yield chunk[offset : offset + amt]
        self.closed = True

    def read(self, amt: int | None = None, decode_content: bool = True) -> bytes:
        del decode_content
        if self.closed and not self._buf:
            return b""
        if amt is None or amt < 0:
            parts = [self._buf] if self._buf else []
            self._buf = b""
            while self._idx < len(self._chunks):
                if self._chunk_delay_s > 0.0:
                    time.sleep(self._chunk_delay_s)
                parts.append(self._chunks[self._idx])
                self._idx += 1
            self.closed = True
            return b"".join(parts)
        while len(self._buf) < amt and self._idx < len(self._chunks):
            if self._chunk_delay_s > 0.0:
                time.sleep(self._chunk_delay_s)
            self._buf += self._chunks[self._idx]
            self._idx += 1
        out = self._buf[:amt]
        self._buf = self._buf[amt:]
        if not self._buf and self._idx >= len(self._chunks):
            self.closed = True
        return out

    def release_conn(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


class _RecordingRequestsRaw:
    """Wraps a live urllib3 raw stream, recording UTF-8 SSE chunks or raw bytes on EOF."""

    def __init__(
        self,
        upstream_raw: Any,
        *,
        is_sse: bool,
        on_complete_sse: Callable[[list[str]], None] | None = None,
        on_complete_bytes: Callable[[bytes], None] | None = None,
    ) -> None:
        self._upstream = upstream_raw
        self._is_sse = is_sse
        self._on_complete_sse = on_complete_sse
        self._on_complete_bytes = on_complete_bytes
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._text_chunks: list[str] = []
        self._byte_chunks: list[bytes] = []
        self._reached_eof = False
        self._committed = False

    def _record_bytes(self, data: bytes) -> None:
        if not data:
            return
        if self._is_sse:
            text = self._utf8_decoder.decode(data, final=False)
            if text:
                self._text_chunks.append(text)
        else:
            self._byte_chunks.append(data)

    def _finish(self) -> None:
        if not self._reached_eof:
            if self._is_sse:
                tail = self._utf8_decoder.decode(b"", final=True)
                if tail:
                    self._text_chunks.append(tail)
            self._reached_eof = True
        self._commit()

    def _commit(self) -> None:
        if not self._committed:
            self._committed = True
            if self._is_sse and self._on_complete_sse is not None:
                self._on_complete_sse(self._text_chunks)
            elif not self._is_sse and self._on_complete_bytes is not None:
                self._on_complete_bytes(b"".join(self._byte_chunks))

    def stream(self, amt: int | None = 65536, decode_content: bool = True) -> Iterator[bytes]:
        for chunk in self._upstream.stream(amt=amt, decode_content=decode_content):
            self._record_bytes(chunk)
            yield chunk
        self._finish()

    def read(self, amt: int | None = None, decode_content: bool = True) -> bytes:
        if amt == 0:
            return b""
        data: bytes = self._upstream.read(amt, decode_content=decode_content)
        if data:
            self._record_bytes(data)
            if amt is None or amt < 0:
                self._finish()
        else:
            self._finish()
        return data

    def release_conn(self) -> None:
        try:
            if hasattr(self._upstream, "release_conn"):
                self._upstream.release_conn()
        finally:
            if self._reached_eof:
                self._commit()

    def close(self) -> None:
        try:
            if hasattr(self._upstream, "close"):
                self._upstream.close()
        finally:
            if self._reached_eof:
                self._commit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upstream, name)


def _status_reason(status_code: int) -> str:
    try:
        return http.HTTPStatus(status_code).phrase
    except ValueError:
        return ""


def _build_replay_requests_response(
    requests_mod: Any,
    prepared_request: Any,
    interaction: Interaction,
    *,
    stream: bool,
    chunk_delay_s: float,
) -> Any:
    resp = requests_mod.Response()
    resp.status_code = interaction.response.status_code
    resp.reason = _status_reason(interaction.response.status_code)
    resp.url = str(getattr(prepared_request, "url", interaction.request.url))
    resp.request = prepared_request
    resp.encoding = "utf-8"

    replay_headers = {
        k: v
        for k, v in interaction.response.headers.items()
        if k.lower() not in _STRIP_REPLAY_HEADERS
    }
    resp.headers = requests_mod.structures.CaseInsensitiveDict(replay_headers)

    if interaction.response.stream_chunks is not None:
        encoded_chunks = tuple(c.encode("utf-8") for c in interaction.response.stream_chunks)
        if stream:
            resp.raw = _ReplayRequestsRaw(encoded_chunks, chunk_delay_s=chunk_delay_s)
            resp._content = False
            resp._content_consumed = False
        else:
            raw_bytes = b"".join(encoded_chunks)
            resp.raw = _ReplayRequestsRaw((raw_bytes,))
            resp._content = raw_bytes
            resp._content_consumed = True
        return resp

    raw_bytes = encode_body_with_encoding(
        interaction.response.body, interaction.response.body_encoding
    )
    if stream:
        resp.raw = _ReplayRequestsRaw((raw_bytes,))
        resp._content = False
        resp._content_consumed = False
    else:
        resp.raw = _ReplayRequestsRaw((raw_bytes,))
        resp._content = raw_bytes
        resp._content_consumed = True
    return resp


def _dispatched_requests_send(req_session: Any, prepared_request: Any, **kwargs: Any) -> Any:
    requests_mod: Any = importlib.import_module("requests")

    with _patch_lock:
        orig = _orig_session_send
        getter = _session_getter

    if orig is None:
        orig = requests_mod.Session.send
        if orig is _dispatched_requests_send:
            raise RuntimeError("requests.Session.send original reference is unavailable")

    if _in_requests_send.get():
        return orig(req_session, prepared_request, **kwargs)

    session = getter() if getter is not None else None
    if session is None or session.mode is TapeMode.NONE:
        return orig(req_session, prepared_request, **kwargs)

    url = str(getattr(prepared_request, "url", ""))
    if session._should_ignore_url(url):
        return orig(req_session, prepared_request, **kwargs)

    stream = bool(kwargs.get("stream", False))
    method = str(getattr(prepared_request, "method", "GET") or "GET").upper()
    headers_items = list(getattr(prepared_request, "headers", {}).items())
    body_bytes = _extract_body_bytes(prepared_request)

    recorded_req = session._build_recorded_request_parts(
        method=method,
        url=url,
        headers=headers_items,
        body_bytes=body_bytes,
    )

    if session._is_replaying:
        interaction = session._pop_replay_interaction(recorded_req)
        delay_s = (
            (interaction.latency_ms * session.replay_latency_scale) / 1000.0
            if session.replay_latency_scale > 0.0 and interaction.latency_ms > 0.0
            else 0.0
        )
        chunk_delay_s = 0.0
        if delay_s > 0.0:
            if stream and interaction.response.stream_chunks:
                chunk_delay_s = delay_s / len(interaction.response.stream_chunks)
            else:
                time.sleep(delay_s)
        return _build_replay_requests_response(
            requests_mod,
            prepared_request,
            interaction,
            stream=stream,
            chunk_delay_s=chunk_delay_s,
        )

    start = time.perf_counter()
    reentrancy_token = _in_requests_send.set(True)
    try:
        response = orig(req_session, prepared_request, **kwargs)
    finally:
        _in_requests_send.reset(reentrancy_token)

    content_type = str(response.headers.get("content-type", "")).lower()
    is_sse = any(ct in content_type for ct in _SSE_CONTENT_TYPES)

    if (
        stream
        and response.status_code < 400
        and getattr(response, "raw", None) is not None
        and not getattr(response, "_content_consumed", False)
        and (is_sse or hasattr(response.raw, "stream"))
    ):
        resp_headers_items = list(response.headers.items())
        resp_status = int(response.status_code)

        def on_complete_sse(chunks: list[str]) -> None:
            latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
            session._record_interaction_parts(
                recorded_req=recorded_req,
                status_code=resp_status,
                response_headers=resp_headers_items,
                response_bytes=None,
                latency_ms=latency_ms,
                stream_chunks=chunks,
            )

        def on_complete_bytes(data: bytes) -> None:
            latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
            session._record_interaction_parts(
                recorded_req=recorded_req,
                status_code=resp_status,
                response_headers=resp_headers_items,
                response_bytes=data,
                latency_ms=latency_ms,
            )

        response.raw = _RecordingRequestsRaw(
            response.raw,
            is_sse=is_sse,
            on_complete_sse=on_complete_sse,
            on_complete_bytes=on_complete_bytes,
        )
        return response

    raw_content: bytes = response.content
    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
    resp_headers_items = list(response.headers.items())
    if is_sse and response.status_code < 400:
        decoded_text = raw_content.decode("utf-8", errors="replace")
        parts = decoded_text.split("\n\n")
        sse_chunks = [
            f"{p}\n\n" if i < len(parts) - 1 else p
            for i, p in enumerate(parts)
            if p or i < len(parts) - 1
        ]
        session._record_interaction_parts(
            recorded_req=recorded_req,
            status_code=int(response.status_code),
            response_headers=resp_headers_items,
            response_bytes=None,
            latency_ms=latency_ms,
            stream_chunks=sse_chunks,
        )
    else:
        session._record_interaction_parts(
            recorded_req=recorded_req,
            status_code=int(response.status_code),
            response_headers=resp_headers_items,
            response_bytes=raw_content,
            latency_ms=latency_ms,
        )
    return response


def acquire_requests_patch(get_active_session: Callable[[], TapeSession | None]) -> bool:
    """Install ref-counted requests.Session.send hook if `requests` is installed."""
    global _patch_refcount, _orig_session_send, _session_getter
    try:
        requests_mod: Any = importlib.import_module("requests")
    except ImportError:
        return False

    with _patch_lock:
        if _patch_refcount == 0:
            _orig_session_send = requests_mod.Session.send
            _session_getter = get_active_session
            requests_mod.Session.send = _dispatched_requests_send
        _patch_refcount += 1
    return True


def release_requests_patch() -> None:
    """Release ref-counted requests.Session.send hook."""
    global _patch_refcount
    with _patch_lock:
        if _patch_refcount > 0:
            _patch_refcount -= 1
            if _patch_refcount == 0 and _orig_session_send is not None:
                try:
                    requests_mod: Any = importlib.import_module("requests")
                    requests_mod.Session.send = _orig_session_send
                except ImportError:
                    pass
