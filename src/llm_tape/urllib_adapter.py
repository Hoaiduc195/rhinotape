"""Optional thread-safe, contextvars-isolated stdlib adapter for `urllib.request`.

Enabled via `patch_urllib=True` on `TapeSession`, `tape()`, or `@pytest.mark.tape(...)`.
"""

from __future__ import annotations

import contextvars
import email.message
import http
import io
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from llm_tape.models import Interaction, TapeMode
from llm_tape.scrubbing import encode_body_with_encoding

if TYPE_CHECKING:
    from llm_tape.session import TapeSession

_STRIP_REPLAY_HEADERS = frozenset({"content-encoding", "transfer-encoding", "content-length"})
_SSE_CONTENT_TYPES = ("text/event-stream", "application/x-ndjson", "application/ndjson")

_patch_lock = threading.Lock()
_patch_refcount = 0
_orig_opener_open: Any = urllib.request.OpenerDirector.open
_session_getter: Callable[[], TapeSession | None] | None = None
_in_urllib_open: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "llm_tape_in_urllib_open", default=False
)


def _status_reason(status_code: int) -> str:
    try:
        return http.HTTPStatus(status_code).phrase
    except ValueError:
        return ""


def _build_email_headers(headers: dict[str, str]) -> email.message.Message:
    msg = email.message.Message()
    for k, v in headers.items():
        if k.lower() not in _STRIP_REPLAY_HEADERS:
            msg[k] = v
    return msg


class _ReplayUrllibResponse:
    """Minimal http.client.HTTPResponse / addinfourl-compatible replay object for urllib."""

    def __init__(
        self,
        *,
        url: str,
        status: int,
        headers: dict[str, str],
        body_bytes: bytes,
    ) -> None:
        self.url = url
        self.status = status
        self.code = status
        self.reason = _status_reason(status)
        self.headers = _build_email_headers(headers)
        self._fp = io.BytesIO(body_bytes)
        self.closed = False

    def read(self, amt: int | None = None) -> bytes:
        return self._fp.read() if amt is None else self._fp.read(amt)

    def readline(self, limit: int = -1) -> bytes:
        return self._fp.readline(limit)

    def readlines(self, hint: int = -1) -> list[bytes]:
        return self._fp.readlines(hint)

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._fp)

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self.url

    def info(self) -> email.message.Message:
        return self.headers

    def close(self) -> None:
        self.closed = True
        self._fp.close()

    def __enter__(self) -> _ReplayUrllibResponse:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.close()


def _extract_urllib_request(
    fullurl: str | urllib.request.Request, data: Any
) -> tuple[str, str, list[tuple[str, str]], bytes]:
    if isinstance(fullurl, urllib.request.Request):
        req = fullurl
        url = req.full_url
        req_data = data if data is not None else req.data
        method = (
            req.get_method().upper()
            if req_data is None
            else (req.method.upper() if req.method else "POST")
        )
        headers = list(req.header_items())
    else:
        url = str(fullurl)
        req_data = data
        method = "POST" if req_data is not None else "GET"
        headers = []

    if req_data is None:
        body_bytes = b""
    elif isinstance(req_data, bytes):
        body_bytes = req_data
    elif isinstance(req_data, (bytearray, memoryview)):
        body_bytes = bytes(req_data)
    elif isinstance(req_data, str):
        body_bytes = req_data.encode("utf-8")
    else:
        body_bytes = str(req_data).encode("utf-8")
    return method, url, headers, body_bytes


def _build_replay_urllib_result(url: str, interaction: Interaction) -> _ReplayUrllibResponse:
    if interaction.response.stream_chunks is not None:
        raw_bytes = "".join(interaction.response.stream_chunks).encode("utf-8")
    else:
        raw_bytes = encode_body_with_encoding(
            interaction.response.body, interaction.response.body_encoding
        )

    status = interaction.response.status_code
    if status >= 400:
        raise urllib.error.HTTPError(
            url=url,
            code=status,
            msg=_status_reason(status),
            hdrs=_build_email_headers(interaction.response.headers),
            fp=io.BytesIO(raw_bytes),
        )

    return _ReplayUrllibResponse(
        url=url,
        status=status,
        headers=interaction.response.headers,
        body_bytes=raw_bytes,
    )


def _dispatched_opener_open(
    opener: urllib.request.OpenerDirector,
    fullurl: str | urllib.request.Request,
    data: Any = None,
    timeout: Any = None,
) -> Any:
    with _patch_lock:
        orig = _orig_opener_open
        getter = _session_getter

    if _in_urllib_open.get():
        return orig(opener, fullurl, data=data, timeout=timeout)

    session = getter() if getter is not None else None
    if session is None or session.mode is TapeMode.NONE:
        return orig(opener, fullurl, data=data, timeout=timeout)

    method, url, headers, body_bytes = _extract_urllib_request(fullurl, data)
    if session._should_ignore_url(url):
        return orig(opener, fullurl, data=data, timeout=timeout)

    recorded_req = session._build_recorded_request_parts(
        method=method,
        url=url,
        headers=headers,
        body_bytes=body_bytes,
    )

    if session._is_replaying:
        interaction = session._pop_replay_interaction(recorded_req)
        delay_s = (
            (interaction.latency_ms * session.replay_latency_scale) / 1000.0
            if session.replay_latency_scale > 0.0 and interaction.latency_ms > 0.0
            else 0.0
        )
        if delay_s > 0.0:
            time.sleep(delay_s)
        return _build_replay_urllib_result(url, interaction)

    start = time.perf_counter()
    token = _in_urllib_open.set(True)
    try:
        try:
            resp = orig(opener, fullurl, data=data, timeout=timeout)
        except urllib.error.HTTPError as err:
            raw_err = err.read()
            latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
            err_headers = list(err.headers.items()) if err.headers is not None else []
            session._record_interaction_parts(
                recorded_req=recorded_req,
                status_code=int(err.code),
                response_headers=err_headers,
                response_bytes=raw_err,
                latency_ms=latency_ms,
            )
            raise urllib.error.HTTPError(
                url=err.url,
                code=err.code,
                msg=err.msg,
                hdrs=err.hdrs,
                fp=io.BytesIO(raw_err),
            ) from None
    finally:
        _in_urllib_open.reset(token)

    raw_bytes: bytes = resp.read()
    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
    resp_status = int(getattr(resp, "status", getattr(resp, "code", 200)))
    resp_headers_dict = dict(resp.headers.items()) if getattr(resp, "headers", None) else {}
    resp_headers_items = list(resp_headers_dict.items())
    content_type = str(resp.headers.get("Content-Type", "") if resp.headers else "").lower()
    is_sse = any(ct in content_type for ct in _SSE_CONTENT_TYPES)

    if is_sse and resp_status < 400:
        decoded_text = raw_bytes.decode("utf-8", errors="replace")
        parts = decoded_text.split("\n\n")
        sse_chunks = [
            f"{p}\n\n" if i < len(parts) - 1 else p
            for i, p in enumerate(parts)
            if p or i < len(parts) - 1
        ]
        session._record_interaction_parts(
            recorded_req=recorded_req,
            status_code=resp_status,
            response_headers=resp_headers_items,
            response_bytes=None,
            latency_ms=latency_ms,
            stream_chunks=sse_chunks,
        )
    else:
        session._record_interaction_parts(
            recorded_req=recorded_req,
            status_code=resp_status,
            response_headers=resp_headers_items,
            response_bytes=raw_bytes,
            latency_ms=latency_ms,
        )

    final_url = str(getattr(resp, "url", url))
    try:
        resp.close()
    except Exception:
        pass
    return _ReplayUrllibResponse(
        url=final_url,
        status=resp_status,
        headers=resp_headers_dict,
        body_bytes=raw_bytes,
    )


def acquire_urllib_patch(get_active_session: Callable[[], TapeSession | None]) -> bool:
    """Install ref-counted urllib.request.OpenerDirector.open hook."""
    global _patch_refcount, _orig_opener_open, _session_getter
    with _patch_lock:
        if _patch_refcount == 0:
            _orig_opener_open = urllib.request.OpenerDirector.open
            _session_getter = get_active_session
            urllib.request.OpenerDirector.open = _dispatched_opener_open  # type: ignore[method-assign,assignment]
        _patch_refcount += 1
    return True


def release_urllib_patch() -> None:
    """Release ref-counted urllib.request.OpenerDirector.open hook."""
    global _patch_refcount
    with _patch_lock:
        if _patch_refcount > 0:
            _patch_refcount -= 1
            if _patch_refcount == 0:
                urllib.request.OpenerDirector.open = _orig_opener_open  # type: ignore[method-assign]
