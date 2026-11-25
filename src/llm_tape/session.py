"""TapeSession: contextvars-safe recording, replaying, SSE streaming, and LLM assertions."""

from __future__ import annotations

import codecs
import contextvars
import fnmatch
import functools
import inspect
import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import httpx

from llm_tape.cassette import Cassette
from llm_tape.models import (
    Interaction,
    RecordedRequest,
    RecordedResponse,
    TapeAssertionError,
    TapeError,
    TapeMode,
    ToolCall,
    UnmatchedRequestError,
    Usage,
)
from llm_tape.parsers import parse_llm_interaction
from llm_tape.pricing import ModelPrice
from llm_tape.scrubbing import (
    decode_body_with_encoding,
    encode_body_with_encoding,
    fingerprint_request,
    format_request_diff,
    scrub_headers,
    scrub_url,
)

P = ParamSpec("P")
R = TypeVar("R")

_STRIP_REPLAY_HEADERS = frozenset({"content-encoding", "transfer-encoding", "content-length"})
_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_HANDLED_EXT_KEY = "_llm_tape_handled"

# --- Global contextvars + ref-counted httpx dispatcher ---------------------

_active_session: contextvars.ContextVar[TapeSession | None] = contextvars.ContextVar(
    "llm_tape_active_session", default=None
)
_patch_lock = threading.Lock()
_patch_refcount = 0
_orig_sync_send: Any = None
_orig_async_send: Any = None


def _dispatched_sync_send(
    client: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any
) -> httpx.Response:
    session = _active_session.get()
    if session is None:
        return _orig_sync_send(client, request, *args, **kwargs)  # type: ignore[no-any-return]
    stream = bool(kwargs.get("stream", False))
    return session.handle_sync(
        request,
        lambda req: _orig_sync_send(client, req, *args, **kwargs),
        stream=stream,
    )


async def _dispatched_async_send(
    client: httpx.AsyncClient, request: httpx.Request, *args: Any, **kwargs: Any
) -> httpx.Response:
    session = _active_session.get()
    if session is None:
        return await _orig_async_send(client, request, *args, **kwargs)  # type: ignore[no-any-return]
    stream = bool(kwargs.get("stream", False))
    return await session.handle_async(
        request,
        lambda req: _orig_async_send(client, req, *args, **kwargs),
        stream=stream,
    )


def _acquire_httpx_patch() -> None:
    global _patch_refcount, _orig_sync_send, _orig_async_send
    with _patch_lock:
        if _patch_refcount == 0:
            _orig_sync_send = httpx.Client.send
            _orig_async_send = httpx.AsyncClient.send
            httpx.Client.send = _dispatched_sync_send  # type: ignore[method-assign,assignment]
            httpx.AsyncClient.send = _dispatched_async_send  # type: ignore[method-assign,assignment]
        _patch_refcount += 1


def _release_httpx_patch() -> None:
    global _patch_refcount, _orig_sync_send, _orig_async_send
    with _patch_lock:
        if _patch_refcount > 0:
            _patch_refcount -= 1
            if _patch_refcount == 0:
                if _orig_sync_send is not None:
                    httpx.Client.send = _orig_sync_send  # type: ignore[method-assign]
                    _orig_sync_send = None
                if _orig_async_send is not None:
                    httpx.AsyncClient.send = _orig_async_send  # type: ignore[method-assign]
                    _orig_async_send = None


# --- Streaming ByteStream wrappers -----------------------------------------


class _ReplayByteStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Yields recorded SSE/stream chunks for both sync and async httpx replay."""

    def __init__(self, chunks: tuple[str, ...]) -> None:
        self._chunks = [c.encode("utf-8") for c in chunks]

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _ReplayRawBytesStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Yields a single raw bytes buffer (used when a streamed response had a non-SSE error body)."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __iter__(self) -> Iterator[bytes]:
        if self._data:
            yield self._data

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._data:
            yield self._data


class _RecordingSyncStream(httpx.SyncByteStream):
    """Wraps a live sync stream, decoding incrementally and committing only on EOF."""

    def __init__(
        self,
        upstream_stream: httpx.SyncByteStream,
        on_complete: Callable[[list[str]], None],
    ) -> None:
        self._upstream = upstream_stream
        self._on_complete = on_complete
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._chunks: list[str] = []
        self._reached_eof = False
        self._committed = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._upstream:
            text = self._decoder.decode(chunk, final=False)
            if text:
                self._chunks.append(text)
            yield chunk
        tail = self._decoder.decode(b"", final=True)
        if tail:
            self._chunks.append(tail)
        self._reached_eof = True
        self._commit()

    def close(self) -> None:
        try:
            self._upstream.close()
        finally:
            if self._reached_eof:
                self._commit()

    def _commit(self) -> None:
        if not self._committed:
            self._committed = True
            self._on_complete(self._chunks)


class _RecordingAsyncStream(httpx.AsyncByteStream):
    """Wraps a live async stream, decoding incrementally and committing only on EOF."""

    def __init__(
        self,
        upstream_stream: httpx.AsyncByteStream,
        on_complete: Callable[[list[str]], None],
    ) -> None:
        self._upstream = upstream_stream
        self._on_complete = on_complete
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._chunks: list[str] = []
        self._reached_eof = False
        self._committed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._upstream:
            text = self._decoder.decode(chunk, final=False)
            if text:
                self._chunks.append(text)
            yield chunk
        tail = self._decoder.decode(b"", final=True)
        if tail:
            self._chunks.append(tail)
        self._reached_eof = True
        self._commit()

    async def aclose(self) -> None:
        try:
            await self._upstream.aclose()
        finally:
            if self._reached_eof:
                self._commit()

    def _commit(self) -> None:
        if not self._committed:
            self._committed = True
            self._on_complete(self._chunks)


# --- TapeSession -----------------------------------------------------------


class TapeSession:
    """Active recording/replaying session with LLM-aware inspection and assertions."""

    def __init__(
        self,
        cassette_path: str | Path,
        *,
        mode: TapeMode | str = TapeMode.ONCE,
        custom_pricing: dict[str, ModelPrice] | None = None,
        extra_sensitive_headers: Iterable[str] = (),
        extra_sensitive_query_params: Iterable[str] = (),
        ignore_hosts: Iterable[str] = (),
        ignore_localhost: bool = False,
        ignore_request: Callable[[httpx.Request], bool] | None = None,
        ignore_fields: Iterable[str] = (),
        ignore_query_params: Iterable[str] = (),
        allow_playback_repeats: bool = False,
        save_on_error: bool = False,
        patch_httpx: bool = True,
    ) -> None:
        self.cassette_path = Path(cassette_path)
        self.mode = TapeMode.coerce(mode)
        self.custom_pricing = custom_pricing
        self.extra_sensitive_headers = tuple(extra_sensitive_headers)
        self.extra_sensitive_query_params = tuple(extra_sensitive_query_params)
        self.ignore_hosts = tuple(h.lower() for h in ignore_hosts)
        self.ignore_localhost = ignore_localhost
        self.ignore_request = ignore_request
        self.ignore_fields = tuple(ignore_fields)
        self.ignore_query_params = tuple(ignore_query_params)
        self.allow_playback_repeats = allow_playback_repeats
        self.save_on_error = save_on_error
        self.patch_httpx = patch_httpx

        self._lock = threading.Lock()
        self._interactions: list[Interaction] = []
        self._recorded: list[Interaction] = []
        self._replay_queues: dict[str, deque[Interaction]] = defaultdict(deque)
        self._known_fingerprints: dict[str, int] = {}
        self._all_loaded_requests: list[RecordedRequest] = []
        self._is_replaying = False
        self._is_recording = False
        self._active = False
        self._ctx_token: contextvars.Token[TapeSession | None] | None = None
        self._patch_acquired = False

    # --- Lifecycle ---------------------------------------------------------

    def _prepare(self) -> None:
        if self._active:
            raise TapeError("TapeSession is already active and cannot be re-entered concurrently.")
        self._active = True
        with self._lock:
            self._interactions.clear()
            self._recorded.clear()
            self._replay_queues.clear()
            self._known_fingerprints.clear()
            self._all_loaded_requests.clear()

        if self.mode is TapeMode.NONE:
            self._is_replaying = False
            self._is_recording = False
            return

        if self.mode is TapeMode.REPLAY:
            cassette = Cassette.load(self.cassette_path)
            self._load_replay_queues(cassette)
            self._is_replaying = True
            self._is_recording = False
        elif self.mode is TapeMode.RECORD:
            self._is_replaying = False
            self._is_recording = True
        elif self.mode is TapeMode.ONCE:
            if self.cassette_path.exists():
                cassette = Cassette.load(self.cassette_path)
                self._load_replay_queues(cassette)
                self._is_replaying = True
                self._is_recording = False
            else:
                self._is_replaying = False
                self._is_recording = True

    def _load_replay_queues(self, cassette: Cassette) -> None:
        with self._lock:
            for interaction in cassette.interactions:
                fp = fingerprint_request(
                    interaction.request.method,
                    interaction.request.url,
                    interaction.request.body,
                    ignore_fields=self.ignore_fields,
                    ignore_query_params=self.ignore_query_params,
                )
                self._replay_queues[fp].append(interaction)
                self._known_fingerprints[fp] = self._known_fingerprints.get(fp, 0) + 1
                self._all_loaded_requests.append(interaction.request)

    def _install_httpx_patch(self) -> None:
        if not self.patch_httpx or self.mode is TapeMode.NONE:
            return
        self._ctx_token = _active_session.set(self)
        _acquire_httpx_patch()
        self._patch_acquired = True

    def _uninstall_httpx_patch(self) -> None:
        try:
            if self._patch_acquired:
                _release_httpx_patch()
                self._patch_acquired = False
        finally:
            if self._ctx_token is not None:
                try:
                    _active_session.reset(self._ctx_token)
                except ValueError:
                    # Context mismatch (e.g. async generator finalized in another task)
                    pass
                self._ctx_token = None

    def _finalize(self, *, exc_occurred: bool = False) -> None:
        try:
            self._uninstall_httpx_patch()
        finally:
            self._active = False

        if not self._is_recording:
            return
        if exc_occurred and not self.save_on_error:
            return
        with self._lock:
            recorded_copy = list(self._recorded)
        should_save = bool(recorded_copy) or (self.mode is TapeMode.RECORD and not exc_occurred)
        if should_save:
            cassette = Cassette(
                path=self.cassette_path,
                interactions=recorded_copy,
                metadata={"llm_tape_version": "0.2.0"},
            )
            cassette.save()

    def __enter__(self) -> TapeSession:
        self._prepare()
        try:
            self._install_httpx_patch()
        except Exception:
            self._active = False
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._finalize(exc_occurred=exc_type is not None)

    async def __aenter__(self) -> TapeSession:
        self._prepare()
        try:
            self._install_httpx_patch()
        except Exception:
            self._active = False
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._finalize(exc_occurred=exc_type is not None)

    # --- Filtering & Matching ----------------------------------------------

    def _should_ignore(self, request: httpx.Request) -> bool:
        host = (request.url.host or "").lower()
        if self.ignore_localhost and host in _LOCALHOST_HOSTS:
            return True
        if any(fnmatch.fnmatch(host, pattern) for pattern in self.ignore_hosts):
            return True
        if self.ignore_request is not None and self.ignore_request(request):
            return True
        return False

    def _build_recorded_request(self, request: httpx.Request, body_bytes: bytes) -> RecordedRequest:
        clean_url = scrub_url(str(request.url), self.extra_sensitive_query_params)
        clean_headers = scrub_headers(request.headers.items(), self.extra_sensitive_headers)
        decoded_body, body_encoding = decode_body_with_encoding(body_bytes)
        fp = fingerprint_request(
            request.method,
            clean_url,
            decoded_body,
            ignore_fields=self.ignore_fields,
            ignore_query_params=self.ignore_query_params,
        )
        return RecordedRequest(
            method=request.method.upper(),
            url=clean_url,
            headers=clean_headers,
            body=decoded_body,
            fingerprint=fp,
            body_encoding=body_encoding,
        )

    def _replay_response(
        self, request: httpx.Request, recorded_req: RecordedRequest
    ) -> httpx.Response:
        with self._lock:
            queue = self._replay_queues.get(recorded_req.fingerprint)
            if not queue:
                if recorded_req.fingerprint in self._known_fingerprints:
                    total = self._known_fingerprints[recorded_req.fingerprint]
                    raise UnmatchedRequestError(
                        f"Replay queue exhausted for {recorded_req.method} {recorded_req.url}: "
                        f"cassette {self.cassette_path} only recorded {total} matching call(s), "
                        f"but code made call #{total + 1}. "
                        "Pass allow_playback_repeats=True if repeated identical calls are expected."
                    )
                diff_text = format_request_diff(
                    recorded_req,
                    self._all_loaded_requests,
                    ignore_fields=self.ignore_fields,
                    ignore_query_params=self.ignore_query_params,
                )
                raise UnmatchedRequestError(
                    f"No recorded interaction matches {recorded_req.method} {recorded_req.url} "
                    f"(fingerprint={recorded_req.fingerprint}) in cassette {self.cassette_path}.\n"
                    f"Closest recorded request diff:\n{diff_text}"
                )

            interaction = (
                queue[0] if (self.allow_playback_repeats and len(queue) == 1) else queue.popleft()
            )
            self._interactions.append(interaction)

        replay_headers = {
            k: v
            for k, v in interaction.response.headers.items()
            if k.lower() not in _STRIP_REPLAY_HEADERS
        }
        if interaction.response.stream_chunks is not None:
            return httpx.Response(
                status_code=interaction.response.status_code,
                headers=replay_headers,
                stream=_ReplayByteStream(interaction.response.stream_chunks),
                request=request,
            )
        return httpx.Response(
            status_code=interaction.response.status_code,
            headers=replay_headers,
            content=encode_body_with_encoding(
                interaction.response.body, interaction.response.body_encoding
            ),
            request=request,
        )

    def _record_interaction(
        self,
        recorded_req: RecordedRequest,
        response: httpx.Response,
        response_bytes: bytes | None,
        latency_ms: float,
        *,
        stream_chunks: list[str] | None = None,
    ) -> Interaction:
        clean_resp_headers = scrub_headers(response.headers.items(), self.extra_sensitive_headers)
        content_type = response.headers.get("content-type", "")

        if stream_chunks is not None:
            chunks_tuple = tuple(stream_chunks)
            recorded_resp = RecordedResponse(
                status_code=response.status_code,
                headers=clean_resp_headers,
                body=None,
                body_encoding="stream",
                stream_chunks=chunks_tuple,
            )
            llm_summary = parse_llm_interaction(
                url=recorded_req.url,
                request_body=recorded_req.body,
                response_body=None,
                custom_pricing=self.custom_pricing,
                stream_chunks=chunks_tuple,
                content_type=content_type,
            )
        else:
            decoded_resp_body, resp_encoding = decode_body_with_encoding(response_bytes or b"")
            recorded_resp = RecordedResponse(
                status_code=response.status_code,
                headers=clean_resp_headers,
                body=decoded_resp_body,
                body_encoding=resp_encoding,
            )
            llm_summary = parse_llm_interaction(
                url=recorded_req.url,
                request_body=recorded_req.body,
                response_body=decoded_resp_body,
                custom_pricing=self.custom_pricing,
                content_type=content_type,
            )

        interaction = Interaction(
            request=recorded_req,
            response=recorded_resp,
            latency_ms=latency_ms,
            llm=llm_summary,
        )
        with self._lock:
            self._interactions.append(interaction)
            self._recorded.append(interaction)
        return interaction

    def handle_sync(
        self,
        request: httpx.Request,
        sender: Callable[[httpx.Request], httpx.Response],
        *,
        stream: bool = False,
    ) -> httpx.Response:
        if request.extensions.get(_HANDLED_EXT_KEY):
            return sender(request)
        if self.mode is TapeMode.NONE or self._should_ignore(request):
            return sender(request)

        request.extensions[_HANDLED_EXT_KEY] = True
        body_bytes = request.read()
        recorded_req = self._build_recorded_request(request, body_bytes)

        if self._is_replaying:
            return self._replay_response(request, recorded_req)

        start = time.perf_counter()
        response = sender(request)
        content_type = response.headers.get("content-type", "").lower()
        is_sse = "text/event-stream" in content_type
        should_wrap_stream = (is_sse or (stream and response.status_code < 400)) and isinstance(
            response.stream, httpx.SyncByteStream
        )

        if should_wrap_stream:

            def _on_complete(chunks: list[str]) -> None:
                latency_ms = (time.perf_counter() - start) * 1000.0
                self._record_interaction(
                    recorded_req, response, None, latency_ms, stream_chunks=chunks
                )

            assert isinstance(response.stream, httpx.SyncByteStream)
            response.stream = _RecordingSyncStream(response.stream, _on_complete)
            return response

        response_bytes = response.read()
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._record_interaction(recorded_req, response, response_bytes, latency_ms)
        if stream:
            # Preserve readable stream interface for 4xx/5xx error responses on client.stream()
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=_ReplayRawBytesStream(response_bytes),
                request=request,
            )
        return response

    async def handle_async(
        self,
        request: httpx.Request,
        sender: Callable[[httpx.Request], Any],
        *,
        stream: bool = False,
    ) -> httpx.Response:
        if request.extensions.get(_HANDLED_EXT_KEY):
            return await sender(request)  # type: ignore[no-any-return]
        if self.mode is TapeMode.NONE or self._should_ignore(request):
            return await sender(request)  # type: ignore[no-any-return]

        request.extensions[_HANDLED_EXT_KEY] = True
        body_bytes = await request.aread()
        recorded_req = self._build_recorded_request(request, body_bytes)

        if self._is_replaying:
            return self._replay_response(request, recorded_req)

        start = time.perf_counter()
        response: httpx.Response = await sender(request)
        content_type = response.headers.get("content-type", "").lower()
        is_sse = "text/event-stream" in content_type
        should_wrap_stream = (is_sse or (stream and response.status_code < 400)) and isinstance(
            response.stream, httpx.AsyncByteStream
        )

        if should_wrap_stream:

            def _on_complete(chunks: list[str]) -> None:
                latency_ms = (time.perf_counter() - start) * 1000.0
                self._record_interaction(
                    recorded_req, response, None, latency_ms, stream_chunks=chunks
                )

            assert isinstance(response.stream, httpx.AsyncByteStream)
            response.stream = _RecordingAsyncStream(response.stream, _on_complete)
            return response

        response_bytes = await response.aread()
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._record_interaction(recorded_req, response, response_bytes, latency_ms)
        if stream:
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=_ReplayRawBytesStream(response_bytes),
                request=request,
            )
        return response

    # --- Inspection properties ---------------------------------------------

    @property
    def interactions(self) -> list[Interaction]:
        with self._lock:
            return list(self._interactions)

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self._interactions)

    @property
    def total_usage(self) -> Usage:
        with self._lock:
            return Usage(
                input_tokens=sum(i.llm.usage.input_tokens for i in self._interactions),
                output_tokens=sum(i.llm.usage.output_tokens for i in self._interactions),
            )

    @property
    def total_tokens(self) -> int:
        return self.total_usage.total_tokens

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return sum(i.llm.estimated_cost_usd for i in self._interactions)

    @property
    def tool_calls(self) -> list[ToolCall]:
        with self._lock:
            calls: list[ToolCall] = []
            for interaction in self._interactions:
                calls.extend(interaction.llm.tool_calls)
            return calls

    @property
    def tool_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]

    @property
    def is_replaying(self) -> bool:
        return self._is_replaying

    # --- Assertions --------------------------------------------------------

    def assert_call_count(self, expected: int) -> None:
        actual = self.call_count
        if actual != expected:
            raise TapeAssertionError(
                f"Expected {expected} LLM/HTTP call(s), but observed {actual}."
            )

    def assert_tokens_under(self, max_tokens: int) -> None:
        actual = self.total_tokens
        if actual > max_tokens:
            raise TapeAssertionError(
                f"Token budget exceeded: used {actual} tokens (limit: {max_tokens})."
            )

    def assert_cost_under(self, max_usd: float) -> None:
        actual = self.total_cost_usd
        if actual > max_usd:
            raise TapeAssertionError(
                f"Cost budget exceeded: estimated ${actual:.6f} (limit: ${max_usd:.6f})."
            )

    def assert_tool_called(self, name: str, *, min_times: int = 1) -> None:
        calls = self.tool_calls
        count = sum(1 for tc in calls if tc.name == name)
        if count < min_times:
            raise TapeAssertionError(
                f"Expected tool {name!r} to be called at least {min_times} time(s), "
                f"but it was called {count} time(s). Observed tools: {[tc.name for tc in calls]}"
            )

    def assert_tool_sequence(self, expected_names: list[str] | tuple[str, ...]) -> None:
        actual = self.tool_names
        expected = list(expected_names)
        if actual != expected:
            raise TapeAssertionError(
                f"Tool call sequence mismatch: expected {expected}, got {actual}."
            )


def tape(
    cassette_path: str | Path,
    *,
    mode: TapeMode | str = TapeMode.ONCE,
    custom_pricing: dict[str, ModelPrice] | None = None,
    extra_sensitive_headers: Iterable[str] = (),
    extra_sensitive_query_params: Iterable[str] = (),
    ignore_hosts: Iterable[str] = (),
    ignore_localhost: bool = False,
    ignore_request: Callable[[httpx.Request], bool] | None = None,
    ignore_fields: Iterable[str] = (),
    ignore_query_params: Iterable[str] = (),
    allow_playback_repeats: bool = False,
    save_on_error: bool = False,
    patch_httpx: bool = True,
) -> TapeSession:
    """Create a TapeSession context manager for recording/replaying LLM calls."""
    return TapeSession(
        cassette_path=cassette_path,
        mode=mode,
        custom_pricing=custom_pricing,
        extra_sensitive_headers=extra_sensitive_headers,
        extra_sensitive_query_params=extra_sensitive_query_params,
        ignore_hosts=ignore_hosts,
        ignore_localhost=ignore_localhost,
        ignore_request=ignore_request,
        ignore_fields=ignore_fields,
        ignore_query_params=ignore_query_params,
        allow_playback_repeats=allow_playback_repeats,
        save_on_error=save_on_error,
        patch_httpx=patch_httpx,
    )


def use_tape(
    cassette_path: str | Path,
    *,
    mode: TapeMode | str = TapeMode.ONCE,
    **session_kwargs: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that wraps a sync or async function inside a TapeSession."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                async with TapeSession(cassette_path, mode=mode, **session_kwargs):
                    return await fn(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with TapeSession(cassette_path, mode=mode, **session_kwargs):
                return fn(*args, **kwargs)

        return sync_wrapper

    return decorator
