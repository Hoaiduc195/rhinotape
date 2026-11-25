"""TapeSession: contextvars-safe recording, replaying, SSE streaming, and LLM assertions."""

from __future__ import annotations

import asyncio
import codecs
import contextvars
import fnmatch
import functools
import inspect
import math
import re
import threading
import time
import zlib
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, ParamSpec, TypeVar
from urllib.parse import urlsplit

import httpx

from llm_tape._version import __version__
from llm_tape.aiohttp_adapter import acquire_aiohttp_patch, release_aiohttp_patch
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
from llm_tape.parsers import parse_llm_interaction, split_sse_chunks
from llm_tape.pricing import ModelPrice
from llm_tape.requests_adapter import acquire_requests_patch, release_requests_patch
from llm_tape.scrubbing import (
    DEFAULT_MATCH_ON,
    compile_patterns,
    decode_body_with_encoding,
    encode_body_with_encoding,
    fingerprint_request,
    format_request_diff,
    redact_body,
    redact_text,
    scrub_headers,
    scrub_url,
    validate_match_on,
)
from llm_tape.urllib_adapter import acquire_urllib_patch, release_urllib_patch

P = ParamSpec("P")
R = TypeVar("R")

_STRIP_REPLAY_HEADERS = frozenset({"content-encoding", "transfer-encoding", "content-length"})
_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_HANDLED_EXT_KEY = "_llm_tape_handled"
_UNSET: Any = object()


def replace_response(
    interaction: Interaction,
    *,
    status_code: int | None = None,
    body: Any = _UNSET,
    headers: dict[str, str] | None = None,
    stream_chunks: tuple[str, ...] | None = _UNSET,
    latency_ms: float | None = None,
    custom_pricing: dict[str, ModelPrice] | None = None,
) -> Interaction:
    """Return a copy of an Interaction with modified response fields for chaos/fault replay."""
    orig = interaction.response
    new_status = orig.status_code if status_code is None else int(status_code)
    new_headers = dict(orig.headers) if headers is None else dict(headers)
    new_latency = interaction.latency_ms if latency_ms is None else float(latency_ms)

    body_changed = body is not _UNSET
    chunks_changed = stream_chunks is not _UNSET

    if chunks_changed and stream_chunks is not None:
        new_chunks: tuple[str, ...] | None = tuple(stream_chunks)
        new_body: Any = None
        new_encoding = "stream"
    elif body_changed:
        new_chunks = None
        if isinstance(body, (bytes, bytearray, memoryview)):
            new_body, new_encoding = decode_body_with_encoding(bytes(body))
        elif body is None:
            new_body = None
            new_encoding = "empty"
        elif isinstance(body, (dict, list, int, float, bool)):
            new_body = body
            new_encoding = "json"
        else:
            new_body = body
            new_encoding = "text"
    else:
        new_chunks = orig.stream_chunks if stream_chunks is _UNSET else stream_chunks
        new_body = orig.body
        new_encoding = orig.body_encoding
        if new_chunks is None and new_encoding == "stream":
            new_encoding = "empty" if new_body is None else "text"

    new_resp = RecordedResponse(
        status_code=new_status,
        headers=new_headers,
        body=new_body,
        body_encoding=new_encoding,
        stream_chunks=new_chunks,
    )
    if body_changed or chunks_changed:
        ct = ""
        for k, v in new_headers.items():
            if k.lower() == "content-type":
                ct = v
                break
        new_llm = parse_llm_interaction(
            url=interaction.request.url,
            request_body=interaction.request.body,
            response_body=new_body,
            custom_pricing=custom_pricing,
            stream_chunks=new_chunks,
            content_type=ct,
        )
    else:
        new_llm = interaction.llm

    return Interaction(
        request=interaction.request,
        response=new_resp,
        latency_ms=new_latency,
        llm=new_llm,
    )


# --- Global contextvars + ref-counted httpx dispatcher ---------------------

_active_session: contextvars.ContextVar[TapeSession | None] = contextvars.ContextVar(
    "llm_tape_active_session", default=None
)
_patch_lock = threading.Lock()
_patch_refcount = 0
_orig_sync_send: Any = httpx.Client.send
_orig_async_send: Any = httpx.AsyncClient.send


def _dispatched_sync_send(
    client: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any
) -> httpx.Response:
    with _patch_lock:
        orig = _orig_sync_send
    session = _active_session.get()
    if session is None:
        return orig(client, request, *args, **kwargs)  # type: ignore[no-any-return]
    stream = bool(kwargs.get("stream", False))
    return session.handle_sync(
        request,
        lambda req: orig(client, req, *args, **kwargs),
        stream=stream,
    )


async def _dispatched_async_send(
    client: httpx.AsyncClient, request: httpx.Request, *args: Any, **kwargs: Any
) -> httpx.Response:
    with _patch_lock:
        orig = _orig_async_send
    session = _active_session.get()
    if session is None:
        return await orig(client, request, *args, **kwargs)  # type: ignore[no-any-return]
    stream = bool(kwargs.get("stream", False))
    return await session.handle_async(
        request,
        lambda req: orig(client, req, *args, **kwargs),
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
    global _patch_refcount
    with _patch_lock:
        if _patch_refcount > 0:
            _patch_refcount -= 1
            if _patch_refcount == 0:
                httpx.Client.send = _orig_sync_send  # type: ignore[method-assign]
                httpx.AsyncClient.send = _orig_async_send  # type: ignore[method-assign]


# --- Streaming Content Decompression & ByteStream wrappers -----------------


class _StreamContentDecoder:
    """Incrementally decompresses gzip/deflate/br/zstd stream chunks for recording."""

    def __init__(self, content_encoding: str) -> None:
        enc = content_encoding.strip().lower()
        self._mode = "identity"
        self._zlib_obj: Any = None
        self._brotli_obj: Any = None
        self._deflate_pending: bytearray | None = None
        if enc in {"gzip", "x-gzip"}:
            self._mode = "zlib"
            self._zlib_obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif enc == "deflate":
            self._mode = "zlib"
            self._deflate_pending = bytearray()
        elif enc == "br":
            try:
                import brotli  # type: ignore[import-not-found]

                self._mode = "brotli"
                self._brotli_obj = brotli.Decompressor()
            except ImportError:
                self._mode = "identity"
        elif enc == "zstd":
            try:
                import zstandard  # type: ignore[import-not-found]

                self._mode = "zstd"
                self._zstd_obj: Any = zstandard.ZstdDecompressor().decompressobj()
            except ImportError:
                self._mode = "identity"

    def _init_deflate(self, data: bytes) -> bytes:
        self._deflate_pending = None
        try:
            obj = zlib.decompressobj(zlib.MAX_WBITS)
            out = obj.decompress(data)
            self._zlib_obj = obj
            return bytes(out)
        except zlib.error:
            # Fallback to raw deflate (RFC 1951 without zlib wrapper)
            obj = zlib.decompressobj(-zlib.MAX_WBITS)
            out = obj.decompress(data)
            self._zlib_obj = obj
            return bytes(out)

    def decompress(self, chunk: bytes) -> bytes:
        if not chunk or self._mode == "identity":
            return chunk
        if self._mode == "zlib":
            if self._deflate_pending is not None:
                self._deflate_pending.extend(chunk)
                if len(self._deflate_pending) < 2:
                    return b""
                return self._init_deflate(bytes(self._deflate_pending))
            return bytes(self._zlib_obj.decompress(chunk))
        if self._mode == "brotli":
            return bytes(self._brotli_obj.process(chunk))
        if self._mode == "zstd":
            return bytes(self._zstd_obj.decompress(chunk))
        return chunk

    def flush(self) -> bytes:
        if self._mode == "zlib":
            if self._deflate_pending is not None:
                if not self._deflate_pending:
                    self._deflate_pending = None
                    return b""
                out = self._init_deflate(bytes(self._deflate_pending))
                return out + bytes(self._zlib_obj.flush())
            return bytes(self._zlib_obj.flush())
        if self._mode == "zstd":
            return bytes(self._zstd_obj.flush())
        return b""


class _ReplayByteStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Yields recorded SSE/stream chunks for both sync and async httpx replay."""

    def __init__(self, chunks: tuple[str, ...], *, chunk_delay_s: float = 0.0) -> None:
        self._chunks = [c.encode("utf-8") for c in chunks]
        self._chunk_delay_s = max(0.0, chunk_delay_s)

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            if self._chunk_delay_s > 0.0:
                time.sleep(self._chunk_delay_s)
            yield chunk

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            if self._chunk_delay_s > 0.0:
                await asyncio.sleep(self._chunk_delay_s)
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
    """Wraps a live sync stream, decompressing & decoding incrementally and committing on EOF."""

    def __init__(
        self,
        upstream_stream: httpx.SyncByteStream,
        on_complete: Callable[[list[str]], None],
        *,
        content_encoding: str = "",
    ) -> None:
        self._upstream = upstream_stream
        self._on_complete = on_complete
        self._content_decoder = _StreamContentDecoder(content_encoding)
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._chunks: list[str] = []
        self._reached_eof = False
        self._committed = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._upstream:
            decoded_bytes = self._content_decoder.decompress(chunk)
            text = self._utf8_decoder.decode(decoded_bytes, final=False)
            if text:
                self._chunks.append(text)
            yield chunk
        flushed = self._content_decoder.flush()
        tail = self._utf8_decoder.decode(flushed, final=True)
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
    """Wraps a live async stream, decompressing & decoding incrementally and committing on EOF."""

    def __init__(
        self,
        upstream_stream: httpx.AsyncByteStream,
        on_complete: Callable[[list[str]], None],
        *,
        content_encoding: str = "",
    ) -> None:
        self._upstream = upstream_stream
        self._on_complete = on_complete
        self._content_decoder = _StreamContentDecoder(content_encoding)
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._chunks: list[str] = []
        self._reached_eof = False
        self._committed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._upstream:
            decoded_bytes = self._content_decoder.decompress(chunk)
            text = self._utf8_decoder.decode(decoded_bytes, final=False)
            if text:
                self._chunks.append(text)
            yield chunk
        flushed = self._content_decoder.flush()
        tail = self._utf8_decoder.decode(flushed, final=True)
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
        redact_fields: Iterable[str] = (),
        redact_patterns: Iterable[str | re.Pattern[str]] = (),
        scrub_body: Callable[[Any], Any] | None = None,
        match_on: Iterable[str] = DEFAULT_MATCH_ON,
        custom_matcher: Callable[[RecordedRequest, RecordedRequest], bool] | None = None,
        on_replay: Callable[[Interaction, int], Interaction | None] | None = None,
        replay_latency_scale: float = 0.0,
        allow_playback_repeats: bool = False,
        save_on_error: bool = False,
        patch_httpx: bool = True,
        patch_requests: bool = True,
        patch_aiohttp: bool = True,
        patch_urllib: bool = False,
    ) -> None:
        if not math.isfinite(replay_latency_scale) or replay_latency_scale < 0.0:
            raise ValueError("replay_latency_scale must be a finite number >= 0.0")
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
        self.redact_fields = tuple(redact_fields)
        self.redact_patterns = tuple(redact_patterns)
        self._compiled_redact_patterns = compile_patterns(self.redact_patterns)
        self._has_body_redaction = bool(
            self.redact_fields or self._compiled_redact_patterns or scrub_body is not None
        )
        self.scrub_body = scrub_body
        self.match_on = validate_match_on(match_on)
        self.custom_matcher = custom_matcher
        self.on_replay = on_replay
        self.replay_latency_scale = float(replay_latency_scale)
        self.allow_playback_repeats = allow_playback_repeats
        self.save_on_error = save_on_error
        self.patch_httpx = patch_httpx
        self.patch_requests = patch_requests
        self.patch_aiohttp = patch_aiohttp
        self.patch_urllib = patch_urllib

        self._lock = threading.Lock()
        self._interactions: list[Interaction] = []
        self._recorded: list[Interaction] = []
        self._replay_queues: dict[str, deque[tuple[int, Interaction]]] = defaultdict(deque)
        self._custom_replay_list: list[tuple[int, Interaction]] = []
        self._total_loaded_count = 0
        self._consumed_indices: set[int] = set()
        self._replay_call_seq = 0
        self._known_fingerprints: dict[str, int] = {}
        self._all_loaded_requests: list[RecordedRequest] = []
        self._is_replaying = False
        self._is_recording = False
        self._active = False
        self._ctx_token: contextvars.Token[TapeSession | None] | None = None
        self._patch_acquired = False
        self._requests_patch_acquired = False
        self._aiohttp_patch_acquired = False
        self._urllib_patch_acquired = False

    # --- Lifecycle ---------------------------------------------------------

    def _prepare(self) -> None:
        if self._active:
            raise TapeError("TapeSession is already active and cannot be re-entered concurrently.")
        self._active = True
        try:
            with self._lock:
                self._interactions.clear()
                self._recorded.clear()
                self._replay_queues.clear()
                self._custom_replay_list.clear()
                self._total_loaded_count = 0
                self._consumed_indices.clear()
                self._replay_call_seq = 0
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
        except Exception:
            self._active = False
            raise

    def _load_replay_queues(self, cassette: Cassette) -> None:
        with self._lock:
            self._total_loaded_count = len(cassette.interactions)
            for idx, interaction in enumerate(cassette.interactions):
                fp = fingerprint_request(
                    interaction.request.method,
                    interaction.request.url,
                    interaction.request.body,
                    headers=interaction.request.headers,
                    ignore_fields=self.ignore_fields,
                    ignore_query_params=self.ignore_query_params,
                    match_on=self.match_on,
                )
                self._replay_queues[fp].append((idx, interaction))
                self._custom_replay_list.append((idx, interaction))
                self._known_fingerprints[fp] = self._known_fingerprints.get(fp, 0) + 1
                self._all_loaded_requests.append(interaction.request)

    def _install_httpx_patch(self) -> None:
        if self.mode is TapeMode.NONE or (
            not self.patch_httpx
            and not self.patch_requests
            and not self.patch_aiohttp
            and not self.patch_urllib
        ):
            return
        self._ctx_token = _active_session.set(self)
        try:
            if self.patch_httpx:
                _acquire_httpx_patch()
                self._patch_acquired = True
            if self.patch_requests:
                self._requests_patch_acquired = acquire_requests_patch(_active_session.get)
            if self.patch_aiohttp:
                self._aiohttp_patch_acquired = acquire_aiohttp_patch(_active_session.get)
            if self.patch_urllib:
                self._urllib_patch_acquired = acquire_urllib_patch(_active_session.get)
        except Exception:
            self._uninstall_httpx_patch()
            raise

    def _uninstall_httpx_patch(self) -> None:
        try:
            if self._patch_acquired:
                _release_httpx_patch()
                self._patch_acquired = False
        finally:
            try:
                if self._requests_patch_acquired:
                    release_requests_patch()
                    self._requests_patch_acquired = False
            finally:
                try:
                    if self._aiohttp_patch_acquired:
                        release_aiohttp_patch()
                        self._aiohttp_patch_acquired = False
                finally:
                    try:
                        if self._urllib_patch_acquired:
                            release_urllib_patch()
                            self._urllib_patch_acquired = False
                    finally:
                        if self._ctx_token is not None:
                            try:
                                _active_session.reset(self._ctx_token)
                            except ValueError:
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
                metadata={"llm_tape_version": __version__},
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

    def _should_ignore_url(self, url: str) -> bool:
        host = (urlsplit(url).hostname or "").lower()
        if self.ignore_localhost and host in _LOCALHOST_HOSTS:
            return True
        if any(fnmatch.fnmatch(host, pattern) for pattern in self.ignore_hosts):
            return True
        return False

    def _should_ignore(self, request: httpx.Request) -> bool:
        host = (request.url.host or "").lower()
        if self.ignore_localhost and host in _LOCALHOST_HOSTS:
            return True
        if any(fnmatch.fnmatch(host, pattern) for pattern in self.ignore_hosts):
            return True
        if self.ignore_request is not None and self.ignore_request(request):
            return True
        return False

    def _build_recorded_request_parts(
        self,
        *,
        method: str,
        url: str,
        headers: Iterable[tuple[str, str]],
        body_bytes: bytes,
    ) -> RecordedRequest:
        clean_url = scrub_url(url, self.extra_sensitive_query_params)
        clean_headers = scrub_headers(headers, self.extra_sensitive_headers)
        decoded_body, body_encoding = decode_body_with_encoding(body_bytes)
        if self._has_body_redaction and body_encoding not in {"base64", "empty"}:
            decoded_body = redact_body(
                decoded_body,
                redact_fields=self.redact_fields,
                redact_patterns=self._compiled_redact_patterns,
                scrub_body=self.scrub_body,
            )
        fp = fingerprint_request(
            method,
            clean_url,
            decoded_body,
            headers=clean_headers,
            ignore_fields=self.ignore_fields,
            ignore_query_params=self.ignore_query_params,
            match_on=self.match_on,
        )
        return RecordedRequest(
            method=method.upper(),
            url=clean_url,
            headers=clean_headers,
            body=decoded_body,
            fingerprint=fp,
            body_encoding=body_encoding,
        )

    def _build_recorded_request(self, request: httpx.Request, body_bytes: bytes) -> RecordedRequest:
        return self._build_recorded_request_parts(
            method=request.method,
            url=str(request.url),
            headers=request.headers.items(),
            body_bytes=body_bytes,
        )

    def _pop_replay_interaction(self, recorded_req: RecordedRequest) -> Interaction:
        with self._lock:
            if self.custom_matcher is not None:
                matched: Interaction | None = None
                for pos, (orig_idx, candidate) in enumerate(self._custom_replay_list):
                    if self.custom_matcher(candidate.request, recorded_req):
                        has_later_match = self.allow_playback_repeats and any(
                            self.custom_matcher(later.request, recorded_req)
                            for _, later in self._custom_replay_list[pos + 1 :]
                        )
                        if self.allow_playback_repeats and not has_later_match:
                            matched = candidate
                        else:
                            _, matched = self._custom_replay_list.pop(pos)
                        self._consumed_indices.add(orig_idx)
                        break
                if matched is None:
                    diff_text = format_request_diff(
                        recorded_req,
                        self._all_loaded_requests,
                        ignore_fields=self.ignore_fields,
                        ignore_query_params=self.ignore_query_params,
                    )
                    raise UnmatchedRequestError(
                        f"No recorded interaction matched custom_matcher for "
                        f"{recorded_req.method} {recorded_req.url} in cassette "
                        f"{self.cassette_path}.\nClosest recorded request diff:\n{diff_text}"
                    )
                interaction = matched
            else:
                queue = self._replay_queues.get(recorded_req.fingerprint)
                if not queue:
                    if recorded_req.fingerprint in self._known_fingerprints:
                        total = self._known_fingerprints[recorded_req.fingerprint]
                        raise UnmatchedRequestError(
                            f"Replay queue exhausted for {recorded_req.method} {recorded_req.url}: "
                            f"cassette {self.cassette_path} only recorded {total} matching "
                            f"call(s), but code made call #{total + 1}. "
                            "Pass allow_playback_repeats=True if repeated identical calls are "
                            "expected."
                        )
                    diff_text = format_request_diff(
                        recorded_req,
                        self._all_loaded_requests,
                        ignore_fields=self.ignore_fields,
                        ignore_query_params=self.ignore_query_params,
                    )
                    raise UnmatchedRequestError(
                        f"No recorded interaction matches {recorded_req.method} {recorded_req.url} "
                        f"(fingerprint={recorded_req.fingerprint}) in cassette "
                        f"{self.cassette_path}.\nClosest recorded request diff:\n{diff_text}"
                    )

                orig_idx, interaction = (
                    queue[0]
                    if (self.allow_playback_repeats and len(queue) == 1)
                    else queue.popleft()
                )
                self._consumed_indices.add(orig_idx)

            call_idx = self._replay_call_seq
            self._replay_call_seq += 1

        if self.on_replay is not None:
            transformed = self.on_replay(interaction, call_idx)
            if transformed is not None:
                if not isinstance(transformed, Interaction):
                    raise TypeError(
                        f"on_replay callback must return Interaction or None, "
                        f"got {type(transformed).__name__}"
                    )
                interaction = transformed

        with self._lock:
            self._interactions.append(interaction)
        return interaction

    def _build_replay_httpx_response(
        self,
        request: httpx.Request,
        interaction: Interaction,
        *,
        stream: bool = False,
        chunk_delay_s: float = 0.0,
    ) -> httpx.Response:
        replay_headers = {
            k: v
            for k, v in interaction.response.headers.items()
            if k.lower() not in _STRIP_REPLAY_HEADERS
        }
        if interaction.response.stream_chunks is not None:
            if stream:
                return httpx.Response(
                    status_code=interaction.response.status_code,
                    headers=replay_headers,
                    stream=_ReplayByteStream(
                        interaction.response.stream_chunks, chunk_delay_s=chunk_delay_s
                    ),
                    request=request,
                )
            joined = "".join(interaction.response.stream_chunks).encode("utf-8")
            return httpx.Response(
                status_code=interaction.response.status_code,
                headers=replay_headers,
                content=joined,
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

    def _replay_response(
        self,
        request: httpx.Request,
        recorded_req: RecordedRequest,
        *,
        stream: bool = False,
    ) -> httpx.Response:
        interaction = self._pop_replay_interaction(recorded_req)
        delay_s = (
            (interaction.latency_ms * self.replay_latency_scale) / 1000.0
            if self.replay_latency_scale > 0.0 and interaction.latency_ms > 0.0
            else 0.0
        )
        chunk_delay_s = 0.0
        if delay_s > 0.0:
            if stream and interaction.response.stream_chunks:
                chunk_delay_s = delay_s / len(interaction.response.stream_chunks)
            else:
                time.sleep(delay_s)
        return self._build_replay_httpx_response(
            request, interaction, stream=stream, chunk_delay_s=chunk_delay_s
        )

    async def _replay_response_async(
        self,
        request: httpx.Request,
        recorded_req: RecordedRequest,
        *,
        stream: bool = False,
    ) -> httpx.Response:
        interaction = self._pop_replay_interaction(recorded_req)
        delay_s = (
            (interaction.latency_ms * self.replay_latency_scale) / 1000.0
            if self.replay_latency_scale > 0.0 and interaction.latency_ms > 0.0
            else 0.0
        )
        chunk_delay_s = 0.0
        if delay_s > 0.0:
            if stream and interaction.response.stream_chunks:
                chunk_delay_s = delay_s / len(interaction.response.stream_chunks)
            else:
                await asyncio.sleep(delay_s)
        return self._build_replay_httpx_response(
            request, interaction, stream=stream, chunk_delay_s=chunk_delay_s
        )

    def _record_interaction_parts(
        self,
        *,
        recorded_req: RecordedRequest,
        status_code: int,
        response_headers: Iterable[tuple[str, str]],
        response_bytes: bytes | None,
        latency_ms: float,
        stream_chunks: list[str] | None = None,
    ) -> Interaction:
        headers_list = list(response_headers)
        clean_resp_headers = scrub_headers(headers_list, self.extra_sensitive_headers)
        content_type = ""
        for k, v in headers_list:
            if k.lower() == "content-type":
                content_type = v
                break

        if stream_chunks is not None:
            if self._compiled_redact_patterns:
                per_chunk = [redact_text(c, self._compiled_redact_patterns) for c in stream_chunks]
                joined_per_chunk = "".join(per_chunk)
                joined_redacted = redact_text(
                    "".join(stream_chunks), self._compiled_redact_patterns
                )
                if joined_redacted != joined_per_chunk:
                    # Secret spanned across TCP chunk boundary: re-split on SSE event boundaries
                    stream_chunks = split_sse_chunks(joined_redacted)
                else:
                    stream_chunks = per_chunk
            chunks_tuple = tuple(stream_chunks)
            recorded_resp = RecordedResponse(
                status_code=status_code,
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
            if self._has_body_redaction and resp_encoding not in {"base64", "empty"}:
                decoded_resp_body = redact_body(
                    decoded_resp_body,
                    redact_fields=self.redact_fields,
                    redact_patterns=self._compiled_redact_patterns,
                    scrub_body=self.scrub_body,
                )
            recorded_resp = RecordedResponse(
                status_code=status_code,
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

    def _record_interaction(
        self,
        recorded_req: RecordedRequest,
        response: httpx.Response,
        response_bytes: bytes | None,
        latency_ms: float,
        *,
        stream_chunks: list[str] | None = None,
    ) -> Interaction:
        return self._record_interaction_parts(
            recorded_req=recorded_req,
            status_code=response.status_code,
            response_headers=response.headers.items(),
            response_bytes=response_bytes,
            latency_ms=latency_ms,
            stream_chunks=stream_chunks,
        )

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
        try:
            body_bytes = request.read()
            recorded_req = self._build_recorded_request(request, body_bytes)

            if self._is_replaying:
                return self._replay_response(request, recorded_req, stream=stream)

            start = time.perf_counter()
            response = sender(request)
            content_type = response.headers.get("content-type", "").lower()
            is_sse = (
                "text/event-stream" in content_type
                or "application/x-ndjson" in content_type
                or "application/ndjson" in content_type
            )
            should_wrap_stream = (
                not response.is_stream_consumed
                and (is_sse or (stream and response.status_code < 400))
                and isinstance(response.stream, httpx.SyncByteStream)
            )

            if should_wrap_stream:
                content_encoding = response.headers.get("content-encoding", "")

                def _on_complete(chunks: list[str]) -> None:
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    self._record_interaction(
                        recorded_req, response, None, latency_ms, stream_chunks=chunks
                    )

                assert isinstance(response.stream, httpx.SyncByteStream)
                response.stream = _RecordingSyncStream(
                    response.stream, _on_complete, content_encoding=content_encoding
                )
                return response

            response_bytes = response.content if response.is_stream_consumed else response.read()
            latency_ms = (time.perf_counter() - start) * 1000.0
            if is_sse:
                text_chunk = response_bytes.decode("utf-8", errors="replace")
                self._record_interaction(
                    recorded_req, response, None, latency_ms, stream_chunks=[text_chunk]
                )
            else:
                self._record_interaction(recorded_req, response, response_bytes, latency_ms)
            if stream:
                cleaned_headers = {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower() not in _STRIP_REPLAY_HEADERS
                }
                return httpx.Response(
                    status_code=response.status_code,
                    headers=cleaned_headers,
                    stream=_ReplayRawBytesStream(response_bytes),
                    request=request,
                )
            return response
        finally:
            request.extensions.pop(_HANDLED_EXT_KEY, None)

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
        try:
            body_bytes = await request.aread()
            recorded_req = self._build_recorded_request(request, body_bytes)

            if self._is_replaying:
                return await self._replay_response_async(request, recorded_req, stream=stream)

            start = time.perf_counter()
            response: httpx.Response = await sender(request)
            content_type = response.headers.get("content-type", "").lower()
            is_sse = (
                "text/event-stream" in content_type
                or "application/x-ndjson" in content_type
                or "application/ndjson" in content_type
            )
            should_wrap_stream = (
                not response.is_stream_consumed
                and (is_sse or (stream and response.status_code < 400))
                and isinstance(response.stream, httpx.AsyncByteStream)
            )

            if should_wrap_stream:
                content_encoding = response.headers.get("content-encoding", "")

                def _on_complete(chunks: list[str]) -> None:
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    self._record_interaction(
                        recorded_req, response, None, latency_ms, stream_chunks=chunks
                    )

                assert isinstance(response.stream, httpx.AsyncByteStream)
                response.stream = _RecordingAsyncStream(
                    response.stream, _on_complete, content_encoding=content_encoding
                )
                return response

            response_bytes = (
                response.content if response.is_stream_consumed else await response.aread()
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            if is_sse:
                text_chunk = response_bytes.decode("utf-8", errors="replace")
                self._record_interaction(
                    recorded_req, response, None, latency_ms, stream_chunks=[text_chunk]
                )
            else:
                self._record_interaction(recorded_req, response, response_bytes, latency_ms)
            if stream:
                cleaned_headers = {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower() not in _STRIP_REPLAY_HEADERS
                }
                return httpx.Response(
                    status_code=response.status_code,
                    headers=cleaned_headers,
                    stream=_ReplayRawBytesStream(response_bytes),
                    request=request,
                )
            return response
        finally:
            request.extensions.pop(_HANDLED_EXT_KEY, None)

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
                cached_input_tokens=sum(
                    i.llm.usage.cached_input_tokens for i in self._interactions
                ),
                reasoning_tokens=sum(i.llm.usage.reasoning_tokens for i in self._interactions),
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
    def all_texts(self) -> list[str]:
        with self._lock:
            return [i.llm.text for i in self._interactions if i.llm.text]

    @property
    def last_text(self) -> str:
        with self._lock:
            return self._interactions[-1].llm.text if self._interactions else ""

    @property
    def last_thinking(self) -> str:
        with self._lock:
            return self._interactions[-1].llm.thinking if self._interactions else ""

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
        if round(actual, 8) > max_usd:
            raise TapeAssertionError(
                f"Cost budget exceeded: estimated ${actual:.6f} (limit: ${max_usd:.6f})."
            )

    def assert_cached_tokens_at_least(self, min_cached: int) -> None:
        actual = self.total_usage.cached_input_tokens
        if actual < min_cached:
            raise TapeAssertionError(
                f"Expected at least {min_cached} cached input token(s), but observed {actual}."
            )

    def assert_model(self, expected_model: str) -> None:
        interactions = self.interactions
        if not interactions:
            raise TapeAssertionError(
                f"Expected model {expected_model!r}, but no interactions were recorded/replayed."
            )
        mismatched = [i.llm.model for i in interactions if i.llm.model != expected_model]
        if mismatched:
            raise TapeAssertionError(
                f"Model mismatch: expected {expected_model!r} for all calls, "
                f"but observed {[i.llm.model for i in interactions]}."
            )

    def assert_provider(self, expected_provider: str) -> None:
        interactions = self.interactions
        if not interactions:
            raise TapeAssertionError(
                f"Expected provider {expected_provider!r}, "
                "but no interactions were recorded/replayed."
            )
        mismatched = [i.llm.provider for i in interactions if i.llm.provider != expected_provider]
        if mismatched:
            raise TapeAssertionError(
                f"Provider mismatch: expected {expected_provider!r} for all calls, "
                f"but observed {[i.llm.provider for i in interactions]}."
            )

    def assert_no_errors(self) -> None:
        failed = [
            f"{i.request.method} {i.request.url} -> {i.response.status_code}"
            for i in self.interactions
            if i.response.status_code >= 400
        ]
        if failed:
            raise TapeAssertionError(
                f"Expected all HTTP calls to succeed (< 400), but got error(s): {failed}."
            )

    def assert_text_contains(self, substring: str, *, case_sensitive: bool = True) -> None:
        texts = self.all_texts
        needle = substring if case_sensitive else substring.lower()
        found = any((needle in t) if case_sensitive else (needle in t.lower()) for t in texts)
        if not found:
            raise TapeAssertionError(
                f"Expected LLM response text to contain {substring!r}, "
                f"but observed texts: {texts!r}."
            )

    def assert_text_matches(self, pattern: str | re.Pattern[str]) -> None:
        regex = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
        texts = self.all_texts
        if not any(regex.search(t) for t in texts):
            raise TapeAssertionError(
                f"Expected LLM response text to match pattern {regex.pattern!r}, "
                f"but observed texts: {texts!r}."
            )

    def assert_no_tool_called(self, name: str) -> None:
        calls = self.tool_calls
        count = sum(1 for tc in calls if tc.name == name)
        if count > 0:
            raise TapeAssertionError(
                f"Expected tool {name!r} NOT to be called, but it was called {count} time(s)."
            )

    def assert_thinking_present(self) -> None:
        if not any(i.llm.thinking for i in self.interactions):
            raise TapeAssertionError(
                "Expected at least one LLM call to include non-empty thinking output, "
                "but none was observed."
            )

    def assert_reasoning_tokens_under(self, max_tokens: int) -> None:
        actual = self.total_usage.reasoning_tokens
        if actual > max_tokens:
            raise TapeAssertionError(
                f"Reasoning token budget exceeded: used {actual} tokens (limit: {max_tokens})."
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

    def assert_tool_args(
        self,
        name: str,
        expected: dict[str, Any],
        *,
        exact: bool = False,
    ) -> None:
        """Assert at least one call to `name` matches `expected` arguments (subset by default)."""
        matching_calls = [tc for tc in self.tool_calls if tc.name == name]
        if not matching_calls:
            raise TapeAssertionError(
                f"Expected tool {name!r} to be called with {expected!r}, "
                f"but it was never called. Observed tools: {self.tool_names}"
            )
        for tc in matching_calls:
            if not isinstance(tc.arguments, dict):
                continue
            if exact:
                if tc.arguments == expected:
                    return
            else:
                if all(k in tc.arguments and tc.arguments[k] == v for k, v in expected.items()):
                    return
        observed = [tc.arguments for tc in matching_calls]
        mode_label = "exact" if exact else "subset"
        raise TapeAssertionError(
            f"Tool {name!r} arguments did not match ({mode_label}): "
            f"expected {expected!r}, observed {observed!r}."
        )

    def assert_no_unused_interactions(self) -> None:
        """Assert that all recorded interactions in the cassette were consumed during replay."""
        if not self._is_replaying:
            return
        with self._lock:
            remaining = max(0, self._total_loaded_count - len(self._consumed_indices))
        if remaining > 0:
            raise TapeAssertionError(
                f"Cassette {self.cassette_path} has {remaining} unused recorded interaction(s). "
                "Re-record with mode='record' if the test no longer makes those calls."
            )

    def assert_max_latency_ms(self, max_ms: float) -> None:
        """Assert that no recorded/replayed call exceeded `max_ms` milliseconds."""
        with self._lock:
            slow = [i.latency_ms for i in self._interactions if i.latency_ms > max_ms]
        if slow:
            raise TapeAssertionError(
                f"Latency budget exceeded: slowest call took {max(slow):.2f}ms "
                f"(limit: {max_ms:.2f}ms)."
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
    redact_fields: Iterable[str] = (),
    redact_patterns: Iterable[str | re.Pattern[str]] = (),
    scrub_body: Callable[[Any], Any] | None = None,
    match_on: Iterable[str] = DEFAULT_MATCH_ON,
    custom_matcher: Callable[[RecordedRequest, RecordedRequest], bool] | None = None,
    on_replay: Callable[[Interaction, int], Interaction | None] | None = None,
    replay_latency_scale: float = 0.0,
    allow_playback_repeats: bool = False,
    save_on_error: bool = False,
    patch_httpx: bool = True,
    patch_requests: bool = True,
    patch_aiohttp: bool = True,
    patch_urllib: bool = False,
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
        redact_fields=redact_fields,
        redact_patterns=redact_patterns,
        scrub_body=scrub_body,
        match_on=match_on,
        custom_matcher=custom_matcher,
        on_replay=on_replay,
        replay_latency_scale=replay_latency_scale,
        allow_playback_repeats=allow_playback_repeats,
        save_on_error=save_on_error,
        patch_httpx=patch_httpx,
        patch_requests=patch_requests,
        patch_aiohttp=patch_aiohttp,
        patch_urllib=patch_urllib,
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
