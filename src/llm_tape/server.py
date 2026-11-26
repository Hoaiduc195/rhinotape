"""Local HTTP replay & reverse-proxy server with polyglot control plane for any HTTP client."""

from __future__ import annotations

import codecs
import http.server
import json
import math
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from llm_tape._version import __version__
from llm_tape.cassette import MAX_CASSETTE_BYTES, Cassette, is_cassette_file, iter_cassette_files
from llm_tape.models import (
    CorruptCassetteError,
    Interaction,
    RecordedRequest,
    RecordedResponse,
    TapeMode,
)
from llm_tape.parsers import parse_llm_interaction, redact_stream_chunks, split_sse_chunks
from llm_tape.pricing import ModelPrice
from llm_tape.scrubbing import (
    DEFAULT_MATCH_ON,
    compile_patterns,
    compile_sensitive_headers,
    decode_body_with_encoding,
    encode_body_with_encoding,
    fingerprint_request,
    redact_body,
    resolve_pii_patterns,
    scrub_headers,
    scrub_url,
    validate_match_on,
)

_STRIP_HEADERS = frozenset(
    {"content-encoding", "transfer-encoding", "content-length", "connection"}
)
_HOP_BY_HOP_REQ_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
        "accept-encoding",
    }
)
_CORS_ALLOW_METHODS = "GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD"
_HEX_DIGITS = frozenset(b"0123456789abcdefABCDEF")
_MAX_CHUNK_LINE_BYTES = 4096
_MAX_TRAILER_LINES = 64
_HANDLED_EXT_KEY = "_llm_tape_handled"
_VALID_ASSERT_KEYS = frozenset(
    {
        "call_count",
        "max_tokens",
        "max_cost_usd",
        "max_latency_ms",
        "no_errors",
        "tool_called",
        "no_tool_called",
        "tool_sequence",
        "text_contains",
        "no_unused_interactions",
    }
)


def _extract_path_and_query(raw_target: str) -> tuple[str, str]:
    """Extract normalized HTTP path and query from a request target safely via urlsplit."""
    if raw_target.startswith("//"):
        parts = urlsplit("/" + raw_target.lstrip("/"))
    else:
        parts = urlsplit(raw_target)
    req_path = parts.path if parts.path.startswith("/") else f"/{parts.path}"
    return req_path, parts.query


def _reject_nonfinite_constant(val: str) -> Any:
    raise ValueError(f"Non-finite JSON constant {val!r} is not allowed")


def _parse_json_object_strict(raw_bytes: bytes) -> dict[str, Any] | None:
    try:
        text = raw_bytes.decode("utf-8")
        parsed = json.loads(text, parse_constant=_reject_nonfinite_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class _BorrowedSyncTransport(httpx.BaseTransport):
    """Delegates to a caller-owned httpx.BaseTransport without closing it on Client.close()."""

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._inner.handle_request(request)

    def close(self) -> None:
        pass


class TapeReplayServer:
    """Thread-safe local HTTP reverse-proxy & replay server with `/__tape__/*` control plane.

    Matches incoming requests by ``(method, path, body)`` by default so SDKs in Python,
    Node/TypeScript, Go, Rust, Java, Ruby, or browsers pointed at ``http://127.0.0.1:<port>``
    transparently record and replay cassettes offline.
    """

    def __init__(
        self,
        source: str | Path | Cassette,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        mode: TapeMode | str = TapeMode.REPLAY,
        upstream_url: str | None = None,
        upstream_transport: httpx.BaseTransport | None = None,
        allow_playback_repeats: bool = True,
        ignore_fields: Iterable[str] = (),
        ignore_query_params: Iterable[str] = (),
        ignore_json_array_order: Iterable[str] = (),
        match_on: Iterable[str] = ("method", "path", "body"),
        match_headers: Iterable[str] = (),
        extra_sensitive_headers: Iterable[str] = (),
        extra_sensitive_query_params: Iterable[str] = (),
        redact_fields: Iterable[str] = (),
        redact_patterns: Iterable[str | re.Pattern[str]] = (),
        redact_pii: bool | str | Iterable[str] = False,
        redact_url_patterns: Iterable[str | re.Pattern[str]] = (),
        custom_pricing: dict[str, ModelPrice] | None = None,
        enable_cors: bool = False,
        cassette_dir: str | Path | None = None,
        auto_save: bool = True,
        replay_latency_scale: float = 0.0,
        max_request_bytes: int = 10 * 1024 * 1024,
        request_timeout: float = 30.0,
    ) -> None:
        coerced_mode = TapeMode.coerce(mode)
        if coerced_mode is TapeMode.NONE:
            raise ValueError(
                "TapeReplayServer does not support mode='none'; "
                "expected replay, once, record, or new_episodes."
            )
        if (
            isinstance(replay_latency_scale, bool)
            or not math.isfinite(replay_latency_scale)
            or replay_latency_scale < 0.0
        ):
            raise ValueError("replay_latency_scale must be a finite number >= 0.0")
        if (
            isinstance(max_request_bytes, bool)
            or not isinstance(max_request_bytes, int)
            or max_request_bytes < 0
        ):
            raise ValueError("max_request_bytes must be an integer >= 0")
        if (
            isinstance(request_timeout, bool)
            or not math.isfinite(request_timeout)
            or request_timeout <= 0.0
        ):
            raise ValueError("request_timeout must be a finite number > 0.0")

        cleaned_upstream: str | None = None
        if upstream_url is not None:
            cleaned_upstream = upstream_url.strip().rstrip("/")
            up_parts = urlsplit(cleaned_upstream)
            if up_parts.scheme.lower() not in {"http", "https"} or not up_parts.netloc:
                raise ValueError(
                    f"Invalid upstream_url {upstream_url!r}: expected http(s)://host[:port]"
                )

        self.host = host
        self.requested_port = port
        self.mode: TapeMode = coerced_mode
        self.upstream_url = cleaned_upstream
        self.upstream_transport = upstream_transport
        self.allow_playback_repeats = allow_playback_repeats
        self.ignore_fields = tuple(ignore_fields)
        self.ignore_query_params = tuple(ignore_query_params)
        self.ignore_json_array_order = tuple(ignore_json_array_order)
        self.match_on = validate_match_on(match_on)
        self.match_headers = tuple(h.lower().strip() for h in match_headers if h.strip())
        self.extra_sensitive_headers = tuple(extra_sensitive_headers)
        self._compiled_sensitive_headers = compile_sensitive_headers(self.extra_sensitive_headers)
        self.extra_sensitive_query_params = tuple(extra_sensitive_query_params)
        self.redact_fields = tuple(f.strip() for f in redact_fields if f.strip())
        self.redact_patterns = tuple(redact_patterns)
        self.redact_pii = redact_pii
        self.redact_url_patterns = tuple(redact_url_patterns)
        self._compiled_url_patterns = compile_patterns(self.redact_url_patterns)
        self._compiled_redact_patterns = (
            *compile_patterns(self.redact_patterns),
            *resolve_pii_patterns(redact_pii),
        )
        self._has_body_redaction = bool(self.redact_fields or self._compiled_redact_patterns)
        self.custom_pricing = custom_pricing
        self.enable_cors = bool(enable_cors)
        self.auto_save = bool(auto_save)
        self.replay_latency_scale = float(replay_latency_scale)
        self.max_request_bytes = max_request_bytes
        self.request_timeout = float(request_timeout)

        if cassette_dir is not None:
            self.cassette_dir = Path(cassette_dir).resolve()
        elif isinstance(source, Cassette):
            self.cassette_dir = source.path.parent.resolve()
        else:
            src_path = Path(source)
            self.cassette_dir = (
                src_path.resolve() if src_path.is_dir() else src_path.parent.resolve()
            )

        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._queues: dict[str, deque[tuple[int, Interaction]]] = defaultdict(deque)
        self._initial_interactions: list[Interaction] = []
        self._existing_metadata: dict[str, Any] = {}
        self._recorded: list[Interaction] = []
        self._observed_interactions: list[Interaction] = []
        self._consumed_indices: set[int] = set()
        self._active_cassette_path: Path | None = None
        self._is_replaying = True
        self._is_recording = False
        self._dirty = False
        self._loaded_count = 0
        self._served_count = 0
        self._unmatched_count = 0
        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._http_client: httpx.Client | None = None

        self._configure_source(source, self.mode)

    def _compute_interaction_fp(self, inter: Interaction) -> str:
        clean_inter_url = scrub_url(
            inter.request.url,
            self.extra_sensitive_query_params,
            redact_url_patterns=self._compiled_url_patterns,
        )
        clean_inter_headers = scrub_headers(
            inter.request.headers, compiled=self._compiled_sensitive_headers
        )
        return fingerprint_request(
            inter.request.method,
            clean_inter_url,
            inter.request.body,
            headers=clean_inter_headers,
            ignore_fields=self.ignore_fields,
            ignore_query_params=self.ignore_query_params,
            ignore_json_array_order=self.ignore_json_array_order,
            match_on=self.match_on,
            match_headers=self.match_headers,
        )

    def _rebuild_queues_locked(self) -> None:
        self._queues.clear()
        for idx, inter in enumerate(self._initial_interactions):
            fp = self._compute_interaction_fp(inter)
            self._queues[fp].append((idx, inter))
        self._loaded_count = len(self._initial_interactions)

    def _configure_source(self, source: str | Path | Cassette, mode: TapeMode) -> None:
        initial: list[Interaction] = []
        existing_meta: dict[str, Any] = {}
        active_path: Path | None = None
        is_replaying = False
        is_recording = False

        if isinstance(source, Cassette):
            active_path = source.path.resolve()
            if mode is TapeMode.RECORD:
                is_replaying = False
                is_recording = True
            elif mode is TapeMode.ONCE:
                if source.interactions or active_path.is_file():
                    loaded = source if source.interactions else Cassette.load(active_path)
                    initial = list(loaded.interactions)
                    existing_meta = dict(loaded.metadata)
                    is_replaying = True
                    is_recording = False
                else:
                    is_replaying = False
                    is_recording = True
            elif mode is TapeMode.NEW_EPISODES:
                initial = list(source.interactions)
                existing_meta = dict(source.metadata)
                is_replaying = True
                is_recording = True
            else:  # REPLAY
                initial = list(source.interactions)
                existing_meta = dict(source.metadata)
                is_replaying = True
                is_recording = False
        else:
            p = Path(source)
            if mode is TapeMode.REPLAY:
                if p.is_file():
                    c = Cassette.load(p)
                    active_path = p.resolve()
                    initial = list(c.interactions)
                    existing_meta = dict(c.metadata)
                elif p.is_dir():
                    for file_path in iter_cassette_files(p):
                        initial.extend(Cassette.load(file_path).interactions)
                else:
                    raise FileNotFoundError(f"Cassette source not found: {p}")
                is_replaying = True
                is_recording = False
            elif mode is TapeMode.RECORD:
                active_path = (p / "recorded.tape.json").resolve() if p.is_dir() else p.resolve()
                is_replaying = False
                is_recording = True
            elif mode is TapeMode.ONCE:
                if p.is_file():
                    c = Cassette.load(p)
                    active_path = p.resolve()
                    initial = list(c.interactions)
                    existing_meta = dict(c.metadata)
                    is_replaying = True
                    is_recording = False
                elif p.is_dir():
                    files = iter_cassette_files(p)
                    if files:
                        for file_path in files:
                            initial.extend(Cassette.load(file_path).interactions)
                        is_replaying = True
                        is_recording = False
                    else:
                        active_path = (p / "recorded.tape.json").resolve()
                        is_replaying = False
                        is_recording = True
                else:
                    active_path = p.resolve()
                    is_replaying = False
                    is_recording = True
            elif mode is TapeMode.NEW_EPISODES:
                if p.is_file():
                    c = Cassette.load(p)
                    active_path = p.resolve()
                    initial = list(c.interactions)
                    existing_meta = dict(c.metadata)
                    is_replaying = True
                    is_recording = True
                elif p.is_dir():
                    for file_path in iter_cassette_files(p):
                        initial.extend(Cassette.load(file_path).interactions)
                    active_path = (p / "recorded.tape.json").resolve()
                    is_replaying = True
                    is_recording = True
                else:
                    active_path = p.resolve()
                    is_replaying = False
                    is_recording = True

        with self._lock:
            self.mode = mode
            self._active_cassette_path = active_path
            self._initial_interactions = initial
            self._existing_metadata = existing_meta
            self._recorded.clear()
            self._observed_interactions.clear()
            self._consumed_indices.clear()
            self._served_count = 0
            self._unmatched_count = 0
            self._is_replaying = is_replaying
            self._is_recording = is_recording
            self._dirty = False
            self._rebuild_queues_locked()

    def _join_upstream_path(self, req_path: str) -> str:
        if not self.upstream_url:
            return req_path
        base_path = urlsplit(self.upstream_url).path.rstrip("/")
        if not base_path or req_path == base_path or req_path.startswith(f"{base_path}/"):
            return req_path
        return f"{base_path}{req_path}"

    def _build_scrubbed_synthetic_url(self, raw_target: str) -> str:
        req_path, req_query = _extract_path_and_query(raw_target)
        effective_path = self._join_upstream_path(req_path)
        raw_synthetic = urlunsplit(("http", "localhost", effective_path, req_query, ""))
        return scrub_url(
            raw_synthetic,
            self.extra_sensitive_query_params,
            redact_url_patterns=self._compiled_url_patterns,
        )

    def _match(
        self,
        method: str,
        path: str,
        body_bytes: bytes,
        headers: dict[str, str] | None = None,
        *,
        count_unmatched: bool = True,
    ) -> Interaction | None:
        decoded_body, body_enc = decode_body_with_encoding(body_bytes)
        if self._has_body_redaction and body_enc not in {"base64", "empty"}:
            decoded_body = redact_body(
                decoded_body,
                redact_fields=self.redact_fields,
                redact_patterns=self._compiled_redact_patterns,
            )
        synthetic_url = self._build_scrubbed_synthetic_url(path)
        fp = fingerprint_request(
            method,
            synthetic_url,
            decoded_body,
            headers=headers,
            ignore_fields=self.ignore_fields,
            ignore_query_params=self.ignore_query_params,
            ignore_json_array_order=self.ignore_json_array_order,
            match_on=self.match_on,
            match_headers=self.match_headers,
        )
        with self._lock:
            q = self._queues.get(fp)
            if not q:
                if count_unmatched:
                    self._unmatched_count += 1
                return None
            orig_idx, inter = q[0] if (self.allow_playback_repeats and len(q) == 1) else q.popleft()
            self._consumed_indices.add(orig_idx)
            self._served_count += 1
            self._observed_interactions.append(inter)
            return inter

    def _get_upstream_client(self) -> httpx.Client:
        with self._lock:
            if self._http_client is None:
                transport = (
                    _BorrowedSyncTransport(self.upstream_transport)
                    if self.upstream_transport is not None
                    else None
                )
                self._http_client = httpx.Client(
                    transport=transport,
                    timeout=self.request_timeout,
                    follow_redirects=True,
                )
            return self._http_client

    def _build_upstream_target_url(self, raw_target: str) -> str:
        assert self.upstream_url is not None
        req_path, req_query = _extract_path_and_query(raw_target)
        up_parts = urlsplit(self.upstream_url)
        joined_path = self._join_upstream_path(req_path)
        return urlunsplit((up_parts.scheme, up_parts.netloc, joined_path, req_query, ""))

    def _current_cassette_interactions_locked(self) -> list[Interaction]:
        if self.mode is TapeMode.NEW_EPISODES:
            return [*self._initial_interactions, *self._recorded]
        if self._is_recording:
            return list(self._recorded)
        return list(self._initial_interactions)

    def save_cassette(self, target_path: Path | None = None) -> tuple[Path, int]:
        """Atomically persist current session interactions to `target_path` or active cassette."""
        with self._save_lock:
            with self._lock:
                dest = target_path if target_path is not None else self._active_cassette_path
                if dest is None:
                    raise ValueError("No active cassette file path configured to save.")
                interactions = self._current_cassette_interactions_locked()
                meta = {**self._existing_metadata, "llm_tape_version": __version__}
                if target_path is None:
                    self._dirty = False
            Cassette(path=dest, interactions=interactions, metadata=meta).save()
            return dest, len(interactions)

    def _record_live_interaction(
        self,
        *,
        method: str,
        target_url: str,
        req_headers: Iterable[tuple[str, str]],
        req_body_bytes: bytes,
        status_code: int,
        resp_headers: Iterable[tuple[str, str]],
        resp_bytes: bytes | None,
        latency_ms: float,
        stream_chunks: list[str] | None = None,
    ) -> Interaction:
        clean_url = scrub_url(
            target_url,
            self.extra_sensitive_query_params,
            redact_url_patterns=self._compiled_url_patterns,
        )
        clean_req_headers = scrub_headers(req_headers, compiled=self._compiled_sensitive_headers)
        decoded_req_body, req_enc = decode_body_with_encoding(req_body_bytes)
        if self._has_body_redaction and req_enc not in {"base64", "empty"}:
            decoded_req_body = redact_body(
                decoded_req_body,
                redact_fields=self.redact_fields,
                redact_patterns=self._compiled_redact_patterns,
            )
        stored_match_on = (
            DEFAULT_MATCH_ON if self.match_on == ("method", "path", "body") else self.match_on
        )
        fp = fingerprint_request(
            method,
            clean_url,
            decoded_req_body,
            headers=clean_req_headers,
            ignore_fields=self.ignore_fields,
            ignore_query_params=self.ignore_query_params,
            ignore_json_array_order=self.ignore_json_array_order,
            match_on=stored_match_on,
            match_headers=self.match_headers,
        )
        recorded_req = RecordedRequest(
            method=method.upper(),
            url=clean_url,
            headers=clean_req_headers,
            body=decoded_req_body,
            fingerprint=fp,
            body_encoding=req_enc,
        )

        resp_headers_list = list(resp_headers)
        clean_resp_headers = scrub_headers(
            resp_headers_list, compiled=self._compiled_sensitive_headers
        )
        content_type = ""
        for k, v in resp_headers_list:
            if k.lower() == "content-type":
                content_type = v
                break

        if stream_chunks is not None:
            chunks = (
                split_sse_chunks(stream_chunks[0])
                if len(stream_chunks) == 1
                else list(stream_chunks)
            )
            chunks_tuple = tuple(redact_stream_chunks(chunks, self._compiled_redact_patterns))
            recorded_resp = RecordedResponse(
                status_code=status_code,
                headers=clean_resp_headers,
                body=None,
                body_encoding="stream",
                stream_chunks=chunks_tuple,
            )
            llm_summary = parse_llm_interaction(
                url=clean_url,
                request_body=decoded_req_body,
                response_body=None,
                custom_pricing=self.custom_pricing,
                stream_chunks=chunks_tuple,
                content_type=content_type,
            )
        else:
            decoded_resp_body, resp_enc = decode_body_with_encoding(resp_bytes or b"")
            if self._has_body_redaction and resp_enc not in {"base64", "empty"}:
                decoded_resp_body = redact_body(
                    decoded_resp_body,
                    redact_fields=self.redact_fields,
                    redact_patterns=self._compiled_redact_patterns,
                )
            recorded_resp = RecordedResponse(
                status_code=status_code,
                headers=clean_resp_headers,
                body=decoded_resp_body,
                body_encoding=resp_enc,
            )
            llm_summary = parse_llm_interaction(
                url=clean_url,
                request_body=decoded_req_body,
                response_body=decoded_resp_body,
                custom_pricing=self.custom_pricing,
                content_type=content_type,
            )

        inter = Interaction(
            request=recorded_req,
            response=recorded_resp,
            latency_ms=latency_ms,
            llm=llm_summary,
        )
        should_autosave = False
        with self._lock:
            self._recorded.append(inter)
            self._observed_interactions.append(inter)
            self._dirty = True
            should_autosave = self.auto_save and self._active_cassette_path is not None
        if should_autosave:
            self.save_cassette()
        return inter

    def _resolve_safe_cassette_path(self, raw_rel: Any) -> Path | None:
        if not isinstance(raw_rel, str):
            return None
        cleaned = raw_rel.strip()
        if not cleaned or "\x00" in cleaned or cleaned.startswith(("/", "\\")):
            return None
        rel_path = Path(cleaned)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            return None
        base_dir = self.cassette_dir.resolve()
        candidate = (base_dir / rel_path).resolve()
        if candidate == base_dir or not candidate.is_relative_to(base_dir):
            return None
        if not is_cassette_file(candidate):
            return None
        if candidate.exists() and candidate.is_dir():
            return None
        return candidate

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "mode": self.mode.value,
                "loaded_interactions": self._loaded_count,
                "active_cassette": (
                    str(self._active_cassette_path)
                    if self._active_cassette_path is not None
                    else None
                ),
            }

    def stats(self) -> dict[str, Any]:
        """Return live server replay/record counters and LLM usage summary."""
        with self._lock:
            return {
                "loaded_interactions": self._loaded_count,
                "served_requests": self._served_count,
                "unmatched_requests": self._unmatched_count,
                "recorded_interactions": len(self._recorded),
                "total_tokens": sum(i.llm.usage.total_tokens for i in self._observed_interactions),
                "estimated_cost_usd": round(
                    sum(
                        i.llm.estimated_cost_usd
                        for i in self._observed_interactions
                        if math.isfinite(i.llm.estimated_cost_usd)
                    ),
                    8,
                ),
                "tool_calls": [
                    tc.name for i in self._observed_interactions for tc in i.llm.tool_calls
                ],
            }

    def reset(self) -> dict[str, Any]:
        """Reset replay queues, counters, and observed interactions."""
        reload_path: Path | None = None
        with self._lock:
            if (
                self.mode in {TapeMode.ONCE, TapeMode.NEW_EPISODES}
                and self._active_cassette_path is not None
                and self._active_cassette_path.is_file()
            ):
                reload_path = self._active_cassette_path

        reloaded_cassette = Cassette.load(reload_path) if reload_path is not None else None

        with self._lock:
            if reloaded_cassette is not None:
                self._initial_interactions = list(reloaded_cassette.interactions)
                self._existing_metadata = dict(reloaded_cassette.metadata)
                self._is_replaying = True
                self._is_recording = self.mode is TapeMode.NEW_EPISODES
            elif self.mode is TapeMode.RECORD:
                self._initial_interactions = []
                self._is_replaying = False
                self._is_recording = True
            self._recorded.clear()
            self._observed_interactions.clear()
            self._consumed_indices.clear()
            self._served_count = 0
            self._unmatched_count = 0
            self._dirty = False
            self._rebuild_queues_locked()
            return {"status": "reset", "loaded_interactions": self._loaded_count}

    def interactions_summary(self) -> dict[str, Any]:
        with self._lock:
            snapshot = list(self._observed_interactions)
        items = [
            {
                "method": inter.request.method,
                "url": inter.request.url,
                "status_code": inter.response.status_code,
                "latency_ms": round(inter.latency_ms, 2)
                if math.isfinite(inter.latency_ms)
                else 0.0,
                "provider": inter.llm.provider,
                "model": inter.llm.model,
                "usage": inter.llm.usage.to_dict(),
                "estimated_cost_usd": round(inter.llm.estimated_cost_usd, 8)
                if math.isfinite(inter.llm.estimated_cost_usd)
                else 0.0,
                "tool_calls": [tc.to_dict() for tc in inter.llm.tool_calls],
                "text": inter.llm.text,
                "thinking": inter.llm.thinking,
                "is_stream": inter.llm.is_stream,
            }
            for inter in snapshot
        ]
        return {"interactions": items}

    def evaluate_assertions(self, spec: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        unknown = [k for k in spec if k not in _VALID_ASSERT_KEYS]
        if unknown:
            return (
                400,
                {
                    "ok": False,
                    "error": "invalid_assertion_spec",
                    "failures": [f"Unknown assertion key(s): {sorted(unknown)}"],
                },
            )

        # Validate assertion parameter types & finite bounds up front
        if "call_count" in spec:
            v = spec["call_count"]
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                return (
                    400,
                    {"ok": False, "failures": ["call_count must be an integer >= 0."]},
                )
        if "max_tokens" in spec:
            v = spec["max_tokens"]
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                return (
                    400,
                    {"ok": False, "failures": ["max_tokens must be an integer >= 0."]},
                )
        if "max_cost_usd" in spec:
            v = spec["max_cost_usd"]
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v < 0.0
            ):
                return (
                    400,
                    {"ok": False, "failures": ["max_cost_usd must be a finite number >= 0.0."]},
                )
        if "max_latency_ms" in spec:
            v = spec["max_latency_ms"]
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v < 0.0
            ):
                return (
                    400,
                    {
                        "ok": False,
                        "failures": ["max_latency_ms must be a finite number >= 0.0."],
                    },
                )
        for bool_key in ("no_errors", "no_unused_interactions"):
            if bool_key in spec and not isinstance(spec[bool_key], bool):
                return (
                    400,
                    {"ok": False, "failures": [f"{bool_key} must be a boolean."]},
                )
        for str_or_list_key in ("tool_called", "no_tool_called", "text_contains"):
            if str_or_list_key in spec:
                v = spec[str_or_list_key]
                if not isinstance(v, str) and not (
                    isinstance(v, list) and all(isinstance(x, str) for x in v)
                ):
                    return (
                        400,
                        {
                            "ok": False,
                            "failures": [f"{str_or_list_key} must be a string or list of strings."],
                        },
                    )
        if "tool_sequence" in spec:
            v = spec["tool_sequence"]
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                return (
                    400,
                    {"ok": False, "failures": ["tool_sequence must be a list of strings."]},
                )

        with self._lock:
            observed = list(self._observed_interactions)
            total_initial = len(self._initial_interactions)
            consumed_count = len(self._consumed_indices)
            is_replaying = self._is_replaying

        failures: list[str] = []
        observed_tools = [tc.name for i in observed for tc in i.llm.tool_calls]

        if "call_count" in spec:
            expected_calls = int(spec["call_count"])
            if len(observed) != expected_calls:
                failures.append(
                    f"call_count: expected {expected_calls} call(s), observed {len(observed)}."
                )

        if "max_tokens" in spec:
            max_tok = int(spec["max_tokens"])
            actual_tok = sum(i.llm.usage.total_tokens for i in observed)
            if actual_tok > max_tok:
                failures.append(f"max_tokens: used {actual_tok} tokens (limit: {max_tok}).")

        if "max_cost_usd" in spec:
            max_usd = float(spec["max_cost_usd"])
            if any(not math.isfinite(i.llm.estimated_cost_usd) for i in observed):
                failures.append("max_cost_usd: observed non-finite interaction cost.")
            else:
                actual_cost = sum(i.llm.estimated_cost_usd for i in observed)
                if not math.isfinite(actual_cost) or round(actual_cost, 8) > max_usd:
                    failures.append(
                        f"max_cost_usd: estimated ${actual_cost:.6f} exceeded limit ${max_usd:.6f}."
                    )

        if "max_latency_ms" in spec:
            max_ms = float(spec["max_latency_ms"])
            if any(not math.isfinite(i.latency_ms) for i in observed):
                failures.append("max_latency_ms: observed non-finite interaction latency.")
            else:
                slow = [i.latency_ms for i in observed if i.latency_ms > max_ms]
                if slow:
                    failures.append(
                        f"max_latency_ms: slowest call took {max(slow):.2f}ms "
                        f"(limit: {max_ms:.2f}ms)."
                    )

        if spec.get("no_errors") is True:
            failed_http = [
                f"{i.request.method} {i.request.url} -> {i.response.status_code}"
                for i in observed
                if i.response.status_code >= 400
            ]
            if failed_http:
                failures.append(f"no_errors: observed HTTP error(s): {failed_http}.")

        if "tool_called" in spec:
            raw_tc = spec["tool_called"]
            req_tools = [raw_tc] if isinstance(raw_tc, str) else list(raw_tc)
            for tname in req_tools:
                if tname not in observed_tools:
                    failures.append(
                        f"tool_called: expected tool {tname!r} to be called, "
                        f"observed {observed_tools}."
                    )

        if "no_tool_called" in spec:
            raw_ntc = spec["no_tool_called"]
            forbidden = [raw_ntc] if isinstance(raw_ntc, str) else list(raw_ntc)
            for tname in forbidden:
                cnt = sum(1 for n in observed_tools if n == tname)
                if cnt > 0:
                    failures.append(
                        f"no_tool_called: expected tool {tname!r} NOT to be called, "
                        f"but it was called {cnt} time(s)."
                    )

        if "tool_sequence" in spec:
            expected_seq = list(spec["tool_sequence"])
            if observed_tools != expected_seq:
                failures.append(f"tool_sequence: expected {expected_seq}, got {observed_tools}.")

        if "text_contains" in spec:
            raw_txt = spec["text_contains"]
            needles = [raw_txt] if isinstance(raw_txt, str) else list(raw_txt)
            texts = [i.llm.text for i in observed if i.llm.text]
            for needle in needles:
                if not any(needle in t for t in texts):
                    failures.append(
                        f"text_contains: expected response text containing {needle!r}, "
                        f"observed {texts!r}."
                    )

        if spec.get("no_unused_interactions") is True and is_replaying:
            remaining = max(0, total_initial - consumed_count)
            if remaining > 0:
                failures.append(
                    f"no_unused_interactions: {remaining} recorded interaction(s) unused."
                )

        if failures:
            return 400, {"ok": False, "failures": failures}
        return 200, {"ok": True, "failures": []}

    def _make_handler_cls(self) -> type[http.server.BaseHTTPRequestHandler]:
        server_ref = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def setup(self) -> None:
                self.request.settimeout(server_ref.request_timeout)
                super().setup()

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

            def _send_cors_headers(self) -> None:
                if server_ref.enable_cors:
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Methods", _CORS_ALLOW_METHODS)
                    self.send_header("Access-Control-Allow-Headers", "*")

            def _write_filtered_headers(self, headers_items: Iterable[tuple[str, str]]) -> None:
                for k, v in headers_items:
                    low_k = k.lower()
                    if low_k in _STRIP_HEADERS:
                        continue
                    if server_ref.enable_cors and low_k.startswith("access-control-"):
                        continue
                    self.send_header(k, v)

            def _send_json(
                self,
                status_code: int,
                payload: dict[str, Any],
                *,
                close_conn: bool = False,
            ) -> None:
                data = json.dumps(payload).encode("utf-8")
                if close_conn:
                    self.close_connection = True
                try:
                    self.send_response(status_code)
                    self._send_cors_headers()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    if close_conn:
                        self.send_header("Connection", "close")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

            def _read_exact(self, n: int) -> bytes | None:
                chunks: list[bytes] = []
                remaining = n
                while remaining > 0:
                    piece = self.rfile.read(min(remaining, 65536))
                    if not piece:
                        return None
                    chunks.append(piece)
                    remaining -= len(piece)
                return b"".join(chunks)

            def _read_chunked_body(self) -> bytes | None:
                out: list[bytes] = []
                total = 0
                while True:
                    line = self.rfile.readline(_MAX_CHUNK_LINE_BYTES + 1)
                    if not line or len(line) > _MAX_CHUNK_LINE_BYTES or not line.endswith(b"\r\n"):
                        self._send_json(400, {"error": "invalid_chunked_encoding"}, close_conn=True)
                        return None
                    size_token = line[:-2].split(b";", 1)[0].strip()
                    if not size_token or not all(b in _HEX_DIGITS for b in size_token):
                        self._send_json(400, {"error": "invalid_chunked_encoding"}, close_conn=True)
                        return None
                    chunk_size = int(size_token, 16)
                    if chunk_size == 0:
                        for _ in range(_MAX_TRAILER_LINES + 1):
                            trailer = self.rfile.readline(_MAX_CHUNK_LINE_BYTES + 1)
                            if (
                                not trailer
                                or len(trailer) > _MAX_CHUNK_LINE_BYTES
                                or not trailer.endswith(b"\r\n")
                            ):
                                self._send_json(
                                    400, {"error": "invalid_chunked_encoding"}, close_conn=True
                                )
                                return None
                            if trailer == b"\r\n":
                                return b"".join(out)
                        self._send_json(400, {"error": "invalid_chunked_encoding"}, close_conn=True)
                        return None

                    if total + chunk_size > server_ref.max_request_bytes:
                        self._send_json(413, {"error": "payload_too_large"}, close_conn=True)
                        return None

                    chunk_data = self._read_exact(chunk_size)
                    if chunk_data is None:
                        self._send_json(400, {"error": "invalid_chunked_encoding"}, close_conn=True)
                        return None
                    crlf = self._read_exact(2)
                    if crlf != b"\r\n":
                        self._send_json(400, {"error": "invalid_chunked_encoding"}, close_conn=True)
                        return None
                    out.append(chunk_data)
                    total += chunk_size

            def _read_request_body(self) -> bytes | None:
                te_raw = self.headers.get("Transfer-Encoding")
                if te_raw is not None:
                    encodings = [t.strip().lower() for t in te_raw.split(",") if t.strip()]
                    if "chunked" in encodings:
                        return self._read_chunked_body()
                    self._send_json(
                        400, {"error": "unsupported_transfer_encoding"}, close_conn=True
                    )
                    return None

                raw_len = self.headers.get("Content-Length", "0") or "0"
                try:
                    length = int(raw_len)
                except ValueError:
                    self._send_json(400, {"error": "invalid_content_length"}, close_conn=True)
                    return None
                if length < 0:
                    self._send_json(400, {"error": "invalid_content_length"}, close_conn=True)
                    return None
                if length > server_ref.max_request_bytes:
                    self._send_json(413, {"error": "payload_too_large"}, close_conn=True)
                    return None
                return self.rfile.read(length) if length > 0 else b""

            def _handle_control_plane(self, method: str, req_path: str) -> bool:
                if not req_path.startswith("/__tape__/"):
                    return False

                if method in {"GET", "HEAD"} and req_path == "/__tape__/health":
                    self._send_json(200, server_ref.health_payload())
                    return True
                if method in {"GET", "HEAD"} and req_path == "/__tape__/stats":
                    self._send_json(200, server_ref.stats())
                    return True
                if method in {"GET", "HEAD"} and req_path == "/__tape__/interactions":
                    self._send_json(200, server_ref.interactions_summary())
                    return True

                if method == "POST" and req_path == "/__tape__/reset":
                    body = self._read_request_body()
                    if body is None:
                        return True
                    self._send_json(200, server_ref.reset())
                    return True

                if method == "POST" and req_path == "/__tape__/load":
                    body = self._read_request_body()
                    if body is None:
                        return True
                    payload = _parse_json_object_strict(body)
                    if payload is None:
                        self._send_json(400, {"error": "invalid_json"})
                        return True
                    candidate = server_ref._resolve_safe_cassette_path(payload.get("cassette"))
                    if candidate is None:
                        self._send_json(400, {"error": "invalid_cassette_path"})
                        return True
                    raw_mode = payload.get("mode")
                    if raw_mode is not None:
                        if not isinstance(raw_mode, str):
                            self._send_json(400, {"error": "invalid_mode"})
                            return True
                        try:
                            target_mode = TapeMode.coerce(raw_mode)
                        except ValueError:
                            self._send_json(400, {"error": "invalid_mode"})
                            return True
                        if target_mode is TapeMode.NONE:
                            self._send_json(400, {"error": "invalid_mode"})
                            return True
                    else:
                        target_mode = server_ref.mode

                    with server_ref._lock:
                        should_flush = (
                            server_ref._dirty and server_ref._active_cassette_path is not None
                        )
                    if should_flush:
                        server_ref.save_cassette()

                    try:
                        server_ref._configure_source(candidate, target_mode)
                    except FileNotFoundError:
                        self._send_json(
                            404,
                            {"error": "cassette_not_found", "cassette": payload.get("cassette")},
                        )
                        return True
                    except CorruptCassetteError as exc:
                        self._send_json(400, {"error": "corrupt_cassette", "detail": str(exc)})
                        return True

                    rel_str = str(candidate.relative_to(server_ref.cassette_dir.resolve()))
                    self._send_json(
                        200,
                        {
                            "status": "loaded",
                            "cassette": rel_str,
                            "mode": server_ref.mode.value,
                            "loaded_interactions": server_ref._loaded_count,
                        },
                    )
                    return True

                if method == "POST" and req_path == "/__tape__/save":
                    body = self._read_request_body()
                    if body is None:
                        return True
                    target_path: Path | None = None
                    if body.strip():
                        payload = _parse_json_object_strict(body)
                        if payload is None:
                            self._send_json(400, {"error": "invalid_json"})
                            return True
                        if payload.get("cassette") is not None:
                            target_path = server_ref._resolve_safe_cassette_path(
                                payload.get("cassette")
                            )
                            if target_path is None:
                                self._send_json(400, {"error": "invalid_cassette_path"})
                                return True
                    try:
                        saved_path, count = server_ref.save_cassette(target_path)
                    except ValueError as exc:
                        self._send_json(400, {"error": "no_active_cassette", "detail": str(exc)})
                        return True
                    self._send_json(
                        200,
                        {"status": "saved", "path": str(saved_path), "interactions": count},
                    )
                    return True

                if method == "POST" and req_path == "/__tape__/assert":
                    body = self._read_request_body()
                    if body is None:
                        return True
                    spec = _parse_json_object_strict(body) if body.strip() else {}
                    if spec is None:
                        self._send_json(
                            400,
                            {
                                "ok": False,
                                "error": "invalid_json",
                                "failures": ["Invalid JSON assertion payload."],
                            },
                        )
                        return True
                    code, result = server_ref.evaluate_assertions(spec)
                    self._send_json(code, result)
                    return True

                self._send_json(404, {"error": "unknown_control_endpoint", "path": req_path})
                return True

            def _proxy_and_record(self, method: str, body_bytes: bytes) -> None:
                if server_ref.upstream_url is None:
                    self._send_json(502, {"error": "upstream_not_configured"})
                    return

                target_url = server_ref._build_upstream_target_url(self.path)
                raw_req_headers = list(self.headers.items())
                forward_headers = [
                    (k, v) for k, v in raw_req_headers if k.lower() not in _HOP_BY_HOP_REQ_HEADERS
                ]
                client = server_ref._get_upstream_client()
                req = client.build_request(
                    method=method,
                    url=target_url,
                    headers=forward_headers,
                    content=body_bytes,
                    extensions={_HANDLED_EXT_KEY: True},
                )

                start = time.perf_counter()
                try:
                    up_resp = client.send(req, stream=True)
                except httpx.HTTPError as exc:
                    self._send_json(502, {"error": "upstream_error", "detail": str(exc)})
                    return

                try:
                    resp_headers_list = list(up_resp.headers.items())
                    resp_ct = up_resp.headers.get("content-type", "").lower()
                    is_sse = (
                        (
                            "text/event-stream" in resp_ct
                            or "application/x-ndjson" in resp_ct
                            or "application/ndjson" in resp_ct
                        )
                        and up_resp.status_code < 400
                        and self.command != "HEAD"
                    )

                    if is_sse:
                        self.close_connection = True
                        self.send_response(up_resp.status_code)
                        self._send_cors_headers()
                        self._write_filtered_headers(resp_headers_list)
                        self.send_header("Connection", "close")
                        self.end_headers()

                        utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                        text_chunks: list[str] = []
                        recorded_bytes = 0
                        oversized = False
                        for raw_chunk in up_resp.iter_bytes():
                            if not raw_chunk:
                                continue
                            recorded_bytes += len(raw_chunk)
                            if recorded_bytes <= MAX_CASSETTE_BYTES:
                                decoded_piece = utf8_decoder.decode(raw_chunk, final=False)
                                if decoded_piece:
                                    text_chunks.append(decoded_piece)
                            else:
                                oversized = True
                            try:
                                self.wfile.write(raw_chunk)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                return
                        if not oversized:
                            tail = utf8_decoder.decode(b"", final=True)
                            if tail:
                                text_chunks.append(tail)
                            latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
                            server_ref._record_live_interaction(
                                method=method,
                                target_url=target_url,
                                req_headers=raw_req_headers,
                                req_body_bytes=body_bytes,
                                status_code=up_resp.status_code,
                                resp_headers=resp_headers_list,
                                resp_bytes=None,
                                latency_ms=latency_ms,
                                stream_chunks=text_chunks,
                            )
                        return

                    buf: list[bytes] = []
                    total_up = 0
                    for piece in up_resp.iter_bytes():
                        if not piece:
                            continue
                        total_up += len(piece)
                        if total_up > MAX_CASSETTE_BYTES:
                            self._send_json(
                                502,
                                {"error": "upstream_response_too_large"},
                                close_conn=True,
                            )
                            return
                        buf.append(piece)
                    resp_bytes = b"".join(buf)
                    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
                    server_ref._record_live_interaction(
                        method=method,
                        target_url=target_url,
                        req_headers=raw_req_headers,
                        req_body_bytes=body_bytes,
                        status_code=up_resp.status_code,
                        resp_headers=resp_headers_list,
                        resp_bytes=resp_bytes,
                        latency_ms=latency_ms,
                    )
                    self.send_response(up_resp.status_code)
                    self._send_cors_headers()
                    self._write_filtered_headers(resp_headers_list)
                    self.send_header("Content-Length", str(len(resp_bytes)))
                    self.end_headers()
                    if self.command != "HEAD" and resp_bytes:
                        self.wfile.write(resp_bytes)
                finally:
                    up_resp.close()

            def _handle(self, method: str) -> None:
                req_path, _req_query = _extract_path_and_query(self.path)
                if self._handle_control_plane(method, req_path):
                    return

                body_bytes = self._read_request_body()
                if body_bytes is None:
                    return
                clean_headers = scrub_headers(
                    self.headers.items(), compiled=server_ref._compiled_sensitive_headers
                )

                with server_ref._lock:
                    is_replaying = server_ref._is_replaying
                    is_recording = server_ref._is_recording

                if is_replaying:
                    inter = server_ref._match(
                        method,
                        self.path,
                        body_bytes,
                        headers=clean_headers,
                        count_unmatched=not is_recording,
                    )
                    if inter is not None:
                        self._write_interaction(inter)
                        return

                if is_recording:
                    self._proxy_and_record(method, body_bytes)
                    return

                safe_url = server_ref._build_scrubbed_synthetic_url(self.path)
                safe_parts = urlsplit(safe_url)
                safe_path = (
                    f"{safe_parts.path}?{safe_parts.query}" if safe_parts.query else safe_parts.path
                )
                self._send_json(
                    404,
                    {"error": "unmatched_request", "method": method, "path": safe_path},
                )

            def _write_interaction(self, inter: Interaction) -> None:
                delay_s = (
                    (inter.latency_ms * server_ref.replay_latency_scale) / 1000.0
                    if server_ref.replay_latency_scale > 0.0 and inter.latency_ms > 0.0
                    else 0.0
                )
                if inter.response.stream_chunks is not None:
                    chunks = inter.response.stream_chunks
                    encoded_chunks = [c.encode("utf-8") for c in chunks]
                    total_len = sum(len(b) for b in encoded_chunks)
                    chunk_delay = (
                        delay_s / len(encoded_chunks) if (delay_s > 0 and encoded_chunks) else 0.0
                    )
                    self.send_response(inter.response.status_code)
                    self._send_cors_headers()
                    self._write_filtered_headers(inter.response.headers.items())
                    self.send_header("Content-Length", str(total_len))
                    self.end_headers()
                    if self.command == "HEAD":
                        return
                    for chunk_bytes in encoded_chunks:
                        if chunk_delay > 0.0:
                            time.sleep(chunk_delay)
                        if chunk_bytes:
                            self.wfile.write(chunk_bytes)
                            self.wfile.flush()
                    return

                if delay_s > 0.0:
                    time.sleep(delay_s)
                payload = encode_body_with_encoding(
                    inter.response.body, inter.response.body_encoding
                )
                self.send_response(inter.response.status_code)
                self._send_cors_headers()
                self._write_filtered_headers(inter.response.headers.items())
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if self.command != "HEAD" and payload:
                    self.wfile.write(payload)

            def do_GET(self) -> None:
                self._handle("GET")

            def do_HEAD(self) -> None:
                self._handle("HEAD")

            def do_OPTIONS(self) -> None:
                if server_ref.enable_cors:
                    self.send_response(204)
                    self._send_cors_headers()
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self._handle("OPTIONS")

            def do_POST(self) -> None:
                self._handle("POST")

            def do_PUT(self) -> None:
                self._handle("PUT")

            def do_PATCH(self) -> None:
                self._handle("PATCH")

            def do_DELETE(self) -> None:
                self._handle("DELETE")

        return _Handler

    @property
    def port(self) -> int:
        if self._httpd is None:
            return self.requested_port
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> TapeReplayServer:
        if self._httpd is not None:
            return self
        self._httpd = http.server.ThreadingHTTPServer(
            (self.host, self.requested_port), self._make_handler_cls()
        )
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            client = self._http_client
            self._http_client = None
            should_save = self.auto_save and self._dirty and self._active_cassette_path is not None
        if client is not None:
            client.close()
        if should_save:
            self.save_cassette()

    def __enter__(self) -> TapeReplayServer:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.stop()
