"""Zero-dependency framework integrations (ASGI, WSGI, unittest, SDKs, generic HTTP)."""

from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
import math
import os
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import httpx

from rhinotape.models import TapeError, TapeMode
from rhinotape.pricing import ModelPrice
from rhinotape.scrubbing import DEFAULT_MATCH_ON, resolve_pii_patterns, validate_match_on
from rhinotape.session import TapeSession, _active_session
from rhinotape.stores import CassetteStore
from rhinotape.transport import AsyncTapeTransport, TapeTransport

P = ParamSpec("P")
R = TypeVar("R")

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_STEM_LEN = 120
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off"})


# --- 1. Context Accessors --------------------------------------------------


def current_session() -> TapeSession | None:
    """Return the active TapeSession in the current context, or None if inactive."""
    return _active_session.get()


def require_session() -> TapeSession:
    """Return the active TapeSession in the current context or raise TapeError."""
    session = current_session()
    if session is None:
        raise TapeError("No active rhinotape session in current context.")
    return session


# --- Internal Helpers ------------------------------------------------------


def _parse_env_bool(env_name: str, raw: str | None, *, default: bool) -> bool:
    if raw is None or not raw.strip():
        return default
    cleaned = raw.strip().lower()
    if cleaned in _TRUTHY_VALUES:
        return True
    if cleaned in _FALSY_VALUES:
        return False
    raise ValueError(
        f"Invalid boolean value for {env_name}: {raw!r}. "
        "Expected one of: true, false, 1, 0, yes, no, on, off."
    )


def _parse_env_float(env_name: str, raw: str | None) -> float | None:
    if raw is None or not raw.strip():
        return None
    try:
        val = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for {env_name}: {raw!r}.") from exc
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{env_name} must be a finite number >= 0, got {raw!r}.")
    return val


def _parse_env_int(env_name: str, raw: str | None) -> int | None:
    if raw is None or not raw.strip():
        return None
    try:
        val = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for {env_name}: {raw!r}.") from exc
    if val < 0:
        raise ValueError(f"{env_name} must be >= 0, got {raw!r}.")
    return val


def _parse_env_csv(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_env_redact_pii(env_name: str, raw: str | None) -> bool | tuple[str, ...]:
    if raw is None or not raw.strip():
        return False
    cleaned = raw.strip().lower()
    if cleaned in _FALSY_VALUES:
        return False
    if cleaned in _TRUTHY_VALUES or cleaned == "all":
        return True
    parts = tuple(part.strip() for part in cleaned.split(",") if part.strip())
    try:
        resolve_pii_patterns(parts)
    except ValueError as exc:
        raise ValueError(f"Invalid {env_name}: {exc}") from exc
    return parts


def _has_path_traversal(raw: str) -> bool:
    if "\x00" in raw or "/" in raw or "\\" in raw or ".." in raw:
        return True
    return Path(raw).is_absolute()


def _sanitize_cassette_stem(raw_stem: str) -> str:
    cleaned = raw_stem.replace("\x00", "")
    sanitized = _SAFE_NAME_RE.sub("_", cleaned)
    while ".." in sanitized:
        sanitized = sanitized.replace("..", "_")
    sanitized = sanitized.strip("._") or "tape"
    needs_hash = sanitized != raw_stem or len(sanitized) > _MAX_STEM_LEN
    if len(sanitized) > _MAX_STEM_LEN:
        sanitized = sanitized[:_MAX_STEM_LEN].rstrip("._") or "tape"
    if needs_hash:
        short_hash = hashlib.sha256(raw_stem.encode("utf-8")).hexdigest()[:8]
        return f"{sanitized}_{short_hash}"
    return sanitized


def _default_route_cassette_name(method: str, path: str) -> str:
    clean_method = _SAFE_NAME_RE.sub("_", method.upper()).strip("._") or "HTTP"
    clean_path = _SAFE_NAME_RE.sub("_", path.strip("/")).strip("._") or "root"
    return f"{clean_method}__{clean_path}"


def _ensure_context_session(session: TapeSession) -> contextvars.Token[TapeSession | None] | None:
    if _active_session.get() is session:
        return None
    return _active_session.set(session)


def _reset_context_session(token: contextvars.Token[TapeSession | None] | None) -> None:
    if token is not None:
        try:
            _active_session.reset(token)
        except ValueError:
            pass


# --- 2. TapeConfig & Env Bootstrap -----------------------------------------


@dataclass(frozen=True)
class TapeConfig:
    """Reusable, immutable configuration for creating rhinotape sessions across frameworks."""

    mode: TapeMode | str = TapeMode.ONCE
    tape_dir: Path = Path("tapes")
    compress: bool = False
    ignore_localhost: bool = False
    ignore_hosts: tuple[str, ...] = ()
    ignore_fields: tuple[str, ...] = ()
    ignore_query_params: tuple[str, ...] = ()
    ignore_json_array_order: tuple[str, ...] = ()
    redact_fields: tuple[str, ...] = ()
    redact_patterns: tuple[str | re.Pattern[str], ...] = ()
    redact_pii: bool | str | tuple[str, ...] = False
    redact_url_patterns: tuple[str | re.Pattern[str], ...] = ()
    match_on: tuple[str, ...] = DEFAULT_MATCH_ON
    match_headers: tuple[str, ...] = ()
    max_interactions: int | None = None
    max_cost_usd: float | None = None
    replay_latency_scale: float = 0.0
    allow_playback_repeats: bool = False
    save_on_error: bool = False
    patch_httpx: bool = True
    patch_requests: bool = True
    patch_aiohttp: bool = True
    patch_urllib: bool = False
    store: CassetteStore | None = None
    custom_pricing: dict[str, ModelPrice] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        TapeMode.coerce(self.mode)
        validate_match_on(self.match_on)
        if self.max_interactions is not None and (
            isinstance(self.max_interactions, bool) or self.max_interactions < 0
        ):
            raise ValueError("max_interactions must be an integer >= 0.")
        if self.max_cost_usd is not None and (
            isinstance(self.max_cost_usd, bool)
            or not math.isfinite(self.max_cost_usd)
            or self.max_cost_usd < 0.0
        ):
            raise ValueError("max_cost_usd must be a finite number >= 0.0.")
        if (
            isinstance(self.replay_latency_scale, bool)
            or not math.isfinite(self.replay_latency_scale)
            or self.replay_latency_scale < 0.0
        ):
            raise ValueError("replay_latency_scale must be a finite number >= 0.0.")

    @classmethod
    def from_env(cls, prefix: str = "RHINOTAPE_") -> TapeConfig:
        """Build a TapeConfig from environment variables prefixed with `prefix`."""
        raw_mode = os.environ.get(f"{prefix}MODE", "").strip()
        mode: TapeMode = TapeMode.coerce(raw_mode) if raw_mode else TapeMode.ONCE

        raw_dir = os.environ.get(f"{prefix}DIR", "").strip()
        tape_dir = Path(raw_dir) if raw_dir else Path("tapes")

        compress = _parse_env_bool(
            f"{prefix}COMPRESS", os.environ.get(f"{prefix}COMPRESS"), default=False
        )
        ignore_localhost = _parse_env_bool(
            f"{prefix}IGNORE_LOCALHOST",
            os.environ.get(f"{prefix}IGNORE_LOCALHOST"),
            default=False,
        )
        ignore_hosts = _parse_env_csv(os.environ.get(f"{prefix}IGNORE_HOSTS"))
        ignore_fields = _parse_env_csv(os.environ.get(f"{prefix}IGNORE_FIELDS"))
        ignore_query_params = _parse_env_csv(os.environ.get(f"{prefix}IGNORE_QUERY_PARAMS"))
        ignore_json_array_order = _parse_env_csv(os.environ.get(f"{prefix}IGNORE_JSON_ARRAY_ORDER"))
        redact_fields = _parse_env_csv(os.environ.get(f"{prefix}REDACT_FIELDS"))
        redact_pii = _parse_env_redact_pii(
            f"{prefix}REDACT_PII", os.environ.get(f"{prefix}REDACT_PII")
        )

        raw_match_on = _parse_env_csv(os.environ.get(f"{prefix}MATCH_ON"))
        match_on = validate_match_on(raw_match_on) if raw_match_on else DEFAULT_MATCH_ON
        match_headers = _parse_env_csv(os.environ.get(f"{prefix}MATCH_HEADERS"))

        max_cost = _parse_env_float(f"{prefix}MAX_COST", os.environ.get(f"{prefix}MAX_COST"))
        max_interactions = _parse_env_int(
            f"{prefix}MAX_INTERACTIONS", os.environ.get(f"{prefix}MAX_INTERACTIONS")
        )
        latency_scale = _parse_env_float(
            f"{prefix}REPLAY_LATENCY_SCALE",
            os.environ.get(f"{prefix}REPLAY_LATENCY_SCALE"),
        )
        allow_repeats = _parse_env_bool(
            f"{prefix}ALLOW_PLAYBACK_REPEATS",
            os.environ.get(f"{prefix}ALLOW_PLAYBACK_REPEATS"),
            default=False,
        )
        save_on_error = _parse_env_bool(
            f"{prefix}SAVE_ON_ERROR", os.environ.get(f"{prefix}SAVE_ON_ERROR"), default=False
        )
        patch_httpx = _parse_env_bool(
            f"{prefix}PATCH_HTTPX", os.environ.get(f"{prefix}PATCH_HTTPX"), default=True
        )
        patch_requests = _parse_env_bool(
            f"{prefix}PATCH_REQUESTS", os.environ.get(f"{prefix}PATCH_REQUESTS"), default=True
        )
        patch_aiohttp = _parse_env_bool(
            f"{prefix}PATCH_AIOHTTP", os.environ.get(f"{prefix}PATCH_AIOHTTP"), default=True
        )
        patch_urllib = _parse_env_bool(
            f"{prefix}PATCH_URLLIB", os.environ.get(f"{prefix}PATCH_URLLIB"), default=False
        )

        return cls(
            mode=mode,
            tape_dir=tape_dir,
            compress=compress,
            ignore_localhost=ignore_localhost,
            ignore_hosts=ignore_hosts,
            ignore_fields=ignore_fields,
            ignore_query_params=ignore_query_params,
            ignore_json_array_order=ignore_json_array_order,
            redact_fields=redact_fields,
            redact_pii=redact_pii,
            match_on=match_on,
            match_headers=match_headers,
            max_interactions=max_interactions,
            max_cost_usd=max_cost,
            replay_latency_scale=latency_scale if latency_scale is not None else 0.0,
            allow_playback_repeats=allow_repeats,
            save_on_error=save_on_error,
            patch_httpx=patch_httpx,
            patch_requests=patch_requests,
            patch_aiohttp=patch_aiohttp,
            patch_urllib=patch_urllib,
        )

    def resolve_path(self, name: str, *, compress: bool | None = None) -> Path:
        """Sanitize a logical cassette name safely inside `tape_dir` and append extension."""
        cleaned = name.strip()
        lower = cleaned.lower()
        use_compress = self.compress if compress is None else compress
        if lower.endswith(".tape.json.gz"):
            raw_stem = cleaned[: -len(".tape.json.gz")]
            ext = ".tape.json.gz"
        elif lower.endswith(".tape.json"):
            raw_stem = cleaned[: -len(".tape.json")]
            ext = ".tape.json"
        else:
            raw_stem = cleaned
            ext = ".tape.json.gz" if use_compress else ".tape.json"

        safe_stem = _sanitize_cassette_stem(raw_stem)
        base_dir = Path(self.tape_dir)
        candidate = base_dir / f"{safe_stem}{ext}"
        if not candidate.resolve().is_relative_to(base_dir.resolve()):
            raise ValueError(f"Cassette name {name!r} resolved outside tape_dir {base_dir}.")
        return candidate

    def session(self, name_or_path: str | Path, **overrides: Any) -> TapeSession:
        """Construct a TapeSession using this config plus optional keyword overrides."""
        override_copy = dict(overrides)
        compress_override = override_copy.pop("compress", None)
        if isinstance(name_or_path, Path):
            path = name_or_path
        else:
            path = self.resolve_path(
                name_or_path,
                compress=bool(compress_override) if compress_override is not None else None,
            )

        kwargs: dict[str, Any] = {
            "mode": self.mode,
            "custom_pricing": self.custom_pricing,
            "ignore_localhost": self.ignore_localhost,
            "ignore_hosts": self.ignore_hosts,
            "ignore_fields": self.ignore_fields,
            "ignore_query_params": self.ignore_query_params,
            "ignore_json_array_order": self.ignore_json_array_order,
            "redact_fields": self.redact_fields,
            "redact_patterns": self.redact_patterns,
            "redact_pii": self.redact_pii,
            "redact_url_patterns": self.redact_url_patterns,
            "match_on": self.match_on,
            "match_headers": self.match_headers,
            "metadata": dict(self.metadata) if self.metadata is not None else None,
            "store": self.store,
            "max_interactions": self.max_interactions,
            "max_cost_usd": self.max_cost_usd,
            "replay_latency_scale": self.replay_latency_scale,
            "allow_playback_repeats": self.allow_playback_repeats,
            "save_on_error": self.save_on_error,
            "patch_httpx": self.patch_httpx,
            "patch_requests": self.patch_requests,
            "patch_aiohttp": self.patch_aiohttp,
            "patch_urllib": self.patch_urllib,
        }
        kwargs.update(override_copy)
        return TapeSession(path, **kwargs)

    def decorator(
        self,
        name_or_path: str | Path,
        **overrides: Any,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Return a decorator wrapping a sync or async callable in a configured TapeSession."""

        def _decorate(fn: Callable[P, R]) -> Callable[P, R]:
            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                    async with self.session(name_or_path, **overrides):
                        return await fn(*args, **kwargs)

                return _async_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def _sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with self.session(name_or_path, **overrides):
                    return fn(*args, **kwargs)

            return _sync_wrapper

        return _decorate


def tape_from_env(
    cassette_path: str | Path | None = None,
    *,
    prefix: str = "RHINOTAPE_",
    **overrides: Any,
) -> TapeSession:
    """Create a TapeSession configured from environment variables."""
    cfg = TapeConfig.from_env(prefix=prefix)
    if cassette_path is None:
        raw_cassette = os.environ.get(f"{prefix}CASSETTE", "").strip()
        if raw_cassette:
            target: str | Path = (
                Path(raw_cassette) if Path(raw_cassette).is_absolute() else raw_cassette
            )
        else:
            target = "default"
    else:
        target = cassette_path
    return cfg.session(target, **overrides)


# --- 3. Drop-in SDK / httpx Client Factories -------------------------------


def make_httpx_client(
    session: TapeSession,
    *,
    upstream: httpx.BaseTransport | None = None,
    **client_kwargs: Any,
) -> httpx.Client:
    """Create a sync httpx.Client wired directly to `session` via TapeTransport."""
    if "transport" in client_kwargs:
        raise TypeError(
            "make_httpx_client() manages 'transport' via TapeTransport; pass 'upstream' instead."
        )
    return httpx.Client(transport=TapeTransport(session, upstream=upstream), **client_kwargs)


def make_async_httpx_client(
    session: TapeSession,
    *,
    upstream: httpx.AsyncBaseTransport | None = None,
    **client_kwargs: Any,
) -> httpx.AsyncClient:
    """Create an async httpx.AsyncClient wired directly to `session` via AsyncTapeTransport."""
    if "transport" in client_kwargs:
        raise TypeError(
            "make_async_httpx_client() manages 'transport' via AsyncTapeTransport; "
            "pass 'upstream' instead."
        )
    return httpx.AsyncClient(
        transport=AsyncTapeTransport(session, upstream=upstream), **client_kwargs
    )


# --- 4. Pure ASGI 3.0 Middleware -------------------------------------------


class TapeASGIMiddleware:
    """Zero-dependency ASGI 3.0 middleware wrapping HTTP requests in a TapeSession.

    Works with FastAPI, Starlette, Litestar, Quart, and Django ASGI.
    """

    def __init__(
        self,
        app: Any,
        *,
        config: TapeConfig | None = None,
        cassette_resolver: Callable[[dict[str, Any]], str | Path | None] | None = None,
        header_name: str = "x-tape-name",
        expose_headers: bool = False,
        skip_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.config = config if config is not None else TapeConfig()
        self.cassette_resolver = cassette_resolver
        self.header_name = header_name.lower()
        self._header_name_bytes = self.header_name.encode("latin-1")
        self.expose_headers = expose_headers
        self._skip_paths: frozenset[str] = frozenset(skip_paths)

    def _extract_header(self, scope: dict[str, Any]) -> str | None:
        raw_headers = scope.get("headers") or ()
        for key, value in raw_headers:
            if isinstance(key, bytes) and key.lower() == self._header_name_bytes:
                decoded = (
                    value.decode("latin-1") if isinstance(value, bytes) else str(value)
                ).strip()
                return decoded or None
        return None

    def _resolve_target(self, scope: dict[str, Any], path: str) -> str | Path | None:
        if self.cassette_resolver is not None:
            return self.cassette_resolver(scope)
        header_val = self._extract_header(scope)
        if header_val and not _has_path_traversal(header_val):
            return header_val
        method = str(scope.get("method", "GET") or "GET")
        return _default_route_cassette_name(method, path)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", "") or "")
        if path in self._skip_paths:
            await self.app(scope, receive, send)
            return

        target = self._resolve_target(scope, path)
        if target is None:
            await self.app(scope, receive, send)
            return

        session = self.config.session(target)
        async with session:
            ctx_token = _ensure_context_session(session)
            try:
                state = scope.get("state")
                if state is None:
                    scope["state"] = {"rhinotape": session}
                elif isinstance(state, dict):
                    state["rhinotape"] = session

                if not self.expose_headers:
                    await self.app(scope, receive, send)
                    return

                async def _send_with_headers(message: dict[str, Any]) -> None:
                    if message.get("type") == "http.response.start":
                        existing = list(message.get("headers") or ())
                        extra = [
                            (b"x-tape-calls", str(session.call_count).encode("latin-1")),
                            (b"x-tape-tokens", str(session.total_tokens).encode("latin-1")),
                            (b"x-tape-cost-usd", f"{session.total_cost_usd:.8f}".encode("latin-1")),
                        ]
                        message = {**message, "headers": [*existing, *extra]}
                    await send(message)

                await self.app(scope, receive, _send_with_headers)
            finally:
                _reset_context_session(ctx_token)


# --- 5. Pure PEP 3333 WSGI Middleware --------------------------------------


class _WSGIResponseWrapper:
    """Wrap a WSGI response iterable so the active TapeSession closes after exhaustion/close()."""

    def __init__(
        self,
        iterable: Iterable[bytes],
        finalize: Callable[[bool], None],
    ) -> None:
        self._iterable = iterable
        self._finalize = finalize
        self._closed = False
        self._errored = False

    def __iter__(self) -> Iterator[bytes]:
        try:
            yield from self._iterable
        except BaseException:
            self._errored = True
            self._close_inner()
            self._finish(exc_occurred=True)
            raise
        else:
            self._close_inner()
            self._finish(exc_occurred=False)

    def _close_inner(self) -> None:
        close_fn = getattr(self._iterable, "close", None)
        if callable(close_fn):
            close_fn()

    def _finish(self, *, exc_occurred: bool) -> None:
        if not self._closed:
            self._closed = True
            self._finalize(exc_occurred)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._close_inner()
        finally:
            self._finish(exc_occurred=self._errored)


class TapeWSGIMiddleware:
    """Zero-dependency PEP 3333 WSGI middleware wrapping requests in a TapeSession.

    Works with Flask, Django WSGI, Bottle, Pyramid, and Falcon.
    """

    def __init__(
        self,
        app: Any,
        *,
        config: TapeConfig | None = None,
        cassette_resolver: Callable[[dict[str, Any]], str | Path | None] | None = None,
        header_name: str = "HTTP_X_TAPE_NAME",
        expose_headers: bool = False,
        skip_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.config = config if config is not None else TapeConfig()
        self.cassette_resolver = cassette_resolver
        self.header_name = header_name
        self._alt_header_name = (
            header_name
            if header_name.startswith("HTTP_")
            else f"HTTP_{header_name.upper().replace('-', '_')}"
        )
        self.expose_headers = expose_headers
        self._skip_paths: frozenset[str] = frozenset(skip_paths)

    def _extract_header(self, environ: dict[str, Any]) -> str | None:
        raw = environ.get(self.header_name)
        if raw is None and self._alt_header_name != self.header_name:
            raw = environ.get(self._alt_header_name)
        if raw is None:
            return None
        cleaned = str(raw).strip()
        return cleaned or None

    def _resolve_target(self, environ: dict[str, Any], path: str) -> str | Path | None:
        if self.cassette_resolver is not None:
            return self.cassette_resolver(environ)
        header_val = self._extract_header(environ)
        if header_val and not _has_path_traversal(header_val):
            return header_val
        method = str(environ.get("REQUEST_METHOD", "GET") or "GET")
        return _default_route_cassette_name(method, path)

    def __call__(self, environ: dict[str, Any], start_response: Any) -> Iterable[bytes]:
        path = str(environ.get("PATH_INFO", "") or "")
        if path in self._skip_paths:
            result: Iterable[bytes] = self.app(environ, start_response)
            return result

        target = self._resolve_target(environ, path)
        if target is None:
            passthrough: Iterable[bytes] = self.app(environ, start_response)
            return passthrough

        session = self.config.session(target)
        session.__enter__()
        ctx_token = _ensure_context_session(session)
        environ["rhinotape.session"] = session

        def _finalize(exc_occurred: bool) -> None:
            try:
                session._finalize(exc_occurred=exc_occurred)
            finally:
                _reset_context_session(ctx_token)

        def _start_response_with_headers(
            status: str,
            response_headers: list[tuple[str, str]],
            exc_info: Any = None,
        ) -> Any:
            headers = [
                *response_headers,
                ("X-Tape-Calls", str(session.call_count)),
                ("X-Tape-Tokens", str(session.total_tokens)),
                ("X-Tape-Cost-Usd", f"{session.total_cost_usd:.8f}"),
            ]
            if exc_info is not None:
                return start_response(status, headers, exc_info)
            return start_response(status, headers)

        try:
            app_iter: Iterable[bytes] = self.app(
                environ,
                _start_response_with_headers if self.expose_headers else start_response,
            )
        except BaseException:
            _finalize(True)
            raise

        return _WSGIResponseWrapper(app_iter, _finalize)


# --- 6. stdlib unittest Mixin ----------------------------------------------


def _unittest_did_fail(test_case: Any) -> bool:
    """Safely detect whether a unittest.TestCase failed across Python 3.10-3.13."""
    outcome = getattr(test_case, "_outcome", None)
    if outcome is None:
        return False

    result = getattr(outcome, "result", None)
    if result is not None:
        for collection_name in ("failures", "errors"):
            entries = getattr(result, collection_name, ())
            for entry in entries:
                if isinstance(entry, tuple) and entry:
                    tc = entry[0]
                    if tc is test_case or getattr(tc, "test_case", None) is test_case:
                        return True
        unexpected = getattr(result, "unexpectedSuccesses", ())
        if test_case in unexpected:
            return True

    # Python 3.10 _Outcome.errors list of (test, exc_info | None)
    legacy_errors = getattr(outcome, "errors", None)
    if isinstance(legacy_errors, list):
        for entry in legacy_errors:
            if isinstance(entry, tuple) and len(entry) >= 2 and entry[1] is not None:
                return True

    return False


class TapeTestCaseMixin:
    """Mixin for unittest.TestCase / Django TestCase managing per-test TapeSession lifecycle."""

    tape_config: TapeConfig | None = None
    tape_dir: str | Path | None = None
    tape_mode: TapeMode | str | None = None
    tape_strict_unused: bool = False
    tape_kwargs: dict[str, Any] = {}
    rhinotape: TapeSession
    _rhinotape_session: TapeSession | None = None
    _rhinotape_ctx_token: contextvars.Token[TapeSession | None] | None = None

    def setUp(self) -> None:
        base_cfg = self.tape_config if self.tape_config is not None else TapeConfig()
        method_name = str(getattr(self, "_testMethodName", "test"))
        cassette_name = f"{type(self).__name__}__{method_name}"

        overrides = dict(self.tape_kwargs)
        if self.tape_mode is not None and "mode" not in overrides:
            overrides["mode"] = self.tape_mode

        if self.tape_dir is not None:
            resolved_name = base_cfg.resolve_path(cassette_name).name
            target: str | Path = Path(self.tape_dir) / resolved_name
        else:
            target = cassette_name

        session = base_cfg.session(target, **overrides)
        session.__enter__()
        ctx_token = _ensure_context_session(session)
        self.rhinotape = session
        self._rhinotape_session = session
        self._rhinotape_ctx_token = ctx_token

        try:
            super_setup = getattr(super(), "setUp", None)
            if callable(super_setup):
                super_setup()
        except BaseException:
            self._rhinotape_session = None
            self._rhinotape_ctx_token = None
            try:
                session._finalize(exc_occurred=True)
            finally:
                _reset_context_session(ctx_token)
            raise

    def tearDown(self) -> None:
        session = self._rhinotape_session
        ctx_token = self._rhinotape_ctx_token
        self._rhinotape_session = None
        self._rhinotape_ctx_token = None
        unused_err: Exception | None = None

        try:
            if session is not None:
                test_failed = _unittest_did_fail(self)
                if (
                    self.tape_strict_unused
                    and not test_failed
                    and session._is_replaying
                    and not session._is_recording
                ):
                    try:
                        session.assert_no_unused_interactions()
                    except Exception as exc:
                        unused_err = exc
                try:
                    session._finalize(exc_occurred=test_failed or unused_err is not None)
                finally:
                    _reset_context_session(ctx_token)
        finally:
            super_teardown = getattr(super(), "tearDown", None)
            if callable(super_teardown):
                super_teardown()

        if unused_err is not None:
            raise unused_err


# --- 7. Global / Process-level Activation Handle ---------------------------


class GlobalTapeHandle:
    """Lifecycle handle for an activated process/worker-level TapeSession."""

    def __init__(
        self,
        session: TapeSession,
        *,
        ctx_token: contextvars.Token[TapeSession | None] | None = None,
    ) -> None:
        self._session = session
        self._ctx_token = ctx_token
        self._closed = False

    @property
    def session(self) -> TapeSession:
        return self._session

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self, exc_occurred: bool = False) -> None:
        """Finalize and save (if recording) the underlying TapeSession idempotently."""
        if self._closed:
            return
        self._closed = True
        try:
            self._session._finalize(exc_occurred=exc_occurred)
        finally:
            _reset_context_session(self._ctx_token)
            self._ctx_token = None

    def __enter__(self) -> GlobalTapeHandle:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc, tb
        self.close(exc_occurred=exc_type is not None)


def activate_tape(
    cassette_path: str | Path,
    *,
    config: TapeConfig | None = None,
    **overrides: Any,
) -> GlobalTapeHandle:
    """Activate a TapeSession immediately and return a handle to close it on shutdown."""
    cfg = config if config is not None else TapeConfig()
    session = cfg.session(cassette_path, **overrides)
    session.__enter__()
    ctx_token = _ensure_context_session(session)
    return GlobalTapeHandle(session, ctx_token=ctx_token)


def activate_from_env(
    *,
    prefix: str = "RHINOTAPE_",
    **overrides: Any,
) -> GlobalTapeHandle | None:
    """Activate rhinotape from env vars only when enabled or cassette is configured.

    Returns None when neither `{prefix}ENABLED` is truthy nor `{prefix}CASSETTE` is set,
    making it safe to call unconditionally at application startup.
    """
    raw_enabled = os.environ.get(f"{prefix}ENABLED")
    raw_cassette = os.environ.get(f"{prefix}CASSETTE", "").strip()

    if raw_enabled is not None and raw_enabled.strip():
        if not _parse_env_bool(f"{prefix}ENABLED", raw_enabled, default=False):
            return None
    elif not raw_cassette:
        return None

    session = tape_from_env(prefix=prefix, **overrides)
    session.__enter__()
    ctx_token = _ensure_context_session(session)
    return GlobalTapeHandle(session, ctx_token=ctx_token)


# --- 8. Generic Adapter Bridge ---------------------------------------------


@dataclass(frozen=True)
class GenericHTTPRequest:
    """Library-agnostic HTTP request value object for integrating custom HTTP clients."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


@dataclass(frozen=True)
class GenericHTTPResponse:
    """Library-agnostic HTTP response value object for integrating custom HTTP clients."""

    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


def handle_generic_request(
    session: TapeSession,
    request: GenericHTTPRequest,
    sender: Callable[[GenericHTTPRequest], GenericHTTPResponse],
) -> GenericHTTPResponse:
    """Route a sync custom HTTP client request through TapeSession record/replay/scrubbing."""
    httpx_req = httpx.Request(
        method=request.method.upper(),
        url=request.url,
        headers=request.headers,
        content=request.body,
    )

    def _bridge_sender(req: httpx.Request) -> httpx.Response:
        generic_resp = sender(request)
        return httpx.Response(
            status_code=generic_resp.status_code,
            headers=generic_resp.headers,
            content=generic_resp.body,
            request=req,
        )

    httpx_resp = session.handle_sync(httpx_req, _bridge_sender)
    resp_bytes = httpx_resp.content if httpx_resp.is_stream_consumed else httpx_resp.read()
    return GenericHTTPResponse(
        status_code=httpx_resp.status_code,
        headers=dict(httpx_resp.headers.items()),
        body=resp_bytes,
    )


async def handle_generic_request_async(
    session: TapeSession,
    request: GenericHTTPRequest,
    sender: Callable[[GenericHTTPRequest], Awaitable[GenericHTTPResponse]],
) -> GenericHTTPResponse:
    """Route an async custom HTTP client request through TapeSession record/replay/scrubbing."""
    httpx_req = httpx.Request(
        method=request.method.upper(),
        url=request.url,
        headers=request.headers,
        content=request.body,
    )

    async def _bridge_sender(req: httpx.Request) -> httpx.Response:
        generic_resp = await sender(request)
        return httpx.Response(
            status_code=generic_resp.status_code,
            headers=generic_resp.headers,
            content=generic_resp.body,
            request=req,
        )

    httpx_resp = await session.handle_async(httpx_req, _bridge_sender)
    resp_bytes = httpx_resp.content if httpx_resp.is_stream_consumed else await httpx_resp.aread()
    return GenericHTTPResponse(
        status_code=httpx_resp.status_code,
        headers=dict(httpx_resp.headers.items()),
        body=resp_bytes,
    )


__all__ = [
    "GenericHTTPRequest",
    "GenericHTTPResponse",
    "GlobalTapeHandle",
    "TapeASGIMiddleware",
    "TapeConfig",
    "TapeTestCaseMixin",
    "TapeWSGIMiddleware",
    "activate_from_env",
    "activate_tape",
    "current_session",
    "handle_generic_request",
    "handle_generic_request_async",
    "make_async_httpx_client",
    "make_httpx_client",
    "require_session",
    "tape_from_env",
]
