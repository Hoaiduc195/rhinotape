"""TapeSession: orchestrates recording, replaying, patching, and LLM assertions."""

from __future__ import annotations

import functools
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

import httpx

from llm_tape.cassette import Cassette
from llm_tape.models import (
    Interaction,
    RecordedRequest,
    RecordedResponse,
    TapeAssertionError,
    TapeMode,
    ToolCall,
    UnmatchedRequestError,
    Usage,
)
from llm_tape.parsers import parse_llm_interaction
from llm_tape.pricing import ModelPrice
from llm_tape.scrubbing import (
    decode_body,
    encode_body,
    fingerprint_request,
    scrub_headers,
    scrub_url,
)

F = TypeVar("F", bound=Callable[..., Any])

_STRIP_REPLAY_HEADERS = frozenset(
    {"content-encoding", "transfer-encoding", "content-length"}
)


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
        patch_httpx: bool = True,
    ) -> None:
        self.cassette_path = Path(cassette_path)
        self.mode = TapeMode.coerce(mode)
        self.custom_pricing = custom_pricing
        self.extra_sensitive_headers = tuple(extra_sensitive_headers)
        self.extra_sensitive_query_params = tuple(extra_sensitive_query_params)
        self.patch_httpx = patch_httpx

        self._interactions: list[Interaction] = []
        self._recorded: list[Interaction] = []
        self._replay_queues: dict[str, deque[Interaction]] = defaultdict(deque)
        self._is_replaying = False
        self._is_recording = False
        self._orig_sync_send: Any = None
        self._orig_async_send: Any = None

    # --- Lifecycle ---------------------------------------------------------

    def _prepare(self) -> None:
        self._interactions.clear()
        self._recorded.clear()
        self._replay_queues.clear()

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
        for interaction in cassette.interactions:
            self._replay_queues[interaction.request.fingerprint].append(interaction)

    def _install_httpx_patch(self) -> None:
        if not self.patch_httpx or self.mode is TapeMode.NONE:
            return
        session = self
        orig_sync = httpx.Client.send
        orig_async = httpx.AsyncClient.send
        self._orig_sync_send = orig_sync
        self._orig_async_send = orig_async

        def patched_sync_send(
            client: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any
        ) -> httpx.Response:
            return session.handle_sync(
                request, lambda req: orig_sync(client, req, *args, **kwargs)
            )

        async def patched_async_send(
            client: httpx.AsyncClient, request: httpx.Request, *args: Any, **kwargs: Any
        ) -> httpx.Response:
            return await session.handle_async(
                request, lambda req: orig_async(client, req, *args, **kwargs)
            )

        httpx.Client.send = patched_sync_send  # type: ignore[method-assign]
        httpx.AsyncClient.send = patched_async_send  # type: ignore[method-assign]

    def _uninstall_httpx_patch(self) -> None:
        if self._orig_sync_send is not None:
            httpx.Client.send = self._orig_sync_send  # type: ignore[method-assign]
            self._orig_sync_send = None
        if self._orig_async_send is not None:
            httpx.AsyncClient.send = self._orig_async_send  # type: ignore[method-assign]
            self._orig_async_send = None

    def _finalize(self) -> None:
        self._uninstall_httpx_patch()
        if self._is_recording and self._recorded:
            cassette = Cassette(path=self.cassette_path, interactions=list(self._recorded))
            cassette.save()

    def __enter__(self) -> TapeSession:
        self._prepare()
        self._install_httpx_patch()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._finalize()

    async def __aenter__(self) -> TapeSession:
        self._prepare()
        self._install_httpx_patch()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._finalize()

    # --- Request / Response handling ---------------------------------------

    def _build_recorded_request(self, request: httpx.Request, body_bytes: bytes) -> RecordedRequest:
        clean_url = scrub_url(str(request.url), self.extra_sensitive_query_params)
        clean_headers = scrub_headers(request.headers.items(), self.extra_sensitive_headers)
        decoded_body = decode_body(body_bytes)
        fp = fingerprint_request(request.method, clean_url, decoded_body)
        return RecordedRequest(
            method=request.method.upper(),
            url=clean_url,
            headers=clean_headers,
            body=decoded_body,
            fingerprint=fp,
        )

    def _replay_response(
        self, request: httpx.Request, recorded_req: RecordedRequest
    ) -> httpx.Response:
        queue = self._replay_queues.get(recorded_req.fingerprint)
        if not queue:
            raise UnmatchedRequestError(
                f"No recorded interaction matches {recorded_req.method} {recorded_req.url} "
                f"(fingerprint={recorded_req.fingerprint}) in cassette {self.cassette_path}"
            )
        interaction = queue.popleft()
        self._interactions.append(interaction)
        replay_headers = {
            k: v
            for k, v in interaction.response.headers.items()
            if k.lower() not in _STRIP_REPLAY_HEADERS
        }
        return httpx.Response(
            status_code=interaction.response.status_code,
            headers=replay_headers,
            content=encode_body(interaction.response.body),
            request=request,
        )

    def _record_interaction(
        self,
        recorded_req: RecordedRequest,
        response: httpx.Response,
        response_bytes: bytes,
        latency_ms: float,
    ) -> Interaction:
        clean_resp_headers = scrub_headers(
            response.headers.items(), self.extra_sensitive_headers
        )
        decoded_resp_body = decode_body(response_bytes)
        recorded_resp = RecordedResponse(
            status_code=response.status_code,
            headers=clean_resp_headers,
            body=decoded_resp_body,
        )
        llm_summary = parse_llm_interaction(
            url=recorded_req.url,
            request_body=recorded_req.body,
            response_body=decoded_resp_body,
            custom_pricing=self.custom_pricing,
        )
        interaction = Interaction(
            request=recorded_req,
            response=recorded_resp,
            latency_ms=latency_ms,
            llm=llm_summary,
        )
        self._interactions.append(interaction)
        self._recorded.append(interaction)
        return interaction

    def handle_sync(
        self,
        request: httpx.Request,
        sender: Callable[[httpx.Request], httpx.Response],
    ) -> httpx.Response:
        if self.mode is TapeMode.NONE:
            return sender(request)

        body_bytes = request.read()
        recorded_req = self._build_recorded_request(request, body_bytes)

        if self._is_replaying:
            return self._replay_response(request, recorded_req)

        start = time.perf_counter()
        response = sender(request)
        response_bytes = response.read()
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._record_interaction(recorded_req, response, response_bytes, latency_ms)
        return response

    async def handle_async(
        self,
        request: httpx.Request,
        sender: Callable[[httpx.Request], Any],
    ) -> httpx.Response:
        if self.mode is TapeMode.NONE:
            return await sender(request)

        body_bytes = await request.aread()
        recorded_req = self._build_recorded_request(request, body_bytes)

        if self._is_replaying:
            return self._replay_response(request, recorded_req)

        start = time.perf_counter()
        response = await sender(request)
        response_bytes = await response.aread()
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._record_interaction(recorded_req, response, response_bytes, latency_ms)
        return response

    # --- Inspection properties ---------------------------------------------

    @property
    def interactions(self) -> list[Interaction]:
        return list(self._interactions)

    @property
    def call_count(self) -> int:
        return len(self._interactions)

    @property
    def total_usage(self) -> Usage:
        return Usage(
            input_tokens=sum(i.llm.usage.input_tokens for i in self._interactions),
            output_tokens=sum(i.llm.usage.output_tokens for i in self._interactions),
        )

    @property
    def total_tokens(self) -> int:
        return self.total_usage.total_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(i.llm.estimated_cost_usd for i in self._interactions)

    @property
    def tool_calls(self) -> list[ToolCall]:
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
        if self.call_count != expected:
            raise TapeAssertionError(
                f"Expected {expected} LLM/HTTP call(s), but observed {self.call_count}."
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
        count = sum(1 for tc in self.tool_calls if tc.name == name)
        if count < min_times:
            raise TapeAssertionError(
                f"Expected tool {name!r} to be called at least {min_times} time(s), "
                f"but it was called {count} time(s). Observed tools: {self.tool_names}"
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
    patch_httpx: bool = True,
) -> TapeSession:
    """Create a TapeSession context manager for recording/replaying LLM calls."""
    return TapeSession(
        cassette_path=cassette_path,
        mode=mode,
        custom_pricing=custom_pricing,
        extra_sensitive_headers=extra_sensitive_headers,
        extra_sensitive_query_params=extra_sensitive_query_params,
        patch_httpx=patch_httpx,
    )


def use_tape(
    cassette_path: str | Path,
    *,
    mode: TapeMode | str = TapeMode.ONCE,
) -> Callable[[F], F]:
    """Decorator that wraps a sync function inside a TapeSession."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TapeSession(cassette_path, mode=mode):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
