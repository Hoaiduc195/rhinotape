"""Explicit httpx BaseTransport / AsyncBaseTransport wrappers for llm-tape."""

from __future__ import annotations

import httpx

from llm_tape.session import TapeSession


class TapeTransport(httpx.BaseTransport):
    """Sync httpx transport that records/replays through a TapeSession without global patching."""

    def __init__(
        self,
        session: TapeSession,
        upstream: httpx.BaseTransport | None = None,
        *,
        owns_upstream: bool | None = None,
    ) -> None:
        self._session = session
        self._owns_upstream = (upstream is None) if owns_upstream is None else owns_upstream
        self._upstream = upstream or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._session.handle_sync(request, self._upstream.handle_request)

    def close(self) -> None:
        if self._owns_upstream:
            self._upstream.close()


class AsyncTapeTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that records/replays through a TapeSession without global patching."""

    def __init__(
        self,
        session: TapeSession,
        upstream: httpx.AsyncBaseTransport | None = None,
        *,
        owns_upstream: bool | None = None,
    ) -> None:
        self._session = session
        self._owns_upstream = (upstream is None) if owns_upstream is None else owns_upstream
        self._upstream = upstream or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._session.handle_async(request, self._upstream.handle_async_request)

    async def aclose(self) -> None:
        if self._owns_upstream:
            await self._upstream.aclose()
