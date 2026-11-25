"""Composable chaos/fault-injection hooks for TapeSession(..., on_replay=...)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_tape.models import Interaction
from llm_tape.session import replace_response

ReplayHook = Callable[[Interaction, int], Interaction | None]


def fail_nth(
    n: int,
    *,
    status_code: int = 429,
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> ReplayHook:
    """Return an on_replay hook that fails the 0-based `n`-th replayed call."""
    if n < 0:
        raise ValueError("n must be >= 0")
    err_body = (
        {"error": {"message": f"Injected HTTP {status_code} fault on call {n}"}}
        if body is None
        else body
    )

    def _hook(interaction: Interaction, call_index: int) -> Interaction | None:
        if call_index != n:
            return None
        merged_headers = dict(interaction.response.headers)
        if headers:
            merged_headers.update(headers)
        return replace_response(
            interaction,
            status_code=status_code,
            body=err_body,
            headers=merged_headers,
        )

    return _hook


def fail_first(
    k: int,
    *,
    status_code: int = 503,
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> ReplayHook:
    """Return an on_replay hook that fails the first `k` replayed calls (0..k-1)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    err_body = (
        {"error": {"message": f"Injected HTTP {status_code} transient fault"}}
        if body is None
        else body
    )

    def _hook(interaction: Interaction, call_index: int) -> Interaction | None:
        if call_index >= k:
            return None
        merged_headers = dict(interaction.response.headers)
        if headers:
            merged_headers.update(headers)
        return replace_response(
            interaction,
            status_code=status_code,
            body=err_body,
            headers=merged_headers,
        )

    return _hook
