"""Composable chaos/fault-injection hooks for TapeSession(..., on_replay=...)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from rhinotape.models import Interaction
from rhinotape.session import replace_response

ReplayHook = Callable[[Interaction, int], Interaction | None]


def _merge_fault_headers(
    orig_headers: dict[str, str],
    extra_headers: dict[str, str] | None,
    body: Any,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    key_map: dict[str, str] = {}
    for k, v in orig_headers.items():
        low = k.lower()
        if low in key_map:
            del merged[key_map[low]]
        key_map[low] = k
        merged[k] = v

    explicit_ct = False
    if extra_headers:
        for k, v in extra_headers.items():
            low = k.lower()
            if low == "content-type":
                explicit_ct = True
            if low in key_map:
                del merged[key_map[low]]
            key_map[low] = k
            merged[k] = v

    if isinstance(body, (dict, list)) and not explicit_ct:
        if "content-type" in key_map:
            del merged[key_map["content-type"]]
        merged["content-type"] = "application/json"
    return merged


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
        return replace_response(
            interaction,
            status_code=status_code,
            body=err_body,
            headers=_merge_fault_headers(interaction.response.headers, headers, err_body),
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
        return replace_response(
            interaction,
            status_code=status_code,
            body=err_body,
            headers=_merge_fault_headers(interaction.response.headers, headers, err_body),
        )

    return _hook


def inject_latency(extra_ms: float, *, nth: int | None = None) -> ReplayHook:
    """Return an on_replay hook that adds `extra_ms` to interaction latency_ms."""
    if not math.isfinite(extra_ms) or extra_ms < 0.0:
        raise ValueError("extra_ms must be a finite number >= 0.0")
    if nth is not None and nth < 0:
        raise ValueError("nth must be >= 0")

    def _hook(interaction: Interaction, call_index: int) -> Interaction | None:
        if nth is not None and call_index != nth:
            return None
        return replace_response(interaction, latency_ms=interaction.latency_ms + extra_ms)

    return _hook


def corrupt_json_body(
    nth: int | None = None,
    *,
    payload: str = '{"choices":[{"message":',
) -> ReplayHook:
    """Return an on_replay hook that replaces the response body with malformed JSON text."""
    if nth is not None and nth < 0:
        raise ValueError("nth must be >= 0")

    def _hook(interaction: Interaction, call_index: int) -> Interaction | None:
        if nth is not None and call_index != nth:
            return None
        return replace_response(interaction, body=payload)

    return _hook


def truncate_stream(max_chunks: int, *, nth: int | None = None) -> ReplayHook:
    """Return an on_replay hook that truncates SSE stream_chunks to at most `max_chunks`."""
    if max_chunks < 0:
        raise ValueError("max_chunks must be >= 0")
    if nth is not None and nth < 0:
        raise ValueError("nth must be >= 0")

    def _hook(interaction: Interaction, call_index: int) -> Interaction | None:
        if nth is not None and call_index != nth:
            return None
        chunks = interaction.response.stream_chunks
        if chunks is None:
            return None
        return replace_response(interaction, stream_chunks=chunks[:max_chunks])

    return _hook


def chain_replay_hooks(*hooks: ReplayHook) -> ReplayHook:
    """Compose multiple on_replay hooks sequentially, feeding transformed Interactions forward."""

    def _hook(interaction: Interaction, call_index: int) -> Interaction | None:
        current = interaction
        changed = False
        for hook in hooks:
            result = hook(current, call_index)
            if result is not None:
                current = result
                changed = True
        return current if changed else None

    return _hook
