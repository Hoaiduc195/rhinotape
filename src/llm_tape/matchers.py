"""Composable custom_matcher builders for TapeSession request matching."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlsplit

from llm_tape.models import RecordedRequest

RequestMatcher = Callable[[RecordedRequest, RecordedRequest], bool]
_MISSING: Any = object()


def _get_dot_path(node: Any, path: str) -> Any:
    cur = node
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return _MISSING
        else:
            return _MISSING
    return cur


def _is_subset(subset: Any, superset: Any) -> bool:
    if isinstance(subset, dict):
        if not isinstance(superset, dict):
            return False
        return all(k in superset and _is_subset(v, superset[k]) for k, v in subset.items())
    return bool(subset == superset)


def match_method() -> RequestMatcher:
    """Match when incoming and recorded HTTP methods are equal (case-insensitive)."""

    def _matcher(incoming: RecordedRequest, recorded: RecordedRequest) -> bool:
        return incoming.method.upper() == recorded.method.upper()

    return _matcher


def match_path() -> RequestMatcher:
    """Match when incoming and recorded URL paths are equal."""

    def _matcher(incoming: RecordedRequest, recorded: RecordedRequest) -> bool:
        return urlsplit(incoming.url).path == urlsplit(recorded.url).path

    return _matcher


def match_json_subset(keys: Iterable[str] | None = None) -> RequestMatcher:
    """Match JSON bodies either on specific dot-path `keys` or recursive dict subset."""
    key_tuple = tuple(k.strip() for k in keys if k.strip()) if keys is not None else None

    def _matcher(incoming: RecordedRequest, recorded: RecordedRequest) -> bool:
        if key_tuple is not None:
            return all(
                _get_dot_path(incoming.body, k) == _get_dot_path(recorded.body, k)
                for k in key_tuple
            )
        return _is_subset(recorded.body, incoming.body)

    return _matcher


def match_all(*matchers: RequestMatcher) -> RequestMatcher:
    """Logical AND combinator over multiple RequestMatchers."""

    def _matcher(incoming: RecordedRequest, recorded: RecordedRequest) -> bool:
        return all(m(incoming, recorded) for m in matchers)

    return _matcher


def match_any(*matchers: RequestMatcher) -> RequestMatcher:
    """Logical OR combinator over multiple RequestMatchers."""

    def _matcher(incoming: RecordedRequest, recorded: RecordedRequest) -> bool:
        return any(m(incoming, recorded) for m in matchers)

    return _matcher
