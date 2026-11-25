"""Composable custom_matcher builders for TapeSession request matching."""

from __future__ import annotations

import re
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


def match_json_regex(path: str, pattern: str | re.Pattern[str]) -> RequestMatcher:
    """Match when dot-path `path` in both incoming and recorded bodies matches `pattern`."""
    compiled = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)

    def _matcher(incoming: RecordedRequest, recorded: RecordedRequest) -> bool:
        inc_val = _get_dot_path(incoming.body, path)
        rec_val = _get_dot_path(recorded.body, path)
        if inc_val is _MISSING or rec_val is _MISSING:
            return False
        return bool(compiled.search(str(inc_val))) and bool(compiled.search(str(rec_val)))

    return _matcher


def _lookup_header(headers: dict[str, str], target_lower: str) -> str | None:
    for k, v in headers.items():
        if k.lower() == target_lower:
            return v
    return None


def match_header_equals(
    name: str,
    value: str | None = None,
    *,
    case_sensitive: bool = False,
) -> RequestMatcher:
    """Match on a specific HTTP header (exact equality or matching a fixed expected value)."""
    target_lower = name.lower().strip()

    def _norm(s: str | None) -> str | None:
        if s is None:
            return None
        return s if case_sensitive else s.lower()

    expected_norm = _norm(value)

    def _matcher(incoming: RecordedRequest, recorded: RecordedRequest) -> bool:
        inc_val = _norm(_lookup_header(incoming.headers, target_lower))
        rec_val = _norm(_lookup_header(recorded.headers, target_lower))
        if expected_norm is not None:
            return inc_val == expected_norm and rec_val == expected_norm
        return inc_val == rec_val

    return _matcher
