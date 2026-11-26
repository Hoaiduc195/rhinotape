"""Composable custom_matcher builders for TapeSession request matching."""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlsplit

from rhinotape.models import RecordedRequest

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
    """Match when recorded and incoming HTTP methods are equal (case-insensitive)."""

    def _matcher(recorded: RecordedRequest, incoming: RecordedRequest) -> bool:
        return recorded.method.upper() == incoming.method.upper()

    return _matcher


def match_path() -> RequestMatcher:
    """Match when recorded and incoming URL paths are equal."""

    def _matcher(recorded: RecordedRequest, incoming: RecordedRequest) -> bool:
        return urlsplit(recorded.url).path == urlsplit(incoming.url).path

    return _matcher


def match_json_subset(keys: Iterable[str] | None = None) -> RequestMatcher:
    """Match JSON bodies either on specific dot-path `keys` or recursive dict subset.

    When `keys` is None, returns True iff `recorded.body` is a recursive subset of `incoming.body`.
    """
    key_tuple = tuple(k.strip() for k in keys if k.strip()) if keys is not None else None

    def _matcher(recorded: RecordedRequest, incoming: RecordedRequest) -> bool:
        if key_tuple is not None:
            return all(
                _get_dot_path(recorded.body, k) == _get_dot_path(incoming.body, k)
                for k in key_tuple
            )
        return _is_subset(recorded.body, incoming.body)

    return _matcher


def match_all(*matchers: RequestMatcher) -> RequestMatcher:
    """Logical AND combinator over multiple RequestMatchers."""

    def _matcher(recorded: RecordedRequest, incoming: RecordedRequest) -> bool:
        return all(m(recorded, incoming) for m in matchers)

    return _matcher


def match_any(*matchers: RequestMatcher) -> RequestMatcher:
    """Logical OR combinator over multiple RequestMatchers."""

    def _matcher(recorded: RecordedRequest, incoming: RecordedRequest) -> bool:
        return any(m(recorded, incoming) for m in matchers)

    return _matcher


def match_json_regex(path: str, pattern: str | re.Pattern[str]) -> RequestMatcher:
    """Match when dot-path `path` in both recorded and incoming bodies matches `pattern`."""
    compiled = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)

    def _matcher(recorded: RecordedRequest, incoming: RecordedRequest) -> bool:
        rec_val = _get_dot_path(recorded.body, path)
        inc_val = _get_dot_path(incoming.body, path)
        if rec_val is _MISSING or inc_val is _MISSING:
            return False
        return bool(compiled.search(str(rec_val))) and bool(compiled.search(str(inc_val)))

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

    def _matcher(recorded: RecordedRequest, incoming: RecordedRequest) -> bool:
        rec_val = _norm(_lookup_header(recorded.headers, target_lower))
        inc_val = _norm(_lookup_header(incoming.headers, target_lower))
        if expected_norm is not None:
            return rec_val == expected_norm and inc_val == expected_norm
        return rec_val == inc_val

    return _matcher


def _flatten_text_value(val: Any) -> str | None:
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts: list[str] = []
        for item in val:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif "content" in item:
                    sub = _flatten_text_value(item["content"])
                    if sub:
                        parts.append(sub)
                elif isinstance(item.get("parts"), list):
                    sub = _flatten_text_value(item["parts"])
                    if sub:
                        parts.append(sub)
        return "\n".join(parts) if parts else None
    if isinstance(val, dict):
        if isinstance(val.get("text"), str):
            return str(val["text"])
        if "content" in val:
            return _flatten_text_value(val["content"])
    return None


def _extract_prompt_text(body: Any, path: str | None) -> str | None:
    if path is not None:
        val = _get_dot_path(body, path)
        if val is _MISSING:
            return None
        flat = _flatten_text_value(val)
        return flat if flat is not None else str(val)
    if isinstance(body, str):
        return body
    if not isinstance(body, dict):
        return None
    # Auto-detect common LLM prompt containers (OpenAI, Anthropic, Gemini, Ollama, Cohere, Bedrock)
    for key in ("messages", "input", "contents", "prompt", "message"):
        if key in body:
            extracted = _flatten_text_value(body[key])
            if extracted is not None:
                return extracted
    return None


def match_fuzzy_text(
    path: str | None = None,
    *,
    threshold: float = 0.85,
    case_sensitive: bool = False,
    normalize_whitespace: bool = True,
    scorer: Callable[[str, str], float] | None = None,
) -> RequestMatcher:
    """Match requests when extracted prompt/field text meets a similarity threshold.

    When `path` is None, automatically extracts prompt text from common LLM request formats
    (`messages`, `input`, `contents`, `prompt`, `message`, or a plain string body).
    Uses stdlib `difflib.SequenceMatcher` by default, or a pluggable `scorer(a, b) -> float`.
    """
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be a finite float in [0.0, 1.0], got {threshold!r}")

    def _normalize(text: str) -> str:
        out = " ".join(text.split()) if normalize_whitespace else text
        return out if case_sensitive else out.lower()

    def _matcher(recorded: RecordedRequest, incoming: RecordedRequest) -> bool:
        rec_raw = _extract_prompt_text(recorded.body, path)
        inc_raw = _extract_prompt_text(incoming.body, path)
        if rec_raw is None or inc_raw is None:
            return False
        rec_norm = _normalize(rec_raw)
        inc_norm = _normalize(inc_raw)
        score = (
            float(scorer(rec_norm, inc_norm))
            if scorer is not None
            else difflib.SequenceMatcher(None, rec_norm, inc_norm).ratio()
        )
        return math.isfinite(score) and score >= threshold

    return _matcher
