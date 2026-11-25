"""Secret scrubbing, body encoding, field stripping, fingerprinting, and request diffs."""

from __future__ import annotations

import base64
import copy
import difflib
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from llm_tape.models import RecordedRequest

REDACTED = "[REDACTED]"
URL_USERINFO_REDACTED = "REDACTED"
VALID_MATCH_ON: frozenset[str] = frozenset(
    {"method", "url", "host", "path", "query", "body", "headers"}
)
DEFAULT_MATCH_ON: tuple[str, ...] = ("method", "url", "body")

DEFAULT_SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "x-goog-api-key",
        "x-auth-token",
        "api-key",
        "ocp-apim-subscription-key",
        "x-amz-security-token",
        "proxy-api-key",
        "anthropic-api-key",
        "openai-api-key",
        "cookie",
        "set-cookie",
    }
)

DEFAULT_SENSITIVE_QUERY_PARAMS: frozenset[str] = frozenset(
    {
        "key",
        "api_key",
        "apikey",
        "token",
        "access_token",
    }
)


def scrub_headers(
    headers: Mapping[str, str] | Iterable[tuple[str, str]],
    extra_sensitive: Iterable[str] = (),
) -> dict[str, str]:
    """Return a copy of headers with sensitive values redacted."""
    extra_set = {h.lower() for h in extra_sensitive} if extra_sensitive else None
    sensitive = DEFAULT_SENSITIVE_HEADERS | extra_set if extra_set else DEFAULT_SENSITIVE_HEADERS
    items = headers.items() if isinstance(headers, Mapping) else headers
    cleaned: dict[str, str] = {}
    for key, value in items:
        if key.lower() in sensitive:
            cleaned[key] = REDACTED
        else:
            cleaned[key] = value
    return cleaned


def scrub_url(
    url: str,
    extra_sensitive: Iterable[str] = (),
    ignore_query_params: Iterable[str] = (),
) -> str:
    """Return URL with basic-auth userinfo and sensitive query params redacted and sorted."""
    parts = urlsplit(url)
    has_userinfo = "@" in parts.netloc
    if not parts.query and not has_userinfo:
        return url

    extra_set = {p.lower() for p in extra_sensitive} if extra_sensitive else None
    sensitive = (
        DEFAULT_SENSITIVE_QUERY_PARAMS | extra_set if extra_set else DEFAULT_SENSITIVE_QUERY_PARAMS
    )
    ignored = {p.lower() for p in ignore_query_params} if ignore_query_params else frozenset()

    netloc = parts.netloc
    if has_userinfo:
        _, hostport = netloc.rsplit("@", 1)
        netloc = f"{URL_USERINFO_REDACTED}@{hostport}"

    new_query = ""
    if parts.query:
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        cleaned_pairs = [
            (k, REDACTED if k.lower() in sensitive else v)
            for k, v in sorted(pairs)
            if k.lower() not in ignored
        ]
        new_query = urlencode(cleaned_pairs)

    return urlunsplit((parts.scheme, netloc, parts.path, new_query, parts.fragment))


def decode_body_with_encoding(raw_bytes: bytes) -> tuple[Any, str]:
    """Decode bytes into (value, encoding) preserving JSON primitives and binary payloads."""
    if not raw_bytes:
        return None, "empty"
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return base64.b64encode(raw_bytes).decode("ascii"), "base64"

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text, "text"
    return parsed, "json"


def encode_body_with_encoding(body: Any, encoding: str) -> bytes:
    """Encode stored body back into bytes faithfully according to its recorded encoding."""
    if encoding == "empty":
        return b""
    if encoding == "base64":
        return base64.b64decode(str(body).encode("ascii"))
    if encoding == "json":
        return json.dumps(body, ensure_ascii=False).encode("utf-8")
    return b"" if body is None else str(body).encode("utf-8")


def decode_body(raw_bytes: bytes) -> Any:
    """Backwards-compatible helper returning only the decoded body value."""
    body, _ = decode_body_with_encoding(raw_bytes)
    return body


def encode_body(body: Any) -> bytes:
    """Backwards-compatible helper encoding a body value."""
    if body is None:
        return b""
    if isinstance(body, (dict, list, int, float, bool)):
        return json.dumps(body, ensure_ascii=False).encode("utf-8")
    return str(body).encode("utf-8")


_SENTINEL = object()


def _mark_delete_path(node: Any, parts: list[str]) -> None:
    if not parts:
        return
    head, *rest = parts
    if isinstance(node, dict):
        if head == "*":
            if not rest:
                node.clear()
            else:
                for val in node.values():
                    _mark_delete_path(val, rest)
        elif head in node:
            if not rest:
                del node[head]
            else:
                _mark_delete_path(node[head], rest)
    elif isinstance(node, list):
        if head == "*":
            for item in node:
                _mark_delete_path(item, rest)
        elif head.isdigit():
            idx = int(head)
            if 0 <= idx < len(node):
                if not rest:
                    node[idx] = _SENTINEL
                else:
                    _mark_delete_path(node[idx], rest)


def _prune_sentinels(node: Any) -> Any:
    if isinstance(node, list):
        return [_prune_sentinels(item) for item in node if item is not _SENTINEL]
    if isinstance(node, dict):
        return {k: _prune_sentinels(v) for k, v in node.items()}
    return node


def _redact_path(node: Any, parts: list[str]) -> None:
    if not parts:
        return
    head, *rest = parts
    if isinstance(node, dict):
        if head == "*":
            if not rest:
                for key in list(node.keys()):
                    node[key] = REDACTED
            else:
                for val in node.values():
                    _redact_path(val, rest)
        elif head in node:
            if not rest:
                node[head] = REDACTED
            else:
                _redact_path(node[head], rest)
    elif isinstance(node, list):
        if head == "*":
            if not rest:
                for idx in range(len(node)):
                    node[idx] = REDACTED
            else:
                for item in node:
                    _redact_path(item, rest)
        elif head.isdigit():
            idx = int(head)
            if 0 <= idx < len(node):
                if not rest:
                    node[idx] = REDACTED
                else:
                    _redact_path(node[idx], rest)


def compile_patterns(patterns: Iterable[str | re.Pattern[str]]) -> tuple[re.Pattern[str], ...]:
    """Compile an iterable of regex strings or pre-compiled Patterns into a tuple."""
    compiled: list[re.Pattern[str]] = []
    for pat in patterns:
        if isinstance(pat, re.Pattern):
            compiled.append(pat)
        elif isinstance(pat, str) and pat:
            compiled.append(re.compile(pat))
    return tuple(compiled)


def redact_text(text: str, redact_patterns: Iterable[str | re.Pattern[str]]) -> str:
    """Apply regex redaction patterns to a string."""
    if not text or not redact_patterns:
        return text
    compiled = compile_patterns(redact_patterns)
    if not compiled:
        return text
    result = text
    for pat in compiled:
        result = pat.sub(REDACTED, result)
    return result


def _redact_patterns_in_node(node: Any, compiled: Sequence[re.Pattern[str]]) -> Any:
    if isinstance(node, str):
        result = node
        for pat in compiled:
            result = pat.sub(REDACTED, result)
        return result
    if isinstance(node, list):
        return [_redact_patterns_in_node(item, compiled) for item in node]
    if isinstance(node, dict):
        return {k: _redact_patterns_in_node(v, compiled) for k, v in node.items()}
    return node


def redact_body(
    body: Any,
    *,
    redact_fields: Iterable[str] = (),
    redact_patterns: Iterable[str | re.Pattern[str]] = (),
    scrub_body: Callable[[Any], Any] | None = None,
) -> Any:
    """Return a copy of body with dot-path fields, regex patterns, and custom scrubber applied."""
    fields = [f.strip() for f in redact_fields if f.strip()] if redact_fields else []
    compiled = compile_patterns(redact_patterns) if redact_patterns else ()
    if not fields and not compiled and scrub_body is None:
        return body

    result = copy.deepcopy(body)
    if fields and isinstance(result, (dict, list)):
        for field_path in fields:
            _redact_path(result, field_path.split("."))
    if compiled:
        result = _redact_patterns_in_node(result, compiled)
    if scrub_body is not None:
        result = scrub_body(result)
    return result


def strip_ignored_fields(body: Any, ignore_fields: Iterable[str]) -> Any:
    """Return a deep copy of body with dot-path fields (supporting '*' wildcards) removed."""
    fields = [f.strip() for f in ignore_fields if f.strip()]
    if not fields or not isinstance(body, (dict, list)):
        return body
    cloned = copy.deepcopy(body)
    for field_path in fields:
        _mark_delete_path(cloned, field_path.split("."))
    return _prune_sentinels(cloned)


def validate_match_on(match_on: Iterable[str]) -> tuple[str, ...]:
    """Validate and normalize a sequence of match_on keys."""
    normalized = tuple(k.lower().strip() for k in match_on if k.strip())
    if not normalized:
        raise ValueError(f"match_on must be non-empty. Valid keys: {sorted(VALID_MATCH_ON)}")
    invalid = [k for k in normalized if k not in VALID_MATCH_ON]
    if invalid:
        raise ValueError(
            f"Invalid match_on key(s): {invalid}. Expected subset of: {sorted(VALID_MATCH_ON)}"
        )
    return normalized


def fingerprint_request(
    method: str,
    scrubbed_url: str,
    body: Any,
    *,
    headers: Mapping[str, str] | None = None,
    ignore_fields: Iterable[str] = (),
    ignore_query_params: Iterable[str] = (),
    match_on: Sequence[str] = DEFAULT_MATCH_ON,
) -> str:
    """Compute a deterministic SHA-256 fingerprint for matching HTTP requests."""
    normalized_url = (
        scrub_url(scrubbed_url, ignore_query_params=ignore_query_params)
        if ignore_query_params
        else scrubbed_url
    )
    filtered_body = strip_ignored_fields(body, ignore_fields)
    canonical_body = (
        json.dumps(filtered_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if isinstance(filtered_body, (dict, list))
        else ("" if filtered_body is None else str(filtered_body))
    )

    keys = validate_match_on(match_on)
    if keys == DEFAULT_MATCH_ON:
        # Preserve exact v0.1/v0.2 fingerprint hash format for full backwards compatibility
        payload = f"{method.upper()}\n{normalized_url}\n{canonical_body}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    url_parts = urlsplit(normalized_url)
    pieces: list[str] = []
    for key in keys:
        if key == "method":
            pieces.append(f"method:{method.upper()}")
        elif key == "url":
            pieces.append(f"url:{normalized_url}")
        elif key == "host":
            pieces.append(f"host:{(url_parts.netloc or '').lower()}")
        elif key == "path":
            pieces.append(f"path:{url_parts.path}")
        elif key == "query":
            pieces.append(f"query:{url_parts.query}")
        elif key == "body":
            pieces.append(f"body:{canonical_body}")
        elif key == "headers":
            norm_headers = {k.lower(): v for k, v in (headers or {}).items()}
            pieces.append(
                "headers:"
                + json.dumps(
                    norm_headers, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
            )
    payload = "\n".join(pieces)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _render_request_for_diff(
    req: RecordedRequest,
    *,
    ignore_fields: Iterable[str] = (),
    ignore_query_params: Iterable[str] = (),
) -> list[str]:
    url = (
        scrub_url(req.url, ignore_query_params=ignore_query_params)
        if ignore_query_params
        else req.url
    )
    body = strip_ignored_fields(req.body, ignore_fields)
    body_str = (
        json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False)
        if isinstance(body, (dict, list))
        else str(body)
    )
    return [f"{req.method} {url}", *body_str.splitlines()]


def format_request_diff(
    incoming: RecordedRequest,
    candidates: Sequence[RecordedRequest],
    *,
    ignore_fields: Iterable[str] = (),
    ignore_query_params: Iterable[str] = (),
    max_lines: int = 40,
) -> str:
    """Format a compact unified diff against the closest recorded request."""
    if not candidates:
        return "(cassette contains no recorded requests)"

    incoming_lines = _render_request_for_diff(
        incoming, ignore_fields=ignore_fields, ignore_query_params=ignore_query_params
    )
    incoming_text = "\n".join(incoming_lines)

    rendered_candidates = [
        _render_request_for_diff(
            c, ignore_fields=ignore_fields, ignore_query_params=ignore_query_params
        )
        for c in candidates
    ]
    recorded_lines = max(
        rendered_candidates,
        key=lambda lines: difflib.SequenceMatcher(None, "\n".join(lines), incoming_text).ratio(),
    )
    diff_lines = list(
        difflib.unified_diff(
            recorded_lines,
            incoming_lines,
            fromfile="recorded",
            tofile="incoming",
            lineterm="",
        )
    )
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines] + ["... (diff truncated)"]
    return "\n".join(diff_lines)
