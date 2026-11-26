"""Secret scrubbing, body encoding, field stripping, fingerprinting, and request diffs."""

from __future__ import annotations

import base64
import copy
import difflib
import fnmatch
import functools
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from rhinotape.models import RecordedRequest

REDACTED = "[REDACTED]"
URL_USERINFO_REDACTED = "REDACTED"
VALID_MATCH_ON: frozenset[str] = frozenset(
    {"method", "url", "host", "path", "query", "body", "headers"}
)
DEFAULT_MATCH_ON: tuple[str, ...] = ("method", "url", "body")
MATCH_STRICT: tuple[str, ...] = ("method", "url", "body", "headers")
MATCH_RELAXED: tuple[str, ...] = ("method", "path")
MATCH_PATH_AND_BODY: tuple[str, ...] = ("method", "path", "body")

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
        "secret",
        "client_secret",
        "sig",
        "signature",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
    }
)

PII_EMAIL: re.Pattern[str] = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PII_PHONE: re.Pattern[str] = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)
PII_SSN: re.Pattern[str] = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
PII_CREDIT_CARD: re.Pattern[str] = re.compile(r"(?<!\d)(?:\d{4}[-\s]?){3}\d{4}(?!\d)")
PII_IPV4: re.Pattern[str] = re.compile(
    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d)"
)
PII_JWT: re.Pattern[str] = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
)

PII_PRESETS: dict[str, re.Pattern[str]] = {
    "email": PII_EMAIL,
    "phone": PII_PHONE,
    "ssn": PII_SSN,
    "credit_card": PII_CREDIT_CARD,
    "ipv4": PII_IPV4,
    "jwt": PII_JWT,
}


def resolve_pii_patterns(redact_pii: bool | str | Iterable[str]) -> tuple[re.Pattern[str], ...]:
    """Resolve `redact_pii` flag or preset names into compiled regex patterns."""
    if redact_pii is False or redact_pii is None:
        return ()
    if redact_pii is True:
        return tuple(PII_PRESETS.values())
    names = [redact_pii] if isinstance(redact_pii, str) else list(redact_pii)
    resolved: list[re.Pattern[str]] = []
    seen: set[str] = set()
    for raw in names:
        key = raw.strip().lower()
        if not key:
            continue
        if key == "all":
            return tuple(PII_PRESETS.values())
        if key not in PII_PRESETS:
            valid = ", ".join(sorted(PII_PRESETS))
            raise ValueError(f"Unknown PII preset {raw!r}. Expected one of: {valid}, all")
        if key not in seen:
            seen.add(key)
            resolved.append(PII_PRESETS[key])
    return tuple(resolved)


_GLOB_CHARS = frozenset("*?[")


def compile_sensitive_headers(
    extra_sensitive: Iterable[str] = (),
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Pre-split extra sensitive header rules into (exact_frozenset, glob_patterns_tuple)."""
    if not extra_sensitive:
        return DEFAULT_SENSITIVE_HEADERS, ()
    exact_extra: set[str] = set()
    glob_patterns: list[str] = []
    for h in extra_sensitive:
        low = h.lower()
        if any(ch in low for ch in _GLOB_CHARS):
            glob_patterns.append(low)
        else:
            exact_extra.add(low)
    sensitive = (
        DEFAULT_SENSITIVE_HEADERS | exact_extra if exact_extra else DEFAULT_SENSITIVE_HEADERS
    )
    return sensitive, tuple(glob_patterns)


def scrub_headers(
    headers: Mapping[str, str] | Iterable[tuple[str, str]],
    extra_sensitive: Iterable[str] = (),
    *,
    compiled: tuple[frozenset[str], tuple[str, ...]] | None = None,
) -> dict[str, str]:
    """Return a copy of headers with sensitive values redacted (supports '*' wildcards)."""
    if compiled is not None:
        sensitive, glob_patterns = compiled
    else:
        sensitive, glob_patterns = compile_sensitive_headers(extra_sensitive)
    items = headers.items() if isinstance(headers, Mapping) else headers
    cleaned: dict[str, str] = {}
    for key, value in items:
        low_key = key.lower()
        if low_key in sensitive or (
            glob_patterns and any(fnmatch.fnmatchcase(low_key, pat) for pat in glob_patterns)
        ):
            cleaned[key] = REDACTED
        else:
            cleaned[key] = value
    return cleaned


def scrub_url(
    url: str,
    extra_sensitive: Iterable[str] = (),
    ignore_query_params: Iterable[str] = (),
    redact_url_patterns: Iterable[str | re.Pattern[str]] = (),
) -> str:
    """Return URL with basic-auth userinfo, sensitive query params, and path patterns redacted."""
    parts = urlsplit(url)
    has_userinfo = "@" in parts.netloc
    compiled_path = compile_patterns(redact_url_patterns) if redact_url_patterns else ()
    if not parts.query and not has_userinfo and not compiled_path:
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

    path = parts.path
    if compiled_path and path:
        for pat in compiled_path:
            path = pat.sub(URL_USERINFO_REDACTED, path)

    new_query = ""
    if parts.query:
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        cleaned_pairs = [
            (k, REDACTED if k.lower() in sensitive else v)
            for k, v in sorted(pairs)
            if k.lower() not in ignored
        ]
        new_query = urlencode(cleaned_pairs)

    return urlunsplit((parts.scheme, netloc, path, new_query, parts.fragment))


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
    if isinstance(patterns, tuple) and all(isinstance(p, re.Pattern) for p in patterns):
        return patterns
    compiled: list[re.Pattern[str]] = []
    for pat in patterns:
        if isinstance(pat, re.Pattern):
            compiled.append(pat)
        elif isinstance(pat, str) and pat:
            compiled.append(re.compile(pat))
    return tuple(compiled)


_UNSAFE_COMBINE_RE = re.compile(r"(?<!\\)\\[1-9]|(?<!\\)\(\?P[<=]|(?<!\\)\(\?[aiLmsux-]+[:)]")


@functools.lru_cache(maxsize=64)
def _combine_patterns(
    compiled: tuple[re.Pattern[str], ...],
) -> re.Pattern[str] | None:
    """Combine compatible regexes into a single-pass alternation when safe."""
    if len(compiled) < 2:
        return compiled[0] if compiled else None
    flags = compiled[0].flags
    for pat in compiled:
        if pat.flags != flags or _UNSAFE_COMBINE_RE.search(pat.pattern):
            return None
    try:
        return re.compile("|".join(f"(?:{p.pattern})" for p in compiled), flags)
    except re.error:
        return None


def _apply_compiled_patterns(text: str, compiled: tuple[re.Pattern[str], ...]) -> str:
    combined = _combine_patterns(compiled)
    if combined is not None:
        return combined.sub(REDACTED, text)
    result = text
    for pat in compiled:
        result = pat.sub(REDACTED, result)
    return result


def redact_text(text: str, redact_patterns: Iterable[str | re.Pattern[str]]) -> str:
    """Apply regex redaction patterns to a string."""
    if not text or not redact_patterns:
        return text
    compiled = compile_patterns(redact_patterns)
    if not compiled:
        return text
    return _apply_compiled_patterns(text, compiled)


def _redact_patterns_in_node(node: Any, compiled: tuple[re.Pattern[str], ...]) -> Any:
    combined = _combine_patterns(compiled)
    if combined is not None:
        return _redact_single_pattern_in_node(node, combined)
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


def _redact_single_pattern_in_node(node: Any, pattern: re.Pattern[str]) -> Any:
    if isinstance(node, str):
        return pattern.sub(REDACTED, node)
    if isinstance(node, list):
        return [_redact_single_pattern_in_node(item, pattern) for item in node]
    if isinstance(node, dict):
        return {k: _redact_single_pattern_in_node(v, pattern) for k, v in node.items()}
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

    # _redact_patterns_in_node already constructs fresh dicts/lists, so skip deepcopy when only
    # regex patterns are active (cuts allocations and tree traversal in half).
    if not fields and scrub_body is None:
        return _redact_patterns_in_node(body, compiled)

    result = copy.deepcopy(body)
    if fields and isinstance(result, (dict, list)):
        for field_path in fields:
            _redact_path(result, field_path.split("."))
    if compiled:
        result = _redact_patterns_in_node(result, compiled)
    if scrub_body is not None:
        result = scrub_body(result)
    return result


def _sort_array_path_cow(node: Any, parts: list[str]) -> Any:
    if not parts:
        if isinstance(node, list):
            return sorted(
                node,
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ),
            )
        return node
    head, *rest = parts
    if isinstance(node, dict):
        if head == "*":
            return {k: _sort_array_path_cow(v, rest) for k, v in node.items()}
        if head in node:
            new_val = _sort_array_path_cow(node[head], rest)
            if new_val is node[head]:
                return node
            out = dict(node)
            out[head] = new_val
            return out
        return node
    if isinstance(node, list):
        if head == "*":
            return [_sort_array_path_cow(item, rest) for item in node]
        if head.isdigit():
            idx = int(head)
            if 0 <= idx < len(node):
                new_val = _sort_array_path_cow(node[idx], rest)
                if new_val is node[idx]:
                    return node
                out_list = list(node)
                out_list[idx] = new_val
                return out_list
        return node
    return node


_SENTINEL = object()


def _strip_path_cow(node: Any, parts: list[str], mark_list: list[bool]) -> Any:
    if not parts:
        return node
    head, *rest = parts
    if isinstance(node, dict):
        if head == "*":
            if not rest:
                return {}
            return {k: _strip_path_cow(v, rest, mark_list) for k, v in node.items()}
        if head in node:
            if not rest:
                return {k: v for k, v in node.items() if k != head}
            new_val = _strip_path_cow(node[head], rest, mark_list)
            if new_val is node[head]:
                return node
            out = dict(node)
            out[head] = new_val
            return out
        return node
    if isinstance(node, list):
        if head == "*":
            return [_strip_path_cow(item, rest, mark_list) for item in node]
        if head.isdigit():
            idx = int(head)
            if 0 <= idx < len(node):
                if not rest:
                    mark_list[0] = True
                    out_list = list(node)
                    out_list[idx] = _SENTINEL
                    return out_list
                new_val = _strip_path_cow(node[idx], rest, mark_list)
                if new_val is node[idx]:
                    return node
                out_list = list(node)
                out_list[idx] = new_val
                return out_list
        return node
    return node


def _prune_sentinels_cow(node: Any) -> Any:
    if isinstance(node, list):
        changed = False
        out_list: list[Any] = []
        for item in node:
            if item is _SENTINEL:
                changed = True
                continue
            pruned = _prune_sentinels_cow(item)
            if pruned is not item:
                changed = True
            out_list.append(pruned)
        return out_list if changed else node
    if isinstance(node, dict):
        changed = False
        out_dict: dict[str, Any] = {}
        for k, v in node.items():
            pruned = _prune_sentinels_cow(v)
            if pruned is not v:
                changed = True
            out_dict[k] = pruned
        return out_dict if changed else node
    return node


def normalize_json_array_order(body: Any, array_paths: Iterable[str]) -> Any:
    """Return a copy-on-write view of body with JSON arrays at dot-paths sorted canonically."""
    if not array_paths or not isinstance(body, (dict, list)):
        return body
    paths = [p.strip() for p in array_paths if p.strip()]
    if not paths:
        return body
    result = body
    for field_path in paths:
        result = _sort_array_path_cow(result, field_path.split("."))
    return result


def strip_ignored_fields(body: Any, ignore_fields: Iterable[str]) -> Any:
    """Return a copy-on-write view of body with ignored dot-path fields ('*' wildcards) removed."""
    if not ignore_fields or not isinstance(body, (dict, list)):
        return body
    fields = [f.strip() for f in ignore_fields if f.strip()]
    if not fields:
        return body
    has_sentinel = [False]
    result = body
    for field_path in fields:
        result = _strip_path_cow(result, field_path.split("."), has_sentinel)
    return _prune_sentinels_cow(result) if has_sentinel[0] else result


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
    ignore_json_array_order: Iterable[str] = (),
    match_on: Sequence[str] = DEFAULT_MATCH_ON,
    match_headers: Iterable[str] = (),
) -> str:
    """Compute a deterministic SHA-256 fingerprint for matching HTTP requests."""
    normalized_url = (
        scrub_url(scrubbed_url, ignore_query_params=ignore_query_params)
        if ignore_query_params
        else scrubbed_url
    )
    filtered_body = strip_ignored_fields(body, ignore_fields)
    if ignore_json_array_order:
        filtered_body = normalize_json_array_order(filtered_body, ignore_json_array_order)
    canonical_body = (
        json.dumps(filtered_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if isinstance(filtered_body, (dict, list))
        else ("" if filtered_body is None else str(filtered_body))
    )

    if match_on is DEFAULT_MATCH_ON and not match_headers:
        # Fast path: preserve exact v0.1/v0.2 fingerprint hash format without re-validating
        payload = f"{method.upper()}\n{normalized_url}\n{canonical_body}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    keys = validate_match_on(match_on)
    header_filter = frozenset(h.lower().strip() for h in match_headers if h.strip())
    if keys == DEFAULT_MATCH_ON and not header_filter:
        # Preserve exact v0.1/v0.2 fingerprint hash format for full backwards compatibility
        payload = f"{method.upper()}\n{normalized_url}\n{canonical_body}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    effective_keys = keys if ("headers" in keys or not header_filter) else (*keys, "headers")
    url_parts = urlsplit(normalized_url)
    pieces: list[str] = []
    for key in effective_keys:
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
            norm_headers = {
                k.lower(): v
                for k, v in (headers or {}).items()
                if not header_filter or k.lower() in header_filter
            }
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


def _collect_json_diffs(
    recorded: Any,
    incoming: Any,
    prefix: str,
    out: list[str],
    max_paths: int,
) -> None:
    if len(out) >= max_paths or recorded == incoming:
        return
    if isinstance(recorded, dict) and isinstance(incoming, dict):
        for key in sorted(set(recorded) | set(incoming)):
            if len(out) >= max_paths:
                return
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in recorded:
                out.append(f"+ {path}")
            elif key not in incoming:
                out.append(f"- {path}")
            else:
                _collect_json_diffs(recorded[key], incoming[key], path, out, max_paths)
    elif isinstance(recorded, list) and isinstance(incoming, list):
        if len(recorded) != len(incoming):
            out.append(f"~ {prefix or '$'} (len {len(recorded)} -> {len(incoming)})")
            return
        for idx, (a, b) in enumerate(zip(recorded, incoming, strict=True)):
            if len(out) >= max_paths:
                return
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _collect_json_diffs(a, b, path, out, max_paths)
    else:
        out.append(f"~ {prefix or '$'}")


def summarize_json_diff(recorded_body: Any, incoming_body: Any, *, max_paths: int = 8) -> list[str]:
    """Return a concise list of semantic JSON key-path differences between two bodies."""
    if not isinstance(recorded_body, (dict, list)) or not isinstance(incoming_body, (dict, list)):
        return []
    out: list[str] = []
    _collect_json_diffs(recorded_body, incoming_body, "", out, max_paths)
    return out


def find_closest_request(
    incoming: RecordedRequest,
    candidates: Sequence[RecordedRequest],
    *,
    ignore_fields: Iterable[str] = (),
    ignore_query_params: Iterable[str] = (),
) -> RecordedRequest | None:
    """Return the recorded candidate request most similar to `incoming`."""
    if not candidates:
        return None
    incoming_text = "\n".join(
        _render_request_for_diff(
            incoming, ignore_fields=ignore_fields, ignore_query_params=ignore_query_params
        )
    )
    return max(
        candidates,
        key=lambda c: difflib.SequenceMatcher(
            None,
            "\n".join(
                _render_request_for_diff(
                    c, ignore_fields=ignore_fields, ignore_query_params=ignore_query_params
                )
            ),
            incoming_text,
        ).ratio(),
    )


def format_request_diff(
    incoming: RecordedRequest,
    candidates: Sequence[RecordedRequest],
    *,
    ignore_fields: Iterable[str] = (),
    ignore_query_params: Iterable[str] = (),
    max_lines: int = 40,
) -> str:
    """Format a compact semantic summary + unified diff against the closest recorded request."""
    if not candidates:
        return "(cassette contains no recorded requests)"

    incoming_lines = _render_request_for_diff(
        incoming, ignore_fields=ignore_fields, ignore_query_params=ignore_query_params
    )
    incoming_text = "\n".join(incoming_lines)

    best_candidate, recorded_lines = max(
        (
            (
                c,
                _render_request_for_diff(
                    c, ignore_fields=ignore_fields, ignore_query_params=ignore_query_params
                ),
            )
            for c in candidates
        ),
        key=lambda pair: difflib.SequenceMatcher(None, "\n".join(pair[1]), incoming_text).ratio(),
    )
    rec_body = strip_ignored_fields(best_candidate.body, ignore_fields)
    inc_body = strip_ignored_fields(incoming.body, ignore_fields)
    changed_paths = summarize_json_diff(rec_body, inc_body)

    diff_lines = list(
        difflib.unified_diff(
            recorded_lines,
            incoming_lines,
            fromfile="recorded",
            tofile="incoming",
            lineterm="",
        )
    )
    if changed_paths:
        diff_lines = [f"Changed JSON fields: {', '.join(changed_paths)}", *diff_lines]
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines] + ["... (diff truncated)"]
    return "\n".join(diff_lines)
