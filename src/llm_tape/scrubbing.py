"""Secret scrubbing and deterministic request fingerprinting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"

DEFAULT_SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "x-goog-api-key",
        "api-key",
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
    sensitive = DEFAULT_SENSITIVE_HEADERS | {h.lower() for h in extra_sensitive}
    items = headers.items() if isinstance(headers, Mapping) else headers
    cleaned: dict[str, str] = {}
    for key, value in items:
        if key.lower() in sensitive:
            cleaned[key] = REDACTED
        else:
            cleaned[key] = value
    return cleaned


def scrub_url(url: str, extra_sensitive: Iterable[str] = ()) -> str:
    """Return URL with sensitive query parameters redacted and sorted."""
    sensitive = DEFAULT_SENSITIVE_QUERY_PARAMS | {p.lower() for p in extra_sensitive}
    parts = urlsplit(url)
    if not parts.query:
        return url
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    cleaned_pairs = [
        (k, REDACTED if k.lower() in sensitive else v) for k, v in sorted(pairs)
    ]
    new_query = urlencode(cleaned_pairs)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


def decode_body(raw_bytes: bytes) -> Any:
    """Decode bytes into parsed JSON if possible, otherwise a UTF-8 string."""
    if not raw_bytes:
        return None
    text = raw_bytes.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def encode_body(body: Any) -> bytes:
    """Encode stored body (JSON value or string or None) back into bytes."""
    if body is None:
        return b""
    if isinstance(body, (dict, list, int, float, bool)):
        return json.dumps(body, ensure_ascii=False).encode("utf-8")
    return str(body).encode("utf-8")


def fingerprint_request(method: str, scrubbed_url: str, body: Any) -> str:
    """Compute a deterministic SHA-256 fingerprint for matching HTTP requests."""
    canonical_body = (
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if isinstance(body, (dict, list))
        else ("" if body is None else str(body))
    )
    payload = f"{method.upper()}\n{scrubbed_url}\n{canonical_body}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
