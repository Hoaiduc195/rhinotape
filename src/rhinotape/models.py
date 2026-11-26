"""Core data models and exceptions for rhinotape."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TapeMode(str, Enum):
    """Controls how rhinotape records or replays HTTP interactions."""

    ONCE = "once"  # Replay if cassette exists; otherwise record and save.
    REPLAY = "replay"  # Strict offline replay; fail if cassette or interaction missing.
    RECORD = "record"  # Always hit live network and overwrite cassette.
    NEW_EPISODES = "new_episodes"  # Replay existing matches; record and append new requests.
    NONE = "none"  # Passthrough without recording or replaying.

    @classmethod
    def coerce(cls, value: TapeMode | str) -> TapeMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except ValueError as exc:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"Invalid tape mode {value!r}. Expected one of: {valid}") from exc


class TapeError(Exception):
    """Base exception for rhinotape."""


class TapeNotFoundError(TapeError):
    """Raised in replay mode when the expected cassette file does not exist."""


class CorruptCassetteError(TapeError):
    """Raised when a cassette file cannot be parsed or has an unsupported schema."""


class UnmatchedRequestError(TapeError):
    """Raised when an outgoing request does not match any recorded interaction."""

    def __init__(
        self,
        message: str,
        *,
        incoming: RecordedRequest | None = None,
        closest: RecordedRequest | None = None,
        diff_text: str = "",
    ) -> None:
        super().__init__(message)
        self.incoming = incoming
        self.closest = closest
        self.diff_text = diff_text


class TapeAssertionError(AssertionError, TapeError):
    """Raised when a tape assertion (budget, tokens, tool calls) fails."""


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return value


def _validate_headers(raw_headers: Any) -> dict[str, str]:
    if raw_headers is None:
        return {}
    if not isinstance(raw_headers, dict):
        raise TypeError(f"headers must be a dict or None, got {type(raw_headers).__name__}")
    return {str(k): str(v) for k, v in raw_headers.items()}


@dataclass(frozen=True)
class ToolCall:
    """Normalized representation of a single LLM tool/function call."""

    name: str
    arguments: dict[str, Any] | str = field(default_factory=dict)
    call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        if not isinstance(data, dict):
            raise TypeError(f"ToolCall must be a dict, got {type(data).__name__}")
        raw_call_id = data.get("call_id")
        return cls(
            name=str(data["name"]),
            arguments=data.get("arguments", {}),
            call_id=None if raw_call_id is None else str(raw_call_id),
        )


@dataclass(frozen=True)
class Usage:
    """Normalized token usage for an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cached_input_tokens:
            payload["cached_input_tokens"] = self.cached_input_tokens
        if self.cache_creation_tokens:
            payload["cache_creation_tokens"] = self.cache_creation_tokens
        if self.reasoning_tokens:
            payload["reasoning_tokens"] = self.reasoning_tokens
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Usage:
        if not isinstance(data, dict):
            raise TypeError(f"Usage must be a dict, got {type(data).__name__}")
        return cls(
            input_tokens=_require_int(data.get("input_tokens", 0), "input_tokens"),
            output_tokens=_require_int(data.get("output_tokens", 0), "output_tokens"),
            cached_input_tokens=_require_int(
                data.get("cached_input_tokens", 0), "cached_input_tokens"
            ),
            cache_creation_tokens=_require_int(
                data.get("cache_creation_tokens", 0), "cache_creation_tokens"
            ),
            reasoning_tokens=_require_int(data.get("reasoning_tokens", 0), "reasoning_tokens"),
        )


@dataclass(frozen=True)
class LLMCallSummary:
    """Extracted LLM-level metadata for a single recorded HTTP interaction."""

    provider: str = "unknown"
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    estimated_cost_usd: float = 0.0
    tool_calls: tuple[ToolCall, ...] = ()
    text: str = ""
    thinking: str = ""
    is_stream: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "text": self.text,
            "is_stream": self.is_stream,
        }
        if self.thinking:
            payload["thinking"] = self.thinking
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMCallSummary:
        if not isinstance(data, dict):
            raise TypeError(f"LLMCallSummary must be a dict, got {type(data).__name__}")
        raw_usage = data.get("usage")
        raw_cost = data.get("estimated_cost_usd", 0.0)
        if isinstance(raw_cost, bool):
            raise TypeError("estimated_cost_usd must be a number, got bool")
        raw_tool_calls = data.get("tool_calls", [])
        if not isinstance(raw_tool_calls, list):
            raise TypeError("tool_calls must be a list")
        raw_model = data.get("model")
        return cls(
            provider=str(data.get("provider", "unknown")),
            model=None if raw_model is None else str(raw_model),
            usage=Usage.from_dict({} if raw_usage is None else raw_usage),
            estimated_cost_usd=float(raw_cost),
            tool_calls=tuple(ToolCall.from_dict(tc) for tc in raw_tool_calls),
            text=str(data.get("text", "")),
            thinking=str(data.get("thinking", "")),
            is_stream=bool(data.get("is_stream", False)),
        )


@dataclass(frozen=True)
class RecordedRequest:
    """Sanitized HTTP request saved inside a cassette."""

    method: str
    url: str
    headers: dict[str, str]
    body: Any
    fingerprint: str
    body_encoding: str = "json"  # "json" | "text" | "base64" | "empty"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "url": self.url,
            "headers": self.headers,
            "body": self.body,
            "fingerprint": self.fingerprint,
            "body_encoding": self.body_encoding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordedRequest:
        if not isinstance(data, dict):
            raise TypeError(f"RecordedRequest must be a dict, got {type(data).__name__}")
        body = data.get("body")
        default_enc = (
            "empty" if body is None else ("json" if isinstance(body, (dict, list)) else "text")
        )
        return cls(
            method=str(data["method"]),
            url=str(data["url"]),
            headers=_validate_headers(data.get("headers")),
            body=body,
            fingerprint=str(data["fingerprint"]),
            body_encoding=str(data.get("body_encoding", default_enc)),
        )


@dataclass(frozen=True)
class RecordedResponse:
    """Sanitized HTTP response saved inside a cassette."""

    status_code: int
    headers: dict[str, str]
    body: Any
    body_encoding: str = "json"  # "json" | "text" | "base64" | "empty" | "stream"
    stream_chunks: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status_code": self.status_code,
            "headers": self.headers,
            "body": self.body,
            "body_encoding": self.body_encoding,
        }
        if self.stream_chunks is not None:
            payload["stream_chunks"] = list(self.stream_chunks)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordedResponse:
        if not isinstance(data, dict):
            raise TypeError(f"RecordedResponse must be a dict, got {type(data).__name__}")
        body = data.get("body")
        raw_chunks = data.get("stream_chunks")
        if raw_chunks is not None and not isinstance(raw_chunks, (list, tuple)):
            raise TypeError("stream_chunks must be a list or None")
        chunks = tuple(str(c) for c in raw_chunks) if raw_chunks is not None else None
        if chunks is not None:
            default_enc = "stream"
        elif body is None:
            default_enc = "empty"
        elif isinstance(body, (dict, list)):
            default_enc = "json"
        else:
            default_enc = "text"
        return cls(
            status_code=_require_int(data["status_code"], "status_code"),
            headers=_validate_headers(data.get("headers")),
            body=body,
            body_encoding=str(data.get("body_encoding", default_enc)),
            stream_chunks=chunks,
        )


@dataclass(frozen=True)
class Interaction:
    """A single recorded HTTP request/response pair plus parsed LLM metadata."""

    request: RecordedRequest
    response: RecordedResponse
    latency_ms: float = 0.0
    llm: LLMCallSummary = field(default_factory=LLMCallSummary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "response": self.response.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "llm": self.llm.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Interaction:
        if not isinstance(data, dict):
            raise TypeError(f"Interaction must be a dict, got {type(data).__name__}")
        raw_latency = data.get("latency_ms", 0.0)
        if isinstance(raw_latency, bool):
            raise TypeError("latency_ms must be a number, got bool")
        raw_llm = data.get("llm")
        return cls(
            request=RecordedRequest.from_dict(data["request"]),
            response=RecordedResponse.from_dict(data["response"]),
            latency_ms=float(raw_latency),
            llm=LLMCallSummary.from_dict({} if raw_llm is None else raw_llm),
        )
