"""Core data models and exceptions for llm-tape."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TapeMode(str, Enum):
    """Controls how llm-tape records or replays HTTP interactions."""

    ONCE = "once"  # Replay if cassette exists; otherwise record and save.
    REPLAY = "replay"  # Strict offline replay; fail if cassette or interaction missing.
    RECORD = "record"  # Always hit live network and overwrite cassette.
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
    """Base exception for llm-tape."""


class TapeNotFoundError(TapeError):
    """Raised in replay mode when the expected cassette file does not exist."""


class CorruptCassetteError(TapeError):
    """Raised when a cassette file cannot be parsed or has an unsupported schema."""


class UnmatchedRequestError(TapeError):
    """Raised when an outgoing request does not match any recorded interaction."""


class TapeAssertionError(AssertionError, TapeError):
    """Raised when a tape assertion (budget, tokens, tool calls) fails."""


@dataclass(frozen=True)
class ToolCall:
    """Normalized representation of a single LLM tool/function call."""

    name: str
    arguments: dict[str, Any] | str = field(default_factory=dict)
    call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            name=str(data["name"]),
            arguments=data.get("arguments", {}),
            call_id=data.get("call_id"),
        )


@dataclass(frozen=True)
class Usage:
    """Normalized token usage for an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
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
        if self.reasoning_tokens:
            payload["reasoning_tokens"] = self.reasoning_tokens
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Usage:
        return cls(
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cached_input_tokens=int(data.get("cached_input_tokens", 0)),
            reasoning_tokens=int(data.get("reasoning_tokens", 0)),
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
    is_stream: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "text": self.text,
            "is_stream": self.is_stream,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMCallSummary:
        return cls(
            provider=str(data.get("provider", "unknown")),
            model=data.get("model"),
            usage=Usage.from_dict(data.get("usage", {})),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0)),
            tool_calls=tuple(ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])),
            text=str(data.get("text", "")),
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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordedRequest:
        body = data.get("body")
        default_enc = (
            "empty" if body is None else ("json" if isinstance(body, (dict, list)) else "text")
        )
        return cls(
            method=str(data["method"]),
            url=str(data["url"]),
            headers=dict(data.get("headers", {})),
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
        body = data.get("body")
        raw_chunks = data.get("stream_chunks")
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
            status_code=int(data["status_code"]),
            headers=dict(data.get("headers", {})),
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
        return cls(
            request=RecordedRequest.from_dict(data["request"]),
            response=RecordedResponse.from_dict(data["response"]),
            latency_ms=float(data.get("latency_ms", 0.0)),
            llm=LLMCallSummary.from_dict(data.get("llm", {})),
        )
