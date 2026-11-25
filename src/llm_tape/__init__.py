"""llm-tape: Record, replay, and assert on LLM HTTP calls in pytest."""

from llm_tape.cassette import Cassette
from llm_tape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    TapeAssertionError,
    TapeError,
    TapeMode,
    TapeNotFoundError,
    ToolCall,
    UnmatchedRequestError,
    Usage,
)
from llm_tape.pricing import DEFAULT_PRICING, ModelPrice, estimate_cost_usd
from llm_tape.session import TapeSession, tape, use_tape
from llm_tape.transport import AsyncTapeTransport, TapeTransport

__all__ = [
    "AsyncTapeTransport",
    "Cassette",
    "DEFAULT_PRICING",
    "Interaction",
    "LLMCallSummary",
    "ModelPrice",
    "RecordedRequest",
    "RecordedResponse",
    "TapeAssertionError",
    "TapeError",
    "TapeMode",
    "TapeNotFoundError",
    "TapeSession",
    "TapeTransport",
    "ToolCall",
    "UnmatchedRequestError",
    "Usage",
    "estimate_cost_usd",
    "tape",
    "use_tape",
]

__version__ = "0.1.0"
