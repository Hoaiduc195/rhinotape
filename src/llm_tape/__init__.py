"""llm-tape: Record, replay, and assert on LLM HTTP & SSE calls in pytest."""

from llm_tape._version import __version__
from llm_tape.cassette import Cassette
from llm_tape.models import (
    CorruptCassetteError,
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
from llm_tape.pricing import DEFAULT_PRICING, ModelPrice, estimate_cost_usd, load_pricing_file
from llm_tape.scrubbing import DEFAULT_MATCH_ON, VALID_MATCH_ON, redact_body, redact_text
from llm_tape.server import TapeReplayServer
from llm_tape.session import TapeSession, replace_response, tape, use_tape
from llm_tape.transport import AsyncTapeTransport, TapeTransport

__all__ = [
    "AsyncTapeTransport",
    "Cassette",
    "CorruptCassetteError",
    "DEFAULT_MATCH_ON",
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
    "TapeReplayServer",
    "TapeSession",
    "TapeTransport",
    "ToolCall",
    "UnmatchedRequestError",
    "Usage",
    "VALID_MATCH_ON",
    "__version__",
    "estimate_cost_usd",
    "load_pricing_file",
    "redact_body",
    "redact_text",
    "replace_response",
    "tape",
    "use_tape",
]
