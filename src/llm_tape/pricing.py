"""Token pricing table and cost estimation for popular LLM models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BOUNDARY_CHARS = frozenset({"-", ".", ":", "@"})
PRICING_FILE_ENV = "LLM_TAPE_PRICING_FILE"


@dataclass(frozen=True)
class ModelPrice:
    """Price in USD per 1 million tokens (with optional cached input token rate)."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None

    def compute(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_input_tokens: int = 0,
    ) -> float:
        if cached_input_tokens > 0 and self.cached_input_per_million is not None:
            billable_cached = min(cached_input_tokens, input_tokens)
            non_cached_input = max(0, input_tokens - billable_cached)
            return (
                non_cached_input * self.input_per_million
                + billable_cached * self.cached_input_per_million
                + output_tokens * self.output_per_million
            ) / 1_000_000.0
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000.0


# Reference USD per 1M tokens (input, output, cached_input). Users can override via
# custom_pricing or LLM_TAPE_PRICING_FILE.
DEFAULT_PRICING: dict[str, ModelPrice] = {
    # OpenAI
    "gpt-4o": ModelPrice(2.50, 10.00, cached_input_per_million=1.25),
    "gpt-4o-mini": ModelPrice(0.15, 0.60, cached_input_per_million=0.075),
    "gpt-4.1": ModelPrice(2.00, 8.00, cached_input_per_million=0.50),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60, cached_input_per_million=0.10),
    "gpt-4.1-nano": ModelPrice(0.10, 0.40, cached_input_per_million=0.025),
    "o3-mini": ModelPrice(1.10, 4.40, cached_input_per_million=0.55),
    # Anthropic
    "claude-3-5-sonnet": ModelPrice(3.00, 15.00, cached_input_per_million=0.30),
    "claude-3-5-haiku": ModelPrice(0.80, 4.00, cached_input_per_million=0.08),
    "claude-3-7-sonnet": ModelPrice(3.00, 15.00, cached_input_per_million=0.30),
    "claude-sonnet-4": ModelPrice(3.00, 15.00, cached_input_per_million=0.30),
    "claude-opus-4": ModelPrice(15.00, 75.00, cached_input_per_million=1.50),
    # Google Gemini
    "gemini-2.0-flash": ModelPrice(0.10, 0.40, cached_input_per_million=0.025),
    "gemini-2.5-flash": ModelPrice(0.15, 0.60, cached_input_per_million=0.0375),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00, cached_input_per_million=0.3125),
}


def load_pricing_file(path: str | Path) -> dict[str, ModelPrice]:
    """Load custom model pricing overrides from a JSON file.

    Expected JSON structure:
        {
          "my-model": {
            "input_per_million": 1.0,
            "output_per_million": 2.0,
            "cached_input_per_million": 0.25
          }
        }
    """
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Pricing file {path} must contain a JSON object at top level.")
    table: dict[str, ModelPrice] = {}
    for model_name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Pricing entry for {model_name!r} must be a JSON object.")
        cached = entry.get("cached_input_per_million")
        table[str(model_name).lower().strip()] = ModelPrice(
            input_per_million=float(entry["input_per_million"]),
            output_per_million=float(entry["output_per_million"]),
            cached_input_per_million=float(cached) if cached is not None else None,
        )
    return table


def _matches_model_prefix(normalized: str, key: str) -> bool:
    if normalized == key:
        return True
    return normalized.startswith(key) and normalized[len(key)] in _BOUNDARY_CHARS


def estimate_cost_usd(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    custom_pricing: dict[str, ModelPrice] | None = None,
    *,
    cached_input_tokens: int = 0,
) -> float:
    """Estimate USD cost for a model call using exact or boundary-safe longest-prefix match."""
    if not model or (input_tokens == 0 and output_tokens == 0):
        return 0.0

    table: dict[str, ModelPrice] = dict(DEFAULT_PRICING)
    env_file = os.environ.get(PRICING_FILE_ENV)
    if env_file and Path(env_file).is_file():
        try:
            table.update(load_pricing_file(env_file))
        except (OSError, ValueError, KeyError):
            pass
    if custom_pricing:
        table.update({k.lower().strip(): v for k, v in custom_pricing.items()})

    normalized = model.lower().strip()
    if normalized.startswith("models/"):
        normalized = normalized[len("models/") :]
    # Strip provider prefixes used by OpenRouter / Bedrock (e.g. "openai/gpt-4o-mini",
    # "anthropic.claude-3-5-sonnet-20241022-v2:0")
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    for prefix in ("anthropic.", "us.anthropic.", "eu.anthropic.", "apac.anthropic."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    if normalized in table:
        return table[normalized].compute(
            input_tokens, output_tokens, cached_input_tokens=cached_input_tokens
        )

    # Longest boundary-safe prefix match (e.g., "gpt-4o-mini-2024-07-18" -> "gpt-4o-mini",
    # while preventing "gpt-4" from matching "gpt-4o").
    matches = [key for key in table if _matches_model_prefix(normalized, key)]
    if matches:
        best = max(matches, key=len)
        return table[best].compute(
            input_tokens, output_tokens, cached_input_tokens=cached_input_tokens
        )

    return 0.0
