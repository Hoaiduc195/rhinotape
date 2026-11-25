"""Token pricing table and cost estimation for popular LLM models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """Price in USD per 1 million tokens."""

    input_per_million: float
    output_per_million: float

    def compute(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000.0


# Reference USD per 1M tokens (input, output). Users can override via PricingTable.
DEFAULT_PRICING: dict[str, ModelPrice] = {
    # OpenAI
    "gpt-4o": ModelPrice(2.50, 10.00),
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "gpt-4.1": ModelPrice(2.00, 8.00),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60),
    "gpt-4.1-nano": ModelPrice(0.10, 0.40),
    "o3-mini": ModelPrice(1.10, 4.40),
    # Anthropic
    "claude-3-5-sonnet": ModelPrice(3.00, 15.00),
    "claude-3-5-haiku": ModelPrice(0.80, 4.00),
    "claude-3-7-sonnet": ModelPrice(3.00, 15.00),
    "claude-sonnet-4": ModelPrice(3.00, 15.00),
    "claude-opus-4": ModelPrice(15.00, 75.00),
    # Google Gemini
    "gemini-2.0-flash": ModelPrice(0.10, 0.40),
    "gemini-2.5-flash": ModelPrice(0.15, 0.60),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00),
}


def estimate_cost_usd(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    custom_pricing: dict[str, ModelPrice] | None = None,
) -> float:
    """Estimate USD cost for a model call using exact or longest-prefix match."""
    if not model or (input_tokens == 0 and output_tokens == 0):
        return 0.0

    table: dict[str, ModelPrice] = dict(DEFAULT_PRICING)
    if custom_pricing:
        table.update(custom_pricing)

    normalized = model.lower().strip()
    if normalized.startswith("models/"):
        normalized = normalized[len("models/") :]

    if normalized in table:
        return table[normalized].compute(input_tokens, output_tokens)

    # Longest prefix match (e.g., "gpt-4o-mini-2024-07-18" -> "gpt-4o-mini")
    matches = [key for key in table if normalized.startswith(key)]
    if matches:
        best = max(matches, key=len)
        return table[best].compute(input_tokens, output_tokens)

    return 0.0
