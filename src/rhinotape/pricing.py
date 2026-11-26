"""Token pricing table and cost estimation for popular LLM models."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)
_BOUNDARY_CHARS = frozenset({"-", ".", ":", "@"})
PRICING_FILE_ENV = "RHINOTAPE_PRICING_FILE"
_pricing_cache_lock = threading.Lock()
_pricing_file_cache: dict[str, tuple[int, int, dict[str, ModelPrice]]] = {}


def _validate_price_rate(field_name: str, value: Any, *, optional: bool = False) -> None:
    if value is None:
        if optional:
            return
        raise ValueError(f"{field_name} must be a finite non-negative number, got None.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite non-negative number, got {type(value).__name__}."
        )
    fval = float(value)
    if not math.isfinite(fval) or fval < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number, got {value!r}.")


def _parse_json_price_rate(field_name: str, raw: Any, *, optional: bool = False) -> float | None:
    if raw is None:
        if optional:
            return None
        raise ValueError(f"missing required numeric field {field_name!r}")
    if isinstance(raw, bool):
        raise ValueError(f"{field_name!r} must be numeric, not bool")
    val = float(raw)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{field_name!r} must be a finite non-negative number, got {raw!r}")
    return val


@dataclass(frozen=True)
class ModelPrice:
    """Price in USD per 1 million tokens (with optional cached read/write input rates)."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None
    cache_write_per_million: float | None = None

    def __post_init__(self) -> None:
        _validate_price_rate("input_per_million", self.input_per_million)
        _validate_price_rate("output_per_million", self.output_per_million)
        _validate_price_rate(
            "cached_input_per_million", self.cached_input_per_million, optional=True
        )
        _validate_price_rate("cache_write_per_million", self.cache_write_per_million, optional=True)

    def compute(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_input_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> float:
        billable_cached = (
            min(cached_input_tokens, input_tokens)
            if cached_input_tokens > 0 and self.cached_input_per_million is not None
            else 0
        )
        remaining = max(0, input_tokens - billable_cached)
        billable_write = (
            min(cache_creation_tokens, remaining)
            if cache_creation_tokens > 0 and self.cache_write_per_million is not None
            else 0
        )
        non_cached_input = max(0, remaining - billable_write)
        cached_rate = (
            self.cached_input_per_million
            if self.cached_input_per_million is not None
            else self.input_per_million
        )
        write_rate = (
            self.cache_write_per_million
            if self.cache_write_per_million is not None
            else self.input_per_million
        )
        return (
            non_cached_input * self.input_per_million
            + billable_cached * cached_rate
            + billable_write * write_rate
            + output_tokens * self.output_per_million
        ) / 1_000_000.0


# Reference USD per 1M tokens (input, output, cached_input). Users can override via
# custom_pricing or RHINOTAPE_PRICING_FILE.
DEFAULT_PRICING: dict[str, ModelPrice] = {
    # OpenAI
    "gpt-4o": ModelPrice(2.50, 10.00, cached_input_per_million=1.25),
    "gpt-4o-mini": ModelPrice(0.15, 0.60, cached_input_per_million=0.075),
    "gpt-4.1": ModelPrice(2.00, 8.00, cached_input_per_million=0.50),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60, cached_input_per_million=0.10),
    "gpt-4.1-nano": ModelPrice(0.10, 0.40, cached_input_per_million=0.025),
    "o3": ModelPrice(2.00, 8.00, cached_input_per_million=0.50),
    "o3-mini": ModelPrice(1.10, 4.40, cached_input_per_million=0.55),
    "o4-mini": ModelPrice(1.10, 4.40, cached_input_per_million=0.275),
    "text-embedding-3-small": ModelPrice(0.02, 0.0),
    "text-embedding-3-large": ModelPrice(0.13, 0.0),
    "text-embedding-ada-002": ModelPrice(0.10, 0.0),
    # Anthropic
    "claude-3-5-sonnet": ModelPrice(
        3.00, 15.00, cached_input_per_million=0.30, cache_write_per_million=3.75
    ),
    "claude-3-5-haiku": ModelPrice(
        0.80, 4.00, cached_input_per_million=0.08, cache_write_per_million=1.00
    ),
    "claude-3-7-sonnet": ModelPrice(
        3.00, 15.00, cached_input_per_million=0.30, cache_write_per_million=3.75
    ),
    "claude-sonnet-4": ModelPrice(
        3.00, 15.00, cached_input_per_million=0.30, cache_write_per_million=3.75
    ),
    "claude-opus-4": ModelPrice(
        15.00, 75.00, cached_input_per_million=1.50, cache_write_per_million=18.75
    ),
    "claude-haiku-4": ModelPrice(
        1.00, 5.00, cached_input_per_million=0.10, cache_write_per_million=1.25
    ),
    # Google Gemini
    "gemini-2.0-flash": ModelPrice(0.10, 0.40, cached_input_per_million=0.025),
    "gemini-2.5-flash": ModelPrice(0.15, 0.60, cached_input_per_million=0.0375),
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40, cached_input_per_million=0.025),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00, cached_input_per_million=0.3125),
    # DeepSeek
    "deepseek-chat": ModelPrice(0.27, 1.10, cached_input_per_million=0.07),
    "deepseek-reasoner": ModelPrice(0.55, 2.19, cached_input_per_million=0.14),
    # Mistral
    "mistral-large": ModelPrice(2.00, 6.00),
    "mistral-small": ModelPrice(0.10, 0.30),
    "codestral": ModelPrice(0.30, 0.90),
    "ministral-8b": ModelPrice(0.10, 0.10),
    "ministral-3b": ModelPrice(0.04, 0.04),
    # Cohere
    "command-a": ModelPrice(2.50, 10.00),
    "command-r-plus": ModelPrice(2.50, 10.00),
    "command-r": ModelPrice(0.15, 0.60),
    # xAI Grok
    "grok-3": ModelPrice(3.00, 15.00, cached_input_per_million=0.75),
    "grok-3-mini": ModelPrice(0.30, 0.50, cached_input_per_million=0.075),
    "grok-2": ModelPrice(2.00, 10.00),
    # Perplexity
    "sonar": ModelPrice(1.00, 1.00),
    "sonar-pro": ModelPrice(3.00, 15.00),
    "sonar-reasoning": ModelPrice(1.00, 5.00),
    "sonar-reasoning-pro": ModelPrice(2.00, 8.00),
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
        raw_in = entry.get("input_per_million")
        raw_out = entry.get("output_per_million")
        if raw_in is None or raw_out is None:
            raise ValueError(
                f"Pricing entry for {model_name!r} must specify numeric "
                "'input_per_million' and 'output_per_million'."
            )
        cached = entry.get("cached_input_per_million")
        cache_write = entry.get("cache_write_per_million")
        try:
            in_rate = _parse_json_price_rate("input_per_million", raw_in)
            out_rate = _parse_json_price_rate("output_per_million", raw_out)
            cached_rate = _parse_json_price_rate("cached_input_per_million", cached, optional=True)
            write_rate = _parse_json_price_rate(
                "cache_write_per_million", cache_write, optional=True
            )
            assert in_rate is not None and out_rate is not None
            table[str(model_name).lower().strip()] = ModelPrice(
                input_per_million=in_rate,
                output_per_million=out_rate,
                cached_input_per_million=cached_rate,
                cache_write_per_million=write_rate,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric price for {model_name!r}: {exc}") from exc
    return table


def _matches_model_prefix(normalized: str, key: str) -> bool:
    if normalized == key:
        return True
    return normalized.startswith(key) and normalized[len(key)] in _BOUNDARY_CHARS


def _load_env_pricing_cached(env_file: str) -> dict[str, ModelPrice]:
    path = Path(env_file)
    try:
        st = path.stat()
    except OSError as exc:
        _logger.warning("Failed to access %s (%s): %s", PRICING_FILE_ENV, env_file, exc)
        return {}
    if not path.is_file():
        _logger.warning("%s path is not a regular file: %s", PRICING_FILE_ENV, env_file)
        return {}
    with _pricing_cache_lock:
        cached = _pricing_file_cache.get(env_file)
        if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            return cached[2]
    try:
        loaded = load_pricing_file(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _logger.warning("Failed to load %s (%s): %s", PRICING_FILE_ENV, env_file, exc)
        return {}
    with _pricing_cache_lock:
        _pricing_file_cache[env_file] = (st.st_mtime_ns, st.st_size, loaded)
    return loaded


def _normalize_model_key(model: str) -> tuple[str, str]:
    raw_normalized = model.lower().strip()
    normalized = raw_normalized
    if normalized.startswith("models/"):
        normalized = normalized[len("models/") :]
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    for prefix in ("anthropic.", "us.anthropic.", "eu.anthropic.", "apac.anthropic."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return raw_normalized, normalized


def _resolve_model_price(
    model: str | None,
    custom_pricing: dict[str, ModelPrice] | None = None,
) -> ModelPrice | None:
    if not model:
        return None

    env_file = os.environ.get(PRICING_FILE_ENV)
    env_table = _load_env_pricing_cached(env_file) if env_file else {}
    if env_table or custom_pricing:
        table: dict[str, ModelPrice] = dict(DEFAULT_PRICING)
        if env_table:
            table.update(env_table)
        if custom_pricing:
            table.update({k.lower().strip(): v for k, v in custom_pricing.items()})
    else:
        table = DEFAULT_PRICING

    raw_normalized, normalized = _normalize_model_key(model)
    if raw_normalized in table:
        return table[raw_normalized]
    raw_matches = [key for key in table if _matches_model_prefix(raw_normalized, key)]
    if raw_matches:
        return table[max(raw_matches, key=len)]

    if normalized in table:
        return table[normalized]

    matches = [key for key in table if _matches_model_prefix(normalized, key)]
    if matches:
        return table[max(matches, key=len)]
    return None


def estimate_cost_usd(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    custom_pricing: dict[str, ModelPrice] | None = None,
    *,
    cached_input_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Estimate USD cost for a model call using exact or boundary-safe longest-prefix match."""
    if not model or (input_tokens == 0 and output_tokens == 0):
        return 0.0
    price = _resolve_model_price(model, custom_pricing)
    if price is None:
        return 0.0
    return price.compute(
        input_tokens,
        output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )


def estimate_cache_savings_usd(
    model: str | None,
    input_tokens: int,
    output_tokens: int = 0,
    custom_pricing: dict[str, ModelPrice] | None = None,
    *,
    cached_input_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Estimate net USD saved by prompt caching compared to uncached input pricing."""
    if not model or (cached_input_tokens <= 0 and cache_creation_tokens <= 0):
        return 0.0
    price = _resolve_model_price(model, custom_pricing)
    if price is None:
        return 0.0
    uncached = price.compute(input_tokens, output_tokens)
    actual = price.compute(
        input_tokens,
        output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )
    return round(uncached - actual, 10)


# Known retired or deprecated LLM models mapped to concise migration guidance.
DEPRECATED_MODELS: dict[str, str] = {
    # OpenAI legacy / retired snapshots
    "text-davinci-003": "Retired legacy completions model; migrate to gpt-4.1-mini or gpt-4o-mini.",
    "gpt-3.5-turbo-0301": "Retired snapshot; migrate to gpt-4.1-mini or gpt-4o-mini.",
    "gpt-3.5-turbo-0613": "Retired snapshot; migrate to gpt-4.1-mini or gpt-4o-mini.",
    "gpt-3.5-turbo-16k-0613": "Retired snapshot; migrate to gpt-4.1-mini or gpt-4o-mini.",
    "gpt-4-0314": "Retired snapshot; migrate to gpt-4.1 or gpt-4o.",
    "gpt-4-0613": "Legacy snapshot; migrate to gpt-4.1 or gpt-4o.",
    "gpt-4-32k": "Retired model family; migrate to gpt-4.1 or gpt-4o.",
    "gpt-4-vision-preview": "Retired preview model; migrate to gpt-4o or gpt-4.1.",
    "gpt-4-1106-preview": "Legacy preview model; migrate to gpt-4.1 or gpt-4o.",
    "gpt-4-0125-preview": "Legacy preview model; migrate to gpt-4.1 or gpt-4o.",
    "gpt-4.5-preview": "Deprecated preview model; migrate to gpt-4.1.",
    "o1-preview": "Deprecated preview model; migrate to o3 or o4-mini.",
    "o1-mini": "Deprecated model; migrate to o3-mini or o4-mini.",
    # Anthropic retired / deprecated models
    "claude-2.0": "Retired model; migrate to claude-sonnet-4 or claude-3-7-sonnet.",
    "claude-2.1": "Retired model; migrate to claude-sonnet-4 or claude-3-7-sonnet.",
    "claude-instant-1.2": "Retired model; migrate to claude-haiku-4 or claude-3-5-haiku.",
    "claude-3-sonnet": "Retired model; migrate to claude-sonnet-4 or claude-3-7-sonnet.",
    "claude-3-opus": "Deprecated model; migrate to claude-opus-4.",
    "claude-3-5-sonnet-20240620": "Deprecated snapshot; migrate to claude-sonnet-4.",
    # Google Gemini legacy / deprecated models
    "gemini-1.0-pro": "Retired model; migrate to gemini-2.5-flash or gemini-2.5-pro.",
    "gemini-pro": "Retired legacy alias; migrate to gemini-2.5-flash or gemini-2.5-pro.",
    "gemini-1.5-pro": "Deprecated model; migrate to gemini-2.5-pro.",
    "gemini-1.5-flash": "Deprecated model; migrate to gemini-2.5-flash.",
}


def get_deprecation_notice(
    model: str | None,
    *,
    deprecated_models: dict[str, str] | None = None,
) -> str | None:
    """Return a deprecation notice if `model` matches a known deprecated/retired model."""
    if not model:
        return None
    table = (
        {k.lower().strip(): v for k, v in deprecated_models.items()}
        if deprecated_models is not None
        else DEPRECATED_MODELS
    )
    raw_norm, norm = _normalize_model_key(model)
    for candidate in (raw_norm, norm):
        if candidate in table:
            return table[candidate]
        matches = [key for key in table if _matches_model_prefix(candidate, key)]
        if matches:
            return table[max(matches, key=len)]
    return None
