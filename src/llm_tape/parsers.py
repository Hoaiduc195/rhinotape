"""Provider-aware parsers for OpenAI, Anthropic, and Google Gemini payloads."""

from __future__ import annotations

import json
import re
from typing import Any

from llm_tape.models import LLMCallSummary, ToolCall, Usage
from llm_tape.pricing import ModelPrice, estimate_cost_usd

_GEMINI_MODEL_RE = re.compile(r"/models/([^/:?]+)")


def _parse_json_args(raw: Any) -> dict[str, Any] | str:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else raw
        except json.JSONDecodeError:
            return raw
    return {}


def _detect_provider(url: str, response_body: Any) -> str:
    lower_url = url.lower()
    if "anthropic.com" in lower_url or (
        isinstance(response_body, dict) and response_body.get("type") == "message"
    ):
        return "anthropic"
    if (
        "generativelanguage.googleapis.com" in lower_url
        or ":generatecontent" in lower_url
        or (isinstance(response_body, dict) and "candidates" in response_body)
    ):
        return "gemini"
    if (
        "openai.com" in lower_url
        or "/chat/completions" in lower_url
        or "/responses" in lower_url
        or (isinstance(response_body, dict) and "choices" in response_body)
    ):
        return "openai"
    return "unknown"


def _parse_openai(
    request_body: Any,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model = response_body.get("model") or (
        request_body.get("model") if isinstance(request_body, dict) else None
    )
    usage_data = response_body.get("usage") or {}
    input_tokens = int(
        usage_data.get("prompt_tokens", usage_data.get("input_tokens", 0)) or 0
    )
    output_tokens = int(
        usage_data.get("completion_tokens", usage_data.get("output_tokens", 0)) or 0
    )
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    tool_calls: list[ToolCall] = []
    texts: list[str] = []

    # Chat Completions format
    for choice in response_body.get("choices", []) or []:
        msg = choice.get("message") or {}
        if isinstance(msg.get("content"), str) and msg["content"]:
            texts.append(msg["content"])
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name"):
                tool_calls.append(
                    ToolCall(
                        name=fn["name"],
                        arguments=_parse_json_args(fn.get("arguments")),
                        call_id=tc.get("id"),
                    )
                )

    # Responses API format
    for item in response_body.get("output", []) or []:
        item_type = item.get("type")
        if item_type == "function_call" and item.get("name"):
            tool_calls.append(
                ToolCall(
                    name=item["name"],
                    arguments=_parse_json_args(item.get("arguments")),
                    call_id=item.get("call_id") or item.get("id"),
                )
            )
        elif item_type == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])

    cost = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens, custom_pricing)
    return LLMCallSummary(
        provider="openai",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="\n".join(texts),
    )


def _parse_anthropic(
    request_body: Any,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model = response_body.get("model") or (
        request_body.get("model") if isinstance(request_body, dict) else None
    )
    usage_data = response_body.get("usage") or {}
    usage = Usage(
        input_tokens=int(usage_data.get("input_tokens", 0) or 0),
        output_tokens=int(usage_data.get("output_tokens", 0) or 0),
    )

    tool_calls: list[ToolCall] = []
    texts: list[str] = []
    for block in response_body.get("content") or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif btype == "tool_use" and block.get("name"):
            tool_calls.append(
                ToolCall(
                    name=block["name"],
                    arguments=_parse_json_args(block.get("input")),
                    call_id=block.get("id"),
                )
            )

    cost = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens, custom_pricing)
    return LLMCallSummary(
        provider="anthropic",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="\n".join(texts),
    )


def _parse_gemini(
    url: str,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model = response_body.get("modelVersion")
    if not model:
        match = _GEMINI_MODEL_RE.search(url)
        if match:
            model = match.group(1)

    usage_meta = response_body.get("usageMetadata") or {}
    usage = Usage(
        input_tokens=int(usage_meta.get("promptTokenCount", 0) or 0),
        output_tokens=int(usage_meta.get("candidatesTokenCount", 0) or 0),
    )

    tool_calls: list[ToolCall] = []
    texts: list[str] = []
    for cand in response_body.get("candidates") or []:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
            fn_call = part.get("functionCall")
            if isinstance(fn_call, dict) and fn_call.get("name"):
                tool_calls.append(
                    ToolCall(
                        name=fn_call["name"],
                        arguments=_parse_json_args(fn_call.get("args")),
                    )
                )

    cost = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens, custom_pricing)
    return LLMCallSummary(
        provider="gemini",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="\n".join(texts),
    )


def parse_llm_interaction(
    url: str,
    request_body: Any,
    response_body: Any,
    custom_pricing: dict[str, ModelPrice] | None = None,
) -> LLMCallSummary:
    """Extract normalized LLM metadata from an HTTP interaction."""
    if not isinstance(response_body, dict):
        return LLMCallSummary(provider=_detect_provider(url, response_body))

    provider = _detect_provider(url, response_body)
    if provider == "openai":
        return _parse_openai(request_body, response_body, custom_pricing)
    if provider == "anthropic":
        return _parse_anthropic(request_body, response_body, custom_pricing)
    if provider == "gemini":
        return _parse_gemini(url, response_body, custom_pricing)
    return LLMCallSummary(provider="unknown")
