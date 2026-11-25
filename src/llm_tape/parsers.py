"""Provider-aware JSON and SSE streaming parsers for OpenAI, Anthropic, and Google Gemini."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from llm_tape.models import LLMCallSummary, ToolCall, Usage
from llm_tape.pricing import ModelPrice, estimate_cost_usd

_GEMINI_MODEL_RE = re.compile(r"/models/([^/:?]+)")


def _parse_json_args(raw: Any) -> dict[str, Any] | str:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else raw
        except json.JSONDecodeError:
            return raw
    return {}


def _detect_provider(
    url: str,
    response_body: Any,
    sse_events: Sequence[tuple[str | None, Any]] = (),
) -> str:
    lower_url = url.lower()
    if "anthropic.com" in lower_url:
        return "anthropic"
    if (
        "generativelanguage.googleapis.com" in lower_url
        or ":generatecontent" in lower_url
        or ":streamgeneratecontent" in lower_url
    ):
        return "gemini"
    if "openai.com" in lower_url or "/chat/completions" in lower_url or "/responses" in lower_url:
        return "openai"

    if isinstance(response_body, dict):
        if response_body.get("type") == "message":
            return "anthropic"
        if "candidates" in response_body:
            return "gemini"
        if "choices" in response_body or "output" in response_body:
            return "openai"

    for event_name, payload in sse_events:
        if event_name in {"message_start", "content_block_start", "message_delta"}:
            return "anthropic"
        if isinstance(payload, dict):
            if payload.get("type") in {"message_start", "content_block_delta"}:
                return "anthropic"
            if "candidates" in payload:
                return "gemini"
            if "choices" in payload or str(payload.get("type", "")).startswith("response."):
                return "openai"

    return "unknown"


def _extract_openai_text(content: Any) -> list[str]:
    if isinstance(content, str) and content:
        return [content]
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"]:
                parts.append(item["text"])
        return parts
    return []


def _parse_openai_json(
    request_body: Any,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model = response_body.get("model") or (
        request_body.get("model") if isinstance(request_body, dict) else None
    )
    usage_data = response_body.get("usage") or {}
    input_tokens = int(usage_data.get("prompt_tokens", usage_data.get("input_tokens", 0)) or 0)
    output_tokens = int(
        usage_data.get("completion_tokens", usage_data.get("output_tokens", 0)) or 0
    )
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    tool_calls: list[ToolCall] = []
    texts: list[str] = []

    for choice in response_body.get("choices", []) or []:
        msg = choice.get("message") or {}
        texts.extend(_extract_openai_text(msg.get("content")))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name"):
                tool_calls.append(
                    ToolCall(
                        name=str(fn["name"]),
                        arguments=_parse_json_args(fn.get("arguments")),
                        call_id=tc.get("id"),
                    )
                )

    for item in response_body.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call" and item.get("name"):
            tool_calls.append(
                ToolCall(
                    name=str(item["name"]),
                    arguments=_parse_json_args(item.get("arguments")),
                    call_id=item.get("call_id") or item.get("id"),
                )
            )
        elif item_type == "message":
            texts.extend(_extract_openai_text(item.get("content")))

    cost = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens, custom_pricing)
    return LLMCallSummary(
        provider="openai",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="\n".join(texts),
    )


def _parse_anthropic_json(
    request_body: Any,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model = response_body.get("model") or (
        request_body.get("model") if isinstance(request_body, dict) else None
    )
    usage_data = response_body.get("usage") or {}
    input_tokens = (
        int(usage_data.get("input_tokens", 0) or 0)
        + int(usage_data.get("cache_creation_input_tokens", 0) or 0)
        + int(usage_data.get("cache_read_input_tokens", 0) or 0)
    )
    output_tokens = int(usage_data.get("output_tokens", 0) or 0)
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

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
                    name=str(block["name"]),
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


def _parse_gemini_json(
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
    input_tokens = int(usage_meta.get("promptTokenCount", 0) or 0)
    output_tokens = int(usage_meta.get("candidatesTokenCount", 0) or 0) + int(
        usage_meta.get("thoughtsTokenCount", 0) or 0
    )
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

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
                        name=str(fn_call["name"]),
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


# --- SSE Streaming Parsers -------------------------------------------------


_SSE_LINE_SPLIT_RE = re.compile(r"\r\n|\n|\r")


def parse_sse_events(raw_sse: str) -> list[tuple[str | None, Any]]:
    """Parse SSE text into a list of (event_name, parsed_json_or_str) tuples per WHATWG SSE spec."""
    events: list[tuple[str | None, Any]] = []
    current_event: str | None = None
    data_lines: list[str] = []

    for line in _SSE_LINE_SPLIT_RE.split(raw_sse):
        if line == "":
            if data_lines:
                raw_data = "\n".join(data_lines)
                if raw_data.strip() != "[DONE]":
                    try:
                        parsed: Any = json.loads(raw_data)
                    except json.JSONDecodeError:
                        parsed = raw_data
                    events.append((current_event, parsed))
                data_lines = []
            current_event = None
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip(" "))

    if data_lines:
        raw_data = "\n".join(data_lines)
        if raw_data.strip() != "[DONE]":
            try:
                parsed = json.loads(raw_data)
            except json.JSONDecodeError:
                parsed = raw_data
            events.append((current_event, parsed))

    return events


def _parse_openai_sse(
    request_body: Any,
    events: Sequence[tuple[str | None, Any]],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model: str | None = request_body.get("model") if isinstance(request_body, dict) else None
    input_tokens = 0
    output_tokens = 0
    text_parts: list[str] = []
    # index -> {"id": str|None, "name": str, "args": list[str]}
    tool_acc: dict[int, dict[str, Any]] = {}
    responses_calls: list[ToolCall] = []

    for _ev, payload in events:
        if not isinstance(payload, dict):
            continue

        if payload.get("model"):
            model = str(payload["model"])

        # OpenAI Responses API streaming
        ptype = payload.get("type")
        if ptype == "response.output_text.delta" and isinstance(payload.get("delta"), str):
            text_parts.append(payload["delta"])
            continue
        if ptype == "response.function_call_arguments.done" and payload.get("name"):
            responses_calls.append(
                ToolCall(
                    name=str(payload["name"]),
                    arguments=_parse_json_args(payload.get("arguments")),
                    call_id=payload.get("call_id") or payload.get("item_id"),
                )
            )
            continue
        if ptype == "response.completed" and isinstance(payload.get("response"), dict):
            summary = _parse_openai_json(request_body, payload["response"], custom_pricing)
            return LLMCallSummary(
                provider="openai",
                model=summary.model or model,
                usage=summary.usage,
                estimated_cost_usd=summary.estimated_cost_usd,
                tool_calls=summary.tool_calls or tuple(responses_calls),
                text=summary.text or "".join(text_parts),
                is_stream=True,
            )

        # OpenAI Chat Completions streaming
        usage_data = payload.get("usage")
        if isinstance(usage_data, dict):
            input_tokens = int(
                usage_data.get("prompt_tokens", usage_data.get("input_tokens", input_tokens)) or 0
            )
            output_tokens = int(
                usage_data.get("completion_tokens", usage_data.get("output_tokens", output_tokens))
                or 0
            )

        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", 0))
                slot = tool_acc.setdefault(idx, {"id": None, "name": "", "args": []})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name") and not slot["name"]:
                    slot["name"] = str(fn["name"])
                if isinstance(fn.get("arguments"), str):
                    slot["args"].append(fn["arguments"])

    tool_calls = tuple(
        ToolCall(
            name=slot["name"],
            arguments=_parse_json_args("".join(slot["args"])),
            call_id=slot["id"],
        )
        for _, slot in sorted(tool_acc.items())
        if slot["name"]
    ) or tuple(responses_calls)
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    cost = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens, custom_pricing)
    return LLMCallSummary(
        provider="openai",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tool_calls,
        text="".join(text_parts),
        is_stream=True,
    )


def _parse_anthropic_sse(
    request_body: Any,
    events: Sequence[tuple[str | None, Any]],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model: str | None = request_body.get("model") if isinstance(request_body, dict) else None
    input_tokens = 0
    output_tokens = 0
    text_parts: list[str] = []
    tool_acc: dict[int, dict[str, Any]] = {}

    for ev_name, payload in events:
        if not isinstance(payload, dict):
            continue
        etype = payload.get("type") or ev_name

        if etype == "message_start":
            msg = payload.get("message") or {}
            if msg.get("model"):
                model = str(msg["model"])
            u = msg.get("usage") or {}
            input_tokens = (
                int(u.get("input_tokens", 0) or 0)
                + int(u.get("cache_creation_input_tokens", 0) or 0)
                + int(u.get("cache_read_input_tokens", 0) or 0)
            )
            output_tokens = int(u.get("output_tokens", output_tokens) or 0)
        elif etype == "content_block_start":
            idx = int(payload.get("index", 0))
            block = payload.get("content_block") or {}
            if block.get("type") == "tool_use":
                tool_acc[idx] = {
                    "id": block.get("id"),
                    "name": str(block.get("name", "")),
                    "args": [],
                }
        elif etype == "content_block_delta":
            idx = int(payload.get("index", 0))
            delta = payload.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta" and isinstance(delta.get("text"), str):
                text_parts.append(delta["text"])
            elif dtype == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                if idx in tool_acc:
                    tool_acc[idx]["args"].append(delta["partial_json"])
        elif etype == "message_delta":
            u = payload.get("usage") or {}
            if "output_tokens" in u:
                output_tokens = int(u.get("output_tokens", 0) or 0)

    tool_calls = tuple(
        ToolCall(
            name=slot["name"],
            arguments=_parse_json_args("".join(slot["args"])),
            call_id=slot["id"],
        )
        for _, slot in sorted(tool_acc.items())
        if slot["name"]
    )
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    cost = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens, custom_pricing)
    return LLMCallSummary(
        provider="anthropic",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tool_calls,
        text="".join(text_parts),
        is_stream=True,
    )


def _parse_gemini_sse(
    url: str,
    events: Sequence[tuple[str | None, Any]],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model: str | None = None
    match = _GEMINI_MODEL_RE.search(url)
    if match:
        model = match.group(1)

    input_tokens = 0
    output_tokens = 0
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for _ev, payload in events:
        if not isinstance(payload, dict):
            continue
        if payload.get("modelVersion"):
            model = str(payload["modelVersion"])
        um = payload.get("usageMetadata")
        if isinstance(um, dict):
            input_tokens = int(um.get("promptTokenCount", input_tokens) or 0)
            output_tokens = int(um.get("candidatesTokenCount", 0) or 0) + int(
                um.get("thoughtsTokenCount", 0) or 0
            )
        for cand in payload.get("candidates") or []:
            content = cand.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                fn_call = part.get("functionCall")
                if isinstance(fn_call, dict) and fn_call.get("name"):
                    tool_calls.append(
                        ToolCall(
                            name=str(fn_call["name"]),
                            arguments=_parse_json_args(fn_call.get("args")),
                        )
                    )

    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    cost = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens, custom_pricing)
    return LLMCallSummary(
        provider="gemini",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="".join(text_parts),
        is_stream=True,
    )


def parse_llm_interaction(
    url: str,
    request_body: Any,
    response_body: Any,
    custom_pricing: dict[str, ModelPrice] | None = None,
    *,
    stream_chunks: Sequence[str] | None = None,
    content_type: str = "",
) -> LLMCallSummary:
    """Extract normalized LLM metadata from a JSON or SSE streaming HTTP interaction."""
    is_sse = stream_chunks is not None or "text/event-stream" in content_type.lower()
    if is_sse:
        raw_sse = "".join(stream_chunks) if stream_chunks is not None else str(response_body or "")
        events = parse_sse_events(raw_sse)
        provider = _detect_provider(url, None, events)
        if provider == "openai":
            return _parse_openai_sse(request_body, events, custom_pricing)
        if provider == "anthropic":
            return _parse_anthropic_sse(request_body, events, custom_pricing)
        if provider == "gemini":
            return _parse_gemini_sse(url, events, custom_pricing)
        return LLMCallSummary(provider="unknown", is_stream=True)

    if not isinstance(response_body, dict):
        return LLMCallSummary(provider=_detect_provider(url, response_body))

    provider = _detect_provider(url, response_body)
    if provider == "openai":
        return _parse_openai_json(request_body, response_body, custom_pricing)
    if provider == "anthropic":
        return _parse_anthropic_json(request_body, response_body, custom_pricing)
    if provider == "gemini":
        return _parse_gemini_json(url, response_body, custom_pricing)
    return LLMCallSummary(provider="unknown")
