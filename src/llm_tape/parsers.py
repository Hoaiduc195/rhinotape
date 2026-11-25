"""Provider-aware JSON and SSE streaming parsers for OpenAI, Anthropic, and Google Gemini."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from llm_tape.models import LLMCallSummary, ToolCall, Usage
from llm_tape.pricing import ModelPrice, estimate_cost_usd

_GEMINI_MODEL_RE = re.compile(r"/models/([^/:?]+)")
_AZURE_DEPLOYMENT_RE = re.compile(r"/openai/deployments/([^/?]+)")
_BEDROCK_MODEL_RE = re.compile(r"/model/([^/?]+)/converse")


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
        or "aiplatform.googleapis.com" in lower_url
        or ":generatecontent" in lower_url
        or ":streamgeneratecontent" in lower_url
    ):
        return "gemini"
    if "/converse" in lower_url and ("bedrock" in lower_url or "/model/" in lower_url):
        return "bedrock"
    url_path = urlsplit(lower_url).path.rstrip("/")
    if url_path.endswith(("/api/chat", "/api/generate")):
        return "ollama"
    if "cohere.com" in lower_url or "cohere.ai" in lower_url:
        return "cohere"
    if (
        "openai.com" in lower_url
        or "openai.azure.com" in lower_url
        or "openrouter.ai" in lower_url
        or "deepseek.com" in lower_url
        or "api.groq.com" in lower_url
        or "api.mistral.ai" in lower_url
        or "api.together.xyz" in lower_url
        or "api.fireworks.ai" in lower_url
        or "api.x.ai" in lower_url
        or "/chat/completions" in lower_url
        or "/fim/completions" in lower_url
        or "/responses" in lower_url
        or "/embeddings" in lower_url
    ):
        return "openai"

    if isinstance(response_body, dict):
        if response_body.get("type") == "message":
            return "anthropic"
        if "candidates" in response_body:
            return "gemini"
        if isinstance(response_body.get("output"), dict) and "message" in response_body["output"]:
            return "bedrock"
        if isinstance(response_body.get("meta"), dict) and (
            "billed_units" in response_body["meta"] or "tokens" in response_body["meta"]
        ):
            return "cohere"
        if "prompt_eval_count" in response_body or (
            "done" in response_body and ("message" in response_body or "response" in response_body)
        ):
            return "ollama"
        if (
            "choices" in response_body
            or "output" in response_body
            or (
                response_body.get("object") == "list"
                and isinstance(response_body.get("data"), list)
                and bool(response_body["data"])
                and isinstance(response_body["data"][0], dict)
                and response_body["data"][0].get("object") == "embedding"
            )
        ):
            return "openai"

    for event_name, payload in sse_events:
        if event_name in {"message_start", "content_block_start", "message_delta"}:
            return "anthropic"
        if event_name in {"content-delta", "message-end", "tool-call-start"}:
            return "cohere"
        if isinstance(payload, dict):
            ptype = payload.get("type")
            if ptype in {"message_start", "content_block_delta"}:
                return "anthropic"
            if ptype in {"content-delta", "message-end", "tool-call-start"}:
                return "cohere"
            if "candidates" in payload:
                return "gemini"
            if "choices" in payload or str(ptype or "").startswith("response."):
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


def _extract_openai_cached_tokens(usage_data: dict[str, Any]) -> int:
    details = usage_data.get("prompt_tokens_details") or usage_data.get("input_tokens_details")
    if isinstance(details, dict):
        return int(details.get("cached_tokens", 0) or 0)
    return 0


def _extract_openai_reasoning_tokens(usage_data: dict[str, Any]) -> int:
    details = usage_data.get("completion_tokens_details") or usage_data.get("output_tokens_details")
    if isinstance(details, dict):
        return int(details.get("reasoning_tokens", 0) or 0)
    return 0


def _parse_openai_json(
    url: str,
    request_body: Any,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model = response_body.get("model") or (
        request_body.get("model") if isinstance(request_body, dict) else None
    )
    if not model:
        azure_match = _AZURE_DEPLOYMENT_RE.search(url)
        if azure_match:
            model = azure_match.group(1)

    usage_data = response_body.get("usage") or {}
    input_tokens = int(usage_data.get("prompt_tokens", usage_data.get("input_tokens", 0)) or 0)
    output_tokens = int(
        usage_data.get("completion_tokens", usage_data.get("output_tokens", 0)) or 0
    )
    cached_tokens = _extract_openai_cached_tokens(usage_data)
    reasoning_tokens = _extract_openai_reasoning_tokens(usage_data)
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )

    tool_calls: list[ToolCall] = []
    texts: list[str] = []
    thinking_parts: list[str] = []

    for choice in response_body.get("choices", []) or []:
        msg = choice.get("message") or {}
        texts.extend(_extract_openai_text(msg.get("content")))
        rc = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(rc, str) and rc:
            thinking_parts.append(rc)
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

    cost = estimate_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        custom_pricing,
        cached_input_tokens=usage.cached_input_tokens,
    )
    return LLMCallSummary(
        provider="openai",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="\n".join(texts),
        thinking="\n".join(thinking_parts),
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
    cache_read = int(usage_data.get("cache_read_input_tokens", 0) or 0)
    input_tokens = (
        int(usage_data.get("input_tokens", 0) or 0)
        + int(usage_data.get("cache_creation_input_tokens", 0) or 0)
        + cache_read
    )
    output_tokens = int(usage_data.get("output_tokens", 0) or 0)
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cache_read,
    )

    tool_calls: list[ToolCall] = []
    texts: list[str] = []
    thinking_parts: list[str] = []
    for block in response_body.get("content") or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif btype == "thinking" and isinstance(block.get("thinking"), str):
            thinking_parts.append(block["thinking"])
        elif btype == "tool_use" and block.get("name"):
            tool_calls.append(
                ToolCall(
                    name=str(block["name"]),
                    arguments=_parse_json_args(block.get("input")),
                    call_id=block.get("id"),
                )
            )

    cost = estimate_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        custom_pricing,
        cached_input_tokens=usage.cached_input_tokens,
    )
    return LLMCallSummary(
        provider="anthropic",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="\n".join(texts),
        thinking="\n".join(thinking_parts),
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
    cached_tokens = int(usage_meta.get("cachedContentTokenCount", 0) or 0)
    thoughts_tokens = int(usage_meta.get("thoughtsTokenCount", 0) or 0)
    output_tokens = int(usage_meta.get("candidatesTokenCount", 0) or 0) + thoughts_tokens
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=thoughts_tokens,
    )

    tool_calls: list[ToolCall] = []
    texts: list[str] = []
    thinking_parts: list[str] = []
    for cand in response_body.get("candidates") or []:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                if part.get("thought") is True:
                    thinking_parts.append(part["text"])
                else:
                    texts.append(part["text"])
            fn_call = part.get("functionCall")
            if isinstance(fn_call, dict) and fn_call.get("name"):
                tool_calls.append(
                    ToolCall(
                        name=str(fn_call["name"]),
                        arguments=_parse_json_args(fn_call.get("args")),
                    )
                )

    cost = estimate_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        custom_pricing,
        cached_input_tokens=usage.cached_input_tokens,
    )
    return LLMCallSummary(
        provider="gemini",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="\n".join(texts),
        thinking="\n".join(thinking_parts),
    )


def _parse_ollama_json(
    request_body: Any,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model = response_body.get("model") or (
        request_body.get("model") if isinstance(request_body, dict) else None
    )
    input_tokens = int(response_body.get("prompt_eval_count", 0) or 0)
    output_tokens = int(response_body.get("eval_count", 0) or 0)
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    if isinstance(response_body.get("response"), str) and response_body["response"]:
        texts.append(response_body["response"])
    msg = response_body.get("message")
    if isinstance(msg, dict):
        if isinstance(msg.get("content"), str) and msg["content"]:
            texts.append(msg["content"])
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

    cost = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens, custom_pricing)
    return LLMCallSummary(
        provider="ollama",
        model=str(model) if model else None,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="\n".join(texts),
    )


def _parse_bedrock_json(
    url: str,
    request_body: Any,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model: str | None = request_body.get("modelId") if isinstance(request_body, dict) else None
    if not model:
        match = _BEDROCK_MODEL_RE.search(url)
        if match:
            from urllib.parse import unquote

            model = unquote(match.group(1))

    usage_data = response_body.get("usage") or {}
    cache_read = int(usage_data.get("cacheReadInputTokens", 0) or 0)
    cache_write = int(usage_data.get("cacheWriteInputTokens", 0) or 0)
    input_tokens = int(usage_data.get("inputTokens", 0) or 0) + cache_read + cache_write
    output_tokens = int(usage_data.get("outputTokens", 0) or 0)
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cache_read,
    )

    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    output_obj = response_body.get("output")
    msg = output_obj.get("message") if isinstance(output_obj, dict) else None
    if isinstance(msg, dict):
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("text"), str):
                texts.append(block["text"])
            tu = block.get("toolUse")
            if isinstance(tu, dict) and tu.get("name"):
                tool_calls.append(
                    ToolCall(
                        name=str(tu["name"]),
                        arguments=_parse_json_args(tu.get("input")),
                        call_id=tu.get("toolUseId"),
                    )
                )

    cost = estimate_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        custom_pricing,
        cached_input_tokens=usage.cached_input_tokens,
    )
    return LLMCallSummary(
        provider="bedrock",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="\n".join(texts),
    )


# --- SSE Streaming Parsers -------------------------------------------------


_SSE_LINE_SPLIT_RE = re.compile(r"\r\n|\n|\r")
_SSE_EVENT_BOUNDARY_RE = re.compile(r"(\r\n\r\n|\n\n)")


def split_sse_chunks(text: str) -> list[str]:
    """Split raw SSE text into event-delimited chunks, preserving LF or CRLF separators."""
    if not text:
        return []
    tokens = _SSE_EVENT_BOUNDARY_RE.split(text)
    chunks: list[str] = []
    idx = 0
    while idx < len(tokens):
        body = tokens[idx]
        sep = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        piece = body + sep
        if piece:
            chunks.append(piece)
        idx += 2
    return chunks


def _strip_one_leading_space(value: str) -> str:
    return value[1:] if value.startswith(" ") else value


def parse_sse_events(raw_sse: str) -> list[tuple[str | None, Any]]:
    """Parse SSE text into a list of (event_name, parsed_json_or_str) tuples per WHATWG SSE spec."""
    if raw_sse.startswith("\ufeff"):
        raw_sse = raw_sse[1:]

    events: list[tuple[str | None, Any]] = []
    current_event: str | None = None
    data_lines: list[str] = []

    def _dispatch() -> None:
        nonlocal current_event, data_lines
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

    for line in _SSE_LINE_SPLIT_RE.split(raw_sse):
        if line == "":
            _dispatch()
            continue
        if line.startswith(":"):
            continue
        if ":" in line:
            field_name, raw_val = line.split(":", 1)
            value = _strip_one_leading_space(raw_val)
        else:
            field_name = line
            value = ""

        if field_name == "event":
            current_event = value or None
        elif field_name == "data":
            data_lines.append(value)

    _dispatch()
    return events


def _parse_openai_sse(
    url: str,
    request_body: Any,
    events: Sequence[tuple[str | None, Any]],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model: str | None = request_body.get("model") if isinstance(request_body, dict) else None
    if not model:
        azure_match = _AZURE_DEPLOYMENT_RE.search(url)
        if azure_match:
            model = azure_match.group(1)
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    text_parts: list[str] = []
    thinking_parts: list[str] = []
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
            summary = _parse_openai_json(url, request_body, payload["response"], custom_pricing)
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
            c_toks = _extract_openai_cached_tokens(usage_data)
            if c_toks:
                cached_tokens = c_toks
            r_toks = _extract_openai_reasoning_tokens(usage_data)
            if r_toks:
                reasoning_tokens = r_toks

        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            rc = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(rc, str) and rc:
                thinking_parts.append(rc)
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
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    cost = estimate_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        custom_pricing,
        cached_input_tokens=usage.cached_input_tokens,
    )
    return LLMCallSummary(
        provider="openai",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tool_calls,
        text="".join(text_parts),
        thinking="".join(thinking_parts),
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
    cached_tokens = 0
    reasoning_tokens = 0
    text_parts: list[str] = []
    thinking_parts: list[str] = []
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
            cached_tokens = int(u.get("cache_read_input_tokens", 0) or 0)
            input_tokens = (
                int(u.get("input_tokens", 0) or 0)
                + int(u.get("cache_creation_input_tokens", 0) or 0)
                + cached_tokens
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
            elif dtype == "thinking_delta" and isinstance(delta.get("thinking"), str):
                thinking_parts.append(delta["thinking"])
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
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    cost = estimate_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        custom_pricing,
        cached_input_tokens=usage.cached_input_tokens,
    )
    return LLMCallSummary(
        provider="anthropic",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tool_calls,
        text="".join(text_parts),
        thinking="".join(thinking_parts),
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
    cached_tokens = 0
    reasoning_tokens = 0
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for _ev, payload in events:
        if not isinstance(payload, dict):
            continue
        if payload.get("modelVersion"):
            model = str(payload["modelVersion"])
        um = payload.get("usageMetadata")
        if isinstance(um, dict):
            input_tokens = int(um.get("promptTokenCount", input_tokens) or 0)
            thoughts = int(um.get("thoughtsTokenCount", 0) or 0)
            if thoughts:
                reasoning_tokens = thoughts
            output_tokens = int(um.get("candidatesTokenCount", 0) or 0) + thoughts
            if "cachedContentTokenCount" in um:
                cached_tokens = int(um.get("cachedContentTokenCount", 0) or 0)
        for cand in payload.get("candidates") or []:
            content = cand.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    if part.get("thought") is True:
                        thinking_parts.append(part["text"])
                    else:
                        text_parts.append(part["text"])
                fn_call = part.get("functionCall")
                if isinstance(fn_call, dict) and fn_call.get("name"):
                    tool_calls.append(
                        ToolCall(
                            name=str(fn_call["name"]),
                            arguments=_parse_json_args(fn_call.get("args")),
                        )
                    )

    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    cost = estimate_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        custom_pricing,
        cached_input_tokens=usage.cached_input_tokens,
    )
    return LLMCallSummary(
        provider="gemini",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="".join(text_parts),
        thinking="".join(thinking_parts),
        is_stream=True,
    )


def _parse_ollama_ndjson(
    request_body: Any,
    raw_stream: str,
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model: str | None = request_body.get("model") if isinstance(request_body, dict) else None
    input_tokens = 0
    output_tokens = 0
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for line in raw_stream.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("model"):
            model = str(payload["model"])
        if "prompt_eval_count" in payload:
            input_tokens = int(payload.get("prompt_eval_count", 0) or 0)
        if "eval_count" in payload:
            output_tokens = int(payload.get("eval_count", 0) or 0)
        if isinstance(payload.get("response"), str) and payload["response"]:
            text_parts.append(payload["response"])
        msg = payload.get("message")
        if isinstance(msg, dict):
            if isinstance(msg.get("content"), str) and msg["content"]:
                text_parts.append(msg["content"])
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

    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    cost = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens, custom_pricing)
    return LLMCallSummary(
        provider="ollama",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="".join(text_parts),
        is_stream=True,
    )


def _parse_cohere_json(
    request_body: Any,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model = response_body.get("model") or (
        request_body.get("model") if isinstance(request_body, dict) else None
    )
    meta = response_body.get("meta") or {}
    usage_map = meta.get("tokens") or meta.get("billed_units") or {}
    input_tokens = int(usage_map.get("input_tokens", 0) or 0)
    output_tokens = int(usage_map.get("output_tokens", 0) or 0)
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    texts: list[str] = []
    tool_calls: list[ToolCall] = []

    msg = response_body.get("message")
    if isinstance(msg, dict):
        for block in msg.get("content") or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if fn.get("name"):
                tool_calls.append(
                    ToolCall(
                        name=str(fn["name"]),
                        arguments=_parse_json_args(fn.get("arguments")),
                        call_id=tc.get("id"),
                    )
                )
    elif isinstance(response_body.get("text"), str) and response_body["text"]:
        texts.append(response_body["text"])

    for tc in response_body.get("tool_calls") or []:
        if isinstance(tc, dict) and tc.get("name"):
            tool_calls.append(
                ToolCall(
                    name=str(tc["name"]),
                    arguments=_parse_json_args(tc.get("parameters")),
                    call_id=tc.get("id"),
                )
            )

    cost = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens, custom_pricing)
    return LLMCallSummary(
        provider="cohere",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="\n".join(texts),
    )


def _parse_cohere_sse(
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
        delta = payload.get("delta") or {}

        if etype == "content-delta":
            content = (delta.get("message") or {}).get("content") or {}
            if isinstance(content.get("text"), str):
                text_parts.append(content["text"])
        elif etype == "tool-call-start":
            idx = int(payload.get("index", 0))
            tc = (delta.get("message") or {}).get("tool_calls") or {}
            fn = tc.get("function") or {}
            tool_acc[idx] = {
                "id": tc.get("id"),
                "name": str(fn.get("name", "")),
                "args": [str(fn["arguments"])] if isinstance(fn.get("arguments"), str) else [],
            }
        elif etype == "tool-call-delta":
            idx = int(payload.get("index", 0))
            tc = (delta.get("message") or {}).get("tool_calls") or {}
            fn = tc.get("function") or {}
            if idx in tool_acc and isinstance(fn.get("arguments"), str):
                tool_acc[idx]["args"].append(fn["arguments"])
        elif etype == "message-end":
            u = delta.get("usage") or {}
            toks = u.get("tokens") or u.get("billed_units") or {}
            input_tokens = int(toks.get("input_tokens", input_tokens) or 0)
            output_tokens = int(toks.get("output_tokens", output_tokens) or 0)

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
        provider="cohere",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tool_calls,
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
    """Extract normalized LLM metadata from a JSON, SSE, or NDJSON streaming HTTP interaction."""
    lower_ct = content_type.lower()
    is_ndjson = "application/x-ndjson" in lower_ct or "application/ndjson" in lower_ct
    is_stream = stream_chunks is not None or "text/event-stream" in lower_ct or is_ndjson
    if is_stream:
        raw_stream = (
            "".join(stream_chunks) if stream_chunks is not None else str(response_body or "")
        )
        if _detect_provider(url, None) == "ollama" or is_ndjson:
            return _parse_ollama_ndjson(request_body, raw_stream, custom_pricing)
        events = parse_sse_events(raw_stream)
        provider = _detect_provider(url, None, events)
        if provider == "openai":
            return _parse_openai_sse(url, request_body, events, custom_pricing)
        if provider == "anthropic":
            return _parse_anthropic_sse(request_body, events, custom_pricing)
        if provider == "gemini":
            return _parse_gemini_sse(url, events, custom_pricing)
        if provider == "cohere":
            return _parse_cohere_sse(request_body, events, custom_pricing)
        return LLMCallSummary(provider="unknown", is_stream=True)

    if not isinstance(response_body, dict):
        return LLMCallSummary(provider=_detect_provider(url, response_body))

    provider = _detect_provider(url, response_body)
    if provider == "openai":
        return _parse_openai_json(url, request_body, response_body, custom_pricing)
    if provider == "anthropic":
        return _parse_anthropic_json(request_body, response_body, custom_pricing)
    if provider == "gemini":
        return _parse_gemini_json(url, response_body, custom_pricing)
    if provider == "ollama":
        return _parse_ollama_json(request_body, response_body, custom_pricing)
    if provider == "bedrock":
        return _parse_bedrock_json(url, request_body, response_body, custom_pricing)
    if provider == "cohere":
        return _parse_cohere_json(request_body, response_body, custom_pricing)
    return LLMCallSummary(provider="unknown")
