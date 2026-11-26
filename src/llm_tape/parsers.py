"""Provider-aware JSON and SSE streaming parsers for OpenAI, Anthropic, and Google Gemini."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from llm_tape.models import LLMCallSummary, ToolCall, Usage
from llm_tape.pricing import ModelPrice, estimate_cost_usd
from llm_tape.scrubbing import redact_text

_GEMINI_MODEL_RE = re.compile(r"/models/([^/:?]+)")
_AZURE_DEPLOYMENT_RE = re.compile(r"/openai/deployments/([^/?]+)")
_BEDROCK_MODEL_RE = re.compile(r"/model/([^/?]+)/converse")

_GEMINI_URL_TOKENS: tuple[str, ...] = (
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
    ":generatecontent",
    ":streamgeneratecontent",
)
_COHERE_URL_TOKENS: tuple[str, ...] = ("cohere.com", "cohere.ai")
_OLLAMA_PATH_SUFFIXES: tuple[str, ...] = ("/api/chat", "/api/generate")
_OPENAI_URL_TOKENS: tuple[str, ...] = (
    "openai.com",
    "openai.azure.com",
    "openrouter.ai",
    "deepseek.com",
    "api.groq.com",
    "api.mistral.ai",
    "api.together.xyz",
    "api.fireworks.ai",
    "api.x.ai",
    "api.perplexity.ai",
    "api-inference.huggingface.co",
    "router.huggingface.co",
    "/chat/completions",
    "/fim/completions",
    "/responses",
    "/embeddings",
    "/audio/transcriptions",
    "/audio/translations",
    "/audio/speech",
    "/images/generations",
    "/images/edits",
)
_ANTHROPIC_SSE_EVENTS: frozenset[str] = frozenset(
    {"message_start", "content_block_start", "message_delta"}
)
_ANTHROPIC_SSE_TYPES: frozenset[str] = frozenset({"message_start", "content_block_delta"})
_COHERE_SSE_EVENTS: frozenset[str] = frozenset({"content-delta", "message-end", "tool-call-start"})


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else default
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _extract_model_name(*candidates: Any) -> str | None:
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _as_optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


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
    lower_path = urlsplit(lower_url).path
    if "anthropic.com" in lower_url:
        return "anthropic"
    if any(
        (tok in lower_path if tok.startswith(("/", ":")) else tok in lower_url)
        for tok in _GEMINI_URL_TOKENS
    ):
        return "gemini"
    if "/converse" in lower_path and ("bedrock" in lower_url or "/model/" in lower_path):
        return "bedrock"
    if "/api/" in lower_path and lower_path.rstrip("/").endswith(_OLLAMA_PATH_SUFFIXES):
        return "ollama"
    if any(tok in lower_url for tok in _COHERE_URL_TOKENS):
        return "cohere"
    if any(
        (tok in lower_path if tok.startswith("/") else tok in lower_url)
        for tok in _OPENAI_URL_TOKENS
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
        if event_name in _ANTHROPIC_SSE_EVENTS:
            return "anthropic"
        if event_name in _COHERE_SSE_EVENTS:
            return "cohere"
        if isinstance(payload, dict):
            ptype = payload.get("type")
            if ptype in _ANTHROPIC_SSE_TYPES:
                return "anthropic"
            if ptype in _COHERE_SSE_EVENTS:
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
        return _safe_int(details.get("cached_tokens"), 0)
    return 0


def _extract_openai_reasoning_tokens(usage_data: dict[str, Any]) -> int:
    details = usage_data.get("completion_tokens_details") or usage_data.get("output_tokens_details")
    if isinstance(details, dict):
        return _safe_int(details.get("reasoning_tokens"), 0)
    return 0


def _parse_openai_json(
    url: str,
    request_body: Any,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model = _extract_model_name(
        response_body.get("model"),
        request_body.get("model") if isinstance(request_body, dict) else None,
    )
    if not model:
        azure_match = _AZURE_DEPLOYMENT_RE.search(url)
        if azure_match:
            model = azure_match.group(1)

    usage_data = _as_dict(response_body.get("usage"))
    raw_in = usage_data.get("prompt_tokens")
    if raw_in is None:
        raw_in = usage_data.get("input_tokens")
    input_tokens = _safe_int(raw_in, 0)
    raw_out = usage_data.get("completion_tokens")
    if raw_out is None:
        raw_out = usage_data.get("output_tokens")
    output_tokens = _safe_int(raw_out, 0)
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

    for choice in _as_list(response_body.get("choices")):
        if not isinstance(choice, dict):
            continue
        msg = _as_dict(choice.get("message"))
        texts.extend(_extract_openai_text(msg.get("content")))
        rc = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(rc, str) and rc:
            thinking_parts.append(rc)
        for tc in _as_list(msg.get("tool_calls")):
            if not isinstance(tc, dict):
                continue
            fn = _as_dict(tc.get("function"))
            fn_name = fn.get("name")
            if isinstance(fn_name, str) and fn_name:
                tool_calls.append(
                    ToolCall(
                        name=fn_name,
                        arguments=_parse_json_args(fn.get("arguments")),
                        call_id=_as_optional_str(tc.get("id")),
                    )
                )

    for item in _as_list(response_body.get("output")):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call" and isinstance(item.get("name"), str) and item["name"]:
            call_id = _as_optional_str(item.get("call_id")) or _as_optional_str(item.get("id"))
            tool_calls.append(
                ToolCall(
                    name=item["name"],
                    arguments=_parse_json_args(item.get("arguments")),
                    call_id=call_id,
                )
            )
        elif item_type == "message":
            texts.extend(_extract_openai_text(item.get("content")))
        elif item_type == "reasoning":
            for part in _as_list(item.get("summary")):
                if (
                    isinstance(part, dict)
                    and part.get("type") in {"summary_text", "text"}
                    and isinstance(part.get("text"), str)
                    and part["text"]
                ):
                    thinking_parts.append(part["text"])

    if not texts and isinstance(response_body.get("text"), str) and response_body["text"]:
        texts.append(response_body["text"])
    elif not texts and isinstance(response_body.get("data"), list):
        for item in response_body["data"]:
            if isinstance(item, dict) and isinstance(item.get("revised_prompt"), str):
                texts.append(item["revised_prompt"])

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
    model = _extract_model_name(
        response_body.get("model"),
        request_body.get("model") if isinstance(request_body, dict) else None,
    )
    usage_data = _as_dict(response_body.get("usage"))
    cache_read = _safe_int(usage_data.get("cache_read_input_tokens"), 0)
    cache_write = _safe_int(usage_data.get("cache_creation_input_tokens"), 0)
    input_tokens = _safe_int(usage_data.get("input_tokens"), 0) + cache_write + cache_read
    output_tokens = _safe_int(usage_data.get("output_tokens"), 0)
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cache_read,
        cache_creation_tokens=cache_write,
    )

    tool_calls: list[ToolCall] = []
    texts: list[str] = []
    thinking_parts: list[str] = []
    for block in _as_list(response_body.get("content")):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif btype == "thinking" and isinstance(block.get("thinking"), str):
            thinking_parts.append(block["thinking"])
        elif btype == "tool_use" and isinstance(block.get("name"), str) and block["name"]:
            tool_calls.append(
                ToolCall(
                    name=block["name"],
                    arguments=_parse_json_args(block.get("input")),
                    call_id=_as_optional_str(block.get("id")),
                )
            )

    cost = estimate_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        custom_pricing,
        cached_input_tokens=usage.cached_input_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
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
    model = _extract_model_name(response_body.get("modelVersion"))
    if not model:
        match = _GEMINI_MODEL_RE.search(url)
        if match:
            model = match.group(1)

    usage_meta = _as_dict(response_body.get("usageMetadata"))
    input_tokens = _safe_int(usage_meta.get("promptTokenCount"), 0)
    cached_tokens = _safe_int(usage_meta.get("cachedContentTokenCount"), 0)
    thoughts_tokens = _safe_int(usage_meta.get("thoughtsTokenCount"), 0)
    output_tokens = _safe_int(usage_meta.get("candidatesTokenCount"), 0) + thoughts_tokens
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=thoughts_tokens,
    )

    tool_calls: list[ToolCall] = []
    texts: list[str] = []
    thinking_parts: list[str] = []
    for cand in _as_list(response_body.get("candidates")):
        if not isinstance(cand, dict):
            continue
        content = _as_dict(cand.get("content"))
        for part in _as_list(content.get("parts")):
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                if part.get("thought") is True:
                    thinking_parts.append(part["text"])
                else:
                    texts.append(part["text"])
            fn_call = part.get("functionCall")
            if (
                isinstance(fn_call, dict)
                and isinstance(fn_call.get("name"), str)
                and fn_call["name"]
            ):
                tool_calls.append(
                    ToolCall(
                        name=fn_call["name"],
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
    model = _extract_model_name(
        response_body.get("model"),
        request_body.get("model") if isinstance(request_body, dict) else None,
    )
    input_tokens = _safe_int(response_body.get("prompt_eval_count"), 0)
    output_tokens = _safe_int(response_body.get("eval_count"), 0)
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    texts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    if isinstance(response_body.get("response"), str) and response_body["response"]:
        texts.append(response_body["response"])
    msg = response_body.get("message")
    if isinstance(msg, dict):
        if isinstance(msg.get("content"), str) and msg["content"]:
            texts.append(msg["content"])
        if isinstance(msg.get("thinking"), str) and msg["thinking"]:
            thinking_parts.append(msg["thinking"])
        for tc in _as_list(msg.get("tool_calls")):
            if not isinstance(tc, dict):
                continue
            fn = _as_dict(tc.get("function"))
            fn_name = fn.get("name")
            if isinstance(fn_name, str) and fn_name:
                tool_calls.append(
                    ToolCall(
                        name=fn_name,
                        arguments=_parse_json_args(fn.get("arguments")),
                        call_id=_as_optional_str(tc.get("id")),
                    )
                )

    cost = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens, custom_pricing)
    return LLMCallSummary(
        provider="ollama",
        model=model,
        usage=usage,
        estimated_cost_usd=cost,
        tool_calls=tuple(tool_calls),
        text="\n".join(texts),
        thinking="\n".join(thinking_parts),
    )


def _parse_bedrock_json(
    url: str,
    request_body: Any,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model = _extract_model_name(
        request_body.get("modelId") if isinstance(request_body, dict) else None
    )
    if not model:
        match = _BEDROCK_MODEL_RE.search(url)
        if match:
            from urllib.parse import unquote

            model = unquote(match.group(1))

    usage_data = _as_dict(response_body.get("usage"))
    cache_read = _safe_int(usage_data.get("cacheReadInputTokens"), 0)
    cache_write = _safe_int(usage_data.get("cacheWriteInputTokens"), 0)
    input_tokens = _safe_int(usage_data.get("inputTokens"), 0) + cache_read + cache_write
    output_tokens = _safe_int(usage_data.get("outputTokens"), 0)
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cache_read,
        cache_creation_tokens=cache_write,
    )

    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    output_obj = response_body.get("output")
    msg = output_obj.get("message") if isinstance(output_obj, dict) else None
    if isinstance(msg, dict):
        for block in _as_list(msg.get("content")):
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("text"), str):
                texts.append(block["text"])
            tu = block.get("toolUse")
            if isinstance(tu, dict) and isinstance(tu.get("name"), str) and tu["name"]:
                tool_calls.append(
                    ToolCall(
                        name=tu["name"],
                        arguments=_parse_json_args(tu.get("input")),
                        call_id=_as_optional_str(tu.get("toolUseId")),
                    )
                )

    cost = estimate_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        custom_pricing,
        cached_input_tokens=usage.cached_input_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
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


def redact_stream_chunks(
    chunks: Sequence[str],
    patterns: Sequence[re.Pattern[str]],
) -> list[str]:
    """Redact regex patterns across stream chunks, re-splitting if a match spans chunk edges."""
    if not patterns:
        return list(chunks)
    per_chunk = [redact_text(c, patterns) for c in chunks]
    joined_redacted = redact_text("".join(chunks), patterns)
    if joined_redacted != "".join(per_chunk):
        return split_sse_chunks(joined_redacted)
    return per_chunk


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
    model = _extract_model_name(
        request_body.get("model") if isinstance(request_body, dict) else None
    )
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
    responses_items: dict[str, dict[str, str | None]] = {}
    responses_calls: list[ToolCall] = []

    for _ev, payload in events:
        if not isinstance(payload, dict):
            continue

        if isinstance(payload.get("model"), str) and payload["model"]:
            model = payload["model"]

        # OpenAI Responses API streaming
        ptype = payload.get("type")
        if ptype == "response.output_text.delta" and isinstance(payload.get("delta"), str):
            text_parts.append(payload["delta"])
            continue
        if ptype in {
            "response.reasoning_summary_text.delta",
            "response.reasoning.delta",
        } and isinstance(payload.get("delta"), str):
            thinking_parts.append(payload["delta"])
            continue
        if ptype == "response.output_item.added":
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                item_id = _as_optional_str(item.get("id"))
                if item_id:
                    raw_item_name = item.get("name")
                    item_call_id = _as_optional_str(item.get("call_id")) or item_id
                    responses_items[item_id] = {
                        "name": raw_item_name if isinstance(raw_item_name, str) else "",
                        "call_id": item_call_id,
                    }
            continue
        if ptype == "response.function_call_arguments.done":
            item_id = _as_optional_str(payload.get("item_id"))
            tracked = responses_items.get(item_id) if item_id else None
            raw_done_name = payload.get("name")
            tool_name = (
                raw_done_name
                if isinstance(raw_done_name, str) and raw_done_name
                else (tracked["name"] if tracked and tracked["name"] else "")
            )
            if tool_name:
                call_id = (
                    _as_optional_str(payload.get("call_id"))
                    or (tracked["call_id"] if tracked else None)
                    or item_id
                )
                responses_calls.append(
                    ToolCall(
                        name=tool_name,
                        arguments=_parse_json_args(payload.get("arguments")),
                        call_id=call_id,
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
                thinking=summary.thinking or "".join(thinking_parts),
                is_stream=True,
            )

        # OpenAI Chat Completions streaming
        usage_data = payload.get("usage")
        if isinstance(usage_data, dict):
            raw_in = usage_data.get("prompt_tokens")
            if raw_in is None:
                raw_in = usage_data.get("input_tokens")
            if raw_in is not None:
                input_tokens = _safe_int(raw_in, input_tokens)
            raw_out = usage_data.get("completion_tokens")
            if raw_out is None:
                raw_out = usage_data.get("output_tokens")
            if raw_out is not None:
                output_tokens = _safe_int(raw_out, output_tokens)
            c_toks = _extract_openai_cached_tokens(usage_data)
            if c_toks:
                cached_tokens = c_toks
            r_toks = _extract_openai_reasoning_tokens(usage_data)
            if r_toks:
                reasoning_tokens = r_toks

        for choice in _as_list(payload.get("choices")):
            if not isinstance(choice, dict):
                continue
            delta = _as_dict(choice.get("delta"))
            if isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            rc = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(rc, str) and rc:
                thinking_parts.append(rc)
            for tc in _as_list(delta.get("tool_calls")):
                if not isinstance(tc, dict):
                    continue
                idx = _safe_int(tc.get("index"), 0)
                slot = tool_acc.setdefault(idx, {"id": None, "name": "", "args": []})
                tc_id = _as_optional_str(tc.get("id"))
                if tc_id:
                    slot["id"] = tc_id
                fn = _as_dict(tc.get("function"))
                fn_name = fn.get("name")
                if isinstance(fn_name, str) and fn_name and not slot["name"]:
                    slot["name"] = fn_name
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
    model = _extract_model_name(
        request_body.get("model") if isinstance(request_body, dict) else None
    )
    base_input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    cache_write_tokens = 0
    reasoning_tokens = 0
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_acc: dict[int, dict[str, Any]] = {}

    for ev_name, payload in events:
        if not isinstance(payload, dict):
            continue
        etype = payload.get("type") or ev_name

        if etype == "message_start":
            msg = _as_dict(payload.get("message"))
            if isinstance(msg.get("model"), str) and msg["model"]:
                model = msg["model"]
            u = _as_dict(msg.get("usage"))
            if u.get("cache_read_input_tokens") is not None:
                cached_tokens = _safe_int(u.get("cache_read_input_tokens"), cached_tokens)
            if u.get("cache_creation_input_tokens") is not None:
                cache_write_tokens = _safe_int(
                    u.get("cache_creation_input_tokens"), cache_write_tokens
                )
            if u.get("input_tokens") is not None:
                base_input_tokens = _safe_int(u.get("input_tokens"), base_input_tokens)
            if u.get("output_tokens") is not None:
                output_tokens = _safe_int(u.get("output_tokens"), output_tokens)
        elif etype == "content_block_start":
            idx = _safe_int(payload.get("index"), 0)
            block = _as_dict(payload.get("content_block"))
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str) and block["text"]:
                text_parts.append(block["text"])
            elif (
                btype == "thinking" and isinstance(block.get("thinking"), str) and block["thinking"]
            ):
                thinking_parts.append(block["thinking"])
            elif btype == "tool_use":
                raw_name = block.get("name")
                initial_input = block.get("input")
                has_initial_input = (isinstance(initial_input, dict) and bool(initial_input)) or (
                    isinstance(initial_input, str) and bool(initial_input.strip())
                )
                tool_acc[idx] = {
                    "id": _as_optional_str(block.get("id")),
                    "name": raw_name if isinstance(raw_name, str) else "",
                    "args": [],
                    "initial_input": initial_input if has_initial_input else None,
                }
        elif etype == "content_block_delta":
            idx = _safe_int(payload.get("index"), 0)
            delta = _as_dict(payload.get("delta"))
            dtype = delta.get("type")
            if dtype == "text_delta" and isinstance(delta.get("text"), str):
                text_parts.append(delta["text"])
            elif dtype == "thinking_delta" and isinstance(delta.get("thinking"), str):
                thinking_parts.append(delta["thinking"])
            elif dtype == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                if idx in tool_acc:
                    tool_acc[idx]["args"].append(delta["partial_json"])
        elif etype == "message_delta":
            u = _as_dict(payload.get("usage"))
            if u.get("input_tokens") is not None:
                base_input_tokens = _safe_int(u.get("input_tokens"), base_input_tokens)
            if u.get("cache_read_input_tokens") is not None:
                cached_tokens = _safe_int(u.get("cache_read_input_tokens"), cached_tokens)
            if u.get("cache_creation_input_tokens") is not None:
                cache_write_tokens = _safe_int(
                    u.get("cache_creation_input_tokens"), cache_write_tokens
                )
            if u.get("output_tokens") is not None:
                output_tokens = _safe_int(u.get("output_tokens"), output_tokens)

    input_tokens = base_input_tokens + cache_write_tokens + cached_tokens
    tool_calls = tuple(
        ToolCall(
            name=slot["name"],
            arguments=(
                _parse_json_args("".join(slot["args"]))
                if slot["args"]
                else (
                    _parse_json_args(slot["initial_input"])
                    if slot.get("initial_input") is not None
                    else {}
                )
            ),
            call_id=slot["id"],
        )
        for _, slot in sorted(tool_acc.items())
        if slot["name"]
    )
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        cache_creation_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    cost = estimate_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        custom_pricing,
        cached_input_tokens=usage.cached_input_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
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
    candidates_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for _ev, payload in events:
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("modelVersion"), str) and payload["modelVersion"]:
            model = payload["modelVersion"]
        um = payload.get("usageMetadata")
        if isinstance(um, dict):
            if um.get("promptTokenCount") is not None:
                input_tokens = _safe_int(um.get("promptTokenCount"), input_tokens)
            thoughts = _safe_int(um.get("thoughtsTokenCount"), 0)
            if thoughts:
                reasoning_tokens = thoughts
            if um.get("candidatesTokenCount") is not None:
                candidates_tokens = _safe_int(um.get("candidatesTokenCount"), candidates_tokens)
            output_tokens = candidates_tokens + reasoning_tokens
            if um.get("cachedContentTokenCount") is not None:
                cached_tokens = _safe_int(um.get("cachedContentTokenCount"), cached_tokens)
        for cand in _as_list(payload.get("candidates")):
            if not isinstance(cand, dict):
                continue
            content = _as_dict(cand.get("content"))
            for part in _as_list(content.get("parts")):
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    if part.get("thought") is True:
                        thinking_parts.append(part["text"])
                    else:
                        text_parts.append(part["text"])
                fn_call = part.get("functionCall")
                if (
                    isinstance(fn_call, dict)
                    and isinstance(fn_call.get("name"), str)
                    and fn_call["name"]
                ):
                    tool_calls.append(
                        ToolCall(
                            name=fn_call["name"],
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
    model = _extract_model_name(
        request_body.get("model") if isinstance(request_body, dict) else None
    )
    input_tokens = 0
    output_tokens = 0
    text_parts: list[str] = []
    thinking_parts: list[str] = []
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
        if isinstance(payload.get("model"), str) and payload["model"]:
            model = payload["model"]
        if payload.get("prompt_eval_count") is not None:
            input_tokens = _safe_int(payload.get("prompt_eval_count"), input_tokens)
        if payload.get("eval_count") is not None:
            output_tokens = _safe_int(payload.get("eval_count"), output_tokens)
        if isinstance(payload.get("response"), str) and payload["response"]:
            text_parts.append(payload["response"])
        msg = payload.get("message")
        if isinstance(msg, dict):
            if isinstance(msg.get("content"), str) and msg["content"]:
                text_parts.append(msg["content"])
            if isinstance(msg.get("thinking"), str) and msg["thinking"]:
                thinking_parts.append(msg["thinking"])
            for tc in _as_list(msg.get("tool_calls")):
                if not isinstance(tc, dict):
                    continue
                fn = _as_dict(tc.get("function"))
                fn_name = fn.get("name")
                if isinstance(fn_name, str) and fn_name:
                    tool_calls.append(
                        ToolCall(
                            name=fn_name,
                            arguments=_parse_json_args(fn.get("arguments")),
                            call_id=_as_optional_str(tc.get("id")),
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
        thinking="".join(thinking_parts),
        is_stream=True,
    )


def _parse_cohere_json(
    request_body: Any,
    response_body: dict[str, Any],
    custom_pricing: dict[str, ModelPrice] | None,
) -> LLMCallSummary:
    model = _extract_model_name(
        response_body.get("model"),
        request_body.get("model") if isinstance(request_body, dict) else None,
    )
    meta = _as_dict(response_body.get("meta"))
    usage_map = _as_dict(meta.get("tokens")) or _as_dict(meta.get("billed_units"))
    input_tokens = _safe_int(usage_map.get("input_tokens"), 0)
    output_tokens = _safe_int(usage_map.get("output_tokens"), 0)
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    texts: list[str] = []
    tool_calls: list[ToolCall] = []

    msg = response_body.get("message")
    if isinstance(msg, dict):
        for block in _as_list(msg.get("content")):
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])
        for tc in _as_list(msg.get("tool_calls")):
            if not isinstance(tc, dict):
                continue
            fn = _as_dict(tc.get("function"))
            fn_name = fn.get("name")
            if isinstance(fn_name, str) and fn_name:
                tool_calls.append(
                    ToolCall(
                        name=fn_name,
                        arguments=_parse_json_args(fn.get("arguments")),
                        call_id=_as_optional_str(tc.get("id")),
                    )
                )
    elif isinstance(response_body.get("text"), str) and response_body["text"]:
        texts.append(response_body["text"])

    for tc in _as_list(response_body.get("tool_calls")):
        if isinstance(tc, dict) and isinstance(tc.get("name"), str) and tc["name"]:
            tool_calls.append(
                ToolCall(
                    name=tc["name"],
                    arguments=_parse_json_args(tc.get("parameters")),
                    call_id=_as_optional_str(tc.get("id")),
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
    model = _extract_model_name(
        request_body.get("model") if isinstance(request_body, dict) else None
    )
    input_tokens = 0
    output_tokens = 0
    text_parts: list[str] = []
    tool_acc: dict[int, dict[str, Any]] = {}

    for ev_name, payload in events:
        if not isinstance(payload, dict):
            continue
        etype = payload.get("type") or ev_name
        delta = _as_dict(payload.get("delta"))

        if etype == "content-delta":
            content = _as_dict(_as_dict(delta.get("message")).get("content"))
            if isinstance(content.get("text"), str):
                text_parts.append(content["text"])
        elif etype == "tool-call-start":
            idx = _safe_int(payload.get("index"), 0)
            tc = _as_dict(_as_dict(delta.get("message")).get("tool_calls"))
            fn = _as_dict(tc.get("function"))
            fn_name = fn.get("name")
            tool_acc[idx] = {
                "id": _as_optional_str(tc.get("id")),
                "name": fn_name if isinstance(fn_name, str) else "",
                "args": [fn["arguments"]] if isinstance(fn.get("arguments"), str) else [],
            }
        elif etype == "tool-call-delta":
            idx = _safe_int(payload.get("index"), 0)
            tc = _as_dict(_as_dict(delta.get("message")).get("tool_calls"))
            fn = _as_dict(tc.get("function"))
            if idx in tool_acc and isinstance(fn.get("arguments"), str):
                tool_acc[idx]["args"].append(fn["arguments"])
        elif etype == "message-end":
            u = _as_dict(delta.get("usage"))
            toks = _as_dict(u.get("tokens")) or _as_dict(u.get("billed_units"))
            if toks.get("input_tokens") is not None:
                input_tokens = _safe_int(toks.get("input_tokens"), input_tokens)
            if toks.get("output_tokens") is not None:
                output_tokens = _safe_int(toks.get("output_tokens"), output_tokens)

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
