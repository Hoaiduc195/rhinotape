"""Fluent synthetic cassette builder for OpenAI, Anthropic, Gemini, and HTTP error mocks."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from rhinotape.cassette import Cassette
from rhinotape.models import Interaction, RecordedRequest, RecordedResponse
from rhinotape.parsers import parse_llm_interaction
from rhinotape.scrubbing import (
    decode_body_with_encoding,
    fingerprint_request,
    scrub_headers,
    scrub_url,
)

__all__ = ["CassetteBuilder"]


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def _require_non_neg_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _require_status_code(value: object, *, min_code: int = 100) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (min_code <= value <= 599):
        raise ValueError(f"status_code must be an integer in {min_code}..599, got {value!r}")
    return value


def _require_latency(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"latency_ms must be a finite number >= 0.0, got {value!r}")
    num = float(value)
    if not math.isfinite(num) or num < 0.0:
        raise ValueError(f"latency_ms must be a finite number >= 0.0, got {value!r}")
    return num


def _require_non_empty_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value.strip()


def _require_http_url(value: object, name: str = "url") -> str:
    cleaned = _require_non_empty_str(value, name)
    parts = urlsplit(cleaned)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"{name} must be a valid http(s) URL, got {value!r}")
    return cleaned


def _normalize_optional_text(value: object, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str or None, got {type(value).__name__}")
    return value


def _validate_extra_body(extra_request_body: object) -> dict[str, Any]:
    if extra_request_body is None:
        return {}
    if not isinstance(extra_request_body, dict):
        raise ValueError(
            f"extra_request_body must be a dict or None, got {type(extra_request_body).__name__}"
        )
    return copy.deepcopy(extra_request_body)


def _normalize_messages(prompt: object) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, Sequence) and not isinstance(prompt, (bytes, bytearray)):
        items = list(prompt)
        if not items or not all(isinstance(m, dict) for m in items):
            raise ValueError("prompt sequence must be a non-empty sequence of message dicts")
        return copy.deepcopy(items)
    raise ValueError(f"prompt must be a str or non-empty sequence of dicts, got {type(prompt)!r}")


def _extract_raw_tool_args(item: dict[str, Any]) -> Any:
    for key in ("arguments", "input", "args"):
        if key in item:
            return item[key]
    return {}


def _normalize_tool_calls(
    tool_calls: Sequence[dict[str, Any]] | None,
    *,
    id_prefix: str,
) -> list[tuple[str, str, dict[str, Any] | str, str]]:
    if tool_calls is None:
        return []
    if isinstance(tool_calls, (str, bytes, bytearray, dict)) or not isinstance(
        tool_calls, Sequence
    ):
        raise ValueError(
            f"tool_calls must be a sequence of dicts or None, got {type(tool_calls).__name__}"
        )
    normalized: list[tuple[str, str, dict[str, Any] | str, str]] = []
    for idx, item in enumerate(tool_calls, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"tool_calls[{idx - 1}] must be a dict, got {type(item).__name__}")
        name = _require_non_empty_str(item.get("name"), f"tool_calls[{idx - 1}]['name']")
        raw_id = item["id"] if "id" in item else item.get("call_id")
        if raw_id is None:
            call_id = f"{id_prefix}_{idx}"
        elif isinstance(raw_id, str):
            call_id = raw_id.strip() or f"{id_prefix}_{idx}"
        elif isinstance(raw_id, int) and not isinstance(raw_id, bool):
            call_id = str(raw_id)
        else:
            raise ValueError(
                f"tool_calls[{idx - 1}]['id'] must be a str, int, or None, "
                f"got {type(raw_id).__name__}"
            )
        raw_args = _extract_raw_tool_args(item)
        if isinstance(raw_args, dict):
            args_val: dict[str, Any] | str = copy.deepcopy(raw_args)
            args_json = json.dumps(args_val, ensure_ascii=False)
        elif isinstance(raw_args, str):
            args_json = raw_args
            stripped = raw_args.strip()
            if not stripped:
                args_val = {}
            else:
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    args_val = raw_args
                else:
                    args_val = parsed if isinstance(parsed, dict) else raw_args
        else:
            raise ValueError(
                f"tool_calls[{idx - 1}]['arguments'] must be a dict or JSON str, "
                f"got {type(raw_args).__name__}"
            )
        normalized.append((call_id, name, args_val, args_json))
    return normalized


def _split_text_for_stream(text: str) -> list[str]:
    if not text:
        return []
    if len(text) < 4:
        return [text]
    mid = len(text) // 2
    return [text[:mid], text[mid:]]


def _validate_path(path: str | Path) -> Path:
    if isinstance(path, str):
        _require_non_empty_str(path, "path")
        return Path(path)
    if isinstance(path, Path):
        if not str(path).strip():
            raise ValueError("path must not be empty")
        return path
    raise ValueError(f"path must be a str or Path, got {type(path).__name__}")


def _write_bytes_atomic(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.stem}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class CassetteBuilder:
    """Fluent builder for constructing authentic synthetic LLM cassettes for offline tests."""

    def __init__(
        self,
        path: str | Path = "synthetic.tape.json",
        *,
        recorded_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._path = _validate_path(path)
        if recorded_at is not None and not isinstance(recorded_at, str):
            raise ValueError(f"recorded_at must be a str or None, got {type(recorded_at).__name__}")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"metadata must be a dict or None, got {type(metadata).__name__}")
        self._recorded_at = (
            recorded_at if recorded_at is not None else datetime.now(timezone.utc).isoformat()
        )
        self._metadata: dict[str, Any] = copy.deepcopy(metadata) if metadata is not None else {}
        self._interactions: list[Interaction] = []

    def _append_interaction(
        self,
        *,
        method: str,
        url: str,
        req_headers: dict[str, str],
        req_body: Any,
        req_encoding: str,
        status_code: int,
        resp_headers: dict[str, str],
        resp_body: Any,
        resp_encoding: str,
        stream_chunks: tuple[str, ...] | None,
        latency_ms: float,
    ) -> CassetteBuilder:
        scrubbed_url = scrub_url(url)
        clean_req_headers = scrub_headers(req_headers)
        clean_resp_headers = scrub_headers(resp_headers)
        fp = fingerprint_request(
            method,
            scrubbed_url,
            req_body,
            headers=clean_req_headers,
        )
        ct = next(
            (v for k, v in clean_resp_headers.items() if k.lower() == "content-type"),
            "",
        )
        llm = parse_llm_interaction(
            url=scrubbed_url,
            request_body=req_body,
            response_body=resp_body,
            stream_chunks=stream_chunks,
            content_type=ct,
        )
        self._interactions.append(
            Interaction(
                request=RecordedRequest(
                    method=method,
                    url=scrubbed_url,
                    headers=clean_req_headers,
                    body=req_body,
                    fingerprint=fp,
                    body_encoding=req_encoding,
                ),
                response=RecordedResponse(
                    status_code=status_code,
                    headers=clean_resp_headers,
                    body=resp_body,
                    body_encoding=resp_encoding,
                    stream_chunks=stream_chunks,
                ),
                latency_ms=latency_ms,
                llm=llm,
            )
        )
        return self

    def add_openai_chat(
        self,
        prompt: str | Sequence[dict[str, Any]],
        *,
        reply: str | None = "",
        model: str = "gpt-4o",
        tool_calls: Sequence[dict[str, Any]] | None = None,
        thinking: str | None = None,
        input_tokens: int = 10,
        output_tokens: int = 5,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
        stream: bool = False,
        status_code: int = 200,
        latency_ms: float = 25.0,
        url: str = "https://api.openai.com/v1/chat/completions",
        extra_request_body: dict[str, Any] | None = None,
    ) -> CassetteBuilder:
        """Append a synthetic OpenAI Chat Completions interaction (JSON or SSE stream)."""
        messages = _normalize_messages(prompt)
        reply_text = _normalize_optional_text(reply, "reply")
        thinking_text = _normalize_optional_text(thinking, "thinking")
        model_name = _require_non_empty_str(model, "model")
        target_url = _require_http_url(url, "url")
        in_tok = _require_non_neg_int(input_tokens, "input_tokens")
        out_tok = _require_non_neg_int(output_tokens, "output_tokens")
        cached_tok = _require_non_neg_int(cached_input_tokens, "cached_input_tokens")
        reason_tok = _require_non_neg_int(reasoning_tokens, "reasoning_tokens")
        is_stream = _require_bool(stream, "stream")
        code = _require_status_code(status_code)
        lat = _require_latency(latency_ms)
        extra = _validate_extra_body(extra_request_body)
        seq = len(self._interactions) + 1
        norm_tools = _normalize_tool_calls(tool_calls, id_prefix=f"call_synth_{seq}")

        req_body: dict[str, Any] = {"model": model_name, "messages": messages, **extra}
        if is_stream:
            req_body["stream"] = True

        if code >= 400:
            err_body = {
                "error": {
                    "message": reply_text or f"Synthetic OpenAI HTTP {code} error",
                    "type": "api_error",
                    "code": code,
                }
            }
            return self._append_interaction(
                method="POST",
                url=target_url,
                req_headers={"content-type": "application/json"},
                req_body=req_body,
                req_encoding="json",
                status_code=code,
                resp_headers={"content-type": "application/json"},
                resp_body=err_body,
                resp_encoding="json",
                stream_chunks=None,
                latency_ms=lat,
            )

        eff_in = max(in_tok, cached_tok)
        eff_out = max(out_tok, reason_tok)
        usage_obj: dict[str, Any] = {
            "prompt_tokens": eff_in,
            "completion_tokens": eff_out,
            "total_tokens": eff_in + eff_out,
        }
        if cached_tok > 0:
            usage_obj["prompt_tokens_details"] = {"cached_tokens": cached_tok}
        if reason_tok > 0:
            usage_obj["completion_tokens_details"] = {"reasoning_tokens": reason_tok}

        cmpl_id = f"chatcmpl-synth-{seq}"
        finish_reason = "tool_calls" if norm_tools else "stop"

        if is_stream:
            chunks: list[str] = []

            def _sse(choices: list[dict[str, Any]], usage: dict[str, Any] | None = None) -> str:
                obj: dict[str, Any] = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model_name,
                    "choices": choices,
                }
                if usage is not None:
                    obj["usage"] = usage
                return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

            chunks.append(
                _sse([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])
            )
            if thinking_text:
                chunks.append(
                    _sse(
                        [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": thinking_text},
                                "finish_reason": None,
                            }
                        ]
                    )
                )
            for piece in _split_text_for_stream(reply_text):
                chunks.append(
                    _sse([{"index": 0, "delta": {"content": piece}, "finish_reason": None}])
                )
            for tc_idx, (call_id, fn_name, _args_val, args_json) in enumerate(norm_tools):
                chunks.append(
                    _sse(
                        [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": tc_idx,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {"name": fn_name, "arguments": args_json},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    )
                )
            chunks.append(_sse([{"index": 0, "delta": {}, "finish_reason": finish_reason}]))
            chunks.append(_sse([], usage=usage_obj))
            chunks.append("data: [DONE]\n\n")
            return self._append_interaction(
                method="POST",
                url=target_url,
                req_headers={"content-type": "application/json"},
                req_body=req_body,
                req_encoding="json",
                status_code=code,
                resp_headers={"content-type": "text/event-stream"},
                resp_body=None,
                resp_encoding="stream",
                stream_chunks=tuple(chunks),
                latency_ms=lat,
            )

        msg: dict[str, Any] = {
            "role": "assistant",
            "content": reply_text if (reply_text or not norm_tools) else None,
        }
        if thinking_text:
            msg["reasoning_content"] = thinking_text
        if norm_tools:
            msg["tool_calls"] = [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": fn_name, "arguments": args_json},
                }
                for call_id, fn_name, _args_val, args_json in norm_tools
            ]
        resp_body = {
            "id": cmpl_id,
            "object": "chat.completion",
            "created": 1700000000,
            "model": model_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
            "usage": usage_obj,
        }
        return self._append_interaction(
            method="POST",
            url=target_url,
            req_headers={"content-type": "application/json"},
            req_body=req_body,
            req_encoding="json",
            status_code=code,
            resp_headers={"content-type": "application/json"},
            resp_body=resp_body,
            resp_encoding="json",
            stream_chunks=None,
            latency_ms=lat,
        )

    def add_anthropic_message(
        self,
        prompt: str | Sequence[dict[str, Any]],
        *,
        reply: str | None = "",
        model: str = "claude-sonnet-4",
        system: str | None = None,
        tool_calls: Sequence[dict[str, Any]] | None = None,
        thinking: str | None = None,
        input_tokens: int = 10,
        output_tokens: int = 5,
        cached_input_tokens: int = 0,
        cache_creation_tokens: int = 0,
        stream: bool = False,
        status_code: int = 200,
        latency_ms: float = 25.0,
        url: str = "https://api.anthropic.com/v1/messages",
        extra_request_body: dict[str, Any] | None = None,
    ) -> CassetteBuilder:
        """Append a synthetic Anthropic Messages API interaction (JSON or SSE stream)."""
        messages = _normalize_messages(prompt)
        reply_text = _normalize_optional_text(reply, "reply")
        thinking_text = _normalize_optional_text(thinking, "thinking")
        if system is not None and not isinstance(system, str):
            raise ValueError(f"system must be a str or None, got {type(system).__name__}")
        model_name = _require_non_empty_str(model, "model")
        target_url = _require_http_url(url, "url")
        in_tok = _require_non_neg_int(input_tokens, "input_tokens")
        out_tok = _require_non_neg_int(output_tokens, "output_tokens")
        cached_tok = _require_non_neg_int(cached_input_tokens, "cached_input_tokens")
        create_tok = _require_non_neg_int(cache_creation_tokens, "cache_creation_tokens")
        is_stream = _require_bool(stream, "stream")
        code = _require_status_code(status_code)
        lat = _require_latency(latency_ms)
        extra = _validate_extra_body(extra_request_body)
        seq = len(self._interactions) + 1
        norm_tools = _normalize_tool_calls(tool_calls, id_prefix=f"toolu_synth_{seq}")

        req_body: dict[str, Any] = {"model": model_name, "messages": messages}
        if system is not None:
            req_body["system"] = system
        req_body.update(extra)
        if is_stream:
            req_body["stream"] = True

        if code >= 400:
            err_body = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": reply_text or f"Synthetic Anthropic HTTP {code} error",
                },
            }
            return self._append_interaction(
                method="POST",
                url=target_url,
                req_headers={"content-type": "application/json"},
                req_body=req_body,
                req_encoding="json",
                status_code=code,
                resp_headers={"content-type": "application/json"},
                resp_body=err_body,
                resp_encoding="json",
                stream_chunks=None,
                latency_ms=lat,
            )

        base_uncached = max(0, in_tok - cached_tok - create_tok)
        usage_obj: dict[str, Any] = {
            "input_tokens": base_uncached,
            "output_tokens": out_tok,
        }
        if cached_tok > 0:
            usage_obj["cache_read_input_tokens"] = cached_tok
        if create_tok > 0:
            usage_obj["cache_creation_input_tokens"] = create_tok

        blocks: list[dict[str, Any]] = []
        if thinking_text:
            blocks.append({"type": "thinking", "thinking": thinking_text})
        if reply_text or (not norm_tools and not thinking_text):
            blocks.append({"type": "text", "text": reply_text})
        for call_id, fn_name, args_val, _args_json in norm_tools:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": fn_name,
                    "input": args_val,
                }
            )

        msg_id = f"msg_synth_{seq}"
        stop_reason = "tool_use" if norm_tools else "end_turn"

        if is_stream:
            chunks: list[str] = []

            def _ev(event_name: str, payload: dict[str, Any]) -> str:
                return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

            start_usage = {**usage_obj, "output_tokens": 0}
            chunks.append(
                _ev(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "model": model_name,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": start_usage,
                        },
                    },
                )
            )
            for idx, block in enumerate(blocks):
                btype = block["type"]
                if btype == "thinking":
                    chunks.append(
                        _ev(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": {"type": "thinking", "thinking": ""},
                            },
                        )
                    )
                    chunks.append(
                        _ev(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "thinking_delta", "thinking": block["thinking"]},
                            },
                        )
                    )
                elif btype == "text":
                    chunks.append(
                        _ev(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    )
                    for piece in _split_text_for_stream(block["text"]):
                        chunks.append(
                            _ev(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": idx,
                                    "delta": {"type": "text_delta", "text": piece},
                                },
                            )
                        )
                elif btype == "tool_use":
                    raw_input = block["input"]
                    partial_json = (
                        raw_input
                        if isinstance(raw_input, str)
                        else json.dumps(raw_input, ensure_ascii=False)
                    )
                    chunks.append(
                        _ev(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": block["id"],
                                    "name": block["name"],
                                    "input": {},
                                },
                            },
                        )
                    )
                    chunks.append(
                        _ev(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": partial_json,
                                },
                            },
                        )
                    )
                chunks.append(
                    _ev("content_block_stop", {"type": "content_block_stop", "index": idx})
                )

            chunks.append(
                _ev(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": {"output_tokens": out_tok},
                    },
                )
            )
            chunks.append(_ev("message_stop", {"type": "message_stop"}))
            return self._append_interaction(
                method="POST",
                url=target_url,
                req_headers={"content-type": "application/json"},
                req_body=req_body,
                req_encoding="json",
                status_code=code,
                resp_headers={"content-type": "text/event-stream"},
                resp_body=None,
                resp_encoding="stream",
                stream_chunks=tuple(chunks),
                latency_ms=lat,
            )

        resp_body = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage_obj,
        }
        return self._append_interaction(
            method="POST",
            url=target_url,
            req_headers={"content-type": "application/json"},
            req_body=req_body,
            req_encoding="json",
            status_code=code,
            resp_headers={"content-type": "application/json"},
            resp_body=resp_body,
            resp_encoding="json",
            stream_chunks=None,
            latency_ms=lat,
        )

    def add_gemini_generate(
        self,
        prompt: str,
        *,
        reply: str | None = "",
        model: str = "gemini-2.5-flash",
        tool_calls: Sequence[dict[str, Any]] | None = None,
        thinking: str | None = None,
        input_tokens: int = 10,
        output_tokens: int = 5,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
        stream: bool = False,
        status_code: int = 200,
        latency_ms: float = 25.0,
        url: str | None = None,
        extra_request_body: dict[str, Any] | None = None,
    ) -> CassetteBuilder:
        """Append a synthetic Google Gemini generateContent interaction (JSON or SSE stream)."""
        if not isinstance(prompt, str):
            raise ValueError(f"prompt must be a str, got {type(prompt).__name__}")
        reply_text = _normalize_optional_text(reply, "reply")
        thinking_text = _normalize_optional_text(thinking, "thinking")
        model_name = _require_non_empty_str(model, "model")
        in_tok = _require_non_neg_int(input_tokens, "input_tokens")
        out_tok = _require_non_neg_int(output_tokens, "output_tokens")
        cached_tok = _require_non_neg_int(cached_input_tokens, "cached_input_tokens")
        reason_tok = _require_non_neg_int(reasoning_tokens, "reasoning_tokens")
        is_stream = _require_bool(stream, "stream")
        code = _require_status_code(status_code)
        lat = _require_latency(latency_ms)
        extra = _validate_extra_body(extra_request_body)
        seq = len(self._interactions) + 1
        norm_tools = _normalize_tool_calls(tool_calls, id_prefix=f"gem_call_{seq}")

        if url is None:
            model_slug = (
                model_name[len("models/") :] if model_name.startswith("models/") else model_name
            )
            if not model_slug:
                raise ValueError(f"model must specify a valid model id, got {model!r}")
            safe_slug = quote(model_slug, safe="-._~")
            endpoint = "streamGenerateContent?alt=sse" if is_stream else "generateContent"
            target_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{safe_slug}:{endpoint}"
            )
        else:
            target_url = _require_http_url(url, "url")

        req_body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            **extra,
        }

        if code >= 400:
            err_body = {
                "error": {
                    "code": code,
                    "message": reply_text or f"Synthetic Gemini HTTP {code} error",
                    "status": "INTERNAL" if code >= 500 else "INVALID_ARGUMENT",
                }
            }
            return self._append_interaction(
                method="POST",
                url=target_url,
                req_headers={"content-type": "application/json"},
                req_body=req_body,
                req_encoding="json",
                status_code=code,
                resp_headers={"content-type": "application/json"},
                resp_body=err_body,
                resp_encoding="json",
                stream_chunks=None,
                latency_ms=lat,
            )

        eff_in = max(in_tok, cached_tok)
        eff_out = max(out_tok, reason_tok)
        cand_tok = eff_out - reason_tok
        usage_meta: dict[str, Any] = {
            "promptTokenCount": eff_in,
            "candidatesTokenCount": cand_tok,
            "totalTokenCount": eff_in + eff_out,
        }
        if cached_tok > 0:
            usage_meta["cachedContentTokenCount"] = cached_tok
        if reason_tok > 0:
            usage_meta["thoughtsTokenCount"] = reason_tok

        thought_parts: list[dict[str, Any]] = (
            [{"text": thinking_text, "thought": True}] if thinking_text else []
        )
        fn_parts: list[dict[str, Any]] = [
            {"functionCall": {"name": fn_name, "args": args_val}}
            for _call_id, fn_name, args_val, _args_json in norm_tools
        ]

        if is_stream:
            chunks: list[str] = []
            pieces = _split_text_for_stream(reply_text)
            if thought_parts:
                thought_payload = {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": thought_parts},
                            "index": 0,
                        }
                    ],
                    "modelVersion": model_name,
                }
                chunks.append(f"data: {json.dumps(thought_payload, ensure_ascii=False)}\n\n")
            if len(pieces) > 1:
                first_payload = {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": pieces[0]}]},
                            "index": 0,
                        }
                    ],
                    "modelVersion": model_name,
                }
                chunks.append(f"data: {json.dumps(first_payload, ensure_ascii=False)}\n\n")
                tail_parts: list[dict[str, Any]] = [{"text": pieces[1]}, *fn_parts]
            else:
                tail_parts = (
                    ([{"text": pieces[0]}] if pieces else []) + fn_parts
                    if (pieces or fn_parts or thought_parts)
                    else [{"text": ""}]
                )
            final_payload = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": tail_parts},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": usage_meta,
                "modelVersion": model_name,
            }
            chunks.append(f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n")
            return self._append_interaction(
                method="POST",
                url=target_url,
                req_headers={"content-type": "application/json"},
                req_body=req_body,
                req_encoding="json",
                status_code=code,
                resp_headers={"content-type": "text/event-stream"},
                resp_body=None,
                resp_encoding="stream",
                stream_chunks=tuple(chunks),
                latency_ms=lat,
            )

        parts: list[dict[str, Any]] = [*thought_parts]
        if reply_text or (not fn_parts and not thought_parts):
            parts.append({"text": reply_text})
        parts.extend(fn_parts)
        resp_body = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": parts},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": usage_meta,
            "modelVersion": model_name,
        }
        return self._append_interaction(
            method="POST",
            url=target_url,
            req_headers={"content-type": "application/json"},
            req_body=req_body,
            req_encoding="json",
            status_code=code,
            resp_headers={"content-type": "application/json"},
            resp_body=resp_body,
            resp_encoding="json",
            stream_chunks=None,
            latency_ms=lat,
        )

    def add_error(
        self,
        url: str,
        *,
        status_code: int = 429,
        message: str = "Rate limit exceeded",
        request_body: Any = None,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        latency_ms: float = 15.0,
    ) -> CassetteBuilder:
        """Append a synthetic HTTP error interaction (e.g. 429 rate-limit or 500 server error)."""
        target_url = _require_http_url(url, "url")
        method_norm = _require_non_empty_str(method, "method").upper()
        code = _require_status_code(status_code, min_code=400)
        lat = _require_latency(latency_ms)
        if not isinstance(message, str):
            raise ValueError(f"message must be a str, got {type(message).__name__}")
        if headers is not None and not isinstance(headers, dict):
            raise ValueError(f"headers must be a dict or None, got {type(headers).__name__}")

        if request_body is None:
            req_body_val: Any = None
            req_enc = "empty"
        elif isinstance(request_body, (dict, list)):
            req_body_val = copy.deepcopy(request_body)
            req_enc = "json"
        elif isinstance(request_body, (bytes, bytearray, memoryview)):
            req_body_val, req_enc = decode_body_with_encoding(bytes(request_body))
        else:
            req_body_val = str(request_body)
            req_enc = "text"

        req_headers = {"content-type": "application/json"} if req_enc == "json" else {}
        resp_headers: dict[str, str] = {"content-type": "application/json"}
        if headers:
            for k, v in headers.items():
                if k.lower() == "content-type":
                    resp_headers["content-type"] = str(v)
                else:
                    resp_headers[str(k)] = str(v)

        err_body = {
            "error": {
                "message": message,
                "type": "synthetic_error",
                "code": code,
            }
        }
        return self._append_interaction(
            method=method_norm,
            url=target_url,
            req_headers=req_headers,
            req_body=req_body_val,
            req_encoding=req_enc,
            status_code=code,
            resp_headers=resp_headers,
            resp_body=err_body,
            resp_encoding="json",
            stream_chunks=None,
            latency_ms=lat,
        )

    def build(self) -> Cassette:
        """Return a fresh Cassette copy with deep-copied interactions and metadata."""
        return Cassette(
            path=self._path,
            interactions=copy.deepcopy(self._interactions),
            recorded_at=self._recorded_at,
            metadata=copy.deepcopy(self._metadata),
        )

    def save(
        self,
        path: str | Path | None = None,
        *,
        compress: bool | None = None,
    ) -> Path:
        """Persist the built cassette to disk (.tape.json or .tape.json.gz) and return its Path."""
        if compress is not None and not isinstance(compress, bool):
            raise ValueError(f"compress must be a bool or None, got {compress!r}")
        cassette = self.build()
        if path is not None:
            cassette.path = _validate_path(path)
        use_gzip = cassette.path.suffix.lower() == ".gz" if compress is None else compress
        if use_gzip == (cassette.path.suffix.lower() == ".gz"):
            cassette.save()
        else:
            _write_bytes_atomic(cassette.path, cassette.to_bytes(compress=use_gzip))
        return cassette.path
