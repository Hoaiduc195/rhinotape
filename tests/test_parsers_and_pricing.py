"""Tests for provider-aware LLM payload parsing and token pricing."""

from __future__ import annotations

import pytest

from llm_tape.parsers import parse_llm_interaction
from llm_tape.pricing import ModelPrice, estimate_cost_usd


def test_parse_openai_chat_completions_with_tools() -> None:
    req = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Weather in Hanoi?"}]}
    resp = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Hanoi"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    summary = parse_llm_interaction("https://api.openai.com/v1/chat/completions", req, resp)
    assert summary.provider == "openai"
    assert summary.model == "gpt-4o-mini-2024-07-18"
    assert summary.usage.input_tokens == 100
    assert summary.usage.output_tokens == 50
    assert summary.usage.total_tokens == 150
    assert len(summary.tool_calls) == 1
    assert summary.tool_calls[0].name == "get_weather"
    assert summary.tool_calls[0].arguments == {"city": "Hanoi"}
    assert summary.estimated_cost_usd == pytest.approx((100 * 0.15 + 50 * 0.60) / 1_000_000)


def test_parse_openai_responses_api() -> None:
    resp = {
        "model": "gpt-4.1-mini",
        "output": [
            {
                "type": "function_call",
                "call_id": "fc_1",
                "name": "search_docs",
                "arguments": '{"query": "pytest"}',
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Done searching."}],
            },
        ],
        "usage": {"input_tokens": 200, "output_tokens": 40},
    }
    summary = parse_llm_interaction("https://api.openai.com/v1/responses", {}, resp)
    assert summary.provider == "openai"
    assert summary.usage.total_tokens == 240
    assert [tc.name for tc in summary.tool_calls] == ["search_docs"]
    assert summary.text == "Done searching."


def test_parse_anthropic_messages() -> None:
    resp = {
        "id": "msg_1",
        "type": "message",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {"type": "text", "text": "Let me look that up."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "query_db",
                "input": {"sql": "SELECT 1"},
            },
        ],
        "usage": {"input_tokens": 300, "output_tokens": 120},
    }
    summary = parse_llm_interaction("https://api.anthropic.com/v1/messages", {}, resp)
    assert summary.provider == "anthropic"
    assert summary.model == "claude-3-5-sonnet-20241022"
    assert summary.usage.total_tokens == 420
    assert summary.text == "Let me look that up."
    assert len(summary.tool_calls) == 1
    assert summary.tool_calls[0].name == "query_db"
    assert summary.tool_calls[0].arguments == {"sql": "SELECT 1"}
    assert summary.estimated_cost_usd > 0


def test_parse_gemini_generate_content() -> None:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    resp = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Checking stock price."},
                        {"functionCall": {"name": "get_price", "args": {"ticker": "GOOG"}}},
                    ]
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 80},
    }
    summary = parse_llm_interaction(url, {}, resp)
    assert summary.provider == "gemini"
    assert summary.model == "gemini-2.5-flash"
    assert summary.usage.input_tokens == 500
    assert summary.usage.output_tokens == 80
    assert summary.text == "Checking stock price."
    assert summary.tool_calls[0].name == "get_price"
    assert summary.tool_calls[0].arguments == {"ticker": "GOOG"}
    assert summary.estimated_cost_usd > 0


def test_custom_pricing_override() -> None:
    cost = estimate_cost_usd(
        "my-local-model-v2",
        input_tokens=1_000_000,
        output_tokens=500_000,
        custom_pricing={"my-local-model": ModelPrice(1.0, 2.0)},
    )
    assert cost == pytest.approx(2.0)
