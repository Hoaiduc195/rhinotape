"""Tests for Wave 5: OpenAI Embeddings and Cohere v1/v2 JSON & SSE parsers + pricing."""

from __future__ import annotations

import pytest

from llm_tape.parsers import parse_llm_interaction


def test_parse_openai_embeddings_and_pricing() -> None:
    s = parse_llm_interaction(
        url="https://api.openai.com/v1/embeddings",
        request_body={"model": "text-embedding-3-small", "input": ["hello", "world"]},
        response_body={
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 1_000_000, "total_tokens": 1_000_000},
        },
    )
    assert s.provider == "openai"
    assert s.model == "text-embedding-3-small"
    assert s.usage.input_tokens == 1_000_000
    assert s.usage.output_tokens == 0
    assert s.estimated_cost_usd == pytest.approx(0.02)

    # Proxy URL detected via embedding list response shape
    s_proxy = parse_llm_interaction(
        url="https://proxy.internal/embed",
        request_body={"model": "text-embedding-3-large"},
        response_body={
            "object": "list",
            "model": "text-embedding-3-large",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5]}],
            "usage": {"prompt_tokens": 1_000_000, "total_tokens": 1_000_000},
        },
    )
    assert s_proxy.provider == "openai"
    assert s_proxy.estimated_cost_usd == pytest.approx(0.13)


def test_parse_cohere_v2_v1_json_and_sse() -> None:
    # Cohere v2 JSON with tool call
    v2 = parse_llm_interaction(
        url="https://api.cohere.com/v2/chat",
        request_body={"model": "command-r-plus"},
        response_body={
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Checking weather."}],
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Hanoi"}'},
                    }
                ],
            },
            "meta": {"tokens": {"input_tokens": 1000, "output_tokens": 200}},
        },
    )
    assert v2.provider == "cohere"
    assert v2.model == "command-r-plus"
    assert v2.text == "Checking weather."
    assert len(v2.tool_calls) == 1
    assert v2.tool_calls[0].name == "get_weather"
    assert v2.tool_calls[0].arguments == {"city": "Hanoi"}
    assert v2.usage.total_tokens == 1200
    assert v2.estimated_cost_usd > 0

    # Cohere v1 JSON
    v1 = parse_llm_interaction(
        url="https://api.cohere.ai/v1/chat",
        request_body={"model": "command-r"},
        response_body={
            "text": "Hello v1",
            "tool_calls": [{"name": "lookup", "parameters": {"id": 7}}],
            "meta": {"billed_units": {"input_tokens": 500, "output_tokens": 100}},
        },
    )
    assert v1.provider == "cohere"
    assert v1.text == "Hello v1"
    assert v1.tool_calls[0].name == "lookup"
    assert v1.tool_calls[0].arguments == {"id": 7}
    assert v1.usage.total_tokens == 600

    # Cohere v2 SSE streaming
    sse = parse_llm_interaction(
        url="https://api.cohere.com/v2/chat",
        request_body={"model": "command-a"},
        response_body=None,
        stream_chunks=(
            'event: content-delta\ndata: {"type":"content-delta",'
            '"delta":{"message":{"content":{"text":"Hi "}}}}\n\n',
            'event: content-delta\ndata: {"type":"content-delta",'
            '"delta":{"message":{"content":{"text":"there!"}}}}\n\n',
            'event: tool-call-start\ndata: {"type":"tool-call-start","index":0,'
            '"delta":{"message":{"tool_calls":{"id":"tc1",'
            '"function":{"name":"search","arguments":"{\\"q\\":"}}}}}\n\n',
            'event: tool-call-delta\ndata: {"type":"tool-call-delta","index":0,'
            '"delta":{"message":{"tool_calls":{"function":{"arguments":"\\"ok\\"}"}}}}}\n\n',
            'event: message-end\ndata: {"type":"message-end",'
            '"delta":{"usage":{"tokens":{"input_tokens":200,"output_tokens":50}}}}\n\n',
        ),
        content_type="text/event-stream",
    )
    assert sse.provider == "cohere"
    assert sse.is_stream is True
    assert sse.text == "Hi there!"
    assert sse.tool_calls[0].name == "search"
    assert sse.tool_calls[0].arguments == {"q": "ok"}
    assert sse.usage.total_tokens == 250
    assert sse.estimated_cost_usd > 0
