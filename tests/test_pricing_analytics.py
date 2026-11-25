"""Tests for cache-write pricing, cache savings analytics, and llm-tape pricing CLI."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from llm_tape import TapeMode, estimate_cache_savings_usd, estimate_cost_usd, tape
from llm_tape.cli import main as cli_main
from llm_tape.parsers import parse_llm_interaction


def test_anthropic_cache_write_and_read_pricing() -> None:
    # claude-3-5-sonnet: input=$3/M, cached_read=$0.30/M, cache_write=$3.75/M, output=$15/M
    cost = estimate_cost_usd(
        "claude-3-5-sonnet-20241022",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=400_000,
        cache_creation_tokens=200_000,
    )
    # 400k uncached * 3.0 + 400k read * 0.30 + 200k write * 3.75 = 1.20 + 0.12 + 0.75 = 2.07
    assert round(cost, 6) == 2.07

    savings = estimate_cache_savings_usd(
        "claude-3-5-sonnet-20241022",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=400_000,
        cache_creation_tokens=200_000,
    )
    # Uncached = 3.00, actual = 2.07 -> net savings = +0.93
    assert round(savings, 6) == 0.93

    # Write-only call has negative net savings (surcharge of $0.75 per 1M write tokens)
    write_only = estimate_cache_savings_usd(
        "claude-3-5-sonnet",
        input_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
    )
    assert round(write_only, 6) == -0.75
    assert estimate_cache_savings_usd("unknown-model", 1000, cached_input_tokens=500) == 0.0


def test_anthropic_json_sse_and_bedrock_cache_creation_tokens() -> None:
    anth = parse_llm_interaction(
        url="https://api.anthropic.com/v1/messages",
        request_body={"model": "claude-sonnet-4"},
        response_body={
            "type": "message",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 500,
                "cache_read_input_tokens": 400,
                "output_tokens": 50,
            },
        },
    )
    assert anth.usage.input_tokens == 1000
    assert anth.usage.cached_input_tokens == 400
    assert anth.usage.cache_creation_tokens == 500
    assert anth.usage.to_dict()["cache_creation_tokens"] == 500

    sse_chunks = (
        'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4",'
        '"usage":{"input_tokens":50,"cache_creation_input_tokens":200,"cache_read_input_tokens":300}}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"hi"}}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":10}}\n\n',
    )
    anth_sse = parse_llm_interaction(
        url="https://api.anthropic.com/v1/messages",
        request_body={"model": "claude-sonnet-4", "stream": True},
        response_body=None,
        stream_chunks=sse_chunks,
        content_type="text/event-stream",
    )
    assert anth_sse.usage.cache_creation_tokens == 200
    assert anth_sse.usage.cached_input_tokens == 300
    assert anth_sse.usage.input_tokens == 550

    bedrock = parse_llm_interaction(
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4/converse",
        request_body={},
        response_body={
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "usage": {
                "inputTokens": 100,
                "cacheReadInputTokens": 200,
                "cacheWriteInputTokens": 300,
                "outputTokens": 20,
            },
        },
    )
    assert bedrock.usage.cache_creation_tokens == 300
    assert bedrock.usage.cached_input_tokens == 200
    assert bedrock.usage.input_tokens == 600


def test_stats_cache_savings_and_cli_pricing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cassette_path = tmp_path / "cached.tape.json"
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "1",
                "model": "gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": "cached reply"}}],
                "usage": {
                    "prompt_tokens": 10_000,
                    "completion_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 8_000},
                },
            },
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD) as session:
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert session.total_usage.cached_input_tokens == 8_000

    assert cli_main(["stats", str(cassette_path), "--json"]) == 0
    stats_data = json.loads(capsys.readouterr().out)
    assert stats_data["totals"]["cache_savings_usd"] > 0.0

    assert cli_main(["stats", str(cassette_path)]) == 0
    assert "Estimated cache savings:" in capsys.readouterr().out

    assert cli_main(["pricing", "--filter", "claude", "--json"]) == 0
    pricing_data = json.loads(capsys.readouterr().out)
    assert "claude-sonnet-4" in pricing_data
    assert pricing_data["claude-sonnet-4"]["cache_write_per_million"] == 3.75

    assert cli_main(["pricing", "--filter", "gpt-4o"]) == 0
    table_out = capsys.readouterr().out
    assert "gpt-4o" in table_out
    assert "Write / 1M" in table_out
