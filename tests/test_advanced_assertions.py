"""Tests for Batch 9 advanced TapeSession assertions."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from rhinotape import TapeAssertionError, TapeMode, tape


def test_assert_json_output_and_per_call_budgets(tmp_path: Path) -> None:
    cassette_path = tmp_path / "json_out.tape.json"
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "gpt-4o",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"status": "ok", "score": 95}\n```',
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "score_item", "arguments": '{"id": 1}'},
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            },
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD) as session:
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "score"}]},
            )
        parsed = session.assert_json_output(
            required_keys=["status", "score"],
            predicate=lambda d: d["score"] >= 90,
        )
        assert parsed == {"status": "ok", "score": 95}

        session.assert_max_tokens_per_call(1200)
        session.assert_max_cost_per_call(0.05)
        session.assert_not_streamed()
        session.assert_status_codes([200])
        session.assert_tool_call_count(1)
        session.assert_cache_hit_rate_at_least(80.0)

        with pytest.raises(TapeAssertionError, match="missing required key"):
            session.assert_json_output(required_keys=["missing_key"])
        with pytest.raises(TapeAssertionError, match="predicate"):
            session.assert_json_output(predicate=lambda d: d["score"] > 100)
        with pytest.raises(TapeAssertionError, match="Per-call token budget"):
            session.assert_max_tokens_per_call(500)
        with pytest.raises(TapeAssertionError, match="Per-call cost budget"):
            session.assert_max_cost_per_call(0.00001)
        with pytest.raises(TapeAssertionError, match="streamed"):
            session.assert_streamed()
        with pytest.raises(TapeAssertionError, match="status code sequence"):
            session.assert_status_codes([200, 200])
        with pytest.raises(TapeAssertionError, match="tool call"):
            session.assert_tool_call_count(2)
        with pytest.raises(TapeAssertionError, match="cache hit rate"):
            session.assert_cache_hit_rate_at_least(90.0)
        with pytest.raises(ValueError):
            session.assert_cache_hit_rate_at_least(150.0)


def test_assert_streamed_and_invalid_json_output(tmp_path: Path) -> None:
    cassette_path = tmp_path / "stream_assert.tape.json"

    class _Stream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"not json"}}]}\n\n'
            yield b"data: [DONE]\n\n"

    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_Stream(),
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD) as session:
        with httpx.Client(transport=transport) as client:
            with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "stream": True},
            ) as resp:
                resp.read()
        session.assert_streamed()
        with pytest.raises(TapeAssertionError, match="no streamed calls"):
            session.assert_not_streamed()
        with pytest.raises(TapeAssertionError, match="valid JSON"):
            session.assert_json_output()
        with pytest.raises(TapeAssertionError, match="input_tokens is 0"):
            session.assert_cache_hit_rate_at_least(10.0)
        session.assert_cache_hit_rate_at_least(0.0)
