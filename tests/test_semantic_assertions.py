"""Tests for Wave 8: semantic text/thinking/tool-safety/reasoning-token assertions."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from llm_tape import TapeAssertionError, TapeMode, TapeTransport, tape


def test_semantic_assertions_and_text_thinking_properties(tmp_path: Path) -> None:
    tape_path = tmp_path / "sem.tape.json"
    upstream = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "model": "deepseek-reasoner",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "First check the safety rules.",
                            "content": "The answer is 42.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "read_docs", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 80,
                    "completion_tokens_details": {"reasoning_tokens": 55},
                },
            },
        )
    )

    with tape(tape_path, mode=TapeMode.RECORD, patch_httpx=False) as s:
        assert s.last_text == ""
        assert s.last_thinking == ""
        assert s.all_texts == []

        with httpx.Client(transport=TapeTransport(s, upstream=upstream)) as client:
            client.post("https://api.deepseek.com/chat/completions", json={"q": "hi"})

        assert s.last_text == "The answer is 42."
        assert s.last_thinking == "First check the safety rules."
        assert s.all_texts == ["The answer is 42."]

        s.assert_text_contains("42")
        s.assert_text_contains("the answer", case_sensitive=False)
        s.assert_text_matches(r"\b\d{2}\b")
        s.assert_text_matches(re.compile(r"answer is 42"))
        s.assert_no_tool_called("delete_db")
        s.assert_thinking_present()
        s.assert_reasoning_tokens_under(60)

        with pytest.raises(TapeAssertionError, match="contain"):
            s.assert_text_contains("the answer", case_sensitive=True)
        with pytest.raises(TapeAssertionError, match="match pattern"):
            s.assert_text_matches(r"999")
        with pytest.raises(TapeAssertionError, match="NOT to be called"):
            s.assert_no_tool_called("read_docs")
        with pytest.raises(TapeAssertionError, match="Reasoning token budget"):
            s.assert_reasoning_tokens_under(50)

    # assert_thinking_present fails when no thinking output exists
    empty_tape = tmp_path / "no_think.tape.json"
    with tape(empty_tape, mode=TapeMode.RECORD, patch_httpx=False) as s2:
        with pytest.raises(TapeAssertionError, match="non-empty thinking"):
            s2.assert_thinking_present()
