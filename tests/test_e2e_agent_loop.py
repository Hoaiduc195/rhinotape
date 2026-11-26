"""End-to-end multi-turn agent loop test with NEW_EPISODES and guardrails (Wave 96)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from rhinotape import TapeAssertionError, TapeMode, tape


def _agent_transport() -> httpx.MockTransport:
    def _handle(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content.decode("utf-8"))
        turn = payload["messages"][-1]["content"]
        if turn == "step-1":
            return httpx.Response(
                200,
                json={
                    "id": "t1",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "fetch_price",
                                            "arguments": '{"ticker":"NVDA"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 200, "completion_tokens": 50},
                },
            )
        if turn == "step-2":
            return httpx.Response(
                200,
                json={
                    "id": "t2",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "c2",
                                        "type": "function",
                                        "function": {
                                            "name": "compute_var",
                                            "arguments": '{"ticker":"NVDA","window":30}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 300, "completion_tokens": 60},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "t3",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"decision": "hold", "confidence": 0.87}',
                        }
                    }
                ],
                "usage": {"prompt_tokens": 400, "completion_tokens": 40},
            },
        )

    return httpx.MockTransport(_handle)


def test_multi_turn_agent_loop_new_episodes_and_assertions(tmp_path: Path) -> None:
    cassette_path = tmp_path / "agent_loop.tape.json"

    # Phase 1: record turns 1 and 2
    with tape(cassette_path, mode=TapeMode.RECORD, max_cost_usd=0.05):
        with httpx.Client(transport=_agent_transport()) as client:
            for step in ("step-1", "step-2"):
                r = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": step}]},
                )
                assert r.status_code == 200

    # Phase 2: NEW_EPISODES replays turns 1-2 and records new turn 3
    with tape(cassette_path, mode=TapeMode.NEW_EPISODES, max_cost_usd=0.05) as session:
        with httpx.Client(transport=_agent_transport()) as client:
            for step in ("step-1", "step-2", "step-3"):
                r = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": step}]},
                )
                assert r.status_code == 200
        session.assert_call_count(3)
        session.assert_tool_sequence(["fetch_price", "compute_var"])
        session.assert_tool_args("compute_var", {"ticker": "NVDA", "window": 30}, exact=True)
        out = session.assert_json_output(
            required_keys=["decision", "confidence"],
            predicate=lambda d: d["confidence"] > 0.8,
        )
        assert out["decision"] == "hold"

    # Phase 3: strict offline replay of all 3 turns
    with tape(cassette_path, mode=TapeMode.REPLAY) as session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            for step in ("step-1", "step-2", "step-3"):
                r = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": step}]},
                )
                assert r.status_code == 200
        session.assert_status_codes([200, 200, 200])
        session.assert_tool_call_count(2)
        session.assert_no_unused_interactions()

    # Phase 4: max_interactions guardrail blocks runaway loop in NEW_EPISODES
    with pytest.raises(TapeAssertionError, match="max_interactions=0"):
        with tape(cassette_path, mode=TapeMode.NEW_EPISODES, max_interactions=0):
            with httpx.Client(transport=_agent_transport()) as client:
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "step-4"}],
                    },
                )
