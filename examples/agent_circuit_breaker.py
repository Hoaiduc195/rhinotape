"""Example: Live cost/call circuit breakers, chaos retry testing, and JSON assertions.

Run with:
    uv run python examples/agent_circuit_breaker.py
"""

from __future__ import annotations

import httpx

from llm_tape import (
    MemoryCassetteStore,
    TapeAssertionError,
    TapeMode,
    fail_first,
    tape,
)


def main() -> None:
    store = MemoryCassetteStore()
    mock_upstream = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "id": "chatcmpl-agent",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"action": "approve", "risk": 0.12}',
                        }
                    }
                ],
                "usage": {"prompt_tokens": 500, "completion_tokens": 50},
            },
        )
    )

    # 1. Record safely with live circuit breakers (max_interactions + max_cost_usd)
    with tape(
        "agent/decision.tape.json",
        mode=TapeMode.RECORD,
        store=store,
        max_interactions=3,
        max_cost_usd=0.05,
    ) as session:
        with httpx.Client(transport=mock_upstream) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "evaluate"}],
                },
            )
        parsed = session.assert_json_output(
            required_keys=["action", "risk"],
            predicate=lambda d: d["risk"] < 0.5,
        )
        print("Validated JSON output:", parsed)

    # 2. Chaos-test retry handling: inject HTTP 429 on first attempt, succeed on retry
    with tape(
        "agent/decision.tape.json",
        mode=TapeMode.REPLAY,
        store=store,
        allow_playback_repeats=True,
        on_replay=fail_first(1, status_code=429),
    ) as replay_session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            first = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "evaluate"}],
                },
            )
            second = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "evaluate"}],
                },
            )
            print("Chaos retry statuses:", first.status_code, "->", second.status_code)
        replay_session.assert_status_codes([429, 200])

    # 3. Demonstrate circuit breaker tripping on runaway calls
    try:
        with tape("agent/runaway.tape.json", mode=TapeMode.RECORD, store=store, max_interactions=1):
            with httpx.Client(transport=mock_upstream) as client:
                for i in range(3):
                    client.post(
                        "https://api.openai.com/v1/chat/completions",
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": str(i)}],
                        },
                    )
    except TapeAssertionError as exc:
        print("Circuit breaker tripped as expected:", exc)


if __name__ == "__main__":
    main()
