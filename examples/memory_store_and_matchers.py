"""Example: Zero-disk MemoryCassetteStore + composable RequestMatchers.

Run with:
    uv run python examples/memory_store_and_matchers.py
"""

from __future__ import annotations

import httpx

from llm_tape import (
    MemoryCassetteStore,
    TapeMode,
    match_all,
    match_json_regex,
    match_json_subset,
    match_method,
    match_path,
    tape,
)


def main() -> None:
    store = MemoryCassetteStore(compress=True)
    mock_upstream = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "id": "chatcmpl-match",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "Order shipped."}}],
            },
        )
    )

    # Record into memory (zero filesystem writes)
    with tape("orders/status.tape.json", mode=TapeMode.RECORD, store=store):
        with httpx.Client(transport=mock_upstream) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Check order #1001"}],
                },
            )

    # Replay with a composable matcher that tolerates dynamic order IDs
    matcher = match_all(
        match_method(),
        match_path(),
        match_json_subset(["model"]),
        match_json_regex("messages.0.content", r"Check order #\d+"),
    )
    with tape("orders/status.tape.json", mode=TapeMode.REPLAY, store=store, custom_matcher=matcher):
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Check order #9999"}],
                },
            )
            print("Matched reply:", resp.json()["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
