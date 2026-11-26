"""Example: Serving a recorded cassette over HTTP via TapeReplayServer.

Run with:
    uv run python examples/local_replay_server.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import httpx

from rhinotape import TapeMode, TapeReplayServer, tape


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cassette_path = Path(tmp) / "openai_demo.tape.json"
        payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello!"}]}

        # 1. Record a sample interaction (mocked here so this script runs offline)
        mock_upstream = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "id": "chatcmpl-demo",
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"role": "assistant", "content": "Hi from tape!"}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                },
            )
        )
        with tape(cassette_path, mode=TapeMode.RECORD):
            with httpx.Client(transport=mock_upstream) as client:
                client.post("https://api.openai.com/v1/chat/completions", json=payload)

        # 2. Serve the cassette over localhost for any SDK or language (Node, Go, curl, Python)
        with TapeReplayServer(cassette_path) as server:
            with httpx.Client() as client:
                resp = client.post(f"{server.url}/v1/chat/completions", json=payload)
                print("Replayed response:", resp.json()["choices"][0]["message"]["content"])
                print("Server stats:", client.get(f"{server.url}/__tape__/stats").json())


if __name__ == "__main__":
    main()
