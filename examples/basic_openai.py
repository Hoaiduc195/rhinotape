"""Example: recording and replaying an OpenAI-compatible chat & SSE stream with llm-tape."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from llm_tape import tape


def ask_llm(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "sk-demo-key")
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])


if __name__ == "__main__":
    cassette = Path("tapes/basic_openai.tape.json")
    with tape(cassette, mode="once", ignore_localhost=True) as session:
        answer = ask_llm("Explain pytest fixtures in one sentence.")
        print("Answer:", answer)
        session.assert_call_count(1)
        session.assert_tokens_under(500)
        session.assert_cost_under(0.005)
