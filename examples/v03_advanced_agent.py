"""Example: v0.3.0 features — body redaction, flexible matching, and tool-arg assertions."""

from __future__ import annotations

from pathlib import Path

import httpx

from rhinotape import tape


def run_customer_support_agent(user_email: str, query: str) -> str:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "metadata": {"customer_email": user_email},
                "messages": [{"role": "user", "content": query}],
            },
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"].get("content", ""))


if __name__ == "__main__":
    cassette = Path("tapes/v03_support_agent.tape.json")
    with tape(
        cassette,
        mode="once",
        redact_fields=["metadata.customer_email"],
        redact_patterns=[r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"],
        match_on=("method", "path", "body"),
    ) as session:
        answer = run_customer_support_agent("alice@example.com", "Check order #1234")
        print("Answer:", answer)
        session.assert_call_count(1)
        session.assert_cost_under(0.01)
        session.assert_max_latency_ms(10_000.0)
        session.assert_no_unused_interactions()
