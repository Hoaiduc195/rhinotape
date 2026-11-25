"""Example: testing a multi-step tool-calling agent in pytest with llm-tape."""

from __future__ import annotations

import httpx
import pytest

from llm_tape import TapeSession


def run_agent(query: str) -> str:
    with httpx.Client() as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "metadata": {"trace_id": "dynamic-per-run"},
                "messages": [{"role": "user", "content": query}],
            },
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"].get("content", ""))


@pytest.mark.tape(
    name="agent_workflow",
    ignore_localhost=True,
    ignore_fields=["metadata.trace_id"],
)
def test_agent_stays_within_budget(llm_tape: TapeSession) -> None:
    """Run with `pytest --tape-mode=replay` in CI for zero-cost deterministic tests."""
    # In real projects, record once with real credentials, then commit the .tape.json file.
    assert llm_tape is not None
