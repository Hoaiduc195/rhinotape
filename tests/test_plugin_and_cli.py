"""Tests for the pytest plugin fixture and CLI commands."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from llm_tape.cli import main as cli_main
from llm_tape.session import TapeSession


@pytest.mark.tape(name="plugin_demo")
def test_pytest_fixture_records_and_asserts(llm_tape: TapeSession, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 25},
            },
        )

    # Redirect cassette into tmp_path so running pytest doesn't dirty the repo
    llm_tape.cassette_path = tmp_path / "plugin_demo.tape.json"

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resp = client.post("https://api.openai.com/v1/chat/completions", json={"prompt": "hi"})
        assert resp.status_code == 200

    llm_tape.assert_call_count(1)
    llm_tape.assert_tokens_under(100)


def test_cli_inspect_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cassette_path = tmp_path / "demo.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 100},
            },
        )

    with TapeSession(cassette_path, mode="record"):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"hello": True})

    exit_code = cli_main(["inspect", str(cassette_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "gpt-4o-mini" in out
    assert "Tokens   : 300" in out

    exit_code = cli_main(["summary", str(tmp_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "demo.tape.json" in out
    assert "TOTAL (1 files)" in out
