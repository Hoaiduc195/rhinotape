"""CLI integration tests for gen-test, inspect --graph, and top-level exports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import rhinotape
from rhinotape.cassette import Cassette
from rhinotape.cli import main
from rhinotape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    ToolCall,
    Usage,
)


def _sample_agent_cassette(path: Path) -> Path:
    interactions = [
        Interaction(
            request=RecordedRequest(
                method="POST",
                url="https://api.openai.com/v1/chat/completions",
                headers={"content-type": "application/json"},
                body={"model": "gpt-4o", "messages": [{"role": "user", "content": "weather?"}]},
                fingerprint="fp-1",
            ),
            response=RecordedResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                body={"choices": []},
            ),
            llm=LLMCallSummary(
                provider="openai",
                model="gpt-4o",
                text="",
                tool_calls=[ToolCall(name="get_weather", arguments={"city": "Tokyo"})],
                usage=Usage(input_tokens=50, output_tokens=20),
                estimated_cost_usd=0.0004,
            ),
            latency_ms=80.0,
        ),
        Interaction(
            request=RecordedRequest(
                method="POST",
                url="https://api.openai.com/v1/chat/completions",
                headers={"content-type": "application/json"},
                body={"model": "gpt-4o", "messages": [{"role": "tool", "content": "22C"}]},
                fingerprint="fp-2",
            ),
            response=RecordedResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                body={"choices": []},
            ),
            llm=LLMCallSummary(
                provider="openai",
                model="gpt-4o",
                text="It is 22C in Tokyo.",
                usage=Usage(input_tokens=90, output_tokens=15),
                estimated_cost_usd=0.0005,
            ),
            latency_ms=65.0,
        ),
    ]
    cassette = Cassette(path=path, interactions=interactions)
    cassette.save()
    return path


def test_cli_gen_test_stdout_file_force_and_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tape_path = _sample_agent_cassette(tmp_path / "weather_agent.tape.json")

    assert main(["gen-test", str(tape_path)]) == 0
    stdout_code = capsys.readouterr().out
    ast.parse(stdout_code)
    assert "def test_weather_agent() -> None:" in stdout_code
    assert "tape.assert_call_count(2)" in stdout_code
    assert 'tape.assert_tool_sequence(["get_weather"])' in stdout_code

    out_file = tmp_path / "generated" / "test_weather.py"
    assert (
        main(
            [
                "gen-test",
                str(tape_path),
                "-o",
                str(out_file),
                "--test-name",
                "custom_weather",
                "--no-text",
            ]
        )
        == 0
    )
    assert "Generated pytest test from" in capsys.readouterr().out
    written = out_file.read_text(encoding="utf-8")
    ast.parse(written)
    assert "def test_custom_weather() -> None:" in written
    assert "assert_text_contains" not in written

    # Refuses to overwrite without --force, overwrites with --force
    assert main(["gen-test", str(tape_path), "-o", str(out_file)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main(["gen-test", str(tape_path), "-o", str(out_file), "--force"]) == 0
    capsys.readouterr()

    # Error handling: missing file and invalid headroom
    assert main(["gen-test", str(tmp_path / "missing.tape.json")]) == 1
    assert "not found" in capsys.readouterr().err
    assert main(["gen-test", str(tape_path), "--cost-headroom", "0.5"]) == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_inspect_graph_ascii_and_mermaid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tape_path = _sample_agent_cassette(tmp_path / "agent_flow.tape.json")

    assert main(["inspect", str(tape_path), "--graph"]) == 0
    ascii_out = capsys.readouterr().out
    assert "Trajectory: agent_flow.tape.json" in ascii_out
    assert "tool: get_weather(" in ascii_out
    assert "It is 22C in Tokyo." in ascii_out

    assert main(["inspect", str(tape_path), "--graph", "mermaid"]) == 0
    mermaid_out = capsys.readouterr().out
    assert mermaid_out.startswith("flowchart TD")
    assert "C1 --> T1_1" in mermaid_out
    assert "T1_1 --> C2" in mermaid_out

    # --interaction slices the graph cleanly and validates bounds
    assert main(["inspect", str(tape_path), "--graph", "--interaction", "2"]) == 0
    sliced_out = capsys.readouterr().out
    assert "It is 22C in Tokyo." in sliced_out
    assert "get_weather" not in sliced_out

    assert main(["inspect", str(tape_path), "--graph", "--interaction", "99"]) == 1
    assert "out of range" in capsys.readouterr().err


def test_public_exports_v05_features() -> None:
    for name in (
        "DEPRECATED_MODELS",
        "generate_pytest_code",
        "get_deprecation_notice",
        "render_ascii_graph",
        "render_mermaid_graph",
    ):
        assert hasattr(rhinotape, name)
        assert name in rhinotape.__all__
