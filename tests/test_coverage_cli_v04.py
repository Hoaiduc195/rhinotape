"""Targeted coverage tests for Batch 6-14 CLI subcommands (Wave 110)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from llm_tape import TapeMode, tape
from llm_tape.cli import main as cli_main


def test_cli_v04_empty_missing_and_edge_branches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope"
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    # pricing no match
    assert cli_main(["pricing", "--filter", "nonexistent-model-xyz"]) == 0
    assert "No models matching" in capsys.readouterr().out

    # search missing path + empty dir + no matches
    assert cli_main(["search", str(missing)]) == 1
    assert cli_main(["search", str(empty_dir)]) == 0
    assert "No matching interactions" in capsys.readouterr().out

    # replay-check missing + empty (plain & json) + corrupt
    assert cli_main(["replay-check", str(missing)]) == 1
    assert cli_main(["replay-check", str(empty_dir)]) == 0
    assert cli_main(["replay-check", str(empty_dir), "--json"]) == 0
    bad_file = empty_dir / "bad.tape.json"
    bad_file.write_text("{broken", encoding="utf-8")
    assert cli_main(["replay-check", str(empty_dir)]) == 1
    capsys.readouterr()
    bad_file.unlink()

    # reparse missing + empty dir
    assert cli_main(["reparse", str(missing)]) == 1
    assert cli_main(["reparse", str(empty_dir)]) == 0

    # list missing + negative limit + empty dir
    assert cli_main(["list", str(missing)]) == 1
    assert cli_main(["list", str(empty_dir), "--limit", "-1"]) == 1
    assert cli_main(["list", str(empty_dir)]) == 0
    capsys.readouterr()

    # Record one cassette to cover search filters, list sort modes, and identical text diff
    cpath = empty_dir / "one.tape.json"
    t = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "1",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "same text"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )
    )
    with tape(cpath, mode=TapeMode.RECORD):
        with httpx.Client(transport=t) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )

    assert (
        cli_main(["search", str(empty_dir), "--model", "gpt-4o-mini", "--provider", "openai"]) == 0
    )
    assert cli_main(["search", str(empty_dir), "--min-cost", "999.0"]) == 0
    for sort_key in ("tokens", "calls", "latency"):
        assert cli_main(["list", str(empty_dir), "--sort", sort_key]) == 0
    assert cli_main(["diff", str(cpath), str(cpath), "--show-text-diff"]) == 0
    assert "(identical)" in capsys.readouterr().out
