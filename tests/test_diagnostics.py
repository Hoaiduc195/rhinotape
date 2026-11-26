"""Tests for Batch 14 diagnostics & diff UX (Waves 83-86)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from rhinotape import TapeMode, UnmatchedRequestError, tape
from rhinotape.cli import main as cli_main
from rhinotape.scrubbing import summarize_json_diff


def test_summarize_json_diff_and_unmatched_error_attributes(tmp_path: Path) -> None:
    diffs = summarize_json_diff(
        {"model": "gpt-4o", "messages": [{"content": "a"}], "removed_key": 1},
        {"model": "gpt-4o-mini", "messages": [{"content": "b"}], "added_key": 2},
    )
    assert "+ added_key" in diffs
    assert "- removed_key" in diffs
    assert "~ model" in diffs
    assert "~ messages[0].content" in diffs

    cassette_path = tmp_path / "diag.tape.json"
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "1",
                "model": "gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD):
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "old"}]},
            )

    with pytest.raises(UnmatchedRequestError) as exc_info:
        with tape(cassette_path, mode=TapeMode.REPLAY):
            with httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(500))
            ) as client:
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": [{"role": "user", "content": "new"}]},
                )

    err = exc_info.value
    assert err.incoming is not None
    assert err.closest is not None
    assert err.incoming.body["messages"][0]["content"] == "new"
    assert err.closest.body["messages"][0]["content"] == "old"
    assert "Changed JSON fields:" in err.diff_text
    assert "~ messages[0].content" in err.diff_text


def test_cli_diff_show_text_diff_and_summary_group_by_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dir_a = tmp_path / "suite_a"
    dir_b = tmp_path / "suite_b"
    c1 = dir_a / "t1.tape.json"
    c2 = dir_b / "t2.tape.json"

    for cpath, reply in ((c1, "alpha line\nshared"), (c2, "beta line\nshared")):
        t = httpx.MockTransport(
            lambda r, txt=reply: httpx.Response(
                200,
                json={
                    "id": "1",
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"role": "assistant", "content": txt}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )
        )
        with tape(cpath, mode=TapeMode.RECORD):
            with httpx.Client(transport=t) as client:
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
                )

    assert cli_main(["diff", str(c1), str(c2), "--show-text-diff"]) == 0
    out = capsys.readouterr().out
    assert "Response Text Diff:" in out
    assert "-alpha line" in out
    assert "+beta line" in out

    assert cli_main(["diff", str(c1), str(c2), "--show-text-diff", "--json"]) == 0
    jdiff = json.loads(capsys.readouterr().out)
    assert any("-alpha line" in ln for ln in jdiff["text_diff"])

    assert cli_main(["summary", str(tmp_path), "--group-by", "dir", "--json"]) == 0
    grouped = json.loads(capsys.readouterr().out)
    dirs = {g["directory"] for g in grouped["groups"]}
    assert dirs == {"suite_a", "suite_b"}

    assert cli_main(["summary", str(tmp_path), "--group-by", "dir"]) == 0
    table_out = capsys.readouterr().out
    assert "suite_a" in table_out
    assert "TOTAL (2 dirs)" in table_out
