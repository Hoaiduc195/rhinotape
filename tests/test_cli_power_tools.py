"""Tests for Batch 10 CLI power tools: search, replay-check, reparse, export junit, list."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import pytest

from llm_tape import Cassette, TapeMode, tape
from llm_tape.cli import main as cli_main
from llm_tape.models import Interaction, LLMCallSummary, RecordedRequest, RecordedResponse


def _seed_cassettes(tmp_path: Path) -> tuple[Path, Path]:
    c1 = tmp_path / "cheap.tape.json"
    c2 = tmp_path / "expensive.tape.json"

    t1 = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "1",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "weather is sunny",
                            "tool_calls": [
                                {
                                    "id": "tc1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )
    )
    t2 = httpx.MockTransport(
        lambda r: httpx.Response(
            500,
            json={
                "id": "2",
                "model": "gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": "error fallback"}}],
                "usage": {"prompt_tokens": 10000, "completion_tokens": 5000},
            },
        )
    )
    with tape(c1, mode=TapeMode.RECORD):
        with httpx.Client(transport=t1) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "w"}]},
            )
    with tape(c2, mode=TapeMode.RECORD):
        with httpx.Client(transport=t2) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "e"}]},
            )
    return c1, c2


def test_cli_search_and_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed_cassettes(tmp_path)

    assert cli_main(["search", str(tmp_path), "--tool", "weather", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["count"] == 1
    assert res["matches"][0]["cassette"] == "cheap.tape.json"

    assert cli_main(["search", str(tmp_path), "--status", "500", "--text", "fallback"]) == 0
    assert "expensive.tape.json" in capsys.readouterr().out

    assert cli_main(["list", str(tmp_path), "--sort", "cost", "--limit", "1", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["count"] == 1
    assert listed["cassettes"][0]["relative_path"] == "expensive.tape.json"

    assert cli_main(["list", str(tmp_path), "--sort", "name"]) == 0
    out = capsys.readouterr().out
    assert out.index("cheap.tape.json") < out.index("expensive.tape.json")


def test_cli_replay_check_and_reparse(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    c1, _ = _seed_cassettes(tmp_path)

    assert cli_main(["replay-check", str(tmp_path)]) == 0
    assert "OK:" in capsys.readouterr().out

    # Tamper with stored fingerprint -> replay-check must fail with exit code 1
    cass = Cassette.load(c1)
    bad_req = RecordedRequest(
        method=cass.interactions[0].request.method,
        url=cass.interactions[0].request.url,
        headers=cass.interactions[0].request.headers,
        body=cass.interactions[0].request.body,
        fingerprint="deadbeef0000000000000000",
    )
    cass.interactions[0] = Interaction(
        request=bad_req,
        response=cass.interactions[0].response,
        latency_ms=cass.interactions[0].latency_ms,
        llm=LLMCallSummary(provider="unknown", model=None),
    )
    cass.save()

    assert cli_main(["replay-check", str(c1), "--json"]) == 1
    chk = json.loads(capsys.readouterr().out)
    assert len(chk["mismatches"]) == 1

    # Reparse restores parsed provider/model/cost
    assert cli_main(["reparse", str(c1), "--dry-run"]) == 0
    assert "WOULD REPARSE" in capsys.readouterr().out
    assert cli_main(["reparse", str(c1)]) == 0
    assert "REPARSED" in capsys.readouterr().out
    reloaded = Cassette.load(c1)
    assert reloaded.interactions[0].llm.provider == "openai"
    assert reloaded.interactions[0].llm.model == "gpt-4o-mini"


def test_cli_export_junit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed_cassettes(tmp_path)
    assert cli_main(["export", str(tmp_path), "--format", "junit"]) == 0
    xml_text = capsys.readouterr().out
    root = ET.fromstring(xml_text)
    assert root.tag == "testsuite"
    assert root.attrib["tests"] == "2"
    assert root.attrib["failures"] == "1"
    cases = {tc.attrib["name"]: tc for tc in root.findall("testcase")}
    assert cases["cheap.tape.json"].find("failure") is None
    assert cases["expensive.tape.json"].find("failure") is not None
    assert RecordedResponse is not None
