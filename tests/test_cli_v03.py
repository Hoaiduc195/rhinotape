"""Tests for enhanced CLI commands: verify --check-secrets, diff dirs/gates, and prune."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

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


def _write_cassette(
    path: Path,
    *,
    recorded_at: str | None = None,
    url: str = "https://api.openai.com/v1/chat/completions",
    req_headers: dict[str, str] | None = None,
    req_body: Any = None,
    resp_body: Any = None,
    stream_chunks: tuple[str, ...] | None = None,
    input_tokens: int = 100,
    cached_input_tokens: int = 0,
    output_tokens: int = 50,
    cost: float = 0.001,
    latency_ms: float = 120.0,
    tools: tuple[ToolCall, ...] = (),
    text: str = "hello",
) -> Path:
    inter = Interaction(
        request=RecordedRequest(
            method="POST",
            url=url,
            headers=req_headers if req_headers is not None else {"authorization": "[REDACTED]"},
            body=req_body if req_body is not None else {"model": "gpt-4o-mini"},
            fingerprint="fp",
            body_encoding="json",
        ),
        response=RecordedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=resp_body if resp_body is not None else {"choices": []},
            body_encoding="json",
            stream_chunks=stream_chunks,
        ),
        latency_ms=latency_ms,
        llm=LLMCallSummary(
            provider="openai",
            model="gpt-4o-mini",
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            ),
            estimated_cost_usd=cost,
            tool_calls=tools,
            text=text,
            is_stream=stream_chunks is not None,
        ),
    )
    c = Cassette(
        path=path,
        interactions=[inter],
        recorded_at=recorded_at or datetime.now(timezone.utc).isoformat(),
    )
    c.save()
    return path


def test_verify_single_file_and_secret_detection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean = _write_cassette(tmp_path / "clean.tape.json")
    assert main(["verify", str(clean), "--check-secrets"]) == 0
    assert "OK: verified 1 cassette(s)" in capsys.readouterr().out

    leaked = _write_cassette(
        tmp_path / "leaked.tape.json",
        req_headers={"x-custom": "Bearer supersecrettoken1234567890"},
        req_body={"note": "key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234"},
        stream_chunks=("data: sk-ant-ABCDEFGHIJ", "KLMNOPQRSTUVWXYZ1234\n\n"),
    )
    rc = main(["verify", str(leaked), "--check-secrets", "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    patterns = {f["pattern"] for f in payload["secret_findings"]}
    assert "bearer_token" in patterns
    assert "openai_key" in patterns
    assert "anthropic_key" in patterns
    # Never print the raw secret value in output
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234" not in json.dumps(payload)


def test_verify_custom_secret_pattern_and_invalid_regex(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tape = _write_cassette(
        tmp_path / "custom.tape.json",
        tools=(ToolCall(name="lookup", arguments={"ssn": "123-45-6789"}),),
    )
    rc = main(["verify", str(tape), "--secret-pattern", r"\b\d{3}-\d{2}-\d{4}\b"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "SECRET" in err
    assert "llm.tool_calls[0].arguments" in err

    # Invalid regex is caught cleanly
    assert main(["verify", str(tape), "--secret-pattern", "[unclosed"]) == 1
    assert "Error:" in capsys.readouterr().err


def test_diff_files_with_regression_gates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write_cassette(
        tmp_path / "a.tape.json",
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=50,
        cost=0.01,
        latency_ms=100.0,
        tools=(ToolCall(name="search", arguments={}),),
    )
    b = _write_cassette(
        tmp_path / "b.tape.json",
        input_tokens=200,
        cached_input_tokens=50,
        output_tokens=100,
        cost=0.025,
        latency_ms=180.0,
        tools=(ToolCall(name="search", arguments={}), ToolCall(name="fetch", arguments={})),
    )

    assert main(["diff", str(a), str(b), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["delta"]["total_tokens"] == 150
    assert data["delta"]["cached_input_tokens"] == 30
    assert data["delta"]["estimated_cost_usd"] == 0.015
    assert data["delta"]["latency_ms"] == 80.0
    assert data["delta"]["tools_changed"] is True

    # Trigger all 3 CI gates
    rc = main(
        [
            "diff",
            str(a),
            str(b),
            "--max-cost-increase",
            "0.01",
            "--max-cost-increase-pct",
            "50",
            "--fail-on-tool-change",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "--max-cost-increase" in err
    assert "--max-cost-increase-pct" in err
    assert "--fail-on-tool-change" in err


def test_diff_directories_and_mismatch_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dir_a = tmp_path / "before"
    dir_b = tmp_path / "after"
    _write_cassette(dir_a / "same.tape.json", cost=0.01)
    _write_cassette(dir_b / "same.tape.json", cost=0.01)
    _write_cassette(dir_a / "changed.tape.json", cost=0.01)
    _write_cassette(dir_b / "changed.tape.json", cost=0.03)
    _write_cassette(dir_a / "removed.tape.json", cost=0.005)
    _write_cassette(dir_b / "added.tape.json", cost=0.02)

    rc = main(["diff", str(dir_a), str(dir_b), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["added"] == ["added.tape.json"]
    assert data["removed"] == ["removed.tape.json"]
    assert len(data["changed"]) == 1
    assert data["unchanged_count"] == 1
    assert data["delta"]["estimated_cost_usd"] == 0.035

    # Human output + gate failure
    rc_fail = main(["diff", str(dir_a), str(dir_b), "--max-cost-increase", "0.01"])
    assert rc_fail == 1
    out = capsys.readouterr().out
    assert "Comparing directories" in out

    # File vs directory error
    assert main(["diff", str(dir_a / "same.tape.json"), str(dir_b)]) == 1
    assert (
        "both diff arguments must be files or both must be directories" in capsys.readouterr().err
    )


def test_prune_unreferenced_and_stale_dry_run_and_apply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_agent.py").write_text(
        """
import pytest

def test_keep_me():
    pass

class TestSuite:
    async def test_class_method(self):
        pass

@pytest.mark.tape(name="custom_named_tape")
def test_with_marker():
    pass
""",
        encoding="utf-8",
    )

    tapes_dir = tmp_path / "tapes"
    old_ts = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    recent_ts = datetime.now(timezone.utc).isoformat()

    keep1 = _write_cassette(tapes_dir / "test_agent" / "test_keep_me.tape.json", recorded_at=old_ts)
    keep2 = _write_cassette(
        tapes_dir / "test_agent" / "test_keep_me_param1_a1b2c3d4.tape.json", recorded_at=recent_ts
    )
    keep3 = _write_cassette(
        tapes_dir / "test_agent" / "TestSuite__test_class_method.tape.json", recorded_at=old_ts
    )
    keep4 = _write_cassette(
        tapes_dir / "test_agent" / "custom_named_tape.tape.json", recorded_at=old_ts
    )
    orphan_recent = _write_cassette(
        tapes_dir / "orphan_dir" / "test_deleted_recently.tape.json", recorded_at=recent_ts
    )
    orphan_old = _write_cassette(
        tapes_dir / "orphan_old_dir" / "test_deleted_long_ago.tape.json", recorded_at=old_ts
    )

    # Error when neither filter given
    assert main(["prune", str(tapes_dir)]) == 1
    assert "specify at least one filter" in capsys.readouterr().err

    # Dry-run with both --tests-dir AND --older-than-days 30 -> only orphan_old matches
    assert (
        main(
            [
                "prune",
                str(tapes_dir),
                "--tests-dir",
                str(tests_dir),
                "--older-than-days",
                "30",
                "--json",
            ]
        )
        == 0
    )
    dry_data = json.loads(capsys.readouterr().out)
    assert dry_data["dry_run"] is True
    assert dry_data["deleted"] == 0
    assert len(dry_data["candidates"]) == 1
    assert dry_data["candidates"][0]["path"] == str(orphan_old)
    assert orphan_old.exists()

    # Apply with --tests-dir only -> deletes both orphans and removes their now-empty parent dirs
    assert main(["prune", str(tapes_dir), "--tests-dir", str(tests_dir), "--apply"]) == 0
    out = capsys.readouterr().out
    assert "DELETED:" in out
    assert keep1.exists() and keep2.exists() and keep3.exists() and keep4.exists()
    assert not orphan_recent.exists()
    assert not orphan_old.exists()
    assert not (tapes_dir / "orphan_dir").exists()
    assert not (tapes_dir / "orphan_old_dir").exists()
    assert tapes_dir.exists()


def test_rv_cli_review_regressions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # P1 #1: --tests-dir pointing to a single file must fail instead of wiping all cassettes
    test_file = tmp_path / "test_one.py"
    test_file.write_text("def test_one(): pass\n", encoding="utf-8")
    tapes_dir = tmp_path / "tapes"
    keep = _write_cassette(tapes_dir / "test_one.tape.json")
    assert main(["prune", str(tapes_dir), "--tests-dir", str(test_file), "--apply"]) == 1
    assert keep.exists()
    capsys.readouterr()

    # P1 #2: symlinks (including broken/external) and dirs named *.tape.json are skipped safely
    outside = _write_cassette(tmp_path / "outside.tape.json")
    (tapes_dir / "ext_link.tape.json").symlink_to(outside)
    (tapes_dir / "broken_link.tape.json").symlink_to(tmp_path / "does_not_exist.tape.json")
    (tapes_dir / "fake_dir.tape.json").mkdir()
    assert main(["prune", str(tapes_dir), "--older-than-days", "0", "--apply"]) == 0
    assert outside.exists()
    assert (tapes_dir / "ext_link.tape.json").is_symlink()
    assert (tapes_dir / "broken_link.tape.json").is_symlink()
    assert (tapes_dir / "fake_dir.tape.json").is_dir()
    capsys.readouterr()

    # P2 #3 & #4: Google API key ending in '-', no duplicate openai+anthropic hit, metadata scan
    google_key_trailing_dash = "AIza" + "A" * 34 + "-"
    anth_key = "sk-ant-" + "B" * 24
    c_path = _write_cassette(
        tmp_path / "sec.tape.json",
        req_body={"g": google_key_trailing_dash, "a": anth_key},
    )
    c = Cassette.load(c_path)
    c.metadata["leaked_meta"] = "ghp_" + "C" * 36
    c.save()
    assert main(["verify", str(c_path), "--check-secrets", "--json"]) == 1
    sec_data = json.loads(capsys.readouterr().out)
    patterns_found = [f["pattern"] for f in sec_data["secret_findings"]]
    assert "google_api_key" in patterns_found
    assert "anthropic_key" in patterns_found
    assert "openai_key" not in patterns_found
    assert any(
        f["location"] == "metadata" and f["pattern"] == "github_token"
        for f in sec_data["secret_findings"]
    )

    # P2 #5: nan/inf/overflow in --older-than-days rejected cleanly
    assert main(["prune", str(tapes_dir), "--older-than-days", "nan"]) == 1
    assert main(["prune", str(tapes_dir), "--older-than-days", "inf"]) == 1
    assert main(["prune", str(tapes_dir), "--older-than-days", "1e12"]) == 1
    capsys.readouterr()

    # P2 #6: exact percentage boundary (0.0003 -> 0.00039 = 30.0%) does not false-fail
    b1 = _write_cassette(tmp_path / "b1.tape.json", cost=0.0003)
    b2 = _write_cassette(tmp_path / "b2.tape.json", cost=0.00039)
    assert main(["diff", str(b1), str(b2), "--max-cost-increase-pct", "30"]) == 0
    capsys.readouterr()
