"""Tests for deprecated model detection and verify --max-age / --warn-deprecated-models."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rhinotape.cassette import Cassette
from rhinotape.cli import main
from rhinotape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    Usage,
)
from rhinotape.pricing import DEPRECATED_MODELS, get_deprecation_notice


def _write_cassette(
    path: Path,
    *,
    model: str = "gpt-4o",
    provider: str = "openai",
    recorded_at: str | None = None,
) -> Path:
    inter = Interaction(
        request=RecordedRequest(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={"model": model, "messages": [{"role": "user", "content": "hi"}]},
            fingerprint="fp-verify-test",
        ),
        response=RecordedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        ),
        llm=LLMCallSummary(
            provider=provider,
            model=model,
            text="ok",
            usage=Usage(input_tokens=10, output_tokens=5),
            estimated_cost_usd=0.0001,
        ),
        latency_ms=42.0,
    )
    cassette = Cassette(
        path=path,
        interactions=[inter],
        recorded_at=recorded_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    cassette.save()
    return path


def test_get_deprecation_notice_exact_prefix_and_normalization() -> None:
    assert "gpt-3.5-turbo-0301" in DEPRECATED_MODELS
    assert get_deprecation_notice("gpt-3.5-turbo-0301") is not None
    assert get_deprecation_notice("models/gemini-1.5-flash-002") is not None
    assert get_deprecation_notice("us.anthropic.claude-3-sonnet-20240229-v1:0") is not None
    assert get_deprecation_notice("openai/o1-preview") is not None
    assert get_deprecation_notice("gpt-4o") is None
    assert get_deprecation_notice("claude-sonnet-4") is None
    assert get_deprecation_notice(None) is None
    assert get_deprecation_notice("") is None
    custom = get_deprecation_notice("custom-old", deprecated_models={"custom-old": "use v2"})
    assert custom == "use v2"


def test_cli_verify_max_age_pass_fail_and_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fresh = _write_cassette(
        tmp_path / "fresh.tape.json",
        recorded_at=(datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
    )
    stale = _write_cassette(
        tmp_path / "stale.tape.json",
        recorded_at=(datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
    )

    assert main(["verify", str(fresh), "--max-age", "30d"]) == 0
    assert "OK: verified 1 cassette(s)" in capsys.readouterr().out

    assert main(["verify", str(stale), "--max-age", "30d"]) == 1
    err = capsys.readouterr().err
    assert "STALE" in err
    assert "stale cassette(s)" in err

    assert main(["verify", str(stale), "--max-age", "2w", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert len(payload["stale_findings"]) == 1
    assert payload["stale_findings"][0]["max_age"] == "2w"
    assert payload["stale_findings"][0]["age_days"] >= 44.0

    # Hours and plain numeric days are also accepted
    assert main(["verify", str(fresh), "--max-age", "72h"]) == 0
    capsys.readouterr()
    assert main(["verify", str(fresh), "--max-age", "5"]) == 0
    capsys.readouterr()

    # Invalid --max-age values fail with clean error
    assert main(["verify", str(fresh), "--max-age=-1d"]) == 1
    assert "Error:" in capsys.readouterr().err
    assert main(["verify", str(fresh), "--max-age", "abc"]) == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_verify_deprecated_models_warn_and_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old_tape = _write_cassette(
        tmp_path / "legacy.tape.json",
        model="gpt-4-0314",
        provider="openai",
    )
    modern_tape = _write_cassette(
        tmp_path / "modern.tape.json",
        model="gpt-4.1-mini",
        provider="openai",
    )

    # Warn mode exits 0 while printing warning
    assert main(["verify", str(old_tape), "--warn-deprecated-models"]) == 0
    captured = capsys.readouterr()
    assert "WARN DEPRECATED" in captured.err
    assert "gpt-4-0314" in captured.err
    assert "1 deprecated model warning(s)" in captured.out

    # Fail mode exits 1 and emits JSON findings
    assert main(["verify", str(old_tape), "--fail-on-deprecated-models", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert len(payload["deprecated_findings"]) == 1
    assert payload["deprecated_findings"][0]["model"] == "gpt-4-0314"

    # Modern model passes cleanly in fail mode
    assert main(["verify", str(modern_tape), "--fail-on-deprecated-models"]) == 0
    assert "OK: verified 1 cassette(s)" in capsys.readouterr().out

    # Case-insensitive model deduplication in deprecated warnings
    dup_cassette = Cassette.load(old_tape)
    inter2 = Interaction(
        request=dup_cassette.interactions[0].request,
        response=dup_cassette.interactions[0].response,
        llm=LLMCallSummary(provider="openai", model="GPT-4-0314", text="ok"),
        latency_ms=10.0,
    )
    dup_path = tmp_path / "dup_case.tape.json"
    Cassette(path=dup_path, interactions=[dup_cassette.interactions[0], inter2]).save()
    assert main(["verify", str(dup_path), "--warn-deprecated-models", "--json"]) == 0
    dup_payload = json.loads(capsys.readouterr().out)
    assert len(dup_payload["deprecated_findings"]) == 1
