"""Tests for Wave 7: Cassette.content_hash() and CLI `llm-tape stats` breakdown."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_tape import Cassette
from llm_tape.cli import main
from llm_tape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    Usage,
)


def _inter(
    provider: str,
    model: str,
    *,
    inp: int,
    cached: int,
    out: int,
    cost: float,
    latency_ms: float,
) -> Interaction:
    return Interaction(
        request=RecordedRequest("POST", "https://api.example.com/v1", {}, {"m": model}, "fp"),
        response=RecordedResponse(200, {"content-type": "application/json"}, {"ok": True}),
        latency_ms=latency_ms,
        llm=LLMCallSummary(
            provider=provider,
            model=model,
            usage=Usage(input_tokens=inp, output_tokens=out, cached_input_tokens=cached),
            estimated_cost_usd=cost,
        ),
    )


def test_cassette_content_hash_is_deterministic(tmp_path: Path) -> None:
    p1 = tmp_path / "a.tape.json"
    p2 = tmp_path / "b.tape.json.gz"
    inter = _inter("openai", "gpt-4o-mini", inp=100, cached=50, out=20, cost=0.001, latency_ms=50.0)
    c1 = Cassette(path=p1, interactions=[inter], recorded_at="2026-01-01T00:00:00Z")
    c2 = Cassette(path=p2, interactions=[inter], recorded_at="2026-12-31T00:00:00Z")
    c1.save()
    c2.save()

    h1 = Cassette.load(p1).content_hash()
    h2 = Cassette.load(p2).content_hash()
    assert len(h1) == 64
    assert h1 == h2


def test_cli_stats_provider_model_breakdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tapes = tmp_path / "tapes"
    Cassette(
        path=tapes / "c1.tape.json",
        interactions=[
            _inter(
                "openai", "gpt-4o-mini", inp=1000, cached=500, out=100, cost=0.002, latency_ms=100.0
            ),
            _inter(
                "openai", "gpt-4o-mini", inp=1000, cached=500, out=100, cost=0.002, latency_ms=300.0
            ),
        ],
    ).save()
    Cassette(
        path=tapes / "c2.tape.json.gz",
        interactions=[
            _inter(
                "anthropic",
                "claude-sonnet-4",
                inp=500,
                cached=250,
                out=200,
                cost=0.010,
                latency_ms=450.0,
            ),
        ],
    ).save()

    assert main(["stats", str(tapes), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    # Sorted by highest cost first
    assert data["models"][0]["model"] == "claude-sonnet-4"
    assert data["models"][1]["model"] == "gpt-4o-mini"
    assert data["models"][1]["calls"] == 2
    assert data["models"][1]["cache_hit_pct"] == 50.0
    assert data["models"][1]["avg_latency_ms"] == 200.0
    assert data["models"][1]["p95_latency_ms"] == 300.0
    assert data["totals"]["calls"] == 3
    assert data["totals"]["estimated_cost_usd"] == 0.014

    # Human table output
    assert main(["stats", str(tapes)]) == 0
    out = capsys.readouterr().out
    assert "claude-sonnet-4" in out
    assert "gpt-4o-mini" in out
    assert "TOTAL" in out
