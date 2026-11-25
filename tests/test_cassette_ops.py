"""Tests for Batch 1 (Waves 18-22): Cassette merge, filter, split, clone, stats, sort, validate."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_tape import Cassette
from llm_tape.cli import main as cli_main
from llm_tape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    Usage,
)


def _mk_inter(
    fp: str, *, latency: float = 10.0, tokens: int = 10, cost: float = 0.001
) -> Interaction:
    return Interaction(
        request=RecordedRequest("POST", f"https://api.openai.com/v1/{fp}", {}, {"id": fp}, fp),
        response=RecordedResponse(200, {"content-type": "application/json"}, {"ok": fp}),
        latency_ms=latency,
        llm=LLMCallSummary(
            provider="openai",
            model="gpt-4o-mini",
            usage=Usage(input_tokens=tokens, output_tokens=0),
            estimated_cost_usd=cost,
            text=f"reply-{fp}",
        ),
    )


def test_wave18_cassette_merge_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p1 = tmp_path / "a.tape.json"
    p2 = tmp_path / "b.tape.json"
    Cassette(path=p1, interactions=[_mk_inter("fp1"), _mk_inter("fp2")], metadata={"a": 1}).save()
    Cassette(path=p2, interactions=[_mk_inter("fp2"), _mk_inter("fp3")], metadata={"b": 2}).save()

    out = tmp_path / "merged.tape.json.gz"
    assert cli_main(["merge", str(p1), str(p2), "-o", str(out), "--deduplicate"]) == 0
    assert "Merged 2 cassette(s) (3 interaction(s))" in capsys.readouterr().out
    loaded = Cassette.load(out)
    assert [i.request.fingerprint for i in loaded.interactions] == ["fp1", "fp2", "fp3"]
    assert loaded.metadata == {"a": 1, "b": 2}


def test_wave19_cassette_filter_slice_and_deduplicate(tmp_path: Path) -> None:
    c = Cassette(
        path=tmp_path / "ops.tape.json",
        interactions=[_mk_inter("a", latency=10), _mk_inter("b", latency=50), _mk_inter("a")],
    )
    filtered = c.filter(lambda i: i.latency_ms >= 20)
    assert [i.request.fingerprint for i in filtered.interactions] == ["b"]
    sliced = c.slice(0, 2)
    assert [i.request.fingerprint for i in sliced.interactions] == ["a", "b"]
    deduped = c.deduplicate()
    assert [i.request.fingerprint for i in deduped.interactions] == ["a", "b"]


def test_wave20_cli_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "multi.tape.json"
    Cassette(path=src, interactions=[_mk_inter("x"), _mk_inter("y"), _mk_inter("z")]).save()
    out_dir = tmp_path / "parts"
    assert cli_main(["split", str(src), "-o", str(out_dir), "--gzip"]) == 0
    assert "Split" in capsys.readouterr().out
    parts = sorted(out_dir.glob("*.tape.json.gz"))
    assert len(parts) == 3
    assert Cassette.load(parts[1]).interactions[0].request.fingerprint == "y"


def test_wave21_cassette_clone_and_stats(tmp_path: Path) -> None:
    c = Cassette(
        path=tmp_path / "orig.tape.json",
        interactions=[_mk_inter("a", latency=12.5, tokens=30, cost=0.002)],
        metadata={"env": "ci"},
    )
    cloned = c.clone(tmp_path / "copy.tape.json.gz")
    assert cloned.path == tmp_path / "copy.tape.json.gz"
    assert cloned.content_hash() == c.content_hash()
    st = cloned.stats()
    assert st["calls"] == 1
    assert st["total_tokens"] == 30
    assert st["estimated_cost_usd"] == 0.002
    assert st["latency_ms"] == 12.5


def test_wave22_cassette_sort_and_validate(tmp_path: Path) -> None:
    c = Cassette(
        path=tmp_path / "sort.tape.json",
        interactions=[
            _mk_inter("c", latency=5),
            _mk_inter("a", latency=30),
            _mk_inter("b", latency=15),
        ],
    )
    assert [i.request.fingerprint for i in c.sort_interactions().interactions] == ["a", "b", "c"]
    by_lat = c.sort_interactions(key=lambda i: i.latency_ms, reverse=True)
    assert [i.request.fingerprint for i in by_lat.interactions] == ["a", "b", "c"]
    assert c.validate() == []

    bad = Cassette(
        path=tmp_path / "bad.tape.json",
        interactions=[
            Interaction(
                request=RecordedRequest("", "", {}, None, "", body_encoding="weird"),
                response=RecordedResponse(
                    999, {}, None, body_encoding="stream", stream_chunks=None
                ),
                latency_ms=-1.0,
            )
        ],
    )
    issues = bad.validate()
    assert len(issues) == 7
