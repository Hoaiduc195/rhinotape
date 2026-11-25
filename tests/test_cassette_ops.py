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
