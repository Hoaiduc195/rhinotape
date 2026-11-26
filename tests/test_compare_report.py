"""Tests for Model Migration A/B Comparison Report engine and `rhinotape compare` CLI."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rhinotape import (
    Cassette,
    CassettePairComparison,
    MigrationReport,
    TapeNotFoundError,
    compare_cassettes,
    evaluate_compare_gates,
    render_compare_markdown,
    render_compare_text,
)
from rhinotape.cli import main as cli_main
from rhinotape.compare import _normalize_cassette_stem
from rhinotape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    ToolCall,
    Usage,
)


def _mk_inter(
    fp: str,
    *,
    model: str | None = "gpt-4o",
    in_tok: int = 100,
    out_tok: int = 50,
    cost: float = 0.01,
    latency: float = 200.0,
    tools: tuple[str, ...] = (),
    text: str = "The quick brown fox jumps over the lazy dog.",
    status_code: int = 200,
) -> Interaction:
    return Interaction(
        request=RecordedRequest(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={"id": fp},
            fingerprint=fp,
        ),
        response=RecordedResponse(
            status_code=status_code,
            headers={"content-type": "application/json"},
            body={"ok": status_code < 400},
        ),
        latency_ms=latency,
        llm=LLMCallSummary(
            provider="openai",
            model=model,
            usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
            estimated_cost_usd=cost,
            tool_calls=tuple(ToolCall(name=t) for t in tools),
            text=text,
        ),
    )


def test_single_file_comparison_metrics_and_strict_json(tmp_path: Path) -> None:
    base_file = tmp_path / "agent.tape.json"
    cand_file = tmp_path / "agent_mini.tape.json"

    Cassette(
        path=base_file,
        interactions=[
            _mk_inter(
                "1",
                model="gpt-4o",
                in_tok=200,
                out_tok=100,
                cost=0.010,
                latency=400.0,
                tools=("search_docs",),
                text="Answer: Paris is the capital of France.",
            ),
            _mk_inter(
                "2",
                model="gpt-4o",
                in_tok=100,
                out_tok=50,
                cost=0.005,
                latency=100.0,
                status_code=500,
                text="",
            ),
        ],
    ).save()

    Cassette(
        path=cand_file,
        interactions=[
            _mk_inter(
                "1",
                model="gpt-4o-mini",
                in_tok=200,
                out_tok=90,
                cost=0.003,
                latency=150.0,
                tools=("search_docs",),
                text="Answer:   Paris is the capital of France.\n",
            ),
            _mk_inter(
                "2",
                model="gpt-4o-mini",
                in_tok=100,
                out_tok=40,
                cost=0.0015,
                latency=50.0,
                status_code=200,
                text="",
            ),
        ],
    ).save()

    report = compare_cassettes(base_file, cand_file)
    assert isinstance(report, MigrationReport)
    assert len(report.pairs) == 1
    assert report.only_in_baseline == []
    assert report.only_in_candidate == []

    pair = report.pairs[0]
    assert isinstance(pair, CassettePairComparison)
    assert pair.baseline_models == ["gpt-4o"]
    assert pair.candidate_models == ["gpt-4o-mini"]
    assert pair.baseline_calls == 2
    assert pair.candidate_calls == 2
    assert pair.baseline_tokens == 450
    assert pair.candidate_tokens == 430
    assert pair.baseline_cost_usd == pytest.approx(0.015)
    assert pair.candidate_cost_usd == pytest.approx(0.0045)
    assert pair.cost_delta_usd == pytest.approx(-0.0105)
    assert pair.cost_change_pct == pytest.approx(-70.0)
    assert pair.baseline_latency_ms == pytest.approx(500.0)
    assert pair.candidate_latency_ms == pytest.approx(200.0)
    assert pair.latency_delta_ms == pytest.approx(-300.0)
    assert pair.latency_change_pct == pytest.approx(-60.0)
    assert pair.baseline_tools == ["search_docs"]
    assert pair.candidate_tools == ["search_docs"]
    assert pair.tools_match is True
    assert pair.baseline_errors == 1
    assert pair.candidate_errors == 0
    # Normalized whitespace makes identical text -> similarity 1.0
    assert pair.text_similarity == pytest.approx(1.0)

    assert report.tool_drift_count == 0
    assert report.avg_text_similarity == pytest.approx(1.0)
    assert report.cost_change_pct == pytest.approx(-70.0)

    # Strict RFC 8259 JSON serialization
    serialized = json.dumps(report.to_dict(), allow_nan=False)
    parsed = json.loads(serialized)
    assert parsed["pairs"][0]["tools_match"] is True


def test_directory_pairing_gz_unpaired_zero_cost_and_renderers(tmp_path: Path) -> None:
    base_dir = tmp_path / "baseline"
    cand_dir = tmp_path / "candidate"

    # Pair 1: nested .tape.json vs .tape.json.gz, 0 -> 0 cost, both empty text
    Cassette(
        path=base_dir / "flows" / "empty_text.tape.json",
        interactions=[
            _mk_inter("a", model="m|base", cost=0.0, latency=0.0, tools=("t1",), text="")
        ],
    ).save()
    Cassette(
        path=cand_dir / "flows" / "empty_text.tape.json.gz",
        interactions=[
            _mk_inter("a", model="m|cand", cost=0.0, latency=0.0, tools=("t2",), text="   ")
        ],
    ).save()

    # Pair 2: 0 -> positive cost & latency, partial text similarity, pipe in filename
    Cassette(
        path=base_dir / "zero|to_paid.tape.json",
        interactions=[
            _mk_inter("b", model="local-llama", cost=0.0, latency=0.0, text="Hello world")
        ],
    ).save()
    Cassette(
        path=cand_dir / "zero|to_paid.tape.json",
        interactions=[
            _mk_inter("b", model="gpt-4o-mini", cost=0.002, latency=50.0, text="Hello brave world")
        ],
    ).save()

    # Unpaired files
    Cassette(path=base_dir / "legacy_only.tape.json", interactions=[_mk_inter("x")]).save()
    Cassette(path=cand_dir / "new_only.tape.json.gz", interactions=[_mk_inter("y")]).save()

    report = compare_cassettes(base_dir, cand_dir)
    assert report.only_in_baseline == ["legacy_only"]
    assert report.only_in_candidate == ["new_only"]
    assert [p.name for p in report.pairs] == ["flows/empty_text", "zero|to_paid"]
    assert report.tool_drift_count == 1

    p_empty, p_inf = report.pairs
    assert p_empty.cost_change_pct is None
    assert p_empty.latency_change_pct is None
    assert p_empty.text_similarity == 1.0
    assert p_empty.tools_match is False

    assert p_inf.cost_change_pct is not None and math.isinf(p_inf.cost_change_pct)
    assert p_inf.latency_change_pct is not None and math.isinf(p_inf.latency_change_pct)
    assert 0.5 < p_inf.text_similarity < 1.0

    assert report.cost_change_pct is not None and math.isinf(report.cost_change_pct)
    # to_dict() must still serialize strictly without NaN/Infinity tokens
    raw_json = json.dumps(report.to_dict(), allow_nan=False)
    assert "flows/empty_text" in raw_json

    txt = render_compare_text(report)
    assert "flows/empty_text" in txt
    assert "legacy_only" in txt
    assert "new_only" in txt

    md = render_compare_markdown(report)
    assert "| Cassette |" in md
    # Unescaped pipes inside cell names/models must be escaped as \|
    assert "zero\\|to_paid" in md
    assert "m\\|base" in md
    assert "m\\|cand" in md
    assert "legacy_only" in md
    assert "new_only" in md


def test_compare_validation_errors(tmp_path: Path) -> None:
    f = tmp_path / "one.tape.json"
    Cassette(path=f, interactions=[]).save()
    d = tmp_path / "dir"
    d.mkdir()

    with pytest.raises(ValueError, match="directories"):
        compare_cassettes(f, d)

    with pytest.raises((FileNotFoundError, TapeNotFoundError)):
        compare_cassettes(tmp_path / "missing.tape.json", f)

    with pytest.raises((FileNotFoundError, TapeNotFoundError)):
        compare_cassettes(f, tmp_path / "missing2.tape.json")


def test_cli_compare_modes_gates_and_invalid_args(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_p = tmp_path / "base.tape.json"
    cand_p = tmp_path / "cand.tape.json"

    Cassette(
        path=base_p,
        interactions=[
            _mk_inter(
                "q1",
                model="gpt-4o",
                cost=0.010,
                tools=("lookup",),
                text="Tokyo is the capital of Japan.",
            )
        ],
    ).save()
    Cassette(
        path=cand_p,
        interactions=[
            _mk_inter(
                "q1",
                model="gpt-4o-mini",
                cost=0.012,  # +20% cost
                tools=("lookup", "extra_tool"),  # tool drift
                text="Tokyo is Japan's capital city.",
            )
        ],
    ).save()

    # Passing gates
    rc_pass = cli_main(
        [
            "compare",
            str(base_p),
            str(cand_p),
            "--min-similarity",
            "0.5",
            "--max-cost-increase-pct",
            "25.0",
        ]
    )
    assert rc_pass == 0
    out_txt = capsys.readouterr().out
    assert "gpt-4o" in out_txt and "gpt-4o-mini" in out_txt

    # Markdown output + failing --fail-on-tool-drift
    rc_drift = cli_main(["compare", str(base_p), str(cand_p), "--markdown", "--fail-on-tool-drift"])
    captured_drift = capsys.readouterr()
    assert rc_drift == 1
    assert "###" in captured_drift.out
    assert "tool" in captured_drift.err.lower()

    # Failing --min-similarity and --max-cost-increase-pct in --json mode
    rc_json = cli_main(
        [
            "compare",
            str(base_p),
            str(cand_p),
            "--json",
            "--min-similarity",
            "0.99",
            "--max-cost-increase-pct",
            "10.0",
        ]
    )
    captured_json = capsys.readouterr()
    assert rc_json == 1
    payload = json.loads(captured_json.out)
    assert payload["passed"] is False
    assert len(payload["violations"]) == 2

    # 0 -> positive cost fails --max-cost-increase-pct 0.0
    zero_base = tmp_path / "zero.tape.json"
    Cassette(path=zero_base, interactions=[_mk_inter("q1", cost=0.0)]).save()
    assert (
        cli_main(
            [
                "compare",
                str(zero_base),
                str(cand_p),
                "--max-cost-increase-pct",
                "100.0",
            ]
        )
        == 1
    )
    assert "cost" in capsys.readouterr().err.lower()

    # Invalid flag values exit 1
    for bad_args in [
        ["compare", str(base_p), str(cand_p), "--min-similarity", "-0.1"],
        ["compare", str(base_p), str(cand_p), "--min-similarity", "1.2"],
        ["compare", str(base_p), str(cand_p), "--min-similarity", "nan"],
        ["compare", str(base_p), str(cand_p), "--max-cost-increase-pct", "-5"],
        ["compare", str(base_p), str(cand_p), "--max-cost-increase-pct", "inf"],
        ["compare", str(base_p), str(cand_p), "--markdown", "--json"],
        ["compare", str(base_p), str(tmp_path)],
    ]:
        assert cli_main(bad_args) == 1
        capsys.readouterr()


def test_stem_collision_windows_separators_and_empty_dirs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. Stem collision (foo.tape.json AND foo.tape.json.gz in same dir) raises ValueError.
    col_base = tmp_path / "col_base"
    col_cand = tmp_path / "col_cand"
    Cassette(path=col_base / "sub" / "foo.tape.json", interactions=[_mk_inter("1")]).save()
    Cassette(path=col_base / "sub" / "foo.tape.json.gz", interactions=[_mk_inter("2")]).save()
    Cassette(path=col_cand / "sub" / "foo.tape.json", interactions=[_mk_inter("1")]).save()

    with pytest.raises(ValueError, match="Ambiguous cassette stem 'sub/foo'"):
        compare_cassettes(col_base, col_cand)
    assert cli_main(["compare", str(col_base), str(col_cand)]) == 1
    assert "Ambiguous cassette stem" in capsys.readouterr().err

    # 2. Windows backslash relative paths normalize cleanly to POSIX stems.
    assert _normalize_cassette_stem(r"flows\nested\agent.tape.json.gz") == "flows/nested/agent"

    # 3. Empty directories (0 matched pairs): clean metrics, strict JSON, and CI gate failure.
    empty_b = tmp_path / "empty_b"
    empty_c = tmp_path / "empty_c"
    empty_b.mkdir()
    empty_c.mkdir()

    rep_empty = compare_cassettes(empty_b, empty_c)
    assert rep_empty.pairs == []
    assert rep_empty.avg_text_similarity == 0.0
    assert rep_empty.cost_change_pct is None
    assert rep_empty.latency_change_pct is None
    assert json.loads(json.dumps(rep_empty.to_dict(), allow_nan=False))["matched_pairs"] == 0
    assert "n/a" in render_compare_text(rep_empty)
    assert "n/a" in render_compare_markdown(rep_empty)

    # Without gates, 0 matched pairs exits 0; WITH any gate, 0 matched pairs fails (exits 1)!
    assert cli_main(["compare", str(empty_b), str(empty_c)]) == 0
    capsys.readouterr()
    for gate_flag in (
        ["--min-similarity", "0.8"],
        ["--max-cost-increase-pct", "10.0"],
        ["--fail-on-tool-drift"],
    ):
        assert cli_main(["compare", str(empty_b), str(empty_c), *gate_flag]) == 1
        assert "No matched cassette pairs" in capsys.readouterr().err


def test_text_similarity_edge_cases_markdown_escaping_and_gate_boundaries(
    tmp_path: Path,
) -> None:
    # One side empty & other non-empty -> 0.0
    b_empty = tmp_path / "b_empty.tape.json"
    c_nonempty = tmp_path / "c_nonempty.tape.json"
    Cassette(path=b_empty, interactions=[_mk_inter("1", text="")]).save()
    Cassette(path=c_nonempty, interactions=[_mk_inter("1", text="Non-empty output")]).save()
    assert compare_cassettes(b_empty, c_nonempty).pairs[0].text_similarity == 0.0
    assert compare_cassettes(c_nonempty, b_empty).pairs[0].text_similarity == 0.0

    # Multi-interaction concatenated text across turns vs single combined turn -> 1.0
    b_multi = tmp_path / "b_multi.tape.json"
    c_single = tmp_path / "c_single.tape.json"
    Cassette(
        path=b_multi,
        interactions=[_mk_inter("1", text="Step one."), _mk_inter("2", text="Step two.")],
    ).save()
    Cassette(
        path=c_single,
        interactions=[_mk_inter("1", text="Step one. Step two.")],
    ).save()
    assert compare_cassettes(b_multi, c_single).pairs[0].text_similarity == 1.0

    # Exact gate boundary equality: text_similarity == min_similarity AND
    # cost_change_pct == max_cost_increase_pct MUST pass!
    b_bound = tmp_path / "b_bound.tape.json"
    c_bound = tmp_path / "c_bound.tape.json"
    Cassette(
        path=b_bound, interactions=[_mk_inter("1", cost=0.010, text="Exact match text")]
    ).save()
    Cassette(
        path=c_bound, interactions=[_mk_inter("1", cost=0.012, text="Exact match text")]
    ).save()
    rep_bound = compare_cassettes(b_bound, c_bound)
    assert rep_bound.cost_change_pct == pytest.approx(20.0)
    assert rep_bound.pairs[0].text_similarity == 1.0
    assert (
        evaluate_compare_gates(
            rep_bound,
            min_similarity=1.0,
            max_cost_increase_pct=20.0,
            fail_on_tool_drift=True,
        )
        == []
    )

    # Python API gate validation rejects NaN, negative, >1.0, and bools
    for bad_sim in (-0.01, 1.01, float("nan"), float("inf"), True, False):
        with pytest.raises(ValueError, match="min_similarity"):
            evaluate_compare_gates(rep_bound, min_similarity=bad_sim)
    for bad_cost in (-0.1, float("nan"), float("inf"), True, False):
        with pytest.raises(ValueError, match="max_cost_increase_pct"):
            evaluate_compare_gates(rep_bound, max_cost_increase_pct=bad_cost)
    with pytest.raises(ValueError, match="fail_on_tool_drift"):
        evaluate_compare_gates(rep_bound, fail_on_tool_drift="yes")  # type: ignore[arg-type]

    # Non-finite values on synthetic/manual MigrationReport still serialize as strict RFC 8259 JSON
    nan_pair = CassettePairComparison(
        name="bad`name|row\n2",
        baseline_models=["gpt`4o|v1\nx"],
        candidate_models=["gpt`4o-mini"],
        baseline_calls=1,
        candidate_calls=1,
        baseline_tokens=10,
        candidate_tokens=10,
        baseline_cost_usd=float("nan"),
        candidate_cost_usd=0.01,
        cost_delta_usd=float("nan"),
        cost_change_pct=float("nan"),
        baseline_latency_ms=float("inf"),
        candidate_latency_ms=10.0,
        latency_delta_ms=float("-inf"),
        latency_change_pct=float("-inf"),
        baseline_tools=[],
        candidate_tools=[],
        tools_match=True,
        baseline_errors=0,
        candidate_errors=0,
        text_similarity=float("nan"),
    )
    nan_report = MigrationReport(
        baseline_path="b",
        candidate_path="c",
        pairs=[nan_pair],
        only_in_baseline=["unpaired`one|a\nb"],
        only_in_candidate=[],
        baseline_models=["gpt`4o|v1\nx"],
        candidate_models=["gpt`4o-mini"],
        baseline_calls=1,
        candidate_calls=1,
        baseline_tokens=10,
        candidate_tokens=10,
        token_delta=0,
        baseline_cost_usd=float("nan"),
        candidate_cost_usd=0.01,
        cost_delta_usd=float("nan"),
        cost_change_pct=float("inf"),
        baseline_latency_ms=10.0,
        candidate_latency_ms=5.0,
        latency_delta_ms=-5.0,
        latency_change_pct=float("-inf"),
        baseline_errors=0,
        candidate_errors=0,
        tool_drift_count=0,
        avg_text_similarity=float("nan"),
    )
    strict_json = json.dumps(nan_report.to_dict(), allow_nan=False)
    parsed_nan = json.loads(strict_json)
    assert parsed_nan["cost_change_pct"] == "inf"
    assert parsed_nan["latency_change_pct"] == "-inf"
    assert parsed_nan["baseline_cost_usd"] is None

    # Markdown escaping: no multi-line table rows, escaped pipes and safe backticks
    md = render_compare_markdown(nan_report)
    table_rows = [line for line in md.splitlines() if line.startswith("|")]
    assert len(table_rows) == 14
    for row in table_rows:
        assert "\r" not in row and "\n" not in row
    assert "bad\\`name\\|row 2" in md
    assert "`gpt'4o\\|v1 x`" in md
    assert "`unpaired'one\\|a b`" in md
