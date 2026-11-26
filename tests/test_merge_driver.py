"""Tests for Cassette.merge strategies, 3-way merge, and CLI Git merge driver."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rhinotape import Cassette, TapeError
from rhinotape.cli import main as cli_main
from rhinotape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    Usage,
)


def _inter(
    fp: str,
    *,
    text: str = "ok",
    latency: float = 10.0,
    cost: float = 0.001,
    tokens: int = 20,
) -> Interaction:
    return Interaction(
        request=RecordedRequest(
            method="POST",
            url=f"https://api.openai.com/v1/chat/{fp}",
            headers={"content-type": "application/json"},
            body={"prompt": fp},
            fingerprint=fp,
        ),
        response=RecordedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"reply": text},
        ),
        latency_ms=latency,
        llm=LLMCallSummary(
            provider="openai",
            model="gpt-4o-mini",
            usage=Usage(input_tokens=tokens, output_tokens=5),
            estimated_cost_usd=cost,
            text=text,
        ),
    )


def test_merge_on_duplicate_strategies_and_backward_compat(tmp_path: Path) -> None:
    c1 = Cassette(
        path=tmp_path / "c1.tape.json",
        interactions=[_inter("fp_shared", text="v1"), _inter("fp_only_1", text="o1")],
        recorded_at="2026-05-10T12:00:00Z",
        metadata={"k": 1, "from_1": True},
    )
    c2 = Cassette(
        path=tmp_path / "c2.tape.json",
        interactions=[_inter("fp_shared", text="v2"), _inter("fp_only_2", text="o2")],
        recorded_at="2026-05-09T12:00:00Z",
        metadata={"k": 2, "from_2": True},
    )
    out = tmp_path / "out.tape.json"

    # Default keep-all
    m_all = Cassette.merge(c1, c2, output_path=out)
    assert [i.llm.text for i in m_all.interactions] == ["v1", "o1", "v2", "o2"]
    assert m_all.metadata == {"k": 2, "from_1": True, "from_2": True}

    # Backward-compat deduplicate=True -> keep-first
    m_dedup = Cassette.merge(c1, c2, output_path=out, deduplicate=True)
    assert [i.llm.text for i in m_dedup.interactions] == ["v1", "o1", "o2"]

    # Explicit keep-first
    m_first = Cassette.merge(c1, c2, output_path=out, on_duplicate="keep-first")
    assert [i.llm.text for i in m_first.interactions] == ["v1", "o1", "o2"]

    # Explicit keep-last
    m_last = Cassette.merge(c1, c2, output_path=out, on_duplicate="keep-last")
    texts_by_fp = {i.request.fingerprint: i.llm.text for i in m_last.interactions}
    assert len(m_last.interactions) == 3
    assert texts_by_fp["fp_shared"] == "v2"

    # keep-newest: c1 (May 10) is newer than c2 (May 09) -> v1 wins
    m_newest_c1 = Cassette.merge(c1, c2, output_path=out, on_duplicate="keep-newest")
    assert {i.request.fingerprint: i.llm.text for i in m_newest_c1.interactions}["fp_shared"] == (
        "v1"
    )

    # keep-newest: reverse order c2 then c1 -> c1 (May 10) still wins
    m_newest_rev = Cassette.merge(c2, c1, output_path=out, on_duplicate="keep-newest")
    assert {i.request.fingerprint: i.llm.text for i in m_newest_rev.interactions}["fp_shared"] == (
        "v1"
    )

    # keep-newest tie and missing recorded_at -> fall back to later input order
    c_tie_1 = Cassette(
        path=tmp_path / "t1.tape.json",
        interactions=[_inter("fp_shared", text="tie1")],
        recorded_at="2026-05-10T00:00:00+00:00",
    )
    c_tie_2 = Cassette(
        path=tmp_path / "t2.tape.json",
        interactions=[_inter("fp_shared", text="tie2")],
        recorded_at="2026-05-10T00:00:00Z",
    )
    m_tie = Cassette.merge(c_tie_1, c_tie_2, output_path=out, on_duplicate="keep-newest")
    assert [i.llm.text for i in m_tie.interactions] == ["tie2"]

    c_miss_1 = Cassette(
        path=tmp_path / "m1.tape.json",
        interactions=[_inter("fp_shared", text="miss1")],
        recorded_at="",
    )
    c_miss_2 = Cassette(
        path=tmp_path / "m2.tape.json",
        interactions=[_inter("fp_shared", text="miss2")],
        recorded_at="not-a-timestamp",
    )
    m_miss = Cassette.merge(c_miss_1, c_miss_2, output_path=out, on_duplicate="keep-newest")
    assert [i.llm.text for i in m_miss.interactions] == ["miss2"]

    # error strategy raises TapeError naming the duplicate fingerprint
    with pytest.raises(TapeError, match="fp_shared"):
        Cassette.merge(c1, c2, output_path=out, on_duplicate="error")

    # Unknown strategy raises ValueError
    with pytest.raises(ValueError, match="on_duplicate"):
        Cassette.merge(c1, c2, output_path=out, on_duplicate="bogus")


def test_merge_sort_by_modes(tmp_path: Path) -> None:
    c_late = Cassette(
        path=tmp_path / "late.tape.json",
        interactions=[_inter("a", latency=50.0, cost=0.02)],
        recorded_at="2026-06-02T00:00:00Z",
    )
    c_early = Cassette(
        path=tmp_path / "early.tape.json",
        interactions=[_inter("b", latency=5.0, cost=0.09), _inter("c", latency=20.0, cost=0.001)],
        recorded_at="2026-06-01T00:00:00Z",
    )
    c_unparsed = Cassette(
        path=tmp_path / "unparsed.tape.json",
        interactions=[_inter("d", latency=1.0, cost=0.5)],
        recorded_at="",
    )
    out = tmp_path / "sorted.tape.json"

    by_rec = Cassette.merge(c_late, c_early, c_unparsed, output_path=out, sort_by="recorded_at")
    assert [i.request.fingerprint for i in by_rec.interactions] == ["b", "c", "a", "d"]

    by_lat = Cassette.merge(c_late, c_early, c_unparsed, output_path=out, sort_by="latency")
    assert [i.request.fingerprint for i in by_lat.interactions] == ["d", "b", "c", "a"]

    by_cost = Cassette.merge(c_late, c_early, c_unparsed, output_path=out, sort_by="cost")
    assert [i.request.fingerprint for i in by_cost.interactions] == ["c", "a", "b", "d"]

    with pytest.raises(ValueError, match="sort_by"):
        Cassette.merge(c_late, c_early, output_path=out, sort_by="invalid_field")


def test_merge_three_way_clean_add_delete_modify_duplicates_and_metadata(tmp_path: Path) -> None:
    # Base has: keep_same, del_in_ours, del_in_theirs, mod_in_ours, mod_in_theirs, and 2x dup_fp
    base = Cassette(
        path=tmp_path / "base.tape.json",
        interactions=[
            _inter("keep_same", text="same"),
            _inter("del_in_ours", text="bye_o"),
            _inter("del_in_theirs", text="bye_t"),
            _inter("mod_in_ours", text="orig_o"),
            _inter("mod_in_theirs", text="orig_t"),
            _inter("dup_fp", text="dup_turn_1"),
            _inter("dup_fp", text="dup_turn_2"),
        ],
        metadata={"unchanged": 1, "drop_ours": "x", "mod_theirs": "v0"},
    )
    ours = Cassette(
        path=tmp_path / "ours.tape.json",
        interactions=[
            _inter("keep_same", text="same"),
            # del_in_ours removed
            _inter("del_in_theirs", text="bye_t"),
            _inter("mod_in_ours", text="updated_by_ours"),
            _inter("mod_in_theirs", text="orig_t"),
            _inter("dup_fp", text="dup_turn_1"),
            _inter("dup_fp", text="dup_turn_2_ours"),
            _inter("added_ours", text="new_o"),
            _inter("added_both_same", text="convergent"),
        ],
        metadata={"unchanged": 1, "mod_theirs": "v0", "ours_meta": True},
    )
    theirs = Cassette(
        path=tmp_path / "theirs.tape.json",
        interactions=[
            _inter("keep_same", text="same"),
            _inter("del_in_ours", text="bye_o"),
            # del_in_theirs removed
            _inter("mod_in_ours", text="orig_o"),
            _inter("mod_in_theirs", text="updated_by_theirs"),
            _inter("dup_fp", text="dup_turn_1"),
            _inter("dup_fp", text="dup_turn_2"),
            _inter("added_theirs", text="new_t"),
            _inter("added_both_same", text="convergent"),
        ],
        metadata={"unchanged": 1, "drop_ours": "x", "mod_theirs": "v1", "theirs_meta": True},
    )

    merged, conflicts = Cassette.merge_three_way(
        base,
        ours,
        theirs,
        output_path=tmp_path / "merged.tape.json",
        on_conflict="error",
    )
    assert conflicts == []
    pairs = [(i.request.fingerprint, i.llm.text) for i in merged.interactions]
    assert pairs == [
        ("keep_same", "same"),
        ("mod_in_ours", "updated_by_ours"),
        ("mod_in_theirs", "updated_by_theirs"),
        ("dup_fp", "dup_turn_1"),
        ("dup_fp", "dup_turn_2_ours"),
        ("added_ours", "new_o"),
        ("added_both_same", "convergent"),
        ("added_theirs", "new_t"),
    ]
    assert merged.metadata == {
        "unchanged": 1,
        "mod_theirs": "v1",
        "ours_meta": True,
        "theirs_meta": True,
    }


def test_merge_three_way_conflicts_and_resolutions(tmp_path: Path) -> None:
    base = Cassette(
        path=tmp_path / "base.tape.json",
        interactions=[
            _inter("both_mod", text="base_v"),
            _inter("mod_ours_del_theirs", text="base_od"),
            _inter("del_ours_mod_theirs", text="base_do"),
        ],
        recorded_at="2026-05-01T00:00:00Z",
        metadata={"env": "base"},
    )
    ours = Cassette(
        path=tmp_path / "ours.tape.json",
        interactions=[
            _inter("both_mod", text="ours_v"),
            _inter("mod_ours_del_theirs", text="ours_modified"),
            # del_ours_mod_theirs deleted in ours
            _inter("both_added_diff", text="ours_added"),
        ],
        recorded_at="2026-05-02T00:00:00Z",
        metadata={"env": "ours"},
    )
    theirs = Cassette(
        path=tmp_path / "theirs.tape.json",
        interactions=[
            _inter("both_mod", text="theirs_v"),
            # mod_ours_del_theirs deleted in theirs
            _inter("del_ours_mod_theirs", text="theirs_modified"),
            _inter("both_added_diff", text="theirs_added"),
        ],
        recorded_at="2026-05-03T00:00:00Z",
        metadata={"env": "theirs"},
    )
    out = tmp_path / "conflict_out.tape.json"

    # keep-ours
    m_ours, conf_ours = Cassette.merge_three_way(
        base, ours, theirs, output_path=out, on_conflict="keep-ours"
    )
    assert len(conf_ours) >= 4
    assert [(i.request.fingerprint, i.llm.text) for i in m_ours.interactions] == [
        ("both_mod", "ours_v"),
        ("mod_ours_del_theirs", "ours_modified"),
        ("both_added_diff", "ours_added"),
    ]
    assert m_ours.metadata["env"] == "ours"

    # keep-theirs
    m_theirs, conf_theirs = Cassette.merge_three_way(
        base, ours, theirs, output_path=out, on_conflict="keep-theirs"
    )
    assert len(conf_theirs) >= 4
    assert [(i.request.fingerprint, i.llm.text) for i in m_theirs.interactions] == [
        ("both_mod", "theirs_v"),
        ("del_ours_mod_theirs", "theirs_modified"),
        ("both_added_diff", "theirs_added"),
    ]
    assert m_theirs.metadata["env"] == "theirs"

    # keep-newest (theirs May 03 > ours May 02)
    m_newest, _ = Cassette.merge_three_way(
        base, ours, theirs, output_path=out, on_conflict="keep-newest"
    )
    assert [(i.request.fingerprint, i.llm.text) for i in m_newest.interactions] == [
        ("both_mod", "theirs_v"),
        ("del_ours_mod_theirs", "theirs_modified"),
        ("both_added_diff", "theirs_added"),
    ]
    assert m_newest.metadata["env"] == "theirs"

    # error raises TapeError
    with pytest.raises(TapeError, match="both_mod"):
        Cassette.merge_three_way(base, ours, theirs, output_path=out, on_conflict="error")

    with pytest.raises(ValueError, match="on_conflict"):
        Cassette.merge_three_way(base, ours, theirs, output_path=out, on_conflict="unknown")


def test_cli_merge_flags_and_git_driver_json_and_gzip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p1 = tmp_path / "one.tape.json"
    p2 = tmp_path / "two.tape.json"
    Cassette(
        path=p1,
        interactions=[_inter("dup", text="first", cost=0.05), _inter("x", cost=0.01)],
        recorded_at="2026-05-01T00:00:00Z",
    ).save()
    Cassette(
        path=p2,
        interactions=[_inter("dup", text="second", cost=0.02)],
        recorded_at="2026-05-02T00:00:00Z",
    ).save()

    # Missing -o without --git-driver fails cleanly
    assert cli_main(["merge", str(p1), str(p2)]) == 1
    assert "--output" in capsys.readouterr().err

    # CLI --on-duplicate keep-last --sort-by cost
    merged_out = tmp_path / "merged_cli.tape.json"
    assert (
        cli_main(
            [
                "merge",
                str(p1),
                str(p2),
                "-o",
                str(merged_out),
                "--on-duplicate",
                "keep-last",
                "--sort-by",
                "cost",
            ]
        )
        == 0
    )
    capsys.readouterr()
    loaded_cli = Cassette.load(merged_out)
    assert [(i.request.fingerprint, i.llm.text) for i in loaded_cli.interactions] == [
        ("x", "ok"),
        ("dup", "second"),
    ]

    # CLI --git-driver clean merge on .tape.json (writes in-place to <ours>)
    base_p = tmp_path / "base.tape.json"
    ours_p = tmp_path / "ours.tape.json"
    theirs_p = tmp_path / "theirs.tape.json"
    Cassette(path=base_p, interactions=[_inter("shared", text="s0")]).save()
    Cassette(path=ours_p, interactions=[_inter("shared", text="s0"), _inter("o_add")]).save()
    Cassette(path=theirs_p, interactions=[_inter("shared", text="s0"), _inter("t_add")]).save()

    assert cli_main(["merge", "--git-driver", str(base_p), str(ours_p), str(theirs_p)]) == 0
    capsys.readouterr()
    merged_ours = Cassette.load(ours_p)
    assert [i.request.fingerprint for i in merged_ours.interactions] == [
        "shared",
        "o_add",
        "t_add",
    ]

    # CLI --git-driver with .tape.json.gz AND extensionless %A temp file (like real Git)
    gz_base = tmp_path / "stage_base"
    gz_ours = tmp_path / "stage_ours"
    gz_theirs = tmp_path / "stage_theirs"
    gz_base.write_bytes(
        Cassette(path=Path("x.tape.json.gz"), interactions=[_inter("a")]).to_bytes(compress=True)
    )
    gz_ours.write_bytes(
        Cassette(
            path=Path("x.tape.json.gz"), interactions=[_inter("a"), _inter("b", text="ours_b")]
        ).to_bytes(compress=True)
    )
    gz_theirs.write_bytes(
        Cassette(
            path=Path("x.tape.json.gz"), interactions=[_inter("a"), _inter("b", text="theirs_b")]
        ).to_bytes(compress=True)
    )

    # Conflict with --on-conflict error -> exits 1, stderr mentions conflict, ours untouched!
    before_bytes = gz_ours.read_bytes()
    assert (
        cli_main(
            [
                "merge",
                "--git-driver",
                str(gz_base),
                str(gz_ours),
                str(gz_theirs),
                "--on-conflict",
                "error",
            ]
        )
        == 1
    )
    assert "b" in capsys.readouterr().err
    assert gz_ours.read_bytes() == before_bytes

    # Resolve conflict with --on-conflict keep-theirs -> exits 0, preserves gzip magic bytes!
    assert (
        cli_main(
            [
                "merge",
                "--git-driver",
                str(gz_base),
                str(gz_ours),
                str(gz_theirs),
                "--on-conflict",
                "keep-theirs",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert gz_ours.read_bytes()[:2] == b"\x1f\x8b"
    reloaded_gz = Cassette.load(gz_ours)
    assert [(i.request.fingerprint, i.llm.text) for i in reloaded_gz.interactions] == [
        ("a", "ok"),
        ("b", "theirs_b"),
    ]


@pytest.mark.skipif(shutil.which("git") is None, reason="git binary not installed")
def test_real_git_custom_merge_driver_integration(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    rhinotape_bin = Path(sys.executable).parent / "rhinotape"

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )

    git("init", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test User")
    git(
        "config",
        "merge.rhinotape.driver",
        f"{rhinotape_bin} merge --git-driver %O %A %B",
    )
    (repo / ".gitattributes").write_text(
        "*.tape.json merge=rhinotape\n*.tape.json.gz merge=rhinotape\n",
        encoding="utf-8",
    )
    cass_file = repo / "chat.tape.json.gz"
    Cassette(path=cass_file, interactions=[_inter("init_call", text="hello")]).save()
    git("add", ".gitattributes", "chat.tape.json.gz")
    git("commit", "-m", "base")

    git("checkout", "-b", "feature")
    Cassette(
        path=cass_file,
        interactions=[_inter("init_call", text="hello"), _inter("feat_call", text="from_feat")],
    ).save()
    git("add", "chat.tape.json.gz")
    git("commit", "-m", "feature call")

    git("checkout", "main")
    Cassette(
        path=cass_file,
        interactions=[_inter("init_call", text="hello"), _inter("main_call", text="from_main")],
    ).save()
    git("add", "chat.tape.json.gz")
    git("commit", "-m", "main call")

    git("merge", "--no-edit", "feature")
    assert cass_file.read_bytes()[:2] == b"\x1f\x8b"
    final = Cassette.load(cass_file)
    assert [i.request.fingerprint for i in final.interactions] == [
        "init_call",
        "main_call",
        "feat_call",
    ]


def test_merge_keep_newest_missing_vs_valid_and_subsecond_timestamps(tmp_path: Path) -> None:
    out = tmp_path / "out.tape.json"
    c_valid = Cassette(
        path=tmp_path / "valid.tape.json",
        interactions=[_inter("fp_shared", text="valid_ts")],
        recorded_at="2026-05-10T12:00:00Z",
    )
    c_empty = Cassette(
        path=tmp_path / "empty.tape.json",
        interactions=[_inter("fp_shared", text="empty_ts")],
        recorded_at="",
    )
    c_bad = Cassette(
        path=tmp_path / "bad.tape.json",
        interactions=[_inter("fp_shared", text="bad_ts")],
        recorded_at="not-a-timestamp",
    )

    # Valid timestamp MUST beat missing/unparseable timestamp regardless of input order.
    for missing in (c_empty, c_bad):
        m1 = Cassette.merge(c_valid, missing, output_path=out, on_duplicate="keep-newest")
        m2 = Cassette.merge(missing, c_valid, output_path=out, on_duplicate="keep-newest")
        assert [i.llm.text for i in m1.interactions] == ["valid_ts"]
        assert [i.llm.text for i in m2.interactions] == ["valid_ts"]

    # Subsecond / nanosecond and compact timezone offset parsing (Python 3.10+ compat).
    c_nano = Cassette(
        path=tmp_path / "nano.tape.json",
        interactions=[_inter("fp_shared", text="nano")],
        recorded_at="2026-05-10T12:00:00.123456789Z",
    )
    c_half = Cassette(
        path=tmp_path / "half.tape.json",
        interactions=[_inter("fp_shared", text="half")],
        recorded_at="2026-05-10T12:00:00.5Z",
    )
    c_tz = Cassette(
        path=tmp_path / "tz.tape.json",
        interactions=[_inter("fp_shared", text="tz_later")],
        recorded_at="2026-05-10T08:30:00-0400",  # 12:30:00 UTC > 12:00:00.5 UTC
    )
    m_sub = Cassette.merge(c_tz, c_nano, c_half, output_path=out, on_duplicate="keep-newest")
    assert [i.llm.text for i in m_sub.interactions] == ["tz_later"]

    m_sub2 = Cassette.merge(c_half, c_nano, output_path=out, on_duplicate="keep-newest")
    assert [i.llm.text for i in m_sub2.interactions] == ["half"]

    # deduplicate=True combined with explicit on_duplicate strategy honors explicit strategy.
    m_dedup_last = Cassette.merge(
        c_valid, c_empty, output_path=out, deduplicate=True, on_duplicate="keep-last"
    )
    assert [i.llm.text for i in m_dedup_last.interactions] == ["empty_ts"]


def test_merge_sort_by_handles_nan_inf_and_none_recorded_at(tmp_path: Path) -> None:
    out = tmp_path / "nan_sorted.tape.json"
    c1 = Cassette(
        path=tmp_path / "c1.tape.json",
        interactions=[
            _inter("slow", latency=50.0, cost=0.50),
            _inter("nan_val", latency=float("nan"), cost=float("nan")),
            _inter("fast", latency=5.0, cost=0.01),
            _inter("inf_val", latency=float("inf"), cost=float("inf")),
        ],
        recorded_at=None,  # type: ignore[arg-type]
    )
    c2 = Cassette(
        path=tmp_path / "c2.tape.json",
        interactions=[_inter("mid", latency=20.0, cost=0.10)],
        recorded_at="2026-05-01T00:00:00Z",
    )

    by_lat = Cassette.merge(c1, c2, output_path=out, sort_by="latency")
    assert [i.request.fingerprint for i in by_lat.interactions] == [
        "fast",
        "mid",
        "slow",
        "nan_val",
        "inf_val",
    ]

    by_cost = Cassette.merge(c1, c2, output_path=out, sort_by="cost")
    assert [i.request.fingerprint for i in by_cost.interactions] == [
        "fast",
        "mid",
        "slow",
        "nan_val",
        "inf_val",
    ]

    by_rec = Cassette.merge(c1, c2, output_path=out, sort_by="recorded_at")
    assert [i.request.fingerprint for i in by_rec.interactions] == [
        "mid",
        "slow",
        "nan_val",
        "fast",
        "inf_val",
    ]


def test_merge_three_way_edge_cases_metadata_bool_int_and_duplicates(tmp_path: Path) -> None:
    out = tmp_path / "edge_3w.tape.json"
    base = Cassette(
        path=tmp_path / "base.tape.json",
        interactions=[
            _inter("dup", text="d1"),
            _inter("dup", text="d2_base"),
            _inter("both_del", text="gone"),
        ],
        recorded_at="2026-05-01T00:00:00Z",
        metadata={
            "flag_zero": 0,
            "flag_one": 1,
            "del_both": "bye",
            "del_ours_mod_theirs": "v0",
            "mod_ours_del_theirs": "v0",
            "nested": {"items": [1, 2]},
        },
    )
    ours = Cassette(
        path=tmp_path / "ours.tape.json",
        interactions=[
            _inter("dup", text="d1"),
            _inter("dup", text="d2_ours"),
            _inter("dup", text="d3_ours_only"),
        ],
        recorded_at="2026-05-10T00:00:00Z",
        metadata={
            "flag_zero": 0,
            "flag_one": True,  # changed 1 -> True (int -> bool!)
            "mod_ours_del_theirs": "v_ours",
            "bool_vs_int_add": 1,
            "nested": {"items": [1, 2]},
        },
    )
    theirs = Cassette(
        path=tmp_path / "theirs.tape.json",
        interactions=[
            _inter("dup", text="d1"),
            _inter("dup", text="d2_theirs"),
        ],
        recorded_at="invalid-timestamp",
        metadata={
            "flag_zero": False,  # changed 0 -> False (int -> bool!)
            "flag_one": 1,
            "del_ours_mod_theirs": "v_theirs",
            "bool_vs_int_add": True,
            "nested": {"items": [1, 2]},
        },
    )

    # keep-newest: ours has valid timestamp while theirs is invalid -> ours wins!
    merged, conflicts = Cassette.merge_three_way(
        base, ours, theirs, output_path=out, on_conflict="keep-newest"
    )
    assert any("dup#2" in c for c in conflicts)
    assert any("metadata['del_ours_mod_theirs']" in c for c in conflicts)
    assert any("metadata['mod_ours_del_theirs']" in c for c in conflicts)
    assert any("metadata['bool_vs_int_add']" in c for c in conflicts)
    assert [(i.request.fingerprint, i.llm.text) for i in merged.interactions] == [
        ("dup", "d1"),
        ("dup", "d2_ours"),
        ("dup", "d3_ours_only"),
    ]
    assert merged.recorded_at == "2026-05-10T00:00:00Z"
    assert merged.metadata["flag_zero"] is False
    assert merged.metadata["flag_one"] is True
    assert "del_both" not in merged.metadata
    assert "del_ours_mod_theirs" not in merged.metadata
    assert merged.metadata["mod_ours_del_theirs"] == "v_ours"
    assert merged.metadata["bool_vs_int_add"] == 1 and not isinstance(
        merged.metadata["bool_vs_int_add"], bool
    )

    # Deepcopy check: mutating merged.metadata does not mutate base/ours/theirs.
    merged.metadata["nested"]["items"].append(99)
    assert ours.metadata["nested"]["items"] == [1, 2]

    # Exact timestamp tie in merge_three_way keep-newest breaks toward theirs.
    ours_tie = ours.clone(tmp_path / "ours_tie.tape.json")
    theirs_tie = theirs.clone(tmp_path / "theirs_tie.tape.json")
    ours_tie.recorded_at = "2026-05-10T00:00:00+00:00"
    theirs_tie.recorded_at = "2026-05-10T00:00:00Z"
    merged_tie, _ = Cassette.merge_three_way(
        base, ours_tie, theirs_tie, output_path=out, on_conflict="keep-newest"
    )
    assert merged_tie.interactions[1].llm.text == "d2_theirs"
    assert merged_tie.metadata["del_ours_mod_theirs"] == "v_theirs"
    assert "mod_ours_del_theirs" not in merged_tie.metadata


def test_cli_merge_git_driver_validation_and_empty_ancestor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_p = tmp_path / "base.tape.json"
    ours_p = tmp_path / "ours.tape.json"
    theirs_p = tmp_path / "theirs.tape.json"
    other_out = tmp_path / "other.tape.json"

    Cassette(path=base_p, interactions=[_inter("a")]).save()
    Cassette(path=ours_p, interactions=[_inter("a"), _inter("o1")]).save()
    Cassette(path=theirs_p, interactions=[_inter("a"), _inter("t1")]).save()
    orig_ours_bytes = ours_p.read_bytes()

    # 1. --git-driver combined with -o/--output must fail and leave <ours> untouched.
    assert (
        cli_main(
            [
                "merge",
                "--git-driver",
                str(base_p),
                str(ours_p),
                str(theirs_p),
                "-o",
                str(other_out),
            ]
        )
        == 1
    )
    assert "--output" in capsys.readouterr().err
    assert not other_out.exists()
    assert ours_p.read_bytes() == orig_ours_bytes

    # 2. --git-driver combined with 2-way flags (--deduplicate / --on-duplicate / --sort-by) fails.
    for extra_flags in (
        ["--deduplicate"],
        ["--on-duplicate", "keep-last"],
        ["--sort-by", "cost"],
    ):
        assert (
            cli_main(
                ["merge", "--git-driver", str(base_p), str(ours_p), str(theirs_p), *extra_flags]
            )
            == 1
        )
        assert "--git-driver" in capsys.readouterr().err
        assert ours_p.read_bytes() == orig_ours_bytes

    # 3. Wrong arg counts (2 args or 4 args) and missing file with --git-driver fail.
    assert cli_main(["merge", "--git-driver", str(base_p), str(ours_p)]) == 1
    assert "3 arguments" in capsys.readouterr().err
    assert (
        cli_main(["merge", "--git-driver", str(base_p), str(ours_p), str(theirs_p), str(base_p)])
        == 1
    )
    assert "3 arguments" in capsys.readouterr().err
    assert (
        cli_main(["merge", "--git-driver", str(tmp_path / "nope"), str(ours_p), str(theirs_p)]) == 1
    )
    assert "not found" in capsys.readouterr().err
    assert ours_p.read_bytes() == orig_ours_bytes

    # 4. --on-conflict without --git-driver and --deduplicate + --on-duplicate keep-all fail.
    assert (
        cli_main(
            ["merge", str(ours_p), str(theirs_p), "-o", str(other_out), "--on-conflict", "error"]
        )
        == 1
    )
    assert "--on-conflict" in capsys.readouterr().err
    assert (
        cli_main(
            [
                "merge",
                str(ours_p),
                str(theirs_p),
                "-o",
                str(other_out),
                "--deduplicate",
                "--on-duplicate",
                "keep-all",
            ]
        )
        == 1
    )
    assert "--deduplicate" in capsys.readouterr().err

    # 5. 0-byte %O ancestor file (Git add/add merge with no common ancestor) merges cleanly.
    empty_base = tmp_path / "empty_stage_base"
    empty_base.write_bytes(b"")
    assert cli_main(["merge", "--git-driver", str(empty_base), str(ours_p), str(theirs_p)]) == 0
    capsys.readouterr()
    merged_add_add = Cassette.load(ours_p)
    assert [i.request.fingerprint for i in merged_add_add.interactions] == ["a", "o1", "t1"]
