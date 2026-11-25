"""Tests for pytest reporting (--tape-report, --tape-json-report, budget gates) and assertions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_tape import TapeAssertionError, TapeMode, TapeTransport, tape

pytest_plugins = ["pytester"]


def test_session_new_assertions(tmp_path: Path) -> None:
    cassette_path = tmp_path / "assertions.tape.json"

    def handler(req: httpx.Request) -> httpx.Response:
        if "fail" in str(req.url):
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 60},
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        )

    upstream = httpx.MockTransport(handler)
    with tape(cassette_path, mode=TapeMode.RECORD, patch_httpx=False) as s:
        with httpx.Client(transport=TapeTransport(s, upstream=upstream)) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4o-mini"})

        s.assert_model("gpt-4o-mini")
        s.assert_provider("openai")
        s.assert_cached_tokens_at_least(60)
        s.assert_no_errors()

        with pytest.raises(TapeAssertionError, match="Model mismatch"):
            s.assert_model("gpt-4o")
        with pytest.raises(TapeAssertionError, match="Provider mismatch"):
            s.assert_provider("anthropic")
        with pytest.raises(TapeAssertionError, match="cached input token"):
            s.assert_cached_tokens_at_least(61)

    # Empty session & HTTP error assertions
    err_tape = tmp_path / "err.tape.json"
    with tape(err_tape, mode=TapeMode.RECORD, patch_httpx=False) as s2:
        with pytest.raises(TapeAssertionError, match="no interactions"):
            s2.assert_model("gpt-4o-mini")
        with pytest.raises(TapeAssertionError, match="no interactions"):
            s2.assert_provider("openai")
        with httpx.Client(transport=TapeTransport(s2, upstream=upstream)) as client:
            client.post("https://api.openai.com/v1/fail", json={})
        with pytest.raises(TapeAssertionError, match="500"):
            s2.assert_no_errors()


def test_pytester_report_json_and_budget_gates(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    pytester.makepyfile(
        """
        import httpx
        from llm_tape import TapeTransport

        def test_llm_cost_reporting(llm_tape):
            upstream = httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    json={
                        "model": "gpt-4o-mini",
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                        "usage": {
                            "prompt_tokens": 1000,
                            "prompt_tokens_details": {"cached_tokens": 400},
                            "completion_tokens": 200,
                            "total_tokens": 1200,
                        },
                    },
                )
            )
            with httpx.Client(transport=TapeTransport(llm_tape, upstream=upstream)) as c:
                r = c.post("https://api.openai.com/v1/chat/completions", json={"q": "hi"})
                assert r.status_code == 200
        """
    )

    json_report = pytester.path / "reports" / "tape-report.json"
    res = pytester.runpytest("--tape-report", f"--tape-json-report={json_report}")
    res.assert_outcomes(passed=1)
    stdout = res.stdout.str()
    assert "llm-tape summary" in stdout
    assert "recorded" in stdout
    assert "1200" in stdout
    assert "400" in stdout

    data = json.loads(json_report.read_text(encoding="utf-8"))
    assert data["totals"]["tests"] == 1
    assert data["totals"]["calls"] == 1
    assert data["totals"]["total_tokens"] == 1200
    assert data["totals"]["cached_input_tokens"] == 400
    assert data["budget_violations"] == []

    # Second run replays and fails session budget gate (--tape-max-tokens)
    res_fail = pytester.runpytest("--tape-max-tokens=500", f"--tape-json-report={json_report}")
    assert res_fail.ret == pytest.ExitCode.TESTS_FAILED
    assert "LLM token budget exceeded" in res_fail.stdout.str()
    data2 = json.loads(json_report.read_text(encoding="utf-8"))
    assert len(data2["budget_violations"]) == 1

    # Invalid negative/nan budget options raise pytest UsageError
    res_bad_cost = pytester.runpytest("--tape-max-cost=nan")
    assert res_bad_cost.ret == pytest.ExitCode.USAGE_ERROR
    res_bad_tokens = pytester.runpytest("--tape-max-tokens=-1")
    assert res_bad_tokens.ret == pytest.ExitCode.USAGE_ERROR


def test_xdist_worker_and_controller_aggregation(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from llm_tape.pytest_plugin import (
        _REPORT_KEY,
        _VIOLATIONS_KEY,
        pytest_sessionfinish,
        pytest_testnodedown,
    )

    json_report = tmp_path / "xdist-report.json"

    def _make_config(*, is_worker: bool, max_tokens: int | None) -> Any:
        stash = pytest.Stash()
        stash[_REPORT_KEY] = []
        stash[_VIOLATIONS_KEY] = []
        opts = {
            "--tape-max-tokens": max_tokens,
            "--tape-max-cost": None,
            "--tape-json-report": str(json_report),
        }
        cfg = SimpleNamespace(
            stash=stash,
            getoption=lambda name, default=None: opts.get(name, default),
        )
        if is_worker:
            cfg.workerinput = {"workerid": "gw0"}
            cfg.workeroutput = {}
        return cfg

    # Worker 1 & Worker 2 each stay under 500 tokens individually (300 each)
    w1_cfg = _make_config(is_worker=True, max_tokens=500)
    w1_cfg.stash[_REPORT_KEY].append(
        {
            "nodeid": "test_a.py::test_1",
            "cassette": "a.tape.json",
            "action": "replayed",
            "calls": 1,
            "input_tokens": 200,
            "cached_input_tokens": 50,
            "output_tokens": 100,
            "total_tokens": 300,
            "estimated_cost_usd": 0.001,
            "latency_ms": 10.0,
        }
    )
    w1_session = SimpleNamespace(config=w1_cfg, exitstatus=pytest.ExitCode.OK)
    pytest_sessionfinish(w1_session, 0)  # type: ignore[arg-type]
    # Worker must NOT write the JSON report or fail locally
    assert not json_report.exists()
    assert w1_session.exitstatus == pytest.ExitCode.OK
    assert len(w1_cfg.workeroutput["llm_tape_entries"]) == 1

    w2_cfg = _make_config(is_worker=True, max_tokens=500)
    w2_cfg.stash[_REPORT_KEY].append(
        {
            "nodeid": "test_b.py::test_2",
            "cassette": "b.tape.json",
            "action": "replayed",
            "calls": 1,
            "input_tokens": 200,
            "cached_input_tokens": 0,
            "output_tokens": 100,
            "total_tokens": 300,
            "estimated_cost_usd": 0.001,
            "latency_ms": 15.0,
        }
    )
    w2_session = SimpleNamespace(config=w2_cfg, exitstatus=pytest.ExitCode.OK)
    pytest_sessionfinish(w2_session, 0)  # type: ignore[arg-type]

    # Controller merges both workers via pytest_testnodedown and enforces total cap (600 > 500)
    ctrl_cfg = _make_config(is_worker=False, max_tokens=500)
    pytest_testnodedown(SimpleNamespace(config=ctrl_cfg, workeroutput=w1_cfg.workeroutput), None)
    pytest_testnodedown(SimpleNamespace(config=ctrl_cfg, workeroutput=w2_cfg.workeroutput), None)
    ctrl_session = SimpleNamespace(config=ctrl_cfg, exitstatus=pytest.ExitCode.OK)
    pytest_sessionfinish(ctrl_session, 0)  # type: ignore[arg-type]

    assert ctrl_session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert json_report.exists()
    report = json.loads(json_report.read_text(encoding="utf-8"))
    assert report["totals"]["tests"] == 2
    assert report["totals"]["total_tokens"] == 600
    assert len(report["budget_violations"]) == 1
