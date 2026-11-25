"""Pytester tests for Batch 11 pytest plugin enhancements (Waves 68-71)."""

from __future__ import annotations

import json

import pytest

pytest_plugins = ["pytester"]


def test_ini_budget_and_llm_tape_dir_and_redact_pii(pytester: pytest.Pytester) -> None:
    pytester.makeini(
        """
        [pytest]
        asyncio_default_fixture_loop_scope = function
        tape_redact_pii = email
        tape_max_tokens = 5
        """
    )
    pytester.makepyfile(
        """
        import httpx
        import pytest

        def _transport():
            return httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-1",
                        "model": "gpt-4o-mini",
                        "choices": [
                            {"message": {"role": "assistant", "content": "hi bob@test.com"}}
                        ],
                        "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
                    },
                )
            )

        @pytest.mark.tape(mode="record")
        def test_pii_and_dir(llm_tape, llm_tape_dir):
            assert llm_tape.cassette_path.parent == llm_tape_dir
            with httpx.Client(transport=_transport()) as client:
                r = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "alice@test.com"}],
                    },
                )
                assert r.status_code == 200
        """
    )
    result = pytester.runpytest()
    # Test itself passes, but session fails due to tape_max_tokens=5 < 10
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*LLM token budget exceeded: 10 > --tape-max-tokens 5*"])

    cassette_files = list((pytester.path / "tapes").rglob("*.tape.json"))
    assert len(cassette_files) == 1
    raw_text = cassette_files[0].read_text(encoding="utf-8")
    assert "alice@test.com" not in raw_text
    assert "bob@test.com" not in raw_text


def test_env_vars_and_update_stale(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    custom_dir = pytester.path / "env_tapes"
    monkeypatch.setenv("LLM_TAPE_DIR", str(custom_dir))
    monkeypatch.setenv("LLM_TAPE_MODE", "once")

    pytester.makepyfile(
        """
        import httpx

        COUNTER = {"n": 0}

        def _transport():
            def _handle(req):
                COUNTER["n"] += 1
                msg = f"v{COUNTER['n']}"
                return httpx.Response(
                    200,
                    json={
                        "id": f"chatcmpl-{COUNTER['n']}",
                        "model": "gpt-4o-mini",
                        "choices": [{"message": {"role": "assistant", "content": msg}}],
                    },
                )
            return httpx.MockTransport(_handle)

        def test_stale(llm_tape):
            with httpx.Client(transport=_transport()) as client:
                r = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                )
                assert r.status_code == 200
        """
    )
    # 1st run: records into LLM_TAPE_DIR
    r1 = pytester.runpytest()
    r1.assert_outcomes(passed=1)
    cassette_files = list(custom_dir.rglob("*.tape.json"))
    assert len(cassette_files) == 1
    cpath = cassette_files[0]

    # Backdate recorded_at by 100 days
    data = json.loads(cpath.read_text(encoding="utf-8"))
    data["recorded_at"] = "2020-01-01T00:00:00Z"
    cpath.write_text(json.dumps(data), encoding="utf-8")

    # 2nd run with --tape-update-stale=30 re-records and refreshes recorded_at
    r2 = pytester.runpytest("--tape-update-stale=30")
    r2.assert_outcomes(passed=1)
    refreshed = json.loads(cpath.read_text(encoding="utf-8"))
    assert refreshed["recorded_at"] != "2020-01-01T00:00:00Z"
