"""Tests for Wave 11 (match_headers) and Wave 12 (pytest --tape-strict-unused)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from llm_tape import TapeMode, TapeTransport, UnmatchedRequestError, tape

pytest_plugins = ["pytester"]


def test_match_headers_ignores_volatile_headers_and_matches_selected(tmp_path: Path) -> None:
    tape_path = tmp_path / "hdr.tape.json"
    upstream = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": req.headers.get("anthropic-beta", "none")}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )
    )

    with tape(
        tape_path,
        mode=TapeMode.RECORD,
        match_headers=["anthropic-beta"],
        patch_httpx=False,
    ) as rec:
        with httpx.Client(transport=TapeTransport(rec, upstream=upstream)) as c:
            c.post(
                "https://api.anthropic.com/v1/messages",
                headers={"anthropic-beta": "prompt-caching-2024-07-31", "x-request-id": "req-1"},
                json={"model": "claude-sonnet-4"},
            )

    # Different volatile x-request-id still matches!
    with tape(
        tape_path,
        mode=TapeMode.REPLAY,
        match_headers=["Anthropic-Beta"],
        patch_httpx=False,
    ) as rep:
        with httpx.Client(transport=TapeTransport(rep)) as c:
            r = c.post(
                "https://api.anthropic.com/v1/messages",
                headers={"anthropic-beta": "prompt-caching-2024-07-31", "x-request-id": "req-999"},
                json={"model": "claude-sonnet-4"},
            )
            assert r.status_code == 200

    # Different selected header (anthropic-beta) fails with UnmatchedRequestError!
    with tape(
        tape_path,
        mode=TapeMode.REPLAY,
        match_headers=["anthropic-beta"],
        patch_httpx=False,
    ) as rep2:
        with httpx.Client(transport=TapeTransport(rep2)) as c:
            with pytest.raises(UnmatchedRequestError):
                c.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"anthropic-beta": "other-beta"},
                    json={"model": "claude-sonnet-4"},
                )


def test_pytester_strict_unused_fails_on_dead_recorded_interactions(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    pytester.makepyfile(
        """
        import os
        import httpx
        from llm_tape import TapeTransport

        def test_agent_two_calls(llm_tape):
            upstream = httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    json={
                        "model": "gpt-4o-mini",
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    },
                )
            )
            with httpx.Client(transport=TapeTransport(llm_tape, upstream=upstream)) as c:
                c.post("https://api.openai.com/v1/chat/completions", json={"step": 1})
                if os.environ.get("RUN_STEP_2", "1") == "1":
                    c.post("https://api.openai.com/v1/chat/completions", json={"step": 2})
        """
    )

    # Record 2 interactions
    r1 = pytester.runpytest()
    r1.assert_outcomes(passed=1)

    # Replay with only 1 call + --tape-strict-unused -> fails teardown due to unused interaction
    monkeypatch.setenv("RUN_STEP_2", "0")
    r2 = pytester.runpytest("--tape-mode=replay", "--tape-strict-unused")
    r2.assert_outcomes(passed=1, errors=1)
    assert "unused" in r2.stdout.str().lower()
