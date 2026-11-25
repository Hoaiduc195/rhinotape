"""End-to-end pytest plugin tests using pytest's built-in pytester fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


def test_pytester_cli_options_and_parametrized_cassettes(
    pytester: pytest.Pytester, tmp_path: Path
) -> None:
    tape_dir = tmp_path / "tapes"
    pytester.makeini(
        """
        [pytest]
        asyncio_default_fixture_loop_scope = function
        """
    )
    pytester.makepyfile(
        test_sample="""
        import httpx
        import pytest

        def _mock_handler(request: httpx.Request) -> httpx.Response:
            body = request.read().decode("utf-8")
            return httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"role": "assistant", "content": body}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )

        @pytest.mark.parametrize("city", ["Hanoi/VN", "Hanoi:VN"])
        @pytest.mark.tape(name="weather")
        def test_weather(llm_tape, city):
            with httpx.Client(transport=httpx.MockTransport(_mock_handler)) as client:
                resp = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"city": city},
                )
                assert city in resp.json()["choices"][0]["message"]["content"]
            llm_tape.assert_call_count(1)
            llm_tape.assert_tokens_under(50)
        """
    )

    # First run: records two distinct cassettes (no collision between Hanoi/VN and Hanoi:VN)
    res1 = pytester.runpytest("--tape-mode=once", f"--tape-dir={tape_dir}")
    res1.assert_outcomes(passed=2)

    recorded_files = list(tape_dir.rglob("*.tape.json"))
    assert len(recorded_files) == 2

    # Second run: strict replay mode passes cleanly
    res2 = pytester.runpytest("--tape-mode=replay", f"--tape-dir={tape_dir}")
    res2.assert_outcomes(passed=2)


def test_pytester_save_on_error_false_and_class_name_isolation(
    pytester: pytest.Pytester, tmp_path: Path
) -> None:
    tape_dir = tmp_path / "tapes"
    pytester.makeini(
        """
        [pytest]
        asyncio_default_fixture_loop_scope = function
        """
    )
    pytester.makepyfile(
        test_classes="""
        import httpx
        import pytest

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 5}})

        class TestOpenAI:
            def test_chat(self, llm_tape):
                with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
                    client.post("https://api.openai.com/v1/chat/completions", json={"p": "openai"})

        class TestAnthropic:
            def test_chat(self, llm_tape):
                with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
                    client.post("https://api.anthropic.com/v1/messages", json={"p": "anthropic"})

        def test_failing_does_not_save_cassette(llm_tape):
            with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
                client.post("https://api.openai.com/v1/chat/completions", json={"p": "fail"})
            assert False, "intentional failure"
        """
    )

    res = pytester.runpytest("--tape-mode=once", f"--tape-dir={tape_dir}")
    res.assert_outcomes(passed=2, failed=1)

    saved = sorted(p.name for p in tape_dir.rglob("*.tape.json"))
    assert saved == [
        "TestAnthropic__test_chat.tape.json",
        "TestOpenAI__test_chat.tape.json",
    ]
