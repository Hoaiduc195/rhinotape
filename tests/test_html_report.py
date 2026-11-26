"""Tests for self-contained HTML trace report generation, CLI export, and pytest flag."""

from __future__ import annotations

from pathlib import Path

import pytest

from rhinotape.cassette import Cassette
from rhinotape.cli import main as cli_main
from rhinotape.html_report import render_html_report
from rhinotape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    ToolCall,
    Usage,
)

pytest_plugins = ["pytester"]


def _make_sample_cassette(path: Path) -> Cassette:
    inter1 = Interaction(
        request=RecordedRequest(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "<script>alert(1)</script>"}],
            },
            fingerprint="fp1",
        ),
        response=RecordedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"id": "chatcmpl-1", "choices": [{"message": {"content": "<b>hi</b>"}}]},
        ),
        llm=LLMCallSummary(
            provider="openai",
            model="gpt-4o",
            text="<b>hi</b><script>xss()</script>",
            thinking="step <1> & step <2>",
            tool_calls=(
                ToolCall(call_id="call_1", name="get_weather", arguments='{"city":"<NYC>"}'),
            ),
            usage=Usage(
                input_tokens=10,
                output_tokens=20,
                cached_input_tokens=5,
                reasoning_tokens=4,
            ),
            estimated_cost_usd=0.001234,
        ),
        latency_ms=125.5,
    )
    inter2 = Interaction(
        request=RecordedRequest(
            method="POST",
            url="https://api.anthropic.com/v1/messages",
            headers={},
            body="raw-body-string",
            fingerprint="fp2",
        ),
        response=RecordedResponse(
            status_code=429,
            headers={"content-type": "text/event-stream"},
            body=None,
            body_encoding="stream",
            stream_chunks=("data: chunk1\n\n", "data: <img src=x onerror=alert(1)>\n\n"),
        ),
        llm=LLMCallSummary(
            provider="anthropic",
            model="claude-3-5-sonnet",
            is_stream=True,
            text="rate limited",
            usage=Usage(input_tokens=5, output_tokens=0),
            estimated_cost_usd=0.0001,
        ),
        latency_ms=50.0,
    )
    return Cassette(path=path, interactions=[inter1, inter2])


def test_render_html_report_escapes_xss_and_renders_panels(tmp_path: Path) -> None:
    cassette = _make_sample_cassette(tmp_path / "sample.tape.json")
    doc = render_html_report([cassette], title="<script>evil()</script> Report")

    assert "<!DOCTYPE html>" in doc
    assert "<script>evil()</script>" not in doc
    assert "&lt;script&gt;evil()&lt;/script&gt; Report" in doc
    assert "<script>alert(1)</script>" not in doc
    assert "<script>xss()</script>" not in doc
    assert "<img src=x onerror=alert(1)>" not in doc
    assert "&lt;img src=x onerror=alert(1)&gt;" in doc
    assert "Thinking" in doc
    assert "Response Text" in doc
    assert "Tool Calls" in doc
    assert "Stream (2 chunks)" in doc
    assert "badge ok" in doc
    assert "badge err" in doc
    assert "tape-search" in doc
    assert "tape-provider" in doc
    assert "tape-status" in doc


def test_render_html_report_empty(tmp_path: Path) -> None:
    empty = Cassette(path=tmp_path / "empty.tape.json", interactions=[])
    doc = render_html_report([empty])
    assert "No interactions recorded." in doc
    assert render_html_report([]).count("No interactions recorded.") == 1


def test_cli_export_html_stdout_and_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cassette_path = tmp_path / "demo.tape.json"
    _make_sample_cassette(cassette_path).save()

    rc = cli_main(["export", str(cassette_path), "--format", "html"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "<!DOCTYPE html>" in out
    assert "gpt-4o" in out

    out_file = tmp_path / "out.html"
    rc2 = cli_main(["export", str(cassette_path), "--format", "html", "-o", str(out_file)])
    assert rc2 == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "claude-3-5-sonnet" in content


def test_pytest_tape_html_report_flag(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    report_path = pytester.path / "reports" / "trace.html"
    pytester.makepyfile(
        """
        import httpx
        import pytest

        def _transport():
            return httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-test",
                        "model": "gpt-4o-mini",
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                    },
                )
            )

        @pytest.mark.tape(mode="record")
        def test_record_for_html(rhinotape):
            with httpx.Client(transport=_transport()) as client:
                r = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert r.status_code == 200
        """
    )
    result = pytester.runpytest(f"--tape-html-report={report_path}")
    result.assert_outcomes(passed=1)
    assert report_path.exists()
    html_text = report_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "gpt-4o-mini" in html_text
