"""Tests for CLI export (Markdown, CSV, HAR 1.2), convert (gzip <-> json), and multi-file verify."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from rhinotape import Cassette
from rhinotape.cli import main
from rhinotape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    Usage,
)


def _make_sample_cassette(
    path: Path,
    *,
    stream_chunks: tuple[str, ...] | None = None,
    reasoning_tokens: int = 0,
) -> Path:
    inter = Interaction(
        request=RecordedRequest(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            headers={"content-type": "application/json", "authorization": "[REDACTED]"},
            body={"model": "o3", "messages": [{"role": "user", "content": "hi"}]},
            fingerprint="fp1",
            body_encoding="json",
        ),
        response=RecordedResponse(
            status_code=200,
            headers={"content-type": "text/event-stream" if stream_chunks else "application/json"},
            body=None if stream_chunks else {"choices": [{"message": {"content": "hello"}}]},
            body_encoding="stream" if stream_chunks else "json",
            stream_chunks=stream_chunks,
        ),
        latency_ms=85.5,
        llm=LLMCallSummary(
            provider="openai",
            model="o3",
            usage=Usage(
                input_tokens=200,
                output_tokens=100,
                cached_input_tokens=50,
                reasoning_tokens=reasoning_tokens,
            ),
            estimated_cost_usd=0.0012,
            text="hello",
            is_stream=stream_chunks is not None,
        ),
    )
    c = Cassette(path=path, interactions=[inter])
    c.save()
    return path


def test_cli_export_markdown_csv_and_har(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tapes_dir = tmp_path / "tapes"
    _make_sample_cassette(tapes_dir / "a.tape.json", reasoning_tokens=40)
    _make_sample_cassette(
        tapes_dir / "b.tape.json.gz",
        stream_chunks=('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', "data: [DONE]\n\n"),
    )

    # Markdown to stdout
    assert main(["export", str(tapes_dir), "--format", "markdown"]) == 0
    md = capsys.readouterr().out
    assert "| Cassette | Calls | Tokens | Cached | Reasoning | Est. Cost | Latency |" in md
    assert "**TOTAL (2 files)**" in md
    assert "| 40 |" in md

    # CSV to stdout
    assert main(["export", str(tapes_dir), "--format", "csv"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 2
    assert rows[0]["reasoning_tokens"] == "40"
    assert rows[0]["cached_input_tokens"] == "50"

    # HAR 1.2 to file
    har_out = tmp_path / "out" / "trace.har"
    assert main(["export", str(tapes_dir), "--format", "har", "-o", str(har_out)]) == 0
    assert "Exported har report" in capsys.readouterr().out
    har = json.loads(har_out.read_text(encoding="utf-8"))
    assert har["log"]["version"] == "1.2"
    assert len(har["log"]["entries"]) == 2
    assert har["log"]["entries"][0]["request"]["method"] == "POST"
    assert "data: [DONE]" in har["log"]["entries"][1]["response"]["content"]["text"]


def test_cli_export_otel(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from rhinotape.models import ToolCall

    tapes_dir = tmp_path / "otel_tapes"
    _make_sample_cassette(tapes_dir / "first.tape.json", reasoning_tokens=50)

    # Cassette with tool call and error response
    inter_tool = Interaction(
        request=RecordedRequest(
            method="POST",
            url="https://api.anthropic.com/v1/messages",
            headers={"content-type": "application/json"},
            body={"model": "claude-3-5-sonnet", "messages": []},
            fingerprint="fp_tool",
            body_encoding="json",
        ),
        response=RecordedResponse(
            status_code=500,
            headers={"content-type": "application/json"},
            body={"error": {"message": "overloaded"}},
            body_encoding="json",
        ),
        latency_ms=120.0,
        llm=LLMCallSummary(
            provider="anthropic",
            model="claude-3-5-sonnet",
            usage=Usage(input_tokens=150, output_tokens=80),
            tool_calls=(
                ToolCall(
                    name="web_search",
                    arguments={"query": "rhinotape docs"},
                    call_id="toolu_123",
                ),
            ),
        ),
    )
    Cassette(path=tapes_dir / "tool_and_error.tape.json", interactions=[inter_tool]).save()

    # 1. Export OTLP to stdout (supports both --format otel and --format otlp)
    assert main(["export", str(tapes_dir), "--format", "otel"]) == 0
    raw_json = capsys.readouterr().out
    otel = json.loads(raw_json)

    assert main(["export", str(tapes_dir), "--format", "otlp"]) == 0
    otlp_json = json.loads(capsys.readouterr().out)
    assert len(otlp_json["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 2

    assert "resourceSpans" in otel
    assert len(otel["resourceSpans"]) == 1
    res_spans = otel["resourceSpans"][0]
    res_attrs = {a["key"]: a["value"]["stringValue"] for a in res_spans["resource"]["attributes"]}
    assert res_attrs["service.name"] == "rhinotape"

    scope_spans = res_spans["scopeSpans"][0]
    spans = scope_spans["spans"]
    assert len(spans) == 2

    # Check span 1 (o3 from first cassette)
    span1 = spans[0]
    assert span1["name"] == "chat o3"
    assert span1["kind"] == 3
    assert len(span1["traceId"]) == 32
    assert len(span1["spanId"]) == 16
    assert span1["status"] == {"code": 1}

    span1_attrs = {a["key"]: list(a["value"].values())[0] for a in span1["attributes"]}
    assert span1_attrs["gen_ai.system"] == "openai"
    assert span1_attrs["gen_ai.request.model"] == "o3"
    assert span1_attrs["gen_ai.usage.input_tokens"] == 200
    assert span1_attrs["gen_ai.usage.output_tokens"] == 100
    assert span1_attrs["http.response.status_code"] == 200

    # Check span 2 (claude with tool call event and status 500 error)
    span2 = spans[1]
    assert span2["name"] == "chat claude-3-5-sonnet"
    assert span2["status"]["code"] == 2
    assert "HTTP 500" in span2["status"]["message"]

    span2_attrs = {a["key"]: list(a["value"].values())[0] for a in span2["attributes"]}
    assert span2_attrs["gen_ai.system"] == "anthropic"
    assert span2_attrs["gen_ai.request.model"] == "claude-3-5-sonnet"

    # Verify tool call event
    assert len(span2["events"]) == 1
    event = span2["events"][0]
    assert event["name"] == "gen_ai.tool.call"
    event_attrs = {a["key"]: list(a["value"].values())[0] for a in event["attributes"]}
    assert event_attrs["gen_ai.tool.name"] == "web_search"
    assert event_attrs["gen_ai.tool.call_id"] == "toolu_123"
    assert json.loads(event_attrs["gen_ai.tool.arguments"]) == {"query": "rhinotape docs"}

    # 2. Export OTLP to file with -o
    otel_out = tmp_path / "out" / "traces.json"
    assert main(["export", str(tapes_dir), "--format", "otel", "-o", str(otel_out)]) == 0
    assert "Exported otel report (2 cassette(s)) to" in capsys.readouterr().out
    assert otel_out.exists()
    disk_otel = json.loads(otel_out.read_text(encoding="utf-8"))
    assert len(disk_otel["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 2


def test_cli_convert_gzip_json_and_multi_file_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tapes_dir = tmp_path / "tapes"
    p1 = _make_sample_cassette(tapes_dir / "one.tape.json")
    p2 = _make_sample_cassette(tapes_dir / "two.tape.json")

    # Multi-file verify (like pre-commit passing multiple filenames)
    assert main(["verify", "--check-secrets", str(p1), str(p2)]) == 0
    assert "verified 2 cassette(s)" in capsys.readouterr().out

    # Dry-run convert to gzip
    assert main(["convert", str(tapes_dir), "--to", "gzip", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "WOULD CONVERT" in out
    assert not (tapes_dir / "one.tape.json.gz").exists()

    # Apply convert to gzip with --delete-source
    assert main(["convert", str(tapes_dir), "--to", "gzip", "--delete-source"]) == 0
    capsys.readouterr()
    gz1 = tapes_dir / "one.tape.json.gz"
    gz2 = tapes_dir / "two.tape.json.gz"
    assert gz1.exists() and gz2.exists()
    assert not p1.exists() and not p2.exists()
    assert Cassette.load(gz1).interactions[0].llm.model == "o3"

    # Convert back to plain json
    assert main(["convert", str(gz1), "--to", "json", "--delete-source"]) == 0
    capsys.readouterr()
    assert p1.exists() and not gz1.exists()
    assert Cassette.load(p1).interactions[0].llm.model == "o3"


def test_convert_collision_protection_and_har_spec_fixes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import base64

    # P1: collision refuses to overwrite without --force, succeeds with --force
    tapes_dir = tmp_path / "collide"
    plain = _make_sample_cassette(tapes_dir / "dup.tape.json")
    gz = _make_sample_cassette(tapes_dir / "dup.tape.json.gz")
    orig_gz_bytes = gz.read_bytes()

    assert main(["convert", str(tapes_dir), "--to", "gzip", "--delete-source"]) == 1
    assert "CONFLICT:" in capsys.readouterr().err
    assert plain.exists()
    assert gz.read_bytes() == orig_gz_bytes

    assert main(["convert", str(tapes_dir), "--to", "gzip", "--delete-source", "--force"]) == 0
    capsys.readouterr()
    assert not plain.exists()
    assert gz.exists()

    # P2: HAR queryString, decoded base64 content.size, and case-insensitive Content-Type
    raw_bin = b"\x89PNG\r\n\x1a\n123456"
    b64_str = base64.b64encode(raw_bin).decode("ascii")
    har_tape = tmp_path / "har_edge.tape.json"
    Cassette(
        path=har_tape,
        interactions=[
            Interaction(
                request=RecordedRequest(
                    method="POST",
                    url="https://api.example.com/v1/gen?api-version=2025-01&key=[REDACTED]",
                    headers={"CONTENT-TYPE": "application/json"},
                    body={"x": 1},
                    fingerprint="fp_har",
                    body_encoding="json",
                ),
                response=RecordedResponse(
                    status_code=200,
                    headers={"CONTENT-TYPE": "image/png"},
                    body=b64_str,
                    body_encoding="base64",
                ),
                latency_ms=12.0,
            )
        ],
    ).save()

    assert main(["export", str(har_tape), "--format", "har"]) == 0
    har = json.loads(capsys.readouterr().out)
    entry = har["log"]["entries"][0]
    assert entry["request"]["queryString"] == [
        {"name": "api-version", "value": "2025-01"},
        {"name": "key", "value": "[REDACTED]"},
    ]
    assert entry["response"]["content"]["mimeType"] == "image/png"
    assert entry["response"]["content"]["encoding"] == "base64"
    assert entry["response"]["content"]["size"] == len(raw_bin)
