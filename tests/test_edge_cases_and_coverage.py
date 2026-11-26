"""Additional edge-case tests to guarantee >= 95% line coverage across all modules."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from rhinotape import (
    AsyncTapeTransport,
    Cassette,
    CorruptCassetteError,
    TapeMode,
    TapeSession,
    TapeTransport,
    tape,
)
from rhinotape.cli import main as cli_main
from rhinotape.models import RecordedRequest
from rhinotape.parsers import parse_llm_interaction, parse_sse_events
from rhinotape.pricing import estimate_cost_usd
from rhinotape.scrubbing import (
    decode_body,
    decode_body_with_encoding,
    encode_body,
    encode_body_with_encoding,
    format_request_diff,
    strip_ignored_fields,
)


def test_models_and_pricing_edge_cases() -> None:
    with pytest.raises(ValueError, match="Invalid tape mode"):
        TapeMode.coerce("invalid_mode")

    assert estimate_cost_usd(None, 100, 100) == 0.0
    assert estimate_cost_usd("gpt-4o-mini", 0, 0) == 0.0
    assert estimate_cost_usd("models/gemini-2.5-flash", 1000, 500) > 0.0
    assert estimate_cost_usd("completely-unknown-model", 1000, 500) == 0.0


def test_scrubbing_edge_cases() -> None:
    assert decode_body_with_encoding(b"") == (None, "empty")
    assert encode_body_with_encoding(None, "empty") == b""
    assert decode_body_with_encoding(b"plain text") == ("plain text", "text")
    assert encode_body_with_encoding("plain text", "text") == b"plain text"
    assert decode_body(b'{"a":1}') == {"a": 1}
    assert encode_body(None) == b""
    assert encode_body({"a": 1}) == b'{"a": 1}'
    assert encode_body("raw") == b"raw"

    # Wildcard dict clearing + list index deletion (including multiple indices without shift bug!)
    data = {"items": ["a", "b", "c"], "meta": {"k1": 1, "k2": 2}}
    stripped = strip_ignored_fields(data, ["items.0", "items.2", "meta.*"])
    assert stripped == {"items": ["b"], "meta": {}}

    # Empty candidates & truncated diff
    req = RecordedRequest("POST", "https://api.openai.com/v1/chat/completions", {}, {"a": 1}, "fp")
    assert "no recorded requests" in format_request_diff(req, [])
    big_a = RecordedRequest("POST", "https://x.test", {}, {f"k{i}": i for i in range(60)}, "fp1")
    big_b = RecordedRequest(
        "POST", "https://x.test", {}, {f"k{i}": i + 1 for i in range(60)}, "fp2"
    )
    diff = format_request_diff(big_b, [big_a], max_lines=10)
    assert "(diff truncated)" in diff


def test_parsers_edge_cases() -> None:
    # Non-dict JSON body & unknown provider
    assert parse_llm_interaction("https://example.com/api", {}, "plain text").provider == "unknown"
    assert (
        parse_llm_interaction("https://example.com/api", {}, {"hello": "world"}).provider
        == "unknown"
    )

    # OpenAI multi-part content blocks & invalid JSON tool arguments
    resp = {
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "part A"},
                        {"type": "text", "text": "part B"},
                    ],
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "fn", "arguments": "not-json"}},
                        {"id": "c2", "function": {"name": "fn2", "arguments": ""}},
                    ],
                }
            }
        ],
    }
    summary = parse_llm_interaction("https://api.openai.com/v1/chat/completions", {}, resp)
    assert summary.text == "part A\npart B"
    assert summary.tool_calls[0].arguments == "not-json"
    assert summary.tool_calls[1].arguments == {}

    # SSE comments, trailing data without final newline, and non-JSON SSE data
    events = parse_sse_events(': keepalive\nevent: ping\ndata: hello\n\ndata: {"ok":true}')
    assert events == [("ping", "hello"), (None, {"ok": True})]

    # Unknown SSE provider
    unk_sse = parse_llm_interaction(
        "https://custom.example.com/stream",
        {},
        None,
        stream_chunks=['data: {"foo": 1}\n\n'],
        content_type="text/event-stream",
    )
    assert unk_sse.provider == "unknown" and unk_sse.is_stream is True


def test_cassette_corrupt_interactions_and_non_object_top_level(tmp_path: Path) -> None:
    arr_file = tmp_path / "array.tape.json"
    arr_file.write_text("[]", encoding="utf-8")
    with pytest.raises(CorruptCassetteError, match="top level"):
        Cassette.load(arr_file)

    bad_list = tmp_path / "bad_list.tape.json"
    bad_list.write_text('{"schema_version": 2, "interactions": "nope"}', encoding="utf-8")
    with pytest.raises(CorruptCassetteError, match="Corrupt cassette structure"):
        Cassette.load(bad_list)


@pytest.mark.asyncio
async def test_session_mode_none_save_on_error_and_default_transports(tmp_path: Path) -> None:
    # Default transport creation owns upstream and closes cleanly
    s_none = TapeSession(tmp_path / "none.tape.json", mode="none", patch_httpx=False)
    t_sync = TapeTransport(s_none)
    t_sync.close()
    t_async = AsyncTapeTransport(s_none)
    await t_async.aclose()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    # Async mode=none passthrough
    async with tape(tmp_path / "async_none.tape.json", mode="none"):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            r = await client.get("https://api.openai.com/v1/models")
            assert r.json() == {"ok": True}

    # save_on_error=True persists recorded interactions even if exception raised
    err_path = tmp_path / "save_err.tape.json"
    with pytest.raises(ValueError, match="expected"):
        with tape(err_path, mode="once", save_on_error=True):
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                client.get("https://api.openai.com/v1/models")
            raise ValueError("expected")
    assert err_path.exists()


def test_cli_edge_cases(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing_dir = tmp_path / "no_such_dir"
    assert cli_main(["summary", str(missing_dir)]) == 1
    assert cli_main(["verify", str(missing_dir)]) == 1
    assert cli_main(["inspect", str(tmp_path / "missing.tape.json")]) == 1
    capsys.readouterr()

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert cli_main(["summary", str(empty_dir)]) == 0
    assert "No .tape.json" in capsys.readouterr().out

    assert cli_main(["summary", str(empty_dir), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["files"] == []

    # diff --json + summary --json with a valid file
    c1 = empty_dir / "a.tape.json"
    Cassette(path=c1, interactions=[]).save()
    assert cli_main(["summary", str(empty_dir), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["files"] == 1

    assert cli_main(["diff", str(c1), str(c1), "--json"]) == 0
    diff_data = json.loads(capsys.readouterr().out)
    assert diff_data["delta"]["calls"] == 0
