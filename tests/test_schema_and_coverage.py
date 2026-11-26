"""Tests for Wave 13 (JSON Schema CLI + export_summary) and Wave 14 (adapter coverage push)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import httpx
import pytest

from rhinotape import TapeMode, TapeTransport, tape
from rhinotape.aiohttp_adapter import (
    _extract_request_body,
    _RecordingAiohttpStreamReader,
    _ReplayAiohttpStreamReader,
)
from rhinotape.aiohttp_adapter import (
    _status_reason as aio_status_reason,
)
from rhinotape.cli import main as cli_main
from rhinotape.urllib_adapter import (
    _extract_urllib_request,
    _ReplayUrllibResponse,
)
from rhinotape.urllib_adapter import (
    _status_reason as urllib_status_reason,
)


def test_cli_schema_and_session_export_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Stdout JSON schema
    assert cli_main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["maximum"] == 2

    # File output JSON schema
    out_file = tmp_path / "schemas" / "cassette.schema.json"
    assert cli_main(["schema", "-o", str(out_file)]) == 0
    assert "Wrote JSON Schema" in capsys.readouterr().out
    assert json.loads(out_file.read_text(encoding="utf-8")) == schema

    # TapeSession.export_summary()
    tape_path = tmp_path / "sum.tape.json"
    upstream = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 25},
            },
        )
    )
    with tape(tape_path, mode=TapeMode.RECORD, patch_httpx=False) as s:
        with httpx.Client(transport=TapeTransport(s, upstream=upstream)) as c:
            c.post("https://api.openai.com/v1/chat/completions", json={"q": "hi"})
        summary = s.export_summary()
        assert summary["calls"] == 1
        assert summary["total_tokens"] == 125
        assert summary["providers"] == ["openai"]
        assert summary["models"] == ["gpt-4o-mini"]
        assert summary["mode"] == "record"


@pytest.mark.asyncio
async def test_aiohttp_and_urllib_adapter_branch_coverage() -> None:
    # aiohttp helpers
    assert aio_status_reason(200) == "OK"
    assert aio_status_reason(999) == ""
    assert _extract_request_body({"json": {"a": 1}}) == b'{"a": 1}'
    assert _extract_request_body({"data": b"raw"}) == b"raw"
    assert _extract_request_body({"data": bytearray(b"ba")}) == b"ba"
    assert _extract_request_body({"data": "hello"}) == b"hello"
    assert _extract_request_body({"data": {"k": "v"}}) == b"k=v"
    assert _extract_request_body({"data": 42}) == b"42"
    assert _extract_request_body({}) == b""

    # _RecordingAiohttpStreamReader all methods
    committed: list[list[str]] = []
    rec1 = _RecordingAiohttpStreamReader(
        _ReplayAiohttpStreamReader((b"line1\nline2\n",)), committed.append
    )
    assert await rec1.read(0) == b""
    assert await rec1.readline() == b"line1\n"
    assert await rec1.read(6) == b"line2\n"
    assert await rec1.read(6) == b""
    assert committed == [["line1\n", "line2\n"]]

    committed2: list[list[str]] = []
    rec2 = _RecordingAiohttpStreamReader(
        _ReplayAiohttpStreamReader((b"ab", b"cd")), committed2.append
    )
    assert [c async for c in rec2.iter_any()] == [b"ab", b"cd"]
    assert committed2 == [["ab", "cd"]]

    committed3: list[list[str]] = []
    rec3 = _RecordingAiohttpStreamReader(_ReplayAiohttpStreamReader((b"1234",)), committed3.append)
    assert [c async for c in rec3.iter_chunked(2)] == [b"12", b"34"]
    assert committed3 == [["12", "34"]]

    committed4: list[list[str]] = []
    rec4 = _RecordingAiohttpStreamReader(_ReplayAiohttpStreamReader((b"xy",)), committed4.append)
    assert await rec4.readany() == b"xy"
    assert await rec4.readany() == b""
    assert committed4 == [["xy"]]

    committed5: list[list[str]] = []
    rec5 = _RecordingAiohttpStreamReader(_ReplayAiohttpStreamReader((b"pq",)), committed5.append)
    assert await rec5.readchunk() == (b"pq", True)
    assert await rec5.readchunk() == (b"", False)
    assert committed5 == [["pq"]]

    # urllib helpers
    assert urllib_status_reason(200) == "OK"
    assert urllib_status_reason(999) == ""
    resp = _ReplayUrllibResponse(
        url="https://api.openai.com/v1",
        status=200,
        headers={"Content-Type": "text/plain"},
        body_bytes=b"l1\nl2\n",
    )
    assert resp.geturl() == "https://api.openai.com/v1"
    assert resp.info().get("Content-Type") == "text/plain"
    assert resp.readline() == b"l1\n"
    assert resp.readlines() == [b"l2\n"]
    resp.close()
    assert resp.closed is True

    m, u, _h, b, _d = _extract_urllib_request("https://api.openai.com/v1", [b"a", "b"])
    assert m == "POST" and u == "https://api.openai.com/v1" and b == b"ab"
    req_str = urllib.request.Request("https://api.openai.com/v1", data=bytearray(b"xy"))
    _m2, _u2, _h2, b2, _d2 = _extract_urllib_request(req_str, None)
    assert b2 == b"xy"
