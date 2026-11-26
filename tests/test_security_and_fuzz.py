"""Security and fuzz/edge-case tests (Batch 17, Waves 98-101)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import rhinotape.cassette as cassette_mod
from rhinotape import Cassette, CorruptCassetteError, TapeMode, TapeReplayServer, tape
from rhinotape.cli import main as cli_main
from rhinotape.parsers import parse_llm_interaction
from rhinotape.scrubbing import decode_body_with_encoding, encode_body_with_encoding


def test_oversized_plain_cassette_rejected_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "big.tape.json"
    p.write_text('{"schema_version": 2, "interactions": []}\n', encoding="utf-8")
    monkeypatch.setattr(cassette_mod, "MAX_CASSETTE_BYTES", 10)
    with pytest.raises(CorruptCassetteError, match="safety size limit"):
        Cassette.load(p)


def test_server_404_redacts_sensitive_query_params(tmp_path: Path) -> None:
    c = Cassette(path=tmp_path / "empty.tape.json", interactions=[])
    with TapeReplayServer(c) as srv:
        with httpx.Client() as client:
            r = client.get(
                f"{srv.url}/v1/models?key=AIzaSuperSecretKey12345678901234567890&foo=bar"
            )
            assert r.status_code == 404
            payload = r.json()
            assert "AIzaSuperSecretKey" not in payload["path"]
            assert "REDACTED" in payload["path"]
            assert "foo=bar" in payload["path"]


def test_cli_verify_check_pii(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "pii_leak.tape.json"
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "1",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "Email bob@corp.com"}}],
            },
        )
    )
    with tape(p, mode=TapeMode.RECORD):
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )

    assert cli_main(["verify", str(p), "--check-pii", "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert any(f["pattern"] == "pii_email" for f in out["secret_findings"])

    # After redacting PII, verify --check-pii passes
    assert cli_main(["redact", str(p), "--pii"]) == 0
    capsys.readouterr()
    assert cli_main(["verify", str(p), "--check-pii"]) == 0


def test_fuzz_malformed_utf8_empty_stream_and_invalid_json() -> None:
    # Malformed UTF-8 round-trips cleanly via base64 encoding
    bad_bytes = b"\xff\xfe\x00\x80invalid-utf8"
    val, enc = decode_body_with_encoding(bad_bytes)
    assert enc == "base64"
    assert encode_body_with_encoding(val, enc) == bad_bytes

    # Empty SSE stream does not crash parser
    summary = parse_llm_interaction(
        url="https://api.openai.com/v1/chat/completions",
        request_body={"model": "gpt-4o", "stream": True},
        response_body=None,
        stream_chunks=("", "   \n\n", "data: [DONE]\n\n"),
        content_type="text/event-stream",
    )
    assert summary.provider == "openai"
    assert summary.is_stream is True
    assert summary.text == ""

    # Corrupt JSON string in Cassette.from_json raises CorruptCassetteError
    with pytest.raises(CorruptCassetteError):
        Cassette.from_json("{not valid json")
    with pytest.raises(CorruptCassetteError):
        Cassette.from_json("[]")
    with pytest.raises(CorruptCassetteError):
        Cassette.from_json('{"schema_version": 999, "interactions": []}')
