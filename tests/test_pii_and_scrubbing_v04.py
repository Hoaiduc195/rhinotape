"""Tests for PII presets, redact_pii, wildcard header scrubbing, and URL path redaction."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from rhinotape import (
    PII_PRESETS,
    Cassette,
    TapeMode,
    resolve_pii_patterns,
    tape,
)
from rhinotape.cli import main as cli_main
from rhinotape.scrubbing import (
    REDACTED,
    URL_USERINFO_REDACTED,
    redact_text,
    scrub_headers,
    scrub_url,
)


def test_pii_presets_and_resolver() -> None:
    assert set(PII_PRESETS) == {"email", "phone", "ssn", "credit_card", "ipv4", "jwt"}
    assert resolve_pii_patterns(False) == ()
    assert len(resolve_pii_patterns(True)) == len(PII_PRESETS)
    assert len(resolve_pii_patterns("all")) == len(PII_PRESETS)
    assert len(resolve_pii_patterns("email")) == 1
    assert len(resolve_pii_patterns(["email", "ssn", "email"])) == 2

    with pytest.raises(ValueError, match="Unknown PII preset"):
        resolve_pii_patterns(["not_a_preset"])

    sample = (
        "Contact alice@example.com or 415-555-1234, SSN 123-45-6789, "
        "card 4111-2222-3333-4444, ip 192.168.1.42, "
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature1234-"
    )
    redacted = redact_text(sample, resolve_pii_patterns(True))
    assert "alice@example.com" not in redacted
    assert "415-555-1234" not in redacted
    assert "123-45-6789" not in redacted
    assert "4111-2222-3333-4444" not in redacted
    assert "192.168.1.42" not in redacted
    assert "eyJhbGciOiJIUzI1NiJ9" not in redacted
    assert f"{REDACTED}-" not in redacted
    assert redacted.count(REDACTED) == 6


def test_tape_session_redact_pii_and_url_patterns(tmp_path: Path) -> None:
    cassette_path = tmp_path / "pii.tape.json"

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-pii",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Sent to bob@company.org from 10.0.0.5",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            },
        )

    transport = httpx.MockTransport(handler)
    with tape(
        cassette_path,
        mode=TapeMode.RECORD,
        redact_pii=["email", "ipv4"],
        redact_url_patterns=[r"(?<=/tenants/)[^/]+"],
    ):
        with httpx.Client(transport=transport) as client:
            resp = client.post(
                "https://api.openai.com/tenants/secret-org-42/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "email alice@example.com"}],
                },
            )
            assert resp.status_code == 200

    loaded = Cassette.load(cassette_path)
    inter = loaded.interactions[0]
    assert "secret-org-42" not in inter.request.url
    assert f"/tenants/{URL_USERINFO_REDACTED}/" in inter.request.url
    assert "alice@example.com" not in str(inter.request.body)
    assert "bob@company.org" not in str(inter.response.body)
    assert "10.0.0.5" not in str(inter.response.body)

    # Replaying with the same redact_pii + redact_url_patterns matches cleanly
    with tape(
        cassette_path,
        mode=TapeMode.REPLAY,
        redact_pii=["email", "ipv4"],
        redact_url_patterns=[r"(?<=/tenants/)[^/]+"],
    ) as session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            resp = client.post(
                "https://api.openai.com/tenants/secret-org-42/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "email alice@example.com"}],
                },
            )
            assert resp.status_code == 200
            assert session.call_count == 1


def test_cli_redact_pii(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cassette_path = tmp_path / "raw.tape.json"
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "1",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "SSN 123-45-6789"}}],
            },
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD):
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "a@b.com"}]},
            )

    rc = cli_main(["redact", str(cassette_path), "--pii", "email,ssn"])
    assert rc == 0
    loaded = Cassette.load(cassette_path)
    assert "a@b.com" not in str(loaded.interactions[0].request.body)
    assert "123-45-6789" not in str(loaded.interactions[0].response.body)

    rc_bad = cli_main(["redact", str(cassette_path), "--pii", "bogus"])
    assert rc_bad == 1
    assert "Unknown PII preset" in capsys.readouterr().err


def test_scrub_headers_wildcards() -> None:
    headers = {
        "X-Secret-Key": "s1",
        "X-Secret-Region": "s2",
        "X-Custom-Token": "tok",
        "Content-Type": "application/json",
    }
    cleaned = scrub_headers(headers, extra_sensitive=["x-secret-*", "*-token"])
    assert cleaned["X-Secret-Key"] == REDACTED
    assert cleaned["X-Secret-Region"] == REDACTED
    assert cleaned["X-Custom-Token"] == REDACTED
    assert cleaned["Content-Type"] == "application/json"


def test_scrub_url_path_patterns() -> None:
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj-99/locations/us/models/gemini"
    cleaned = scrub_url(url, redact_url_patterns=[r"(?<=/projects/)[^/]+"])
    assert cleaned == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/REDACTED/locations/us/models/gemini"
    )
