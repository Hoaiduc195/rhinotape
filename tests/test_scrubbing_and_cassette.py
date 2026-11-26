"""Tests for secret scrubbing, fingerprinting, and cassette serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from rhinotape import Cassette, Interaction, TapeNotFoundError
from rhinotape.models import LLMCallSummary, RecordedRequest, RecordedResponse, Usage
from rhinotape.scrubbing import REDACTED, fingerprint_request, scrub_headers, scrub_url


def test_scrub_headers_redacts_secrets() -> None:
    headers = {
        "Authorization": "Bearer sk-live-secret",
        "x-api-key": "ant-secret",
        "X-Goog-Api-Key": "AIza-secret",
        "Content-Type": "application/json",
        "X-Custom-Token": "keep-unless-listed",
    }
    cleaned = scrub_headers(headers, extra_sensitive=["x-custom-token"])
    assert cleaned["Authorization"] == REDACTED
    assert cleaned["x-api-key"] == REDACTED
    assert cleaned["X-Goog-Api-Key"] == REDACTED
    assert cleaned["X-Custom-Token"] == REDACTED
    assert cleaned["Content-Type"] == "application/json"


def test_scrub_url_redacts_and_sorts_query_params() -> None:
    url = "https://generativelanguage.googleapis.com/v1/models/gemini:generateContent?z=1&key=SECRET&a=2"
    cleaned = scrub_url(url)
    assert "SECRET" not in cleaned
    assert "key=%5BREDACTED%5D" in cleaned
    assert cleaned.endswith("?a=2&key=%5BREDACTED%5D&z=1")


def test_fingerprint_is_invariant_to_json_key_order() -> None:
    url = "https://api.openai.com/v1/chat/completions"
    fp1 = fingerprint_request("POST", url, {"b": 2, "a": 1})
    fp2 = fingerprint_request("post", url, {"a": 1, "b": 2})
    assert fp1 == fp2


def test_cassette_roundtrip_and_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.tape.json"
    with pytest.raises(TapeNotFoundError):
        Cassette.load(missing)

    path = tmp_path / "sub" / "sample.tape.json"
    inter = Interaction(
        request=RecordedRequest(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            headers={"authorization": REDACTED},
            body={"model": "gpt-4o-mini"},
            fingerprint="abc123",
        ),
        response=RecordedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"choices": []},
        ),
        latency_ms=42.5,
        llm=LLMCallSummary(
            provider="openai",
            model="gpt-4o-mini",
            usage=Usage(input_tokens=10, output_tokens=20),
            estimated_cost_usd=0.0000135,
        ),
    )
    Cassette(path=path, interactions=[inter]).save()

    loaded = Cassette.load(path)
    assert len(loaded.interactions) == 1
    assert loaded.interactions[0].request.fingerprint == "abc123"
    assert loaded.interactions[0].llm.usage.total_tokens == 30
