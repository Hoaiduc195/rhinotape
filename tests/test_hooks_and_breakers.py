"""Tests for Batch 12 recording hooks, metadata, explicit save, and live circuit breakers."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from llm_tape import Cassette, Interaction, TapeAssertionError, TapeMode, replace_response, tape


def _mock_transport(cost_tokens: int = 1000) -> httpx.MockTransport:
    def _handle(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/health"):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": cost_tokens, "completion_tokens": cost_tokens},
            },
        )

    return httpx.MockTransport(_handle)


def test_on_record_filter_transform_and_metadata_and_save(tmp_path: Path) -> None:
    cassette_path = tmp_path / "hooked.tape.json"
    alt_path = tmp_path / "manual_save.tape.json"

    def _on_record(inter: Interaction) -> Interaction | None:
        if inter.request.url.endswith("/health"):
            return None  # drop healthcheck from recording
        return replace_response(inter, latency_ms=42.0)

    with tape(
        cassette_path,
        mode=TapeMode.RECORD,
        on_record=_on_record,
        metadata={"env": "ci", "suite": "agent-v2"},
    ) as session:
        with httpx.Client(transport=_mock_transport()) as client:
            client.get("https://api.openai.com/health")
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert session.call_count == 1
        assert session.interactions[0].latency_ms == 42.0
        saved_alt = session.save(alt_path)
        assert saved_alt == alt_path
        assert alt_path.exists()

    loaded = Cassette.load(cassette_path)
    assert len(loaded.interactions) == 1
    assert loaded.metadata["env"] == "ci"
    assert loaded.metadata["suite"] == "agent-v2"
    assert "llm_tape_version" in loaded.metadata

    # Cassette.add_interaction fluent API
    loaded.add_interaction(loaded.interactions[0])
    assert len(loaded.interactions) == 2


def test_max_interactions_live_guardrail(tmp_path: Path) -> None:
    cassette_path = tmp_path / "max_inter.tape.json"
    calls_made = 0

    def _handle(req: httpx.Request) -> httpx.Response:
        nonlocal calls_made
        calls_made += 1
        return httpx.Response(
            200,
            json={
                "id": f"c-{calls_made}",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    with pytest.raises(TapeAssertionError, match="max_interactions=2"):
        with tape(cassette_path, mode=TapeMode.RECORD, max_interactions=2):
            with httpx.Client(transport=httpx.MockTransport(_handle)) as client:
                for i in range(5):
                    client.post(
                        "https://api.openai.com/v1/chat/completions",
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": str(i)}],
                        },
                    )
    # The 3rd call was blocked BEFORE hitting the transport!
    assert calls_made == 2


def test_max_cost_usd_live_circuit_breaker(tmp_path: Path) -> None:
    cassette_path = tmp_path / "max_cost.tape.json"
    # gpt-4o: 1000 input ($0.0025) + 1000 output ($0.01) = $0.0125 per call
    with pytest.raises(TapeAssertionError, match="Live cost circuit-breaker tripped"):
        with tape(cassette_path, mode=TapeMode.RECORD, max_cost_usd=0.01):
            with httpx.Client(transport=_mock_transport(cost_tokens=1000)) as client:
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                )

    with pytest.raises(ValueError, match="max_interactions"):
        tape(cassette_path, max_interactions=-1)
    with pytest.raises(ValueError, match="max_cost_usd"):
        tape(cassette_path, max_cost_usd=-0.5)
    with pytest.raises(ValueError, match="max_cost_usd"):
        tape(cassette_path, max_cost_usd=float("nan"))
