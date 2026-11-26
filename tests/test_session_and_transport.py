"""Tests for TapeSession, TapeTransport, AsyncTapeTransport, and assertions."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from rhinotape import (
    AsyncTapeTransport,
    TapeAssertionError,
    TapeNotFoundError,
    TapeSession,
    TapeTransport,
    UnmatchedRequestError,
    tape,
)


def _openai_response(text: str = "Hello!", tool_name: str | None = None) -> dict:
    message: dict = {"role": "assistant", "content": text}
    if tool_name:
        message["tool_calls"] = [
            {
                "id": f"call_{tool_name}",
                "type": "function",
                "function": {"name": tool_name, "arguments": '{"q": "test"}'},
            }
        ]
    return {
        "id": "chatcmpl-test",
        "model": "gpt-4o-mini",
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def test_sync_record_then_replay_with_explicit_transport(tmp_path: Path) -> None:
    cassette_path = tmp_path / "sync.tape.json"
    upstream_hits = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_hits
        upstream_hits += 1
        return httpx.Response(200, json=_openai_response("Recorded answer", tool_name="search"))

    upstream = httpx.MockTransport(handler)

    # First run: records interaction
    with TapeSession(cassette_path, mode="once", patch_httpx=False) as session:
        with httpx.Client(transport=TapeTransport(session, upstream)) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": "Bearer sk-top-secret-123"},
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            )
            assert resp.status_code == 200
            assert not session.is_replaying
            session.assert_call_count(1)
            session.assert_tokens_under(200)
            session.assert_cost_under(0.01)
            session.assert_tool_called("search")
            session.assert_tool_sequence(["search"])

    assert upstream_hits == 1
    assert cassette_path.exists()
    assert "sk-top-secret-123" not in cassette_path.read_text(encoding="utf-8")

    # Second run: replays strictly offline without hitting upstream
    with TapeSession(cassette_path, mode="replay", patch_httpx=False) as session:
        with httpx.Client(transport=TapeTransport(session, upstream)) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": "Bearer different-key"},
                json={"messages": [{"role": "user", "content": "Hi"}], "model": "gpt-4o-mini"},
            )
            assert resp.status_code == 200
            assert resp.json()["choices"][0]["message"]["content"] == "Recorded answer"
            assert session.is_replaying
            session.assert_tool_called("search")

    assert upstream_hits == 1  # Never called again!


def test_auto_httpx_patching_sync_and_assertions(tmp_path: Path) -> None:
    cassette_path = tmp_path / "patched.tape.json"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        tool = "step_one" if calls == 1 else "step_two"
        return httpx.Response(200, json=_openai_response(f"turn {calls}", tool_name=tool))

    mock_transport = httpx.MockTransport(handler)

    with tape(cassette_path, mode="once") as session:
        with httpx.Client(transport=mock_transport) as client:
            client.post("https://api.openai.com/v1/chat/completions", json={"turn": 1})
            client.post("https://api.openai.com/v1/chat/completions", json={"turn": 2})

        session.assert_call_count(2)
        session.assert_tool_sequence(["step_one", "step_two"])
        assert session.total_tokens == 300

        with pytest.raises(TapeAssertionError):
            session.assert_tokens_under(100)
        with pytest.raises(TapeAssertionError):
            session.assert_cost_under(0.000001)
        with pytest.raises(TapeAssertionError):
            session.assert_tool_called("nonexistent")
        with pytest.raises(TapeAssertionError):
            session.assert_tool_sequence(["step_two", "step_one"])
        with pytest.raises(TapeAssertionError):
            session.assert_call_count(99)

    # Replay through patched httpx.Client with a transport that would fail if ever invoked
    def forbidden_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Upstream should not be called during replay!")

    with tape(cassette_path, mode="replay") as session:
        with httpx.Client(transport=httpx.MockTransport(forbidden_handler)) as client:
            r1 = client.post("https://api.openai.com/v1/chat/completions", json={"turn": 1})
            r2 = client.post("https://api.openai.com/v1/chat/completions", json={"turn": 2})
            assert r1.json()["choices"][0]["message"]["content"] == "turn 1"
            assert r2.json()["choices"][0]["message"]["content"] == "turn 2"

            with pytest.raises(UnmatchedRequestError):
                client.post("https://api.openai.com/v1/chat/completions", json={"turn": 999})


@pytest.mark.asyncio
async def test_async_record_and_replay(tmp_path: Path) -> None:
    cassette_path = tmp_path / "async.tape.json"
    hits = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(200, json=_openai_response("Async hello"))

    upstream = httpx.MockTransport(handler)

    async with TapeSession(cassette_path, mode="once", patch_httpx=False) as session:
        async with httpx.AsyncClient(transport=AsyncTapeTransport(session, upstream)) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": []},
            )
            assert resp.json()["choices"][0]["message"]["content"] == "Async hello"

    assert hits == 1

    async with TapeSession(cassette_path, mode="replay", patch_httpx=False) as session:
        async with httpx.AsyncClient(transport=AsyncTapeTransport(session, upstream)) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": []},
            )
            assert resp.json()["choices"][0]["message"]["content"] == "Async hello"
            session.assert_call_count(1)

    assert hits == 1


def test_replay_missing_cassette_raises(tmp_path: Path) -> None:
    with pytest.raises(TapeNotFoundError):
        with tape(tmp_path / "missing.tape.json", mode="replay"):
            pass
