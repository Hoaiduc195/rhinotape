"""Tests for Wave 3: on_replay chaos/fault injection and replace_response helper."""

from __future__ import annotations

from pathlib import Path

import aiohttp
import httpx
import pytest
import requests

from llm_tape import (
    Interaction,
    TapeAssertionError,
    TapeMode,
    TapeTransport,
    replace_response,
    tape,
)


def _record_basic_tape(path: Path) -> None:
    upstream = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "original"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )
    )
    with tape(path, mode=TapeMode.RECORD, patch_httpx=False) as rec:
        with httpx.Client(transport=TapeTransport(rec, upstream=upstream)) as c:
            c.post("https://api.openai.com/v1/chat/completions", json={"q": "hi"})


def test_on_replay_chaos_429_retry_across_httpx_and_requests(tmp_path: Path) -> None:
    tape_path = tmp_path / "chaos.tape.json"
    _record_basic_tape(tape_path)

    def inject_first_429(inter: Interaction, call_idx: int) -> Interaction:
        if call_idx == 0:
            return replace_response(
                inter,
                status_code=429,
                body={"error": "rate_limit"},
                headers={"content-type": "application/json", "retry-after": "1"},
            )
        return inter

    with tape(
        tape_path,
        mode=TapeMode.REPLAY,
        allow_playback_repeats=True,
        on_replay=inject_first_429,
    ) as s:
        r1 = httpx.post("https://api.openai.com/v1/chat/completions", json={"q": "hi"})
        assert r1.status_code == 429
        assert r1.headers["retry-after"] == "1"
        assert r1.json() == {"error": "rate_limit"}

        # Retry succeeds via requests
        r2 = requests.post("https://api.openai.com/v1/chat/completions", json={"q": "hi"})
        assert r2.status_code == 200
        assert r2.json()["choices"][0]["message"]["content"] == "original"

        s.assert_call_count(2)
        with pytest.raises(TapeAssertionError, match="429"):
            s.assert_no_errors()


@pytest.mark.asyncio
async def test_on_replay_mutates_llm_body_and_validates_return_type(tmp_path: Path) -> None:
    tape_path = tmp_path / "mutate.tape.json"
    _record_basic_tape(tape_path)

    def mutate_text(inter: Interaction, call_idx: int) -> Interaction | None:
        del call_idx
        return replace_response(
            inter,
            body={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "mutated"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            },
        )

    async with tape(tape_path, mode=TapeMode.REPLAY, on_replay=mutate_text) as s:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                "https://api.openai.com/v1/chat/completions", json={"q": "hi"}
            ) as resp:
                data = await resp.json()
                assert data["choices"][0]["message"]["content"] == "mutated"
        assert s.total_tokens == 60
        assert s.interactions[0].llm.text == "mutated"

    # Invalid return type raises TypeError cleanly
    with tape(
        tape_path,
        mode=TapeMode.REPLAY,
        on_replay=lambda inter, idx: "bad",  # type: ignore[arg-type,return-value]
    ):
        with pytest.raises(TypeError, match="must return Interaction or None"):
            httpx.post("https://api.openai.com/v1/chat/completions", json={"q": "hi"})


def test_replace_response_bytes_stream_clear_and_custom_pricing(tmp_path: Path) -> None:
    from llm_tape import Cassette, ModelPrice

    tape_path = tmp_path / "edge.tape.json"
    _record_basic_tape(tape_path)
    orig = Cassette.load(tape_path).interactions[0]

    # P1: bytes payload decoded faithfully (binary -> base64, not repr)
    bin_bytes = b"\x89PNG\r\n\x1a\n\xff\xfe"
    with tape(
        tape_path,
        mode=TapeMode.REPLAY,
        on_replay=lambda inter, i: replace_response(inter, body=bin_bytes),
    ):
        r = httpx.post("https://api.openai.com/v1/chat/completions", json={"q": "hi"})
        assert r.content == bin_bytes

    # P2 #2: clearing stream_chunks=None resets body_encoding from "stream" to "empty"
    streamed = replace_response(orig, stream_chunks=('data: {"a":1}\n\n',))
    assert streamed.response.body_encoding == "stream"
    cleared = replace_response(streamed, stream_chunks=None)
    assert cleared.response.stream_chunks is None
    assert cleared.response.body_encoding == "empty"

    # P2 #3: custom_pricing forwarded cleanly
    priced = replace_response(
        orig,
        body={
            "model": "my-custom-model",
            "choices": [{"message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
        },
        custom_pricing={"my-custom-model": ModelPrice(5.0, 10.0)},
    )
    assert priced.llm.estimated_cost_usd == pytest.approx(5.0)
