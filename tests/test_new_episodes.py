"""Tests for Wave 9: TapeMode.NEW_EPISODES (replay existing + record & append new calls)."""

from __future__ import annotations

from pathlib import Path

import httpx

from llm_tape import Cassette, TapeMode, TapeTransport, tape


def test_new_episodes_records_fresh_then_replays_and_appends_only_new(tmp_path: Path) -> None:
    tape_path = tmp_path / "agent.tape.json"
    live_calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(req.content.decode())
        step = str(body["step"])
        live_calls.append(step)
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": f"reply-{step}"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    upstream = httpx.MockTransport(handler)

    # 1. Initial run with NEW_EPISODES creates cassette with step 1
    with tape(tape_path, mode=TapeMode.NEW_EPISODES, patch_httpx=False) as s1:
        with httpx.Client(transport=TapeTransport(s1, upstream=upstream)) as c:
            r1 = c.post("https://api.openai.com/v1/chat/completions", json={"step": "one"})
            assert r1.json()["choices"][0]["message"]["content"] == "reply-one"
    assert live_calls == ["one"]
    assert len(Cassette.load(tape_path).interactions) == 1

    # 2. Add custom metadata, then second run replays step 1 offline AND appends step 2
    c_meta = Cassette.load(tape_path)
    c_meta.metadata["custom_tag"] = "keep-me"
    c_meta.save()

    with tape(tape_path, mode="new_episodes", patch_httpx=False) as s2:
        with httpx.Client(transport=TapeTransport(s2, upstream=upstream)) as c:
            r1 = c.post("https://api.openai.com/v1/chat/completions", json={"step": "one"})
            r2 = c.post("https://api.openai.com/v1/chat/completions", json={"step": "two"})
            assert r1.json()["choices"][0]["message"]["content"] == "reply-one"
            assert r2.json()["choices"][0]["message"]["content"] == "reply-two"
        s2.assert_call_count(2)
    assert live_calls == ["one", "two"]  # step "one" was NOT sent live!
    updated_cassette = Cassette.load(tape_path)
    assert len(updated_cassette.interactions) == 2
    assert updated_cassette.metadata["custom_tag"] == "keep-me"

    # 3. Third run replays both completely offline and does NOT rewrite the file
    mtime_before = tape_path.stat().st_mtime_ns
    with tape(tape_path, mode=TapeMode.NEW_EPISODES, patch_httpx=False) as s3:
        with httpx.Client(transport=TapeTransport(s3, upstream=upstream)) as c:
            c.post("https://api.openai.com/v1/chat/completions", json={"step": "one"})
            c.post("https://api.openai.com/v1/chat/completions", json={"step": "two"})
        s3.assert_call_count(2)
        s3.assert_no_unused_interactions()
    assert live_calls == ["one", "two"]
    assert tape_path.stat().st_mtime_ns == mtime_before

    # 4. Strict REPLAY works on the appended 2-interaction cassette
    with tape(tape_path, mode=TapeMode.REPLAY, patch_httpx=False) as s4:
        with httpx.Client(transport=TapeTransport(s4)) as c:
            assert (
                c.post("https://api.openai.com/v1/chat/completions", json={"step": "one"}).json()[
                    "choices"
                ][0]["message"]["content"]
                == "reply-one"
            )
            assert (
                c.post("https://api.openai.com/v1/chat/completions", json={"step": "two"}).json()[
                    "choices"
                ][0]["message"]["content"]
                == "reply-two"
            )
        s4.assert_no_unused_interactions()
