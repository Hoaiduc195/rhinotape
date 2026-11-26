"""Example: Chaos testing (429 retry injection) + NEW_EPISODES + match_headers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import httpx

from rhinotape import Interaction, TapeMode, TapeTransport, replace_response, tape


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cassette_path = Path(tmp) / "chaos_agent.tape.json.gz"

        def mock_llm(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode())
            step = payload.get("step", "1")
            return httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"role": "assistant", "content": f"Step {step} OK"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                },
            )

        upstream = httpx.MockTransport(mock_llm)

        # 1. Record step 1 into a gzip-compressed cassette
        with tape(
            cassette_path,
            mode=TapeMode.NEW_EPISODES,
            match_headers=["x-agent-tier"],
            patch_httpx=False,
        ) as s1:
            with httpx.Client(transport=TapeTransport(s1, upstream=upstream)) as client:
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"x-agent-tier": "prod"},
                    json={"step": 1},
                )

        # 2. Replay step 1 offline AND append new step 2
        with tape(
            cassette_path,
            mode=TapeMode.NEW_EPISODES,
            match_headers=["x-agent-tier"],
            patch_httpx=False,
        ) as s2:
            with httpx.Client(transport=TapeTransport(s2, upstream=upstream)) as client:
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"x-agent-tier": "prod"},
                    json={"step": 1},
                )
                client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"x-agent-tier": "prod"},
                    json={"step": 2},
                )

        # 3. Chaos replay: inject a 429 on the first attempt, then verify retry succeeds
        def inject_transient_429(inter: Interaction, call_idx: int) -> Interaction:
            if call_idx == 0:
                return replace_response(inter, status_code=429, body={"error": "rate_limit"})
            return inter

        with tape(
            cassette_path,
            mode=TapeMode.REPLAY,
            match_headers=["x-agent-tier"],
            allow_playback_repeats=True,
            on_replay=inject_transient_429,
        ) as s3:
            r1 = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"x-agent-tier": "prod"},
                json={"step": 1},
            )
            assert r1.status_code == 429
            r2 = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"x-agent-tier": "prod"},
                json={"step": 2},
            )
            assert r2.status_code == 200
            s3.assert_text_contains("Step 2 OK")
            print("Chaos replay summary:", json.dumps(s3.export_summary(), indent=2))


if __name__ == "__main__":
    main()
