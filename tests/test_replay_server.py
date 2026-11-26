"""Integration tests for Batch 2 (Waves 23-27): TapeReplayServer and CLI serve."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import httpx
import pytest
import requests

from llm_tape import Cassette, TapeReplayServer
from llm_tape.cli import main as cli_main
from llm_tape.models import Interaction, RecordedRequest, RecordedResponse


def test_tape_replay_server_json_sse_health_stats_and_clients(tmp_path: Path) -> None:
    tapes_dir = tmp_path / "tapes"
    Cassette(
        path=tapes_dir / "chat.tape.json",
        interactions=[
            Interaction(
                request=RecordedRequest(
                    method="POST",
                    url="https://api.openai.com/v1/chat/completions",
                    headers={"content-type": "application/json"},
                    body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
                    fingerprint="orig",
                ),
                response=RecordedResponse(
                    status_code=200,
                    headers={"content-type": "application/json"},
                    body={"choices": [{"message": {"content": "served offline"}}]},
                ),
                latency_ms=5.0,
            ),
            Interaction(
                request=RecordedRequest(
                    method="POST",
                    url="https://api.openai.com/v1/chat/sse",
                    headers={"content-type": "application/json"},
                    body={"stream": True},
                    fingerprint="orig_sse",
                ),
                response=RecordedResponse(
                    status_code=200,
                    headers={"content-type": "text/event-stream"},
                    body=None,
                    body_encoding="stream",
                    stream_chunks=(
                        'data: {"delta":"a"}\n\n',
                        'data: {"delta":"b"}\n\n',
                        "data: [DONE]\n\n",
                    ),
                ),
                latency_ms=10.0,
            ),
        ],
    ).save()

    with TapeReplayServer(tapes_dir, replay_latency_scale=0.1) as srv:
        # Health & stats endpoints
        assert httpx.get(f"{srv.url}/__tape__/health").json()["status"] == "ok"

        # Replay via httpx
        r1 = httpx.post(
            f"{srv.url}/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r1.status_code == 200
        assert r1.json()["choices"][0]["message"]["content"] == "served offline"

        # Replay via requests (SSE stream)
        with requests.post(f"{srv.url}/v1/chat/sse", json={"stream": True}, stream=True) as r2:
            assert r2.status_code == 200
            lines = [ln.decode().strip() for ln in r2.iter_lines() if ln]
            assert len(lines) == 3

        # Replay via stdlib urllib
        req = urllib.request.Request(
            f"{srv.url}/v1/chat/completions",
            data=json.dumps(
                {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r3:
            assert json.loads(r3.read().decode())["choices"][0]["message"]["content"] == (
                "served offline"
            )

        # Unmatched request returns 404
        r404 = httpx.post(f"{srv.url}/v1/unknown", json={"x": 1})
        assert r404.status_code == 404

        st = httpx.get(f"{srv.url}/__tape__/stats").json()
        assert st["loaded_interactions"] == 2
        assert st["served_requests"] == 3
        assert st["unmatched_requests"] == 1

    # Invalid latency scale & missing path CLI error
    with pytest.raises(ValueError, match="finite"):
        TapeReplayServer(tapes_dir, replay_latency_scale=float("nan"))
    assert cli_main(["serve", str(tmp_path / "missing.tape.json")]) == 1
