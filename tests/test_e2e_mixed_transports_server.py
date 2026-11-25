"""End-to-end mixed-transport (httpx, requests, aiohttp, urllib) via TapeReplayServer (Wave 97)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import aiohttp
import httpx
import pytest
import requests

from llm_tape import TapeMode, TapeReplayServer, tape


@pytest.mark.asyncio
async def test_all_four_http_clients_against_replay_server(tmp_path: Path) -> None:
    cassette_path = tmp_path / "shared_server.tape.json"
    body: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "ping"}],
    }
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "id": "chatcmpl-srv",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "pong"}}],
            },
        )
    )
    with tape(cassette_path, mode=TapeMode.RECORD):
        with httpx.Client(transport=transport) as client:
            client.post("https://api.openai.com/v1/chat/completions", json=body)

    with TapeReplayServer(cassette_path, allow_playback_repeats=True) as srv:
        url = f"{srv.url}/v1/chat/completions"

        # 1. httpx
        with httpx.Client() as client:
            r_httpx = client.post(url, json=body)
            assert r_httpx.status_code == 200
            assert r_httpx.json()["choices"][0]["message"]["content"] == "pong"

        # 2. requests
        r_req = requests.post(url, json=body, timeout=5)
        assert r_req.status_code == 200
        assert r_req.json()["choices"][0]["message"]["content"] == "pong"

        # 3. aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body) as r_aio:
                assert r_aio.status == 200
                data = await r_aio.json()
                assert data["choices"][0]["message"]["content"] == "pong"

        # 4. urllib.request
        u_req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(u_req, timeout=5) as u_resp:
            assert u_resp.status == 200
            u_data = json.loads(u_resp.read().decode("utf-8"))
            assert u_data["choices"][0]["message"]["content"] == "pong"

        stats = srv.stats()
        assert stats["served_requests"] == 4
        assert stats["unmatched_requests"] == 0
