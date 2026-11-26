"""Tests for Ponytail review fixes (server security/matching/caps) and `rhinotape run`."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import httpx
import pytest

import rhinotape.server as server_mod
from rhinotape import Cassette, Interaction, RecordedRequest, RecordedResponse, TapeMode
from rhinotape.cli import main as cli_main
from rhinotape.models import LLMCallSummary, Usage
from rhinotape.scrubbing import fingerprint_request
from rhinotape.server import TapeReplayServer


def _make_sample_cassette(path: Path, *, tokens: int = 15, cost: float = 0.001) -> Cassette:
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    url = "https://api.openai.com/v1/chat/completions"
    fp = fingerprint_request("POST", url, body)
    inter = Interaction(
        request=RecordedRequest(
            method="POST",
            url=url,
            headers={"content-type": "application/json"},
            body=body,
            fingerprint=fp,
        ),
        response=RecordedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "pong"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": max(0, tokens - 10)},
            },
        ),
        latency_ms=12.0,
        llm=LLMCallSummary(
            provider="openai",
            model="gpt-4o-mini",
            usage=Usage(input_tokens=10, output_tokens=max(0, tokens - 10)),
            estimated_cost_usd=cost,
            text="pong",
        ),
    )
    c = Cassette(path=path, interactions=[inter])
    c.save()
    return c


def test_server_review_fixes_upstream_prefix_url_redaction_and_control_security(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Upstream with /v1 base path + redact_url_patterns round-trips cleanly on record -> replay
    cass_path = tmp_path / "prefix.tape.json"
    hits: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        hits.append(str(req.url))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
        )

    transport = httpx.MockTransport(handler)
    with TapeReplayServer(
        cass_path,
        mode=TapeMode.ONCE,
        upstream_url="https://api.openai.com/v1",
        upstream_transport=transport,
        redact_url_patterns=[r"/tenants/[^/]+"],
    ) as srv:
        # Control plane works even when query string is attached
        assert httpx.get(f"{srv.url}/__tape__/health?probe=1").status_code == 200

        # Non-cassette filenames rejected by /__tape__/save and /__tape__/load
        assert (
            httpx.post(f"{srv.url}/__tape__/save", json={"cassette": "evil.py"}).status_code == 400
        )
        assert httpx.post(f"{srv.url}/__tape__/load", json={"cassette": ".env"}).status_code == 400

        r1 = httpx.post(
            f"{srv.url}/tenants/acme-99/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r1.status_code == 200
        assert hits == ["https://api.openai.com/v1/tenants/acme-99/chat/completions"]

        # Reset switches ONCE server to replay; path without /v1 matches recorded /v1 path
        assert httpx.post(f"{srv.url}/__tape__/reset").status_code == 200
        r2 = httpx.post(
            f"{srv.url}/tenants/other-tenant/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r2.status_code == 200
        assert len(hits) == 1  # replayed offline

    # 2. Oversized upstream response returns 502 upstream_response_too_large
    monkeypatch.setattr(server_mod, "MAX_CASSETTE_BYTES", 32)
    big_tape = tmp_path / "big.tape.json"
    big_transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b"x" * 64))
    with TapeReplayServer(
        big_tape,
        mode=TapeMode.RECORD,
        upstream_url="https://api.openai.com",
        upstream_transport=big_transport,
    ) as srv:
        r_big = httpx.post(f"{srv.url}/v1/chat/completions", json={"a": 1})
        assert r_big.status_code == 502
        assert r_big.json()["error"] == "upstream_response_too_large"
        assert not big_tape.exists()


def test_server_concurrent_save_preserves_all_recordings(tmp_path: Path) -> None:
    cass_path = tmp_path / "concurrent.tape.json"

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"echo": json.loads(req.content)})

    with TapeReplayServer(
        cass_path,
        mode=TapeMode.RECORD,
        upstream_url="https://api.openai.com",
        upstream_transport=httpx.MockTransport(handler),
    ) as srv:
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                r = httpx.post(f"{srv.url}/v1/chat/completions", json={"id": i})
                assert r.status_code == 200
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    loaded = Cassette.load(cass_path)
    assert len(loaded.interactions) == 6


def test_cli_run_zero_code_replay_budgets_and_module_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cass_path = tmp_path / "agent.tape.json"
    _make_sample_cassette(cass_path, tokens=50, cost=0.02)

    child_script = tmp_path / "app.py"
    child_script.write_text(
        "import json, os, urllib.request\n"
        "assert os.environ['OPENAI_API_KEY'] == 'rhinotape-replay'\n"
        "url = os.environ['OPENAI_BASE_URL'] + '/chat/completions'\n"
        "payload = {'model':'gpt-4o-mini','messages':[{'role':'user','content':'hi'}]}\n"
        "req = urllib.request.Request(\n"
        "    url,\n"
        "    data=json.dumps(payload).encode(),\n"
        "    headers={'Content-Type': 'application/json'},\n"
        ")\n"
        "with urllib.request.urlopen(req) as resp:\n"
        "    data = json.loads(resp.read())\n"
        "assert data['choices'][0]['message']['content'] == 'pong'\n",
        encoding="utf-8",
    )

    # 1. Passing replay run with strict-unused and cost budget
    rc = cli_main(
        [
            "run",
            str(cass_path),
            "--mode",
            "replay",
            "--max-cost",
            "0.05",
            "--max-tokens",
            "100",
            "--strict-unused",
            "--",
            sys.executable,
            str(child_script),
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "calls=1" in err and "tokens=50" in err

    # 2. Exceeding --max-cost fails with exit code 1
    rc_over = cli_main(
        [
            "run",
            str(cass_path),
            "--mode",
            "replay",
            "--max-cost",
            "0.005",
            "--quiet",
            "--",
            sys.executable,
            str(child_script),
        ]
    )
    assert rc_over == 1
    assert "max_cost_usd" in capsys.readouterr().err

    # 3. Child exit code, flag passthrough, missing command (-> 2), invalid budget (-> 1)
    assert cli_main(["run", str(cass_path), "--", sys.executable, "-c", "raise SystemExit(7)"]) == 7
    assert (
        cli_main(
            [
                "run",
                str(cass_path),
                "--quiet",
                "--",
                sys.executable,
                "-c",
                "import sys; assert sys.argv[1:] == ['--mode', 'child', '--help']",
                "--mode",
                "child",
                "--help",
            ]
        )
        == 0
    )
    assert cli_main(["run", str(cass_path)]) == 2
    assert cli_main(["run", str(cass_path), "--"]) == 2
    assert cli_main(["run", str(cass_path), "--max-cost", "nan", "--", sys.executable]) == 1

    # 4. `python -m rhinotape --version` works cleanly
    proc = subprocess.run(
        [sys.executable, "-m", "rhinotape", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "rhinotape" in proc.stdout
