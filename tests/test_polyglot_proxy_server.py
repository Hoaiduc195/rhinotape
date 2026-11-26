"""Tests for polyglot reverse-proxy record/replay server, control plane, and CLI serve flags."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
import requests

from rhinotape import Cassette, ModelPrice, TapeMode, TapeReplayServer
from rhinotape.cli import build_parser
from rhinotape.cli import main as cli_main


def _openai_chat_payload(
    content: str,
    *,
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 20,
    completion_tokens: int = 10,
    tool_name: str | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_name is not None:
        msg["tool_calls"] = [
            {
                "id": f"call_{tool_name}",
                "type": "function",
                "function": {"name": tool_name, "arguments": '{"q":"test"}'},
            }
        ]
    return {
        "id": "chatcmpl-proxy",
        "model": model,
        "choices": [{"message": msg}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def test_reverse_proxy_once_mode_record_reset_and_offline_replay(tmp_path: Path) -> None:
    cassette_path = tmp_path / "cassettes" / "once_flow.tape.json"
    assert not cassette_path.exists()

    # REPLAY mode still raises FileNotFoundError when source does not exist
    with pytest.raises(FileNotFoundError):
        TapeReplayServer(cassette_path, mode="replay")

    upstream_calls: list[httpx.Request] = []

    def _upstream_handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request)
        assert request.headers.get("authorization") == "Bearer sk-live-secret-token-999"
        body = json.loads(request.content.decode("utf-8"))
        prompt = body["messages"][0]["content"]
        return httpx.Response(200, json=_openai_chat_payload(f"reply:{prompt}"))

    transport = httpx.MockTransport(_upstream_handler)

    # 1. ONCE mode on non-existent file records from upstream and saves cassette
    with TapeReplayServer(
        cassette_path,
        mode=TapeMode.ONCE,
        upstream_url="https://api.openai.com",
        upstream_transport=transport,
    ) as srv:
        with httpx.Client() as client:
            r1 = client.post(
                f"{srv.url}/v1/chat/completions",
                headers={"Authorization": "Bearer sk-live-secret-token-999"},
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
            )
            assert r1.status_code == 200
            assert r1.json()["choices"][0]["message"]["content"] == "reply:hello"
            assert len(upstream_calls) == 1

            st1 = client.get(f"{srv.url}/__tape__/stats").json()
            assert st1["recorded_interactions"] == 1
            assert st1["total_tokens"] == 30

            # POST /__tape__/reset transitions ONCE mode to offline replay from saved cassette
            r_reset = client.post(f"{srv.url}/__tape__/reset")
            assert r_reset.status_code == 200
            assert r_reset.json() == {"status": "reset", "loaded_interactions": 1}

            r2 = client.post(
                f"{srv.url}/v1/chat/completions",
                headers={"Authorization": "Bearer sk-different-key"},
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
            )
            assert r2.status_code == 200
            assert r2.json()["choices"][0]["message"]["content"] == "reply:hello"
            # Zero new upstream calls!
            assert len(upstream_calls) == 1

    assert cassette_path.is_file()
    saved = Cassette.load(cassette_path)
    assert len(saved.interactions) == 1
    assert saved.interactions[0].request.headers["Authorization"] == "[REDACTED]"

    # 2. Second server instance in ONCE mode replays completely offline without upstream_url
    with TapeReplayServer(cassette_path, mode="once") as srv2:
        r3 = httpx.post(
            f"{srv2.url}/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert r3.status_code == 200
        assert r3.json()["choices"][0]["message"]["content"] == "reply:hello"
        assert len(upstream_calls) == 1

    # 3. Recording miss without upstream_url returns 502 {"error": "upstream_not_configured"}
    with TapeReplayServer(tmp_path / "no_up.tape.json", mode="record") as srv_no_up:
        r502 = httpx.post(f"{srv_no_up.url}/v1/chat/completions", json={"x": 1})
        assert r502.status_code == 502
        assert r502.json() == {"error": "upstream_not_configured"}


def test_reverse_proxy_record_new_episodes_sse_streaming_and_scrubbing(tmp_path: Path) -> None:
    cassette_path = tmp_path / "stream_and_pii.tape.json"
    upstream_hits = 0

    class _SSEByteStream(httpx.SyncByteStream):
        def __iter__(self) -> Any:
            yield (
                b'data: {"id":"1","model":"custom-m","choices":'
                b'[{"delta":{"content":"Email alice@example.com "}}]}\n\n'
            )
            yield (
                b'data: {"id":"1","model":"custom-m","choices":'
                b'[{"delta":{"content":"code SECRET-42"}}],'
                b'"usage":{"prompt_tokens":100,"completion_tokens":50}}\n\n'
            )
            yield b"data: [DONE]\n\n"

    def _upstream(req: httpx.Request) -> httpx.Response:
        nonlocal upstream_hits
        upstream_hits += 1
        if req.url.path.endswith("/sse"):
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream", "X-Upstream-Secret": "topsecret"},
                stream=_SSEByteStream(),
            )
        body = json.loads(req.content.decode("utf-8"))
        step = body.get("step", 1)
        return httpx.Response(
            200,
            json=_openai_chat_payload(
                f"step-{step} for bob@corp.org",
                model="custom-m",
                tool_name=f"tool_{step}",
            ),
        )

    transport = httpx.MockTransport(_upstream)
    pricing = {"custom-m": ModelPrice(input_per_million=10.0, output_per_million=20.0)}

    # Phase 1: RECORD SSE stream + JSON call with PII, field, pattern, header, and URL scrubbing
    with TapeReplayServer(
        cassette_path,
        mode="record",
        upstream_url="https://api.openai.com",
        upstream_transport=transport,
        extra_sensitive_headers=["x-upstream-secret"],
        extra_sensitive_query_params=["custom_key"],
        redact_fields=[" internal_token ".strip()],
        redact_patterns=[r"SECRET-\d+"],
        redact_pii=True,
        redact_url_patterns=[r"/orgs/org-[0-9]+"],
        ignore_fields=["internal_token"],
        custom_pricing=pricing,
    ) as srv:
        with requests.post(
            f"{srv.url}/orgs/org-9876/v1/chat/sse?custom_key=my_q_secret",
            json={"model": "custom-m", "stream": True, "internal_token": "tok-xyz"},
            stream=True,
            timeout=5,
        ) as sse_resp:
            assert sse_resp.status_code == 200
            lines = [ln.decode("utf-8") for ln in sse_resp.iter_lines() if ln]
            assert len(lines) == 3

        r_json = httpx.post(
            f"{srv.url}/v1/chat/completions",
            json={"model": "custom-m", "step": 1, "internal_token": "tok-111"},
        )
        assert r_json.status_code == 200

    assert upstream_hits == 2
    saved = Cassette.load(cassette_path)
    assert len(saved.interactions) == 2

    sse_inter, json_inter = saved.interactions
    assert "org-9876" not in sse_inter.request.url
    assert "my_q_secret" not in sse_inter.request.url
    assert sse_inter.request.body["internal_token"] == "[REDACTED]"
    assert sse_inter.response.headers["x-upstream-secret"] == "[REDACTED]"
    assert "alice@example.com" not in "".join(sse_inter.response.stream_chunks or ())
    assert "SECRET-42" not in "".join(sse_inter.response.stream_chunks or ())
    assert "[REDACTED]" in sse_inter.llm.text
    assert sse_inter.llm.estimated_cost_usd == pytest.approx(0.002)
    assert "bob@corp.org" not in json_inter.llm.text

    # Phase 2: NEW_EPISODES replays step=1 offline and records new step=2 from upstream
    with TapeReplayServer(
        cassette_path,
        mode="new_episodes",
        upstream_url="https://api.openai.com",
        upstream_transport=transport,
        ignore_fields=["internal_token"],
        redact_pii=True,
        custom_pricing=pricing,
    ) as srv_ne:
        # Existing step=1 -> served from cassette (upstream_hits stays 2)
        r_hit = httpx.post(
            f"{srv_ne.url}/v1/chat/completions",
            json={"model": "custom-m", "step": 1, "internal_token": "different"},
        )
        assert r_hit.status_code == 200
        assert upstream_hits == 2

        # Unseen step=2 -> forwarded to upstream and appended (upstream_hits becomes 3)
        r_miss = httpx.post(
            f"{srv_ne.url}/v1/chat/completions",
            json={"model": "custom-m", "step": 2},
        )
        assert r_miss.status_code == 200
        assert upstream_hits == 3

    updated = Cassette.load(cassette_path)
    assert len(updated.interactions) == 3


def test_cors_headers_and_options_preflight(tmp_path: Path) -> None:
    cassette_path = tmp_path / "cors.tape.json"
    Cassette(path=cassette_path, interactions=[]).save()

    with TapeReplayServer(cassette_path, enable_cors=True) as srv:
        with httpx.Client() as client:
            # OPTIONS preflight succeeds with 204 without needing a cassette match
            opt = client.options(
                f"{srv.url}/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "authorization,content-type",
                },
            )
            assert opt.status_code == 204
            assert opt.headers["access-control-allow-origin"] == "*"
            assert "POST" in opt.headers["access-control-allow-methods"]
            assert "OPTIONS" in opt.headers["access-control-allow-methods"]
            assert opt.headers["access-control-allow-headers"] == "*"

            # Control-plane and 404 responses also carry CORS headers; OPTIONS skips unmatched
            st_resp = client.get(f"{srv.url}/__tape__/stats")
            assert st_resp.headers["access-control-allow-origin"] == "*"
            assert st_resp.json()["unmatched_requests"] == 0

            r404 = client.post(f"{srv.url}/v1/missing", json={"a": 1})
            assert r404.status_code == 404
            assert r404.headers["access-control-allow-origin"] == "*"


def test_control_plane_endpoints_and_path_traversal_guard(tmp_path: Path) -> None:
    root_dir = tmp_path / "tapes"
    root_dir.mkdir()
    outside_secret = tmp_path / "outside.tape.json"
    Cassette(path=outside_secret, interactions=[]).save()

    tape_a = root_dir / "suite" / "a.tape.json"
    tape_a.parent.mkdir(parents=True)

    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json=_openai_chat_payload(
                "used calculator",
                prompt_tokens=40,
                completion_tokens=10,
                tool_name="calc",
            ),
        )
    )

    with TapeReplayServer(
        tape_a,
        mode="record",
        cassette_dir=root_dir,
        upstream_url="https://api.openai.com",
        upstream_transport=transport,
    ) as srv:
        with httpx.Client() as client:
            # 1. /__tape__/health
            h = client.get(f"{srv.url}/__tape__/health").json()
            assert h["status"] == "ok"
            assert h["mode"] == "record"
            assert h["loaded_interactions"] == 0
            assert h["active_cassette"] == str(tape_a.resolve())

            # Record one interaction
            r = client.post(
                f"{srv.url}/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "2+2"}]},
            )
            assert r.status_code == 200

            # 2. /__tape__/stats
            st = client.get(f"{srv.url}/__tape__/stats").json()
            assert st["recorded_interactions"] == 1
            assert st["total_tokens"] == 50
            assert st["estimated_cost_usd"] > 0.0
            assert st["tool_calls"] == ["calc"]

            # 3. /__tape__/interactions
            inters = client.get(f"{srv.url}/__tape__/interactions").json()["interactions"]
            assert len(inters) == 1
            item = inters[0]
            assert item["method"] == "POST"
            assert item["status_code"] == 200
            assert item["provider"] == "openai"
            assert item["model"] == "gpt-4o-mini"
            assert item["usage"]["total_tokens"] == 50
            assert item["tool_calls"][0]["name"] == "calc"
            assert item["text"] == "used calculator"
            assert "thinking" in item
            assert item["is_stream"] is False

            # 4. /__tape__/save to explicit relative path inside cassette_dir
            save_resp = client.post(
                f"{srv.url}/__tape__/save",
                json={"cassette": "saved/copy.tape.json"},
            )
            assert save_resp.status_code == 200
            assert save_resp.json()["status"] == "saved"
            assert save_resp.json()["interactions"] == 1
            assert (root_dir / "saved" / "copy.tape.json").is_file()

            # /__tape__/save rejects path traversal
            bad_save = client.post(
                f"{srv.url}/__tape__/save",
                json={"cassette": "../escaped.tape.json"},
            )
            assert bad_save.status_code == 400
            assert bad_save.json()["error"] == "invalid_cassette_path"

            # 5. /__tape__/load strictly blocks path traversal & absolute paths with 400
            for attack in (
                "../outside.tape.json",
                "suite/../../outside.tape.json",
                str(outside_secret.resolve()),
                "/etc/passwd",
                "",
            ):
                bad_load = client.post(
                    f"{srv.url}/__tape__/load",
                    json={"cassette": attack},
                )
                assert bad_load.status_code == 400
                assert bad_load.json() == {"error": "invalid_cassette_path"}

            # /__tape__/load returns 404 when cassette missing in replay mode
            missing_load = client.post(
                f"{srv.url}/__tape__/load",
                json={"cassette": "does_not_exist.tape.json", "mode": "replay"},
            )
            assert missing_load.status_code == 404

            # /__tape__/load switches active cassette cleanly to saved/copy.tape.json in replay mode
            ok_load = client.post(
                f"{srv.url}/__tape__/load",
                json={"cassette": "saved/copy.tape.json", "mode": "replay"},
            )
            assert ok_load.status_code == 200
            load_body = ok_load.json()
            assert load_body["status"] == "loaded"
            assert load_body["mode"] == "replay"
            assert load_body["loaded_interactions"] == 1

            # Replay works from newly loaded cassette
            r_rep = client.post(
                f"{srv.url}/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "2+2"}]},
            )
            assert r_rep.status_code == 200
            assert r_rep.json()["choices"][0]["message"]["content"] == "used calculator"


def test_declarative_assert_endpoint_pass_fail_and_validation(tmp_path: Path) -> None:
    cassette_path = tmp_path / "assert_suite.tape.json"

    def _upstream(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode("utf-8"))
        idx = body["idx"]
        if idx == 1:
            return httpx.Response(
                200,
                json=_openai_chat_payload(
                    "First answer: 42",
                    model="gpt-4o",
                    prompt_tokens=100,
                    completion_tokens=50,
                    tool_name="search_docs",
                ),
            )
        return httpx.Response(
            200,
            json=_openai_chat_payload(
                "Final summary ready",
                model="gpt-4o",
                prompt_tokens=200,
                completion_tokens=100,
                tool_name="write_report",
            ),
        )

    # Record 2 interactions first
    with TapeReplayServer(
        cassette_path,
        mode="record",
        upstream_url="https://api.openai.com",
        upstream_transport=httpx.MockTransport(_upstream),
    ) as rec_srv:
        httpx.post(f"{rec_srv.url}/v1/chat/completions", json={"idx": 1})
        httpx.post(f"{rec_srv.url}/v1/chat/completions", json={"idx": 2})

    # Replay only 1 of the 2 first to test no_unused_interactions failure, then replay 2nd
    with TapeReplayServer(cassette_path, mode="replay", allow_playback_repeats=False) as srv:
        with httpx.Client() as client:
            client.post(f"{srv.url}/v1/chat/completions", json={"idx": 1})

            # Unused interaction check fails when idx=2 has not been consumed yet
            r_unused = client.post(
                f"{srv.url}/__tape__/assert",
                json={"no_unused_interactions": True},
            )
            assert r_unused.status_code == 400
            assert r_unused.json()["ok"] is False
            assert any("no_unused_interactions" in msg for msg in r_unused.json()["failures"])

            # Consume 2nd interaction
            client.post(f"{srv.url}/v1/chat/completions", json={"idx": 2})

            # All declarative assertions PASS -> 200 {"ok": True, "failures": []}
            r_pass = client.post(
                f"{srv.url}/__tape__/assert",
                json={
                    "call_count": 2,
                    "max_tokens": 500,
                    "max_cost_usd": 0.05,
                    "max_latency_ms": 10000.0,
                    "no_errors": True,
                    "tool_called": ["search_docs", "write_report"],
                    "no_tool_called": "drop_database",
                    "tool_sequence": ["search_docs", "write_report"],
                    "text_contains": "Final summary",
                    "no_unused_interactions": True,
                },
            )
            assert r_pass.status_code == 200
            assert r_pass.json() == {"ok": True, "failures": []}

            # Failing assertions -> 400 {"ok": False, "failures": [...]}
            r_fail = client.post(
                f"{srv.url}/__tape__/assert",
                json={
                    "call_count": 1,
                    "max_tokens": 10,
                    "max_cost_usd": 0.000001,
                    "tool_called": "missing_tool",
                    "no_tool_called": ["search_docs"],
                    "tool_sequence": ["write_report"],
                    "text_contains": "nonexistent phrase",
                },
            )
            assert r_fail.status_code == 400
            fail_body = r_fail.json()
            assert fail_body["ok"] is False
            assert len(fail_body["failures"]) == 7

            # Invalid JSON, NaN, negative thresholds, and unknown keys -> 400
            r_bad_json = client.post(
                f"{srv.url}/__tape__/assert",
                content=b"{not-json",
                headers={"Content-Type": "application/json"},
            )
            assert r_bad_json.status_code == 400

            r_nan = client.post(
                f"{srv.url}/__tape__/assert",
                content=b'{"max_cost_usd": NaN}',
                headers={"Content-Type": "application/json"},
            )
            assert r_nan.status_code == 400

            r_neg = client.post(
                f"{srv.url}/__tape__/assert",
                json={"max_tokens": -5},
            )
            assert r_neg.status_code == 400

            r_unknown = client.post(
                f"{srv.url}/__tape__/assert",
                json={"made_up_assertion": True},
            )
            assert r_unknown.status_code == 400


def test_cli_serve_flags_parsing_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = build_parser()

    # Defaults
    args_default = parser.parse_args(["serve", "tapes/demo.tape.json"])
    assert args_default.mode == "replay"
    assert args_default.upstream is None
    assert args_default.cors is False
    assert args_default.max_request_bytes == 10 * 1024 * 1024
    assert args_default.redact_pii is None

    # Explicit flags + bare --redact-pii
    args_full = parser.parse_args(
        [
            "serve",
            "tapes/new.tape.json",
            "--mode",
            "once",
            "--upstream",
            "https://api.openai.com",
            "--cors",
            "--max-request-bytes",
            "2097152",
            "--redact-pii",
        ]
    )
    assert args_full.mode == "once"
    assert args_full.upstream == "https://api.openai.com"
    assert args_full.cors is True
    assert args_full.max_request_bytes == 2097152
    assert args_full.redact_pii == "all"

    # Non-existent source fails in replay mode, but succeeds in once/record/new_episodes!
    missing = tmp_path / "brand_new.tape.json"
    assert cli_main(["serve", str(missing), "--mode", "replay"]) == 1

    # Invalid max-request-bytes returns exit code 1 cleanly
    assert cli_main(["serve", str(missing), "--mode", "record", "--max-request-bytes", "-1"]) == 1

    # Verify _cmd_serve starts TapeReplayServer with non-existent source in once mode
    orig_wait = threading.Event.wait
    monkeypatch.setattr(
        threading.Event,
        "wait",
        lambda self, timeout=None: True if timeout == 0.5 else orig_wait(self, timeout),
    )
    try:
        rc = cli_main(
            [
                "serve",
                str(missing),
                "--port",
                "0",
                "--mode",
                "once",
                "--upstream",
                "https://api.openai.com",
                "--cors",
                "--redact-pii",
                "email,phone",
            ]
        )
        assert rc == 0
    finally:
        monkeypatch.setattr(threading.Event, "wait", orig_wait)
