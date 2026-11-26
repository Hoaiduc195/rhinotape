"""Offline runnable showcase of llm-tape framework integrations.

Demonstrates:
1. TapeConfig.from_env() + current_session() / require_session()
2. make_httpx_client() drop-in SDK client factory (patch_httpx=False)
3. Pure ASGI 3.0 TapeASGIMiddleware (FastAPI / Starlette / Litestar / Django ASGI)
4. Pure PEP 3333 TapeWSGIMiddleware (Flask / Django WSGI)
5. stdlib unittest TapeTestCaseMixin
6. GenericHTTPRequest / handle_generic_request() adapter bridge

Run with:
    .venv/bin/python examples/framework_integrations.py
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import httpx

from llm_tape import MemoryCassetteStore, TapeMode
from llm_tape.integrations import (
    GenericHTTPRequest,
    GenericHTTPResponse,
    TapeASGIMiddleware,
    TapeConfig,
    TapeTestCaseMixin,
    TapeWSGIMiddleware,
    current_session,
    handle_generic_request,
    make_httpx_client,
    require_session,
)


def _mock_openai_transport(reply: str = "Hello from offline mock!") -> httpx.MockTransport:
    def _handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-demo",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 25, "completion_tokens": 10},
            },
        )

    return httpx.MockTransport(_handler)


def demo_config_and_httpx_factory(tmp_dir: Path) -> None:
    os.environ["LLM_TAPE_DIR"] = str(tmp_dir / "env_tapes")
    os.environ["LLM_TAPE_MODE"] = "once"
    os.environ["LLM_TAPE_REDACT_PII"] = "email"
    try:
        cfg = TapeConfig.from_env()
    finally:
        for key in ("LLM_TAPE_DIR", "LLM_TAPE_MODE", "LLM_TAPE_REDACT_PII"):
            os.environ.pop(key, None)

    # Record with explicit httpx client factory (no global monkey-patching required)
    with cfg.session("sdk_client_demo", patch_httpx=False) as session:
        assert current_session() is session
        with make_httpx_client(session, upstream=_mock_openai_transport("SDK ok")) as client:
            r = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200
        require_session().assert_call_count(1)

    # Replay strictly offline without upstream
    with cfg.session("sdk_client_demo", mode=TapeMode.REPLAY, patch_httpx=False) as replay_sess:
        with make_httpx_client(
            replay_sess, upstream=httpx.MockTransport(lambda req: httpx.Response(500))
        ) as client:
            r = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.json()["choices"][0]["message"]["content"] == "SDK ok"
    print("[1/5] TapeConfig.from_env + make_httpx_client + current_session OK")


async def demo_asgi_middleware(tmp_dir: Path) -> None:
    upstream = _mock_openai_transport("ASGI reply")

    async def mini_asgi_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        del receive
        session = scope["state"]["llm_tape"]
        assert current_session() is session
        with httpx.Client(transport=upstream) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "asgi"}]},
            )
        body = resp.json()["choices"][0]["message"]["content"].encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    cfg = TapeConfig(tape_dir=tmp_dir / "asgi_tapes", mode=TapeMode.ONCE)
    wrapped = TapeASGIMiddleware(mini_asgi_app, config=cfg, expose_headers=True)

    sent: list[dict[str, Any]] = []

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(msg: dict[str, Any]) -> None:
        sent.append(msg)

    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat",
        "headers": [],
    }
    await wrapped(scope, _receive, _send)
    hdrs = dict(sent[0]["headers"])
    assert hdrs[b"x-tape-calls"] == b"1"
    assert sent[1]["body"] == b"ASGI reply"
    print("[2/5] TapeASGIMiddleware (record + X-Tape-* headers) OK")


def demo_wsgi_middleware(tmp_dir: Path) -> None:
    upstream = _mock_openai_transport("WSGI reply")

    def mini_wsgi_app(environ: dict[str, Any], start_response: Any) -> list[bytes]:
        assert environ["llm_tape.session"] is require_session()
        with httpx.Client(transport=upstream) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "wsgi"}]},
            )
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [resp.json()["choices"][0]["message"]["content"].encode("utf-8")]

    cfg = TapeConfig(tape_dir=tmp_dir / "wsgi_tapes", mode=TapeMode.ONCE)
    wrapped = TapeWSGIMiddleware(mini_wsgi_app, config=cfg, expose_headers=True)

    captured_headers: list[tuple[str, str]] = []

    def _start_response(status: str, headers: list[tuple[str, str]]) -> None:
        del status
        captured_headers.extend(headers)

    body = b"".join(wrapped({"REQUEST_METHOD": "POST", "PATH_INFO": "/ask"}, _start_response))
    assert body == b"WSGI reply"
    assert dict(captured_headers)["X-Tape-Calls"] == "1"
    print("[3/5] TapeWSGIMiddleware OK")


def demo_unittest_mixin(tmp_dir: Path) -> None:
    tapes_dir = tmp_dir / "unittest_tapes"

    class DemoAgentTests(TapeTestCaseMixin, unittest.TestCase):
        tape_dir = tapes_dir
        tape_mode = TapeMode.ONCE

        def test_reply(self) -> None:
            with httpx.Client(transport=_mock_openai_transport("unittest ok")) as client:
                r = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "u"}]},
                )
                self.assertEqual(r.json()["choices"][0]["message"]["content"], "unittest ok")
            self.llm_tape.assert_call_count(1)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DemoAgentTests)
    result = unittest.TestResult()
    suite.run(result)
    assert result.wasSuccessful()
    assert (tapes_dir / "DemoAgentTests__test_reply.tape.json").is_file()
    print("[4/5] TapeTestCaseMixin (stdlib unittest) OK")


def demo_generic_bridge() -> None:
    store = MemoryCassetteStore()
    req = GenericHTTPRequest(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": "Bearer sk-demo-secret-1234567890",
            "Content-Type": "application/json",
        },
        body=json.dumps(
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
        ).encode(),
    )

    def fake_custom_http_send(r: GenericHTTPRequest) -> GenericHTTPResponse:
        del r
        payload = {
            "model": "gpt-4o-mini",
            "choices": [{"message": {"role": "assistant", "content": "generic ok"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
        return GenericHTTPResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode("utf-8"),
        )

    cfg = TapeConfig(store=store, patch_httpx=False, patch_requests=False, patch_aiohttp=False)
    with cfg.session("generic_demo", mode=TapeMode.RECORD) as rec:
        resp = handle_generic_request(rec, req, fake_custom_http_send)
        assert resp.status_code == 200
        rec.assert_call_count(1)

    with cfg.session("generic_demo", mode=TapeMode.REPLAY) as rep:
        replayed = handle_generic_request(
            rep,
            req,
            lambda _: (_ for _ in ()).throw(AssertionError("should not call network")),
        )
        assert json.loads(replayed.body)["choices"][0]["message"]["content"] == "generic ok"
        rep.assert_tokens_under(50)
    print("[5/5] GenericHTTPRequest / handle_generic_request bridge OK")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        demo_config_and_httpx_factory(tmp_dir)
        asyncio.run(demo_asgi_middleware(tmp_dir))
        demo_wsgi_middleware(tmp_dir)
        demo_unittest_mixin(tmp_dir)
        demo_generic_bridge()
    print("All llm-tape framework integration demos passed offline!")


if __name__ == "__main__":
    main()
