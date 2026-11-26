"""Comprehensive tests for zero-dependency framework integrations (src/llm_tape/integrations.py)."""

from __future__ import annotations

import json
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_tape import (
    Cassette,
    MemoryCassetteStore,
    ModelPrice,
    TapeAssertionError,
    TapeError,
    TapeMode,
    TapeSession,
)
from llm_tape.integrations import (
    GenericHTTPRequest,
    GenericHTTPResponse,
    TapeASGIMiddleware,
    TapeConfig,
    TapeTestCaseMixin,
    TapeWSGIMiddleware,
    activate_from_env,
    activate_tape,
    current_session,
    handle_generic_request,
    handle_generic_request_async,
    make_async_httpx_client,
    make_httpx_client,
    require_session,
    tape_from_env,
)


def _mock_openai_response(
    content: str = "hello", *, prompt_tokens: int = 20, completion_tokens: int = 10
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-int",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        },
    )


# --- 1. Context Accessors --------------------------------------------------


def test_current_and_require_session(tmp_path: Path) -> None:
    assert current_session() is None
    with pytest.raises(TapeError, match="No active llm-tape session"):
        require_session()

    cpath = tmp_path / "ctx.tape.json"
    with TapeSession(cpath, mode=TapeMode.RECORD) as sess:
        assert current_session() is sess
        assert require_session() is sess

    assert current_session() is None


# --- 2. TapeConfig & Env Bootstrap -----------------------------------------


def test_tape_config_from_env_validation_resolve_session_and_decorator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_TAPE_MODE", "new_episodes")
    monkeypatch.setenv("LLM_TAPE_DIR", str(tmp_path / "env_tapes"))
    monkeypatch.setenv("LLM_TAPE_COMPRESS", "true")
    monkeypatch.setenv("LLM_TAPE_IGNORE_LOCALHOST", "yes")
    monkeypatch.setenv("LLM_TAPE_IGNORE_HOSTS", "internal.local, *.metrics.local")
    monkeypatch.setenv("LLM_TAPE_IGNORE_FIELDS", "request_id, meta.ts")
    monkeypatch.setenv("LLM_TAPE_IGNORE_QUERY_PARAMS", "nonce, sig")
    monkeypatch.setenv("LLM_TAPE_REDACT_PII", "email, ssn")
    monkeypatch.setenv("LLM_TAPE_MAX_COST", "0.5")
    monkeypatch.setenv("LLM_TAPE_MAX_INTERACTIONS", "4")
    monkeypatch.setenv("LLM_TAPE_ALLOW_PLAYBACK_REPEATS", "1")
    monkeypatch.setenv("LLM_TAPE_PATCH_URLLIB", "on")

    cfg = TapeConfig.from_env()
    assert cfg.mode is TapeMode.NEW_EPISODES
    assert cfg.tape_dir == tmp_path / "env_tapes"
    assert cfg.compress is True
    assert cfg.ignore_localhost is True
    assert cfg.ignore_hosts == ("internal.local", "*.metrics.local")
    assert cfg.ignore_fields == ("request_id", "meta.ts")
    assert cfg.ignore_query_params == ("nonce", "sig")
    assert cfg.redact_pii == ("email", "ssn")
    assert cfg.max_cost_usd == 0.5
    assert cfg.max_interactions == 4
    assert cfg.allow_playback_repeats is True
    assert cfg.patch_urllib is True

    # resolve_path adds extension and blocks traversal safely
    normal = cfg.resolve_path("my_flow")
    assert normal == tmp_path / "env_tapes" / "my_flow.tape.json.gz"
    explicit = cfg.resolve_path("keep.tape.json")
    assert explicit == tmp_path / "env_tapes" / "keep.tape.json"
    trav = cfg.resolve_path("../../etc/passwd")
    assert trav.resolve().is_relative_to((tmp_path / "env_tapes").resolve())
    assert ".." not in trav.parts
    assert trav.suffix == ".gz"

    # tape_from_env defaults to default.tape.json(.gz) or uses LLM_TAPE_CASSETTE
    default_sess = tape_from_env()
    assert default_sess.cassette_path == tmp_path / "env_tapes" / "default.tape.json.gz"
    monkeypatch.setenv("LLM_TAPE_CASSETTE", "custom_case")
    named_sess = tape_from_env()
    assert named_sess.cassette_path == tmp_path / "env_tapes" / "custom_case.tape.json.gz"

    # Sync decorator
    sync_cfg = TapeConfig(tape_dir=tmp_path / "dec_tapes", mode=TapeMode.RECORD)

    @sync_cfg.decorator("sync_fn")
    def call_sync() -> str:
        with httpx.Client(
            transport=httpx.MockTransport(lambda r: _mock_openai_response("sync"))
        ) as c:
            return str(
                c.post("https://api.openai.com/v1/chat/completions", json={"x": 1}).json()[
                    "choices"
                ][0]["message"]["content"]
            )

    assert call_sync() == "sync"
    assert (tmp_path / "dec_tapes" / "sync_fn.tape.json").is_file()


@pytest.mark.asyncio
async def test_tape_config_async_decorator_and_env_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = TapeConfig(tape_dir=tmp_path / "async_dec", mode=TapeMode.RECORD)

    @cfg.decorator("async_fn")
    async def call_async() -> str:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: _mock_openai_response("async-ok"))
        ) as c:
            resp = await c.post("https://api.openai.com/v1/chat/completions", json={"y": 2})
            return str(resp.json()["choices"][0]["message"]["content"])

    assert await call_async() == "async-ok"
    assert (tmp_path / "async_dec" / "async_fn.tape.json").is_file()

    # Invalid env values raise ValueError cleanly
    invalid_env_cases: list[tuple[str, str]] = [
        ("TEST_MODE", "not_a_mode"),
        ("TEST_COMPRESS", "maybe"),
        ("TEST_MAX_COST", "nan"),
        ("TEST_MAX_COST", "-1.0"),
        ("TEST_MAX_INTERACTIONS", "-2"),
        ("TEST_MAX_INTERACTIONS", "abc"),
        ("TEST_REDACT_PII", "unknown_preset"),
    ]
    for env_key, env_val in invalid_env_cases:
        monkeypatch.setenv(env_key, env_val)
        with pytest.raises(ValueError):
            TapeConfig.from_env(prefix="TEST_")
        monkeypatch.delenv(env_key)

    # Direct TapeConfig __post_init__ validation
    with pytest.raises(ValueError):
        TapeConfig(max_interactions=-1)
    with pytest.raises(ValueError):
        TapeConfig(max_cost_usd=float("nan"))
    with pytest.raises(ValueError):
        TapeConfig(replay_latency_scale=-0.5)


# --- 3. Drop-in SDK / httpx Client Factories -------------------------------


@pytest.mark.asyncio
async def test_make_httpx_client_sync_and_async_without_global_patching(tmp_path: Path) -> None:
    store = MemoryCassetteStore()
    cfg = TapeConfig(
        store=store,
        patch_httpx=False,
        patch_requests=False,
        patch_aiohttp=False,
    )

    # Sync record + offline replay
    with cfg.session("sync_factory", mode=TapeMode.RECORD) as rec:
        with make_httpx_client(
            rec, upstream=httpx.MockTransport(lambda r: _mock_openai_response("sync-factory"))
        ) as client:
            r = client.post("https://api.openai.com/v1/chat/completions", json={"k": "s"})
            assert r.json()["choices"][0]["message"]["content"] == "sync-factory"

    with cfg.session("sync_factory", mode=TapeMode.REPLAY) as rep:
        with make_httpx_client(
            rep, upstream=httpx.MockTransport(lambda r: httpx.Response(500))
        ) as client:
            r = client.post("https://api.openai.com/v1/chat/completions", json={"k": "s"})
            assert r.json()["choices"][0]["message"]["content"] == "sync-factory"
        rep.assert_call_count(1)

    # Async record + offline replay
    async with cfg.session("async_factory", mode=TapeMode.RECORD) as arec:
        async with make_async_httpx_client(
            arec, upstream=httpx.MockTransport(lambda r: _mock_openai_response("async-factory"))
        ) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", json={"k": "a"})
            assert r.json()["choices"][0]["message"]["content"] == "async-factory"

    async with cfg.session("async_factory", mode=TapeMode.REPLAY) as arep:
        async with make_async_httpx_client(
            arep, upstream=httpx.MockTransport(lambda r: httpx.Response(500))
        ) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", json={"k": "a"})
            assert r.json()["choices"][0]["message"]["content"] == "async-factory"
        arep.assert_call_count(1)

    # Reject passing transport= directly
    with cfg.session("err_factory", mode=TapeMode.NONE) as sess:
        with pytest.raises(TypeError, match="upstream"):
            make_httpx_client(sess, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        with pytest.raises(TypeError, match="upstream"):
            make_async_httpx_client(
                sess, transport=httpx.MockTransport(lambda r: httpx.Response(200))
            )


# --- 4. Pure ASGI 3.0 Middleware -------------------------------------------


@pytest.mark.asyncio
async def test_asgi_middleware_streaming_headers_skip_traversal_and_lifespan(
    tmp_path: Path,
) -> None:
    tapes_dir = tmp_path / "asgi_tapes"
    observed_calls: list[str] = []

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        del receive
        if scope["type"] != "http":
            observed_calls.append(scope["type"])
            await send({"type": "lifespan.startup.complete"})
            return
        if scope["path"] == "/health":
            observed_calls.append("health")
            assert current_session() is None
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
            return

        session = scope["state"]["llm_tape"]
        assert current_session() is session
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: _mock_openai_response("chunk1"))
        ) as client:
            r1 = await client.post("https://api.openai.com/v1/chat/completions", json={"step": 1})
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-app", b"1")]})
        await send(
            {
                "type": "http.response.body",
                "body": r1.json()["choices"][0]["message"]["content"].encode("utf-8"),
                "more_body": True,
            }
        )
        # Second LLM call during streaming body proves session remains active until stream ends
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: _mock_openai_response("chunk2"))
        ) as client:
            r2 = await client.post("https://api.openai.com/v1/chat/completions", json={"step": 2})
        await send(
            {
                "type": "http.response.body",
                "body": r2.json()["choices"][0]["message"]["content"].encode("utf-8"),
                "more_body": False,
            }
        )

    cfg = TapeConfig(tape_dir=tapes_dir, mode=TapeMode.RECORD)
    middleware = TapeASGIMiddleware(app, config=cfg, expose_headers=True, skip_paths=("/health",))

    async def _run_scope(scope: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        async def _recv() -> dict[str, Any]:
            return {"type": "http.request", "body": b""}

        async def _send(msg: dict[str, Any]) -> None:
            events.append(msg)

        await middleware(scope, _recv, _send)
        return events

    # Non-HTTP scope (lifespan) passes straight through
    life_events = await _run_scope({"type": "lifespan"})
    assert life_events == [{"type": "lifespan.startup.complete"}]
    assert "lifespan" in observed_calls

    # Skipped path passes through without TapeSession
    health_events = await _run_scope(
        {"type": "http", "method": "GET", "path": "/health", "headers": []}
    )
    assert health_events[1]["body"] == b"ok"

    # Streaming HTTP request with default method+path cassette naming & expose_headers
    http_events = await _run_scope(
        {"type": "http", "method": "POST", "path": "/v1/chat", "headers": []}
    )
    start_headers = dict(http_events[0]["headers"])
    assert start_headers[b"x-tape-calls"] == b"1"
    assert start_headers[b"x-tape-tokens"] == b"30"
    assert b"x-tape-cost-usd" in start_headers
    assert http_events[1]["body"] + http_events[2]["body"] == b"chunk1chunk2"
    recorded_default = Cassette.load(tapes_dir / "POST__v1_chat.tape.json")
    assert len(recorded_default.interactions) == 2

    # Custom header name selects cassette safely
    await _run_scope(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat",
            "headers": [(b"X-Tape-Name", b"custom_flow")],
        }
    )
    assert (tapes_dir / "custom_flow.tape.json").is_file()

    # Path traversal header is rejected and falls back cleanly inside tape_dir
    await _run_scope(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/safe",
            "headers": [(b"x-tape-name", b"../../etc/passwd")],
        }
    )
    assert not (tmp_path / "etc").exists()
    assert (tapes_dir / "POST__v1_safe.tape.json").is_file()

    # Custom cassette_resolver returning None passes through without creating a tape
    none_mw = TapeASGIMiddleware(
        app,
        config=cfg,
        cassette_resolver=lambda scope: None,
    )
    none_events: list[dict[str, Any]] = []

    async def _recv2() -> dict[str, Any]:
        return {"type": "http.request", "body": b""}

    async def _send2(msg: dict[str, Any]) -> None:
        none_events.append(msg)

    await none_mw(
        {"type": "http", "method": "GET", "path": "/health", "headers": []},
        _recv2,
        _send2,
    )
    assert none_events[1]["body"] == b"ok"

    # expose_headers=False + pre-populated scope["state"] + custom resolver returning Path
    custom_path = tmp_path / "explicit_asgi.tape.json"
    plain_mw = TapeASGIMiddleware(
        app,
        config=cfg,
        cassette_resolver=lambda scope: custom_path,
        expose_headers=False,
    )
    plain_events: list[dict[str, Any]] = []

    async def _send3(msg: dict[str, Any]) -> None:
        plain_events.append(msg)

    existing_state: dict[str, Any] = {"request_id": "req-1"}
    await plain_mw(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat",
            "headers": [],
            "state": existing_state,
        },
        _recv2,
        _send3,
    )
    assert "llm_tape" in existing_state
    assert b"x-tape-calls" not in dict(plain_events[0]["headers"])
    assert custom_path.is_file()


# --- 5. Pure PEP 3333 WSGI Middleware --------------------------------------


def test_wsgi_middleware_streaming_iterator_close_and_error_lifecycle(tmp_path: Path) -> None:
    tapes_dir = tmp_path / "wsgi_tapes"
    inner_closed = {"count": 0}

    class _StreamingBody:
        def __init__(self, fail_midstream: bool = False) -> None:
            self._fail = fail_midstream

        def __iter__(self) -> Iterator[bytes]:
            assert current_session() is not None
            with httpx.Client(
                transport=httpx.MockTransport(lambda r: _mock_openai_response("stream-part"))
            ) as client:
                r = client.post("https://api.openai.com/v1/chat/completions", json={"m": 1})
            yield r.json()["choices"][0]["message"]["content"].encode("utf-8")
            if self._fail:
                raise RuntimeError("boom mid-stream")
            yield b"-end"

        def close(self) -> None:
            inner_closed["count"] += 1

    def wsgi_app(environ: dict[str, Any], start_response: Any) -> Any:
        if environ["PATH_INFO"] == "/skip":
            assert current_session() is None
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"skipped"]
        assert environ["llm_tape.session"] is require_session()
        start_response("200 OK", [("Content-Type", "text/plain")])
        return _StreamingBody(fail_midstream=environ["PATH_INFO"] == "/fail")

    cfg = TapeConfig(tape_dir=tapes_dir, mode=TapeMode.RECORD, save_on_error=False)
    mw = TapeWSGIMiddleware(wsgi_app, config=cfg, expose_headers=True, skip_paths=("/skip",))

    headers_seen: list[tuple[str, str]] = []

    def _start(status: str, headers: list[tuple[str, str]]) -> None:
        del status
        headers_seen.extend(headers)

    # Skipped path
    assert b"".join(mw({"REQUEST_METHOD": "GET", "PATH_INFO": "/skip"}, _start)) == b"skipped"

    # Streaming response: session stays active while iterating and closes & saves on exhaustion
    headers_seen.clear()
    resp_iter = mw(
        {"REQUEST_METHOD": "POST", "PATH_INFO": "/gen", "HTTP_X_TAPE_NAME": "wsgi_stream"},
        _start,
    )
    assert current_session() is not None
    chunks = list(resp_iter)
    assert chunks == [b"stream-part", b"-end"]
    assert current_session() is None
    assert inner_closed["count"] == 1
    assert "X-Tape-Calls" in dict(headers_seen)
    assert (tapes_dir / "wsgi_stream.tape.json").is_file()

    # Calling close() early cleans up session and inner iterable cleanly
    early_iter: Any = mw(
        {"REQUEST_METHOD": "POST", "PATH_INFO": "/gen", "HTTP_X_TAPE_NAME": "early_close"},
        _start,
    )
    assert current_session() is not None
    early_iter.close()
    assert current_session() is None
    assert inner_closed["count"] == 2

    # Mid-stream exception triggers exc_occurred=True -> does not save when save_on_error=False
    fail_iter = mw(
        {"REQUEST_METHOD": "POST", "PATH_INFO": "/fail", "HTTP_X_TAPE_NAME": "failed_stream"},
        _start,
    )
    with pytest.raises(RuntimeError, match="boom mid-stream"):
        list(fail_iter)
    assert current_session() is None
    assert not (tapes_dir / "failed_stream.tape.json").exists()

    # Traversal header rejected -> falls back to default sanitized route name inside tapes_dir
    list(
        mw(
            {"REQUEST_METHOD": "POST", "PATH_INFO": "/safe", "HTTP_X_TAPE_NAME": "../../evil"},
            _start,
        )
    )
    assert (tapes_dir / "POST__safe.tape.json").is_file()

    # Custom resolver returning None passes through without TapeSession
    passthrough_mw = TapeWSGIMiddleware(wsgi_app, config=cfg, cassette_resolver=lambda env: None)
    assert (
        b"".join(passthrough_mw({"REQUEST_METHOD": "GET", "PATH_INFO": "/skip"}, _start))
        == b"skipped"
    )

    # Synchronous exception raised by app cleans up TapeSession immediately
    def crashing_app(environ: dict[str, Any], start_response: Any) -> list[bytes]:
        del environ, start_response
        raise ValueError("app crashed before response")

    crash_mw = TapeWSGIMiddleware(crashing_app, config=cfg)
    with pytest.raises(ValueError, match="app crashed"):
        crash_mw({"REQUEST_METHOD": "POST", "PATH_INFO": "/crash"}, _start)
    assert current_session() is None
    assert not (tapes_dir / "POST__crash.tape.json").exists()


# --- 6. stdlib unittest Mixin ----------------------------------------------


def test_unittest_mixin_pass_failure_no_save_and_strict_unused(tmp_path: Path) -> None:
    tapes_dir = tmp_path / "ut_tapes"

    class PassingAndFailingTests(TapeTestCaseMixin, unittest.TestCase):
        tape_dir = tapes_dir
        tape_mode = TapeMode.RECORD

        def test_passes_and_saves(self) -> None:
            with httpx.Client(
                transport=httpx.MockTransport(lambda r: _mock_openai_response("pass"))
            ) as c:
                c.post("https://api.openai.com/v1/chat/completions", json={"n": 1})
                c.post("https://api.openai.com/v1/chat/completions", json={"n": 2})
            self.llm_tape.assert_call_count(2)

        def test_fails_and_does_not_save(self) -> None:
            with httpx.Client(
                transport=httpx.MockTransport(lambda r: _mock_openai_response("fail"))
            ) as c:
                c.post("https://api.openai.com/v1/chat/completions", json={"n": 1})
            self.fail("intentional test failure")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PassingAndFailingTests)
    res = unittest.TestResult()
    suite.run(res)
    assert len(res.failures) == 1
    assert len(res.errors) == 0
    assert (tapes_dir / "PassingAndFailingTests__test_passes_and_saves.tape.json").is_file()
    assert not (
        tapes_dir / "PassingAndFailingTests__test_fails_and_does_not_save.tape.json"
    ).exists()

    # Copy 2-interaction cassette for strict_unused replay check
    src_cassette = tapes_dir / "PassingAndFailingTests__test_passes_and_saves.tape.json"
    strict_target = tapes_dir / "StrictUnusedTests__test_only_uses_one.tape.json"
    strict_target.write_bytes(src_cassette.read_bytes())

    class StrictUnusedTests(TapeTestCaseMixin, unittest.TestCase):
        tape_dir = tapes_dir
        tape_mode = TapeMode.REPLAY
        tape_strict_unused = True

        def test_only_uses_one(self) -> None:
            with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as c:
                c.post("https://api.openai.com/v1/chat/completions", json={"n": 1})

    res_strict = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(StrictUnusedTests).run(res_strict)
    assert not res_strict.wasSuccessful()
    combined_trace = "\n".join(msg for _, msg in (*res_strict.errors, *res_strict.failures))
    assert "unused recorded interaction" in combined_trace


# --- 7. Global / Process-level Activation Handle ---------------------------


def test_activate_tape_and_activate_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Safe no-op when neither ENABLED nor CASSETTE is set
    monkeypatch.delenv("APP_TAPE_ENABLED", raising=False)
    monkeypatch.delenv("APP_TAPE_CASSETTE", raising=False)
    assert activate_from_env(prefix="APP_TAPE_") is None

    # Explicit ENABLED=false stays disabled even if CASSETTE is set
    monkeypatch.setenv("APP_TAPE_ENABLED", "false")
    monkeypatch.setenv("APP_TAPE_CASSETTE", "ignored")
    assert activate_from_env(prefix="APP_TAPE_") is None

    # Activating via env vars records and saves cleanly on close()
    monkeypatch.setenv("APP_TAPE_ENABLED", "true")
    monkeypatch.setenv("APP_TAPE_DIR", str(tmp_path / "global_tapes"))
    monkeypatch.setenv("APP_TAPE_MODE", "record")
    monkeypatch.setenv("APP_TAPE_CASSETTE", "worker_job")

    handle = activate_from_env(prefix="APP_TAPE_")
    assert handle is not None
    with handle:
        assert current_session() is handle.session
        with httpx.Client(
            transport=httpx.MockTransport(lambda r: _mock_openai_response("worker"))
        ) as c:
            c.post("https://api.openai.com/v1/chat/completions", json={"job": 1})
        handle.session.assert_call_count(1)
    assert handle.closed is True
    assert current_session() is None
    assert (tmp_path / "global_tapes" / "worker_job.tape.json").is_file()

    # activate_tape direct usage with context manager & idempotent close
    h2 = activate_tape(tmp_path / "direct.tape.json", mode=TapeMode.RECORD)
    assert current_session() is h2.session
    h2.close()
    h2.close()  # idempotent
    assert current_session() is None


# --- 8. Generic Adapter Bridge (Sync & Async) ------------------------------


@pytest.mark.asyncio
async def test_generic_http_bridge_sync_and_async_record_replay_scrubbing_and_assertions(
    tmp_path: Path,
) -> None:
    cassette_path = tmp_path / "generic_sync.tape.json"
    req = GenericHTTPRequest(
        method="POST",
        url="https://api.openai.com/v1/chat/completions?api_key=supersecret",
        headers={
            "Authorization": "Bearer sk-proj-SECRET12345678901234567890",
            "Content-Type": "application/json",
        },
        body=json.dumps(
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Contact alice@example.com"}],
            }
        ).encode("utf-8"),
    )

    def sync_sender(r: GenericHTTPRequest) -> GenericHTTPResponse:
        assert r.url.endswith("api_key=supersecret")
        body = json.dumps(
            {
                "id": "chatcmpl-gen",
                "model": "gpt-4o-mini",
                "choices": [
                    {"message": {"role": "assistant", "content": "Reply to bob@example.com"}}
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 10},
            }
        ).encode("utf-8")
        return GenericHTTPResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=body,
        )

    # Sync RECORD with PII redaction + custom pricing
    with TapeSession(
        cassette_path,
        mode=TapeMode.RECORD,
        redact_pii="email",
        custom_pricing={"gpt-4o-mini": ModelPrice(1.0, 2.0)},
        patch_httpx=False,
        patch_requests=False,
        patch_aiohttp=False,
    ) as rec:
        resp = handle_generic_request(rec, req, sync_sender)
        assert resp.status_code == 200
        rec.assert_call_count(1)
        rec.assert_tokens_under(100)
        rec.assert_cost_under(0.01)

    # Verify scrubbing & PII redaction persisted to cassette
    raw_saved = cassette_path.read_text(encoding="utf-8")
    assert "supersecret" not in raw_saved
    assert "SECRET12345678901234567890" not in raw_saved
    assert "alice@example.com" not in raw_saved
    assert "bob@example.com" not in raw_saved

    # Sync REPLAY (network sender must NOT be called)
    with TapeSession(
        cassette_path,
        mode=TapeMode.REPLAY,
        redact_pii="email",
        patch_httpx=False,
        patch_requests=False,
        patch_aiohttp=False,
    ) as rep:
        replayed = handle_generic_request(
            rep,
            req,
            lambda _: (_ for _ in ()).throw(AssertionError("network called in replay")),
        )
        assert replayed.status_code == 200
        assert "[REDACTED]" in json.loads(replayed.body)["choices"][0]["message"]["content"]
        rep.assert_call_count(1)
        with pytest.raises(TapeAssertionError):
            rep.assert_tokens_under(10)

    # Async RECORD + REPLAY via handle_generic_request_async
    async_path = tmp_path / "generic_async.tape.json"

    async def async_sender(r: GenericHTTPRequest) -> GenericHTTPResponse:
        del r
        body = json.dumps(
            {
                "id": "chatcmpl-gen-async",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "async bridge ok"}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 5},
            }
        ).encode("utf-8")
        return GenericHTTPResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=body,
        )

    async with TapeSession(
        async_path,
        mode=TapeMode.RECORD,
        patch_httpx=False,
        patch_requests=False,
        patch_aiohttp=False,
    ) as arec:
        aresp = await handle_generic_request_async(arec, req, async_sender)
        assert json.loads(aresp.body)["choices"][0]["message"]["content"] == "async bridge ok"

    async with TapeSession(
        async_path,
        mode=TapeMode.REPLAY,
        patch_httpx=False,
        patch_requests=False,
        patch_aiohttp=False,
    ) as arep:

        async def _never_called(r: GenericHTTPRequest) -> GenericHTTPResponse:
            del r
            raise AssertionError("network called in async replay")

        areplayed = await handle_generic_request_async(arep, req, _never_called)
        assert json.loads(areplayed.body)["choices"][0]["message"]["content"] == "async bridge ok"
        arep.assert_call_count(1)
        arep.assert_text_contains("async bridge ok")
