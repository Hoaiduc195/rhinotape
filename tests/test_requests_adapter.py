"""Tests for optional `requests` HTTP adapter (JSON, SSE streaming, scrubbing, isolation)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
import requests

from llm_tape import Cassette, TapeMode, tape
from llm_tape.requests_adapter import (
    _ReplayRequestsRaw,
    acquire_requests_patch,
    release_requests_patch,
)


class _FakeUpstreamAdapter(requests.adapters.BaseAdapter):
    """In-memory requests adapter for deterministic offline record tests."""

    def __init__(self, handler: Any) -> None:
        super().__init__()
        self._handler = handler
        self.call_count = 0

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: Any = None,
        verify: Any = True,
        cert: Any = None,
        proxies: Any = None,
    ) -> requests.Response:
        del timeout, verify, cert, proxies
        self.call_count += 1
        return self._handler(request, stream)

    def close(self) -> None:
        pass


def _make_json_response(
    request: requests.PreparedRequest,
    payload: dict[str, Any],
    *,
    status_code: int = 200,
) -> requests.Response:
    raw = json.dumps(payload).encode("utf-8")
    resp = requests.Response()
    resp.status_code = status_code
    resp.reason = "OK"
    resp.url = str(request.url)
    resp.request = request
    resp.encoding = "utf-8"
    resp.headers = requests.structures.CaseInsensitiveDict({"Content-Type": "application/json"})
    resp.raw = _ReplayRequestsRaw((raw,))
    resp._content = raw
    resp._content_consumed = True
    return resp


def _make_sse_response(
    request: requests.PreparedRequest,
    chunks: tuple[bytes, ...],
    *,
    stream: bool,
) -> requests.Response:
    resp = requests.Response()
    resp.status_code = 200
    resp.reason = "OK"
    resp.url = str(request.url)
    resp.request = request
    resp.encoding = "utf-8"
    resp.headers = requests.structures.CaseInsensitiveDict({"Content-Type": "text/event-stream"})
    if stream:
        resp.raw = _ReplayRequestsRaw(chunks)
        resp._content = False
        resp._content_consumed = False
    else:
        raw = b"".join(chunks)
        resp.raw = _ReplayRequestsRaw((raw,))
        resp._content = raw
        resp._content_consumed = True
    return resp


def test_requests_json_record_and_replay_with_scrubbing(tmp_path: Path) -> None:
    cassette_path = tmp_path / "req_json.tape.json"
    orig_send = requests.Session.send

    adapter = _FakeUpstreamAdapter(
        lambda req, stream: _make_json_response(
            req,
            {
                "id": "chatcmpl-req",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "hello from requests"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            },
        )
    )

    with tape(cassette_path, mode=TapeMode.RECORD) as rec:
        sess = requests.Session()
        sess.mount("https://", adapter)
        r = sess.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": "Bearer sk-real-secret-token-1234567890"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "hello from requests"
        rec.assert_call_count(1)
        assert rec.total_tokens == 25

    assert requests.Session.send is orig_send
    assert adapter.call_count == 1

    # Verify secret was scrubbed in saved cassette
    saved = Cassette.load(cassette_path)
    assert saved.interactions[0].request.headers["Authorization"] == "[REDACTED]"

    # Replay without upstream adapter ever being called
    with tape(cassette_path, mode=TapeMode.REPLAY) as rep:
        sess = requests.Session()
        sess.mount("https://", adapter)
        r2 = sess.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": "Bearer sk-different-key"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r2.status_code == 200
        assert r2.json()["choices"][0]["message"]["content"] == "hello from requests"
        rep.assert_call_count(1)
        rep.assert_no_unused_interactions()

    assert adapter.call_count == 1
    assert requests.Session.send is orig_send


def test_requests_sse_stream_record_replay_and_early_close(tmp_path: Path) -> None:
    cassette_path = tmp_path / "req_sse.tape.json"
    sse_chunks = (
        b'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"content":"Hello "}}]}\n\n',
        b'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"content":"world!"}}]}\n\n',
        b"data: [DONE]\n\n",
    )
    adapter = _FakeUpstreamAdapter(
        lambda req, stream: _make_sse_response(req, sse_chunks, stream=stream)
    )

    # Record streaming SSE
    with tape(cassette_path, mode=TapeMode.RECORD) as rec:
        sess = requests.Session()
        sess.mount("https://", adapter)
        with sess.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4o-mini", "stream": True},
            stream=True,
        ) as resp:
            lines = [line.decode("utf-8") for line in resp.iter_lines() if line]
        assert len(lines) == 3
        rec.assert_call_count(1)
        assert rec.interactions[0].llm.text == "Hello world!"
        assert rec.interactions[0].llm.is_stream is True

    # Replay streaming SSE
    with tape(cassette_path, mode=TapeMode.REPLAY) as rep:
        sess = requests.Session()
        sess.mount("https://", adapter)
        with sess.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4o-mini", "stream": True},
            stream=True,
        ) as resp:
            replayed_lines = [line.decode("utf-8") for line in resp.iter_lines() if line]
        assert replayed_lines == lines
        rep.assert_call_count(1)

    assert adapter.call_count == 1

    # Early close must NOT commit partial stream
    early_path = tmp_path / "early.tape.json"
    with tape(early_path, mode=TapeMode.RECORD) as rec2:
        sess = requests.Session()
        sess.mount("https://", adapter)
        resp = sess.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4o-mini", "stream": True},
            stream=True,
        )
        # Read only 1 chunk then close early without reaching EOF
        it = resp.iter_content(chunk_size=16)
        next(it)
        resp.close()
        assert len(rec2.interactions) == 0


def test_requests_non_stream_sse_ignore_localhost_and_patch_false(tmp_path: Path) -> None:
    cassette_path = tmp_path / "non_stream_sse.tape.json"
    sse_chunks = (
        b'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"content":"ok"}}]}\n\n',
        b"data: [DONE]\n\n",
    )
    adapter = _FakeUpstreamAdapter(
        lambda req, stream: _make_sse_response(req, sse_chunks, stream=stream)
    )

    with tape(cassette_path, mode=TapeMode.RECORD, ignore_localhost=True) as rec:
        sess = requests.Session()
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        # Localhost ignored
        sess.get("http://127.0.0.1:9999/health")
        # Non-stream POST receiving SSE
        r = sess.post(
            "https://api.openai.com/v1/chat/completions",
            data=io.BytesIO(b'{"model":"gpt-4o-mini"}'),
            headers={"Content-Type": "application/json"},
        )
        assert "data: [DONE]" in r.text
        rec.assert_call_count(1)
        assert rec.interactions[0].llm.text == "ok"

    # patch_requests=False leaves requests untouched
    orig_send = requests.Session.send
    with tape(cassette_path, mode=TapeMode.REPLAY, patch_requests=False):
        assert requests.Session.send is orig_send


def test_requests_adapter_graceful_when_requests_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "requests":
            raise ImportError("No module named 'requests'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    assert acquire_requests_patch(lambda: None) is False
    release_requests_patch()  # safe no-op


def test_requests_rv_review_regressions_redirects_binary_stream_and_read0(tmp_path: Path) -> None:
    from llm_tape.requests_adapter import _RecordingRequestsRaw

    # P1 #1: Redirect hops must not double-record or break replay
    def redirect_handler(req: requests.PreparedRequest, stream: bool) -> requests.Response:
        del stream
        if str(req.url).endswith("/old"):
            resp = requests.Response()
            resp.status_code = 302
            resp.reason = "Found"
            resp.url = str(req.url)
            resp.request = req
            resp.headers = requests.structures.CaseInsensitiveDict(
                {"Location": "https://api.openai.com/v1/chat/completions"}
            )
            resp.raw = _ReplayRequestsRaw((b"",))
            resp._content = b""
            resp._content_consumed = True
            return resp
        return _make_json_response(
            req,
            {
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "redirected"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
        )

    redirect_tape = tmp_path / "redirect.tape.json"
    adapter = _FakeUpstreamAdapter(redirect_handler)
    with tape(redirect_tape, mode=TapeMode.RECORD) as rec:
        sess = requests.Session()
        sess.mount("https://", adapter)
        r = sess.get("https://api.openai.com/old")
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "redirected"
        rec.assert_call_count(1)

    with tape(redirect_tape, mode=TapeMode.REPLAY) as rep:
        sess = requests.Session()
        sess.mount("https://", adapter)
        r2 = sess.get("https://api.openai.com/old")
        assert r2.status_code == 200
        assert r2.json()["choices"][0]["message"]["content"] == "redirected"
        rep.assert_call_count(1)
        rep.assert_no_unused_interactions()

    # P1 #2: stream=True with JSON & binary responses preserves JSON parsing and binary bytes
    bin_payload = b"\x89PNG\r\n\x1a\n\x00\xff\xfe\xfd"

    def stream_non_sse_handler(req: requests.PreparedRequest, stream: bool) -> requests.Response:
        resp = requests.Response()
        resp.status_code = 200
        resp.reason = "OK"
        resp.url = str(req.url)
        resp.request = req
        if "binary" in str(req.url):
            resp.headers = requests.structures.CaseInsensitiveDict({"Content-Type": "image/png"})
            resp.raw = _ReplayRequestsRaw((bin_payload[:5], bin_payload[5:]))
        else:
            raw_json = json.dumps(
                {
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"role": "assistant", "content": "streamed json"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
                }
            ).encode("utf-8")
            resp.headers = requests.structures.CaseInsensitiveDict(
                {"Content-Type": "application/json"}
            )
            resp.raw = _ReplayRequestsRaw((raw_json[:20], raw_json[20:]))
        resp._content = False if stream else b""
        resp._content_consumed = not stream
        return resp

    non_sse_tape = tmp_path / "stream_non_sse.tape.json"
    non_sse_adapter = _FakeUpstreamAdapter(stream_non_sse_handler)
    with tape(non_sse_tape, mode=TapeMode.RECORD) as rec2:
        sess = requests.Session()
        sess.mount("https://", non_sse_adapter)
        with sess.post("https://api.openai.com/v1/chat/completions", stream=True) as rj:
            body_json = json.loads(b"".join(rj.iter_content(chunk_size=16)))
            assert body_json["choices"][0]["message"]["content"] == "streamed json"
        with sess.get("https://api.openai.com/v1/binary", stream=True) as rb:
            body_bin = b"".join(rb.iter_content(chunk_size=4))
            assert body_bin == bin_payload
        rec2.assert_call_count(2)
        assert rec2.total_tokens == 15
        assert rec2.interactions[0].response.body_encoding == "json"
        assert rec2.interactions[1].response.body_encoding == "base64"

    with tape(non_sse_tape, mode=TapeMode.REPLAY) as rep2:
        sess = requests.Session()
        sess.mount("https://", non_sse_adapter)
        with sess.get("https://api.openai.com/v1/binary", stream=True) as rb2:
            assert b"".join(rb2.iter_content(chunk_size=4)) == bin_payload
        assert rep2.interactions[0].response.body_encoding == "base64"

    # P2 #4: read(0) must not trigger premature EOF commit
    committed: list[bytes] = []
    rec_raw = _RecordingRequestsRaw(
        _ReplayRequestsRaw((b"abc",)),
        is_sse=False,
        on_complete_bytes=committed.append,
    )
    assert rec_raw.read(0) == b""
    assert committed == []
    assert rec_raw.read() == b"abc"
    assert committed == [b"abc"]
