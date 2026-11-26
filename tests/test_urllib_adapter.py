"""Tests for Wave 4: stdlib urllib.request adapter (JSON, SSE, HTTPError, scrubbing)."""

from __future__ import annotations

import http.server
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from rhinotape import Cassette, TapeMode, tape


class _LocalLLMHandler(http.server.BaseHTTPRequestHandler):
    hits = 0

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_POST(self) -> None:
        type(self).hits += 1
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        if self.path.endswith("/sse"):
            body = (
                b'data: {"model":"gpt-4o-mini","choices":[{"delta":{"content":"hello "}}]}\n\n'
                b'data: {"model":"gpt-4o-mini","choices":[{"delta":{"content":"urllib"}}]}\n\n'
                b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = json.dumps(
            {
                "id": "chatcmpl-u",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "from urllib"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        type(self).hits += 1
        body = json.dumps({"error": "rate_limited"}).encode()
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_urllib_record_replay_sse_and_http_error(tmp_path: Path) -> None:
    _LocalLLMHandler.hits = 0
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LocalLLMHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    orig_open = urllib.request.OpenerDirector.open

    try:
        base = f"http://127.0.0.1:{port}"
        tape_path = tmp_path / "urllib.tape.json"

        with tape(tape_path, mode=TapeMode.RECORD, patch_urllib=True) as rec:
            # 1. JSON POST with Authorization secret
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=json.dumps({"model": "gpt-4o-mini"}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer sk-urllib-secret-1234567890",
                },
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                payload = json.loads(resp.read().decode())
                assert payload["choices"][0]["message"]["content"] == "from urllib"

            # 2. SSE POST
            sse_req = urllib.request.Request(
                f"{base}/v1/chat/completions/sse",
                data=b'{"stream":true}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(sse_req) as sse_resp:
                lines = [ln.decode().strip() for ln in sse_resp if ln.strip()]
                assert len(lines) == 3

            # 3. 429 HTTPError
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"{base}/v1/err")
            with exc_info.value as err:
                assert err.code == 429
                assert json.loads(err.read().decode()) == {"error": "rate_limited"}

            rec.assert_call_count(3)
            assert rec.total_tokens == 15
            assert rec.interactions[1].llm.text == "hello urllib"

        assert urllib.request.OpenerDirector.open is orig_open
        assert _LocalLLMHandler.hits == 3

        saved = Cassette.load(tape_path)
        assert saved.interactions[0].request.headers["Authorization"] == "[REDACTED]"

        # Replay completely offline (server hits must NOT increment)
        with tape(tape_path, mode=TapeMode.REPLAY, patch_urllib=True) as rep:
            req2 = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=json.dumps({"model": "gpt-4o-mini"}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer sk-different",
                },
                method="POST",
            )
            with urllib.request.urlopen(req2) as r2:
                assert r2.getcode() == 200
                assert json.loads(r2.read().decode())["choices"][0]["message"]["content"] == (
                    "from urllib"
                )

            sse_req2 = urllib.request.Request(
                f"{base}/v1/chat/completions/sse",
                data=b'{"stream":true}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(sse_req2) as sr2:
                assert [ln.decode().strip() for ln in sr2 if ln.strip()] == lines

            with pytest.raises(urllib.error.HTTPError) as exc_info2:
                urllib.request.urlopen(f"{base}/v1/err")
            with exc_info2.value as err2:
                assert err2.code == 429
                assert json.loads(err2.read().decode()) == {"error": "rate_limited"}

            rep.assert_call_count(3)
            rep.assert_no_unused_interactions()

        assert _LocalLLMHandler.hits == 3
        assert urllib.request.OpenerDirector.open is orig_open
    finally:
        server.shutdown()
        server.server_close()


def test_urllib_rv_review_regressions_file_url_bytesio_and_content_encoding(
    tmp_path: Path,
) -> None:
    import gzip
    import io

    from rhinotape.urllib_adapter import _build_email_headers, _extract_urllib_request

    # P1 #1: Content-Encoding header preserved alongside raw bytes
    hdrs = _build_email_headers({"Content-Encoding": "gzip", "Content-Type": "application/json"})
    assert hdrs.get("Content-Encoding") == "gzip"

    # P2 #3: file-like BytesIO Request.data reads deterministically & updates req.data
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=io.BytesIO(b'{"a":1}'),
        method="POST",
    )
    _m, _u, _h, body_bytes, _d = _extract_urllib_request(req, None)
    assert body_bytes == b'{"a":1}'
    assert req.data == b'{"a":1}'

    # P2 #4: file:// URLs pass through unintercepted
    local_file = tmp_path / "sample.txt"
    local_file.write_text("local-content", encoding="utf-8")
    tape_path = tmp_path / "file_passthrough.tape.json"
    with tape(tape_path, mode=TapeMode.RECORD, patch_urllib=True) as s:
        with urllib.request.urlopen(local_file.as_uri()) as resp:
            assert resp.read().decode() == "local-content"
        s.assert_call_count(0)
    assert gzip.compress(b"x")[:2] == b"\x1f\x8b"
