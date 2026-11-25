"""Local HTTP replay server for serving recorded llm-tape cassettes to any HTTP client."""

from __future__ import annotations

import http.server
import json
import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from llm_tape.cassette import Cassette, iter_cassette_files
from llm_tape.models import Interaction
from llm_tape.scrubbing import (
    decode_body_with_encoding,
    encode_body_with_encoding,
    fingerprint_request,
    scrub_headers,
    scrub_url,
)

_STRIP_HEADERS = frozenset(
    {"content-encoding", "transfer-encoding", "content-length", "connection"}
)


class TapeReplayServer:
    """Thread-safe local HTTP server that replays recorded cassette interactions.

    Matches incoming requests by ``(method, path, body)`` by default so SDKs pointed at
    ``http://127.0.0.1:<port>`` transparently replay cassettes recorded against live hosts.
    """

    def __init__(
        self,
        source: str | Path | Cassette,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        allow_playback_repeats: bool = True,
        ignore_fields: Iterable[str] = (),
        match_on: tuple[str, ...] = ("method", "path", "body"),
        replay_latency_scale: float = 0.0,
    ) -> None:
        self.host = host
        self.requested_port = port
        self.allow_playback_repeats = allow_playback_repeats
        self.ignore_fields = tuple(ignore_fields)
        self.match_on = match_on
        if not math.isfinite(replay_latency_scale) or replay_latency_scale < 0.0:
            raise ValueError("replay_latency_scale must be a finite number >= 0.0")
        self.replay_latency_scale = float(replay_latency_scale)

        self._lock = threading.Lock()
        self._queues: dict[str, deque[Interaction]] = defaultdict(deque)
        self._loaded_count = 0
        self._served_count = 0
        self._unmatched_count = 0
        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

        self._load_source(source)

    def _load_source(self, source: str | Path | Cassette) -> None:
        cassettes: list[Cassette] = []
        if isinstance(source, Cassette):
            cassettes.append(source)
        else:
            p = Path(source)
            if p.is_file():
                cassettes.append(Cassette.load(p))
            elif p.is_dir():
                for file_path in iter_cassette_files(p):
                    cassettes.append(Cassette.load(file_path))
            else:
                raise FileNotFoundError(f"Cassette source not found: {p}")

        for cassette in cassettes:
            for inter in cassette.interactions:
                fp = fingerprint_request(
                    inter.request.method,
                    inter.request.url,
                    inter.request.body,
                    headers=inter.request.headers,
                    ignore_fields=self.ignore_fields,
                    match_on=self.match_on,
                )
                self._queues[fp].append(inter)
                self._loaded_count += 1

    def _match(
        self,
        method: str,
        path: str,
        body_bytes: bytes,
        headers: dict[str, str] | None = None,
    ) -> Interaction | None:
        decoded_body, _ = decode_body_with_encoding(body_bytes)
        synthetic_url = scrub_url(f"http://localhost{path}")
        fp = fingerprint_request(
            method,
            synthetic_url,
            decoded_body,
            headers=headers,
            ignore_fields=self.ignore_fields,
            match_on=self.match_on,
        )
        with self._lock:
            q = self._queues.get(fp)
            if not q:
                self._unmatched_count += 1
                return None
            inter = q[0] if (self.allow_playback_repeats and len(q) == 1) else q.popleft()
            self._served_count += 1
            return inter

    def stats(self) -> dict[str, int]:
        """Return live server replay counters."""
        with self._lock:
            return {
                "loaded_interactions": self._loaded_count,
                "served_requests": self._served_count,
                "unmatched_requests": self._unmatched_count,
            }

    def _make_handler_cls(self) -> type[http.server.BaseHTTPRequestHandler]:
        server_ref = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                del format, args

            def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, method: str) -> None:
                if method == "GET" and self.path == "/__tape__/health":
                    self._send_json(200, {"status": "ok"})
                    return
                if method == "GET" and self.path == "/__tape__/stats":
                    self._send_json(200, server_ref.stats())
                    return

                raw_len = self.headers.get("Content-Length", "0") or "0"
                try:
                    length = int(raw_len)
                except ValueError:
                    self._send_json(400, {"error": "invalid_content_length"})
                    return
                if length < 0:
                    self._send_json(400, {"error": "invalid_content_length"})
                    return
                body_bytes = self.rfile.read(length) if length > 0 else b""
                clean_headers = scrub_headers(self.headers.items())
                inter = server_ref._match(method, self.path, body_bytes, headers=clean_headers)
                if inter is None:
                    safe_url = scrub_url(f"http://localhost{self.path}")
                    safe_path = safe_url[len("http://localhost") :]
                    self._send_json(
                        404,
                        {"error": "unmatched_request", "method": method, "path": safe_path},
                    )
                    return

                self._write_interaction(inter)

            def _write_interaction(self, inter: Interaction) -> None:
                delay_s = (
                    (inter.latency_ms * server_ref.replay_latency_scale) / 1000.0
                    if server_ref.replay_latency_scale > 0.0 and inter.latency_ms > 0.0
                    else 0.0
                )
                if inter.response.stream_chunks is not None:
                    chunks = inter.response.stream_chunks
                    encoded_chunks = [c.encode("utf-8") for c in chunks]
                    total_len = sum(len(b) for b in encoded_chunks)
                    chunk_delay = (
                        delay_s / len(encoded_chunks) if (delay_s > 0 and encoded_chunks) else 0.0
                    )
                    self.send_response(inter.response.status_code)
                    for k, v in inter.response.headers.items():
                        if k.lower() not in _STRIP_HEADERS:
                            self.send_header(k, v)
                    self.send_header("Content-Length", str(total_len))
                    self.end_headers()
                    for chunk_bytes in encoded_chunks:
                        if chunk_delay > 0.0:
                            time.sleep(chunk_delay)
                        if chunk_bytes:
                            self.wfile.write(chunk_bytes)
                            self.wfile.flush()
                    return

                if delay_s > 0.0:
                    time.sleep(delay_s)
                payload = encode_body_with_encoding(
                    inter.response.body, inter.response.body_encoding
                )
                self.send_response(inter.response.status_code)
                for k, v in inter.response.headers.items():
                    if k.lower() not in _STRIP_HEADERS:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if payload:
                    self.wfile.write(payload)

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

            def do_PUT(self) -> None:
                self._handle("PUT")

            def do_PATCH(self) -> None:
                self._handle("PATCH")

            def do_DELETE(self) -> None:
                self._handle("DELETE")

        return _Handler

    @property
    def port(self) -> int:
        if self._httpd is None:
            return self.requested_port
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> TapeReplayServer:
        if self._httpd is not None:
            return self
        self._httpd = http.server.ThreadingHTTPServer(
            (self.host, self.requested_port), self._make_handler_cls()
        )
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> TapeReplayServer:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.stop()
