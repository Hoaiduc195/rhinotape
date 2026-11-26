"""Targeted coverage tests for server.py and html_report.py (Wave 108)."""

from __future__ import annotations

import socket
from pathlib import Path

import httpx
import pytest

from rhinotape import Cassette, TapeReplayServer
from rhinotape.html_report import _fmt_body


def test_server_edge_cases_methods_and_invalid_content_length(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TapeReplayServer(tmp_path / "does_not_exist.tape.json")

    empty = Cassette(path=tmp_path / "empty.tape.json", interactions=[])
    srv = TapeReplayServer(empty, port=0)
    assert srv.port == 0
    srv.start()
    # Calling start() twice is idempotent
    assert srv.start() is srv
    try:
        with httpx.Client() as client:
            for method in ("PUT", "PATCH", "DELETE"):
                r = client.request(method, f"{srv.url}/v1/items/1")
                assert r.status_code == 404

        # Raw socket test for malformed & negative Content-Length -> 400
        for bad_len in ("not-an-int", "-5"):
            with socket.create_connection((srv.host, srv.port), timeout=5) as sock:
                raw_req = (
                    f"POST /v1/chat/completions HTTP/1.1\r\n"
                    f"Host: {srv.host}:{srv.port}\r\n"
                    f"Content-Length: {bad_len}\r\n"
                    f"Connection: close\r\n\r\n"
                ).encode("ascii")
                sock.sendall(raw_req)
                chunks: list[bytes] = []
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
                assert b"400 Bad Request" in data
                assert b"invalid_content_length" in data
    finally:
        srv.stop()


def test_html_report_fmt_body_none() -> None:
    assert _fmt_body(None) == ""
