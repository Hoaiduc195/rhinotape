"""Tests for Wave 6: WHATWG SSE spec hardening (BOM, CRLF, single space strip, bare fields)."""

from __future__ import annotations

from rhinotape.parsers import parse_sse_events, split_sse_chunks


def test_parse_sse_whatwg_bom_crlf_comments_and_multiline() -> None:
    raw = (
        "\ufeff"
        ": keep-alive comment\r\n"
        "event: custom\r\n"
        "data:   two leading spaces preserved\r\n"
        "data\r\n"
        "data: second line\r\n"
        "\r\n"
        "event\r\n"
        'data: {"ok": true}\r\n'
        "\r\n"
        "data: [DONE]\r\n\r\n"
    )
    events = parse_sse_events(raw)
    assert len(events) == 2
    assert events[0] == ("custom", "  two leading spaces preserved\n\nsecond line")
    assert events[1] == (None, {"ok": True})


def test_split_sse_chunks_handles_lf_and_crlf() -> None:
    assert split_sse_chunks("") == []
    assert split_sse_chunks("data: 1\n\ndata: 2\n\n") == ["data: 1\n\n", "data: 2\n\n"]
    assert split_sse_chunks("data: 1\r\n\r\ndata: 2\r\n\r\n") == [
        "data: 1\r\n\r\n",
        "data: 2\r\n\r\n",
    ]
    assert split_sse_chunks("data: trailing") == ["data: trailing"]
