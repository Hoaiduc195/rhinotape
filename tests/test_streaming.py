"""Tests for SSE streaming record/replay and provider SSE parsers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest

from llm_tape import Cassette, tape
from llm_tape.parsers import parse_llm_interaction


class _FakeSyncSSEStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = [c.encode("utf-8") for c in chunks]

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks


class _FakeAsyncSSEStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = [c.encode("utf-8") for c in chunks]

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


OPENAI_SSE_CHUNKS = [
    'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"content":"Hello "}}]}\n\n',
    'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"content":"world!"}}]}\n\n',
    (
        'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"tool_calls":'
        '[{"index":0,"id":"call_1","function":{"name":"search","arguments":"{\\"q\\": "}}]}}]}\n\n'
    ),
    (
        'data: {"id":"1","model":"gpt-4o-mini","choices":[{"delta":{"tool_calls":'
        '[{"index":0,"function":{"arguments":"\\"llm\\"}"}}]}}]}\n\n'
    ),
    (
        'data: {"id":"1","model":"gpt-4o-mini","choices":[],'
        '"usage":{"prompt_tokens":80,"completion_tokens":20}}\n\n'
    ),
    "data: [DONE]\n\n",
]


def test_sync_sse_stream_record_and_replay(tmp_path: Path) -> None:
    cassette_path = tmp_path / "openai_stream.tape.json"
    hits = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_FakeSyncSSEStream(OPENAI_SSE_CHUNKS),
        )

    # Record live stream
    with tape(cassette_path, mode="once") as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "stream": True},
            ) as resp:
                lines = [line for line in resp.iter_lines() if line]
        assert len(lines) == 6
        assert session.interactions[0].llm.is_stream is True
        assert session.interactions[0].llm.text == "Hello world!"
        session.assert_tokens_under(150)
        session.assert_tool_called("search")
        assert session.tool_calls[0].arguments == {"q": "llm"}

    assert hits == 1
    saved = Cassette.load(cassette_path)
    assert saved.interactions[0].response.body_encoding == "stream"
    assert saved.interactions[0].response.stream_chunks is not None

    # Replay offline stream
    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Should not hit upstream during stream replay")

    with tape(cassette_path, mode="replay") as session:
        with httpx.Client(transport=httpx.MockTransport(forbidden)) as client:
            with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "stream": True},
            ) as resp:
                replayed_lines = [line for line in resp.iter_lines() if line]
        assert replayed_lines == lines
        session.assert_tool_sequence(["search"])
        assert session.total_tokens == 100


@pytest.mark.asyncio
async def test_async_sse_stream_record_and_replay(tmp_path: Path) -> None:
    cassette_path = tmp_path / "openai_async_stream.tape.json"
    hits = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_FakeAsyncSSEStream(OPENAI_SSE_CHUNKS),
        )

    async with tape(cassette_path, mode="once") as session:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "stream": True},
            ) as resp:
                collected = [chunk async for chunk in resp.aiter_text()]
        assert len(collected) == len(OPENAI_SSE_CHUNKS)
        assert session.total_tokens == 100

    assert hits == 1

    async with tape(cassette_path, mode="replay") as session:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "stream": True},
            ) as resp:
                replayed = [chunk async for chunk in resp.aiter_text()]
        assert replayed == collected
        session.assert_tool_called("search")

    assert hits == 1


def test_parse_anthropic_sse_with_cache_and_tool_use() -> None:
    chunks = [
        (
            "event: message_start\n"
            'data: {"type":"message_start","message":{"model":"claude-3-5-sonnet-20241022",'
            '"usage":{"input_tokens":100,"cache_creation_input_tokens":50,'
            '"cache_read_input_tokens":200}}}\n\n'
        ),
        (
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"Thinking... "}}\n\n'
        ),
        (
            "event: content_block_start\n"
            'data: {"type":"content_block_start","index":1,'
            '"content_block":{"type":"tool_use","id":"toolu_1","name":"calc"}}\n\n'
        ),
        (
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","index":1,'
            '"delta":{"type":"input_json_delta","partial_json":"{\\"x\\": 42}"}}\n\n'
        ),
        ('event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":35}}\n\n'),
    ]
    summary = parse_llm_interaction(
        "https://api.anthropic.com/v1/messages",
        {},
        None,
        stream_chunks=chunks,
        content_type="text/event-stream",
    )
    assert summary.provider == "anthropic"
    assert summary.is_stream is True
    assert summary.usage.input_tokens == 350
    assert summary.usage.output_tokens == 35
    assert summary.text == "Thinking... "
    assert summary.tool_calls[0].name == "calc"
    assert summary.tool_calls[0].arguments == {"x": 42}


def test_parse_gemini_sse_with_thinking_tokens() -> None:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:streamGenerateContent?alt=sse"
    )
    chunks = [
        'data: {"candidates":[{"content":{"parts":[{"text":"Part 1. "}]}}]}\n\n',
        (
            'data: {"candidates":[{"content":{"parts":['
            '{"text":"Part 2."},{"functionCall":{"name":"lookup","args":{"id":7}}}'
            ']}}],"usageMetadata":{"promptTokenCount":120,'
            '"candidatesTokenCount":30,"thoughtsTokenCount":70}}\n\n'
        ),
    ]
    summary = parse_llm_interaction(
        url, {}, None, stream_chunks=chunks, content_type="text/event-stream"
    )
    assert summary.provider == "gemini"
    assert summary.model == "gemini-2.5-flash"
    assert summary.is_stream is True
    assert summary.text == "Part 1. Part 2."
    assert summary.usage.input_tokens == 120
    assert summary.usage.output_tokens == 100  # 30 + 70 thoughtsTokenCount
    assert summary.tool_calls[0].name == "lookup"
    assert summary.tool_calls[0].arguments == {"id": 7}


class _RawBytesSyncStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks


def test_utf8_multibyte_split_across_tcp_chunks_preserved(tmp_path: Path) -> None:
    """Vietnamese & emoji UTF-8 bytes split across chunks must NOT turn into U+FFFD."""
    cassette_path = tmp_path / "utf8_split.tape.json"
    full_line = (
        'data: {"model":"gpt-4o-mini","choices":[{"delta":{"content":"Xin chào Việt Nam 🚀"}}]}\n\n'
    ).encode()
    # Deliberately slice right in the middle of multi-byte UTF-8 characters
    idx_vn = full_line.index("ệ".encode())
    raw_chunks = [full_line[: idx_vn + 1], full_line[idx_vn + 1 : -4], full_line[-4:]]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_RawBytesSyncStream(raw_chunks),
        )

    with tape(cassette_path, mode="record") as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with client.stream("POST", "https://api.openai.com/v1/chat/completions", json={}) as r:
                _ = list(r.iter_bytes())
        assert "\ufffd" not in session.interactions[0].llm.text
        assert session.interactions[0].llm.text == "Xin chào Việt Nam 🚀"


def test_early_closed_stream_not_committed_to_cassette(tmp_path: Path) -> None:
    cassette_path = tmp_path / "aborted_stream.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_FakeSyncSSEStream(OPENAI_SSE_CHUNKS),
        )

    with tape(cassette_path, mode="once") as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with client.stream("POST", "https://api.openai.com/v1/chat/completions", json={}) as r:
                for _chunk in r.iter_bytes():
                    break  # Caller aborts stream before EOF
        session.assert_call_count(0)

    assert not cassette_path.exists()


def test_stream_429_json_error_recorded_as_json_and_readable(tmp_path: Path) -> None:
    cassette_path = tmp_path / "stream_429.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"content-type": "application/json"},
            json={"error": {"message": "Rate limit exceeded"}},
        )

    with tape(cassette_path, mode="record") as session:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with client.stream("POST", "https://api.openai.com/v1/chat/completions", json={}) as r:
                assert r.status_code == 429
                body = r.read()
                assert b"Rate limit exceeded" in body
        assert session.interactions[0].response.body_encoding == "json"
        assert session.interactions[0].response.stream_chunks is None


def test_tape_transport_works_with_client_stream_and_no_double_interception(
    tmp_path: Path,
) -> None:
    from llm_tape import TapeTransport

    cassette_path = tmp_path / "transport_stream.tape.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_FakeSyncSSEStream(OPENAI_SSE_CHUNKS),
        )

    upstream = httpx.MockTransport(handler)
    # Also intentionally keep patch_httpx=True to verify no double-interception!
    with tape(cassette_path, mode="once", patch_httpx=True) as session:
        with httpx.Client(transport=TapeTransport(session, upstream)) as client:
            with client.stream("POST", "https://api.openai.com/v1/chat/completions", json={}) as r:
                assert not r.is_closed
                raw = list(r.iter_raw())
                assert len(raw) == len(OPENAI_SSE_CHUNKS)
        session.assert_call_count(1)

    with tape(cassette_path, mode="replay", patch_httpx=True) as session:
        with httpx.Client(transport=TapeTransport(session, upstream)) as client:
            with client.stream("POST", "https://api.openai.com/v1/chat/completions", json={}) as r:
                replayed = list(r.iter_raw())
                assert len(replayed) == len(OPENAI_SSE_CHUNKS)
        session.assert_call_count(1)


def test_parse_openai_responses_sse_and_repeated_proxy_tool_name() -> None:
    responses_chunks = [
        'data: {"type":"response.output_text.delta","delta":"Result: "}\n\n',
        'data: {"type":"response.output_text.delta","delta":"42"}\n\n',
        (
            'data: {"type":"response.function_call_arguments.done","call_id":"fc_9",'
            '"name":"compute","arguments":"{\\"n\\":42}"}\n\n'
        ),
    ]
    summary = parse_llm_interaction(
        "https://api.openai.com/v1/responses",
        {"model": "gpt-4.1-mini"},
        None,
        stream_chunks=responses_chunks,
        content_type="text/event-stream",
    )
    assert summary.text == "Result: 42"
    assert [tc.name for tc in summary.tool_calls] == ["compute"]

    # Proxy repeating function.name across every delta chunk must not duplicate name
    proxy_chunks = [
        (
            'data: {"model":"gpt-4o-mini","choices":[{"delta":{"tool_calls":'
            '[{"index":0,"id":"c1","function":{"name":"search","arguments":"{\\"a\\":"}}]}}]}\n\n'
        ),
        (
            'data: {"model":"gpt-4o-mini","choices":[{"delta":{"tool_calls":'
            '[{"index":0,"function":{"name":"search","arguments":"1}"}}]}}]}\n\n'
        ),
    ]
    s2 = parse_llm_interaction(
        "https://api.openai.com/v1/chat/completions",
        {},
        None,
        stream_chunks=proxy_chunks,
        content_type="text/event-stream",
    )
    assert s2.tool_calls[0].name == "search"
    assert s2.tool_calls[0].arguments == {"a": 1}
