"""End-to-end Anthropic SDK wire-protocol integration tests (Wave 94)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx

from rhinotape import TapeMode, tape


def test_anthropic_messages_json_and_sse_with_thinking_and_cache(tmp_path: Path) -> None:
    cassette_path = tmp_path / "anthropic_e2e.tape.json"

    class _AnthropicSSE(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield (
                b'event: message_start\ndata: {"type":"message_start","message":'
                b'{"id":"msg_1","model":"claude-sonnet-4-20250514","usage":'
                b'{"input_tokens":100,"cache_creation_input_tokens":500,'
                b'"cache_read_input_tokens":400,"output_tokens":1}}}\n\n'
            )
            yield (
                b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"thinking_delta","thinking":"reasoning step"}}\n\n'
            )
            yield (
                b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,'
                b'"delta":{"type":"text_delta","text":"final answer"}}\n\n'
            )
            yield (
                b'event: content_block_start\ndata: {"type":"content_block_start","index":2,'
                b'"content_block":{"type":"tool_use","id":"tu_1","name":"run_sql"}}\n\n'
            )
            yield (
                b'event: content_block_delta\ndata: {"type":"content_block_delta","index":2,'
                b'"delta":{"type":"input_json_delta",'
                b'"partial_json":"{\\"q\\":\\"SELECT 1\\"}"}}\n\n'
            )
            yield (
                b'event: message_delta\ndata: {"type":"message_delta",'
                b'"usage":{"output_tokens":40}}\n\n'
            )

    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_AnthropicSSE(),
        )
    )

    with tape(cassette_path, mode=TapeMode.RECORD):
        with httpx.Client(transport=transport) as client:
            with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": "sk-ant-secret-key-01234567890123456789"},
                json={"model": "claude-sonnet-4-20250514", "stream": True, "messages": []},
            ) as resp:
                assert b"final answer" in resp.read()

    # Verify x-api-key was scrubbed on disk
    assert "sk-ant-secret-key" not in cassette_path.read_text(encoding="utf-8")

    with tape(cassette_path, mode=TapeMode.REPLAY) as session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                json={"model": "claude-sonnet-4-20250514", "stream": True, "messages": []},
            ) as resp:
                assert resp.status_code == 200
                resp.read()
        session.assert_provider("anthropic")
        session.assert_streamed()
        session.assert_tool_called("run_sql")
        session.assert_tool_args("run_sql", {"q": "SELECT 1"})
        session.assert_text_contains("final answer")
        assert session.interactions[0].llm.thinking == "reasoning step"
        assert session.total_usage.cache_creation_tokens == 500
        assert session.total_usage.cached_input_tokens == 400
        session.assert_no_unused_interactions()
