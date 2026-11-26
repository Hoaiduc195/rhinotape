"""End-to-end OpenAI SDK wire-protocol integration tests (Wave 93)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx

from rhinotape import TapeMode, tape


def test_openai_chat_and_responses_sse_roundtrip(tmp_path: Path) -> None:
    cassette_path = tmp_path / "openai_e2e.tape.json"

    class _OpenAISSE(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield (
                b'data: {"id":"chatcmpl-s","model":"gpt-4o","choices":[{"delta":'
                b'{"role":"assistant","content":"Hello "}}]}\n\n'
            )
            yield (
                b'data: {"id":"chatcmpl-s","model":"gpt-4o","choices":[{"delta":'
                b'{"content":"world!"}}]}\n\n'
            )
            yield (
                b'data: {"id":"chatcmpl-s","model":"gpt-4o","choices":[],'
                b'"usage":{"prompt_tokens":100,"completion_tokens":20,'
                b'"prompt_tokens_details":{"cached_tokens":60}}}\n\n'
            )
            yield b"data: [DONE]\n\n"

    def _handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/responses":
            return httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "model": "o4-mini",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "fc_1",
                            "name": "lookup_order",
                            "arguments": '{"order_id": 42}',
                        },
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Checking order 42."}],
                        },
                    ],
                    "usage": {
                        "input_tokens": 50,
                        "output_tokens": 30,
                        "output_tokens_details": {"reasoning_tokens": 12},
                    },
                },
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_OpenAISSE(),
        )

    with tape(cassette_path, mode=TapeMode.RECORD):
        with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
            with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": "Bearer sk-test-secret-12345678901234567890"},
                json={
                    "model": "gpt-4o",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as r1:
                assert b"world!" in r1.read()
            r2 = client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": "Bearer sk-test-secret-12345678901234567890"},
                json={"model": "o4-mini", "input": "status of order 42"},
            )
            assert r2.status_code == 200

    # Strict offline replay verifies full parity & assertions
    with tape(cassette_path, mode=TapeMode.REPLAY) as session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as r1:
                assert b"Hello " in r1.read()
            r2 = client.post(
                "https://api.openai.com/v1/responses",
                json={"model": "o4-mini", "input": "status of order 42"},
            )
            assert r2.status_code == 200
        session.assert_call_count(2)
        session.assert_provider("openai")
        session.assert_tool_called("lookup_order")
        session.assert_tool_args("lookup_order", {"order_id": 42})
        session.assert_text_contains("Hello world!")
        session.assert_text_contains("Checking order 42.")
        session.assert_cached_tokens_at_least(60)
        session.assert_reasoning_tokens_under(20)
        session.assert_no_unused_interactions()
