"""End-to-end Google Gemini & Vertex AI wire-protocol integration tests (Wave 95)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx

from llm_tape import TapeMode, tape


def test_gemini_and_vertex_generate_and_stream_roundtrip(tmp_path: Path) -> None:
    cassette_path = tmp_path / "gemini_vertex.tape.json"

    class _GeminiSSE(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield (b'data: {"candidates":[{"content":{"parts":[{"text":"Vertex "}]}}]}\n\n')
            yield (
                b'data: {"candidates":[{"content":{"parts":[{"text":"reply"}]}}],'
                b'"usageMetadata":{"promptTokenCount":40,"candidatesTokenCount":15,'
                b'"cachedContentTokenCount":20}}\n\n'
            )

    def _handler(req: httpx.Request) -> httpx.Response:
        if "streamGenerateContent" in str(req.url):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_GeminiSSE(),
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "search_docs", "args": {"q": "tape"}}},
                                {"text": "Searching docs."},
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 30,
                    "candidatesTokenCount": 10,
                },
            },
        )

    url_patterns = [r"(?<=/projects/)[^/]+"]
    with tape(cassette_path, mode=TapeMode.RECORD, redact_url_patterns=url_patterns):
        with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
            r1 = client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-2.5-flash:generateContent?key=AIzaSecretKey123456789012345678901234567",
                json={"contents": [{"parts": [{"text": "find docs"}]}]},
            )
            assert r1.status_code == 200
            with client.stream(
                "POST",
                "https://us-central1-aiplatform.googleapis.com/v1/projects/my-secret-gcp-proj/"
                "locations/us-central1/publishers/google/models/"
                "gemini-2.5-pro:streamGenerateContent?alt=sse",
                json={"contents": [{"parts": [{"text": "stream me"}]}]},
            ) as r2:
                assert b"Vertex " in r2.read()

    raw_disk = cassette_path.read_text(encoding="utf-8")
    assert "AIzaSecretKey" not in raw_disk
    assert "my-secret-gcp-proj" not in raw_disk

    with tape(cassette_path, mode=TapeMode.REPLAY, redact_url_patterns=url_patterns) as session:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            r1 = client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-2.5-flash:generateContent?key=AIzaDifferentKey123456789012345678901234",
                json={"contents": [{"parts": [{"text": "find docs"}]}]},
            )
            assert r1.status_code == 200
            with client.stream(
                "POST",
                "https://us-central1-aiplatform.googleapis.com/v1/projects/another-gcp-proj/"
                "locations/us-central1/publishers/google/models/"
                "gemini-2.5-pro:streamGenerateContent?alt=sse",
                json={"contents": [{"parts": [{"text": "stream me"}]}]},
            ) as r2:
                assert b"reply" in r2.read()
        session.assert_call_count(2)
        session.assert_provider("gemini")
        session.assert_tool_called("search_docs")
        session.assert_tool_args("search_docs", {"q": "tape"})
        session.assert_text_contains("Vertex reply")
        session.assert_cached_tokens_at_least(20)
        session.assert_no_unused_interactions()
