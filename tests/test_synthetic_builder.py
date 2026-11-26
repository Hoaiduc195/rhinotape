"""End-to-end tests for the fluent CassetteBuilder in rhinotape.mock."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from rhinotape import Cassette, CassetteBuilder, MemoryCassetteStore, TapeMode, TapeSession


def test_openai_json_and_sse_replay_with_tools_and_usage(tmp_path: Path) -> None:
    cassette_path = tmp_path / "openai_flow.tape.json"
    saved = (
        CassetteBuilder(cassette_path, metadata={"suite": "openai"})
        .add_openai_chat(
            "What is the weather in Tokyo?",
            model="gpt-4o",
            tool_calls=[
                {"name": "get_weather", "arguments": {"city": "Tokyo"}, "id": "call_w1"},
            ],
            input_tokens=40,
            output_tokens=15,
            cached_input_tokens=20,
            reasoning_tokens=5,
            extra_request_body={"temperature": 0.0},
        )
        .add_openai_chat(
            [
                {"role": "user", "content": "What is the weather in Tokyo?"},
                {"role": "tool", "tool_call_id": "call_w1", "content": "19C sunny"},
            ],
            reply="It is 19C and sunny in Tokyo.",
            model="gpt-4o",
            input_tokens=60,
            output_tokens=12,
            stream=True,
        )
        .save()
    )
    assert saved == cassette_path
    loaded = Cassette.load(cassette_path)
    assert loaded.validate() == []

    with TapeSession(cassette_path, mode=TapeMode.REPLAY) as session:
        with httpx.Client() as client:
            r1 = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
                    "temperature": 0.0,
                },
            )
            assert r1.status_code == 200
            assert r1.json()["choices"][0]["message"]["tool_calls"][0]["id"] == "call_w1"

            with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": "What is the weather in Tokyo?"},
                        {"role": "tool", "tool_call_id": "call_w1", "content": "19C sunny"},
                    ],
                    "stream": True,
                },
            ) as r2:
                streamed_raw = "".join(r2.iter_text())
                assert "data: [DONE]" in streamed_raw

        session.assert_call_count(2)
        session.assert_no_errors()
        session.assert_provider("openai")
        session.assert_model("gpt-4o")
        session.assert_tool_sequence(["get_weather"])
        session.assert_tool_args("get_weather", {"city": "Tokyo"}, exact=True)
        session.assert_text_contains("19C and sunny in Tokyo")
        session.assert_cached_tokens_at_least(20)
        assert session.total_usage.input_tokens == 100
        assert session.total_usage.output_tokens == 27
        assert session.total_usage.reasoning_tokens == 5
        assert session.total_cost_usd > 0.0


def test_anthropic_json_and_sse_replay_with_thinking_cache_and_tools(tmp_path: Path) -> None:
    cassette_path = tmp_path / "anthropic_flow.tape.json.gz"
    builder = (
        CassetteBuilder(recorded_at="2026-06-01T00:00:00Z")
        .add_anthropic_message(
            "Plan a trip to Kyoto.",
            reply="Let me check train schedules.",
            model="claude-sonnet-4",
            system="You are a travel planner.",
            thinking="User wants Kyoto itinerary; check Shinkansen first.",
            tool_calls=[{"name": "search_trains", "arguments": {"to": "Kyoto"}}],
            input_tokens=120,
            output_tokens=35,
            cached_input_tokens=50,
            cache_creation_tokens=20,
            stream=False,
        )
        .add_anthropic_message(
            "Summarize the train options.",
            reply="Nozomi departs every 10 minutes.",
            model="claude-sonnet-4",
            thinking="Formatting concise schedule.",
            tool_calls=[{"name": "book_seat", "arguments": {"train": "Nozomi"}, "id": "toolu_99"}],
            input_tokens=90,
            output_tokens=25,
            cached_input_tokens=40,
            cache_creation_tokens=10,
            stream=True,
        )
    )
    saved = builder.save(cassette_path)
    assert saved == cassette_path
    assert cassette_path.read_bytes()[:2] == b"\x1f\x8b"
    assert Cassette.load(cassette_path).validate() == []

    with TapeSession(cassette_path, mode=TapeMode.REPLAY) as session:
        with httpx.Client() as client:
            r1 = client.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": "claude-sonnet-4",
                    "system": "You are a travel planner.",
                    "messages": [{"role": "user", "content": "Plan a trip to Kyoto."}],
                },
            )
            assert r1.status_code == 200
            assert r1.json()["type"] == "message"

            with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": "claude-sonnet-4",
                    "messages": [{"role": "user", "content": "Summarize the train options."}],
                    "stream": True,
                },
            ) as r2:
                sse_body = "".join(r2.iter_text())
                assert "event: message_start" in sse_body
                assert "event: content_block_delta" in sse_body
                assert "event: message_stop" in sse_body

        session.assert_call_count(2)
        session.assert_provider("anthropic")
        session.assert_model("claude-sonnet-4")
        session.assert_thinking_present()
        session.assert_tool_sequence(["search_trains", "book_seat"])
        session.assert_tool_args("book_seat", {"train": "Nozomi"}, exact=True)
        session.assert_text_contains("Nozomi departs")
        assert session.total_usage.input_tokens == 210
        assert session.total_usage.output_tokens == 60
        assert session.total_usage.cached_input_tokens == 90
        assert session.total_usage.cache_creation_tokens == 30
        assert session.total_cost_usd > 0.0


def test_gemini_json_and_sse_replay_with_function_call() -> None:
    store = MemoryCassetteStore()
    cassette = (
        CassetteBuilder("mem_gemini.tape.json")
        .add_gemini_generate(
            "Find flights to Seoul",
            model="gemini-2.5-flash",
            tool_calls=[{"name": "find_flights", "arguments": {"dest": "ICN"}}],
            input_tokens=30,
            output_tokens=10,
            cached_input_tokens=12,
            stream=False,
        )
        .add_gemini_generate(
            "Give me the cheapest option",
            reply="Flight KE706 is the cheapest nonstop flight.",
            model="gemini-2.5-flash",
            input_tokens=45,
            output_tokens=18,
            cached_input_tokens=15,
            stream=True,
        )
        .build()
    )
    assert cassette.validate() == []
    store.save("mem_gemini.tape.json", cassette)

    with TapeSession("mem_gemini.tape.json", mode=TapeMode.REPLAY, store=store) as session:
        with httpx.Client() as client:
            r1 = client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                json={"contents": [{"role": "user", "parts": [{"text": "Find flights to Seoul"}]}]},
            )
            assert r1.status_code == 200
            assert "candidates" in r1.json()

            with client.stream(
                "POST",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": "Give me the cheapest option"}]}
                    ]
                },
            ) as r2:
                sse = "".join(r2.iter_text())
                assert "KE706" in sse

        session.assert_call_count(2)
        session.assert_provider("gemini")
        session.assert_model("gemini-2.5-flash")
        session.assert_tool_sequence(["find_flights"])
        session.assert_tool_args("find_flights", {"dest": "ICN"}, exact=True)
        session.assert_text_contains("Flight KE706")
        assert session.total_usage.input_tokens == 75
        assert session.total_usage.output_tokens == 28
        assert session.total_usage.cached_input_tokens == 27


def test_error_injection_retry_and_build_immutability() -> None:
    store = MemoryCassetteStore()
    url = "https://api.openai.com/v1/chat/completions"
    req_payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Ping"}]}

    builder = CassetteBuilder("retry.tape.json", metadata={"team": "core"})
    builder.add_error(
        url,
        status_code=429,
        message="Rate limit exceeded",
        request_body=req_payload,
        headers={"retry-after": "1"},
        latency_ms=8.5,
    )
    snap_one = builder.build()
    builder.add_openai_chat("Ping", reply="Pong!", model="gpt-4o")
    snap_two = builder.build()

    # Mutating snap_one does not affect snap_two or builder
    snap_one.metadata["team"] = "mutated"
    snap_one.interactions.clear()
    assert len(snap_two.interactions) == 2
    assert snap_two.metadata["team"] == "core"
    assert snap_two.validate() == []

    store.save("retry.tape.json", snap_two)
    with TapeSession("retry.tape.json", mode=TapeMode.REPLAY, store=store) as session:
        with httpx.Client() as client:
            first = client.post(url, json=req_payload)
            assert first.status_code == 429
            assert first.headers["retry-after"] == "1"
            assert "Rate limit exceeded" in first.json()["error"]["message"]

            second = client.post(url, json=req_payload)
            assert second.status_code == 200

        session.assert_status_codes([429, 200])
        session.assert_text_contains("Pong!")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"input_tokens": -1}, "input_tokens"),
        ({"output_tokens": True}, "output_tokens"),
        ({"cached_input_tokens": -5}, "cached_input_tokens"),
        ({"reasoning_tokens": -1}, "reasoning_tokens"),
        ({"latency_ms": -0.1}, "latency_ms"),
        ({"latency_ms": float("nan")}, "latency_ms"),
        ({"latency_ms": float("inf")}, "latency_ms"),
        ({"latency_ms": False}, "latency_ms"),
        ({"status_code": 99}, "status_code"),
        ({"status_code": 600}, "status_code"),
        ({"status_code": True}, "status_code"),
        ({"model": "   "}, "model"),
        ({"tool_calls": [{"arguments": {}}]}, "name"),
    ],
)
def test_invalid_arguments_raise_value_error(kwargs: dict[str, Any], match: str) -> None:
    builder = CassetteBuilder()
    with pytest.raises(ValueError, match=match):
        builder.add_openai_chat("hello", **kwargs)


def test_invalid_error_and_prompt_arguments() -> None:
    builder = CassetteBuilder()
    with pytest.raises(ValueError, match="status_code"):
        builder.add_error("https://api.openai.com/v1/chat/completions", status_code=200)
    with pytest.raises(ValueError, match="url"):
        builder.add_error("   ", status_code=500)
    with pytest.raises(ValueError, match="method"):
        builder.add_error("https://api.openai.com/v1/chat/completions", method="")
    with pytest.raises(ValueError, match="prompt"):
        builder.add_openai_chat([])
    with pytest.raises(ValueError, match="cache_creation_tokens"):
        builder.add_anthropic_message("hi", cache_creation_tokens=-2)


def test_empty_reply_multiple_tools_and_string_tool_args() -> None:
    """Verify tool-only turns, multiple tool_calls, JSON-str & raw-str args across providers."""
    tools = [
        {"name": "query_sql", "arguments": '{"table": "users", "limit": 5}'},
        {"name": "raw_shell", "arguments": "ls -la /tmp"},
        {"name": "ping", "arguments": ""},
    ]
    cassette = (
        CassetteBuilder("multi_tools.tape.json")
        .add_openai_chat("run tools", reply=None, tool_calls=tools, stream=False)
        .add_openai_chat("run tools sse", reply="", tool_calls=tools, stream=True)
        .add_anthropic_message("run tools", reply=None, tool_calls=tools, stream=False)
        .add_anthropic_message("run tools sse", reply="", tool_calls=tools, stream=True)
        .add_gemini_generate("run tools", reply=None, tool_calls=tools, stream=False)
        .add_gemini_generate("run tools sse", reply="", tool_calls=tools, stream=True)
        .build()
    )
    assert cassette.validate() == []
    assert len(cassette.interactions) == 6

    for inter in cassette.interactions:
        assert inter.llm.text == ""
        assert len(inter.llm.tool_calls) == 3
        assert inter.llm.tool_calls[0].name == "query_sql"
        assert inter.llm.tool_calls[0].arguments == {"table": "users", "limit": 5}
        assert inter.llm.tool_calls[1].name == "raw_shell"
        assert inter.llm.tool_calls[1].arguments == "ls -la /tmp"
        assert inter.llm.tool_calls[2].name == "ping"
        assert inter.llm.tool_calls[2].arguments == {}


def test_error_status_codes_on_provider_builders_emit_error_payloads() -> None:
    """4xx/5xx status_code on provider methods must emit error JSON and 0 tokens/cost."""
    store = MemoryCassetteStore()
    cassette = (
        CassetteBuilder("provider_errors.tape.json")
        .add_openai_chat(
            "hi",
            reply="Quota exceeded",
            status_code=429,
            input_tokens=500,
            output_tokens=200,
            stream=True,
        )
        .add_anthropic_message(
            "hi",
            reply="Overloaded",
            status_code=529,
            input_tokens=500,
            output_tokens=200,
        )
        .add_gemini_generate(
            "hi",
            reply="Backend unavailable",
            status_code=503,
            input_tokens=500,
            output_tokens=200,
        )
        .build()
    )
    assert cassette.validate() == []
    for inter in cassette.interactions:
        assert inter.response.status_code >= 400
        assert inter.response.stream_chunks is None
        assert inter.response.body_encoding == "json"
        assert "error" in inter.response.body
        assert inter.llm.usage.total_tokens == 0
        assert inter.llm.estimated_cost_usd == 0.0
        assert inter.llm.text == ""
        assert inter.llm.tool_calls == ()

    store.save("provider_errors.tape.json", cassette)
    with TapeSession("provider_errors.tape.json", mode=TapeMode.REPLAY, store=store) as session:
        with httpx.Client() as client:
            with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            ) as r1:
                r1.read()
                assert r1.status_code == 429
                assert r1.json()["error"]["message"] == "Quota exceeded"

            r2 = client.post(
                "https://api.anthropic.com/v1/messages",
                json={"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r2.status_code == 529
            assert r2.json()["error"]["message"] == "Overloaded"

            r3 = client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            )
            assert r3.status_code == 503
            assert r3.json()["error"]["message"] == "Backend unavailable"

        session.assert_status_codes([429, 529, 503])
        assert session.total_tokens == 0
        assert session.total_cost_usd == 0.0


def test_cache_and_reasoning_overflow_extra_body_isolation_and_gemini_url_escaping(
    tmp_path: Path,
) -> None:
    shared_extra: dict[str, Any] = {"stream": False, "nested": {"tags": ["v1"]}}
    cassette = (
        CassetteBuilder("edge_cases.tape.json")
        # cached_input_tokens > input_tokens and reasoning_tokens > output_tokens
        .add_openai_chat(
            "q1",
            reply="a1",
            thinking="step-by-step",
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=80,
            reasoning_tokens=30,
            stream=True,
            extra_request_body=shared_extra,
        )
        # cached + cache_creation > input_tokens
        .add_anthropic_message(
            "q2",
            reply="a2",
            input_tokens=10,
            output_tokens=12,
            cached_input_tokens=40,
            cache_creation_tokens=25,
            stream=True,
            extra_request_body=shared_extra,
        )
        # Gemini thinking + reasoning_tokens + special chars / models/ prefix in model
        .add_gemini_generate(
            "q3",
            reply="a3",
            model="models/gemini-2.5-flash",
            thinking="gemini thought",
            input_tokens=10,
            output_tokens=15,
            cached_input_tokens=50,
            reasoning_tokens=6,
            stream=True,
        )
        .add_gemini_generate(
            "q4",
            reply="a4",
            model="gemini-exp#v1?beta=1",
        )
        .build()
    )
    # Mutating caller's dict after build does not affect recorded request bodies.
    shared_extra["nested"]["tags"].append("mutated")
    assert cassette.validate() == []

    oai, ant, gem_sse, gem_special = cassette.interactions
    # stream=True must NOT be clobbered by extra_request_body={"stream": False}
    assert oai.request.body["stream"] is True
    assert oai.request.body["nested"]["tags"] == ["v1"]
    assert oai.llm.thinking == "step-by-step"
    assert oai.llm.usage.input_tokens == 80
    assert oai.llm.usage.cached_input_tokens == 80
    assert oai.llm.usage.output_tokens == 30
    assert oai.llm.usage.reasoning_tokens == 30
    assert oai.llm.estimated_cost_usd > 0.0

    assert ant.request.body["stream"] is True
    assert ant.llm.usage.input_tokens == 65
    assert ant.llm.usage.cached_input_tokens == 40
    assert ant.llm.usage.cache_creation_tokens == 25
    assert ant.llm.usage.output_tokens == 12
    assert ant.llm.estimated_cost_usd > 0.0

    assert "/models/models/" not in gem_sse.request.url
    assert gem_sse.request.url.endswith("/models/gemini-2.5-flash:streamGenerateContent?alt=sse")
    assert gem_sse.llm.thinking == "gemini thought"
    assert gem_sse.llm.usage.input_tokens == 50
    assert gem_sse.llm.usage.cached_input_tokens == 50
    assert gem_sse.llm.usage.output_tokens == 15
    assert gem_sse.llm.usage.reasoning_tokens == 6
    assert gem_sse.llm.estimated_cost_usd > 0.0

    # Special chars in Gemini model did not hijack URL query/fragment
    assert ":generateContent" in gem_special.request.url
    assert "#" not in gem_special.request.url
    assert gem_special.llm.model == "gemini-exp#v1?beta=1"

    # Explicit compress=True on plain .tape.json path and compress=False on .gz path
    builder = CassetteBuilder(tmp_path / "plain_compressed.tape.json").add_openai_chat(
        "hi", reply="hello"
    )
    p_comp = builder.save(compress=True)
    assert p_comp.read_bytes()[:2] == b"\x1f\x8b"
    assert Cassette.load(p_comp).interactions[0].llm.text == "hello"

    p_uncomp = builder.save(tmp_path / "uncompressed.tape.json.gz", compress=False)
    assert p_uncomp.read_bytes()[:2] != b"\x1f\x8b"


def test_additional_validation_edge_cases() -> None:
    builder = CassetteBuilder()
    with pytest.raises(ValueError, match="url"):
        builder.add_openai_chat("hi", url="not-a-valid-url")
    with pytest.raises(ValueError, match="url"):
        builder.add_anthropic_message("hi", url="   ")
    with pytest.raises(ValueError, match="url"):
        builder.add_gemini_generate("hi", url="ftp://example.com/v1")
    with pytest.raises(ValueError, match="model"):
        builder.add_gemini_generate("hi", model="models/")
    with pytest.raises(ValueError, match="tool_calls"):
        builder.add_openai_chat("hi", tool_calls={"name": "oops"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="arguments"):
        builder.add_openai_chat("hi", tool_calls=[{"name": "oops", "arguments": [1, 2]}])
    with pytest.raises(ValueError, match="extra_request_body"):
        builder.add_openai_chat("hi", extra_request_body=["bad"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="path"):
        CassetteBuilder("   ")
