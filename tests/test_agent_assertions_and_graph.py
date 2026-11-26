"""Tests for agent loop/recovery assertions and ASCII/Mermaid trajectory renderers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from rhinotape.cassette import Cassette
from rhinotape.graph import render_ascii_graph, render_mermaid_graph
from rhinotape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    TapeAssertionError,
    TapeMode,
    ToolCall,
    Usage,
)
from rhinotape.session import TapeSession, tape


def _make_interaction(
    *,
    url: str = "https://api.openai.com/v1/chat/completions",
    request_body: Any = None,
    status_code: int = 200,
    provider: str = "openai",
    model: str | None = "gpt-4o",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost: float = 0.001,
    latency_ms: float = 120.5,
    tool_calls: tuple[ToolCall, ...] = (),
    text: str = "",
    is_stream: bool = False,
    fingerprint: str = "fp",
) -> Interaction:
    return Interaction(
        request=RecordedRequest(
            method="POST",
            url=url,
            headers={"content-type": "application/json"},
            body=request_body,
            fingerprint=fingerprint,
            body_encoding="json" if isinstance(request_body, (dict, list)) else "empty",
        ),
        response=RecordedResponse(
            status_code=status_code,
            headers={"content-type": "application/json"},
            body={"ok": status_code < 400},
            body_encoding="json",
        ),
        latency_ms=latency_ms,
        llm=LLMCallSummary(
            provider=provider,
            model=model,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            estimated_cost_usd=cost,
            tool_calls=tool_calls,
            text=text,
            is_stream=is_stream,
        ),
    )


def _session_with_interactions(tmp_path: Path, interactions: list[Interaction]) -> TapeSession:
    session = TapeSession(tmp_path / "agent.tape.json", mode=TapeMode.NONE)
    session._interactions = list(interactions)
    return session


# --- assert_no_tool_loop ---------------------------------------------------


def test_assert_no_tool_loop_passes_within_limit(tmp_path: Path) -> None:
    interactions = [
        _make_interaction(tool_calls=(ToolCall("search", {"q": "rhino"}),)),
        _make_interaction(tool_calls=(ToolCall("search", {"q": "rhino"}),)),
        _make_interaction(tool_calls=(ToolCall("search", {"q": "tape"}),)),
        _make_interaction(tool_calls=(ToolCall("search", {"q": "rhino"}),)),
        _make_interaction(text="Done"),
    ]
    session = _session_with_interactions(tmp_path, interactions)
    session.assert_no_tool_loop(max_consecutive=2, include_args=True)

    # Empty / tool-free session also passes.
    empty = _session_with_interactions(tmp_path, [])
    empty.assert_no_tool_loop()


def test_assert_no_tool_loop_detects_loop_with_and_without_args(tmp_path: Path) -> None:
    # Same tool, same args (different dict insertion order) 3 times in a row.
    same_args_loop = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(tool_calls=(ToolCall("fetch", {"a": 1, "b": 2}),)),
            _make_interaction(tool_calls=(ToolCall("fetch", {"b": 2, "a": 1}),)),
            _make_interaction(tool_calls=(ToolCall("fetch", {"a": 1, "b": 2}),)),
            _make_interaction(tool_calls=(ToolCall("finish", {}),)),
        ],
    )
    with pytest.raises(TapeAssertionError, match=r"fetch.*3.*limit: 2"):
        same_args_loop.assert_no_tool_loop(max_consecutive=2, include_args=True)
    # Passes if limit raised to 3.
    same_args_loop.assert_no_tool_loop(max_consecutive=3, include_args=True)

    # Loop at the very end of the trajectory (4 consecutive calls).
    trailing_loop = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(tool_calls=(ToolCall("retry_me", {"id": 9}),)),
            _make_interaction(
                tool_calls=(
                    ToolCall("retry_me", {"id": 9}),
                    ToolCall("retry_me", {"id": 9}),
                    ToolCall("retry_me", {"id": 9}),
                )
            ),
        ],
    )
    with pytest.raises(TapeAssertionError, match=r"retry_me.*4.*limit: 2"):
        trailing_loop.assert_no_tool_loop(max_consecutive=2)

    # Same tool name with DIFFERENT args passes when include_args=True,
    # fails when include_args=False.
    diff_args = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(tool_calls=(ToolCall("search", {"page": 1}),)),
            _make_interaction(tool_calls=(ToolCall("search", {"page": 2}),)),
            _make_interaction(tool_calls=(ToolCall("search", {"page": 3}),)),
        ],
    )
    diff_args.assert_no_tool_loop(max_consecutive=2, include_args=True)
    with pytest.raises(TapeAssertionError, match=r"search.*3.*limit: 2"):
        diff_args.assert_no_tool_loop(max_consecutive=2, include_args=False)

    # max_consecutive=1 allows single calls, fails on 2 identical calls.
    two_same = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(tool_calls=(ToolCall("ping", {}),)),
            _make_interaction(tool_calls=(ToolCall("ping", {}),)),
        ],
    )
    two_same.assert_no_tool_loop(max_consecutive=2)
    with pytest.raises(TapeAssertionError, match=r"ping.*2.*limit: 1"):
        two_same.assert_no_tool_loop(max_consecutive=1)


def test_assert_no_tool_loop_invalid_max_consecutive(tmp_path: Path) -> None:
    session = _session_with_interactions(tmp_path, [])
    for bad in (0, -1, -5, True, False):
        with pytest.raises(ValueError, match="max_consecutive"):
            session.assert_no_tool_loop(max_consecutive=bad)


# --- assert_tool_recovery --------------------------------------------------


def test_assert_tool_recovery_http_error_and_unrecovered(tmp_path: Path) -> None:
    # HTTP 500 followed by 200 recovers.
    recovered = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(status_code=500),
            _make_interaction(status_code=200, text="Recovered!"),
        ],
    )
    recovered.assert_tool_recovery(require_error=True)
    with pytest.raises(TapeAssertionError, match=r"required: >= 2"):
        recovered.assert_tool_recovery(min_recovery_calls=2, require_error=True)

    # Trailing HTTP 503 after a 200 fails recovery.
    trailing_err = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(status_code=500),
            _make_interaction(status_code=200),
            _make_interaction(status_code=503),
        ],
    )
    with pytest.raises(TapeAssertionError, match=r"call\[2\].*0 successful"):
        trailing_err.assert_tool_recovery()

    # No error observed: passes when require_error=False, fails when require_error=True.
    clean = _session_with_interactions(tmp_path, [_make_interaction(status_code=200, text="OK")])
    clean.assert_tool_recovery(require_error=False)
    with pytest.raises(TapeAssertionError, match="none was observed"):
        clean.assert_tool_recovery(require_error=True)


def test_assert_tool_recovery_anthropic_tool_result_is_error(tmp_path: Path) -> None:
    turn_1_req = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Run SQL"}],
    }
    turn_2_req = {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": "Run SQL"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu_1", "name": "sql", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "is_error": True,
                        "content": "syntax error near SELECT",
                    }
                ],
            },
        ],
    }
    # Cumulative history on turn 3 carries the old error BEFORE the new assistant turn,
    # and the latest tool_result succeeded (is_error=False).
    turn_3_req = {
        "model": "claude-sonnet-4-6",
        "messages": [
            *turn_2_req["messages"],
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu_2", "name": "sql", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_2",
                        "is_error": False,
                        "content": "rows: 5",
                    }
                ],
            },
        ],
    }

    # Unrecovered (stops right on the error turn).
    unrecovered = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(request_body=turn_1_req, provider="anthropic"),
            _make_interaction(request_body=turn_2_req, provider="anthropic"),
        ],
    )
    with pytest.raises(TapeAssertionError, match=r"call\[1\].*0 successful"):
        unrecovered.assert_tool_recovery(require_error=True)

    # Recovered on turn 3 even with cumulative message history!
    recovered = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(request_body=turn_1_req, provider="anthropic"),
            _make_interaction(request_body=turn_2_req, provider="anthropic"),
            _make_interaction(request_body=turn_3_req, provider="anthropic", text="Found 5 rows."),
        ],
    )
    recovered.assert_tool_recovery(min_recovery_calls=1, require_error=True)

    # Inline 2-turn recovery: turn 2 gets tool error and answers gracefully (200, no pending tools).
    inline_recovered = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(
                request_body=turn_1_req,
                provider="anthropic",
                tool_calls=(ToolCall("sql", {"q": "SELECT *"}),),
            ),
            _make_interaction(
                request_body=turn_2_req,
                provider="anthropic",
                text="Sorry, the SQL query had a syntax error; please check the column name.",
            ),
        ],
    )
    inline_recovered.assert_tool_recovery(min_recovery_calls=1, require_error=True)

    # If turn 2 returns a fallback tool call without a follow-up turn, recovery is NOT complete.
    dangling_fallback = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(request_body=turn_1_req, provider="anthropic"),
            _make_interaction(
                request_body=turn_2_req,
                provider="anthropic",
                text="Retrying SQL...",
                tool_calls=(ToolCall("sql", {"q": "SELECT 1"}),),
            ),
        ],
    )
    with pytest.raises(TapeAssertionError, match=r"call\[1\].*0 successful"):
        dangling_fallback.assert_tool_recovery(require_error=True)


def test_assert_tool_recovery_openai_tool_error_variants(tmp_path: Path) -> None:
    # Variant 1: OpenAI role='tool' with JSON {"error": "..."}
    req_json_err = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Fetch invoice"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"error": "rate_limit"}'},
        ],
    }
    req_json_ok = {
        "model": "gpt-4o",
        "messages": [
            *req_json_err["messages"],
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c2"}]},
            {"role": "tool", "tool_call_id": "c2", "content": '{"error": "", "total": 42}'},
        ],
    }
    s_json = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(request_body=req_json_err),
            _make_interaction(request_body=req_json_ok, text="Invoice total is $42."),
        ],
    )
    s_json.assert_tool_recovery(require_error=True)

    # Variant 2: OpenAI role='tool' starting with "Error:"
    req_prefix_err = {
        "model": "gpt-4o",
        "messages": [
            {"role": "tool", "tool_call_id": "c1", "content": "Error: connection refused"},
        ],
    }
    req_prefix_ok = {
        "model": "gpt-4o",
        "messages": [
            {"role": "tool", "tool_call_id": "c2", "content": "Connected successfully"},
        ],
    }
    s_prefix_fail = _session_with_interactions(
        tmp_path, [_make_interaction(request_body=req_prefix_err)]
    )
    with pytest.raises(TapeAssertionError, match=r"call\[0\]"):
        s_prefix_fail.assert_tool_recovery(require_error=True)

    s_prefix_ok = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(request_body=req_prefix_err),
            _make_interaction(request_body=req_prefix_ok),
            _make_interaction(request_body={"messages": [{"role": "user", "content": "summary"}]}),
        ],
    )
    s_prefix_ok.assert_tool_recovery(min_recovery_calls=2, require_error=True)

    # Variant 3: OpenAI tool message with empty/null error is NOT treated as an error turn.
    s_no_err = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(
                request_body={
                    "messages": [
                        {"role": "tool", "content": '{"error": null, "ok": true}'},
                        {"role": "tool", "content": '{"error": "   ", "ok": true}'},
                        {"role": "tool", "content": "not json"},
                    ]
                }
            )
        ],
    )
    with pytest.raises(TapeAssertionError, match="none was observed"):
        s_no_err.assert_tool_recovery(require_error=True)

    # Variant 4: OpenAI Responses API function_call_output error & recovery.
    s_responses = _session_with_interactions(
        tmp_path,
        [
            _make_interaction(
                request_body={
                    "input": [
                        {"type": "function_call", "name": "lookup"},
                        {"type": "function_call_output", "output": "Error: timeout"},
                    ]
                }
            ),
            _make_interaction(
                request_body={
                    "input": [
                        {"type": "function_call", "name": "lookup"},
                        {"type": "function_call_output", "output": "Error: timeout"},
                        {"type": "function_call", "name": "lookup"},
                        {"type": "function_call_output", "output": '{"ok": true}'},
                    ]
                }
            ),
        ],
    )
    s_responses.assert_tool_recovery(require_error=True)


def test_assert_tool_recovery_invalid_min_recovery_calls(tmp_path: Path) -> None:
    session = _session_with_interactions(tmp_path, [])
    for bad in (0, -2, True, False):
        with pytest.raises(ValueError, match="min_recovery_calls"):
            session.assert_tool_recovery(min_recovery_calls=bad)


def test_agent_assertions_end_to_end_via_httpx_tape(tmp_path: Path) -> None:
    """Exercise both assertions through a real recorded TapeSession."""
    calls = {"n": 0}

    def _handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        if calls["n"] == 2:
            return httpx.Response(
                200,
                json={
                    "id": "c2",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "Tokyo"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "c3",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "Sunny in Tokyo."}}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 10},
            },
        )

    cassette_path = tmp_path / "e2e_recovery.tape.json"
    with tape(cassette_path, mode=TapeMode.RECORD) as session:
        with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "weather"}]},
            )
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "weather retry"}],
                },
            )
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": "weather retry"},
                        {"role": "assistant", "content": ""},
                        {"role": "tool", "content": '{"temp_c": 24}'},
                    ],
                },
            )
        session.assert_no_tool_loop(max_consecutive=2)
        session.assert_tool_recovery(min_recovery_calls=2, require_error=True)


# --- render_ascii_graph & render_mermaid_graph -----------------------------


def test_render_graphs_empty_cassette() -> None:
    empty = Cassette(path=Path("empty.tape.json"), interactions=[])

    ascii_out = render_ascii_graph(empty)
    assert "empty.tape.json" in ascii_out
    assert "0 calls" in ascii_out
    assert "empty cassette" in ascii_out

    mermaid_out = render_mermaid_graph(empty)
    assert mermaid_out.startswith("flowchart TD")
    assert 'EMPTY["No interactions recorded"]' in mermaid_out


def test_render_graphs_single_call() -> None:
    cassette = Cassette(
        path=Path("single.tape.json"),
        interactions=[
            _make_interaction(
                provider="anthropic",
                model="claude-sonnet-4-6",
                status_code=200,
                input_tokens=300,
                output_tokens=100,
                cost=0.0024,
                latency_ms=215.3,
                text="Hello from Claude!",
                is_stream=True,
            )
        ],
    )

    ascii_out = render_ascii_graph(cassette)
    assert "1 call |" in ascii_out
    assert (
        "[1] anthropic/claude-sonnet-4-6 | 200 | 215.3ms | 400 tok | $0.002400 [SSE]" in ascii_out
    )
    assert "`-- text: Hello from Claude!" in ascii_out
    assert "\n  |\n  v\n" not in ascii_out

    mermaid_out = render_mermaid_graph(cassette)
    assert mermaid_out.startswith("flowchart TD\n")
    assert "Call 1: anthropic/claude-sonnet-4-6" in mermaid_out
    assert "400 tok" in mermaid_out
    assert "SSE" in mermaid_out
    assert "Hello from Claude!" in mermaid_out
    assert "-->" not in mermaid_out


def test_render_graphs_multi_turn_tools_loops_and_errors() -> None:
    cassette = Cassette(
        path=Path("agent_loop.tape.json"),
        interactions=[
            _make_interaction(
                provider="openai",
                model="gpt-4o",
                status_code=200,
                latency_ms=100.0,
                input_tokens=100,
                output_tokens=25,
                cost=0.0005,
                is_stream=True,
                tool_calls=(
                    ToolCall("search", {"query": "rhino"}),
                    ToolCall("search", {"query": "rhino"}),
                ),
            ),
            _make_interaction(
                provider="openai",
                model="gpt-4o",
                status_code=500,
                latency_ms=45.0,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
            ),
            _make_interaction(
                provider="openai",
                model="gpt-4o",
                status_code=200,
                latency_ms=150.0,
                input_tokens=200,
                output_tokens=40,
                cost=0.001,
                tool_calls=(
                    ToolCall("search", {"query": "rhino"}),
                    ToolCall("search", {"query": "rhino"}),
                ),
            ),
            _make_interaction(
                provider="openai",
                model=None,
                status_code=200,
                latency_ms=float("nan"),
                input_tokens=50,
                output_tokens=20,
                cost=float("nan"),
                tool_calls=(ToolCall("finish", {}),),
                text="Final answer with   extra\nwhitespace.",
            ),
        ],
    )

    ascii_out = render_ascii_graph(cassette)
    assert "4 calls" in ascii_out
    assert "[1] openai/gpt-4o | 200 | 100.0ms | 125 tok | $0.000500 [SSE]" in ascii_out
    assert "[2] openai/gpt-4o | 500 | 45.0ms | 0 tok | $0.000000 [ERR 500]" in ascii_out
    assert "[LOOP x2]" not in ascii_out
    assert '|-- tool: search({"query":"rhino"}) [LOOP x3]' in ascii_out
    assert '`-- tool: search({"query":"rhino"}) [LOOP x4]' in ascii_out
    assert "[4] openai | 200 | n/a | 70 tok | $n/a" in ascii_out
    assert "|-- tool: finish()" in ascii_out
    assert "`-- text: Final answer with extra whitespace." in ascii_out
    assert ascii_out.count("  |\n  v") == 3

    mermaid_out = render_mermaid_graph(cassette)
    assert mermaid_out.startswith("flowchart TD\n")
    assert "C1 --> T1_1" in mermaid_out
    assert "C1 --> T1_2" in mermaid_out
    assert "T1_1 --> C2" in mermaid_out
    assert "T1_2 --> C2" in mermaid_out
    assert "C2 --> C3" in mermaid_out
    assert "C3 --> T3_1" in mermaid_out
    assert "C3 --> T3_2" in mermaid_out
    assert "T3_1 --> C4" in mermaid_out
    assert "T3_2 --> C4" in mermaid_out
    assert "C4 --> T4_1" in mermaid_out
    assert "ERR 500" in mermaid_out
    assert "LOOP x3" in mermaid_out
    assert "LOOP x4" in mermaid_out
    assert "class C2 err" in mermaid_out
    assert "class T3_1,T3_2 loop" in mermaid_out


def test_render_mermaid_graph_sanitizes_special_characters() -> None:
    nasty_text = 'Line 1 "quoted"\nLine 2 [brackets] {braces} <script>alert(1)</script>; `code` #99'
    nasty_args = {
        "expr": 'a[0] == "x"\n&& b{1} < c > d; `cmd`',
        "long": "x" * 200,
    }
    cassette = Cassette(
        path=Path("special.tape.json"),
        interactions=[
            _make_interaction(
                model='gpt-"4o"[beta]\nv2',
                tool_calls=(
                    ToolCall('run["cmd"]\n;x', nasty_args),
                    ToolCall("raw_str_tool", 'raw "arg"\n[1] {2}'),
                ),
                text=nasty_text,
            )
        ],
    )

    mermaid_out = render_mermaid_graph(cassette)
    lines = mermaid_out.splitlines()
    assert lines[0] == "flowchart TD"

    # Validate every node declaration has balanced outer delimiters and no unsafe inner chars.
    node_pattern = re.compile(
        r'^\s+(?P<id>[A-Z][A-Z0-9_]*)(?P<open>\["|\(\[")(?P<label>.*)(?P<close>"\]|"\)\])'
    )
    checked_nodes = 0
    for line in lines[1:]:
        if "-->" in line or line.strip().startswith(("classDef ", "class ")):
            continue
        match = node_pattern.match(line)
        assert match is not None, f"Invalid Mermaid node line: {line!r}"
        label = match.group("label")
        assert label, f"Empty Mermaid label in line: {line!r}"
        for forbidden in ('"', "[", "]", "{", "}", "<", ">", "`", ";", "\n", "\r", "#"):
            assert forbidden not in label, f"Forbidden char {forbidden!r} in label: {label!r}"
        checked_nodes += 1

    assert checked_nodes == 3  # C1 + T1_1 + T1_2
