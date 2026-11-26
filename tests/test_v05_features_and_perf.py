"""Tests for match_fuzzy_text, Markdown diff, and scrubbing optimizations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from rhinotape import (
    PII_PRESETS,
    Cassette,
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    TapeMode,
    TapeSession,
    ToolCall,
    Usage,
    match_all,
    match_fuzzy_text,
    match_method,
    match_path,
    normalize_json_array_order,
    redact_body,
    redact_text,
)
from rhinotape.cli import main as cli_main
from rhinotape.scrubbing import (
    _combine_patterns,
    compile_patterns,
    fingerprint_request,
    strip_ignored_fields,
)


def _req(body: Any, url: str = "https://api.openai.com/v1/chat/completions") -> RecordedRequest:
    return RecordedRequest(method="POST", url=url, headers={}, body=body, fingerprint="fp")


def _make_cassette(
    path: Path,
    *,
    text: str = "Hello world",
    cost: float = 0.002,
    tokens_in: int = 100,
    tokens_out: int = 50,
    status_code: int = 200,
    tools: list[ToolCall] | None = None,
    recorded_at: str = "2026-11-26T00:00:00Z",
) -> Cassette:
    inter = Interaction(
        request=RecordedRequest(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            fingerprint=f"fp-{path.stem}-{status_code}",
        ),
        response=RecordedResponse(
            status_code=status_code,
            headers={"content-type": "application/json"},
            body={"choices": [{"message": {"role": "assistant", "content": text}}]},
        ),
        latency_ms=42.5,
        llm=LLMCallSummary(
            provider="openai",
            model="gpt-4o-mini",
            usage=Usage(
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                cached_input_tokens=20,
                cache_creation_tokens=10,
                reasoning_tokens=5,
            ),
            estimated_cost_usd=cost,
            text=text,
            tool_calls=tuple(tools or ()),
        ),
    )
    c = Cassette(path=path, interactions=[inter], recorded_at=recorded_at)
    c.save()
    return c


def test_match_fuzzy_text_dot_path_auto_extract_and_session_replay(tmp_path: Path) -> None:
    # 1. Validation of threshold bounds
    for bad in (-0.1, 1.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="threshold"):
            match_fuzzy_text(threshold=bad)  # type: ignore[arg-type]

    # 2. Explicit dot-path with whitespace/case normalization
    m_path = match_fuzzy_text("messages.0.content", threshold=0.8)
    r_rec = _req({"messages": [{"role": "user", "content": "Summarize   the Q4   report!"}]})
    r_inc = _req({"messages": [{"role": "user", "content": "summarize the q4 report."}]})
    r_diff = _req({"messages": [{"role": "user", "content": "Completely unrelated prompt"}]})
    assert m_path(r_rec, r_inc) is True
    assert m_path(r_rec, r_diff) is False
    assert m_path(r_rec, _req({"messages": []})) is False  # missing path -> False

    # Case-sensitive & preserve-whitespace mode
    m_strict = match_fuzzy_text(
        "messages.0.content", threshold=0.99, case_sensitive=True, normalize_whitespace=False
    )
    assert m_strict(r_rec, r_inc) is False

    # 3. Auto-extraction across OpenAI/Anthropic blocks, Gemini contents, Ollama prompt, raw string
    m_auto = match_fuzzy_text(threshold=0.85)
    assert m_auto(
        _req(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello world"}]}]}
        ),
        _req({"messages": [{"role": "user", "content": "hello world!"}]}),
    )
    assert m_auto(
        _req({"contents": [{"parts": [{"text": "Explain quantum computing"}]}]}),
        _req({"contents": [{"parts": [{"text": "Explain quantum computing."}]}]}),
    )
    assert m_auto(_req({"prompt": "Write a poem"}), _req({"prompt": "Write a poem!"}))
    assert m_auto(_req("plain prompt text"), _req("Plain prompt text"))
    assert m_auto(_req({"unrelated": 123}), _req({"unrelated": 123})) is False
    assert m_auto(_req(42), _req(42)) is False

    # Dict-valued path with {"text": ...} or {"content": ...}
    m_obj = match_fuzzy_text("payload", threshold=0.9)
    assert m_obj(_req({"payload": {"text": "alpha"}}), _req({"payload": {"content": "alpha"}}))

    # 4. Custom pluggable scorer (e.g., keyword Jaccard) + NaN score guard
    def jaccard(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0

    m_custom = match_fuzzy_text("prompt", threshold=0.5, scorer=jaccard)
    assert m_custom(_req({"prompt": "alpha beta gamma"}), _req({"prompt": "alpha beta delta"}))
    m_nan = match_fuzzy_text("prompt", threshold=0.5, scorer=lambda a, b: float("nan"))
    assert m_nan(_req({"prompt": "a"}), _req({"prompt": "a"})) is False

    # 5. End-to-end TapeSession replay with match_all + match_fuzzy_text
    cass_path = tmp_path / "fuzzy.tape.json"
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "cached answer"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
    )
    with TapeSession(cass_path, mode=TapeMode.RECORD) as rec:
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hi 2026!"}],
                },
            )
        rec.assert_call_count(1)

    matcher = match_all(match_method(), match_path(), match_fuzzy_text(threshold=0.8))
    with TapeSession(cass_path, mode=TapeMode.REPLAY, custom_matcher=matcher) as rep:
        with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(500))) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hi 2027!"}],
                },
            )
            assert resp.status_code == 200
            assert resp.json()["choices"][0]["message"]["content"] == "cached answer"
        rep.assert_call_count(1)


def test_cli_diff_markdown_file_and_directory_modes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = tmp_path / "a.tape.json"
    b = tmp_path / "b.tape.json"
    _make_cassette(a, text="old answer", cost=0.01)
    _make_cassette(
        b,
        text="new answer",
        cost=0.05,
        tools=[ToolCall(name="search", arguments={})],
    )

    # Single-file Markdown diff with text diff and regression failure
    rc = cli_main(
        [
            "diff",
            str(a),
            str(b),
            "--markdown",
            "--show-text-diff",
            "--max-cost-increase",
            "0.01",
            "--fail-on-tool-change",
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "### rhinotape Diff (`a.tape.json` -> `b.tape.json`) — FAILED" in out
    assert "| Est. Cost (USD) |" in out
    assert "<summary>Response Text Diff</summary>" in out
    assert "-old answer" in out and "+new answer" in out
    assert "Regression gate failed" in out

    # Single-file Markdown diff identical text -> PASS
    rc_same = cli_main(["diff", str(a), str(a), "--markdown", "--show-text-diff"])
    assert rc_same == 0
    out_same = capsys.readouterr().out
    assert "PASS" in out_same and "_Response Text Diff: identical_" in out_same

    # Directory Markdown diff (added, removed, changed, unchanged, gate failure)
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()
    _make_cassette(dir_a / "kept.tape.json", cost=0.01)
    _make_cassette(dir_b / "kept.tape.json", cost=0.01)
    _make_cassette(dir_a / "changed.tape.json", cost=0.01)
    _make_cassette(dir_b / "changed.tape.json", cost=0.04)
    _make_cassette(dir_a / "removed.tape.json", cost=0.005)
    _make_cassette(dir_b / "added.tape.json", cost=0.02)

    rc_dir = cli_main(
        ["diff", str(dir_a), str(dir_b), "--markdown", "--max-cost-increase", "0.005"]
    )
    assert rc_dir == 1
    out_dir = capsys.readouterr().out
    assert "### rhinotape Directory Diff" in out_dir
    assert "| Changed Cassette |" in out_dir
    assert "`changed.tape.json`" in out_dir
    assert "**Added:** `added.tape.json`" in out_dir
    assert "**Removed:** `removed.tape.json`" in out_dir
    assert "Regression gate failed" in out_dir


def test_scrubbing_cow_and_combined_regex_optimizations() -> None:
    # 1. Copy-on-write strip_ignored_fields preserves untouched subtrees without mutating input
    untouched = {"role": "system", "content": "large prompt " * 50}
    body: dict[str, Any] = {
        "messages": [untouched, {"role": "user", "content": "hi"}],
        "metadata": {"trace_id": "abc", "keep": 1},
        "items": [{"id": 1, "secret": "x"}, {"id": 2, "secret": "y"}],
        "tags": ["b", "a"],
    }
    stripped = strip_ignored_fields(body, ["metadata.trace_id", "items.*.secret", "messages.1"])
    assert body["metadata"]["trace_id"] == "abc"
    assert len(body["messages"]) == 2
    assert stripped["metadata"] == {"keep": 1}
    assert stripped["items"] == [{"id": 1}, {"id": 2}]
    assert len(stripped["messages"]) == 1
    assert stripped["messages"][0] is untouched  # zero deepcopy on untouched subtree!

    # Dict wildcard clearing and missing path no-op
    assert strip_ignored_fields({"a": {"x": 1}}, ["a.*"]) == {"a": {}}
    assert strip_ignored_fields(body, ["nonexistent.path"]) is body

    # 2. Copy-on-write normalize_json_array_order preserves untouched subtrees
    sorted_body = normalize_json_array_order(body, ["tags"])
    assert body["tags"] == ["b", "a"]
    assert sorted_body["tags"] == ["a", "b"]
    assert sorted_body["messages"] is body["messages"]
    assert normalize_json_array_order(body, ["missing.path"]) is body
    assert normalize_json_array_order({"a": [{"x": [2, 1]}]}, ["a.*.x"]) == {"a": [{"x": [1, 2]}]}
    assert normalize_json_array_order({"a": [{"x": [2, 1]}]}, ["a.0.x"]) == {"a": [{"x": [1, 2]}]}

    # 3. Single-pass combined regex for PII_PRESETS & safe fallback for incompatible flags/backrefs
    pii_tuple = tuple(PII_PRESETS.values())
    assert compile_patterns(pii_tuple) is pii_tuple
    combined = _combine_patterns(pii_tuple)
    assert combined is not None

    sample = "Email alice@example.com, SSN 123-45-6789, IP 10.20.30.40"
    redacted = redact_text(sample, pii_tuple)
    assert "alice@example.com" not in redacted
    assert "123-45-6789" not in redacted
    assert "10.20.30.40" not in redacted

    # Fallback when patterns have mismatched flags or backreferences
    mixed_flags = (re.compile(r"foo", re.IGNORECASE), re.compile(r"bar"))
    assert _combine_patterns(mixed_flags) is None
    assert redact_text("FOO and bar", mixed_flags) == "[REDACTED] and [REDACTED]"
    assert redact_body({"k": ["FOO", "bar", 1]}, redact_patterns=mixed_flags) == {
        "k": ["[REDACTED]", "[REDACTED]", 1]
    }
    backref = (re.compile(r"(\w+)=\1"), re.compile(r"secret"))
    assert _combine_patterns(backref) is None
    assert redact_text("aa=aa secret", backref) == "[REDACTED] [REDACTED]"

    # 4. fingerprint_request default fast-path matches explicit default match_on identically
    fp_fast = fingerprint_request("POST", "https://api.openai.com/v1/chat", {"a": 1})
    fp_explicit = fingerprint_request(
        "POST", "https://api.openai.com/v1/chat", {"a": 1}, match_on=["method", "url", "body"]
    )
    assert fp_fast == fp_explicit
