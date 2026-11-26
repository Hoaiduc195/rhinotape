"""Tests for rhinotape.codegen pytest test file generation."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from rhinotape.cassette import Cassette
from rhinotape.codegen import generate_pytest_code
from rhinotape.models import (
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    ToolCall,
    Usage,
)
from rhinotape.scrubbing import fingerprint_request


def _make_interaction(
    *,
    url: str = "https://api.openai.com/v1/chat/completions",
    req_body: Any = None,
    status_code: int = 200,
    provider: str = "openai",
    model: str | None = "gpt-4o-mini",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost: float = 0.0012,
    tools: tuple[ToolCall, ...] = (),
    text: str = "Paris is the capital of France.",
    is_stream: bool = False,
) -> Interaction:
    payload = {"messages": [{"role": "user", "content": "hi"}]} if req_body is None else req_body
    fp = fingerprint_request("POST", url, payload)
    if is_stream:
        chunks = ('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', "data: [DONE]\n\n")
        resp = RecordedResponse(
            status_code=status_code,
            headers={"content-type": "text/event-stream"},
            body=None,
            body_encoding="stream",
            stream_chunks=chunks,
        )
    else:
        resp = RecordedResponse(
            status_code=status_code,
            headers={"content-type": "application/json"},
            body={"ok": True},
            body_encoding="json",
        )
    return Interaction(
        request=RecordedRequest(
            method="POST",
            url=url,
            headers={"content-type": "application/json"},
            body=payload,
            fingerprint=fp,
        ),
        response=resp,
        latency_ms=42.0,
        llm=LLMCallSummary(
            provider=provider,
            model=model,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            estimated_cost_usd=cost,
            tool_calls=tools,
            text=text,
            is_stream=is_stream,
        ),
    )


def _assert_valid_python(code: str) -> ast.Module:
    tree = ast.parse(code)
    compile(code, "<generated_test>", "exec")
    return tree


def test_empty_cassette_generates_minimal_valid_test(tmp_path: Path) -> None:
    cassette_path = tmp_path / "empty.tape.json"
    cassette = Cassette(path=cassette_path, interactions=[])
    cassette.save()

    code = generate_pytest_code(cassette)
    _assert_valid_python(code)

    assert "from pathlib import Path" in code
    assert "from rhinotape import TapeMode, TapeSession" in code
    assert "with TapeSession(CASSETTE_PATH, mode=TapeMode.REPLAY) as tape:" in code
    assert "#" in code
    assert "def test_empty() -> None:" in code
    assert "tape.assert_call_count(0)" in code
    assert "tape.assert_no_errors()" in code
    assert "assert_provider" not in code
    assert "assert_model" not in code
    assert "assert_streamed" not in code
    assert "assert_tokens_under" not in code
    assert "assert_cost_under" not in code
    assert "assert_tool_sequence" not in code
    assert "assert_text_contains" not in code

    ns: dict[str, Any] = {}
    exec(compile(code, str(tmp_path / "test_empty.py"), "exec"), ns)
    ns["test_empty"]()


def test_multi_call_tool_and_text_assertions_and_live_replay(tmp_path: Path) -> None:
    cassette_path = tmp_path / "weather-agent.v2.tape.json.gz"
    i1 = _make_interaction(
        req_body={"step": 1},
        input_tokens=80,
        output_tokens=21,
        cost=0.0011,
        tools=(ToolCall(name="get_weather", arguments={"city": "Tokyo"}),),
        text="",
    )
    long_multiline = (
        "\n  Tokyo weather report: currently 19C with clear skies and light breeze today!  \n"
        "Second paragraph with extra details."
    )
    i2 = _make_interaction(
        req_body={"step": 2},
        input_tokens=120,
        output_tokens=30,
        cost=0.0014,
        tools=(ToolCall(name="format_forecast", arguments={"units": "C"}),),
        text=long_multiline,
    )
    cassette = Cassette(path=cassette_path, interactions=[i1, i2])
    cassette.save()

    code = generate_pytest_code(cassette)
    tree = _assert_valid_python(code)

    assert "def test_weather_agent_v2() -> None:" in code
    assert "tape.assert_call_count(2)" in code
    assert "tape.assert_no_errors()" in code
    assert 'tape.assert_provider("openai")' in code
    assert 'tape.assert_model("gpt-4o-mini")' in code
    assert "assert_streamed" not in code
    # total_tokens = 101 + 150 = 251; ceil(251 * 1.25) = ceil(313.75) = 314
    assert "tape.assert_tokens_under(314)" in code
    # est_cost = 0.0025; round(0.0025 * 1.25, 6) = 0.003125
    assert "tape.assert_cost_under(0.003125)" in code
    assert 'tape.assert_tool_sequence(["get_weather", "format_forecast"])' in code

    # Extract excerpt argument from AST and verify single-line <= 60 chars
    excerpt_args = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "assert_text_contains"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]
    assert len(excerpt_args) == 1
    excerpt = excerpt_args[0]
    assert 0 < len(excerpt) <= 60
    assert "\n" not in excerpt and "\r" not in excerpt
    assert excerpt in long_multiline

    # Replace comment placeholder with real replayed httpx calls and execute generated test
    runnable = code.replace(
        "with TapeSession(CASSETTE_PATH, mode=TapeMode.REPLAY) as tape:",
        "with TapeSession(CASSETTE_PATH, mode=TapeMode.REPLAY) as tape:\n"
        '        httpx.post("https://api.openai.com/v1/chat/completions", json={"step": 1})\n'
        '        httpx.post("https://api.openai.com/v1/chat/completions", json={"step": 2})',
    )
    ns: dict[str, Any] = {"httpx": httpx}
    exec(compile(runnable, "<runnable>", "exec"), ns)
    ns["test_weather_agent_v2"]()


def test_optional_flags_tiny_cost_floor_and_cassette_ref() -> None:
    inter = _make_interaction(
        input_tokens=10,
        output_tokens=5,
        cost=1e-7,
        tools=(ToolCall(name="lookup"),),
        text="He said \"hello\" and 'bye'.",
    )
    cassette = Cassette(path=Path("/abs/dir/sample.tape.json"), interactions=[inter])

    code_no_opts = generate_pytest_code(
        cassette,
        cassette_ref="cassettes/sample.tape.json",
        include_tool_assertions=False,
        include_text_assertion=False,
        cost_headroom=1.0,
        token_headroom=1.0,
    )
    _assert_valid_python(code_no_opts)
    assert 'CASSETTE_PATH = Path("cassettes/sample.tape.json")' in code_no_opts
    assert "tape.assert_tokens_under(15)" in code_no_opts
    assert "tape.assert_cost_under(0.000001)" in code_no_opts
    assert "assert_tool_sequence" not in code_no_opts
    assert "assert_text_contains" not in code_no_opts

    code_with_quotes = generate_pytest_code(cassette)
    _assert_valid_python(code_with_quotes)
    assert "assert_text_contains(" in code_with_quotes


def test_streaming_errors_and_mixed_provider_model_conditions() -> None:
    # All streamed + HTTP 429 error + unknown provider + mixed non-None models
    s1 = _make_interaction(
        status_code=200,
        provider="unknown",
        model="gpt-4o-mini",
        input_tokens=0,
        output_tokens=0,
        cost=0.0,
        text="   \n  ",
        is_stream=True,
    )
    s2 = _make_interaction(
        status_code=429,
        provider="unknown",
        model="gpt-4o",
        input_tokens=0,
        output_tokens=0,
        cost=0.0,
        text="",
        is_stream=True,
    )
    cassette = Cassette(path=Path("stream_err.tape.json"), interactions=[s1, s2])
    code = generate_pytest_code(cassette)
    _assert_valid_python(code)

    assert "tape.assert_call_count(2)" in code
    assert "tape.assert_streamed()" in code
    assert "assert_no_errors" not in code
    assert "assert_provider" not in code
    assert "assert_model" not in code
    assert "assert_tokens_under" not in code
    assert "assert_cost_under" not in code
    assert "assert_text_contains" not in code

    # Mixed streaming -> omit assert_streamed; mixed model + None -> omit assert_model
    m1 = _make_interaction(provider="openai", model="gpt-4o", is_stream=True)
    m2 = _make_interaction(provider="anthropic", model=None, is_stream=False)
    mixed_code = generate_pytest_code(Cassette(path=Path("mixed.tape.json"), interactions=[m1, m2]))
    _assert_valid_python(mixed_code)
    assert "assert_streamed" not in mixed_code
    assert "assert_provider" not in mixed_code
    assert "assert_model" not in mixed_code

    # Cost ceiling at headroom=1.0 never rounds below actual cost; non-finite cost is skipped.
    c_ceil = _make_interaction(cost=0.0000014)
    ceil_code = generate_pytest_code(
        Cassette(path=Path("ceil.tape.json"), interactions=[c_ceil]), cost_headroom=1.0
    )
    _assert_valid_python(ceil_code)
    assert "tape.assert_cost_under(0.000002)" in ceil_code

    c_inf = _make_interaction(cost=float("inf"))
    inf_code = generate_pytest_code(Cassette(path=Path("inf.tape.json"), interactions=[c_inf]))
    _assert_valid_python(inf_code)
    assert "assert_cost_under" not in inf_code


@pytest.mark.parametrize(
    ("cassette_name", "custom_name", "expected_fn"),
    [
        ("checkout-flow.tape.json", None, "test_checkout_flow"),
        ("checkout-flow.tape.json.gz", None, "test_checkout_flow"),
        ("test_existing.tape.json", None, "test_existing"),
        ("123-numeric!.tape.json", None, "test_123_numeric"),
        ("class.tape.json", None, "test_class"),
        (".tape.json", None, "test_cassette"),
        ("ignored.tape.json", "my_custom_case", "test_my_custom_case"),
        ("ignored.tape.json", "test_already_prefixed", "test_already_prefixed"),
        ("ignored.tape.json", "return", "test_return"),
        ("ignored.tape.json", "99 Luftballons!", "test_99_luftballons"),
        ("ignored.tape.json", "---", "test_cassette"),
    ],
)
def test_function_name_sanitization(
    cassette_name: str,
    custom_name: str | None,
    expected_fn: str,
) -> None:
    cassette = Cassette(path=Path(cassette_name), interactions=[])
    code = generate_pytest_code(cassette, test_name=custom_name)
    _assert_valid_python(code)
    assert f"def {expected_fn}() -> None:" in code


@pytest.mark.parametrize(
    "bad_value",
    [0.99, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, False],
)
def test_invalid_headroom_raises_value_error(bad_value: Any) -> None:
    cassette = Cassette(path=Path("a.tape.json"), interactions=[])
    with pytest.raises(ValueError, match="cost_headroom"):
        generate_pytest_code(cassette, cost_headroom=bad_value)
    with pytest.raises(ValueError, match="token_headroom"):
        generate_pytest_code(cassette, token_headroom=bad_value)


def test_generated_output_passes_ruff_and_mypy_strict(tmp_path: Path) -> None:
    many_tools = tuple(ToolCall(name=f"tool_number_{idx:02d}") for idx in range(8))
    inter = _make_interaction(
        tools=many_tools,
        text="Single \"double-quoted\" and 'single-quoted' excerpt for formatting check.",
    )
    long_ref = (
        "tests/cassettes/deeply/nested/subfolder/long_cassette_filename_wrapping_check.tape.json.gz"
    )
    cassette = Cassette(path=Path(long_ref), interactions=[inter])
    code = generate_pytest_code(cassette, cassette_ref=long_ref)

    out_file = tmp_path / "test_generated_check.py"
    out_file.write_text(code, encoding="utf-8")

    bin_dir = Path(sys.executable).parent
    ruff_bin = bin_dir / "ruff"
    mypy_bin = bin_dir / "mypy"

    subprocess.run([str(ruff_bin), "check", str(out_file)], check=True, capture_output=True)
    subprocess.run(
        [str(ruff_bin), "format", "--check", str(out_file)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [str(mypy_bin), "--strict", str(out_file)],
        check=True,
        capture_output=True,
    )
