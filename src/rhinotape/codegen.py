"""Generate ready-to-run pytest regression test files from recorded cassettes."""

from __future__ import annotations

import keyword
import math
import re
from pathlib import Path

from rhinotape.cassette import Cassette

__all__ = ["generate_pytest_code"]

_CASSETTE_SUFFIXES: tuple[str, ...] = (
    ".tape.json.gz",
    ".tape.json",
    ".json.gz",
    ".json",
    ".gz",
)
_MAX_EXCERPT_CHARS = 60
_LINE_LIMIT = 100


def _validate_headroom(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float >= 1.0, got {value!r}")
    num = float(value)
    if not math.isfinite(num) or num < 1.0:
        raise ValueError(f"{name} must be a finite float >= 1.0, got {value!r}")
    return num


def _strip_cassette_suffix(filename: str) -> str:
    lower = filename.lower()
    for suffix in _CASSETTE_SUFFIXES:
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def _derive_test_name(cassette: Cassette, test_name: str | None) -> str:
    raw = test_name.strip() if test_name is not None else _strip_cassette_suffix(cassette.path.name)
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", raw.lower()).strip("_")
    if not cleaned or cleaned == "test":
        candidate = "test_cassette"
    elif cleaned.startswith("test_"):
        candidate = cleaned
    else:
        candidate = f"test_{cleaned}"
    if keyword.iskeyword(candidate) or not candidate.isidentifier():
        candidate = f"{candidate}_"
    return candidate


def _py_str(value: str) -> str:
    rendered = repr(value)
    if rendered.startswith('"') or ('"' in value and "'" not in value):
        return rendered
    inner = rendered[1:-1].replace("\\'", "'").replace('"', '\\"')
    return f'"{inner}"'


def _format_cost(value: float) -> str:
    text = f"{value:.6f}".rstrip("0")
    return f"{text}0" if text.endswith(".") else text


def _extract_excerpt(cassette: Cassette) -> str | None:
    for inter in reversed(cassette.interactions):
        if not inter.llm.text:
            continue
        for line in inter.llm.text.splitlines():
            stripped = line.strip()
            if stripped:
                sliced = stripped[:_MAX_EXCERPT_CHARS]
                return sliced.rstrip() or sliced
    return None


def _format_call(indent: str, expr_prefix: str, arg_literal: str) -> list[str]:
    single = f"{indent}{expr_prefix}({arg_literal})"
    if len(single) <= _LINE_LIMIT:
        return [single]
    return [f"{indent}{expr_prefix}(", f"{indent}    {arg_literal}", f"{indent})"]


def generate_pytest_code(
    cassette: Cassette,
    *,
    cassette_ref: str | None = None,
    test_name: str | None = None,
    cost_headroom: float = 1.25,
    token_headroom: float = 1.25,
    include_tool_assertions: bool = True,
    include_text_assertion: bool = True,
) -> str:
    """Generate a ready-to-run pytest module asserting a cassette's recorded behavior."""
    cost_mult = _validate_headroom("cost_headroom", cost_headroom)
    token_mult = _validate_headroom("token_headroom", token_headroom)

    fn_name = _derive_test_name(cassette, test_name)
    ref_str = cassette_ref if cassette_ref is not None else cassette.path.as_posix()
    path_lit = _py_str(ref_str)

    interactions = cassette.interactions
    call_count = len(interactions)
    stats = cassette.stats()
    total_tokens = int(stats["total_tokens"])
    est_cost = float(stats["estimated_cost_usd"])
    tools: list[str] = list(stats["tools"])

    lines: list[str] = [
        "from pathlib import Path",
        "",
        "from rhinotape import TapeMode, TapeSession",
        "",
        *_format_call("", "CASSETTE_PATH = Path", path_lit),
        "",
        "",
        f"def {fn_name}() -> None:",
        "    with TapeSession(CASSETTE_PATH, mode=TapeMode.REPLAY) as tape:",
        "        # Call your LLM function or agent under test here, e.g.: run_agent()",
        f"        tape.assert_call_count({call_count})",
    ]

    if all(i.response.status_code < 400 for i in interactions):
        lines.append("        tape.assert_no_errors()")

    providers = {i.llm.provider for i in interactions}
    if len(providers) == 1:
        provider = next(iter(providers))
        if provider and provider != "unknown":
            lines.extend(_format_call("        ", "tape.assert_provider", _py_str(provider)))

    if interactions and all(i.llm.model for i in interactions):
        models = {i.llm.model for i in interactions if i.llm.model}
        if len(models) == 1:
            model = next(iter(models))
            lines.extend(_format_call("        ", "tape.assert_model", _py_str(model)))

    if call_count > 0 and all(i.llm.is_stream for i in interactions):
        lines.append("        tape.assert_streamed()")

    if total_tokens > 0:
        token_budget = math.ceil(total_tokens * token_mult)
        lines.append(f"        tape.assert_tokens_under({token_budget})")

    if math.isfinite(est_cost) and est_cost > 0.0:
        scaled = est_cost * cost_mult * 1_000_000.0
        cost_budget = max(0.000001, math.ceil(scaled - 1e-12) / 1_000_000.0)
        lines.append(f"        tape.assert_cost_under({_format_cost(cost_budget)})")

    if include_tool_assertions and tools:
        tool_lits = [_py_str(t) for t in tools]
        single_tools = f"        tape.assert_tool_sequence([{', '.join(tool_lits)}])"
        if len(single_tools) <= _LINE_LIMIT:
            lines.append(single_tools)
        else:
            lines.append("        tape.assert_tool_sequence(")
            lines.append("            [")
            for lit in tool_lits:
                lines.append(f"                {lit},")
            lines.append("            ]")
            lines.append("        )")

    if include_text_assertion:
        excerpt = _extract_excerpt(cassette)
        if excerpt is not None:
            lines.extend(_format_call("        ", "tape.assert_text_contains", _py_str(excerpt)))

    return "\n".join(lines) + "\n"
