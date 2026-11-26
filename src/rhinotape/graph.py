"""ASCII and Mermaid trajectory graph renderers for rhinotape cassettes."""

from __future__ import annotations

import json
import math
from typing import Any

from rhinotape.cassette import Cassette
from rhinotape.models import Interaction, ToolCall

_MERMAID_TRANSLATION = str.maketrans(
    {
        '"': "'",
        "`": "'",
        "[": "(",
        "]": ")",
        "{": "(",
        "}": ")",
        "<": "(",
        ">": ")",
        ";": ",",
        "#": "",
    }
)


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3]}..."


def _tool_call_key(tc: ToolCall) -> tuple[str, str]:
    return (tc.name, json.dumps(tc.arguments, sort_keys=True, default=str))


def _format_args_preview(arguments: Any, max_chars: int = 60) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, dict):
        if not arguments:
            return ""
        raw = json.dumps(
            arguments,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    elif isinstance(arguments, str):
        raw = _collapse_whitespace(arguments)
        if not raw:
            return ""
    else:
        raw = _collapse_whitespace(str(arguments))
    return _truncate(_collapse_whitespace(raw), max_chars)


def _format_model_label(inter: Interaction) -> str:
    provider = inter.llm.provider or "unknown"
    model = inter.llm.model
    return f"{provider}/{model}" if model else provider


def _format_latency(latency_ms: float) -> str:
    return f"{latency_ms:.1f}ms" if math.isfinite(latency_ms) else "n/a"


def _format_cost(cost_usd: float) -> str:
    return f"${cost_usd:.6f}" if math.isfinite(cost_usd) else "$n/a"


def _sanitize_mermaid_label(text: str, max_chars: int = 120) -> str:
    """Sanitize quotes, brackets, newlines, and control characters for Mermaid labels."""
    cleaned = _collapse_whitespace(text).translate(_MERMAID_TRANSLATION)
    result = _truncate(_collapse_whitespace(cleaned), max_chars)
    return result or "-"


def render_ascii_graph(cassette: Cassette) -> str:
    """Render a compact ASCII call and tool trajectory graph for a Cassette."""
    interactions = cassette.interactions
    if not interactions:
        return (
            f"Trajectory: {cassette.path.name} (0 calls)\n"
            "(empty cassette - no interactions recorded)"
        )

    total_tokens = sum(i.llm.usage.total_tokens for i in interactions)
    total_cost = sum(
        i.llm.estimated_cost_usd for i in interactions if math.isfinite(i.llm.estimated_cost_usd)
    )
    total_latency = sum(i.latency_ms for i in interactions if math.isfinite(i.latency_ms))
    header = (
        f"Trajectory: {cassette.path.name} "
        f"({len(interactions)} call{'s' if len(interactions) != 1 else ''} | "
        f"{total_tokens} tok | ${total_cost:.6f} | {total_latency:.1f}ms)"
    )
    lines: list[str] = [header, "=" * min(len(header), 80)]

    prev_key: tuple[str, str] | None = None
    run_count = 0

    for idx, inter in enumerate(interactions, start=1):
        status = inter.response.status_code
        tags: list[str] = []
        if inter.llm.is_stream:
            tags.append("[SSE]")
        if status >= 400:
            tags.append(f"[ERR {status}]")
        tag_suffix = f" {' '.join(tags)}" if tags else ""

        lines.append(
            f"[{idx}] {_format_model_label(inter)} | {status} | "
            f"{_format_latency(inter.latency_ms)} | {inter.llm.usage.total_tokens} tok | "
            f"{_format_cost(inter.llm.estimated_cost_usd)}{tag_suffix}"
        )

        children: list[str] = []
        for tc in inter.llm.tool_calls:
            key = _tool_call_key(tc)
            if run_count > 0 and key == prev_key:
                run_count += 1
            else:
                prev_key = key
                run_count = 1
            args_preview = _format_args_preview(tc.arguments)
            loop_tag = f" [LOOP x{run_count}]" if run_count > 2 else ""
            children.append(f"tool: {tc.name}({args_preview}){loop_tag}")

        text_preview = _truncate(_collapse_whitespace(inter.llm.text), 80)
        if text_preview:
            children.append(f"text: {text_preview}")

        for c_idx, child in enumerate(children):
            branch = "`--" if c_idx == len(children) - 1 else "|--"
            lines.append(f"  {branch} {child}")

        if idx < len(interactions):
            lines.append("  |")
            lines.append("  v")

    return "\n".join(lines)


def render_mermaid_graph(cassette: Cassette) -> str:
    """Render a Mermaid `flowchart TD` diagram linking sequential LLM calls and tool calls."""
    interactions = cassette.interactions
    if not interactions:
        return 'flowchart TD\n    EMPTY["No interactions recorded"]'

    lines: list[str] = ["flowchart TD"]
    prev_nodes: list[str] = []
    error_nodes: list[str] = []
    loop_nodes: list[str] = []
    prev_key: tuple[str, str] | None = None
    run_count = 0

    for idx, inter in enumerate(interactions, start=1):
        call_id = f"C{idx}"
        status = inter.response.status_code
        parts: list[str] = [
            f"Call {idx}: {_format_model_label(inter)}",
            str(status),
        ]
        if status >= 400:
            parts.append(f"ERR {status}")
            error_nodes.append(call_id)
        parts.append(_format_latency(inter.latency_ms))
        parts.append(f"{inter.llm.usage.total_tokens} tok")
        parts.append(_format_cost(inter.llm.estimated_cost_usd))
        if inter.llm.is_stream:
            parts.append("SSE")
        text_preview = _truncate(_collapse_whitespace(inter.llm.text), 48)
        if text_preview:
            parts.append(text_preview)

        call_label = _sanitize_mermaid_label(" | ".join(parts), max_chars=160)
        lines.append(f'    {call_id}["{call_label}"]')

        for prev_id in prev_nodes:
            lines.append(f"    {prev_id} --> {call_id}")

        if inter.llm.tool_calls:
            tool_node_ids: list[str] = []
            for t_idx, tc in enumerate(inter.llm.tool_calls, start=1):
                key = _tool_call_key(tc)
                if run_count > 0 and key == prev_key:
                    run_count += 1
                else:
                    prev_key = key
                    run_count = 1
                tool_id = f"T{idx}_{t_idx}"
                args_preview = _format_args_preview(tc.arguments, max_chars=48)
                tool_raw = f"tool: {tc.name}({args_preview})"
                if run_count > 2:
                    tool_raw = f"{tool_raw} | LOOP x{run_count}"
                    loop_nodes.append(tool_id)
                tool_label = _sanitize_mermaid_label(tool_raw, max_chars=120)
                lines.append(f'    {tool_id}(["{tool_label}"])')
                lines.append(f"    {call_id} --> {tool_id}")
                tool_node_ids.append(tool_id)
            prev_nodes = tool_node_ids
        else:
            prev_nodes = [call_id]

    if error_nodes:
        lines.append("    classDef err fill:#fee2e2,stroke:#dc2626,color:#991b1b")
        lines.append(f"    class {','.join(error_nodes)} err")
    if loop_nodes:
        lines.append("    classDef loop fill:#fef3c7,stroke:#d97706,color:#92400e")
        lines.append(f"    class {','.join(loop_nodes)} loop")

    return "\n".join(lines)
