"""Command-line interface for inspecting, verifying, and diffing llm-tape cassettes."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import io
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from llm_tape._version import __version__ as _VERSION
from llm_tape.cassette import Cassette, iter_cassette_files
from llm_tape.models import (
    CorruptCassetteError,
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    TapeError,
    ToolCall,
)
from llm_tape.parsers import parse_llm_interaction, split_sse_chunks
from llm_tape.scrubbing import (
    fingerprint_request,
    redact_body,
    redact_text,
    scrub_headers,
    scrub_url,
)

_BUILTIN_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic_key", re.compile(r"(?<![A-Za-z0-9_-])sk-ant-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])")),
    (
        "openai_key",
        re.compile(
            r"(?<![A-Za-z0-9_-])sk-(?!ant-)(?:proj-|svcacct-|admin-)?"
            r"[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"
        ),
    ),
    ("google_api_key", re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])")),
    ("aws_access_key", re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])")),
    (
        "github_token",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}(?![A-Za-z0-9_])"
            r"|(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{22,}(?![A-Za-z0-9_])"
        ),
    ),
    ("slack_token", re.compile(r"(?<![A-Za-z0-9_-])xox[baprs]-[0-9A-Za-z-]{10,}(?![A-Za-z0-9_-])")),
    (
        "bearer_token",
        re.compile(r"(?i)\bBearer\s+(?!\[REDACTED\]|REDACTED|<REDACTED>)[A-Za-z0-9._~+/=-]{16,}"),
    ),
)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _cassette_stats(cassette: Cassette) -> dict[str, Any]:
    return {
        "path": str(cassette.path),
        "recorded_at": cassette.recorded_at,
        "calls": len(cassette.interactions),
        "input_tokens": sum(i.llm.usage.input_tokens for i in cassette.interactions),
        "cached_input_tokens": sum(i.llm.usage.cached_input_tokens for i in cassette.interactions),
        "output_tokens": sum(i.llm.usage.output_tokens for i in cassette.interactions),
        "reasoning_tokens": sum(i.llm.usage.reasoning_tokens for i in cassette.interactions),
        "total_tokens": sum(i.llm.usage.total_tokens for i in cassette.interactions),
        "estimated_cost_usd": round(
            sum(i.llm.estimated_cost_usd for i in cassette.interactions), 8
        ),
        "latency_ms": round(sum(i.latency_ms for i in cassette.interactions), 2),
        "tools": [tc.name for i in cassette.interactions for tc in i.llm.tool_calls],
    }


def _cmd_inspect(path: Path, *, as_json: bool = False) -> int:
    cassette = Cassette.load(path)
    stats = _cassette_stats(cassette)
    if as_json:
        payload = {
            **stats,
            "interactions": [
                {
                    "method": inter.request.method,
                    "url": inter.request.url,
                    "status_code": inter.response.status_code,
                    "latency_ms": inter.latency_ms,
                    "provider": inter.llm.provider,
                    "model": inter.llm.model,
                    "is_stream": inter.llm.is_stream,
                    "tokens": inter.llm.usage.total_tokens,
                    "estimated_cost_usd": inter.llm.estimated_cost_usd,
                    "tools": [tc.name for tc in inter.llm.tool_calls],
                }
                for inter in cassette.interactions
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Cassette : {cassette.path}")
    print(f"Recorded : {cassette.recorded_at or 'unknown'}")
    print(f"Calls    : {stats['calls']}")
    print(f"Tokens   : {stats['total_tokens']}")
    print(f"Est. Cost: ${stats['estimated_cost_usd']:.6f}")

    if cassette.interactions:
        print("\nInteractions:")
        for idx, inter in enumerate(cassette.interactions, start=1):
            tools = ", ".join(tc.name for tc in inter.llm.tool_calls) or "-"
            model = inter.llm.model or "-"
            stream_tag = " [SSE]" if inter.llm.is_stream else ""
            print(
                f"  [{idx}] {inter.request.method} {inter.request.url} -> "
                f"{inter.response.status_code}{stream_tag} ({inter.latency_ms:.1f}ms) | "
                f"provider={inter.llm.provider} model={model} "
                f"tokens={inter.llm.usage.total_tokens} "
                f"cost=${inter.llm.estimated_cost_usd:.6f} tools=[{tools}]"
            )
    return 0


def _cmd_summary(
    directory: Path,
    *,
    as_json: bool = False,
    max_cost: float | None = None,
    max_tokens: int | None = None,
) -> int:
    if not directory.exists():
        print(f"Error: directory not found: {directory}", file=sys.stderr)
        return 1

    files = iter_cassette_files(directory)
    if not files:
        if as_json:
            print(json.dumps({"files": [], "errors": [], "totals": {}}, indent=2))
        else:
            print(f"No .tape.json cassettes found in {directory}")
        return 0

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    grand_calls = 0
    grand_tokens = 0
    grand_cost = 0.0

    for file_path in files:
        rel = str(file_path.relative_to(directory))
        try:
            cassette = Cassette.load(file_path)
        except CorruptCassetteError as exc:
            errors.append({"path": rel, "error": str(exc)})
            if not as_json:
                print(f"Warning: skipping corrupt cassette {rel}: {exc}", file=sys.stderr)
            continue
        stats = _cassette_stats(cassette)
        stats["relative_path"] = rel
        rows.append(stats)
        grand_calls += int(stats["calls"])
        grand_tokens += int(stats["total_tokens"])
        grand_cost += float(stats["estimated_cost_usd"])

    budget_exceeded = False
    if max_tokens is not None and grand_tokens > max_tokens:
        budget_exceeded = True
        print(
            f"Budget exceeded: total tokens {grand_tokens} > --max-tokens {max_tokens}",
            file=sys.stderr,
        )
    if max_cost is not None and round(grand_cost, 8) > max_cost:
        budget_exceeded = True
        print(
            f"Budget exceeded: total cost ${grand_cost:.6f} > --max-cost ${max_cost:.6f}",
            file=sys.stderr,
        )

    if as_json:
        print(
            json.dumps(
                {
                    "files": rows,
                    "errors": errors,
                    "totals": {
                        "files": len(rows),
                        "calls": grand_calls,
                        "total_tokens": grand_tokens,
                        "estimated_cost_usd": round(grand_cost, 8),
                    },
                },
                indent=2,
            )
        )
        return 1 if (errors or budget_exceeded) else 0

    print(f"{'Cassette':<48} {'Calls':>6} {'Tokens':>9} {'Est. Cost':>12}")
    print("-" * 79)
    for row in rows:
        rel = str(row["relative_path"])
        if len(rel) > 48:
            rel = "..." + rel[-45:]
        print(
            f"{rel:<48} {row['calls']:>6} {row['total_tokens']:>9} "
            f"${row['estimated_cost_usd']:>11.6f}"
        )
    print("-" * 79)
    print(
        f"{f'TOTAL ({len(rows)} files)':<48} {grand_calls:>6} "
        f"{grand_tokens:>9} ${grand_cost:>11.6f}"
    )
    return 1 if (errors or budget_exceeded) else 0


def _body_to_scan_text(body: Any) -> str:
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(body)


def _scan_text_for_secrets(
    text: str,
    location: str,
    patterns: list[tuple[str, re.Pattern[str]]],
) -> list[dict[str, str]]:
    if not text:
        return []
    hits: list[dict[str, str]] = []
    for name, regex in patterns:
        if regex.search(text):
            hits.append({"pattern": name, "location": location})
    return hits


def _scan_cassette_secrets(
    cassette: Cassette,
    patterns: list[tuple[str, re.Pattern[str]]],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if cassette.metadata:
        findings.extend(
            _scan_text_for_secrets(_body_to_scan_text(cassette.metadata), "metadata", patterns)
        )
    for idx, inter in enumerate(cassette.interactions):
        prefix = f"interactions[{idx}]"
        findings.extend(
            _scan_text_for_secrets(inter.request.url, f"{prefix}.request.url", patterns)
        )
        for k, v in inter.request.headers.items():
            if v != "[REDACTED]":
                findings.extend(
                    _scan_text_for_secrets(v, f"{prefix}.request.headers.{k}", patterns)
                )
        if inter.request.body_encoding not in {"base64", "empty"}:
            findings.extend(
                _scan_text_for_secrets(
                    _body_to_scan_text(inter.request.body), f"{prefix}.request.body", patterns
                )
            )
        for k, v in inter.response.headers.items():
            if v != "[REDACTED]":
                findings.extend(
                    _scan_text_for_secrets(v, f"{prefix}.response.headers.{k}", patterns)
                )
        if inter.response.body_encoding not in {"base64", "empty"}:
            findings.extend(
                _scan_text_for_secrets(
                    _body_to_scan_text(inter.response.body), f"{prefix}.response.body", patterns
                )
            )
        if inter.response.stream_chunks:
            joined = "".join(inter.response.stream_chunks)
            findings.extend(
                _scan_text_for_secrets(joined, f"{prefix}.response.stream_chunks", patterns)
            )
        findings.extend(_scan_text_for_secrets(inter.llm.text, f"{prefix}.llm.text", patterns))
        findings.extend(
            _scan_text_for_secrets(inter.llm.thinking, f"{prefix}.llm.thinking", patterns)
        )
        for tc_idx, tc in enumerate(inter.llm.tool_calls):
            findings.extend(
                _scan_text_for_secrets(
                    _body_to_scan_text(tc.arguments),
                    f"{prefix}.llm.tool_calls[{tc_idx}].arguments",
                    patterns,
                )
            )
    return findings


def _cmd_verify(
    targets: list[Path],
    *,
    check_secrets: bool = False,
    secret_patterns: list[str] | None = None,
    as_json: bool = False,
) -> int:
    """Validate .tape.json(.gz) cassette(s) for schema integrity and optional secret leaks."""
    if not targets:
        targets = [Path(".")]
    for target in targets:
        if not target.exists():
            print(f"Error: path not found: {target}", file=sys.stderr)
            return 1

    compiled_patterns: list[tuple[str, re.Pattern[str]]] = []
    if check_secrets or secret_patterns:
        compiled_patterns.extend(_BUILTIN_SECRET_PATTERNS)
        for idx, pat in enumerate(secret_patterns or (), start=1):
            compiled_patterns.append((f"custom_{idx}", re.compile(pat)))

    files_set: set[Path] = set()
    for target in targets:
        if target.is_file():
            files_set.add(target)
        else:
            files_set.update(iter_cassette_files(target))
    files = sorted(files_set)
    schema_errors: list[dict[str, str]] = []
    secret_findings: list[dict[str, str]] = []

    for file_path in files:
        try:
            cassette = Cassette.load(file_path)
        except TapeError as exc:
            schema_errors.append({"path": str(file_path), "error": str(exc)})
            if not as_json:
                print(f"FAIL {file_path}: {exc}", file=sys.stderr)
            continue
        if compiled_patterns:
            for hit in _scan_cassette_secrets(cassette, compiled_patterns):
                entry = {"path": str(file_path), **hit}
                secret_findings.append(entry)
                if not as_json:
                    print(
                        f"SECRET {file_path} [{hit['pattern']}] at {hit['location']}",
                        file=sys.stderr,
                    )

    ok = not schema_errors and not secret_findings
    if as_json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "checked_files": len(files),
                    "schema_errors": schema_errors,
                    "secret_findings": secret_findings,
                },
                indent=2,
            )
        )
        return 0 if ok else 1

    if not ok:
        print(
            f"Verification failed: {len(schema_errors)} invalid cassette(s), "
            f"{len(secret_findings)} secret finding(s) across {len(files)} file(s).",
            file=sys.stderr,
        )
        return 1
    label = str(targets[0]) if len(targets) == 1 else f"{len(targets)} targets"
    print(f"OK: verified {len(files)} cassette(s) in {label}.")
    return 0


def _check_diff_gates(
    *,
    before_cost: float,
    delta_cost: float,
    tools_changed: bool,
    max_cost_increase: float | None,
    max_cost_increase_pct: float | None,
    fail_on_tool_change: bool,
) -> list[str]:
    violations: list[str] = []
    rounded_before = round(before_cost, 8)
    rounded_delta = round(delta_cost, 8)
    if max_cost_increase is not None and rounded_delta > max_cost_increase:
        violations.append(
            f"Cost increased by ${rounded_delta:+.6f} > "
            f"--max-cost-increase ${max_cost_increase:.6f}"
        )
    if max_cost_increase_pct is not None and rounded_delta > 0:
        if rounded_before <= 0:
            pct_increase = math.inf
            exceeds = True
        else:
            pct_increase = (rounded_delta / rounded_before) * 100.0
            exceeds = round(pct_increase, 8) > max_cost_increase_pct
        if exceeds:
            pct_str = "inf%" if math.isinf(pct_increase) else f"{pct_increase:.2f}%"
            violations.append(
                f"Cost increased by {pct_str} > "
                f"--max-cost-increase-pct {max_cost_increase_pct:.2f}%"
            )
    if fail_on_tool_change and tools_changed:
        violations.append("Tool call sequence changed (--fail-on-tool-change)")
    return violations


def _cmd_diff(
    path_a: Path,
    path_b: Path,
    *,
    as_json: bool = False,
    max_cost_increase: float | None = None,
    max_cost_increase_pct: float | None = None,
    fail_on_tool_change: bool = False,
) -> int:
    """Compare calls, tokens, cached tokens, cost, latency, and tools between cassettes or dirs."""
    if not path_a.exists():
        print(f"Error: path not found: {path_a}", file=sys.stderr)
        return 1
    if not path_b.exists():
        print(f"Error: path not found: {path_b}", file=sys.stderr)
        return 1
    if path_a.is_dir() != path_b.is_dir():
        print(
            "Error: both diff arguments must be files or both must be directories.",
            file=sys.stderr,
        )
        return 1

    if path_a.is_dir() and path_b.is_dir():
        return _cmd_diff_dirs(
            path_a,
            path_b,
            as_json=as_json,
            max_cost_increase=max_cost_increase,
            max_cost_increase_pct=max_cost_increase_pct,
            fail_on_tool_change=fail_on_tool_change,
        )

    stats_a = _cassette_stats(Cassette.load(path_a))
    stats_b = _cassette_stats(Cassette.load(path_b))

    delta_calls = int(stats_b["calls"]) - int(stats_a["calls"])
    delta_tokens = int(stats_b["total_tokens"]) - int(stats_a["total_tokens"])
    delta_cached = int(stats_b["cached_input_tokens"]) - int(stats_a["cached_input_tokens"])
    delta_cost = round(
        float(stats_b["estimated_cost_usd"]) - float(stats_a["estimated_cost_usd"]), 8
    )
    delta_latency = round(float(stats_b["latency_ms"]) - float(stats_a["latency_ms"]), 2)
    tools_changed = stats_a["tools"] != stats_b["tools"]

    violations = _check_diff_gates(
        before_cost=float(stats_a["estimated_cost_usd"]),
        delta_cost=delta_cost,
        tools_changed=tools_changed,
        max_cost_increase=max_cost_increase,
        max_cost_increase_pct=max_cost_increase_pct,
        fail_on_tool_change=fail_on_tool_change,
    )
    for msg in violations:
        print(f"Regression gate failed: {msg}", file=sys.stderr)

    if as_json:
        print(
            json.dumps(
                {
                    "before": stats_a,
                    "after": stats_b,
                    "delta": {
                        "calls": delta_calls,
                        "total_tokens": delta_tokens,
                        "cached_input_tokens": delta_cached,
                        "estimated_cost_usd": delta_cost,
                        "latency_ms": delta_latency,
                        "tools_changed": tools_changed,
                    },
                    "violations": violations,
                },
                indent=2,
            )
        )
        return 1 if violations else 0

    print(f"Comparing {path_a} -> {path_b}")
    print(f"  Calls    : {stats_a['calls']} -> {stats_b['calls']} ({delta_calls:+d})")
    print(
        f"  Tokens   : {stats_a['total_tokens']} -> {stats_b['total_tokens']} ({delta_tokens:+d})"
    )
    print(
        f"  Cached   : {stats_a['cached_input_tokens']} -> "
        f"{stats_b['cached_input_tokens']} ({delta_cached:+d})"
    )
    print(
        f"  Est. Cost: ${stats_a['estimated_cost_usd']:.6f} -> "
        f"${stats_b['estimated_cost_usd']:.6f} (${delta_cost:+.6f})"
    )
    print(
        f"  Latency  : {stats_a['latency_ms']:.1f}ms -> "
        f"{stats_b['latency_ms']:.1f}ms ({delta_latency:+.1f}ms)"
    )
    print(f"  Tools    : {stats_a['tools']} -> {stats_b['tools']}")
    return 1 if violations else 0


def _cmd_diff_dirs(
    dir_a: Path,
    dir_b: Path,
    *,
    as_json: bool = False,
    max_cost_increase: float | None = None,
    max_cost_increase_pct: float | None = None,
    fail_on_tool_change: bool = False,
) -> int:
    map_a = {str(p.relative_to(dir_a)): p for p in iter_cassette_files(dir_a)}
    map_b = {str(p.relative_to(dir_b)): p for p in iter_cassette_files(dir_b)}
    all_keys = sorted(set(map_a) | set(map_b))

    added: list[str] = []
    removed: list[str] = []
    changed: list[dict[str, Any]] = []
    unchanged = 0

    before_calls = before_tokens = before_cached = 0
    after_calls = after_tokens = after_cached = 0
    before_cost = after_cost = 0.0
    before_latency = after_latency = 0.0
    any_tools_changed = False

    for key in all_keys:
        if key not in map_a:
            added.append(key)
            sb = _cassette_stats(Cassette.load(map_b[key]))
            after_calls += int(sb["calls"])
            after_tokens += int(sb["total_tokens"])
            after_cached += int(sb["cached_input_tokens"])
            after_cost += float(sb["estimated_cost_usd"])
            after_latency += float(sb["latency_ms"])
            if sb["tools"]:
                any_tools_changed = True
            continue
        if key not in map_b:
            removed.append(key)
            sa = _cassette_stats(Cassette.load(map_a[key]))
            before_calls += int(sa["calls"])
            before_tokens += int(sa["total_tokens"])
            before_cached += int(sa["cached_input_tokens"])
            before_cost += float(sa["estimated_cost_usd"])
            before_latency += float(sa["latency_ms"])
            if sa["tools"]:
                any_tools_changed = True
            continue

        sa = _cassette_stats(Cassette.load(map_a[key]))
        sb = _cassette_stats(Cassette.load(map_b[key]))
        before_calls += int(sa["calls"])
        before_tokens += int(sa["total_tokens"])
        before_cached += int(sa["cached_input_tokens"])
        before_cost += float(sa["estimated_cost_usd"])
        before_latency += float(sa["latency_ms"])
        after_calls += int(sb["calls"])
        after_tokens += int(sb["total_tokens"])
        after_cached += int(sb["cached_input_tokens"])
        after_cost += float(sb["estimated_cost_usd"])
        after_latency += float(sb["latency_ms"])

        d_calls = int(sb["calls"]) - int(sa["calls"])
        d_tokens = int(sb["total_tokens"]) - int(sa["total_tokens"])
        d_cached = int(sb["cached_input_tokens"]) - int(sa["cached_input_tokens"])
        d_cost = round(float(sb["estimated_cost_usd"]) - float(sa["estimated_cost_usd"]), 8)
        d_latency = round(float(sb["latency_ms"]) - float(sa["latency_ms"]), 2)
        t_changed = sa["tools"] != sb["tools"]
        if t_changed:
            any_tools_changed = True

        if d_calls or d_tokens or d_cached or d_cost or t_changed:
            changed.append(
                {
                    "relative_path": key,
                    "delta": {
                        "calls": d_calls,
                        "total_tokens": d_tokens,
                        "cached_input_tokens": d_cached,
                        "estimated_cost_usd": d_cost,
                        "latency_ms": d_latency,
                        "tools_changed": t_changed,
                    },
                }
            )
        else:
            unchanged += 1

    delta_cost = round(after_cost - before_cost, 8)
    delta_latency = round(after_latency - before_latency, 2)
    violations = _check_diff_gates(
        before_cost=before_cost,
        delta_cost=delta_cost,
        tools_changed=any_tools_changed,
        max_cost_increase=max_cost_increase,
        max_cost_increase_pct=max_cost_increase_pct,
        fail_on_tool_change=fail_on_tool_change,
    )
    for msg in violations:
        print(f"Regression gate failed: {msg}", file=sys.stderr)

    payload = {
        "before_dir": str(dir_a),
        "after_dir": str(dir_b),
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged_count": unchanged,
        "delta": {
            "calls": after_calls - before_calls,
            "total_tokens": after_tokens - before_tokens,
            "cached_input_tokens": after_cached - before_cached,
            "estimated_cost_usd": delta_cost,
            "latency_ms": delta_latency,
            "tools_changed": any_tools_changed,
        },
        "violations": violations,
    }
    if as_json:
        print(json.dumps(payload, indent=2))
        return 1 if violations else 0

    print(f"Comparing directories {dir_a} -> {dir_b}")
    print(
        f"  Files    : +{len(added)} added, -{len(removed)} removed, "
        f"~{len(changed)} changed, ={unchanged} unchanged"
    )
    print(f"  Calls    : {before_calls} -> {after_calls} ({after_calls - before_calls:+d})")
    print(f"  Tokens   : {before_tokens} -> {after_tokens} ({after_tokens - before_tokens:+d})")
    print(f"  Est. Cost: ${before_cost:.6f} -> ${after_cost:.6f} (${delta_cost:+.6f})")
    for item in changed:
        d = item["delta"]
        tool_tag = " [tools changed]" if d["tools_changed"] else ""
        print(
            f"  ~ {item['relative_path']}: calls {d['calls']:+d}, "
            f"tokens {d['total_tokens']:+d}, cost ${d['estimated_cost_usd']:+.6f}{tool_tag}"
        )
    return 1 if violations else 0


def _discover_test_names(tests_dir: Path) -> set[str]:
    """Discover pytest test names and explicit @pytest.mark.tape(name=...) targets via AST."""
    names: set[str] = set()

    def _add(raw: str) -> None:
        names.add(raw)
        sanitized = _SAFE_NAME_RE.sub("_", raw).strip("_")
        if sanitized:
            names.add(sanitized)

    def _extract_marker_names(decorators: list[ast.expr]) -> None:
        for dec in decorators:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            is_tape_marker = (isinstance(func, ast.Attribute) and func.attr == "tape") or (
                isinstance(func, ast.Name) and func.id == "tape"
            )
            if not is_tape_marker:
                continue
            for kw in dec.keywords:
                if (
                    kw.arg == "name"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    _add(kw.value.value)

    for py_file in sorted(tests_dir.rglob("*.py")):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    _add(node.name)
                _extract_marker_names(node.decorator_list)
            elif isinstance(node, ast.ClassDef):
                _extract_marker_names(node.decorator_list)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name.startswith("test_"):
                            _add(f"{node.name}__{item.name}")
                            _add(item.name)
                        _extract_marker_names(item.decorator_list)
    return names


def _cassette_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in (".tape.json.gz", ".tape.json"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _is_cassette_referenced(stem: str, discovered_names: set[str]) -> bool:
    if stem in discovered_names:
        return True
    for name in discovered_names:
        if stem.startswith(f"{name}_"):
            return True
    return False


def _cassette_timestamp(path: Path) -> datetime:
    try:
        cassette = Cassette.load(path)
        if cassette.recorded_at:
            raw = cassette.recorded_at
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
    except (TapeError, ValueError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _cmd_prune(
    directory: Path,
    *,
    tests_dir: Path | None = None,
    older_than_days: float | None = None,
    apply: bool = False,
    as_json: bool = False,
) -> int:
    """Find (and optionally delete with --apply) orphaned or stale cassettes."""
    if not directory.exists() or not directory.is_dir():
        print(f"Error: directory not found: {directory}", file=sys.stderr)
        return 1
    if tests_dir is None and older_than_days is None:
        print(
            "Error: specify at least one filter: --tests-dir and/or --older-than-days.",
            file=sys.stderr,
        )
        return 1
    if tests_dir is not None and (not tests_dir.exists() or not tests_dir.is_dir()):
        print(f"Error: tests directory not found: {tests_dir}", file=sys.stderr)
        return 1
    if older_than_days is not None and (not math.isfinite(older_than_days) or older_than_days < 0):
        print("Error: --older-than-days must be a finite number >= 0.", file=sys.stderr)
        return 1

    discovered = _discover_test_names(tests_dir) if tests_dir is not None else None
    cutoff: datetime | None = None
    if older_than_days is not None:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        except OverflowError:
            print(
                f"Error: --older-than-days value {older_than_days} is out of range.",
                file=sys.stderr,
            )
            return 1

    resolved_root = directory.resolve()
    files = [
        p
        for p in iter_cassette_files(directory)
        if not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(resolved_root)
    ]
    candidates: list[dict[str, Any]] = []

    for file_path in files:
        reasons: list[str] = []
        if discovered is not None:
            if not _is_cassette_referenced(_cassette_stem(file_path), discovered):
                reasons.append("unreferenced")
            else:
                continue
        if cutoff is not None:
            ts = _cassette_timestamp(file_path)
            if ts < cutoff:
                reasons.append(f"older_than_{older_than_days:g}d")
            else:
                continue
        if reasons:
            candidates.append(
                {
                    "path": str(file_path),
                    "relative_path": str(file_path.relative_to(directory)),
                    "reasons": reasons,
                }
            )

    deleted_count = 0
    if apply:
        for item in candidates:
            target = Path(item["path"])
            target.unlink(missing_ok=True)
            deleted_count += 1
            # Clean up empty parent directories strictly inside `directory`
            parent = target.parent
            while parent != parent.parent:
                parent_resolved = parent.resolve()
                if parent_resolved == resolved_root or not parent_resolved.is_relative_to(
                    resolved_root
                ):
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

    if as_json:
        print(
            json.dumps(
                {
                    "directory": str(directory),
                    "dry_run": not apply,
                    "scanned": len(files),
                    "candidates": candidates,
                    "deleted": deleted_count,
                },
                indent=2,
            )
        )
        return 0

    for item in candidates:
        action = "DELETED" if apply else "WOULD DELETE"
        print(f"{action}: {item['relative_path']} ({', '.join(item['reasons'])})")
    suffix = "" if apply else " (dry-run; pass --apply to delete)"
    print(f"Done: {len(candidates)}/{len(files)} cassette(s) matched{suffix}.")
    return 0


def _cmd_redact(
    target: Path,
    *,
    headers: list[str],
    query_params: list[str],
    fields: list[str],
    patterns: list[str],
    dry_run: bool = False,
) -> int:
    """Re-scrub one .tape.json file or all cassettes under a directory."""
    if not target.exists():
        print(f"Error: path not found: {target}", file=sys.stderr)
        return 1

    files = [target] if target.is_file() else iter_cassette_files(target)
    if not files:
        print(f"No .tape.json cassettes found in {target}")
        return 0

    updated = 0
    for file_path in files:
        cassette = Cassette.load(file_path)
        new_interactions: list[Interaction] = []
        changed = False
        for inter in cassette.interactions:
            new_url = scrub_url(inter.request.url, extra_sensitive=query_params)
            new_req_headers = scrub_headers(inter.request.headers, extra_sensitive=headers)
            new_req_body = (
                redact_body(inter.request.body, redact_fields=fields, redact_patterns=patterns)
                if inter.request.body_encoding not in {"base64", "empty"}
                else inter.request.body
            )
            if new_url != inter.request.url or new_req_body != inter.request.body:
                new_fp = fingerprint_request(
                    inter.request.method, new_url, new_req_body, headers=new_req_headers
                )
            else:
                new_fp = inter.request.fingerprint
            new_req = RecordedRequest(
                method=inter.request.method,
                url=new_url,
                headers=new_req_headers,
                body=new_req_body,
                fingerprint=new_fp,
                body_encoding=inter.request.body_encoding,
            )

            new_resp_headers = scrub_headers(inter.response.headers, extra_sensitive=headers)
            new_resp_body = (
                redact_body(inter.response.body, redact_fields=fields, redact_patterns=patterns)
                if inter.response.body_encoding not in {"base64", "empty"}
                else inter.response.body
            )
            if inter.response.stream_chunks is not None and patterns:
                per_chunk = [redact_text(c, patterns) for c in inter.response.stream_chunks]
                joined_per_chunk = "".join(per_chunk)
                joined_redacted = redact_text("".join(inter.response.stream_chunks), patterns)
                if joined_redacted != joined_per_chunk:
                    new_chunks: tuple[str, ...] | None = tuple(split_sse_chunks(joined_redacted))
                else:
                    new_chunks = tuple(per_chunk)
            else:
                new_chunks = inter.response.stream_chunks
            new_resp = RecordedResponse(
                status_code=inter.response.status_code,
                headers=new_resp_headers,
                body=new_resp_body,
                body_encoding=inter.response.body_encoding,
                stream_chunks=new_chunks,
            )
            if new_resp != inter.response:
                reparsed = parse_llm_interaction(
                    url=new_req.url,
                    request_body=new_req.body,
                    response_body=new_resp.body,
                    stream_chunks=new_resp.stream_chunks,
                    content_type=new_resp.headers.get("content-type", ""),
                )
                base_text = reparsed.text if reparsed.provider != "unknown" else inter.llm.text
                base_thinking = (
                    reparsed.thinking if reparsed.provider != "unknown" else inter.llm.thinking
                )
                base_calls = (
                    reparsed.tool_calls if reparsed.provider != "unknown" else inter.llm.tool_calls
                )
            else:
                base_text = inter.llm.text
                base_thinking = inter.llm.thinking
                base_calls = inter.llm.tool_calls

            new_llm_text = redact_text(base_text, patterns) if patterns else base_text
            new_llm_thinking = redact_text(base_thinking, patterns) if patterns else base_thinking
            new_tool_calls = tuple(
                ToolCall(
                    name=tc.name,
                    arguments=redact_body(
                        tc.arguments, redact_fields=fields, redact_patterns=patterns
                    ),
                    call_id=tc.call_id,
                )
                for tc in base_calls
            )
            new_llm = LLMCallSummary(
                provider=inter.llm.provider,
                model=inter.llm.model,
                usage=inter.llm.usage,
                estimated_cost_usd=inter.llm.estimated_cost_usd,
                tool_calls=new_tool_calls,
                text=new_llm_text,
                thinking=new_llm_thinking,
                is_stream=inter.llm.is_stream,
            )
            new_inter = Interaction(
                request=new_req,
                response=new_resp,
                latency_ms=inter.latency_ms,
                llm=new_llm,
            )
            if new_inter != inter:
                changed = True
            new_interactions.append(new_inter)

        if changed:
            updated += 1
            if not dry_run:
                cassette.interactions = new_interactions
                cassette.save()
            action = "WOULD UPDATE" if dry_run else "REDACTED"
            print(f"{action}: {file_path}")

    suffix = " (dry-run)" if dry_run else ""
    print(f"Done: {updated}/{len(files)} cassette(s) updated{suffix}.")
    return 0


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _get_header_ci(headers: dict[str, str], name: str, default: str = "") -> str:
    target = name.lower()
    for k, v in headers.items():
        if k.lower() == target:
            return v
    return default


def _render_har(cassettes: list[Cassette]) -> str:
    entries: list[dict[str, Any]] = []
    for cassette in cassettes:
        started = cassette.recorded_at or datetime.now(timezone.utc).isoformat()
        for inter in cassette.interactions:
            req_mime = _get_header_ci(inter.request.headers, "content-type", "application/json")
            req_text = (
                json.dumps(inter.request.body, ensure_ascii=False)
                if isinstance(inter.request.body, (dict, list))
                else ("" if inter.request.body is None else str(inter.request.body))
            )
            resp_mime = _get_header_ci(inter.response.headers, "content-type", "application/json")
            if inter.response.stream_chunks is not None:
                resp_text = "".join(inter.response.stream_chunks)
                resp_encoding: str | None = None
                resp_size = len(resp_text.encode("utf-8"))
            elif inter.response.body_encoding == "base64":
                resp_text = str(inter.response.body or "")
                resp_encoding = "base64"
                try:
                    resp_size = len(base64.b64decode(resp_text))
                except (ValueError, TypeError):
                    resp_size = len(resp_text.encode("utf-8"))
            elif isinstance(inter.response.body, (dict, list)):
                resp_text = json.dumps(inter.response.body, ensure_ascii=False)
                resp_encoding = None
                resp_size = len(resp_text.encode("utf-8"))
            else:
                resp_text = "" if inter.response.body is None else str(inter.response.body)
                resp_encoding = None
                resp_size = len(resp_text.encode("utf-8"))

            resp_content: dict[str, Any] = {
                "size": resp_size,
                "mimeType": resp_mime,
                "text": resp_text,
            }
            if resp_encoding is not None:
                resp_content["encoding"] = resp_encoding

            query_pairs = parse_qsl(urlsplit(inter.request.url).query, keep_blank_values=True)
            har_req: dict[str, Any] = {
                "method": inter.request.method,
                "url": inter.request.url,
                "httpVersion": "HTTP/1.1",
                "headers": [{"name": k, "value": v} for k, v in inter.request.headers.items()],
                "queryString": [{"name": k, "value": v} for k, v in query_pairs],
                "cookies": [],
                "headersSize": -1,
                "bodySize": len(req_text.encode("utf-8")),
            }
            if req_text:
                har_req["postData"] = {"mimeType": req_mime, "text": req_text}

            entries.append(
                {
                    "startedDateTime": started,
                    "time": round(inter.latency_ms, 2),
                    "request": har_req,
                    "response": {
                        "status": inter.response.status_code,
                        "statusText": "",
                        "httpVersion": "HTTP/1.1",
                        "headers": [
                            {"name": k, "value": v} for k, v in inter.response.headers.items()
                        ],
                        "cookies": [],
                        "content": resp_content,
                        "redirectURL": "",
                        "headersSize": -1,
                        "bodySize": resp_content["size"],
                    },
                    "cache": {},
                    "timings": {"send": 0, "wait": round(inter.latency_ms, 2), "receive": 0},
                    "comment": f"{inter.llm.provider}:{inter.llm.model or 'unknown'}",
                }
            )
    payload = {
        "log": {
            "version": "1.2",
            "creator": {"name": "llm-tape", "version": _VERSION},
            "entries": entries,
        }
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _cmd_export(target: Path, *, fmt: str, output: Path | None = None) -> int:
    """Export cassette(s) as Markdown PR summary, CSV cost report, or HAR 1.2 archive."""
    if not target.exists():
        print(f"Error: path not found: {target}", file=sys.stderr)
        return 1

    files = [target] if target.is_file() else iter_cassette_files(target)
    base_dir = target.parent if target.is_file() else target
    cassettes = [Cassette.load(p) for p in files]

    if fmt == "har":
        rendered = _render_har(cassettes)
    elif fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(
            [
                "cassette",
                "calls",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
                "estimated_cost_usd",
                "latency_ms",
            ]
        )
        for c in cassettes:
            s = _cassette_stats(c)
            rel = str(c.path.relative_to(base_dir))
            writer.writerow(
                [
                    rel,
                    s["calls"],
                    s["input_tokens"],
                    s["cached_input_tokens"],
                    s["output_tokens"],
                    s["reasoning_tokens"],
                    s["total_tokens"],
                    f"{s['estimated_cost_usd']:.8f}",
                    f"{s['latency_ms']:.2f}",
                ]
            )
        rendered = buf.getvalue()
    else:  # markdown
        lines = [
            "| Cassette | Calls | Tokens | Cached | Reasoning | Est. Cost | Latency |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        t_calls = t_tokens = t_cached = t_reasoning = 0
        t_cost = 0.0
        t_latency = 0.0
        for c in cassettes:
            s = _cassette_stats(c)
            rel = str(c.path.relative_to(base_dir)).replace("|", "\\|")
            lines.append(
                f"| `{rel}` | {s['calls']} | {s['total_tokens']} | "
                f"{s['cached_input_tokens']} | {s['reasoning_tokens']} | "
                f"${s['estimated_cost_usd']:.6f} | {s['latency_ms']:.1f}ms |"
            )
            t_calls += int(s["calls"])
            t_tokens += int(s["total_tokens"])
            t_cached += int(s["cached_input_tokens"])
            t_reasoning += int(s["reasoning_tokens"])
            t_cost += float(s["estimated_cost_usd"])
            t_latency += float(s["latency_ms"])
        lines.append(
            f"| **TOTAL ({len(cassettes)} files)** | **{t_calls}** | **{t_tokens}** | "
            f"**{t_cached}** | **{t_reasoning}** | **${t_cost:.6f}** | **{t_latency:.1f}ms** |"
        )
        rendered = "\n".join(lines) + "\n"

    if output is not None:
        _write_text_atomic(output, rendered)
        print(f"Exported {fmt} report ({len(cassettes)} cassette(s)) to {output}.")
    else:
        sys.stdout.write(rendered)
    return 0


def _cmd_convert(
    target: Path,
    *,
    to_format: str,
    delete_source: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Convert cassettes between plain .tape.json and gzip-compressed .tape.json.gz."""
    if not target.exists():
        print(f"Error: path not found: {target}", file=sys.stderr)
        return 1

    if target.is_file():
        files = [target]
    else:
        resolved_root = target.resolve()
        files = [
            p
            for p in iter_cassette_files(target)
            if not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(resolved_root)
        ]

    converted = 0
    conflicts = 0
    for file_path in files:
        is_gz = file_path.name.lower().endswith(".gz")
        if (to_format == "gzip" and is_gz) or (to_format == "json" and not is_gz):
            continue
        stem = _cassette_stem(file_path)
        new_ext = ".tape.json.gz" if to_format == "gzip" else ".tape.json"
        dest_path = file_path.with_name(f"{stem}{new_ext}")
        if (dest_path.exists() or dest_path.is_symlink()) and not force:
            conflicts += 1
            print(
                f"CONFLICT: destination already exists: {dest_path} (pass --force to overwrite)",
                file=sys.stderr,
            )
            continue
        converted += 1
        if dry_run:
            print(f"WOULD CONVERT: {file_path} -> {dest_path}")
            continue
        cassette = Cassette.load(file_path)
        cassette.path = dest_path
        cassette.save()
        if delete_source and dest_path.resolve() != file_path.resolve():
            file_path.unlink(missing_ok=True)
        print(f"CONVERTED: {file_path} -> {dest_path}")

    suffix = " (dry-run)" if dry_run else ""
    print(f"Done: {converted}/{len(files)} cassette(s) converted to {to_format}{suffix}.")
    return 1 if conflicts else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-tape",
        description="Inspect, verify, diff, and redact llm-tape .tape.json cassettes.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_VERSION}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect a single .tape.json cassette.")
    p_inspect.add_argument("path", type=Path, help="Path to the .tape.json file.")
    p_inspect.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")

    p_summary = sub.add_parser(
        "summary", help="Summarize all .tape.json cassettes under a directory."
    )
    p_summary.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Directory to scan (default: current directory).",
    )
    p_summary.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")
    p_summary.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Fail with exit code 1 if total estimated USD cost exceeds this limit.",
    )
    p_summary.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Fail with exit code 1 if total tokens exceed this limit.",
    )

    p_verify = sub.add_parser(
        "verify",
        help="Validate .tape.json(.gz) cassette(s) for schema integrity and optional secret leaks.",
    )
    p_verify.add_argument(
        "targets",
        type=Path,
        nargs="*",
        default=[],
        help="Cassette file(s) or directory(ies) to validate (default: current directory).",
    )
    p_verify.add_argument(
        "--check-secrets",
        action="store_true",
        help="Scan cassettes for common unredacted API keys and bearer tokens.",
    )
    p_verify.add_argument(
        "--secret-pattern",
        action="append",
        default=[],
        dest="secret_patterns",
        help="Additional regex pattern to treat as a leaked secret (implies --check-secrets).",
    )
    p_verify.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")

    p_diff = sub.add_parser(
        "diff", help="Compare two .tape.json cassettes or two cassette directories."
    )
    p_diff.add_argument("before", type=Path, help="Baseline .tape.json file or directory.")
    p_diff.add_argument("after", type=Path, help="New .tape.json file or directory.")
    p_diff.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")
    p_diff.add_argument(
        "--max-cost-increase",
        type=float,
        default=None,
        help="Fail with exit code 1 if cost increases by more than this USD amount.",
    )
    p_diff.add_argument(
        "--max-cost-increase-pct",
        type=float,
        default=None,
        help="Fail with exit code 1 if cost increases by more than this percentage.",
    )
    p_diff.add_argument(
        "--fail-on-tool-change",
        action="store_true",
        help="Fail with exit code 1 if tool call sequences changed.",
    )

    p_prune = sub.add_parser(
        "prune",
        help="Find (or delete with --apply) orphaned or stale .tape.json cassettes.",
    )
    p_prune.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Cassette directory to scan (default: current directory).",
    )
    p_prune.add_argument(
        "--tests-dir",
        type=Path,
        default=None,
        help="Test directory to scan for referenced pytest test names.",
    )
    p_prune.add_argument(
        "--older-than-days",
        type=float,
        default=None,
        help="Match cassettes older than this many days (ANDed with --tests-dir if both given).",
    )
    p_prune.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matched cassettes (default is dry-run).",
    )
    p_prune.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")

    p_redact = sub.add_parser(
        "redact", help="Re-scrub secrets/PII in existing .tape.json cassettes."
    )
    p_redact.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Cassette file or directory to redact (default: current directory).",
    )
    p_redact.add_argument(
        "--header",
        action="append",
        default=[],
        dest="headers",
        help="Extra sensitive header name to redact (repeatable).",
    )
    p_redact.add_argument(
        "--query-param",
        action="append",
        default=[],
        dest="query_params",
        help="Extra sensitive query parameter name to redact (repeatable).",
    )
    p_redact.add_argument(
        "--field",
        action="append",
        default=[],
        dest="fields",
        help="JSON dot-path field to redact in request/response bodies (repeatable).",
    )
    p_redact.add_argument(
        "--pattern",
        action="append",
        default=[],
        dest="patterns",
        help="Regex pattern to redact in bodies and stream chunks (repeatable).",
    )
    p_redact.add_argument(
        "--dry-run",
        action="store_true",
        help="Report cassettes that would change without modifying files.",
    )

    p_export = sub.add_parser(
        "export",
        help="Export cassette(s) to Markdown PR summary, CSV cost report, or HAR 1.2 archive.",
    )
    p_export.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Cassette file or directory to export (default: current directory).",
    )
    p_export.add_argument(
        "--format",
        choices=["markdown", "csv", "har"],
        default="markdown",
        dest="fmt",
        help="Output format: markdown (default), csv, or har.",
    )
    p_export.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write output (defaults to stdout).",
    )

    p_convert = sub.add_parser(
        "convert",
        help="Convert cassettes between plain .tape.json and gzip-compressed .tape.json.gz.",
    )
    p_convert.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Cassette file or directory to convert (default: current directory).",
    )
    p_convert.add_argument(
        "--to",
        choices=["gzip", "json"],
        required=True,
        dest="to_format",
        help="Target cassette format: gzip (.tape.json.gz) or json (.tape.json).",
    )
    p_convert.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete the original file after successful conversion.",
    )
    p_convert.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview conversions without modifying files.",
    )
    p_convert.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination cassette files on collision.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _cmd_inspect(args.path, as_json=args.as_json)
        if args.command == "summary":
            return _cmd_summary(
                args.directory,
                as_json=args.as_json,
                max_cost=args.max_cost,
                max_tokens=args.max_tokens,
            )
        if args.command == "verify":
            return _cmd_verify(
                args.targets,
                check_secrets=args.check_secrets,
                secret_patterns=args.secret_patterns,
                as_json=args.as_json,
            )
        if args.command == "diff":
            return _cmd_diff(
                args.before,
                args.after,
                as_json=args.as_json,
                max_cost_increase=args.max_cost_increase,
                max_cost_increase_pct=args.max_cost_increase_pct,
                fail_on_tool_change=args.fail_on_tool_change,
            )
        if args.command == "prune":
            return _cmd_prune(
                args.directory,
                tests_dir=args.tests_dir,
                older_than_days=args.older_than_days,
                apply=args.apply,
                as_json=args.as_json,
            )
        if args.command == "redact":
            return _cmd_redact(
                args.target,
                headers=args.headers,
                query_params=args.query_params,
                fields=args.fields,
                patterns=args.patterns,
                dry_run=args.dry_run,
            )
        if args.command == "export":
            return _cmd_export(args.target, fmt=args.fmt, output=args.output)
        if args.command == "convert":
            return _cmd_convert(
                args.target,
                to_format=args.to_format,
                delete_source=args.delete_source,
                dry_run=args.dry_run,
                force=args.force,
            )
    except (TapeError, re.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
