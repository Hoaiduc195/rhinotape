"""Command-line interface for inspecting, verifying, and diffing llm-tape cassettes."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from llm_tape._version import __version__ as _VERSION
from llm_tape.cassette import Cassette
from llm_tape.models import (
    CorruptCassetteError,
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    TapeError,
    ToolCall,
)
from llm_tape.parsers import parse_llm_interaction
from llm_tape.scrubbing import (
    fingerprint_request,
    redact_body,
    redact_text,
    scrub_headers,
    scrub_url,
)

_BUILTIN_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "github_token",
        re.compile(
            r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b"
        ),
    ),
    ("slack_token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
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

    files = sorted(directory.rglob("*.tape.json"))
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
    target: Path,
    *,
    check_secrets: bool = False,
    secret_patterns: list[str] | None = None,
    as_json: bool = False,
) -> int:
    """Validate .tape.json cassette(s) for schema integrity and optional secret leaks."""
    if not target.exists():
        print(f"Error: path not found: {target}", file=sys.stderr)
        return 1

    compiled_patterns: list[tuple[str, re.Pattern[str]]] = []
    if check_secrets or secret_patterns:
        compiled_patterns.extend(_BUILTIN_SECRET_PATTERNS)
        for idx, pat in enumerate(secret_patterns or (), start=1):
            compiled_patterns.append((f"custom_{idx}", re.compile(pat)))

    files = [target] if target.is_file() else sorted(target.rglob("*.tape.json"))
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
    print(f"OK: verified {len(files)} cassette(s) in {target}.")
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
    if max_cost_increase is not None and round(delta_cost, 8) > max_cost_increase:
        violations.append(
            f"Cost increased by ${delta_cost:+.6f} > --max-cost-increase ${max_cost_increase:.6f}"
        )
    if max_cost_increase_pct is not None and round(delta_cost, 8) > 0:
        if before_cost <= 0:
            pct_increase = math.inf
        else:
            pct_increase = (delta_cost / before_cost) * 100.0
        if pct_increase > max_cost_increase_pct:
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
    map_a = {str(p.relative_to(dir_a)): p for p in sorted(dir_a.rglob("*.tape.json"))}
    map_b = {str(p.relative_to(dir_b)): p for p in sorted(dir_b.rglob("*.tape.json"))}
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
    if name.endswith(".tape.json"):
        return name[: -len(".tape.json")]
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
    if tests_dir is not None and not tests_dir.exists():
        print(f"Error: tests directory not found: {tests_dir}", file=sys.stderr)
        return 1
    if older_than_days is not None and older_than_days < 0:
        print("Error: --older-than-days must be >= 0.", file=sys.stderr)
        return 1

    discovered = _discover_test_names(tests_dir) if tests_dir is not None else None
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=older_than_days)
        if older_than_days is not None
        else None
    )

    files = sorted(directory.rglob("*.tape.json"))
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
        resolved_root = directory.resolve()
        for item in candidates:
            target = Path(item["path"])
            target.unlink(missing_ok=True)
            deleted_count += 1
            # Clean up empty parent directories strictly inside `directory`
            parent = target.parent
            while parent.resolve() != resolved_root:
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

    files = [target] if target.is_file() else sorted(target.rglob("*.tape.json"))
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
                    parts = joined_redacted.split("\n\n")
                    new_chunks: tuple[str, ...] | None = tuple(
                        f"{p}\n\n" if i < len(parts) - 1 else p
                        for i, p in enumerate(parts)
                        if p or i < len(parts) - 1
                    )
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
                base_calls = (
                    reparsed.tool_calls if reparsed.provider != "unknown" else inter.llm.tool_calls
                )
            else:
                base_text = inter.llm.text
                base_calls = inter.llm.tool_calls

            new_llm_text = redact_text(base_text, patterns) if patterns else base_text
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
        help="Validate .tape.json cassette(s) for schema integrity and optional secret leaks.",
    )
    p_verify.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Cassette file or directory to validate (default: current directory).",
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
                args.target,
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
    except (TapeError, re.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
