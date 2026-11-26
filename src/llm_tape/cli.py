"""Command-line interface for inspecting, verifying, and diffing llm-tape cassettes."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import difflib
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from llm_tape._version import __version__ as _VERSION
from llm_tape.cassette import Cassette, cassette_json_schema, iter_cassette_files
from llm_tape.models import (
    CorruptCassetteError,
    Interaction,
    LLMCallSummary,
    RecordedRequest,
    RecordedResponse,
    TapeError,
    ToolCall,
)
from llm_tape.parsers import parse_llm_interaction, redact_stream_chunks
from llm_tape.pricing import (
    DEFAULT_PRICING,
    PRICING_FILE_ENV,
    ModelPrice,
    estimate_cache_savings_usd,
    load_pricing_file,
)
from llm_tape.scrubbing import (
    DEFAULT_MATCH_ON,
    PII_PRESETS,
    compile_patterns,
    fingerprint_request,
    redact_body,
    redact_text,
    resolve_pii_patterns,
    scrub_headers,
    scrub_url,
    validate_match_on,
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
        "huggingface_token",
        re.compile(r"(?<![A-Za-z0-9_-])hf_[A-Za-z0-9]{20,}(?![A-Za-z0-9_-])"),
    ),
    ("groq_key", re.compile(r"(?<![A-Za-z0-9_-])gsk_[A-Za-z0-9]{20,}(?![A-Za-z0-9_-])")),
    ("perplexity_key", re.compile(r"(?<![A-Za-z0-9_-])pplx-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])")),
    ("xai_key", re.compile(r"(?<![A-Za-z0-9_-])xai-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])")),
    ("pem_private_key", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    (
        "bearer_token",
        re.compile(r"(?i)\bBearer\s+(?!\[REDACTED\]|REDACTED|<REDACTED>)[A-Za-z0-9._~+/=-]{16,}"),
    ),
)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _validate_non_negative_float(flag: str, value: float | None) -> str | None:
    if value is not None and (not math.isfinite(value) or value < 0.0):
        return f"Error: {flag} must be a finite number >= 0."
    return None


def _parse_csv_flags(raw_values: list[str] | None) -> list[str]:
    result: list[str] = []
    for item in raw_values or ():
        for part in item.split(","):
            cleaned = part.strip()
            if cleaned:
                result.append(cleaned)
    return result


def _cassette_stats(cassette: Cassette) -> dict[str, Any]:
    return cassette.stats()


def _cmd_inspect(
    path: Path,
    *,
    as_json: bool = False,
    show_bodies: bool = False,
    show_thinking: bool = False,
    interaction_index: int | None = None,
) -> int:
    cassette = Cassette.load(path)
    stats = _cassette_stats(cassette)
    content_hash = cassette.content_hash()

    indexed = list(enumerate(cassette.interactions, start=1))
    if interaction_index is not None:
        if interaction_index < 1 or interaction_index > len(indexed):
            print(
                f"Error: --interaction {interaction_index} out of range "
                f"(cassette has {len(indexed)} interaction(s)).",
                file=sys.stderr,
            )
            return 1
        indexed = [indexed[interaction_index - 1]]

    if as_json:
        items: list[dict[str, Any]] = []
        for idx, inter in indexed:
            row: dict[str, Any] = {
                "index": idx,
                "method": inter.request.method,
                "url": inter.request.url,
                "status_code": inter.response.status_code,
                "latency_ms": inter.latency_ms,
                "provider": inter.llm.provider,
                "model": inter.llm.model,
                "is_stream": inter.llm.is_stream,
                "tokens": inter.llm.usage.total_tokens,
                "cached_input_tokens": inter.llm.usage.cached_input_tokens,
                "reasoning_tokens": inter.llm.usage.reasoning_tokens,
                "estimated_cost_usd": inter.llm.estimated_cost_usd,
                "tools": [tc.name for tc in inter.llm.tool_calls],
            }
            if show_thinking:
                row["text"] = inter.llm.text
                row["thinking"] = inter.llm.thinking
            if show_bodies:
                row["request_body"] = inter.request.body
                row["response_body"] = inter.response.body
                row["tool_calls"] = [tc.to_dict() for tc in inter.llm.tool_calls]
            items.append(row)
        payload = {
            **stats,
            "content_hash": content_hash,
            "interactions": items,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Cassette : {cassette.path}")
    print(f"Recorded : {cassette.recorded_at or 'unknown'}")
    print(f"Hash     : {content_hash[:12]}")
    print(f"Calls    : {stats['calls']}")
    print(f"Tokens   : {stats['total_tokens']}")
    print(f"Est. Cost: ${stats['estimated_cost_usd']:.6f}")

    if indexed:
        print("\nInteractions:")
        for idx, inter in indexed:
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
            if show_thinking:
                if inter.llm.thinking:
                    print(f"      thinking: {inter.llm.thinking}")
                if inter.llm.text:
                    print(f"      text    : {inter.llm.text}")
            if show_bodies:
                req_str = (
                    json.dumps(inter.request.body, ensure_ascii=False)
                    if isinstance(inter.request.body, (dict, list))
                    else str(inter.request.body)
                )
                resp_str = (
                    f"<stream: {len(inter.response.stream_chunks)} chunk(s)>"
                    if inter.response.stream_chunks is not None
                    else (
                        json.dumps(inter.response.body, ensure_ascii=False)
                        if isinstance(inter.response.body, (dict, list))
                        else str(inter.response.body)
                    )
                )
                print(f"      request : {req_str}")
                print(f"      response: {resp_str}")
    return 0


def _cmd_summary(
    directory: Path,
    *,
    as_json: bool = False,
    max_cost: float | None = None,
    max_tokens: int | None = None,
    group_by: str = "file",
) -> int:
    if err := _validate_non_negative_float("--max-cost", max_cost):
        print(err, file=sys.stderr)
        return 1
    if max_tokens is not None and max_tokens < 0:
        print("Error: --max-tokens must be >= 0.", file=sys.stderr)
        return 1
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

    if group_by == "dir":
        dir_buckets: dict[str, dict[str, Any]] = {}
        for row in rows:
            parent = Path(str(row["relative_path"])).parent.as_posix()
            b = dir_buckets.setdefault(
                parent,
                {
                    "directory": parent,
                    "files": 0,
                    "calls": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                },
            )
            b["files"] += 1
            b["calls"] += int(row["calls"])
            b["total_tokens"] += int(row["total_tokens"])
            b["estimated_cost_usd"] = round(
                float(b["estimated_cost_usd"]) + float(row["estimated_cost_usd"]), 8
            )
        grouped_rows = sorted(
            dir_buckets.values(),
            key=lambda r: (-float(r["estimated_cost_usd"]), str(r["directory"])),
        )
        if as_json:
            print(
                json.dumps(
                    {
                        "groups": grouped_rows,
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
        print(f"{'Directory':<42} {'Files':>5} {'Calls':>6} {'Tokens':>9} {'Est. Cost':>12}")
        print("-" * 79)
        for g in grouped_rows:
            dlabel = str(g["directory"])
            if len(dlabel) > 42:
                dlabel = "..." + dlabel[-39:]
            print(
                f"{dlabel:<42} {g['files']:>5} {g['calls']:>6} {g['total_tokens']:>9} "
                f"${g['estimated_cost_usd']:>11.6f}"
            )
        print("-" * 79)
        print(
            f"{f'TOTAL ({len(grouped_rows)} dirs)':<42} {len(rows):>5} {grand_calls:>6} "
            f"{grand_tokens:>9} ${grand_cost:>11.6f}"
        )
        return 1 if (errors or budget_exceeded) else 0

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


def _body_to_secret_scan_text(body: Any, encoding: str) -> str:
    if encoding == "empty" or body is None:
        return ""
    if encoding == "base64":
        if not isinstance(body, str) or not body:
            return ""
        try:
            decoded = base64.b64decode(body, validate=False)
        except (ValueError, TypeError):
            return body
        return decoded.decode("utf-8", errors="ignore")
    return _body_to_scan_text(body)


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
        if inter.request.body_encoding != "empty":
            findings.extend(
                _scan_text_for_secrets(
                    _body_to_secret_scan_text(inter.request.body, inter.request.body_encoding),
                    f"{prefix}.request.body",
                    patterns,
                )
            )
        for k, v in inter.response.headers.items():
            if v != "[REDACTED]":
                findings.extend(
                    _scan_text_for_secrets(v, f"{prefix}.response.headers.{k}", patterns)
                )
        if inter.response.body_encoding != "empty":
            findings.extend(
                _scan_text_for_secrets(
                    _body_to_secret_scan_text(inter.response.body, inter.response.body_encoding),
                    f"{prefix}.response.body",
                    patterns,
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
    check_pii: bool = False,
    secret_patterns: list[str] | None = None,
    as_json: bool = False,
) -> int:
    """Validate .tape.json(.gz) cassette(s) for schema integrity and optional secret/PII leaks."""
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
    if check_pii:
        for pii_name, pii_pat in PII_PRESETS.items():
            compiled_patterns.append((f"pii_{pii_name}", pii_pat))

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
        for issue in cassette.validate():
            schema_errors.append({"path": str(file_path), "error": issue})
            if not as_json:
                print(f"FAIL {file_path}: {issue}", file=sys.stderr)
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
    show_text_diff: bool = False,
) -> int:
    """Compare calls, tokens, cached tokens, cost, latency, and tools between cassettes or dirs."""
    if err := _validate_non_negative_float("--max-cost-increase", max_cost_increase):
        print(err, file=sys.stderr)
        return 1
    if err := _validate_non_negative_float("--max-cost-increase-pct", max_cost_increase_pct):
        print(err, file=sys.stderr)
        return 1
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

    cass_a = Cassette.load(path_a)
    cass_b = Cassette.load(path_b)
    stats_a = _cassette_stats(cass_a)
    stats_b = _cassette_stats(cass_b)
    text_diff_lines: list[str] = []
    if show_text_diff:
        a_lines = [
            line
            for idx, i in enumerate(cass_a.interactions, start=1)
            for line in [f"--- interaction[{idx}] ---", *i.llm.text.splitlines()]
        ]
        b_lines = [
            line
            for idx, i in enumerate(cass_b.interactions, start=1)
            for line in [f"--- interaction[{idx}] ---", *i.llm.text.splitlines()]
        ]
        text_diff_lines = list(
            difflib.unified_diff(
                a_lines,
                b_lines,
                fromfile=str(path_a),
                tofile=str(path_b),
                lineterm="",
            )
        )

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
        payload: dict[str, Any] = {
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
        }
        if show_text_diff:
            payload["text_diff"] = text_diff_lines
        print(json.dumps(payload, indent=2))
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
    if show_text_diff:
        if text_diff_lines:
            print("\nResponse Text Diff:")
            for line in text_diff_lines:
                print(line)
        else:
            print("\nResponse Text Diff: (identical)")
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
    url_patterns: list[str] | None = None,
    pii: str | None = None,
    dry_run: bool = False,
) -> int:
    """Re-scrub one .tape.json file or all cassettes under a directory."""
    if not target.exists():
        print(f"Error: path not found: {target}", file=sys.stderr)
        return 1

    try:
        pii_patterns = (
            resolve_pii_patterns([p.strip() for p in pii.split(",") if p.strip()]) if pii else ()
        )
        effective_patterns = (*compile_patterns(patterns), *pii_patterns)
        compiled_url_patterns = compile_patterns(url_patterns or ())
    except (ValueError, re.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
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
        new_metadata = (
            redact_body(cassette.metadata, redact_fields=fields, redact_patterns=effective_patterns)
            if cassette.metadata
            else cassette.metadata
        )
        if new_metadata != cassette.metadata:
            changed = True
        for inter in cassette.interactions:
            new_url = scrub_url(
                inter.request.url,
                extra_sensitive=query_params,
                redact_url_patterns=compiled_url_patterns,
            )
            new_req_headers = scrub_headers(inter.request.headers, extra_sensitive=headers)
            new_req_body = (
                redact_body(
                    inter.request.body, redact_fields=fields, redact_patterns=effective_patterns
                )
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
                redact_body(
                    inter.response.body, redact_fields=fields, redact_patterns=effective_patterns
                )
                if inter.response.body_encoding not in {"base64", "empty"}
                else inter.response.body
            )
            if inter.response.stream_chunks is not None and effective_patterns:
                new_chunks: tuple[str, ...] | None = tuple(
                    redact_stream_chunks(inter.response.stream_chunks, effective_patterns)
                )
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
                    content_type=_get_header_ci(new_resp.headers, "content-type", ""),
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

            new_llm_text = (
                redact_text(base_text, effective_patterns) if effective_patterns else base_text
            )
            new_llm_thinking = (
                redact_text(base_thinking, effective_patterns)
                if effective_patterns
                else base_thinking
            )
            new_tool_calls = tuple(
                ToolCall(
                    name=tc.name,
                    arguments=redact_body(
                        tc.arguments, redact_fields=fields, redact_patterns=effective_patterns
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
                if isinstance(new_metadata, dict):
                    cassette.metadata = new_metadata
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


def _render_junit(cassettes: list[Cassette], base_dir: Path) -> str:
    failures = 0
    total_time_s = 0.0
    suite = ET.Element("testsuite", name="llm-tape", tests=str(len(cassettes)))
    for c in cassettes:
        s = _cassette_stats(c)
        rel = str(c.path.relative_to(base_dir))
        t_sec = round(float(s["latency_ms"]) / 1000.0, 6)
        total_time_s += t_sec
        tc = ET.SubElement(
            suite,
            "testcase",
            classname="llm_tape",
            name=rel,
            time=f"{t_sec:.6f}",
        )
        err_codes = [
            f"{i.request.method} {i.request.url} -> {i.response.status_code}"
            for i in c.interactions
            if i.response.status_code >= 400
        ]
        if err_codes:
            failures += 1
            fail_el = ET.SubElement(
                tc,
                "failure",
                message=f"{len(err_codes)} HTTP error interaction(s)",
            )
            fail_el.text = "\n".join(err_codes)
        sys_out = ET.SubElement(tc, "system-out")
        sys_out.text = (
            f"calls={s['calls']} tokens={s['total_tokens']} cost_usd={s['estimated_cost_usd']:.6f}"
        )
    suite.set("failures", str(failures))
    suite.set("time", f"{total_time_s:.6f}")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(suite, encoding="unicode") + "\n"
    )


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
    elif fmt == "html":
        from llm_tape.html_report import render_html_report

        rendered = render_html_report(cassettes, title=f"llm-tape Trace ({target})")
    elif fmt == "junit":
        rendered = _render_junit(cassettes, base_dir)
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


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def _cmd_stats(target: Path, *, as_json: bool = False) -> int:
    """Aggregate LLM usage, cache hit rate, cost, and avg/p95 latency grouped by provider/model."""
    if not target.exists():
        print(f"Error: path not found: {target}", file=sys.stderr)
        return 1

    files = [target] if target.is_file() else iter_cassette_files(target)
    if not files:
        if as_json:
            print(json.dumps({"models": [], "errors": [], "totals": {}}, indent=2))
        else:
            print(f"No .tape.json cassettes found in {target}")
        return 0

    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    all_latencies: list[float] = []

    for file_path in files:
        try:
            cassette = Cassette.load(file_path)
        except CorruptCassetteError as exc:
            errors.append({"path": str(file_path), "error": str(exc)})
            if not as_json:
                print(f"Warning: skipping corrupt cassette {file_path}: {exc}", file=sys.stderr)
            continue
        for inter in cassette.interactions:
            key = (inter.llm.provider, inter.llm.model or "<unknown>")
            b = buckets.setdefault(
                key,
                {
                    "provider": key[0],
                    "model": key[1],
                    "calls": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "cache_savings_usd": 0.0,
                    "latencies": [],
                },
            )
            b["calls"] += 1
            b["input_tokens"] += inter.llm.usage.input_tokens
            b["cached_input_tokens"] += inter.llm.usage.cached_input_tokens
            b["output_tokens"] += inter.llm.usage.output_tokens
            b["reasoning_tokens"] += inter.llm.usage.reasoning_tokens
            b["total_tokens"] += inter.llm.usage.total_tokens
            b["estimated_cost_usd"] += inter.llm.estimated_cost_usd
            b["cache_savings_usd"] += estimate_cache_savings_usd(
                inter.llm.model,
                inter.llm.usage.input_tokens,
                inter.llm.usage.output_tokens,
                cached_input_tokens=inter.llm.usage.cached_input_tokens,
                cache_creation_tokens=inter.llm.usage.cache_creation_tokens,
            )
            b["latencies"].append(inter.latency_ms)
            all_latencies.append(inter.latency_ms)

    rows: list[dict[str, Any]] = []
    for b in buckets.values():
        lats: list[float] = b["latencies"]
        inp = int(b["input_tokens"])
        cached = int(b["cached_input_tokens"])
        cache_pct = round((cached / inp) * 100.0, 1) if inp > 0 else 0.0
        rows.append(
            {
                "provider": b["provider"],
                "model": b["model"],
                "calls": b["calls"],
                "input_tokens": inp,
                "cached_input_tokens": cached,
                "cache_hit_pct": cache_pct,
                "output_tokens": b["output_tokens"],
                "reasoning_tokens": b["reasoning_tokens"],
                "total_tokens": b["total_tokens"],
                "estimated_cost_usd": round(float(b["estimated_cost_usd"]), 8),
                "cache_savings_usd": round(float(b["cache_savings_usd"]), 8),
                "avg_latency_ms": round(sum(lats) / len(lats), 2) if lats else 0.0,
                "p95_latency_ms": round(_percentile(lats, 95.0), 2),
            }
        )
    rows.sort(key=lambda r: (-float(r["estimated_cost_usd"]), -int(r["calls"]), str(r["model"])))

    total_calls = sum(int(r["calls"]) for r in rows)
    total_input = sum(int(r["input_tokens"]) for r in rows)
    total_cached = sum(int(r["cached_input_tokens"]) for r in rows)
    total_tokens = sum(int(r["total_tokens"]) for r in rows)
    total_reasoning = sum(int(r["reasoning_tokens"]) for r in rows)
    total_cost = round(sum(float(r["estimated_cost_usd"]) for r in rows), 8)
    total_cache_savings = round(sum(float(r["cache_savings_usd"]) for r in rows), 8)
    totals = {
        "files": len(files) - len(errors),
        "calls": total_calls,
        "input_tokens": total_input,
        "cached_input_tokens": total_cached,
        "cache_hit_pct": round((total_cached / total_input) * 100.0, 1) if total_input > 0 else 0.0,
        "reasoning_tokens": total_reasoning,
        "total_tokens": total_tokens,
        "estimated_cost_usd": total_cost,
        "cache_savings_usd": total_cache_savings,
        "avg_latency_ms": (
            round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0.0
        ),
        "p95_latency_ms": round(_percentile(all_latencies, 95.0), 2),
    }

    if as_json:
        print(json.dumps({"models": rows, "errors": errors, "totals": totals}, indent=2))
        return 1 if errors else 0

    print(
        f"{'Provider':<11} {'Model':<26} {'Calls':>5} {'Tokens':>9} "
        f"{'Cache%':>6} {'Cost':>11} {'Avg ms':>8} {'p95 ms':>8}"
    )
    print("-" * 91)
    for r in rows:
        model_label = str(r["model"])
        if len(model_label) > 26:
            model_label = model_label[:23] + "..."
        print(
            f"{r['provider']:<11} {model_label:<26} {r['calls']:>5} {r['total_tokens']:>9} "
            f"{r['cache_hit_pct']:>5.1f}% ${r['estimated_cost_usd']:>10.6f} "
            f"{r['avg_latency_ms']:>8.1f} {r['p95_latency_ms']:>8.1f}"
        )
    print("-" * 91)
    print(
        f"{'TOTAL':<38} {total_calls:>5} {total_tokens:>9} "
        f"{totals['cache_hit_pct']:>5.1f}% ${total_cost:>10.6f} "
        f"{totals['avg_latency_ms']:>8.1f} {totals['p95_latency_ms']:>8.1f}"
    )
    if total_cache_savings:
        print(f"Estimated cache savings: ${total_cache_savings:.6f}")
    return 1 if errors else 0


def _cmd_pricing(*, prefix: str | None = None, as_json: bool = False) -> int:
    """Display effective built-in (+ LLM_TAPE_PRICING_FILE) model pricing table."""
    table: dict[str, ModelPrice] = dict(DEFAULT_PRICING)
    env_file = os.environ.get(PRICING_FILE_ENV)
    if env_file and Path(env_file).is_file():
        try:
            table.update(load_pricing_file(env_file))
        except (OSError, ValueError) as exc:
            print(f"Warning: failed to load {PRICING_FILE_ENV}: {exc}", file=sys.stderr)

    needle = prefix.lower().strip() if prefix else ""
    items = [
        (name, price)
        for name, price in sorted(table.items())
        if not needle or needle in name.lower()
    ]
    if as_json:
        payload = {
            name: {
                "input_per_million": price.input_per_million,
                "output_per_million": price.output_per_million,
                "cached_input_per_million": price.cached_input_per_million,
                "cache_write_per_million": price.cache_write_per_million,
            }
            for name, price in items
        }
        print(json.dumps(payload, indent=2))
        return 0

    if not items:
        print(f"No models matching {prefix!r}.")
        return 0

    print(
        f"{'Model':<28} {'Input / 1M':>11} {'Output / 1M':>12} "
        f"{'Cached / 1M':>12} {'Write / 1M':>11}"
    )
    print("-" * 78)
    for name, price in items:
        cached_s = (
            f"${price.cached_input_per_million:>10.4f}"
            if price.cached_input_per_million is not None
            else f"{'-':>11}"
        )
        write_s = (
            f"${price.cache_write_per_million:>9.4f}"
            if price.cache_write_per_million is not None
            else f"{'-':>10}"
        )
        print(
            f"{name:<28} ${price.input_per_million:>10.4f} "
            f"${price.output_per_million:>11.4f} {cached_s:>12} {write_s:>11}"
        )
    return 0


def _cmd_search(
    directory: Path,
    *,
    model: str | None = None,
    provider: str | None = None,
    tool: str | None = None,
    text: str | None = None,
    min_cost: float | None = None,
    status: int | None = None,
    as_json: bool = False,
) -> int:
    """Search interactions across all cassettes under a file or directory."""
    if err := _validate_non_negative_float("--min-cost", min_cost):
        print(err, file=sys.stderr)
        return 1
    if not directory.exists():
        print(f"Error: path not found: {directory}", file=sys.stderr)
        return 1
    files = [directory] if directory.is_file() else iter_cassette_files(directory)
    model_q = model.lower() if model else None
    provider_q = provider.lower() if provider else None
    tool_q = tool.lower() if tool else None
    text_q = text.lower() if text else None

    matches: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for file_path in files:
        rel = str(file_path.relative_to(directory)) if directory.is_dir() else file_path.name
        try:
            cassette = Cassette.load(file_path)
        except CorruptCassetteError as exc:
            errors.append({"path": rel, "error": str(exc)})
            if not as_json:
                print(f"Warning: skipping corrupt cassette {rel}: {exc}", file=sys.stderr)
            continue
        for idx, inter in enumerate(cassette.interactions, start=1):
            if model_q and model_q not in (inter.llm.model or "").lower():
                continue
            if provider_q and provider_q != inter.llm.provider.lower():
                continue
            if tool_q and not any(tool_q in tc.name.lower() for tc in inter.llm.tool_calls):
                continue
            if text_q and text_q not in inter.llm.text.lower():
                continue
            if min_cost is not None and inter.llm.estimated_cost_usd < min_cost:
                continue
            if status is not None and inter.response.status_code != status:
                continue
            matches.append(
                {
                    "cassette": rel,
                    "index": idx,
                    "provider": inter.llm.provider,
                    "model": inter.llm.model,
                    "status_code": inter.response.status_code,
                    "tokens": inter.llm.usage.total_tokens,
                    "estimated_cost_usd": inter.llm.estimated_cost_usd,
                    "tools": [tc.name for tc in inter.llm.tool_calls],
                    "text_preview": inter.llm.text[:80],
                }
            )

    if as_json:
        print(json.dumps({"matches": matches, "count": len(matches), "errors": errors}, indent=2))
        return 0
    if not matches:
        print("No matching interactions found.")
        return 0
    for m in matches:
        tools_s = ",".join(m["tools"]) or "-"
        print(
            f"{m['cassette']}[{m['index']}] {m['provider']}/{m['model'] or '-'} "
            f"status={m['status_code']} tok={m['tokens']} "
            f"cost=${m['estimated_cost_usd']:.6f} tools=[{tools_s}]"
        )
    print(f"\nFound {len(matches)} matching interaction(s).")
    return 0


def _cmd_replay_check(
    target: Path,
    *,
    ignore_fields: list[str] | None = None,
    ignore_query_params: list[str] | None = None,
    match_on: list[str] | None = None,
    match_headers: list[str] | None = None,
    as_json: bool = False,
) -> int:
    """Verify stored request fingerprints match recomputed fingerprints across cassettes."""
    if not target.exists():
        print(f"Error: path not found: {target}", file=sys.stderr)
        return 1
    effective_ignore_fields = _parse_csv_flags(ignore_fields)
    effective_ignore_query_params = _parse_csv_flags(ignore_query_params)
    effective_match_headers = _parse_csv_flags(match_headers)
    if match_on:
        try:
            effective_match_on = validate_match_on(_parse_csv_flags(match_on))
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        effective_match_on = DEFAULT_MATCH_ON
    files = [target] if target.is_file() else iter_cassette_files(target)
    if not files:
        if as_json:
            print(json.dumps({"checked": 0, "mismatches": [], "errors": []}, indent=2))
        else:
            print(f"No .tape.json cassettes found in {target}")
        return 0

    checked = 0
    mismatches: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for file_path in files:
        rel = str(file_path.relative_to(target)) if target.is_dir() else file_path.name
        try:
            cassette = Cassette.load(file_path)
        except CorruptCassetteError as exc:
            errors.append({"cassette": rel, "error": str(exc)})
            continue
        for idx, inter in enumerate(cassette.interactions, start=1):
            checked += 1
            expected_fp = fingerprint_request(
                inter.request.method,
                inter.request.url,
                inter.request.body,
                headers=inter.request.headers,
                ignore_fields=effective_ignore_fields,
                ignore_query_params=effective_ignore_query_params,
                match_on=effective_match_on,
                match_headers=effective_match_headers,
            )
            if inter.request.fingerprint != expected_fp:
                mismatches.append(
                    {
                        "cassette": rel,
                        "index": idx,
                        "stored": inter.request.fingerprint,
                        "expected": expected_fp,
                    }
                )

    if as_json:
        print(
            json.dumps({"checked": checked, "mismatches": mismatches, "errors": errors}, indent=2)
        )
        return 1 if (mismatches or errors) else 0

    for err in errors:
        print(f"ERROR: {err['cassette']}: {err['error']}", file=sys.stderr)
    for m in mismatches:
        print(
            f"MISMATCH: {m['cassette']}[{m['index']}] "
            f"stored={m['stored']} expected={m['expected']}",
            file=sys.stderr,
        )
    if mismatches or errors:
        return 1
    print(f"OK: {checked} interaction(s) across {len(files)} cassette(s) have valid fingerprints.")
    return 0


def _cmd_reparse(target: Path, *, dry_run: bool = False) -> int:
    """Re-run LLM metadata and pricing parsers across cassette(s)."""
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
            ct = ""
            for k, v in inter.response.headers.items():
                if k.lower() == "content-type":
                    ct = v
                    break
            reparsed = parse_llm_interaction(
                url=inter.request.url,
                request_body=inter.request.body,
                response_body=inter.response.body,
                stream_chunks=inter.response.stream_chunks,
                content_type=ct,
            )
            if reparsed != inter.llm:
                changed = True
                new_interactions.append(
                    Interaction(
                        request=inter.request,
                        response=inter.response,
                        latency_ms=inter.latency_ms,
                        llm=reparsed,
                    )
                )
            else:
                new_interactions.append(inter)
        if changed:
            updated += 1
            if dry_run:
                print(f"WOULD REPARSE: {file_path}")
            else:
                cassette.interactions = new_interactions
                cassette.save()
                print(f"REPARSED: {file_path}")

    if dry_run:
        print(f"Dry run: {updated} of {len(files)} cassette(s) would be updated.")
    else:
        print(f"Reparsed {updated} of {len(files)} cassette(s).")
    return 0


def _cmd_list(
    directory: Path,
    *,
    sort_by: str = "cost",
    limit: int | None = None,
    as_json: bool = False,
) -> int:
    """List cassettes sorted by cost, tokens, calls, latency, or name."""
    if not directory.exists():
        print(f"Error: path not found: {directory}", file=sys.stderr)
        return 1
    if limit is not None and limit < 0:
        print("Error: --limit must be >= 0.", file=sys.stderr)
        return 1

    files = [directory] if directory.is_file() else iter_cassette_files(directory)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for file_path in files:
        rel = str(file_path.relative_to(directory)) if directory.is_dir() else file_path.name
        try:
            cassette = Cassette.load(file_path)
        except CorruptCassetteError as exc:
            errors.append({"path": rel, "error": str(exc)})
            if not as_json:
                print(f"Warning: skipping corrupt cassette {rel}: {exc}", file=sys.stderr)
            continue
        s = _cassette_stats(cassette)
        s["relative_path"] = rel
        rows.append(s)

    if sort_by == "name":
        rows.sort(key=lambda r: str(r["relative_path"]))
    elif sort_by == "tokens":
        rows.sort(key=lambda r: (-int(r["total_tokens"]), str(r["relative_path"])))
    elif sort_by == "calls":
        rows.sort(key=lambda r: (-int(r["calls"]), str(r["relative_path"])))
    elif sort_by == "latency":
        rows.sort(key=lambda r: (-float(r["latency_ms"]), str(r["relative_path"])))
    else:
        rows.sort(key=lambda r: (-float(r["estimated_cost_usd"]), str(r["relative_path"])))

    if limit is not None:
        rows = rows[:limit]

    if as_json:
        print(json.dumps({"cassettes": rows, "count": len(rows), "errors": errors}, indent=2))
        return 0
    if not rows:
        print(f"No .tape.json cassettes found in {directory}")
        return 0
    print(f"{'Cassette':<48} {'Calls':>5} {'Tokens':>9} {'Cost':>11} {'Latency':>9}")
    print("-" * 86)
    for r in rows:
        name = str(r["relative_path"])
        if len(name) > 48:
            name = "..." + name[-45:]
        print(
            f"{name:<48} {r['calls']:>5} {r['total_tokens']:>9} "
            f"${r['estimated_cost_usd']:>10.6f} {r['latency_ms']:>7.1f}ms"
        )
    return 0


def _cmd_schema(*, output: Path | None = None) -> int:
    """Output the Draft 2020-12 JSON Schema for .tape.json cassettes."""
    rendered = json.dumps(cassette_json_schema(), indent=2, ensure_ascii=False) + "\n"
    if output is not None:
        _write_text_atomic(output, rendered)
        print(f"Wrote JSON Schema to {output}.")
    else:
        sys.stdout.write(rendered)
    return 0


def _cmd_merge(
    sources: list[Path],
    *,
    output: Path,
    deduplicate: bool = False,
) -> int:
    """Merge multiple cassette files or directories into a single output cassette."""
    output_resolved = output.resolve()
    seen_files: set[Path] = set()
    files: list[Path] = []
    for src in sources:
        if not src.exists():
            print(f"Error: path not found: {src}", file=sys.stderr)
            return 1
        candidates = [src] if src.is_file() else iter_cassette_files(src)
        for cand in candidates:
            resolved = cand.resolve()
            if resolved == output_resolved or resolved in seen_files:
                continue
            seen_files.add(resolved)
            files.append(cand)
    if not files:
        print("Error: no cassette files found to merge.", file=sys.stderr)
        return 1
    cassettes = [Cassette.load(p) for p in files]
    merged = Cassette.merge(*cassettes, output_path=output, deduplicate=deduplicate)
    merged.save()
    print(
        f"Merged {len(files)} cassette(s) ({len(merged.interactions)} interaction(s)) -> {output}."
    )
    return 0


def _cmd_split(
    source: Path,
    *,
    output_dir: Path,
    compress: bool = False,
) -> int:
    """Split a multi-interaction cassette into individual single-interaction cassettes."""
    if not source.exists() or not source.is_file():
        print(f"Error: cassette file not found: {source}", file=sys.stderr)
        return 1
    cassette = Cassette.load(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _cassette_stem(source)
    ext = ".tape.json.gz" if compress else ".tape.json"
    width = max(3, len(str(len(cassette.interactions))))
    for idx, inter in enumerate(cassette.interactions, start=1):
        out_path = output_dir / f"{stem}_{idx:0{width}d}{ext}"
        part = Cassette(
            path=out_path,
            interactions=[inter],
            recorded_at=cassette.recorded_at,
            metadata=dict(cassette.metadata),
        )
        part.save()
    print(f"Split {source} into {len(cassette.interactions)} cassette(s) under {output_dir}.")
    return 0


def _build_cli_replay_server(
    source: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    mode: str = "replay",
    upstream: str | None = None,
    enable_cors: bool = False,
    max_request_bytes: int = 10 * 1024 * 1024,
    redact_pii: str | None = None,
    allow_repeats: bool = True,
    latency_scale: float = 0.0,
    ignore_fields: list[str] | None = None,
    match_on: list[str] | None = None,
    auto_save: bool = True,
) -> Any:
    from llm_tape.models import TapeMode
    from llm_tape.server import TapeReplayServer

    coerced_mode = TapeMode.coerce(mode)
    if coerced_mode is TapeMode.REPLAY and not source.exists():
        raise FileNotFoundError(f"path not found: {source}")

    effective_ignore_fields = _parse_csv_flags(ignore_fields)
    effective_match_on = (
        validate_match_on(_parse_csv_flags(match_on)) if match_on else ("method", "path", "body")
    )

    effective_pii: bool | list[str]
    if redact_pii is None:
        effective_pii = False
    else:
        parsed_pii = [p.strip() for p in redact_pii.split(",") if p.strip()]
        effective_pii = parsed_pii if parsed_pii else True

    return TapeReplayServer(
        source,
        host=host,
        port=port,
        mode=coerced_mode,
        upstream_url=upstream,
        enable_cors=enable_cors,
        max_request_bytes=max_request_bytes,
        redact_pii=effective_pii,
        allow_playback_repeats=allow_repeats,
        ignore_fields=effective_ignore_fields,
        match_on=effective_match_on,
        replay_latency_scale=latency_scale,
        auto_save=auto_save,
    )


def _cmd_serve(
    source: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    mode: str = "replay",
    upstream: str | None = None,
    enable_cors: bool = False,
    max_request_bytes: int = 10 * 1024 * 1024,
    redact_pii: str | None = None,
    allow_repeats: bool = True,
    latency_scale: float = 0.0,
    ignore_fields: list[str] | None = None,
    match_on: list[str] | None = None,
) -> int:
    """Start a local HTTP replay/reverse-proxy server serving cassette interaction(s)."""
    try:
        server = _build_cli_replay_server(
            source,
            host=host,
            port=port,
            mode=mode,
            upstream=upstream,
            enable_cors=enable_cors,
            max_request_bytes=max_request_bytes,
            redact_pii=redact_pii,
            allow_repeats=allow_repeats,
            latency_scale=latency_scale,
            ignore_fields=ignore_fields,
            match_on=match_on,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    stop_event = threading.Event()
    prev_sigterm: Any = None
    sigterm_installed = False

    def _handle_sigterm(signum: int, frame: Any) -> None:
        del signum, frame
        stop_event.set()

    if hasattr(signal, "SIGTERM") and threading.current_thread() is threading.main_thread():
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _handle_sigterm)
        sigterm_installed = True

    server.start()
    st = server.stats()
    print(f"Serving {st['loaded_interactions']} interaction(s) at {server.url} (Ctrl+C to stop)...")
    try:
        while not stop_event.wait(timeout=0.5):
            pass
        print("\nStopping llm-tape replay server.")
    except KeyboardInterrupt:
        print("\nStopping llm-tape replay server.")
    finally:
        if sigterm_installed:
            signal.signal(signal.SIGTERM, prev_sigterm)
        server.stop()
    return 0


def _cmd_run(
    source: Path,
    command: list[str],
    *,
    mode: str = "replay",
    upstream: str | None = None,
    enable_cors: bool = False,
    redact_pii: str | None = None,
    allow_repeats: bool = True,
    latency_scale: float = 0.0,
    ignore_fields: list[str] | None = None,
    match_on: list[str] | None = None,
    max_cost: float | None = None,
    max_tokens: int | None = None,
    strict_unused: bool = False,
    save_on_error: bool = False,
    inject_env: bool = True,
    quiet: bool = False,
) -> int:
    """Run any command against an ephemeral local TapeReplayServer with zero code changes."""
    from llm_tape.models import TapeMode

    cmd = list(command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("Error: specify a command to run after `--`.", file=sys.stderr)
        return 2
    if err := _validate_non_negative_float("--max-cost", max_cost):
        print(err, file=sys.stderr)
        return 1
    if max_tokens is not None and max_tokens < 0:
        print("Error: --max-tokens must be >= 0.", file=sys.stderr)
        return 1

    try:
        server = _build_cli_replay_server(
            source,
            host="127.0.0.1",
            port=0,
            mode=mode,
            upstream=upstream,
            enable_cors=enable_cors,
            redact_pii=redact_pii,
            allow_repeats=allow_repeats,
            latency_scale=latency_scale,
            ignore_fields=ignore_fields,
            match_on=match_on,
            auto_save=False,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    server.start()
    try:
        env = dict(os.environ)
        env["LLM_TAPE_URL"] = server.url
        if inject_env:
            env["OPENAI_BASE_URL"] = f"{server.url}/v1"
            env["ANTHROPIC_BASE_URL"] = server.url
            env["GOOGLE_GEMINI_BASE_URL"] = server.url
            env["OLLAMA_HOST"] = server.url
            if server.mode is TapeMode.REPLAY:
                env.setdefault("OPENAI_API_KEY", "llm-tape-replay")
                env.setdefault("ANTHROPIC_API_KEY", "llm-tape-replay")
                env.setdefault("GEMINI_API_KEY", "llm-tape-replay")
        try:
            proc = subprocess.run(cmd, env=env, check=False)
        except OSError as exc:
            print(f"Error: failed to launch command {cmd[0]!r}: {exc}", file=sys.stderr)
            return 1

        if (proc.returncode == 0 or save_on_error) and server._dirty:
            if server._active_cassette_path is not None:
                server.save_cassette()

        spec: dict[str, Any] = {}
        if max_cost is not None:
            spec["max_cost_usd"] = max_cost
        if max_tokens is not None:
            spec["max_tokens"] = max_tokens
        if strict_unused:
            spec["no_unused_interactions"] = True

        st = server.stats()
        if not quiet:
            calls = int(st["served_requests"]) + int(st["recorded_interactions"])
            print(
                f"[llm-tape run] calls={calls} tokens={st['total_tokens']} "
                f"cost=${st['estimated_cost_usd']:.6f}",
                file=sys.stderr,
            )

        if proc.returncode != 0:
            return int(proc.returncode)

        if spec:
            code, res = server.evaluate_assertions(spec)
            if code != 200:
                for msg in res.get("failures", []):
                    print(f"[llm-tape run] FAILED: {msg}", file=sys.stderr)
                return 1
        return 0
    finally:
        server.stop()


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

    p_inspect = sub.add_parser("inspect", help="Inspect a single .tape.json(.gz) cassette.")
    p_inspect.add_argument("path", type=Path, help="Path to the cassette file.")
    p_inspect.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")
    p_inspect.add_argument(
        "--show-bodies",
        action="store_true",
        help="Include request and response bodies in inspect output.",
    )
    p_inspect.add_argument(
        "--show-thinking",
        action="store_true",
        help="Include parsed LLM thinking and response text in inspect output.",
    )
    p_inspect.add_argument(
        "--interaction",
        type=int,
        default=None,
        dest="interaction_index",
        help="Inspect only the given 1-based interaction index.",
    )

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
    p_summary.add_argument(
        "--group-by",
        choices=["file", "dir"],
        default="file",
        help="Group summary rows by file (default) or parent directory.",
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
        "--check-pii",
        action="store_true",
        help="Scan cassettes for unredacted PII (email, phone, SSN, credit card, IPv4, JWT).",
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
    p_diff.add_argument(
        "--show-text-diff",
        action="store_true",
        help="Include unified diff of parsed LLM response text (file vs file).",
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
        "--url-pattern",
        action="append",
        default=[],
        dest="url_patterns",
        help="Regex pattern to redact in request URL paths (repeatable).",
    )
    p_redact.add_argument(
        "--pii",
        nargs="?",
        const="all",
        default=None,
        help=(
            "Redact built-in PII presets (omit value for all, or comma-separated: "
            "email,phone,ssn,credit_card,ipv4,jwt)."
        ),
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
        choices=["markdown", "csv", "har", "html", "junit"],
        default="markdown",
        dest="fmt",
        help="Output format: markdown (default), csv, har, html, or junit.",
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

    p_stats = sub.add_parser(
        "stats",
        help="Show per-provider/model token, cache-hit, cost, and avg/p95 latency breakdown.",
    )
    p_stats.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Cassette file or directory to analyze (default: current directory).",
    )
    p_stats.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")

    p_schema = sub.add_parser(
        "schema",
        help="Output the Draft 2020-12 JSON Schema for .tape.json cassettes.",
    )
    p_schema.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write schema JSON (defaults to stdout).",
    )

    p_merge = sub.add_parser(
        "merge",
        help="Merge multiple cassettes (or directories) into a single cassette file.",
    )
    p_merge.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="Source cassette files or directories to merge in order.",
    )
    p_merge.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Destination cassette path (.tape.json or .tape.json.gz).",
    )
    p_merge.add_argument(
        "--deduplicate",
        action="store_true",
        help="Drop duplicate interactions sharing the same request fingerprint.",
    )

    p_split = sub.add_parser(
        "split",
        help="Split a multi-interaction cassette into one cassette file per interaction.",
    )
    p_split.add_argument("source", type=Path, help="Source cassette file to split.")
    p_split.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for split cassette files.",
    )
    p_split.add_argument(
        "--gzip",
        action="store_true",
        dest="compress",
        help="Write split cassettes as .tape.json.gz.",
    )

    from llm_tape.models import TapeMode

    p_serve = sub.add_parser(
        "serve",
        help="Start a local HTTP replay/reverse-proxy server backed by a cassette or directory.",
    )
    p_serve.add_argument("source", type=Path, help="Cassette file or directory to serve.")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    p_serve.add_argument("--port", type=int, default=8787, help="Bind port (default: 8787).")
    p_serve.add_argument(
        "--mode",
        choices=[m.value for m in TapeMode if m != TapeMode.NONE],
        default="replay",
        help="Server mode: replay (default), once, record, or new_episodes.",
    )
    p_serve.add_argument(
        "--upstream",
        default=None,
        help="Upstream base URL (e.g. https://api.openai.com) for reverse-proxy recording.",
    )
    p_serve.add_argument(
        "--cors",
        action="store_true",
        help="Enable permissive CORS headers and 204 OPTIONS preflight responses.",
    )
    p_serve.add_argument(
        "--max-request-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="Maximum incoming request body size in bytes (default: 10485760).",
    )
    p_serve.add_argument(
        "--redact-pii",
        nargs="?",
        const="all",
        default=None,
        help=(
            "Redact built-in PII presets when recording (omit value for all, or comma-separated: "
            "email,phone,ssn,credit_card,ipv4,jwt)."
        ),
    )
    p_serve.add_argument(
        "--no-repeats",
        action="store_true",
        help="Consume each recorded interaction only once.",
    )
    p_serve.add_argument(
        "--latency-scale",
        type=float,
        default=0.0,
        help="Scale factor for replaying recorded latencies (default: 0.0).",
    )
    p_serve.add_argument(
        "--ignore-fields",
        action="append",
        default=[],
        dest="ignore_fields",
        help=(
            "JSON dot-path field(s) to ignore when matching requests "
            "(repeatable or comma-separated)."
        ),
    )
    p_serve.add_argument(
        "--match-on",
        action="append",
        default=[],
        dest="match_on",
        help="Request matching keys (repeatable or comma-separated; default: method,path,body).",
    )

    p_run = sub.add_parser(
        "run",
        help="Run any command against an ephemeral local TapeReplayServer (zero code changes).",
    )
    p_run.add_argument("source", type=Path, help="Cassette file or directory.")
    p_run.add_argument(
        "--mode",
        choices=[m.value for m in TapeMode if m != TapeMode.NONE],
        default="replay",
        help="Server mode: replay (default), once, record, or new_episodes.",
    )
    p_run.add_argument("--upstream", default=None, help="Upstream base URL for recording.")
    p_run.add_argument("--cors", action="store_true", help="Enable CORS headers and preflight.")
    p_run.add_argument(
        "--redact-pii",
        nargs="?",
        const="all",
        default=None,
        help="Redact built-in PII presets when recording.",
    )
    p_run.add_argument(
        "--no-repeats", action="store_true", help="Consume each recorded interaction only once."
    )
    p_run.add_argument(
        "--latency-scale",
        type=float,
        default=0.0,
        help="Scale factor for replaying recorded latencies.",
    )
    p_run.add_argument("--ignore-fields", action="append", default=[], dest="ignore_fields")
    p_run.add_argument("--match-on", action="append", default=[], dest="match_on")
    p_run.add_argument("--max-cost", type=float, default=None, help="Fail if cost exceeds USD.")
    p_run.add_argument("--max-tokens", type=int, default=None, help="Fail if tokens exceed limit.")
    p_run.add_argument(
        "--strict-unused", action="store_true", help="Fail if any recorded interactions go unused."
    )
    p_run.add_argument(
        "--save-on-error", action="store_true", help="Save cassette even if child command fails."
    )
    p_run.add_argument(
        "--no-inject-env",
        action="store_true",
        help="Only set LLM_TAPE_URL; do not override provider *_BASE_URL env vars.",
    )
    p_run.add_argument("--quiet", action="store_true", help="Suppress summary line on stderr.")
    p_run.add_argument("cmd", nargs="*", help="Command to execute after `--`.")

    p_pricing = sub.add_parser(
        "pricing",
        help="Display built-in (and LLM_TAPE_PRICING_FILE) USD per 1M token pricing table.",
    )
    p_pricing.add_argument(
        "--filter",
        dest="prefix",
        default=None,
        help="Filter models by substring (case-insensitive).",
    )
    p_pricing.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")

    p_search = sub.add_parser(
        "search",
        help="Search recorded interactions by model, provider, tool, text, cost, or HTTP status.",
    )
    p_search.add_argument("directory", type=Path, nargs="?", default=Path("."))
    p_search.add_argument("--model", default=None, help="Filter by model substring.")
    p_search.add_argument("--provider", default=None, help="Filter by provider name.")
    p_search.add_argument("--tool", default=None, help="Filter by tool name substring.")
    p_search.add_argument("--text", default=None, help="Filter by response text substring.")
    p_search.add_argument("--min-cost", type=float, default=None, dest="min_cost")
    p_search.add_argument("--status", type=int, default=None, help="Filter by HTTP status code.")
    p_search.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")

    p_rcheck = sub.add_parser(
        "replay-check",
        help="Verify stored request fingerprints match recomputed fingerprints.",
    )
    p_rcheck.add_argument("target", type=Path, nargs="?", default=Path("."))
    p_rcheck.add_argument(
        "--ignore-fields",
        action="append",
        default=[],
        dest="ignore_fields",
        help="JSON dot-path field(s) ignored during recording (repeatable or comma-separated).",
    )
    p_rcheck.add_argument(
        "--ignore-query-params",
        action="append",
        default=[],
        dest="ignore_query_params",
        help="Query parameter(s) ignored during recording (repeatable or comma-separated).",
    )
    p_rcheck.add_argument(
        "--match-on",
        action="append",
        default=[],
        dest="match_on",
        help="Request matching keys used during recording (repeatable or comma-separated).",
    )
    p_rcheck.add_argument(
        "--match-headers",
        action="append",
        default=[],
        dest="match_headers",
        help=(
            "Header name(s) included in matching during recording (repeatable or comma-separated)."
        ),
    )
    p_rcheck.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")

    p_reparse = sub.add_parser(
        "reparse",
        help="Re-run LLM metadata and cost parsers on existing cassettes.",
    )
    p_reparse.add_argument("target", type=Path, nargs="?", default=Path("."))
    p_reparse.add_argument(
        "--dry-run",
        action="store_true",
        help="Report cassettes that would change without saving.",
    )

    p_list = sub.add_parser(
        "list",
        help="List cassettes sorted by cost, tokens, calls, latency, or name.",
    )
    p_list.add_argument("directory", type=Path, nargs="?", default=Path("."))
    p_list.add_argument(
        "--sort",
        dest="sort_by",
        choices=["cost", "tokens", "calls", "latency", "name"],
        default="cost",
        help="Sort order (default: cost descending).",
    )
    p_list.add_argument("--limit", type=int, default=None, help="Maximum number of rows to show.")
    p_list.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _cmd_inspect(
                args.path,
                as_json=args.as_json,
                show_bodies=args.show_bodies,
                show_thinking=args.show_thinking,
                interaction_index=args.interaction_index,
            )
        if args.command == "summary":
            return _cmd_summary(
                args.directory,
                as_json=args.as_json,
                max_cost=args.max_cost,
                max_tokens=args.max_tokens,
                group_by=args.group_by,
            )
        if args.command == "verify":
            return _cmd_verify(
                args.targets,
                check_secrets=args.check_secrets,
                check_pii=args.check_pii,
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
                show_text_diff=args.show_text_diff,
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
                url_patterns=args.url_patterns,
                pii=args.pii,
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
        if args.command == "stats":
            return _cmd_stats(args.target, as_json=args.as_json)
        if args.command == "schema":
            return _cmd_schema(output=args.output)
        if args.command == "merge":
            return _cmd_merge(args.sources, output=args.output, deduplicate=args.deduplicate)
        if args.command == "split":
            return _cmd_split(args.source, output_dir=args.output_dir, compress=args.compress)
        if args.command == "serve":
            return _cmd_serve(
                args.source,
                host=args.host,
                port=args.port,
                mode=args.mode,
                upstream=args.upstream,
                enable_cors=args.cors,
                max_request_bytes=args.max_request_bytes,
                redact_pii=args.redact_pii,
                allow_repeats=not args.no_repeats,
                latency_scale=args.latency_scale,
                ignore_fields=args.ignore_fields,
                match_on=args.match_on,
            )
        if args.command == "run":
            return _cmd_run(
                args.source,
                args.cmd,
                mode=args.mode,
                upstream=args.upstream,
                enable_cors=args.cors,
                redact_pii=args.redact_pii,
                allow_repeats=not args.no_repeats,
                latency_scale=args.latency_scale,
                ignore_fields=args.ignore_fields,
                match_on=args.match_on,
                max_cost=args.max_cost,
                max_tokens=args.max_tokens,
                strict_unused=args.strict_unused,
                save_on_error=args.save_on_error,
                inject_env=not args.no_inject_env,
                quiet=args.quiet,
            )
        if args.command == "pricing":
            return _cmd_pricing(prefix=args.prefix, as_json=args.as_json)
        if args.command == "search":
            return _cmd_search(
                args.directory,
                model=args.model,
                provider=args.provider,
                tool=args.tool,
                text=args.text,
                min_cost=args.min_cost,
                status=args.status,
                as_json=args.as_json,
            )
        if args.command == "replay-check":
            return _cmd_replay_check(
                args.target,
                ignore_fields=args.ignore_fields,
                ignore_query_params=args.ignore_query_params,
                match_on=args.match_on,
                match_headers=args.match_headers,
                as_json=args.as_json,
            )
        if args.command == "reparse":
            return _cmd_reparse(args.target, dry_run=args.dry_run)
        if args.command == "list":
            return _cmd_list(
                args.directory,
                sort_by=args.sort_by,
                limit=args.limit,
                as_json=args.as_json,
            )
    except (TapeError, re.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
