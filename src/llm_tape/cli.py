"""Command-line interface for inspecting, verifying, and diffing llm-tape cassettes."""

from __future__ import annotations

import argparse
import json
import re
import sys
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


def _cassette_stats(cassette: Cassette) -> dict[str, Any]:
    return {
        "path": str(cassette.path),
        "recorded_at": cassette.recorded_at,
        "calls": len(cassette.interactions),
        "input_tokens": sum(i.llm.usage.input_tokens for i in cassette.interactions),
        "output_tokens": sum(i.llm.usage.output_tokens for i in cassette.interactions),
        "total_tokens": sum(i.llm.usage.total_tokens for i in cassette.interactions),
        "estimated_cost_usd": round(
            sum(i.llm.estimated_cost_usd for i in cassette.interactions), 8
        ),
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


def _cmd_verify(directory: Path) -> int:
    """Validate all .tape.json cassettes in a directory for CI checks."""
    if not directory.exists():
        print(f"Error: directory not found: {directory}", file=sys.stderr)
        return 1

    files = sorted(directory.rglob("*.tape.json"))
    failures = 0
    for file_path in files:
        try:
            Cassette.load(file_path)
        except TapeError as exc:
            failures += 1
            print(f"FAIL {file_path}: {exc}", file=sys.stderr)

    if failures:
        print(f"Verification failed: {failures}/{len(files)} cassette(s) invalid.", file=sys.stderr)
        return 1
    print(f"OK: verified {len(files)} cassette(s) in {directory}.")
    return 0


def _cmd_diff(path_a: Path, path_b: Path, *, as_json: bool = False) -> int:
    """Compare calls, tokens, estimated cost, and tool sequences between two cassettes."""
    stats_a = _cassette_stats(Cassette.load(path_a))
    stats_b = _cassette_stats(Cassette.load(path_b))

    delta_calls = int(stats_b["calls"]) - int(stats_a["calls"])
    delta_tokens = int(stats_b["total_tokens"]) - int(stats_a["total_tokens"])
    delta_cost = round(
        float(stats_b["estimated_cost_usd"]) - float(stats_a["estimated_cost_usd"]), 8
    )

    if as_json:
        print(
            json.dumps(
                {
                    "before": stats_a,
                    "after": stats_b,
                    "delta": {
                        "calls": delta_calls,
                        "total_tokens": delta_tokens,
                        "estimated_cost_usd": delta_cost,
                        "tools_changed": stats_a["tools"] != stats_b["tools"],
                    },
                },
                indent=2,
            )
        )
        return 0

    print(f"Comparing {path_a} -> {path_b}")
    print(f"  Calls    : {stats_a['calls']} -> {stats_b['calls']} ({delta_calls:+d})")
    print(
        f"  Tokens   : {stats_a['total_tokens']} -> {stats_b['total_tokens']} ({delta_tokens:+d})"
    )
    print(
        f"  Est. Cost: ${stats_a['estimated_cost_usd']:.6f} -> "
        f"${stats_b['estimated_cost_usd']:.6f} (${delta_cost:+.6f})"
    )
    print(f"  Tools    : {stats_a['tools']} -> {stats_b['tools']}")
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
                joined_redacted = redact_text(joined_per_chunk, patterns)
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
        "verify", help="Validate all .tape.json cassettes in a directory (for CI)."
    )
    p_verify.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Directory to validate (default: current directory).",
    )

    p_diff = sub.add_parser("diff", help="Compare two .tape.json cassettes.")
    p_diff.add_argument("before", type=Path, help="Baseline .tape.json file.")
    p_diff.add_argument("after", type=Path, help="New .tape.json file.")
    p_diff.add_argument("--json", action="store_true", dest="as_json", help="Output JSON.")

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
            return _cmd_verify(args.directory)
        if args.command == "diff":
            return _cmd_diff(args.before, args.after, as_json=args.as_json)
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
