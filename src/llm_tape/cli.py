"""Command-line interface for inspecting llm-tape cassettes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_tape.cassette import Cassette
from llm_tape.models import TapeError


def _cmd_inspect(path: Path) -> int:
    cassette = Cassette.load(path)
    total_tokens = sum(i.llm.usage.total_tokens for i in cassette.interactions)
    total_cost = sum(i.llm.estimated_cost_usd for i in cassette.interactions)

    print(f"Cassette : {cassette.path}")
    print(f"Recorded : {cassette.recorded_at or 'unknown'}")
    print(f"Calls    : {len(cassette.interactions)}")
    print(f"Tokens   : {total_tokens}")
    print(f"Est. Cost: ${total_cost:.6f}")

    if cassette.interactions:
        print("\nInteractions:")
        for idx, inter in enumerate(cassette.interactions, start=1):
            tools = ", ".join(tc.name for tc in inter.llm.tool_calls) or "-"
            model = inter.llm.model or "-"
            print(
                f"  [{idx}] {inter.request.method} {inter.request.url} -> "
                f"{inter.response.status_code} ({inter.latency_ms:.1f}ms) | "
                f"provider={inter.llm.provider} model={model} "
                f"tokens={inter.llm.usage.total_tokens} "
                f"cost=${inter.llm.estimated_cost_usd:.6f} tools=[{tools}]"
            )
    return 0


def _cmd_summary(directory: Path) -> int:
    if not directory.exists():
        print(f"Error: directory not found: {directory}", file=sys.stderr)
        return 1

    files = sorted(directory.rglob("*.tape.json"))
    if not files:
        print(f"No .tape.json cassettes found in {directory}")
        return 0

    grand_calls = 0
    grand_tokens = 0
    grand_cost = 0.0

    print(f"{'Cassette':<48} {'Calls':>6} {'Tokens':>9} {'Est. Cost':>12}")
    print("-" * 79)
    for file_path in files:
        cassette = Cassette.load(file_path)
        calls = len(cassette.interactions)
        tokens = sum(i.llm.usage.total_tokens for i in cassette.interactions)
        cost = sum(i.llm.estimated_cost_usd for i in cassette.interactions)
        grand_calls += calls
        grand_tokens += tokens
        grand_cost += cost
        rel = str(file_path.relative_to(directory))
        if len(rel) > 48:
            rel = "..." + rel[-45:]
        print(f"{rel:<48} {calls:>6} {tokens:>9} ${cost:>11.6f}")

    print("-" * 79)
    print(
        f"{f'TOTAL ({len(files)} files)':<48} {grand_calls:>6} "
        f"{grand_tokens:>9} ${grand_cost:>11.6f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-tape",
        description="Inspect and summarize llm-tape .tape.json cassettes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect a single .tape.json cassette.")
    p_inspect.add_argument("path", type=Path, help="Path to the .tape.json file.")

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _cmd_inspect(args.path)
        if args.command == "summary":
            return _cmd_summary(args.directory)
    except TapeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
