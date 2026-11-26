"""Model migration A/B comparison engine and text/Markdown report renderers."""

from __future__ import annotations

import difflib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rhinotape.cassette import Cassette, iter_cassette_files
from rhinotape.models import TapeNotFoundError

__all__ = [
    "CassettePairComparison",
    "MigrationReport",
    "compare_cassettes",
    "evaluate_compare_gates",
    "render_compare_markdown",
    "render_compare_text",
]

_CASSETTE_SUFFIXES: tuple[str, ...] = (
    ".tape.json.gz",
    ".tape.json",
    ".json.gz",
    ".json",
    ".gz",
)
_CHAR_SIMILARITY_MAX_LEN = 10_000


def _normalize_cassette_stem(rel_posix: str) -> str:
    normalized = rel_posix.replace("\\", "/")
    lower = normalized.lower()
    for suffix in _CASSETTE_SUFFIXES:
        if lower.endswith(suffix):
            return normalized[: -len(suffix)]
    p = Path(normalized)
    return (p.parent / p.stem).as_posix() if p.parent != Path(".") else p.stem


def _finite_sum(values: Iterable[float], ndigits: int) -> float:
    return round(sum(v for v in values if math.isfinite(v)), ndigits)


def _safe_json_float(value: float, ndigits: int) -> float | None:
    if not math.isfinite(value):
        return None
    return round(value, ndigits)


def _pct_change(baseline_val: float, candidate_val: float) -> float | None:
    if not math.isfinite(baseline_val) or not math.isfinite(candidate_val):
        return None
    if baseline_val == 0.0:
        if candidate_val == 0.0:
            return None
        return math.inf if candidate_val > 0.0 else -math.inf
    return round(((candidate_val - baseline_val) / baseline_val) * 100.0, 6)


def _serialize_pct(value: float | None) -> float | str | None:
    if value is None or math.isnan(value):
        return None
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return round(value, 4)


def _format_pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "+inf%" if value > 0 else "-inf%"
    return f"{value:+.1f}%"


def _combined_normalized_text(cassette: Cassette) -> str:
    raw = " ".join(i.llm.text for i in cassette.interactions if i.llm.text)
    return " ".join(raw.split())


def _compute_text_similarity(baseline: Cassette, candidate: Cassette) -> float:
    base_norm = _combined_normalized_text(baseline)
    cand_norm = _combined_normalized_text(candidate)
    if not base_norm and not cand_norm:
        return 1.0
    if not base_norm or not cand_norm:
        return 0.0
    if base_norm == cand_norm:
        return 1.0
    if max(len(base_norm), len(cand_norm)) > _CHAR_SIMILARITY_MAX_LEN:
        ratio = difflib.SequenceMatcher(
            None, base_norm.split(), cand_norm.split(), autojunk=False
        ).ratio()
    else:
        ratio = difflib.SequenceMatcher(None, base_norm, cand_norm, autojunk=False).ratio()
    return round(ratio, 6)


def _md_escape(text: str) -> str:
    single_line = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return single_line.replace("|", "\\|").replace("`", "\\`")


def _md_inline_code(text: str) -> str:
    single_line = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    cleaned = single_line.replace("`", "'").replace("|", "\\|")
    return f"`{cleaned}`"


@dataclass(frozen=True)
class CassettePairComparison:
    """A/B comparison metrics for a single paired baseline vs. candidate cassette."""

    name: str
    baseline_models: list[str]
    candidate_models: list[str]
    baseline_calls: int
    candidate_calls: int
    baseline_tokens: int
    candidate_tokens: int
    baseline_cost_usd: float
    candidate_cost_usd: float
    cost_delta_usd: float
    cost_change_pct: float | None
    baseline_latency_ms: float
    candidate_latency_ms: float
    latency_delta_ms: float
    latency_change_pct: float | None
    baseline_tools: list[str]
    candidate_tools: list[str]
    tools_match: bool
    baseline_errors: int
    candidate_errors: int
    text_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "baseline_models": list(self.baseline_models),
            "candidate_models": list(self.candidate_models),
            "baseline_calls": self.baseline_calls,
            "candidate_calls": self.candidate_calls,
            "baseline_tokens": self.baseline_tokens,
            "candidate_tokens": self.candidate_tokens,
            "baseline_cost_usd": _safe_json_float(self.baseline_cost_usd, 8),
            "candidate_cost_usd": _safe_json_float(self.candidate_cost_usd, 8),
            "cost_delta_usd": _safe_json_float(self.cost_delta_usd, 8),
            "cost_change_pct": _serialize_pct(self.cost_change_pct),
            "baseline_latency_ms": _safe_json_float(self.baseline_latency_ms, 2),
            "candidate_latency_ms": _safe_json_float(self.candidate_latency_ms, 2),
            "latency_delta_ms": _safe_json_float(self.latency_delta_ms, 2),
            "latency_change_pct": _serialize_pct(self.latency_change_pct),
            "baseline_tools": list(self.baseline_tools),
            "candidate_tools": list(self.candidate_tools),
            "tools_match": self.tools_match,
            "baseline_errors": self.baseline_errors,
            "candidate_errors": self.candidate_errors,
            "text_similarity": _safe_json_float(self.text_similarity, 6),
        }


@dataclass(frozen=True)
class MigrationReport:
    """Aggregate A/B migration comparison report across paired cassettes."""

    baseline_path: str
    candidate_path: str
    pairs: list[CassettePairComparison]
    only_in_baseline: list[str]
    only_in_candidate: list[str]
    baseline_models: list[str]
    candidate_models: list[str]
    baseline_calls: int
    candidate_calls: int
    baseline_tokens: int
    candidate_tokens: int
    token_delta: int
    baseline_cost_usd: float
    candidate_cost_usd: float
    cost_delta_usd: float
    cost_change_pct: float | None
    baseline_latency_ms: float
    candidate_latency_ms: float
    latency_delta_ms: float
    latency_change_pct: float | None
    baseline_errors: int
    candidate_errors: int
    tool_drift_count: int
    avg_text_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_path": self.baseline_path,
            "candidate_path": self.candidate_path,
            "baseline_models": list(self.baseline_models),
            "candidate_models": list(self.candidate_models),
            "matched_pairs": len(self.pairs),
            "only_in_baseline": list(self.only_in_baseline),
            "only_in_candidate": list(self.only_in_candidate),
            "baseline_calls": self.baseline_calls,
            "candidate_calls": self.candidate_calls,
            "baseline_tokens": self.baseline_tokens,
            "candidate_tokens": self.candidate_tokens,
            "token_delta": self.token_delta,
            "baseline_cost_usd": _safe_json_float(self.baseline_cost_usd, 8),
            "candidate_cost_usd": _safe_json_float(self.candidate_cost_usd, 8),
            "cost_delta_usd": _safe_json_float(self.cost_delta_usd, 8),
            "cost_change_pct": _serialize_pct(self.cost_change_pct),
            "baseline_latency_ms": _safe_json_float(self.baseline_latency_ms, 2),
            "candidate_latency_ms": _safe_json_float(self.candidate_latency_ms, 2),
            "latency_delta_ms": _safe_json_float(self.latency_delta_ms, 2),
            "latency_change_pct": _serialize_pct(self.latency_change_pct),
            "baseline_errors": self.baseline_errors,
            "candidate_errors": self.candidate_errors,
            "tool_drift_count": self.tool_drift_count,
            "avg_text_similarity": _safe_json_float(self.avg_text_similarity, 6),
            "pairs": [p.to_dict() for p in self.pairs],
        }


def _compare_pair(name: str, base: Cassette, cand: Cassette) -> CassettePairComparison:
    b_models = sorted({i.llm.model for i in base.interactions if i.llm.model})
    c_models = sorted({i.llm.model for i in cand.interactions if i.llm.model})
    b_calls = len(base.interactions)
    c_calls = len(cand.interactions)
    b_tokens = sum(i.llm.usage.total_tokens for i in base.interactions)
    c_tokens = sum(i.llm.usage.total_tokens for i in cand.interactions)
    b_cost = _finite_sum((i.llm.estimated_cost_usd for i in base.interactions), 8)
    c_cost = _finite_sum((i.llm.estimated_cost_usd for i in cand.interactions), 8)
    cost_delta = round(c_cost - b_cost, 8)
    b_lat = _finite_sum((i.latency_ms for i in base.interactions), 2)
    c_lat = _finite_sum((i.latency_ms for i in cand.interactions), 2)
    lat_delta = round(c_lat - b_lat, 2)
    b_tools = [tc.name for i in base.interactions for tc in i.llm.tool_calls]
    c_tools = [tc.name for i in cand.interactions for tc in i.llm.tool_calls]
    b_errs = sum(1 for i in base.interactions if i.response.status_code >= 400)
    c_errs = sum(1 for i in cand.interactions if i.response.status_code >= 400)

    return CassettePairComparison(
        name=name,
        baseline_models=b_models,
        candidate_models=c_models,
        baseline_calls=b_calls,
        candidate_calls=c_calls,
        baseline_tokens=b_tokens,
        candidate_tokens=c_tokens,
        baseline_cost_usd=b_cost,
        candidate_cost_usd=c_cost,
        cost_delta_usd=cost_delta,
        cost_change_pct=_pct_change(b_cost, c_cost),
        baseline_latency_ms=b_lat,
        candidate_latency_ms=c_lat,
        latency_delta_ms=lat_delta,
        latency_change_pct=_pct_change(b_lat, c_lat),
        baseline_tools=b_tools,
        candidate_tools=c_tools,
        tools_match=(b_tools == c_tools),
        baseline_errors=b_errs,
        candidate_errors=c_errs,
        text_similarity=_compute_text_similarity(base, cand),
    )


def _index_cassette_dir(directory: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for file_path in iter_cassette_files(directory):
        rel = file_path.relative_to(directory).as_posix()
        key = _normalize_cassette_stem(rel)
        if key in mapping:
            raise ValueError(
                f"Ambiguous cassette stem {key!r} in {directory}: "
                f"both {mapping[key].name!r} and {file_path.name!r} exist."
            )
        mapping[key] = file_path
    return mapping


def compare_cassettes(baseline: str | Path, candidate: str | Path) -> MigrationReport:
    """Compare a baseline vs. candidate cassette file or directory of cassettes."""
    base_path = Path(baseline)
    cand_path = Path(candidate)
    if not base_path.exists():
        raise TapeNotFoundError(f"Baseline path not found: {base_path}")
    if not cand_path.exists():
        raise TapeNotFoundError(f"Candidate path not found: {cand_path}")
    both_files = base_path.is_file() and cand_path.is_file()
    both_dirs = base_path.is_dir() and cand_path.is_dir()
    if not (both_files or both_dirs):
        raise ValueError(
            "Both baseline and candidate must be cassette files or both must be directories."
        )

    pairs: list[CassettePairComparison] = []
    only_base: list[str] = []
    only_cand: list[str] = []

    if both_files:
        base_cassette = Cassette.load(base_path)
        cand_cassette = Cassette.load(cand_path)
        pair_name = _normalize_cassette_stem(base_path.name)
        pairs.append(_compare_pair(pair_name, base_cassette, cand_cassette))
    else:
        base_map = _index_cassette_dir(base_path)
        cand_map = _index_cassette_dir(cand_path)
        only_base = sorted(set(base_map) - set(cand_map))
        only_cand = sorted(set(cand_map) - set(base_map))
        for key in sorted(set(base_map) & set(cand_map)):
            pairs.append(
                _compare_pair(
                    key,
                    Cassette.load(base_map[key]),
                    Cassette.load(cand_map[key]),
                )
            )

    all_base_models = sorted({m for p in pairs for m in p.baseline_models})
    all_cand_models = sorted({m for p in pairs for m in p.candidate_models})
    tot_b_calls = sum(p.baseline_calls for p in pairs)
    tot_c_calls = sum(p.candidate_calls for p in pairs)
    tot_b_tokens = sum(p.baseline_tokens for p in pairs)
    tot_c_tokens = sum(p.candidate_tokens for p in pairs)
    tot_b_cost = _finite_sum((p.baseline_cost_usd for p in pairs), 8)
    tot_c_cost = _finite_sum((p.candidate_cost_usd for p in pairs), 8)
    tot_cost_delta = round(tot_c_cost - tot_b_cost, 8)
    tot_b_lat = _finite_sum((p.baseline_latency_ms for p in pairs), 2)
    tot_c_lat = _finite_sum((p.candidate_latency_ms for p in pairs), 2)
    tot_lat_delta = round(tot_c_lat - tot_b_lat, 2)
    tot_b_errs = sum(p.baseline_errors for p in pairs)
    tot_c_errs = sum(p.candidate_errors for p in pairs)
    drift_count = sum(1 for p in pairs if not p.tools_match)
    finite_sims = [p.text_similarity for p in pairs if math.isfinite(p.text_similarity)]
    avg_sim = round(sum(finite_sims) / len(finite_sims), 6) if finite_sims else 0.0

    return MigrationReport(
        baseline_path=str(base_path),
        candidate_path=str(cand_path),
        pairs=pairs,
        only_in_baseline=only_base,
        only_in_candidate=only_cand,
        baseline_models=all_base_models,
        candidate_models=all_cand_models,
        baseline_calls=tot_b_calls,
        candidate_calls=tot_c_calls,
        baseline_tokens=tot_b_tokens,
        candidate_tokens=tot_c_tokens,
        token_delta=tot_c_tokens - tot_b_tokens,
        baseline_cost_usd=tot_b_cost,
        candidate_cost_usd=tot_c_cost,
        cost_delta_usd=tot_cost_delta,
        cost_change_pct=_pct_change(tot_b_cost, tot_c_cost),
        baseline_latency_ms=tot_b_lat,
        candidate_latency_ms=tot_c_lat,
        latency_delta_ms=tot_lat_delta,
        latency_change_pct=_pct_change(tot_b_lat, tot_c_lat),
        baseline_errors=tot_b_errs,
        candidate_errors=tot_c_errs,
        tool_drift_count=drift_count,
        avg_text_similarity=avg_sim,
    )


def _validate_gate_thresholds(
    min_similarity: Any,
    max_cost_increase_pct: Any,
    fail_on_tool_drift: Any,
) -> None:
    if min_similarity is not None:
        if (
            isinstance(min_similarity, bool)
            or not isinstance(min_similarity, (int, float))
            or not math.isfinite(min_similarity)
            or min_similarity < 0.0
            or min_similarity > 1.0
        ):
            raise ValueError("min_similarity must be a finite float between 0.0 and 1.0.")
    if max_cost_increase_pct is not None:
        if (
            isinstance(max_cost_increase_pct, bool)
            or not isinstance(max_cost_increase_pct, (int, float))
            or not math.isfinite(max_cost_increase_pct)
            or max_cost_increase_pct < 0.0
        ):
            raise ValueError("max_cost_increase_pct must be a finite float >= 0.0.")
    if not isinstance(fail_on_tool_drift, bool):
        raise ValueError("fail_on_tool_drift must be a bool.")


def evaluate_compare_gates(
    report: MigrationReport,
    *,
    min_similarity: float | None = None,
    max_cost_increase_pct: float | None = None,
    fail_on_tool_drift: bool = False,
) -> list[str]:
    """Evaluate CI quality/budget gates on a MigrationReport and return violation messages."""
    _validate_gate_thresholds(min_similarity, max_cost_increase_pct, fail_on_tool_drift)

    violations: list[str] = []
    has_gate = min_similarity is not None or max_cost_increase_pct is not None or fail_on_tool_drift
    if has_gate and not report.pairs:
        violations.append("No matched cassette pairs found between baseline and candidate.")
        return violations

    if min_similarity is not None:
        for pair in report.pairs:
            if not math.isfinite(pair.text_similarity) or pair.text_similarity < min_similarity:
                violations.append(
                    f"Cassette '{pair.name}' text_similarity {pair.text_similarity:.3f} "
                    f"< --min-similarity {min_similarity:.3f}"
                )
    if max_cost_increase_pct is not None and report.cost_change_pct is not None:
        if math.isnan(report.cost_change_pct):
            violations.append("Total cost change percentage is non-finite (NaN).")
        elif math.isinf(report.cost_change_pct) and report.cost_change_pct > 0:
            violations.append(
                f"Total cost increased from $0.000000 to ${report.candidate_cost_usd:.6f} "
                f"(+inf% > --max-cost-increase-pct {max_cost_increase_pct:.2f}%)"
            )
        elif report.cost_change_pct > max_cost_increase_pct:
            violations.append(
                f"Total cost increased by {report.cost_change_pct:+.2f}% "
                f"(${report.baseline_cost_usd:.6f} -> ${report.candidate_cost_usd:.6f}), "
                f"exceeding --max-cost-increase-pct {max_cost_increase_pct:.2f}%"
            )
    if fail_on_tool_drift and report.tool_drift_count > 0:
        drifted = ", ".join(p.name for p in report.pairs if not p.tools_match)
        violations.append(
            f"Tool drift detected in {report.tool_drift_count} cassette(s): {drifted}"
        )
    return violations


def render_compare_text(report: MigrationReport) -> str:
    """Render a human-readable terminal executive summary and per-cassette breakdown."""
    b_mods = ", ".join(report.baseline_models) or "<unknown>"
    c_mods = ", ".join(report.candidate_models) or "<unknown>"
    sim_label = f"{report.avg_text_similarity * 100.0:.1f}%" if report.pairs else "n/a"
    lines: list[str] = [
        f"Model Migration Comparison: {report.baseline_path} -> {report.candidate_path}",
        f"  Models        : {b_mods} -> {c_mods}",
        f"  Matched Pairs : {len(report.pairs)}",
        (
            f"  Calls         : {report.baseline_calls} -> {report.candidate_calls} "
            f"({report.candidate_calls - report.baseline_calls:+d})"
        ),
        (
            f"  Total Tokens  : {report.baseline_tokens} -> {report.candidate_tokens} "
            f"({report.token_delta:+d})"
        ),
        (
            f"  Est. Cost     : ${report.baseline_cost_usd:.6f} -> "
            f"${report.candidate_cost_usd:.6f} "
            f"(${report.cost_delta_usd:+.6f}, {_format_pct(report.cost_change_pct)})"
        ),
        (
            f"  Total Latency : {report.baseline_latency_ms:.1f}ms -> "
            f"{report.candidate_latency_ms:.1f}ms "
            f"({report.latency_delta_ms:+.1f}ms, {_format_pct(report.latency_change_pct)})"
        ),
        f"  Avg Similarity: {sim_label}",
        f"  Tool Drift    : {report.tool_drift_count}/{len(report.pairs)} cassette(s)",
        f"  HTTP Errors   : {report.baseline_errors} -> {report.candidate_errors}",
    ]

    if report.pairs:
        lines.append("")
        lines.append("Per-Cassette Breakdown:")
        for p in report.pairs:
            pm_base = ",".join(p.baseline_models) or "-"
            pm_cand = ",".join(p.candidate_models) or "-"
            tool_status = "match" if p.tools_match else "DRIFT"
            lines.append(
                f"  - {p.name} ({pm_base} -> {pm_cand}): "
                f"cost=${p.baseline_cost_usd:.6f}->${p.candidate_cost_usd:.6f} "
                f"({_format_pct(p.cost_change_pct)}) | "
                f"tokens={p.baseline_tokens}->{p.candidate_tokens} | "
                f"latency={p.baseline_latency_ms:.1f}->{p.candidate_latency_ms:.1f}ms | "
                f"tools={tool_status} | sim={p.text_similarity * 100.0:.1f}%"
            )

    if report.only_in_baseline:
        lines.append("")
        lines.append(f"Only in baseline ({len(report.only_in_baseline)}):")
        for item in report.only_in_baseline:
            lines.append(f"  - {item}")
    if report.only_in_candidate:
        lines.append("")
        lines.append(f"Only in candidate ({len(report.only_in_candidate)}):")
        for item in report.only_in_candidate:
            lines.append(f"  - {item}")

    return "\n".join(lines) + "\n"


def render_compare_markdown(report: MigrationReport) -> str:
    """Render a GitHub-Flavored Markdown comparison report suitable for PR comments."""
    b_mods = _md_inline_code(", ".join(report.baseline_models) or "-")
    c_mods = _md_inline_code(", ".join(report.candidate_models) or "-")
    call_delta = report.candidate_calls - report.baseline_calls
    sim_label = f"{report.avg_text_similarity * 100.0:.1f}%" if report.pairs else "n/a"

    lines: list[str] = [
        "### rhinotape Model Migration Report",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| :--- | ---: | ---: | ---: |",
        f"| Models | {b_mods} | {c_mods} | — |",
        f"| Matched Cassettes | {len(report.pairs)} | {len(report.pairs)} | 0 |",
        f"| Calls | {report.baseline_calls} | {report.candidate_calls} | {call_delta:+d} |",
        (
            f"| Total Tokens | {report.baseline_tokens} | {report.candidate_tokens} | "
            f"{report.token_delta:+d} |"
        ),
        (
            f"| Est. Cost (USD) | ${report.baseline_cost_usd:.6f} | "
            f"${report.candidate_cost_usd:.6f} | "
            f"${report.cost_delta_usd:+.6f} ({_format_pct(report.cost_change_pct)}) |"
        ),
        (
            f"| Latency | {report.baseline_latency_ms:.1f}ms | "
            f"{report.candidate_latency_ms:.1f}ms | "
            f"{report.latency_delta_ms:+.1f}ms ({_format_pct(report.latency_change_pct)}) |"
        ),
        (
            f"| HTTP Errors | {report.baseline_errors} | {report.candidate_errors} | "
            f"{report.candidate_errors - report.baseline_errors:+d} |"
        ),
        (f"| Tool Drift | — | — | {report.tool_drift_count}/{len(report.pairs)} |"),
        f"| Avg Text Similarity | — | — | {sim_label} |",
    ]

    if report.pairs:
        lines.extend(
            [
                "",
                "#### Per-Cassette Breakdown",
                "",
                "| Cassette | Models | Cost | Cost Δ% | Tokens | Latency | Tools | Similarity |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | :---: | ---: |",
            ]
        )
        for p in report.pairs:
            name_cell = _md_escape(p.name)
            models_cell = _md_escape(
                f"{','.join(p.baseline_models) or '-'} -> {','.join(p.candidate_models) or '-'}"
            )
            cost_cell = f"${p.baseline_cost_usd:.6f} -> ${p.candidate_cost_usd:.6f}"
            pct_cell = _format_pct(p.cost_change_pct)
            tok_cell = f"{p.baseline_tokens} -> {p.candidate_tokens}"
            lat_cell = f"{p.baseline_latency_ms:.1f} -> {p.candidate_latency_ms:.1f}ms"
            tools_cell = "match" if p.tools_match else "DRIFT"
            sim_cell = f"{p.text_similarity * 100.0:.1f}%"
            lines.append(
                f"| {name_cell} | {models_cell} | {cost_cell} | {pct_cell} | "
                f"{tok_cell} | {lat_cell} | {tools_cell} | {sim_cell} |"
            )

    if report.only_in_baseline or report.only_in_candidate:
        lines.append("")
        lines.append("#### Unpaired Cassettes")
        if report.only_in_baseline:
            items_b = ", ".join(_md_inline_code(k) for k in report.only_in_baseline)
            lines.append(f"- **Only in baseline ({len(report.only_in_baseline)}):** {items_b}")
        if report.only_in_candidate:
            items_c = ", ".join(_md_inline_code(k) for k in report.only_in_candidate)
            lines.append(f"- **Only in candidate ({len(report.only_in_candidate)}):** {items_c}")

    return "\n".join(lines) + "\n"
