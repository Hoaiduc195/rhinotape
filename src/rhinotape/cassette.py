"""Cassette file serialization (.tape.json / .tape.json.gz) with atomic writes and validation."""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import math
import os
import re
import tempfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rhinotape.models import CorruptCassetteError, Interaction, TapeError, TapeNotFoundError

SCHEMA_VERSION = 2
CASSETTE_EXTENSIONS: tuple[str, ...] = (".tape.json", ".tape.json.gz")
MAX_CASSETTE_BYTES = 100 * 1024 * 1024  # 100 MiB decompression safety cap
_GZIP_MAGIC = b"\x1f\x8b"

MERGE_DUPLICATE_STRATEGIES: tuple[str, ...] = (
    "keep-all",
    "keep-first",
    "keep-last",
    "keep-newest",
    "error",
)
MERGE_SORT_KEYS: tuple[str, ...] = ("recorded_at", "latency", "cost")
MERGE_CONFLICT_STRATEGIES: tuple[str, ...] = (
    "keep-ours",
    "keep-theirs",
    "keep-newest",
    "error",
)


_ISO_FRAC_RE = re.compile(r"(\d{2}:\d{2}:\d{2})\.(\d+)")
_ISO_COMPACT_TZ_RE = re.compile(r"([Tt ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?[+-]\d{2})(\d{2})$")


def _parse_recorded_at(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    else:
        text = _ISO_COMPACT_TZ_RE.sub(r"\1:\2", text)
    text = _ISO_FRAC_RE.sub(lambda m: f"{m.group(1)}.{(m.group(2) + '000000')[:6]}", text)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _prefer_later_timestamp(existing_ts: datetime | None, candidate_ts: datetime | None) -> bool:
    if candidate_ts is None:
        return existing_ts is None
    if existing_ts is None:
        return True
    return candidate_ts >= existing_ts


def _finite_sort_key(value: Any) -> tuple[int, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return (0, float(value))
    return (1, 0.0)


def _meta_values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_meta_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_meta_values_equal(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def _index_interactions(interactions: list[Interaction]) -> dict[tuple[str, int], Interaction]:
    counts: dict[str, int] = {}
    indexed: dict[tuple[str, int], Interaction] = {}
    for inter in interactions:
        fp = inter.request.fingerprint
        occ = counts.get(fp, 0)
        counts[fp] = occ + 1
        indexed[(fp, occ)] = inter
    return indexed


def _interactions_equal(a: Interaction, b: Interaction) -> bool:
    return a.to_dict() == b.to_dict()


def cassette_json_schema() -> dict[str, Any]:
    """Return a Draft 2020-12 JSON Schema describing the .tape.json cassette v2 format."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/Hoaiduc195/rhinotape/schemas/cassette-v2.schema.json",
        "title": "rhinotape Cassette",
        "description": "Recorded HTTP & SSE LLM interactions for deterministic offline replay.",
        "type": "object",
        "required": ["schema_version", "interactions"],
        "properties": {
            "schema_version": {"type": "integer", "minimum": 1, "maximum": SCHEMA_VERSION},
            "recorded_at": {"type": "string"},
            "metadata": {"type": ["object", "null"]},
            "summary": {
                "type": "object",
                "properties": {
                    "calls": {"type": "integer", "minimum": 0},
                    "total_tokens": {"type": "integer", "minimum": 0},
                    "estimated_cost_usd": {"type": "number", "minimum": 0},
                },
            },
            "interactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["request", "response"],
                    "properties": {
                        "request": {
                            "type": "object",
                            "required": ["method", "url", "fingerprint"],
                            "properties": {
                                "method": {"type": "string"},
                                "url": {"type": "string"},
                                "headers": {"type": "object"},
                                "body": {},
                                "fingerprint": {"type": "string"},
                                "body_encoding": {
                                    "type": "string",
                                    "enum": ["json", "text", "base64", "empty"],
                                },
                            },
                        },
                        "response": {
                            "type": "object",
                            "required": ["status_code"],
                            "properties": {
                                "status_code": {"type": "integer", "minimum": 100, "maximum": 599},
                                "headers": {"type": "object"},
                                "body": {},
                                "body_encoding": {
                                    "type": "string",
                                    "enum": ["json", "text", "base64", "empty", "stream"],
                                },
                                "stream_chunks": {
                                    "type": ["array", "null"],
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        "latency_ms": {"type": "number", "minimum": 0},
                        "llm": {
                            "type": "object",
                            "properties": {
                                "provider": {"type": "string"},
                                "model": {"type": ["string", "null"]},
                                "usage": {
                                    "type": "object",
                                    "properties": {
                                        "input_tokens": {"type": "integer", "minimum": 0},
                                        "output_tokens": {"type": "integer", "minimum": 0},
                                        "total_tokens": {"type": "integer", "minimum": 0},
                                        "cached_input_tokens": {"type": "integer", "minimum": 0},
                                        "cache_creation_tokens": {"type": "integer", "minimum": 0},
                                        "reasoning_tokens": {"type": "integer", "minimum": 0},
                                    },
                                },
                                "estimated_cost_usd": {"type": "number", "minimum": 0},
                                "tool_calls": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["name"],
                                        "properties": {
                                            "name": {"type": "string"},
                                            "arguments": {},
                                            "call_id": {"type": ["string", "null"]},
                                        },
                                    },
                                },
                                "text": {"type": "string"},
                                "thinking": {"type": "string"},
                                "is_stream": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
        },
    }


def is_cassette_file(path: Path) -> bool:
    """Return True if path has a supported rhinotape cassette extension."""
    lower = path.name.lower()
    return any(lower.endswith(ext) for ext in CASSETTE_EXTENSIONS)


def iter_cassette_files(directory: Path) -> list[Path]:
    """Return sorted list of .tape.json and .tape.json.gz files under directory."""
    return sorted(p for p in directory.rglob("*") if is_cassette_file(p))


def _decompress_gzip_bounded(raw_bytes: bytes, *, max_bytes: int | None = None) -> bytes:
    limit = MAX_CASSETTE_BYTES if max_bytes is None else max_bytes
    with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes), mode="rb") as gz:
        data = gz.read(limit + 1)
    if len(data) > limit:
        raise CorruptCassetteError(
            f"Decompressed cassette exceeds maximum allowed size ({limit} bytes)."
        )
    return data


@dataclass
class Cassette:
    """In-memory representation of a .tape.json or .tape.json.gz cassette file."""

    path: Path
    interactions: list[Interaction] = field(default_factory=list)
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Return deterministic SHA-256 hex digest of this cassette's recorded interactions."""
        canonical = json.dumps(
            [i.to_dict() for i in self.interactions],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def merge(
        cls,
        *cassettes: Cassette,
        output_path: str | Path,
        deduplicate: bool = False,
        on_duplicate: str = "keep-all",
        sort_by: str | None = None,
    ) -> Cassette:
        """Combine multiple cassettes in order into a single Cassette."""
        if on_duplicate not in MERGE_DUPLICATE_STRATEGIES:
            raise ValueError(
                f"Unknown on_duplicate strategy {on_duplicate!r}; "
                f"expected one of {MERGE_DUPLICATE_STRATEGIES}."
            )
        if sort_by is not None and sort_by not in MERGE_SORT_KEYS:
            raise ValueError(
                f"Unknown sort_by value {sort_by!r}; expected None or one of {MERGE_SORT_KEYS}."
            )

        strategy = "keep-first" if (deduplicate and on_duplicate == "keep-all") else on_duplicate
        merged_meta: dict[str, Any] = {}
        items: list[tuple[datetime | None, Interaction]] = []

        if strategy == "keep-all":
            for c in cassettes:
                for mk, mv in c.metadata.items():
                    merged_meta[mk] = copy.deepcopy(mv)
                c_ts = _parse_recorded_at(c.recorded_at)
                for inter in c.interactions:
                    items.append((c_ts, inter))
        else:
            by_fp: dict[str, tuple[datetime | None, Interaction]] = {}
            for c in cassettes:
                for mk, mv in c.metadata.items():
                    merged_meta[mk] = copy.deepcopy(mv)
                c_ts = _parse_recorded_at(c.recorded_at)
                for inter in c.interactions:
                    fp = inter.request.fingerprint
                    if fp not in by_fp:
                        by_fp[fp] = (c_ts, inter)
                    elif strategy == "error":
                        raise TapeError(f"Duplicate interaction fingerprint during merge: {fp}")
                    elif strategy == "keep-last":
                        by_fp[fp] = (c_ts, inter)
                    elif strategy == "keep-newest":
                        prev_ts, _ = by_fp[fp]
                        if _prefer_later_timestamp(prev_ts, c_ts):
                            by_fp[fp] = (c_ts, inter)
            items = list(by_fp.values())

        if sort_by == "recorded_at":
            items.sort(
                key=lambda pair: (0, pair[0].timestamp()) if pair[0] is not None else (1, 0.0)
            )
        elif sort_by == "latency":
            items.sort(key=lambda pair: _finite_sort_key(pair[1].latency_ms))
        elif sort_by == "cost":
            items.sort(key=lambda pair: _finite_sort_key(pair[1].llm.estimated_cost_usd))

        return cls(
            path=Path(output_path),
            interactions=[inter for _, inter in items],
            metadata=merged_meta,
        )

    @classmethod
    def merge_three_way(
        cls,
        base: Cassette,
        ours: Cassette,
        theirs: Cassette,
        *,
        output_path: str | Path,
        on_conflict: str = "keep-ours",
    ) -> tuple[Cassette, list[str]]:
        """Perform a 3-way merge of base, ours, and theirs cassettes."""
        if on_conflict not in MERGE_CONFLICT_STRATEGIES:
            raise ValueError(
                f"Unknown on_conflict strategy {on_conflict!r}; "
                f"expected one of {MERGE_CONFLICT_STRATEGIES}."
            )

        ours_ts = _parse_recorded_at(ours.recorded_at)
        theirs_ts = _parse_recorded_at(theirs.recorded_at)
        prefer_theirs_on_newest = _prefer_later_timestamp(ours_ts, theirs_ts)

        def _pick_side(ours_val: Any, theirs_val: Any) -> Any:
            if on_conflict == "keep-theirs":
                return theirs_val
            if on_conflict == "keep-newest":
                return theirs_val if prefer_theirs_on_newest else ours_val
            return ours_val

        base_idx = _index_interactions(base.interactions)
        ours_idx = _index_interactions(ours.interactions)
        theirs_idx = _index_interactions(theirs.interactions)

        ordered_keys: list[tuple[str, int]] = []
        seen_keys: set[tuple[str, int]] = set()
        for source_map in (base_idx, ours_idx, theirs_idx):
            for k in source_map:
                if k not in seen_keys:
                    seen_keys.add(k)
                    ordered_keys.append(k)

        merged_interactions: list[Interaction] = []
        conflicts: list[str] = []

        for k in ordered_keys:
            fp, occ = k
            label = fp if occ == 0 else f"{fp}#{occ + 1}"
            b_item = base_idx.get(k)
            o_item = ours_idx.get(k)
            t_item = theirs_idx.get(k)

            if b_item is None:
                if o_item is not None and t_item is None:
                    merged_interactions.append(o_item)
                elif o_item is None and t_item is not None:
                    merged_interactions.append(t_item)
                elif o_item is not None and t_item is not None:
                    if _interactions_equal(o_item, t_item):
                        merged_interactions.append(o_item)
                    else:
                        conflicts.append(
                            f"interaction {label}: added with different content in ours and theirs"
                        )
                        chosen = _pick_side(o_item, t_item)
                        if chosen is not None:
                            merged_interactions.append(chosen)
                continue

            if o_item is not None and t_item is not None:
                o_changed = not _interactions_equal(b_item, o_item)
                t_changed = not _interactions_equal(b_item, t_item)
                if not o_changed and not t_changed:
                    merged_interactions.append(o_item)
                elif o_changed and not t_changed:
                    merged_interactions.append(o_item)
                elif not o_changed and t_changed:
                    merged_interactions.append(t_item)
                elif _interactions_equal(o_item, t_item):
                    merged_interactions.append(o_item)
                else:
                    conflicts.append(
                        f"interaction {label}: modified differently in ours and theirs"
                    )
                    merged_interactions.append(_pick_side(o_item, t_item))
            elif o_item is None and t_item is not None:
                if not _interactions_equal(b_item, t_item):
                    conflicts.append(f"interaction {label}: deleted in ours but modified in theirs")
                    chosen = _pick_side(None, t_item)
                    if chosen is not None:
                        merged_interactions.append(chosen)
            elif o_item is not None and t_item is None:
                if not _interactions_equal(b_item, o_item):
                    conflicts.append(f"interaction {label}: modified in ours but deleted in theirs")
                    chosen = _pick_side(o_item, None)
                    if chosen is not None:
                        merged_interactions.append(chosen)

        sentinel = object()
        meta_keys: list[str] = []
        seen_meta: set[str] = set()
        for meta_dict in (base.metadata, ours.metadata, theirs.metadata):
            for mk in meta_dict:
                if mk not in seen_meta:
                    seen_meta.add(mk)
                    meta_keys.append(mk)

        merged_meta: dict[str, Any] = {}
        for mk in meta_keys:
            vb = base.metadata.get(mk, sentinel)
            vo = ours.metadata.get(mk, sentinel)
            vt = theirs.metadata.get(mk, sentinel)
            if vb is sentinel:
                if vo is not sentinel and vt is sentinel:
                    merged_meta[mk] = copy.deepcopy(vo)
                elif vo is sentinel and vt is not sentinel:
                    merged_meta[mk] = copy.deepcopy(vt)
                elif _meta_values_equal(vo, vt):
                    merged_meta[mk] = copy.deepcopy(vo)
                else:
                    conflicts.append(f"metadata[{mk!r}]: added differently in ours and theirs")
                    merged_meta[mk] = copy.deepcopy(_pick_side(vo, vt))
            elif vo is not sentinel and vt is not sentinel:
                o_mod = not _meta_values_equal(vo, vb)
                t_mod = not _meta_values_equal(vt, vb)
                if not o_mod and not t_mod:
                    merged_meta[mk] = copy.deepcopy(vo)
                elif o_mod and not t_mod:
                    merged_meta[mk] = copy.deepcopy(vo)
                elif not o_mod and t_mod:
                    merged_meta[mk] = copy.deepcopy(vt)
                elif _meta_values_equal(vo, vt):
                    merged_meta[mk] = copy.deepcopy(vo)
                else:
                    conflicts.append(f"metadata[{mk!r}]: modified differently in ours and theirs")
                    merged_meta[mk] = copy.deepcopy(_pick_side(vo, vt))
            elif vo is sentinel and vt is not sentinel:
                if not _meta_values_equal(vt, vb):
                    conflicts.append(f"metadata[{mk!r}]: deleted in ours but modified in theirs")
                    picked = _pick_side(sentinel, vt)
                    if picked is not sentinel:
                        merged_meta[mk] = copy.deepcopy(picked)
            elif vo is not sentinel and vt is sentinel:
                if not _meta_values_equal(vo, vb):
                    conflicts.append(f"metadata[{mk!r}]: modified in ours but deleted in theirs")
                    picked = _pick_side(vo, sentinel)
                    if picked is not sentinel:
                        merged_meta[mk] = copy.deepcopy(picked)

        if conflicts and on_conflict == "error":
            raise TapeError(
                f"3-way cassette merge encountered {len(conflicts)} conflict(s): "
                + "; ".join(conflicts)
            )

        base_ts = _parse_recorded_at(base.recorded_at)
        if ours_ts is not None and theirs_ts is not None:
            merged_recorded_at = theirs.recorded_at if theirs_ts >= ours_ts else ours.recorded_at
        elif theirs_ts is not None:
            merged_recorded_at = theirs.recorded_at
        elif ours_ts is not None:
            merged_recorded_at = ours.recorded_at
        elif base_ts is not None:
            merged_recorded_at = base.recorded_at
        else:
            raw_candidates = (ours.recorded_at, theirs.recorded_at, base.recorded_at)
            merged_recorded_at = next(
                (s for s in raw_candidates if isinstance(s, str) and s),
                datetime.now(timezone.utc).isoformat(),
            )

        merged = cls(
            path=Path(output_path),
            interactions=merged_interactions,
            recorded_at=merged_recorded_at,
            metadata=merged_meta,
        )
        return merged, conflicts

    def filter(self, predicate: Callable[[Interaction], bool]) -> Cassette:
        """Return a new Cassette keeping only interactions where predicate(inter) is True."""
        return Cassette(
            path=self.path,
            interactions=[i for i in self.interactions if predicate(i)],
            recorded_at=self.recorded_at,
            metadata=dict(self.metadata),
        )

    def slice(self, start: int | None = None, end: int | None = None) -> Cassette:
        """Return a new Cassette containing interactions[start:end]."""
        return Cassette(
            path=self.path,
            interactions=list(self.interactions[start:end]),
            recorded_at=self.recorded_at,
            metadata=dict(self.metadata),
        )

    def deduplicate(self) -> Cassette:
        """Return a new Cassette with duplicate request fingerprints removed (preserving order)."""
        seen: set[str] = set()
        unique: list[Interaction] = []
        for inter in self.interactions:
            if inter.request.fingerprint not in seen:
                seen.add(inter.request.fingerprint)
                unique.append(inter)
        return Cassette(
            path=self.path,
            interactions=unique,
            recorded_at=self.recorded_at,
            metadata=dict(self.metadata),
        )

    def clone(self, new_path: str | Path) -> Cassette:
        """Return a shallow copy of this Cassette pointing to a new path."""
        return Cassette(
            path=Path(new_path),
            interactions=list(self.interactions),
            recorded_at=self.recorded_at,
            metadata=dict(self.metadata),
        )

    def add_interaction(self, interaction: Interaction) -> Cassette:
        """Append an Interaction to this cassette in place and return self."""
        self.interactions.append(interaction)
        return self

    def stats(self) -> dict[str, Any]:
        """Return summary statistics (calls, token breakdown, cost, latency, tools)."""
        return {
            "path": str(self.path),
            "recorded_at": self.recorded_at,
            "calls": len(self.interactions),
            "input_tokens": sum(i.llm.usage.input_tokens for i in self.interactions),
            "cached_input_tokens": sum(i.llm.usage.cached_input_tokens for i in self.interactions),
            "cache_creation_tokens": sum(
                i.llm.usage.cache_creation_tokens for i in self.interactions
            ),
            "output_tokens": sum(i.llm.usage.output_tokens for i in self.interactions),
            "reasoning_tokens": sum(i.llm.usage.reasoning_tokens for i in self.interactions),
            "total_tokens": sum(i.llm.usage.total_tokens for i in self.interactions),
            "estimated_cost_usd": round(
                sum(i.llm.estimated_cost_usd for i in self.interactions), 8
            ),
            "latency_ms": round(sum(i.latency_ms for i in self.interactions), 2),
            "tools": [tc.name for i in self.interactions for tc in i.llm.tool_calls],
        }

    def sort_interactions(
        self,
        *,
        key: Callable[[Interaction], Any] | None = None,
        reverse: bool = False,
    ) -> Cassette:
        """Return a new Cassette with interactions sorted (default: by url, method, fingerprint)."""
        sort_key = key or (lambda i: (i.request.url, i.request.method, i.request.fingerprint))
        return Cassette(
            path=self.path,
            interactions=sorted(self.interactions, key=sort_key, reverse=reverse),
            recorded_at=self.recorded_at,
            metadata=dict(self.metadata),
        )

    def validate(self) -> list[str]:
        """Validate cassette structure in memory and return a list of issue strings (empty = OK)."""
        issues: list[str] = []
        valid_req_enc = {"json", "text", "base64", "empty"}
        valid_resp_enc = {"json", "text", "base64", "empty", "stream"}
        for idx, inter in enumerate(self.interactions):
            if not inter.request.method:
                issues.append(f"interactions[{idx}].request.method is empty")
            if not inter.request.url:
                issues.append(f"interactions[{idx}].request.url is empty")
            if not inter.request.fingerprint:
                issues.append(f"interactions[{idx}].request.fingerprint is empty")
            if inter.request.body_encoding not in valid_req_enc:
                issues.append(
                    f"interactions[{idx}].request.body_encoding invalid: "
                    f"{inter.request.body_encoding!r}"
                )
            if not (100 <= inter.response.status_code <= 599):
                issues.append(
                    f"interactions[{idx}].response.status_code out of range: "
                    f"{inter.response.status_code}"
                )
            if inter.response.body_encoding not in valid_resp_enc:
                issues.append(
                    f"interactions[{idx}].response.body_encoding invalid: "
                    f"{inter.response.body_encoding!r}"
                )
            if inter.response.body_encoding == "stream" and inter.response.stream_chunks is None:
                issues.append(
                    f"interactions[{idx}].response has body_encoding='stream' "
                    "but stream_chunks is None"
                )
            if not math.isfinite(inter.latency_ms) or inter.latency_ms < 0:
                issues.append(f"interactions[{idx}].latency_ms is invalid: {inter.latency_ms}")
        return issues

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize this cassette to a v2 JSON string."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": self.recorded_at,
            "metadata": self.metadata,
            "summary": {
                "calls": len(self.interactions),
                "total_tokens": sum(i.llm.usage.total_tokens for i in self.interactions),
                "estimated_cost_usd": round(
                    sum(i.llm.estimated_cost_usd for i in self.interactions), 8
                ),
            },
            "interactions": [i.to_dict() for i in self.interactions],
        }
        return json.dumps(payload, indent=indent, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, raw: str, *, path: str | Path = Path("memory.tape.json")) -> Cassette:
        """Deserialize a Cassette from a JSON string."""
        cassette_path = Path(path)

        def _reject_nonfinite(constant: str) -> Any:
            raise ValueError(f"Non-finite JSON constant {constant!r} is not allowed in cassettes")

        try:
            parsed = json.loads(raw, parse_constant=_reject_nonfinite)
        except (json.JSONDecodeError, ValueError) as exc:
            raise CorruptCassetteError(f"Failed to read cassette {cassette_path}: {exc}") from exc

        if not isinstance(parsed, dict):
            raise CorruptCassetteError(
                f"Cassette {cassette_path} must contain a JSON object at top level."
            )

        version = parsed.get("schema_version", 1)
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or not (1 <= version <= SCHEMA_VERSION)
        ):
            raise CorruptCassetteError(
                f"Unsupported schema_version {version!r} in {cassette_path} "
                f"(supported: 1..{SCHEMA_VERSION})."
            )

        try:
            raw_interactions = parsed.get("interactions", [])
            if not isinstance(raw_interactions, list):
                raise TypeError("'interactions' must be a list")
            interactions = [Interaction.from_dict(item) for item in raw_interactions]
            raw_meta = parsed.get("metadata")
            if raw_meta is None:
                metadata: dict[str, Any] = {}
            elif isinstance(raw_meta, dict):
                metadata = dict(raw_meta)
            else:
                raise TypeError("'metadata' must be a JSON object or null")
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptCassetteError(
                f"Corrupt cassette structure in {cassette_path}: {exc}"
            ) from exc

        return cls(
            path=cassette_path,
            interactions=interactions,
            recorded_at=str(parsed.get("recorded_at", "")),
            metadata=metadata,
        )

    def to_bytes(self, *, compress: bool = False, indent: int | None = 2) -> bytes:
        """Serialize this cassette to UTF-8 JSON bytes (optionally gzip-compressed)."""
        serialized = self.to_json(indent=indent).encode("utf-8")
        return gzip.compress(serialized, mtime=0) if compress else serialized

    @classmethod
    def from_bytes(
        cls,
        raw_bytes: bytes,
        *,
        path: str | Path = Path("memory.tape.json"),
        max_bytes: int | None = None,
    ) -> Cassette:
        """Deserialize a Cassette from plain UTF-8 or gzip-compressed bytes."""
        cassette_path = Path(path)
        limit = MAX_CASSETTE_BYTES if max_bytes is None else max_bytes
        try:
            if cassette_path.suffix.lower() == ".gz" or raw_bytes[:2] == _GZIP_MAGIC:
                text = _decompress_gzip_bounded(raw_bytes, max_bytes=limit).decode("utf-8")
            else:
                if len(raw_bytes) > limit:
                    raise CorruptCassetteError(
                        f"Cassette {cassette_path} exceeds safety size limit ({limit} bytes)."
                    )
                text = raw_bytes.decode("utf-8")
        except CorruptCassetteError:
            raise
        except (OSError, gzip.BadGzipFile, zlib.error, EOFError, UnicodeDecodeError) as exc:
            raise CorruptCassetteError(f"Failed to read cassette {cassette_path}: {exc}") from exc
        return cls.from_json(text, path=cassette_path)

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        cassette_path = Path(path)
        if not cassette_path.exists():
            raise TapeNotFoundError(
                f"Cassette file not found: {cassette_path}. "
                "Run with mode='once' or mode='record' first to record interactions."
            )
        limit = MAX_CASSETTE_BYTES
        try:
            with cassette_path.open("rb") as f:
                magic = f.read(2)
                f.seek(0)
                if cassette_path.suffix.lower() == ".gz" or magic == _GZIP_MAGIC:
                    with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                        decompressed = gz.read(limit + 1)
                    if len(decompressed) > limit:
                        raise CorruptCassetteError(
                            f"Decompressed cassette exceeds maximum allowed size ({limit} bytes)."
                        )
                    text = decompressed.decode("utf-8")
                else:
                    st_size = os.fstat(f.fileno()).st_size
                    if st_size > limit:
                        raise CorruptCassetteError(
                            f"Cassette {cassette_path} exceeds safety size limit ({limit} bytes)."
                        )
                    raw_bytes = f.read(limit + 1)
                    if len(raw_bytes) > limit:
                        raise CorruptCassetteError(
                            f"Cassette {cassette_path} exceeds safety size limit ({limit} bytes)."
                        )
                    text = raw_bytes.decode("utf-8")
        except CorruptCassetteError:
            raise
        except FileNotFoundError as exc:
            raise TapeNotFoundError(
                f"Cassette file not found: {cassette_path}. "
                "Run with mode='once' or mode='record' first to record interactions."
            ) from exc
        except (OSError, gzip.BadGzipFile, zlib.error, EOFError, UnicodeDecodeError) as exc:
            raise CorruptCassetteError(f"Failed to read cassette {cassette_path}: {exc}") from exc
        return cls.from_json(text, path=cassette_path)

    def save(self, *, compress: bool | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        use_gzip = self.path.suffix.lower() == ".gz" if compress is None else compress
        data_bytes = self.to_bytes(compress=use_gzip)

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = tmp.name
                tmp.write(data_bytes)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, self.path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
