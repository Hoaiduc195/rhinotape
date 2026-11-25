"""Cassette file serialization (.tape.json / .tape.json.gz) with atomic writes and validation."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import tempfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_tape.models import CorruptCassetteError, Interaction, TapeNotFoundError

SCHEMA_VERSION = 2
CASSETTE_EXTENSIONS: tuple[str, ...] = (".tape.json", ".tape.json.gz")
MAX_CASSETTE_BYTES = 100 * 1024 * 1024  # 100 MiB decompression safety cap
_GZIP_MAGIC = b"\x1f\x8b"


def cassette_json_schema() -> dict[str, Any]:
    """Return a Draft 2020-12 JSON Schema describing the .tape.json cassette v2 format."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/Hoaiduc195/llm-tape/schemas/cassette-v2.schema.json",
        "title": "llm-tape Cassette",
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
    """Return True if path has a supported llm-tape cassette extension."""
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
    ) -> Cassette:
        """Combine multiple cassettes in order into a single Cassette."""
        merged_interactions: list[Interaction] = []
        merged_meta: dict[str, Any] = {}
        seen_fps: set[str] = set()
        for c in cassettes:
            merged_meta.update(c.metadata)
            for inter in c.interactions:
                if deduplicate:
                    if inter.request.fingerprint in seen_fps:
                        continue
                    seen_fps.add(inter.request.fingerprint)
                merged_interactions.append(inter)
        return cls(
            path=Path(output_path),
            interactions=merged_interactions,
            metadata=merged_meta,
        )

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

    def stats(self) -> dict[str, Any]:
        """Return summary statistics (calls, token breakdown, cost, latency, tools)."""
        return {
            "path": str(self.path),
            "recorded_at": self.recorded_at,
            "calls": len(self.interactions),
            "input_tokens": sum(i.llm.usage.input_tokens for i in self.interactions),
            "cached_input_tokens": sum(i.llm.usage.cached_input_tokens for i in self.interactions),
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

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        cassette_path = Path(path)
        if not cassette_path.exists():
            raise TapeNotFoundError(
                f"Cassette file not found: {cassette_path}. "
                "Run with mode='once' or mode='record' first to record interactions."
            )
        try:
            raw_bytes = cassette_path.read_bytes()
            if cassette_path.suffix.lower() == ".gz" or raw_bytes[:2] == _GZIP_MAGIC:
                text = _decompress_gzip_bounded(raw_bytes).decode("utf-8")
            else:
                text = raw_bytes.decode("utf-8")
            raw = json.loads(text)
        except CorruptCassetteError:
            raise
        except (
            OSError,
            gzip.BadGzipFile,
            zlib.error,
            EOFError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise CorruptCassetteError(f"Failed to read cassette {cassette_path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise CorruptCassetteError(
                f"Cassette {cassette_path} must contain a JSON object at top level."
            )

        version = raw.get("schema_version", 1)
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
            raw_interactions = raw.get("interactions", [])
            if not isinstance(raw_interactions, list):
                raise TypeError("'interactions' must be a list")
            interactions = [Interaction.from_dict(item) for item in raw_interactions]
            raw_meta = raw.get("metadata")
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
            recorded_at=str(raw.get("recorded_at", "")),
            metadata=metadata,
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
        serialized = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        if self.path.suffix.lower() == ".gz":
            data_bytes = gzip.compress(serialized, mtime=0)
        else:
            data_bytes = serialized

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
