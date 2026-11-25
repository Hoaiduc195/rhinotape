"""Cassette file serialization (.tape.json / .tape.json.gz) with atomic writes and validation."""

from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_tape.models import CorruptCassetteError, Interaction, TapeNotFoundError

SCHEMA_VERSION = 2
CASSETTE_EXTENSIONS: tuple[str, ...] = (".tape.json", ".tape.json.gz")
MAX_CASSETTE_BYTES = 100 * 1024 * 1024  # 100 MiB decompression safety cap
_GZIP_MAGIC = b"\x1f\x8b"


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
