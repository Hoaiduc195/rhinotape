"""Cassette file serialization (.tape.json) with atomic writes and schema validation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_tape.models import CorruptCassetteError, Interaction, TapeNotFoundError

SCHEMA_VERSION = 2


@dataclass
class Cassette:
    """In-memory representation of a .tape.json cassette file."""

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
            raw = json.loads(cassette_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
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
        serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = tmp.name
                tmp.write(serialized)
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
