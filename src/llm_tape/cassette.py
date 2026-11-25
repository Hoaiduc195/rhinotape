"""Cassette file serialization (.tape.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_tape.models import Interaction, TapeNotFoundError

SCHEMA_VERSION = 1


@dataclass
class Cassette:
    """In-memory representation of a .tape.json cassette file."""

    path: Path
    interactions: list[Interaction] = field(default_factory=list)
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        cassette_path = Path(path)
        if not cassette_path.exists():
            raise TapeNotFoundError(
                f"Cassette file not found: {cassette_path}. "
                "Run with mode='once' or mode='record' first to record interactions."
            )
        raw = json.loads(cassette_path.read_text(encoding="utf-8"))
        interactions = [
            Interaction.from_dict(item) for item in raw.get("interactions", [])
        ]
        return cls(
            path=cassette_path,
            interactions=interactions,
            recorded_at=raw.get("recorded_at", ""),
            metadata=dict(raw.get("metadata", {})),
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
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
