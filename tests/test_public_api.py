"""Public API surface (__all__) and PEP 561 py.typed audit tests (Wave 111)."""

from __future__ import annotations

from pathlib import Path

import rhinotape


def test_public_all_exports_sorted_and_present() -> None:
    exported = rhinotape.__all__
    assert len(exported) == len(set(exported)), "Duplicate entries in rhinotape.__all__"
    assert exported == sorted(exported), "rhinotape.__all__ should remain sorted alphabetically"
    for symbol in exported:
        assert hasattr(rhinotape, symbol), f"Missing exported symbol: {symbol}"


def test_py_typed_marker_present() -> None:
    pkg_dir = Path(rhinotape.__file__).parent
    assert (pkg_dir / "py.typed").is_file()
