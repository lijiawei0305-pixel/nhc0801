"""Small JSON/hash helpers for dataset contracts (no chemistry)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from nhc_deprot.data.errors import DatasetError


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"cannot read JSON: {path}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"invalid JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise DatasetError(f"JSON root is not an object: {path.name}")
    return cast(dict[str, Any], payload), raw


def write_json(
    path: Path, payload: Mapping[str, Any], *, overwrite: bool = False
) -> dict[str, object]:
    """Write canonical JSON object; refuse silent overwrite of divergent content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(dict(payload))
    if path.exists() and not overwrite:
        existing = path.read_bytes()
        if existing != raw:
            raise DatasetError(f"refusing overwrite of divergent file: {path}")
        return {
            "path": str(path),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "wrote": False,
        }
    path.write_bytes(raw)
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "wrote": True,
    }
