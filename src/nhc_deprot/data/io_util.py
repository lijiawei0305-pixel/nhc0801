"""Small JSON/hash helpers for dataset contracts (no chemistry)."""

from __future__ import annotations

import hashlib
import json
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
