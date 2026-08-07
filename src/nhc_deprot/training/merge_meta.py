"""Train merge_meta.json — COMPUTE_DISPATCH_V001 §1.4.4 gate for live train labels.

No merge_meta or ``train_val_disjoint != true`` → refuse live training labels.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

MERGE_META_SCHEMA: Final = "nhc0801-train-merge-meta-v1"
MERGE_META_FILENAME: Final = "merge_meta.json"


class MergeMetaError(RuntimeError):
    """merge_meta missing, invalid, or train/val not disjoint."""


def merge_meta_path(train_batch_dir: Path) -> Path:
    return Path(train_batch_dir) / MERGE_META_FILENAME


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_train_val_disjoint(
    train_roots: Sequence[str], validation_roots: Sequence[str]
) -> bool:
    """Runtime assertion: Train ∩ Val must be empty. Returns True only if disjoint."""

    overlap = sorted(set(train_roots) & set(validation_roots))
    if overlap:
        raise MergeMetaError(
            f"train_val_disjoint false; overlap roots: {overlap[:10]}"
            + ("…" if len(overlap) > 10 else "")
        )
    return True


def build_merge_meta(
    *,
    merge_group_id: str,
    archive_path: str | Path,
    merged_from_groups: Sequence[str],
    train_roots: Sequence[str],
    val_roots: Sequence[str],
    naming_mode: str = "succeed_g001_after_archive",
    teacher_standard: str = "parent_max_steps=250, full_trajectory=true",
    allow_legacy_teacher: bool = False,
    authority: str = (
        "user 2026-08-07 互动确认 / COMPUTE_DISPATCH_V001 §11.1"
    ),
    tvt_split_document: Path | None = None,
    notes: Sequence[str] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build §1.4.4 merge_meta payload. ``train_val_disjoint`` is runtime-asserted."""

    train = sorted({str(r) for r in train_roots})
    val = sorted({str(r) for r in val_roots})
    disjoint = assert_train_val_disjoint(train, val)
    groups = sorted({str(g) for g in merged_from_groups})
    note_list = list(notes or [])
    if tvt_split_document is not None and Path(tvt_split_document).is_file():
        note_list.append(
            f"tvt_split_document_sha256="
            f"{sha256_file(Path(tvt_split_document))}"
            f" path={tvt_split_document}"
        )
    # Always declare quick-val evaluation set change at merge time.
    note_list.append(
        "quick-val evaluation set rebuilt with this merge: "
        "archived train_g001 checkpoints are NOT comparable to new runs"
    )
    return {
        "schema": MERGE_META_SCHEMA,
        "merge_group_id": str(merge_group_id),
        "naming_mode": naming_mode,
        "archive_path": str(archive_path),
        "merged_from_groups": groups,
        "train_roots": train,
        "val_roots": val,
        "train_val_disjoint": bool(disjoint),
        "teacher_standard": teacher_standard,
        "allow_legacy_teacher": bool(allow_legacy_teacher),
        "authority": authority,
        "created_at_utc": created_at_utc or _utc(),
        "notes": note_list,
    }


def write_merge_meta(train_batch_dir: Path, meta: dict[str, Any]) -> Path:
    """Write merge_meta.json; fail closed if train_val_disjoint is not true."""

    if meta.get("schema") != MERGE_META_SCHEMA:
        raise MergeMetaError(f"bad merge_meta schema: {meta.get('schema')!r}")
    if meta.get("train_val_disjoint") is not True:
        raise MergeMetaError("refuse write: train_val_disjoint is not true")
    path = merge_meta_path(train_batch_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_merge_meta(train_batch_dir: Path) -> dict[str, Any]:
    path = merge_meta_path(train_batch_dir)
    if not path.is_file():
        raise MergeMetaError(f"missing merge_meta.json at {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeMetaError(f"unreadable merge_meta.json: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise MergeMetaError(f"merge_meta not an object: {path}")
    return raw


def assert_merge_meta_ready(train_batch_dir: Path) -> dict[str, Any]:
    """Fail closed for live train: meta exists and train_val_disjoint is true."""

    meta = load_merge_meta(train_batch_dir)
    if meta.get("schema") != MERGE_META_SCHEMA:
        raise MergeMetaError(
            f"merge_meta schema mismatch: {meta.get('schema')!r} "
            f"!= {MERGE_META_SCHEMA!r}"
        )
    if meta.get("train_val_disjoint") is not True:
        raise MergeMetaError(
            "merge_meta.train_val_disjoint is not true — refuse live train"
        )
    train = meta.get("train_roots") or []
    val = meta.get("val_roots") or []
    if not isinstance(train, list) or not isinstance(val, list):
        raise MergeMetaError("merge_meta train_roots/val_roots must be lists")
    # Re-assert disjointness from the written lists (not trust the flag alone).
    assert_train_val_disjoint([str(x) for x in train], [str(x) for x in val])
    return meta
