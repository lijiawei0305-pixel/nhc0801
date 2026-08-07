"""Phase-A inventory: double-end m250 teacher PASS roots for expanded Train.

Read-only over ``teacher_gpu_g*`` products and optional ``gpu_teacher_queue/state.json``.
Does not train, does not rewrite split contracts, does not open Final Test identities.

Gate for first large NPZ/train (user 2026-08-05):
  lock Train when ``n_full_pairs_eligible_for_train >= target_train_roots`` (default 150)
  with a fixed Val band of ~15–20 roots (~10%) chosen later — this module only counts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from nhc_deprot.data.paths import (
    SEALED_FINAL_TEST_COMMITMENT_SHA256,
    SEALED_FINAL_TEST_ROOT_COUNT,
    TRAIN_ROOTS,
    VALIDATION_ROOTS,
)
from nhc_deprot.pipeline.gpu_autofill import endpoint_done_ok

INVENTORY_SCHEMA: Final = "nhc0801-eligible-full-pairs-v1"
DEFAULT_TARGET_TRAIN_ROOTS: Final = 150
DEFAULT_TARGET_VAL_ROOTS: Final = 18  # middle of 15–20 (~10% of ~180)


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frame_count(endpoint_dir: Path) -> int:
    if not endpoint_dir.is_dir():
        return 0
    return sum(1 for _ in endpoint_dir.glob("frame_*.json"))


def scan_teacher_products(gen_root: Path) -> dict[str, dict[str, Any]]:
    """Map root_id -> per-endpoint done_ok / frames / product dirs (any teacher_gpu*)."""

    roots: dict[str, dict[str, Any]] = {}
    for tdir in sorted(gen_root.glob("teacher_gpu_g*")):
        if not tdir.is_dir():
            continue
        name = tdir.name
        if name.startswith("_") or "archive" in name.lower():
            continue
        try:
            children = list(tdir.iterdir())
        except OSError:
            continue
        for root_dir in children:
            if not root_dir.is_dir():
                continue
            rid = root_dir.name
            # skip non-inchikey-ish junk
            if rid.startswith(".") or rid.startswith("_"):
                continue
            slot = roots.setdefault(
                rid,
                {
                    "root_id": rid,
                    "cation": {
                        "done_ok": False,
                        "frame_count": 0,
                        "product_dirs": [],
                    },
                    "neutral": {
                        "done_ok": False,
                        "frame_count": 0,
                        "product_dirs": [],
                    },
                },
            )
            for ep in ("cation", "neutral"):
                ep_dir = root_dir / ep
                if not ep_dir.is_dir():
                    continue
                nframes = _frame_count(ep_dir)
                ok = False
                try:
                    ok = endpoint_done_ok(tdir, rid, ep)
                except OSError:
                    ok = False
                info = slot[ep]
                info["frame_count"] = max(int(info["frame_count"]), nframes)
                if ok:
                    info["done_ok"] = True
                rel = f"{name}/{rid}/{ep}"
                if rel not in info["product_dirs"]:
                    info["product_dirs"].append(rel)
    return roots


def merge_queue_state(
    roots: dict[str, dict[str, Any]],
    state_path: Path | None,
) -> dict[str, Any]:
    """Annotate queue running/done/failed keys (non-authoritative vs product done_ok)."""

    meta: dict[str, Any] = {
        "queue_state_path": str(state_path) if state_path else None,
        "queue_len": None,
        "running_count": None,
        "done_pass_count": None,
        "failed_count": None,
        "updated_at_utc": None,
    }
    if state_path is None or not state_path.is_file():
        return meta
    try:
        st = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta["error"] = "unreadable_queue_state"
        return meta

    meta["queue_len"] = len(st.get("queue") or [])
    meta["running_count"] = len(st.get("running") or {})
    meta["failed_count"] = len(st.get("failed") or [])
    meta["updated_at_utc"] = st.get("updated_at_utc")

    done_pass = 0
    for d in st.get("done") or []:
        if not isinstance(d, dict):
            continue
        key = str(d.get("key") or "")
        if ":" not in key:
            continue
        rid, ep = key.split(":", 1)
        if ep not in ("cation", "neutral"):
            continue
        slot = roots.setdefault(
            rid,
            {
                "root_id": rid,
                "cation": {"done_ok": False, "frame_count": 0, "product_dirs": []},
                "neutral": {"done_ok": False, "frame_count": 0, "product_dirs": []},
            },
        )
        q = slot.setdefault("queue_hints", {})
        status = str(d.get("status") or "")
        q[f"{ep}_done_status"] = status
        if status == "PASS":
            done_pass += 1
    meta["done_pass_count"] = done_pass

    for key, v in (st.get("running") or {}).items():
        if not isinstance(key, str) or ":" not in key:
            continue
        rid, ep = key.split(":", 1)
        if ep not in ("cation", "neutral"):
            continue
        slot = roots.setdefault(
            rid,
            {
                "root_id": rid,
                "cation": {"done_ok": False, "frame_count": 0, "product_dirs": []},
                "neutral": {"done_ok": False, "frame_count": 0, "product_dirs": []},
            },
        )
        q = slot.setdefault("queue_hints", {})
        q[f"{ep}_running"] = True
        if isinstance(v, dict):
            q[f"{ep}_running_gpu"] = v.get("gpu_index")
            q[f"{ep}_running_batch"] = v.get("batch_id")

    return meta


def classify_root(
    rid: str,
    slot: Mapping[str, Any],
    *,
    train_roots: Sequence[str],
    val_roots: Sequence[str],
) -> dict[str, Any]:
    """Attach split tags. Final Test identities are never listed (sealed)."""

    train_set = set(train_roots)
    val_set = set(val_roots)
    cat_ok = bool((slot.get("cation") or {}).get("done_ok"))
    neu_ok = bool((slot.get("neutral") or {}).get("done_ok"))
    full_pair = cat_ok and neu_ok
    cat_fr = int((slot.get("cation") or {}).get("frame_count") or 0)
    neu_fr = int((slot.get("neutral") or {}).get("frame_count") or 0)
    has_any = cat_fr > 0 or neu_fr > 0 or cat_ok or neu_ok
    qh = slot.get("queue_hints") or {}
    in_flight = bool(qh.get("cation_running") or qh.get("neutral_running"))

    in_train = rid in train_set
    in_val = rid in val_set
    # Sealed Final Test: identities not exposed — never mark a visible root as test.
    excluded_test = False
    incomplete = (not full_pair) and (has_any or in_flight)

    if in_val:
        role = "val"
    elif in_train:
        role = "train_legacy"
    elif full_pair:
        role = "pool_full_pair"
    elif incomplete:
        role = "incomplete"
    else:
        role = "unknown"

    # Expanded Train candidates: full pair, not Val, not legacy-only restriction
    eligible_for_expanded_train = full_pair and not in_val and not excluded_test

    return {
        "root_id": rid,
        "role": role,
        "in_train": in_train,
        "in_val": in_val,
        "excluded_test": excluded_test,
        "incomplete": incomplete,
        "full_pair_pass": full_pair,
        "eligible_for_expanded_train": eligible_for_expanded_train,
        "cation_done_ok": cat_ok,
        "neutral_done_ok": neu_ok,
        "cation_frame_count": cat_fr,
        "neutral_frame_count": neu_fr,
        "product_dirs": sorted(
            set(
                list((slot.get("cation") or {}).get("product_dirs") or [])
                + list((slot.get("neutral") or {}).get("product_dirs") or [])
            )
        ),
        "queue_hints": qh,
    }


def build_inventory(
    gen_root: Path,
    *,
    queue_state_path: Path | None = None,
    train_roots: Sequence[str] | None = None,
    val_roots: Sequence[str] | None = None,
    target_train_roots: int = DEFAULT_TARGET_TRAIN_ROOTS,
    target_val_roots: int = DEFAULT_TARGET_VAL_ROOTS,
) -> dict[str, Any]:
    """Full Phase-A inventory payload."""

    train = list(train_roots if train_roots is not None else TRAIN_ROOTS)
    val = list(val_roots if val_roots is not None else VALIDATION_ROOTS)
    roots = scan_teacher_products(gen_root)
    qpath = queue_state_path
    if qpath is None:
        cand = gen_root / "gpu_teacher_queue" / "state.json"
        qpath = cand if cand.is_file() else None
    queue_meta = merge_queue_state(roots, qpath)

    classified = [
        classify_root(rid, slot, train_roots=train, val_roots=val)
        for rid, slot in sorted(roots.items())
    ]

    full_pairs = [r for r in classified if r["full_pair_pass"]]
    eligible_train = [r for r in classified if r["eligible_for_expanded_train"]]
    incomplete = [r for r in classified if r["incomplete"]]
    in_val_full = [r for r in classified if r["in_val"] and r["full_pair_pass"]]
    in_train_full = [r for r in classified if r["in_train"] and r["full_pair_pass"]]

    n_eligible = len(eligible_train)
    # After reserving ~target_val from the big pool later, train capacity estimate:
    # user chose Val A 15–20 fixed, Train lock at ≥150. Gap to train lock uses
    # eligible-for-train count (Val-held roots excluded).
    gap_to_train_lock = max(0, int(target_train_roots) - n_eligible)
    # Total full pairs needed if we also want target_val distinct from train:
    # need eligible_train >= 150 AND separately val full pairs (may already be 0 in queue).
    val_full_n = len(in_val_full)
    gap_val_full = max(0, int(target_val_roots) - val_full_n)
    # Practical total full pairs to have on disk before locking split:
    # 150 train-eligible + 15–20 val (val may come from pool reassignment later)
    n_full_all = len(full_pairs)
    suggested_total_before_lock = int(target_train_roots) + int(target_val_roots)
    gap_suggested_total = max(0, suggested_total_before_lock - n_full_all)

    train_ready = n_eligible >= int(target_train_roots)

    return {
        "schema": INVENTORY_SCHEMA,
        "generated_at_utc": _utc(),
        "generation_root": str(gen_root),
        "policy": {
            "target_train_roots": int(target_train_roots),
            "target_val_roots": int(target_val_roots),
            "val_policy": "fixed_15_to_20_about_10_percent",
            "train_lock_rule": "n_eligible_for_expanded_train >= target_train_roots "
            "and both endpoints done_ok (m250 product)",
            "npz_train_only_after_train_lock": True,
            "no_small_groups_of_3": True,
            "final_test": {
                "sealed": True,
                "root_count": SEALED_FINAL_TEST_ROOT_COUNT,
                "commitment_sha256": SEALED_FINAL_TEST_COMMITMENT_SHA256,
                "identities_exposed": False,
            },
        },
        "legacy_split": {
            "train_roots": list(train),
            "val_roots": list(val),
        },
        "queue": queue_meta,
        "counts": {
            "roots_seen": len(classified),
            "n_full_pairs": n_full_all,
            "n_eligible_for_expanded_train": n_eligible,
            "n_incomplete": len(incomplete),
            "n_legacy_train_full": len(in_train_full),
            "n_val_full": val_full_n,
            "gap_to_train_lock_150": gap_to_train_lock,
            "gap_val_full_to_target": gap_val_full,
            "suggested_total_full_pairs_before_split_lock": suggested_total_before_lock,
            "gap_suggested_total_full_pairs": gap_suggested_total,
            "train_lock_ready": train_ready,
        },
        "eligible_full_pairs": [
            {
                "root_id": r["root_id"],
                "in_train": r["in_train"],
                "in_val": r["in_val"],
                "excluded_test": r["excluded_test"],
                "incomplete": r["incomplete"],
                "eligible_for_expanded_train": r["eligible_for_expanded_train"],
                "cation_frame_count": r["cation_frame_count"],
                "neutral_frame_count": r["neutral_frame_count"],
                "product_dirs": r["product_dirs"],
            }
            for r in eligible_train
        ],
        "val_roots_status": [
            r
            for r in classified
            if r["in_val"]
        ],
        "legacy_train_status": [r for r in classified if r["in_train"]],
        "incomplete_roots": [
            {
                "root_id": r["root_id"],
                "cation_done_ok": r["cation_done_ok"],
                "neutral_done_ok": r["neutral_done_ok"],
                "cation_frame_count": r["cation_frame_count"],
                "neutral_frame_count": r["neutral_frame_count"],
                "queue_hints": r["queue_hints"],
            }
            for r in incomplete
        ],
        "all_roots": classified,
    }


def write_inventory(payload: Mapping[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    out_path.write_text(text, encoding="utf-8")
    # also write a tiny status sidecar for daily glance
    side = out_path.with_name("eligible_full_pairs_status.json")
    counts = payload.get("counts") or {}
    side.write_text(
        json.dumps(
            {
                "schema": "nhc0801-eligible-full-pairs-status-v1",
                "generated_at_utc": payload.get("generated_at_utc"),
                "counts": counts,
                "inventory_path": str(out_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_path
