"""Per-batch Epoch-0 on Validation roots ONLY (mindmap step 3).

Standard names + disk layout (user 2026-08-03):

  - g001 Epoch-0  → ``epoch0_val_batches/g001/``
  - g002 Epoch-0  → ``epoch0_val_batches/g002/``
  - g00N Epoch-0  → ``epoch0_val_batches/g00N/``

Same pattern for every batch. Do **not** use a special top-level ``epoch0/``
for g001, and do **not** call these "expansion Val e0".

Never run e0 on Train roots. Policy keyword Val-only ≠ task name.
Uses gold XYZ → AIMNet2 GAU_LOOSE (GPU) → Parent P01 (CPU by default).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from nhc_deprot.contracts.parent_protocol import (
    CATION_CHARGE,
    CATION_MULTIPLICITY,
    NEUTRAL_CHARGE,
    NEUTRAL_MULTIPLICITY,
)
from nhc_deprot.data.io_util import write_json
from nhc_deprot.data.paths import TRAIN_ROOTS
from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout
from nhc_deprot.pipeline.epoch0_runner import Epoch0Config, run_epoch0_campaign
from nhc_deprot.pipeline.live_epoch0 import LiveAimnet2GauLooseEngine, LiveParentP01Engine, load_xyz
from nhc_deprot.pipeline.scientific_validation import FrozenEndpointGeometry

OFFICIAL_WEIGHT = Path("/home/plab/.cache/aimnet/aimnet2_wb97m_d3_0.pt")
DEFAULT_GOLD = Path("/home/plab/test/WJW/data/runs/mol_gold/xyz")
ALT_GOLD = (
    Path("/home/plab/test/WJW/data/candidates/structures_full/xyz"),
    Path("/home/plab/test/WJW/data/candidates/xyz"),
)


class E0ValOnlyError(RuntimeError):
    """Val-only epoch-0 policy violation or setup failure."""


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_epoch0_batch_id(batch_id: str) -> str:
    """Map aliases to the disk/batch name used under epoch0_val_batches/."""
    bid = str(batch_id).strip()
    if bid in {"g001_pilot", "pilot", "g001-pilot"}:
        return "g001"
    # generation id style → still g001 for pilot epoch0 when explicit
    if bid == "nhc0801-g001":
        return "g001"
    if not bid or "/" in bid or ".." in bid:
        raise E0ValOnlyError(f"invalid batch_id for Epoch-0: {batch_id!r}")
    return bid


def refuse_train_roots(val_roots: Sequence[str]) -> list[str]:
    """Hard-filter: drop and error if any Train root sneaks in."""
    train = frozenset(TRAIN_ROOTS)
    roots = list(val_roots)
    bad = [r for r in roots if r in train]
    if bad:
        raise E0ValOnlyError(
            f"REFUSED train roots in e0-val job (never compute train e0): {bad}"
        )
    if not roots:
        raise E0ValOnlyError("val_roots empty — e0 must not run on train")
    return roots


def resolve_xyz(root_id: str, endpoint: str, gold_dirs: Sequence[Path]) -> Path:
    name = f"{root_id}_{endpoint}.xyz"
    for d in gold_dirs:
        p = d / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"missing xyz for {root_id}/{endpoint} in {list(gold_dirs)}")


def load_geo(root_id: str, endpoint: str, gold_dirs: Sequence[Path]) -> FrozenEndpointGeometry:
    xyz = resolve_xyz(root_id, endpoint, gold_dirs)
    elements, coords = load_xyz(xyz)
    charge, mult = (
        (CATION_CHARGE, CATION_MULTIPLICITY)
        if endpoint == "cation"
        else (NEUTRAL_CHARGE, NEUTRAL_MULTIPLICITY)
    )
    xyz_coords = tuple(
        (float(row[0]), float(row[1]), float(row[2])) for row in coords
    )
    return FrozenEndpointGeometry(
        root_id=root_id,
        endpoint=endpoint,
        elements=tuple(elements),
        coordinates=xyz_coords,
        charge=charge,
        multiplicity=mult,
        geometry_sha256="",
    )


def run_e0_for_val_roots(
    *,
    nhc0801_root: Path,
    generation_id: str,
    val_roots: Sequence[str],
    batch_id: str,
    max_steps: int = 100,
    weight: Path = OFFICIAL_WEIGHT,
    gold_dirs: Sequence[Path] | None = None,
    parent_backend: str = "cpu",
    cuda_device: int | None = None,
    use_official_epoch0_dir: bool | None = None,  # deprecated, ignored
) -> dict:
    """Run live Epoch-0 for one batch (g001 / g002 / …) on val_roots only.

    All batches write under ``epoch0_val_batches/<batch_id>/`` (same pattern).
    ``use_official_epoch0_dir`` is deprecated and ignored (legacy top-level epoch0/).
    """
    if use_official_epoch0_dir:
        # kept for CLI compat; uniform path is mandatory now
        pass
    roots = refuse_train_roots(val_roots)
    bid = normalize_epoch0_batch_id(batch_id)
    golds = list(gold_dirs or (DEFAULT_GOLD, *ALT_GOLD))
    layout = resolve_layout(generation_id=generation_id, nhc0801_root=nhc0801_root)
    ensure_generation_tree(layout, exist_ok=True)

    batch_root = layout.epoch0_batch_root(bid)
    work_layout = replace(
        layout,
        epoch0_dir=batch_root / "epoch0",
        logs_dir=batch_root / "logs",
    )
    work_layout.epoch0_dir.mkdir(parents=True, exist_ok=True)
    work_layout.logs_dir.mkdir(parents=True, exist_ok=True)
    side_receipt_dir = layout.generation_root / "epoch0_val_batches"

    geos = []
    for root_id in roots:
        for ep in ("cation", "neutral"):
            geos.append(load_geo(root_id, ep, golds))

    aim = LiveAimnet2GauLooseEngine(weight_path=weight)
    parent_kw: dict = {"max_steps": max_steps}
    if parent_backend == "gpu":
        if cuda_device is None:
            raise ValueError("gpu parent requires cuda_device")
        parent_kw["backend"] = "gpu"
        parent_kw["cuda_device"] = int(cuda_device)
        parent_kw["host_threads"] = 2
    else:
        parent_kw["backend"] = "cpu"
        parent_kw["host_threads"] = 8
    parent = LiveParentP01Engine(**parent_kw)

    print(
        f"[e0] {bid} Epoch-0 roots={roots} endpoints={len(geos)} "
        f"epoch0_dir={work_layout.epoch0_dir} parent={parent_backend}"
        + (f" cuda_device={cuda_device}" if parent_backend == "gpu" else ""),
        flush=True,
    )
    out = run_epoch0_campaign(
        layout=work_layout,
        config=Epoch0Config(validation_roots=tuple(roots)),
        dry_run=False,
        epoch0_execution=True,
        aimnet2=aim,
        parent=parent,
        pure_parent=parent,
        geometries=geos,
    )
    side_receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "nhc0801-epoch0-val-only-receipt-v1",
        "mindmap_step": 3,
        "batch_id": bid,
        "standard_name": f"{bid} Epoch-0",
        "val_roots_only": True,
        "train_roots_forbidden": True,
        "val_roots": roots,
        "refused_train_roots": list(TRAIN_ROOTS),
        "status": out.get("status"),
        "failed_root_count": out.get("failed_root_count"),
        "live_chemistry": True,
        "dry_run": False,
        "disk_layout": "epoch0_val_batches/<batch_id>/",
        "epoch0_dir": str(work_layout.epoch0_dir),
        "batch_root": str(batch_root),
        "created_at_utc": _utc(),
        "campaign": out,
    }
    write_json(side_receipt_dir / f"{bid}_epoch0_val_receipt.json", receipt, overwrite=True)
    write_json(layout.logs_dir / f"epoch0_{bid}.json", receipt, overwrite=True)
    print(f"[e0] finished {bid} Epoch-0 status={out.get('status')}", flush=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nhc0801-root", type=Path, required=True)
    ap.add_argument("--generation-id", default="nhc0801-g001")
    ap.add_argument(
        "--batch-id",
        required=True,
        help="Batch id for this Epoch-0 (g001, g002, …). Aliases: g001_pilot→g001",
    )
    ap.add_argument(
        "--val-roots",
        required=True,
        help="Comma-separated Validation InChIKeys (Train roots refused)",
    )
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--parent-backend", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--cuda-device", type=int, default=None)
    ap.add_argument(
        "--use-official-epoch0-dir",
        action="store_true",
        help="(deprecated, ignored) all batches use epoch0_val_batches/<id>/",
    )
    args = ap.parse_args(argv)
    roots = [r.strip() for r in args.val_roots.split(",") if r.strip()]
    try:
        receipt = run_e0_for_val_roots(
            nhc0801_root=args.nhc0801_root,
            generation_id=args.generation_id,
            val_roots=roots,
            batch_id=args.batch_id,
            max_steps=int(args.max_steps),
            parent_backend=args.parent_backend,
            cuda_device=args.cuda_device,
        )
    except E0ValOnlyError as exc:
        print(f"E0_VAL_EXIT REFUSED {exc}", flush=True)
        return 2
    except Exception as exc:
        print(f"E0_VAL_EXIT FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    bid = receipt.get("batch_id") or normalize_epoch0_batch_id(args.batch_id)
    print(json.dumps({"status": receipt.get("status"), "batch_id": bid, "val_roots": roots}))
    print(f"E0_VAL_EXIT {receipt.get('status')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
