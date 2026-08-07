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
from nhc_deprot.pipeline.parent_handoff import load_gau_loose_profile
from nhc_deprot.pipeline.scientific_validation import (
    FrozenEndpointGeometry,
    assemble_root_label,
    run_endpoint_route,
)

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


def _parent_engine(
    *,
    max_steps: int,
    parent_backend: str,
    cuda_device: int | None,
) -> LiveParentP01Engine:
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
    return LiveParentP01Engine(**parent_kw)


def endpoint_result_path(root_dir: Path, endpoint: str) -> Path:
    """Preferred path for one endpoint result; falls back to legacy ``*_shard.json``."""

    ep = str(endpoint).strip().lower()
    preferred = root_dir / f"{ep}.json"
    if preferred.is_file():
        return preferred
    legacy = root_dir / f"{ep}_shard.json"
    if legacy.is_file():
        return legacy
    return preferred


def run_e0_single_endpoint(
    *,
    nhc0801_root: Path,
    generation_id: str,
    batch_id: str,
    root_id: str,
    endpoint: str,
    max_steps: int = 250,
    weight: Path = OFFICIAL_WEIGHT,
    gold_dirs: Sequence[Path] | None = None,
    parent_backend: str = "cpu",
    cuda_device: int | None = None,
) -> dict:
    """Run Epoch-0 pure+e0 for **one** endpoint on one GPU (cation/neutral 分开算).

    Writes ``epoch0/<root>/<endpoint>.json``. When both cation and neutral exist,
    merges into ``root.json`` (and legacy ``epoch0_root_receipt.json`` for readers).
    """
    roots = refuse_train_roots([root_id])
    rid = roots[0]
    ep = str(endpoint).strip().lower()
    if ep not in {"cation", "neutral"}:
        raise E0ValOnlyError(f"endpoint must be cation|neutral, got {endpoint!r}")
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

    geo = load_geo(rid, ep, golds)
    gau = load_gau_loose_profile()
    aim = LiveAimnet2GauLooseEngine(weight_path=weight)
    parent = _parent_engine(
        max_steps=max_steps, parent_backend=parent_backend, cuda_device=cuda_device
    )
    cfg = Epoch0Config(validation_roots=(rid,))

    print(
        f"[e0] {bid} root={rid} endpoint={ep} "
        f"cuda={cuda_device} parent={parent_backend}",
        flush=True,
    )
    pure_r = run_endpoint_route(
        geometry=geo,
        route_kind="pure_pyscf_reference",
        checkpoint_id="pure-pyscf-reference",
        profile=gau,
        aimnet2=None,
        parent=parent,
    )
    e0_r = run_endpoint_route(
        geometry=geo,
        route_kind="epoch_zero",
        checkpoint_id=cfg.checkpoint_id,
        profile=gau,
        aimnet2=aim,
        parent=parent,
    )
    root_dir = work_layout.epoch0_dir / rid
    root_dir.mkdir(parents=True, exist_ok=True)
    endpoint_payload = {
        "schema": "nhc0801-epoch0-endpoint-v1",
        "batch_id": bid,
        "root_id": rid,
        "endpoint": ep,
        "cuda_device": cuda_device,
        "parent_backend": parent_backend,
        "created_at_utc": _utc(),
        "pure_pyscf_reference": pure_r.as_dict(),
        "epoch0_route": e0_r.as_dict(),
        "status": "PASS" if (not pure_r.catastrophic and not e0_r.catastrophic) else "FAILED",
    }
    # Preferred: cation.json / neutral.json. Legacy *_shard.json kept readable only.
    endpoint_path = root_dir / f"{ep}.json"
    write_json(endpoint_path, endpoint_payload, overwrite=True)
    print(f"[e0] wrote {endpoint_path} status={endpoint_payload['status']}", flush=True)

    merge = try_merge_root_endpoints(
        work_layout.epoch0_dir / rid,
        root_id=rid,
        checkpoint_id=cfg.checkpoint_id,
        official_weight_sha256=cfg.official_weight_sha256,
    )
    if merge is not None:
        print(
            f"[e0] merged root result status={merge.get('status')} path={merge.get('path')}",
            flush=True,
        )
    return {
        "endpoint": endpoint_payload,
        "endpoint_path": str(endpoint_path),
        # backward keys for older CLIs/watchers
        "shard": endpoint_payload,
        "shard_path": str(endpoint_path),
        "root_merge": merge,
    }


def try_merge_root_endpoints(
    root_dir: Path,
    *,
    root_id: str,
    checkpoint_id: str,
    official_weight_sha256: str,
) -> dict | None:
    """If cation.json + neutral.json both exist, write root.json (+ legacy name)."""
    from nhc_deprot.contracts.parent_protocol import PROTOCOL_SHA256
    from nhc_deprot.pipeline.scientific_validation import EndpointRouteReceipt

    cat_p = endpoint_result_path(root_dir, "cation")
    neu_p = endpoint_result_path(root_dir, "neutral")
    if not (cat_p.is_file() and neu_p.is_file()):
        return None

    def _ep(payload: dict, key: str) -> EndpointRouteReceipt:
        raw = payload[key]
        return EndpointRouteReceipt(
            root_id=str(raw["root_id"]),
            endpoint=str(raw["endpoint"]),
            route_kind=str(raw["route_kind"]),
            checkpoint_id=str(raw["checkpoint_id"]),
            stages_completed=list(raw.get("stages_completed") or []),
            aimnet2_converged=bool(raw.get("aimnet2_converged", False)),
            aimnet2_steps=int(raw.get("aimnet2_steps") or 0),
            handoff_classification=raw.get("handoff_classification"),
            continue_parent_optimization=bool(
                raw.get("continue_parent_optimization", False)
            ),
            parent_geometry_converged=bool(raw.get("parent_geometry_converged", False)),
            parent_final_sp_converged=bool(raw.get("parent_final_sp_converged", False)),
            parent_final_state=raw.get("parent_final_state"),
            parent_energy_hartree=(
                float(raw["parent_energy_hartree"])
                if raw.get("parent_energy_hartree") is not None
                else None
            ),
            parent_opt_steps=int(raw.get("parent_opt_steps") or 0),
            parent_opt_steps_is_maxcap=bool(
                raw.get("parent_opt_steps_is_maxcap", True)
            ),
            parent_scf_cycles=int(raw.get("parent_scf_cycles") or 0),
            wall_seconds=float(raw.get("wall_seconds") or 0.0),
            identity_and_structure_ok=bool(raw.get("identity_and_structure_ok", False)),
            catastrophic=bool(raw.get("catastrophic", False)),
            catastrophic_reasons=list(raw.get("catastrophic_reasons") or []),
            aimnet2_energy_used_in_label=bool(
                raw.get("aimnet2_energy_used_in_label", False)
            ),
            single_point_only=bool(raw.get("single_point_only", False)),
            notes=list(raw.get("notes") or []),
        )

    cat_s = json.loads(cat_p.read_text(encoding="utf-8"))
    neu_s = json.loads(neu_p.read_text(encoding="utf-8"))
    pure_c = _ep(cat_s, "pure_pyscf_reference")
    pure_n = _ep(neu_s, "pure_pyscf_reference")
    e0_c = _ep(cat_s, "epoch0_route")
    e0_n = _ep(neu_s, "epoch0_route")

    pure_root = assemble_root_label(pure_c, pure_n, reference=None)
    e0_root = assemble_root_label(e0_c, e0_n, reference=None)
    # bind pure label as reference for error
    if pure_root.label_kcal is not None and e0_root.label_kcal is not None:
        e0_root.reference_label_kcal = pure_root.label_kcal
        e0_root.signed_label_error_kcal = e0_root.label_kcal - pure_root.label_kcal
        e0_root.absolute_label_error_kcal = abs(e0_root.signed_label_error_kcal)

    pure_steps = int(pure_c.parent_opt_steps) + int(pure_n.parent_opt_steps)
    e0_steps = int(e0_c.parent_opt_steps) + int(e0_n.parent_opt_steps)
    step_reduction = None if pure_steps <= 0 else (pure_steps - e0_steps) / pure_steps

    status = (
        "PASS"
        if (
            not pure_root.catastrophic_failure
            and not e0_root.catastrophic_failure
            and pure_root.all_identity_and_structure_hard_gates
            and e0_root.all_identity_and_structure_hard_gates
        )
        else "FAILED"
    )
    root_payload = {
        "schema": "nhc0801-epoch0-root-receipt-v1",
        "mindmap_step": 3,
        "root_id": root_id,
        "dry_run": False,
        "live_chemistry": True,
        "official_weight_sha256": official_weight_sha256,
        "checkpoint_id": checkpoint_id,
        "parent_protocol_sha256": PROTOCOL_SHA256,
        "single_point_only": False,
        "aimnet2_energy_enters_label": False,
        "pure_pyscf_reference": pure_root.as_dict(),
        "epoch0_route": e0_root.as_dict(),
        "comparison": {
            "pure_label_kcal": pure_root.label_kcal,
            "epoch0_label_kcal": e0_root.label_kcal,
            "absolute_label_error_kcal": e0_root.absolute_label_error_kcal,
            "signed_label_error_kcal": e0_root.signed_label_error_kcal,
            "pure_parent_opt_steps": pure_steps,
            "epoch0_parent_opt_steps": e0_steps,
            "parent_opt_step_reduction_fraction": step_reduction,
        },
        "status": status,
        "merged_from_endpoints": True,
    }
    # Preferred short name + legacy path used by rebuild/sci-val loaders.
    out = root_dir / "root.json"
    write_json(out, root_payload, overwrite=True)
    legacy = root_dir / "epoch0_root_receipt.json"
    write_json(legacy, root_payload, overwrite=True)
    return {"status": status, "path": str(out), "legacy_path": str(legacy)}


# Backward-compatible alias (do not use in new code / user-facing text).
try_merge_root_shards = try_merge_root_endpoints


def run_e0_for_val_roots(
    *,
    nhc0801_root: Path,
    generation_id: str,
    val_roots: Sequence[str],
    batch_id: str,
    max_steps: int = 250,
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
    parent = _parent_engine(
        max_steps=max_steps, parent_backend=parent_backend, cuda_device=cuda_device
    )

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
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--parent-backend", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--cuda-device", type=int, default=None)
    ap.add_argument(
        "--endpoint",
        default=None,
        help=(
            "Optional single endpoint (cation|neutral): one GPU runs one endpoint. "
            "Requires exactly one --val-roots entry."
        ),
    )
    ap.add_argument(
        "--use-official-epoch0-dir",
        action="store_true",
        help="(deprecated, ignored) all batches use epoch0_val_batches/<id>/",
    )
    args = ap.parse_args(argv)
    roots = [r.strip() for r in args.val_roots.split(",") if r.strip()]
    try:
        if args.endpoint is not None:
            if len(roots) != 1:
                raise E0ValOnlyError(
                    "--endpoint mode requires exactly one --val-roots root"
                )
            receipt = run_e0_single_endpoint(
                nhc0801_root=args.nhc0801_root,
                generation_id=args.generation_id,
                batch_id=args.batch_id,
                root_id=roots[0],
                endpoint=str(args.endpoint),
                max_steps=int(args.max_steps),
                parent_backend=args.parent_backend,
                cuda_device=args.cuda_device,
            )
        else:
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
    bid = normalize_epoch0_batch_id(args.batch_id)
    if args.endpoint is not None:
        ep_payload = None
        if isinstance(receipt, dict):
            ep_payload = receipt.get("endpoint") or receipt.get("shard")
        st = (ep_payload or {}).get("status") if isinstance(ep_payload, dict) else "UNKNOWN"
        print(
            json.dumps(
                {
                    "mode": "endpoint",
                    "status": st,
                    "batch_id": bid,
                    "root_id": roots[0],
                    "endpoint": args.endpoint,
                    "endpoint_path": receipt.get("endpoint_path")
                    or receipt.get("shard_path"),
                    "root_merge": receipt.get("root_merge"),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        print(f"E0_VAL_EXIT {st}", flush=True)
        return 0 if st == "PASS" else 1
    print(json.dumps({"status": receipt.get("status"), "batch_id": bid, "val_roots": roots}))
    print(f"E0_VAL_EXIT {receipt.get('status')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
