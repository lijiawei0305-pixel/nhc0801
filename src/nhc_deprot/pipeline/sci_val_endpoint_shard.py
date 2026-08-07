"""One sci-val endpoint job (finetuned route) for multi-GPU parallel.

Runs AIMNet2 GAU_LOOSE + full parent GAU on a single (root, cation|neutral).
Parent uses gpu4pyscf via LiveParentP01Engine. Writes one endpoint JSON; the
assembler merges cation+neutral into root-level checkpoint aggregates.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nhc_deprot.data.io_util import write_json
from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout
from nhc_deprot.pipeline.e0_val_only import load_geo
from nhc_deprot.pipeline.live_epoch0 import LiveParentP01Engine
from nhc_deprot.pipeline.live_pre_screen_engine import LiveCheckpointGauLooseEngine
from nhc_deprot.pipeline.parent_handoff import load_gau_loose_profile
from nhc_deprot.pipeline.scientific_validation import run_endpoint_route

DEFAULT_GOLD = Path("/home/plab/test/WJW/data/runs/mol_gold/xyz")
ALT_GOLD = (
    Path("/home/plab/test/WJW/data/candidates/structures_full/xyz"),
    Path("/home/plab/test/WJW/data/candidates/xyz"),
)


class SciValEndpointError(RuntimeError):
    """Sci-val endpoint job failed closed."""


# Deprecated alias
SciValShardError = SciValEndpointError


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_sci_val_endpoint_job(
    *,
    nhc0801_root: Path,
    generation_id: str,
    root_id: str,
    endpoint: str,
    weight_path: Path,
    checkpoint_id: str,
    seed: int,
    epoch: int,
    max_steps: int = 250,
    cuda_device: int,
    gold_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Run finetuned route for one endpoint; write under sci_val/.../endpoints/."""

    rid = str(root_id).strip()
    if rid in TRAIN_ROOTS:
        raise SciValEndpointError(f"refused train root: {rid}")
    if rid not in VALIDATION_ROOTS:
        # allow only Val roots for sci-val pilot
        raise SciValEndpointError(f"root not in VALIDATION_ROOTS: {rid}")
    ep = str(endpoint).strip().lower()
    if ep not in {"cation", "neutral"}:
        raise SciValEndpointError(f"endpoint must be cation|neutral, got {endpoint!r}")
    wp = Path(weight_path)
    if not wp.is_file():
        raise SciValEndpointError(f"weight missing: {wp}")

    layout = resolve_layout(generation_id=generation_id, nhc0801_root=nhc0801_root)
    ensure_generation_tree(layout, exist_ok=True)
    out_dir = (
        layout.sci_val_dir
        / f"seed_{int(seed)}"
        / f"epoch_{int(epoch):04d}"
        / "endpoints"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    golds = list(gold_dirs or (DEFAULT_GOLD, *ALT_GOLD))
    geo = load_geo(rid, ep, golds)
    gau = load_gau_loose_profile()
    # When the process is already pinned via CUDA_VISIBLE_DEVICES=<physical>,
    # AIMNet2 sees a single device as cuda:0. Do not pass cuda:<physical>.
    aim = LiveCheckpointGauLooseEngine(
        weight_path=wp,
        max_steps=None,  # GAU_LOOSE contract budget
        device="cuda:0",
    )
    parent = LiveParentP01Engine(
        max_steps=int(max_steps),
        backend="gpu",
        cuda_device=int(cuda_device),
        host_threads=2,
    )
    print(
        f"[sci-val] root={rid} endpoint={ep} cuda={cuda_device} "
        f"seed={seed} epoch={epoch} max_steps={max_steps}",
        flush=True,
    )
    receipt = run_endpoint_route(
        geometry=geo,
        route_kind="finetuned_checkpoint",
        checkpoint_id=checkpoint_id,
        profile=gau,
        aimnet2=aim,
        parent=parent,
    )
    payload = {
        "schema": "nhc0801-sci-val-endpoint-v1",
        "created_at_utc": _utc(),
        "generation_id": generation_id,
        "root_id": rid,
        "endpoint": ep,
        "seed": int(seed),
        "epoch": int(epoch),
        "checkpoint_id": checkpoint_id,
        "weight_path": str(wp),
        "cuda_device": int(cuda_device),
        "parent_max_steps": int(max_steps),
        "parent_backend": "gpu",
        "route": receipt.as_dict(),
        "status": "PASS" if not receipt.catastrophic else "FAILED",
    }
    path = out_dir / f"{rid}_{ep}.json"
    write_json(path, payload, overwrite=True)
    print(
        f"[sci-val] wrote {path} status={payload['status']} "
        f"opt_steps={receipt.parent_opt_steps} "
        f"maxcap={receipt.parent_opt_steps_is_maxcap}",
        flush=True,
    )
    return {
        "endpoint": payload,
        "endpoint_path": str(path),
        "shard": payload,
        "shard_path": str(path),
    }


# Deprecated alias
run_sci_val_endpoint_shard = run_sci_val_endpoint_job


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nhc0801-root", type=Path, required=True)
    ap.add_argument("--generation-id", default="nhc0801-g001")
    ap.add_argument("--root-id", required=True)
    ap.add_argument("--endpoint", required=True, choices=("cation", "neutral"))
    ap.add_argument("--weight-path", type=Path, required=True)
    ap.add_argument("--checkpoint-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--cuda-device", type=int, required=True)
    args = ap.parse_args(argv)
    try:
        out = run_sci_val_endpoint_job(
            nhc0801_root=args.nhc0801_root,
            generation_id=args.generation_id,
            root_id=args.root_id,
            endpoint=args.endpoint,
            weight_path=args.weight_path,
            checkpoint_id=args.checkpoint_id,
            seed=args.seed,
            epoch=args.epoch,
            max_steps=int(args.max_steps),
            cuda_device=int(args.cuda_device),
        )
    except SciValEndpointError as exc:
        print(json.dumps({"error": str(exc), "status": "FAIL"}, indent=2))
        return 2
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "FAIL"},
                indent=2,
            )
        )
        return 1
    st = (out.get("endpoint") or out.get("shard") or {}).get("status")
    print(
        json.dumps(
            {
                "status": st,
                "endpoint_path": out.get("endpoint_path") or out.get("shard_path"),
            },
            indent=2,
        )
    )
    return 0 if st == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
