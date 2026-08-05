#!/usr/bin/env python3
"""Run multi-start basin perturbation pre-screen matrix (CPU, zero DFT).

Does not modify production ranking. Writes JSONL under
``pre_screen_g001/basin_perturb_v1/`` only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Candidates: (run_id, epoch, group) — seed fixed 20260730
CANDIDATES: list[dict[str, Any]] = [
    # fragile (CUDA→CPU RMSD jump ~0.05 Å)
    {"run_id": "e1f1_mlp_shift", "epoch": 70, "group": "fragile",
     "gpu_rmsd": 0.12401, "cpu_rmsd": 0.18223},
    {"run_id": "e1f1_mlp", "epoch": 70, "group": "fragile",
     "gpu_rmsd": 0.12467, "cpu_rmsd": 0.18248},
    {"run_id": "e1f100_mlp", "epoch": 10, "group": "fragile",
     "gpu_rmsd": 0.17080, "cpu_rmsd": 0.12291},
    {"run_id": "e1f1_mlp", "epoch": 10, "group": "fragile",
     "gpu_rmsd": 0.12605, "cpu_rmsd": 0.17228},
    # robust
    {"run_id": "e1f100_mlp_shift", "epoch": 10, "group": "robust",
     "gpu_rmsd": 0.12181, "cpu_rmsd": 0.12288},
    {"run_id": "e1f1_mlp", "epoch": 30, "group": "robust",
     "gpu_rmsd": 0.12282, "cpu_rmsd": 0.12345},
    {"run_id": "e1f1_mlp_shift", "epoch": 30, "group": "robust",
     "gpu_rmsd": 0.12293, "cpu_rmsd": 0.12351},
    {"run_id": "e1f1_mlp_shift", "epoch": 10, "group": "robust",
     "gpu_rmsd": 0.12575, "cpu_rmsd": 0.12605},
]

EPSILONS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
RNG_SEEDS_NONEZERO = [101, 202, 303, 404, 505, 606]
TRAIN_SEED = 20260730


def _weight_path(nhc: Path, run_id: str, epoch: int) -> Path:
    return (
        nhc
        / "runs"
        / "nhc0801-g001"
        / "train_g001"
        / "runs"
        / run_id
        / f"seed_{TRAIN_SEED}"
        / f"epoch_{epoch:04d}.pt"
    )


def _checkpoint_id(run_id: str, epoch: int) -> str:
    return f"{run_id}_seed_{TRAIN_SEED}_epoch_{epoch:04d}"


def build_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for c in CANDIDATES:
        cid = _checkpoint_id(c["run_id"], c["epoch"])
        for eps in EPSILONS:
            seeds = [0] if eps == 0.0 else list(RNG_SEEDS_NONEZERO)
            for rs in seeds:
                jobs.append(
                    {
                        "run_id": c["run_id"],
                        "epoch": c["epoch"],
                        "group": c["group"],
                        "checkpoint_id": cid,
                        "epsilon_angstrom": eps,
                        "rng_seed": rs,
                        "gpu_rmsd": c["gpu_rmsd"],
                        "cpu_rmsd": c["cpu_rmsd"],
                    }
                )
    return jobs


def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entry — imports inside to keep process spawn clean."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", str(payload.get("omp_threads", 12)))

    from nhc_deprot.data.paths import VALIDATION_ROOTS
    from nhc_deprot.generation.layout import resolve_layout
    from nhc_deprot.pipeline.basin_perturbation import perturb_start_geometry
    from nhc_deprot.pipeline.live_pre_screen_engine import make_engine_factory
    from nhc_deprot.pipeline.pre_screen import (
        CheckpointCandidate,
        load_teacher_references_for_batch,
        screen_checkpoint,
    )

    nhc = Path(payload["nhc0801_root"])
    layout = resolve_layout(
        generation_id=payload["generation_id"],
        nhc0801_root=nhc,
    )
    teacher_dir = (
        Path(payload["teacher_batch_dir"])
        if payload.get("teacher_batch_dir")
        else layout.teacher_batch_dir(payload["batch_id"])
    )

    refs = load_teacher_references_for_batch(
        layout,
        batch_id=payload["batch_id"],
        root_ids=list(VALIDATION_ROOTS),
        teacher_batch_dir=teacher_dir,
    )
    if not refs:
        raise RuntimeError(f"no teacher refs under {teacher_dir}")

    wpath = _weight_path(nhc, payload["run_id"], int(payload["epoch"]))
    if not wpath.is_file():
        raise FileNotFoundError(wpath)

    cand = CheckpointCandidate(
        checkpoint_id=payload["checkpoint_id"],
        run_id=payload["run_id"],
        seed=TRAIN_SEED,
        epoch=int(payload["epoch"]),
        weight_path=str(wpath),
    )

    eps = float(payload["epsilon_angstrom"])
    rs = int(payload["rng_seed"])
    applied: list[float] = []
    pert_refs = []
    for i, ref in enumerate(refs):
        # independent but reproducible per-endpoint seed
        ep_seed = rs if eps == 0.0 else rs + (i + 1) * 1_000_003
        pr = perturb_start_geometry(
            ref, epsilon_angstrom=eps, rng_seed=ep_seed
        )
        applied.append(pr.applied_rms_displacement)
        pert_refs.append(pr.reference)

    t0 = time.perf_counter()
    factory = make_engine_factory(device="cpu")
    engine = factory(cand)
    result = screen_checkpoint(engine, cand, pert_refs)
    wall = time.perf_counter() - t0

    rec = {
        "checkpoint_id": payload["checkpoint_id"],
        "run_id": payload["run_id"],
        "epoch": int(payload["epoch"]),
        "group": payload["group"],
        "epsilon_angstrom": eps,
        "rng_seed": rs,
        "applied_rms_displacement": float(sum(applied) / max(len(applied), 1)),
        "applied_rms_per_endpoint": applied,
        "mean_rmsd_to_reference_angstrom": result.mean_rmsd_to_reference_angstrom,
        "mean_aimnet2_steps_to_gau_loose": result.mean_aimnet2_steps_to_gau_loose,
        "mean_force_rmse_at_reference_ev_per_a": (
            result.mean_force_rmse_at_reference_ev_per_a
        ),
        "hard_gates_passed": result.hard_gates_passed,
        "wall_seconds": wall,
        "gpu_rmsd_ref": payload.get("gpu_rmsd"),
        "cpu_rmsd_ref": payload.get("cpu_rmsd"),
        "per_endpoint": [m.as_dict() for m in result.per_endpoint],
    }
    return rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--batch-id", default="g001")
    p.add_argument(
        "--teacher-batch-dir",
        type=Path,
        default=None,
        help="Default: layout teacher_gpu_g001",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--omp-threads", type=int, default=12)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <nhc>/runs/.../pre_screen_g001/basin_perturb_v1",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only run first N jobs (smoke)",
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help="2 fragile + 2 robust only, eps in {0,1e-4,1e-2}, 2 rng seeds",
    )
    args = p.parse_args(argv)

    nhc = args.nhc0801_root.resolve()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            nhc
            / "runs"
            / args.generation_id
            / "pre_screen_g001"
            / "basin_perturb_v1"
        )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    meta_path = out_dir / "matrix_meta.json"

    jobs = build_jobs()
    if args.pilot:
        # first 2 fragile + first 2 robust
        keep_cids = {
            _checkpoint_id("e1f1_mlp_shift", 70),
            _checkpoint_id("e1f1_mlp", 70),
            _checkpoint_id("e1f100_mlp_shift", 10),
            _checkpoint_id("e1f1_mlp", 30),
        }
        pilot_eps = {0.0, 1e-4, 1e-2}
        pilot_seeds = {0, 101, 202}
        jobs = [
            j
            for j in jobs
            if j["checkpoint_id"] in keep_cids
            and j["epsilon_angstrom"] in pilot_eps
            and (j["epsilon_angstrom"] == 0.0 or j["rng_seed"] in pilot_seeds)
        ]
    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]

    # skip already done
    done_keys: set[tuple[Any, ...]] = set()
    if jsonl_path.is_file():
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done_keys.add(
                    (r["checkpoint_id"], r["epsilon_angstrom"], r["rng_seed"])
                )
    pending = [
        j
        for j in jobs
        if (j["checkpoint_id"], j["epsilon_angstrom"], j["rng_seed"])
        not in done_keys
    ]

    meta = {
        "n_jobs_planned": len(jobs),
        "n_pending": len(pending),
        "n_done_before": len(done_keys),
        "workers": args.workers,
        "omp_threads": args.omp_threads,
        "epsilons": EPSILONS,
        "rng_seeds_nonzero": RNG_SEEDS_NONEZERO,
        "candidates": CANDIDATES,
        "pilot": bool(args.pilot),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"status": "START", **meta}, indent=2), flush=True)

    if not pending:
        print(json.dumps({"status": "NOTHING_TO_DO"}), flush=True)
        return 0

    payloads = []
    for j in pending:
        payloads.append(
            {
                **j,
                "nhc0801_root": str(nhc),
                "generation_id": args.generation_id,
                "batch_id": args.batch_id,
                "teacher_batch_dir": (
                    str(args.teacher_batch_dir)
                    if args.teacher_batch_dir
                    else None
                ),
                "omp_threads": args.omp_threads,
            }
        )

    t0 = time.perf_counter()
    n_ok = 0
    n_fail = 0
    # ProcessPoolExecutor needs picklable top-level function
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {ex.submit(_run_one, pl): pl for pl in payloads}
        with jsonl_path.open("a") as out:
            for fut in as_completed(futs):
                pl = futs[fut]
                try:
                    rec = fut.result()
                    out.write(json.dumps(rec, sort_keys=True) + "\n")
                    out.flush()
                    n_ok += 1
                    print(
                        f"[ok {n_ok}/{len(pending)}] "
                        f"{rec['checkpoint_id']} eps={rec['epsilon_angstrom']} "
                        f"seed={rec['rng_seed']} rmsd={rec['mean_rmsd_to_reference_angstrom']:.5f} "
                        f"F={rec['mean_force_rmse_at_reference_ev_per_a']:.5f} "
                        f"wall={rec['wall_seconds']:.1f}s",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 — report & continue
                    n_fail += 1
                    err = {
                        "error": str(exc),
                        "checkpoint_id": pl["checkpoint_id"],
                        "epsilon_angstrom": pl["epsilon_angstrom"],
                        "rng_seed": pl["rng_seed"],
                    }
                    out.write(json.dumps({"status": "FAIL", **err}) + "\n")
                    out.flush()
                    print(f"[FAIL] {err}", flush=True)

    wall = time.perf_counter() - t0
    summary = {
        "status": "DONE",
        "n_ok": n_ok,
        "n_fail": n_fail,
        "wall_seconds": wall,
        "jsonl": str(jsonl_path),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
