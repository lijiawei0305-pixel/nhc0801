"""Multi-GPU sci-val: cation/neutral 分开算（4 路 × 1 GPU），再组装 + 选模.

Hard rule: each Val root's cation and neutral run on distinct GPUs (same as
e0_val_4gpu). Parent backend = gpu4pyscf. Does not kill teacher daemons.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from nhc_deprot.contracts.tvt_gates import validate_numeric_addendum
from nhc_deprot.data.io_util import load_json_object, write_json
from nhc_deprot.data.paths import VALIDATION_ROOTS
from nhc_deprot.generation.layout import GenerationLayout, ensure_generation_tree
from nhc_deprot.pipeline.epoch0_campaign_rebuild import (
    epoch0_baseline_from_root_receipts,
    load_root_receipts,
    pure_references_from_root_receipts,
)
from nhc_deprot.pipeline.scientific_validation import (
    CheckpointScientificValidation,
    EndpointRouteReceipt,
    RootRouteReceipt,
    aggregate_checkpoint_validation,
    assemble_root_label,
    select_after_scientific_validation,
)
from nhc_deprot.pipeline.training_blockers import load_numeric_calibration
from nhc_deprot.resources.gpu_inventory import (
    GpuInventoryError,
    inventory_as_dict,
    pick_gpus,
)

ENDPOINTS: Final = ("cation", "neutral")
SCI_VAL_CAMPAIGN_SCHEMA: Final = "nhc0801-sci-val-campaign-v1"


class SciValDispatchError(RuntimeError):
    """Sci-val multi-GPU dispatch failed closed."""


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan_endpoint_jobs(
    roots: Sequence[str],
    *,
    gpu_ids: Sequence[int],
) -> list[dict[str, Any]]:
    roots_l = [r.strip() for r in roots if r and r.strip()]
    if len(roots_l) != 2:
        raise SciValDispatchError(f"need exactly 2 Val roots, got {roots_l}")
    if len(gpu_ids) != 4:
        raise SciValDispatchError(f"need exactly 4 GPU ids, got {list(gpu_ids)}")
    jobs: list[dict[str, Any]] = []
    k = 0
    for rid in roots_l:
        for ep in ENDPOINTS:
            jobs.append(
                {
                    "root_id": rid,
                    "endpoint": ep,
                    "gpu_index": int(gpu_ids[k]),
                }
            )
            k += 1
    return jobs


# Deprecated alias
plan_endpoint_shards = plan_endpoint_jobs


def launch_candidate_4gpu(
    *,
    nhc0801_root: Path,
    generation_id: str,
    candidate: Mapping[str, Any],
    max_steps: int = 250,
    max_gpu: int = 8,
    exclude_gpus: Sequence[int] | None = None,
    gpu_ids: Sequence[int] | None = None,
    allow_shared: bool = True,
    require_free: bool = False,
    dry_run: bool = False,
    python_exe: str | None = None,
    wait: bool = True,
    poll_seconds: float = 30.0,
) -> dict[str, Any]:
    """Launch 4 endpoint jobs (2 roots × cation/neutral) for one candidate."""

    seed = candidate.get("seed")
    epoch = candidate.get("epoch")
    wp = Path(str(candidate.get("weight_path") or ""))
    if type(seed) is not int or type(epoch) is not int:
        raise SciValDispatchError(f"candidate needs int seed/epoch: {candidate}")
    if not wp.is_file():
        raise SciValDispatchError(f"weight missing: {wp}")
    ck_id = str(
        candidate.get("checkpoint_id") or f"seed_{seed}_epoch_{epoch:04d}"
    )
    digest = _sha256_file(wp)

    if gpu_ids is not None:
        gpus = [int(x) for x in gpu_ids]
        if len(gpus) != 4:
            raise SciValDispatchError(f"gpu_ids must have length 4, got {gpus}")
    else:
        try:
            gpus = pick_gpus(
                4,
                max_gpu=max_gpu,
                exclude=exclude_gpus,
                allow_shared=allow_shared,
                require_free=require_free,
                # When all cards host other users' VASP, still run sci-val
                # (co-locate; never kill VASP). Teacher should be paused first.
                allow_vasp_share=True,
            )

        except GpuInventoryError as exc:
            raise SciValDispatchError(str(exc)) from exc

    roots = list(VALIDATION_ROOTS)
    jobs_plan = plan_endpoint_jobs(roots, gpu_ids=gpus)
    layout = resolve_or_layout(nhc0801_root, generation_id)
    log_dir = layout.logs_dir / "sci_val_4gpu"
    log_dir.mkdir(parents=True, exist_ok=True)

    py = python_exe or sys.executable
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = str(Path(nhc0801_root) / "src")
    env_base["PYTHONUNBUFFERED"] = "1"

    launched: list[dict[str, Any]] = []
    for job in jobs_plan:
        tag = (
            f"scival_s{seed}_e{epoch:04d}_{job['root_id'][:8]}_"
            f"{job['endpoint']}_gpu{job['gpu_index']}"
        )
        log_path = log_dir / f"{tag}.out"
        cmd = [
            py,
            "-u",
            "-m",
            "nhc_deprot.pipeline.sci_val_endpoint_shard",
            "--nhc0801-root",
            str(nhc0801_root),
            "--generation-id",
            generation_id,
            "--root-id",
            job["root_id"],
            "--endpoint",
            job["endpoint"],
            "--weight-path",
            str(wp),
            "--checkpoint-id",
            ck_id,
            "--seed",
            str(seed),
            "--epoch",
            str(epoch),
            "--max-steps",
            str(int(max_steps)),
            "--cuda-device",
            str(int(job["gpu_index"])),
        ]
        entry: dict[str, Any] = {
            **job,
            "log_path": str(log_path),
            "cmd": cmd,
            "seed": seed,
            "epoch": epoch,
            "checkpoint_id": ck_id,
        }
        if dry_run:
            entry["pid"] = None
            entry["status"] = "DRY_RUN_PLANNED"
        else:
            env = env_base.copy()
            # Physical GPU pin; LiveParentP01Engine also sets CVD for worker.
            env["CUDA_VISIBLE_DEVICES"] = str(int(job["gpu_index"]))
            log_fh = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(nhc0801_root),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            entry["pid"] = int(proc.pid)
            entry["status"] = "LAUNCHED"
            log_fh.close()
        launched.append(entry)

    receipt: dict[str, Any] = {
        "schema": "nhc0801-sci-val-4gpu-dispatch-v1",
        "created_at_utc": _utc(),
        "generation_id": generation_id,
        "seed": seed,
        "epoch": epoch,
        "checkpoint_id": ck_id,
        "checkpoint_sha256": digest,
        "weight_path": str(wp),
        "parent_max_steps": int(max_steps),
        "parent_backend": "gpu",
        "gpu_ids": gpus,
        "val_roots": roots,
        "endpoints": launched,
        "shards": launched,  # legacy key
        "inventory": inventory_as_dict(max_gpu=max_gpu),
        "dry_run": dry_run,
        "notes": [
            "sci-val: 2 roots × cation/neutral 分开算 → 4 GPUs",
            "gpu4pyscf parent; AIMNet2 GAU_LOOSE per endpoint",
            "does not kill teacher daemon",
        ],
    }
    plan_path = log_dir / f"dispatch_s{seed}_e{epoch:04d}_{int(time.time())}.json"
    plan_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["plan_path"] = str(plan_path)

    if dry_run or not wait:
        return receipt

    # Wait for all pids. Note: os.kill(pid, 0) is TRUE for zombies — if the
    # parent never wait()s, EXITED children stay Z and this loop hung forever
    # (2026-08-05: candidate-1 4/4 PASS, GPUs idle, parent stuck ~30min).
    def _pid_finished(pid: int) -> bool:
        try:
            # Reap our own children if any (non-blocking).
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        except OSError:
            return True
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except OSError:
            return True  # gone
        for line in status.splitlines():
            if line.startswith("State:"):
                st = line.split()[1] if len(line.split()) > 1 else ""
                # Z = zombie (work done); anything else still running
                return st == "Z"
        return False

    pending = {int(s["pid"]): s for s in launched if s.get("pid") is not None}
    while pending:
        done: list[int] = []
        for pid, job in pending.items():
            if _pid_finished(pid):
                done.append(pid)
                job["status"] = "EXITED"
        for pid in done:
            del pending[pid]
        if pending:
            time.sleep(poll_seconds)

    # Assemble checkpoint from endpoint results
    assembled = assemble_candidate_from_endpoints(
        layout=layout,
        seed=seed,
        epoch=epoch,
        checkpoint_id=ck_id,
        checkpoint_sha256=digest,
        epoch0_batch_id="g001",
        parent_max_steps=int(max_steps),
    )
    receipt["assemble"] = assembled
    receipt["finished_at_utc"] = _utc()
    plan_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def resolve_or_layout(nhc0801_root: Path, generation_id: str) -> GenerationLayout:
    from nhc_deprot.generation.layout import resolve_layout

    layout = resolve_layout(generation_id=generation_id, nhc0801_root=nhc0801_root)
    ensure_generation_tree(layout, exist_ok=True)
    return layout


def _endpoint_from_payload(raw: Mapping[str, Any]) -> EndpointRouteReceipt:
    route = raw.get("route")
    if not isinstance(route, dict):
        raise SciValDispatchError(f"endpoint result missing route: {raw.get('root_id')}")
    return EndpointRouteReceipt(
        root_id=str(route["root_id"]),
        endpoint=str(route["endpoint"]),
        route_kind=str(route.get("route_kind") or "finetuned_checkpoint"),
        checkpoint_id=str(route.get("checkpoint_id") or ""),
        stages_completed=list(route.get("stages_completed") or []),
        aimnet2_converged=bool(route.get("aimnet2_converged", False)),
        aimnet2_steps=int(route.get("aimnet2_steps") or 0),
        handoff_classification=route.get("handoff_classification"),
        continue_parent_optimization=bool(
            route.get("continue_parent_optimization", False)
        ),
        parent_geometry_converged=bool(route.get("parent_geometry_converged", False)),
        parent_final_sp_converged=bool(route.get("parent_final_sp_converged", False)),
        parent_final_state=route.get("parent_final_state"),
        parent_energy_hartree=(
            float(route["parent_energy_hartree"])
            if route.get("parent_energy_hartree") is not None
            else None
        ),
        parent_opt_steps=int(route.get("parent_opt_steps") or 0),
        parent_opt_steps_is_maxcap=bool(
            route.get("parent_opt_steps_is_maxcap", True)
        ),
        parent_scf_cycles=int(route.get("parent_scf_cycles") or 0),
        wall_seconds=float(route.get("wall_seconds") or 0.0),
        identity_and_structure_ok=bool(route.get("identity_and_structure_ok", False)),
        catastrophic=bool(route.get("catastrophic", False)),
        catastrophic_reasons=list(route.get("catastrophic_reasons") or []),
        aimnet2_energy_used_in_label=bool(
            route.get("aimnet2_energy_used_in_label", False)
        ),
        single_point_only=bool(route.get("single_point_only", False)),
        notes=list(route.get("notes") or []),
    )


def assemble_candidate_from_endpoints(
    *,
    layout: GenerationLayout,
    seed: int,
    epoch: int,
    checkpoint_id: str,
    checkpoint_sha256: str,
    epoch0_batch_id: str = "g001",
    parent_max_steps: int = 250,
) -> dict[str, Any]:
    """Merge 4 endpoint results + e0 pure refs → CheckpointScientificValidation."""

    ep_dir = (
        layout.sci_val_dir / f"seed_{seed}" / f"epoch_{epoch:04d}" / "endpoints"
    )
    # Legacy directory name
    if not ep_dir.is_dir():
        ep_dir = (
            layout.sci_val_dir
            / f"seed_{seed}"
            / f"epoch_{epoch:04d}"
            / "endpoint_shards"
        )
    if not ep_dir.is_dir():
        raise SciValDispatchError(f"missing endpoints dir: {ep_dir}")

    by_root: dict[str, dict[str, EndpointRouteReceipt]] = {}
    paths = list(ep_dir.glob("*.json")) + list(ep_dir.glob("*_shard.json"))
    for path in sorted(set(paths)):
        payload, _ = load_json_object(path)
        if not isinstance(payload, dict):
            continue
        ep_rec = _endpoint_from_payload(payload)
        by_root.setdefault(ep_rec.root_id, {})[ep_rec.endpoint] = ep_rec

    e0_dir = layout.epoch0_batch_dir(epoch0_batch_id)
    root_receipts_raw = load_root_receipts(e0_dir, list(VALIDATION_ROOTS))
    refs = pure_references_from_root_receipts(root_receipts_raw)
    e0_baseline = epoch0_baseline_from_root_receipts(root_receipts_raw)

    # Attach parent_max_steps on baseline for mismatch checks (attribute).
    # CheckpointScientificValidation is a non-frozen @dataclass (not slots);
    # setattr(obj, "const", v) == obj.const = v (ruff B010).
    e0_baseline.parent_max_steps = int(parent_max_steps)

    roots_out: list[RootRouteReceipt] = []
    for rid in VALIDATION_ROOTS:
        eps = by_root.get(rid) or {}
        if set(eps) != set(ENDPOINTS):
            raise SciValDispatchError(
                f"root {rid} incomplete endpoints: {sorted(eps)}"
            )
        ref = refs[rid]
        roots_out.append(
            assemble_root_label(eps["cation"], eps["neutral"], reference=ref)
        )

    # epoch0 means for burden
    e0_mae = e0_baseline.mean_absolute_label_error_kcal_mol
    s, c, w, n = 0.0, 0.0, 0.0, 0
    for r in e0_baseline.root_receipts:
        for ep in (r.cation, r.neutral):
            if ep is None:
                continue
            s += ep.parent_opt_steps
            c += ep.parent_scf_cycles
            w += ep.wall_seconds
            n += 1
    e0_steps = s / n if n else None
    e0_scf = c / n if n else None
    e0_wall = w / n if n else None

    agg = aggregate_checkpoint_validation(
        epoch=int(epoch),
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        route_kind="finetuned_checkpoint",
        root_receipts=roots_out,
        epoch0_mae=e0_mae,
        epoch0_mean_parent_steps=e0_steps,
        epoch0_mean_scf_cycles=e0_scf,
        epoch0_mean_wall=e0_wall,
        live_chemistry_executed=True,
    )
    out_dir = layout.sci_val_dir / f"seed_{seed}" / f"epoch_{epoch:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = agg.as_dict()
    payload["seed"] = seed
    payload["parent_max_steps"] = int(parent_max_steps)
    write_json(out_dir / "sci_val_receipt.json", payload, overwrite=True)
    return {
        "status": "ASSEMBLED",
        "receipt_path": str(out_dir / "sci_val_receipt.json"),
        "aggregate": payload,
        "epoch0_mae": e0_mae,
        "parent_opt_steps_unmeasured": agg.parent_opt_steps_unmeasured,
        "pyscf_geometry_work_reduction_fraction": (
            agg.pyscf_geometry_work_reduction_fraction
        ),
    }


# Deprecated alias
assemble_candidate_from_shards = assemble_candidate_from_endpoints


def run_sci_val_campaign_4gpu(
    *,
    nhc0801_root: Path,
    generation_id: str,
    candidates: Sequence[Mapping[str, Any]],
    max_steps: int = 250,
    epoch0_max_steps: int | None = None,
    max_candidates: int = 2,
    allow_shared: bool = True,
    dry_run: bool = False,
    wait: bool = True,
) -> dict[str, Any]:
    """Run multi-GPU sci-val for shortlist candidates; write campaign receipt."""

    e0_steps = int(epoch0_max_steps if epoch0_max_steps is not None else max_steps)
    if int(e0_steps) != int(max_steps):
        raise SciValDispatchError(
            "BASELINE_CONFIG_MISMATCH: parent_max_steps "
            f"epoch0={e0_steps} vs candidates={max_steps}"
        )

    layout = resolve_or_layout(nhc0801_root, generation_id)
    cand_list = list(candidates)[: int(max_candidates)]
    if not cand_list:
        raise SciValDispatchError("no candidates")

    # Ensure e0 root receipts exist
    e0_dir = layout.epoch0_batch_dir("g001")
    load_root_receipts(e0_dir, list(VALIDATION_ROOTS))

    per_cand: list[dict[str, Any]] = []
    val_objects: list[CheckpointScientificValidation] = []

    for cand in cand_list:
        disp = launch_candidate_4gpu(
            nhc0801_root=nhc0801_root,
            generation_id=generation_id,
            candidate=cand,
            max_steps=int(max_steps),
            allow_shared=allow_shared,
            dry_run=dry_run,
            wait=wait and not dry_run,
        )
        per_cand.append(disp)
        if dry_run or not wait:
            continue
        ass = disp.get("assemble") or {}
        agg_dict = ass.get("aggregate")
        if not isinstance(agg_dict, dict):
            raise SciValDispatchError(
                f"assemble failed for seed={cand.get('seed')} epoch={cand.get('epoch')}"
            )
        # rebuild object for selection via re-read receipt
        seed = int(cand["seed"])
        epoch = int(cand["epoch"])
        receipt_path = (
            layout.sci_val_dir
            / f"seed_{seed}"
            / f"epoch_{epoch:04d}"
            / "sci_val_receipt.json"
        )
        # selection needs CheckpointScientificValidation; re-assemble root objects
        ep_dir = receipt_path.parent / "endpoints"
        if not ep_dir.is_dir():
            ep_dir = receipt_path.parent / "endpoint_shards"
        by_root: dict[str, dict[str, EndpointRouteReceipt]] = {}
        for path in sorted(set(list(ep_dir.glob("*.json")) + list(ep_dir.glob("*_shard.json")))):
            payload, _ = load_json_object(path)
            if isinstance(payload, dict):
                ep_rec = _endpoint_from_payload(payload)
                by_root.setdefault(ep_rec.root_id, {})[ep_rec.endpoint] = ep_rec
        root_receipts_raw = load_root_receipts(e0_dir, list(VALIDATION_ROOTS))
        refs = pure_references_from_root_receipts(root_receipts_raw)
        e0_baseline = epoch0_baseline_from_root_receipts(root_receipts_raw)
        roots_out: list[RootRouteReceipt] = []
        for rid in VALIDATION_ROOTS:
            eps = by_root[rid]
            roots_out.append(
                assemble_root_label(eps["cation"], eps["neutral"], reference=refs[rid])
            )
        s, c, w, n = 0.0, 0.0, 0.0, 0
        for r in e0_baseline.root_receipts:
            for ep in (r.cation, r.neutral):
                if ep is None:
                    continue
                s += ep.parent_opt_steps
                c += ep.parent_scf_cycles
                w += ep.wall_seconds
                n += 1
        agg = aggregate_checkpoint_validation(
            epoch=epoch,
            checkpoint_id=str(cand.get("checkpoint_id") or f"seed_{seed}_epoch_{epoch:04d}"),
            checkpoint_sha256=str(disp.get("checkpoint_sha256") or ("a" * 64)),
            route_kind="finetuned_checkpoint",
            root_receipts=roots_out,
            epoch0_mae=e0_baseline.mean_absolute_label_error_kcal_mol,
            epoch0_mean_parent_steps=(s / n if n else None),
            epoch0_mean_scf_cycles=(c / n if n else None),
            epoch0_mean_wall=(w / n if n else None),
            live_chemistry_executed=True,
        )
        val_objects.append(agg)

    if dry_run:
        campaign = {
            "schema": SCI_VAL_CAMPAIGN_SCHEMA,
            "status": "DRY_RUN_SCI_VAL_4GPU_PLANNED",
            "parent_max_steps": int(max_steps),
            "epoch0_parent_max_steps": e0_steps,
            "candidate_dispatches": per_cand,
            "final_model_selected": False,
            "final_test_authorized": False,
        }
        write_json(layout.sci_val_dir / "campaign_receipt_4gpu_plan.json", campaign, overwrite=True)
        return campaign

    if not wait:
        campaign = {
            "schema": SCI_VAL_CAMPAIGN_SCHEMA,
            "status": "LIVE_SCI_VAL_4GPU_LAUNCHED",
            "parent_max_steps": int(max_steps),
            "epoch0_parent_max_steps": e0_steps,
            "candidate_dispatches": per_cand,
            "final_model_selected": False,
            "final_test_authorized": False,
        }
        write_json(
            layout.sci_val_dir / "campaign_receipt_4gpu_launched.json",
            campaign,
            overwrite=True,
        )
        return campaign

    addendum = load_numeric_calibration()
    validate_numeric_addendum(addendum)
    selection = select_after_scientific_validation(val_objects, numeric_addendum=addendum)
    campaign = {
        "schema": SCI_VAL_CAMPAIGN_SCHEMA,
        "mindmap_steps": [8, 9],
        "generation_id": generation_id,
        "dry_run": False,
        "scientific_validation_live": True,
        "parallel_mode": "4gpu_endpoint_parallel",
        "parent_backend": "gpu",
        "parent_max_steps": int(max_steps),
        "epoch0_parent_max_steps": e0_steps,
        "status": (
            "LIVE_SCI_VAL_PASS"
            if selection.get("outcome") == "VALIDATION_SELECTED"
            else "LIVE_SCI_VAL_REJECTED"
        ),
        "candidate_count": len(val_objects),
        "candidate_results": [v.as_dict() for v in val_objects],
        "candidate_dispatches": per_cand,
        "selection": selection,
        "final_model_selected": selection.get("outcome") == "VALIDATION_SELECTED",
        "final_test_authorized": False,
        "final_test_payload_read": False,
        "numeric_addendum_version": addendum.get("version"),
        "finished_at_utc": _utc(),
        "notes": [
            "P2 multi-GPU sci-val; parent max_steps matched to e0 baseline",
            "burden hard gate only when opt_steps measured (non-maxcap)",
            "Final Test remains sealed",
        ],
    }
    write_json(layout.sci_val_dir / "campaign_receipt.json", campaign, overwrite=True)
    write_json(layout.logs_dir / "sci_val_campaign_receipt.json", campaign, overwrite=True)
    return campaign
