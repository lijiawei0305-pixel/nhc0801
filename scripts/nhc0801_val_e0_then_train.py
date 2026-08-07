#!/usr/bin/env python3
"""Watch Val-16 e0 (TVT resplit) → rebuild campaign → D3 + weighted NPZ → live ablation train.

Server ops entry (mlff env). Does **not** kill gpu_teacher_daemon / e0 workers.
Does **not** open Final Test.

Phases:
  1. poll_val_e0     — until all VALIDATION_ROOTS have terminal root receipts
  2. rebuild_e0      — campaign from root receipts (no DFT)
  3. teacher_view    — symlink ready roots from any teacher_gpu_g* → staging view
  4. d3_weighted     — live Dftd3Projector + assemble_weighted_dataset (ready roots only)
  5. train_ablation  — phase-1 4× run_id live multi-seed (aimnet2_train_authorized)

Status JSON: runs/<gen>/logs/val_e0_then_train/status.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.data.paths import (  # noqa: E402
    DEFAULT_NHC0801,
    TRAIN_ROOTS,
    VALIDATION_ROOTS,
)
from nhc_deprot.generation.layout import (  # noqa: E402
    DEFAULT_GENERATION_ID,
    ensure_generation_tree,
    resolve_layout,
)
from nhc_deprot.pipeline.d3_projection import Dftd3Projector, run_d3_campaign  # noqa: E402
from nhc_deprot.pipeline.eligible_full_pairs import scan_teacher_products  # noqa: E402
from nhc_deprot.pipeline.epoch0_campaign_rebuild import (  # noqa: E402
    rebuild_epoch0_campaign_from_root_receipts,
    resolve_epoch0_root_receipt_path,
)
from nhc_deprot.pipeline.weighted_dataset_writer import assemble_weighted_dataset  # noqa: E402
from nhc_deprot.training.ablation_cli import (  # noqa: E402
    DEFAULT_ABLATION_RUN_IDS,
    run_train_ablation,
)
from nhc_deprot.training.config import TrainingConfig  # noqa: E402
from nhc_deprot.training.merge_meta import (  # noqa: E402
    build_merge_meta,
    write_merge_meta,
)


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _log(log_path: Path, msg: str) -> None:
    line = f"[{_utc()}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def val_e0_status(gen: Path, batch_id: str = "g001") -> dict[str, Any]:
    """Val e0 root status using the same receipt resolution as campaign rebuild (G1)."""

    e0 = gen / "epoch0_val_batches" / batch_id / "epoch0"
    rows: list[dict[str, Any]] = []
    n_pass = n_fail = n_partial = n_miss = 0
    for rid in VALIDATION_ROOTS:
        root_dir = e0 / rid
        rec = resolve_epoch0_root_receipt_path(e0, rid)
        cat = (root_dir / "cation.json").is_file() or (
            root_dir / "cation_shard.json"
        ).is_file()
        neu = (root_dir / "neutral.json").is_file() or (
            root_dir / "neutral_shard.json"
        ).is_file()
        if rec is not None and rec.is_file():
            try:
                st = str(json.loads(rec.read_text(encoding="utf-8")).get("status") or "")
            except (OSError, json.JSONDecodeError):
                st = "UNREADABLE"
            rows.append(
                {
                    "root_id": rid,
                    "status": st,
                    "cation": cat,
                    "neutral": neu,
                    "receipt": rec.name,
                }
            )
            if st == "PASS":
                n_pass += 1
            else:
                n_fail += 1
        elif cat or neu:
            rows.append(
                {"root_id": rid, "status": "PARTIAL", "cation": cat, "neutral": neu}
            )
            n_partial += 1
        else:
            rows.append(
                {"root_id": rid, "status": "MISS", "cation": False, "neutral": False}
            )
            n_miss += 1
    n = len(VALIDATION_ROOTS)
    terminal = n_pass + n_fail
    return {
        "batch_id": batch_id,
        "n_val": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_partial": n_partial,
        "n_miss": n_miss,
        "all_pass": n_pass == n,
        "all_terminal": terminal == n and n_partial == 0 and n_miss == 0,
        "roots": rows,
    }


def archive_if_present(src: Path, dest: Path, log_path: Path) -> Path | None:
    """Move ``src`` → ``dest`` when present; never delete. Returns dest or None."""

    if not src.exists():
        return None
    if dest.exists():
        raise RuntimeError(f"archive destination already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    _log(log_path, f"archived {src} → {dest}")
    return dest


def write_train_merge_meta(
    *,
    gen: Path,
    train_dir: Path,
    archive_path: Path | None,
    train_ready: list[str],
    val_ready: list[str],
    product_root: dict[str, str],
    log_path: Path,
) -> Path:
    """Write train_g00N/merge_meta.json (G2 / COMPUTE_DISPATCH §1.4.4)."""

    # Only groups that actually supply Train∪Val labels (§1.4.4 merged_from_groups).
    groups: set[str] = set()
    for rid in list(train_ready) + list(val_ready):
        rel = product_root.get(rid)
        if not rel:
            continue
        p = Path(rel)
        for part in p.parts:
            if part.startswith("teacher_gpu_"):
                groups.add(part)
                break
        else:
            if p.parent.name.startswith("teacher_gpu_"):
                groups.add(p.parent.name)
    split_doc = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "splits"
        / "nhc0801_g001_tvt_resplit_v001_development.json"
    )
    if not split_doc.is_file():
        # server layout: repo root is nhc0801_root
        alt = Path(DEFAULT_NHC0801) / "data" / "splits" / split_doc.name
        if alt.is_file():
            split_doc = alt
    meta = build_merge_meta(
        merge_group_id="g001",
        archive_path=str(archive_path) if archive_path else "",
        merged_from_groups=sorted(groups),
        train_roots=train_ready,
        val_roots=val_ready,
        tvt_split_document=split_doc if split_doc.is_file() else None,
        notes=[
            "validation pool rebuilt for TVT resplit v001; "
            "quick-val curves vs archived train_g001 are not comparable",
        ],
    )
    path = write_merge_meta(train_dir, meta)
    _log(log_path, f"wrote merge_meta {path}")
    return path


def ready_split_roots(gen: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (train_ready, val_ready, root_id -> teacher product dir)."""

    scan = scan_teacher_products(gen)
    product_root: dict[str, str] = {}
    for rid, info in scan.items():
        if not (info["cation"]["done_ok"] and info["neutral"]["done_ok"]):
            continue
        cdirs = info["cation"].get("product_dirs") or []
        ndirs = info["neutral"].get("product_dirs") or []
        # parents that have both endpoints (same teacher_gpu batch)
        c_parents = {
            str(Path(*Path(p).parts[:2]))
            for p in cdirs
            if len(Path(p).parts) >= 2
        }
        n_parents = {
            str(Path(*Path(p).parts[:2]))
            for p in ndirs
            if len(Path(p).parts) >= 2
        }
        both = sorted(c_parents & n_parents)
        if not both:
            continue
        # prefer higher batch name (more recent g0xx) then max total frames
        best = None
        best_key: tuple[int, str] = (-1, "")
        for rel_parent in both:
            parent = gen / rel_parent
            n = 0
            for ep in ("cation", "neutral"):
                epd = parent / ep
                if epd.is_dir():
                    n += sum(1 for _ in epd.glob("frame_*.json"))
            key = (n, rel_parent)
            if key > best_key:
                best_key = key
                best = str(parent)
        if best:
            product_root[rid] = best
    train = [r for r in TRAIN_ROOTS if r in product_root]
    val = [r for r in VALIDATION_ROOTS if r in product_root]
    return train, val, product_root


def build_teacher_view(
    gen: Path, product_root: dict[str, str], view_name: str = "teacher_view_tvt_v001"
) -> Path:
    """Symlink each ready root into a flat teacher_dir consumable by D3/weighted."""

    view = gen / view_name
    if view.exists():
        shutil.rmtree(view)
    view.mkdir(parents=True, exist_ok=True)
    for rid, src in sorted(product_root.items()):
        src_p = Path(src)
        if not src_p.is_dir():
            continue
        dst = view / rid
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src_p.resolve())
    return view


def pick_train_gpu(exclude: set[int] | None = None) -> int:
    """Pick a low-mem non-VASP GPU for AIMNet2 train (does not kill anything)."""

    from nhc_deprot.resources.gpu_inventory import inventory_gpus, pick_gpus

    try:
        gpus = pick_gpus(
            1,
            max_gpu=8,
            exclude=list(exclude or ()),
            allow_shared=True,
            require_free=False,
        )
        return int(gpus[0])
    except Exception:
        slots = inventory_gpus(8)
        # fallback: lowest used, no vasp
        cands = [s for s in slots if not s.has_vasp]
        if not cands:
            cands = list(slots)
        cands.sort(key=lambda s: (s.used_mib, s.process_count))
        return int(cands[0].index)


def run_pipeline(
    *,
    nhc0801_root: Path,
    generation_id: str,
    e0_batch_id: str,
    train_batch_id: str,
    poll_seconds: int,
    require_all_pass: bool,
    min_train_roots: int,
    base_weight: Path,
    skip_train: bool,
    once: bool,
) -> int:
    gen = nhc0801_root / "runs" / generation_id
    log_dir = gen / "logs" / "val_e0_then_train"
    log_path = log_dir / "watch.log"
    status_path = log_dir / "status.json"
    log_dir.mkdir(parents=True, exist_ok=True)

    phase = "poll_val_e0"
    _log(log_path, f"start generation={generation_id} e0_batch={e0_batch_id}")

    # --- phase 1: wait Val e0 ---
    while True:
        vs = val_e0_status(gen, e0_batch_id)
        status = {
            "schema": "nhc0801-val-e0-then-train-v1",
            "updated_at_utc": _utc(),
            "phase": phase,
            "val_e0": vs,
            "generation_id": generation_id,
            "e0_batch_id": e0_batch_id,
            "train_batch_id": train_batch_id,
        }
        _write(status_path, status)
        _log(
            log_path,
            f"val_e0 pass={vs['n_pass']}/{vs['n_val']} fail={vs['n_fail']} "
            f"partial={vs['n_partial']} miss={vs['n_miss']}",
        )
        if vs["all_pass"]:
            break
        if (not require_all_pass) and vs["all_terminal"]:
            _log(log_path, "all terminal (allow non-PASS); proceeding")
            break
        if once:
            _log(log_path, "once=True and not ready; exit 0 without train")
            return 0
        time.sleep(max(30, int(poll_seconds)))

    # --- phase 2: rebuild e0 campaign ---
    phase = "rebuild_e0"
    layout = resolve_layout(generation_id=generation_id, nhc0801_root=nhc0801_root)
    ensure_generation_tree(layout, exist_ok=True)
    try:
        rebuilt = rebuild_epoch0_campaign_from_root_receipts(
            layout=layout, batch_id=e0_batch_id
        )
        _log(log_path, f"rebuild_e0 status={rebuilt.get('status')} roots={rebuilt.get('root_count')}")
    except Exception as exc:  # noqa: BLE001
        _log(log_path, f"rebuild_e0 error (continue): {exc}")
        rebuilt = {"status": "FAILED", "error": str(exc)}

    # --- phase 3: teacher view + ready roots ---
    phase = "teacher_view"
    train_ready, val_ready, product_root = ready_split_roots(gen)
    _log(
        log_path,
        f"teacher ready train={len(train_ready)}/{len(TRAIN_ROOTS)} "
        f"val={len(val_ready)}/{len(VALIDATION_ROOTS)}",
    )
    if len(train_ready) < min_train_roots:
        _log(
            log_path,
            f"FAILED: train_ready {len(train_ready)} < min_train_roots {min_train_roots}",
        )
        _write(
            status_path,
            {
                "schema": "nhc0801-val-e0-then-train-v1",
                "updated_at_utc": _utc(),
                "phase": "FAILED_TRAIN_READY",
                "train_ready": len(train_ready),
                "min_train_roots": min_train_roots,
                "val_e0": val_e0_status(gen, e0_batch_id),
            },
        )
        print("FAILED", flush=True)
        return 3

    # Val for NPZ: only ready pairs (should be 16)
    if len(val_ready) < 2:
        _log(log_path, f"FAILED: val_ready={len(val_ready)}")
        print("FAILED", flush=True)
        return 3

    view = build_teacher_view(gen, product_root)
    work_layout = replace(layout, teacher_dir=view)
    _log(log_path, f"teacher_view={view} n_links={len(list(view.iterdir()))}")

    # --- phase 4: D3 + weighted ---
    phase = "d3_weighted"
    d3_roots = list(dict.fromkeys(train_ready + val_ready))
    _write(
        status_path,
        {
            "schema": "nhc0801-val-e0-then-train-v1",
            "updated_at_utc": _utc(),
            "phase": phase,
            "train_ready": len(train_ready),
            "val_ready": len(val_ready),
            "d3_root_count": len(d3_roots),
            "rebuild_e0": {
                "status": rebuilt.get("status"),
                "root_count": rebuilt.get("root_count"),
            },
        },
    )
    utc_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        # G4: archive old d3/ before overwrite (do not lose 2026-08 pilot evidence)
        d3_src = work_layout.d3_dir
        if d3_src.is_dir() and any(d3_src.iterdir()):
            d3_arch = gen / f"_archive_d3_pre_merge_{utc_tag}"
            archive_if_present(d3_src, d3_arch, log_path)
            work_layout.d3_dir.mkdir(parents=True, exist_ok=True)

        d3 = run_d3_campaign(
            layout=work_layout,
            root_ids=d3_roots,
            projector=Dftd3Projector(),
            dry_run=False,
            overwrite=True,
        )
        _log(
            log_path,
            f"d3 status={d3.get('status')} frames={d3.get('frame_count')}",
        )
        # archive previous weighted if present (datasets/weighted_m250_path_a untouched)
        ds = work_layout.datasets_dir
        if ds.is_dir() and (ds / "manifest.json").is_file():
            arch = gen / "datasets" / f"_archive_weighted_{utc_tag}"
            if not arch.exists():
                shutil.move(str(ds), str(arch))
                _log(log_path, f"archived old weighted → {arch}")
        work_layout.datasets_dir.mkdir(parents=True, exist_ok=True)
        w = assemble_weighted_dataset(
            layout=work_layout,
            train_roots=train_ready,
            validation_roots=val_ready,
            dry_run=False,
            overwrite=True,
            run_audit=True,
        )
        _log(
            log_path,
            f"weighted status={w.get('status')} frames={w.get('frame_count_by_split')}",
        )
    except Exception as exc:  # noqa: BLE001
        _log(log_path, f"d3_weighted FAILED: {exc}\n{traceback.format_exc()[-2000:]}")
        _write(
            status_path,
            {
                "schema": "nhc0801-val-e0-then-train-v1",
                "updated_at_utc": _utc(),
                "phase": "FAILED_D3_WEIGHTED",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        print("FAILED", flush=True)
        return 4

    # G3: archive existing train_g001/ then re-enable; G2: write merge_meta
    phase = "archive_train_merge_meta"
    train_dir = work_layout.train_batch_dir(train_batch_id)
    train_archive: Path | None = None
    if train_dir.exists() and any(train_dir.iterdir()):
        # Name must NOT match teacher_gpu* or bare train_g001 (scan safety §1.4.1)
        train_archive = gen / f"_archive_g001_pre_merge_{utc_tag}"
        if train_archive.exists():
            train_archive = gen / f"_archive_g001_pre_merge_{utc_tag}_b"
        archive_if_present(train_dir, train_archive, log_path)
        # README in archive
        readme = train_archive / "README.md"
        readme.write_text(
            "\n".join(
                [
                    "# Archived pre-merge train_g001",
                    "",
                    f"- archived_at_utc: {_utc()}",
                    f"- from: {train_dir}",
                    f"- reason: TVT resplit v001 Train150/Val16 merge; "
                    "new train_g001 succeeds this name after archive "
                    "(COMPUTE_DISPATCH_V001 §1.4.1 / §11.1).",
                    "- contents: prior pilot/ablation checkpoints (≈144 .pt historically).",
                    "- **Not comparable** to new training: validation pool changed "
                    "(quick-val evaluation set rebuilt).",
                    "- Do not use as live train labels; allow_legacy_teacher=false.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    train_dir.mkdir(parents=True, exist_ok=True)
    merge_path = write_train_merge_meta(
        gen=gen,
        train_dir=train_dir,
        archive_path=train_archive,
        train_ready=train_ready,
        val_ready=val_ready,
        product_root=product_root,
        log_path=log_path,
    )

    if skip_train:
        _log(log_path, "skip_train=True; DONE after dataset + merge_meta")
        _write(
            status_path,
            {
                "schema": "nhc0801-val-e0-then-train-v1",
                "updated_at_utc": _utc(),
                "phase": "DONE_DATASET_ONLY",
                "train_ready": len(train_ready),
                "val_ready": len(val_ready),
                "merge_meta": str(merge_path),
                "train_archive": str(train_archive) if train_archive else None,
                "weighted": {
                    "status": w.get("status"),
                    "frame_count_by_split": w.get("frame_count_by_split"),
                },
                "d3": {
                    "status": d3.get("status"),
                    "frame_count": d3.get("frame_count"),
                },
            },
        )
        print("DONE", flush=True)
        return 0

    # --- phase 5: live train ablation (GPU 7 exclusive per §11.1) ---
    phase = "train_ablation"
    gpu = 7
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _log(log_path, f"train on CUDA_VISIBLE_DEVICES={gpu} (exclusive claim)")
    _write(
        status_path,
        {
            "schema": "nhc0801-val-e0-then-train-v1",
            "updated_at_utc": _utc(),
            "phase": phase,
            "gpu_index": gpu,
            "train_ready": len(train_ready),
            "val_ready": len(val_ready),
            "run_ids": list(DEFAULT_ABLATION_RUN_IDS),
            "train_batch_id": train_batch_id,
            "merge_meta": str(merge_path),
        },
    )
    try:
        # LiveAimnet2 uses layout.datasets_dir — same as weighted output
        camp = run_train_ablation(
            layout=work_layout,
            run_ids=list(DEFAULT_ABLATION_RUN_IDS),
            train_batch_id=train_batch_id,
            dry_run=False,
            aimnet2_train_authorized=True,
            base_weight=base_weight,
            device="cuda",
            base_config=TrainingConfig(),
        )
        _log(
            log_path,
            f"train status={camp.get('status')} failed_runs={camp.get('failed_run_count')}",
        )
        _write(
            status_path,
            {
                "schema": "nhc0801-val-e0-then-train-v1",
                "updated_at_utc": _utc(),
                "phase": "DONE",
                "gpu_index": gpu,
                "train_ready": len(train_ready),
                "val_ready": len(val_ready),
                "train_ablation": camp,
                "val_e0": val_e0_status(gen, e0_batch_id),
                "merge_meta": str(merge_path),
            },
        )
        print("DONE", flush=True)
        return 0 if str(camp.get("status", "")).endswith("PASS") else 1
    except Exception as exc:  # noqa: BLE001
        _log(log_path, f"train FAILED: {exc}\n{traceback.format_exc()[-2000:]}")
        _write(
            status_path,
            {
                "schema": "nhc0801-val-e0-then-train-v1",
                "updated_at_utc": _utc(),
                "phase": "FAILED_TRAIN",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        print("FAILED", flush=True)
        return 5


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    p.add_argument("--e0-batch-id", default="g001")
    p.add_argument(
        "--train-batch-id",
        default="g001",
        help="train product dir train_<id>/ (default g001 — succeed after archive)",
    )
    p.add_argument("--poll-seconds", type=int, default=300)
    p.add_argument(
        "--require-all-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait until all Val roots PASS (default true)",
    )
    p.add_argument(
        "--min-train-roots",
        type=int,
        default=150,
        help="Minimum teacher-ready Train roots before NPZ/train (default 150)",
    )
    p.add_argument(
        "--base-weight",
        type=Path,
        default=Path("/home/plab/.cache/aimnet/aimnet2_wb97m_d3_0.pt"),
    )
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--once", action="store_true", help="Single poll; no train if not ready")
    args = p.parse_args(argv)
    return run_pipeline(
        nhc0801_root=args.nhc0801_root,
        generation_id=args.generation_id,
        e0_batch_id=args.e0_batch_id,
        train_batch_id=args.train_batch_id,
        poll_seconds=args.poll_seconds,
        require_all_pass=bool(args.require_all_pass),
        min_train_roots=int(args.min_train_roots),
        base_weight=args.base_weight,
        skip_train=bool(args.skip_train),
        once=bool(args.once),
    )


if __name__ == "__main__":
    raise SystemExit(main())
