"""Pre-screen CLI + ablation summary table (mindmap 7 / plan M11).

Logic lives here; scripts only parse argv and call :func:`main_pre_screen` /
:func:`main_ablation_table`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from nhc_deprot.data.io_util import load_json_object, write_json
from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation import artifact_names as anames
from nhc_deprot.generation.layout import (
    DEFAULT_GENERATION_ID,
    GenerationLayout,
    ensure_generation_tree,
    init_generation,
    resolve_layout,
)
from nhc_deprot.pipeline.pre_screen import (
    CheckpointCandidate,
    SimulatedPreScreenEngine,
    TeacherEndpointReference,
    load_teacher_references_for_batch,
    run_pre_screen_campaign,
)
from nhc_deprot.training.ablation_cli import (
    DEFAULT_ABLATION_RUN_IDS,
    AblationCliError,
    parse_run_id_list,
)

SCREEN_RECEIPT_NAME: Final = "screen_campaign.json"
TABLE_SCHEMA: Final = "nhc0801-ablation-pre-screen-table-v1"


class PreScreenCliError(RuntimeError):
    """Pre-screen / ablation-table CLI failed closed."""


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def candidates_from_seed_result(
    seed_payload: Mapping[str, Any],
    *,
    run_id: str | None = None,
    shortlist_only: bool = True,
) -> list[CheckpointCandidate]:
    """Build candidates from one ``seed_result.json`` payload."""

    seed = seed_payload.get("seed")
    if type(seed) is not int:
        raise PreScreenCliError("seed_result missing int seed")
    rid = str(run_id or seed_payload.get("run_id") or "unknown_run")
    checkpoints = seed_payload.get("checkpoints") or []
    if not isinstance(checkpoints, list):
        raise PreScreenCliError(f"seed {seed}: checkpoints not a list")

    shortlist_epochs = seed_payload.get("shortlist_epochs")
    epoch_filter: set[int] | None = None
    if shortlist_only and isinstance(shortlist_epochs, list) and shortlist_epochs:
        epoch_filter = {int(e) for e in shortlist_epochs if type(e) is int}

    out: list[CheckpointCandidate] = []
    for ck in checkpoints:
        if not isinstance(ck, Mapping):
            continue
        epoch = ck.get("epoch")
        if type(epoch) is not int:
            continue
        if epoch_filter is not None and epoch not in epoch_filter:
            continue
        weight = ck.get("weight_path")
        ckpt_id = str(ck.get("checkpoint_id") or f"{rid}_seed_{seed}_epoch_{epoch:04d}")
        out.append(
            CheckpointCandidate(
                checkpoint_id=ckpt_id,
                run_id=rid,
                seed=seed,
                epoch=epoch,
                weight_path=str(weight) if weight is not None else None,
            )
        )
    return out


def collect_candidates_from_train_runs(
    layout: GenerationLayout,
    *,
    batch_id: str,
    run_ids: Sequence[str],
    shortlist_only: bool = True,
) -> list[CheckpointCandidate]:
    """Walk ``train_g00N/runs/<run_id>/seed_*/seed_result.json``."""

    found: list[CheckpointCandidate] = []
    for rid in run_ids:
        run_dir = layout.train_batch_run_dir(batch_id, rid)
        if not run_dir.is_dir():
            continue
        for seed_dir in sorted(run_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            receipt = seed_dir / anames.TRAIN_SEED_RESULT_JSON
            if not receipt.is_file():
                # legacy name
                alt = seed_dir / anames.TRAIN_SEED_RECEIPT_LEGACY
                receipt = alt if alt.is_file() else receipt
            if not receipt.is_file():
                continue
            payload, _ = load_json_object(receipt)
            found.extend(
                candidates_from_seed_result(
                    payload, run_id=rid, shortlist_only=shortlist_only
                )
            )
    return found


def candidates_from_json_file(path: Path) -> list[CheckpointCandidate]:
    """Load a JSON list of candidate dicts or a campaign with ``candidates`` key."""

    payload, _ = load_json_object(path)
    raw_list: Any
    if isinstance(payload, list):
        raw_list = payload
    elif isinstance(payload, Mapping):
        raw_list = payload.get("candidates") or payload.get("ranked") or []
    else:
        raise PreScreenCliError(f"unsupported candidates JSON: {path}")
    if not isinstance(raw_list, list) or not raw_list:
        raise PreScreenCliError(f"no candidates in {path}")
    out: list[CheckpointCandidate] = []
    for row in raw_list:
        if not isinstance(row, Mapping):
            continue
        seed = row.get("seed")
        epoch = row.get("epoch")
        if type(seed) is not int or type(epoch) is not int:
            raise PreScreenCliError(f"candidate needs int seed/epoch: {row}")
        rid = str(row.get("run_id") or "unknown_run")
        ckpt = str(
            row.get("checkpoint_id") or f"{rid}_seed_{seed}_epoch_{epoch:04d}"
        )
        weight = row.get("weight_path")
        out.append(
            CheckpointCandidate(
                checkpoint_id=ckpt,
                run_id=rid,
                seed=seed,
                epoch=epoch,
                weight_path=str(weight) if weight is not None else None,
            )
        )
    if not out:
        raise PreScreenCliError(f"parsed zero candidates from {path}")
    return out


def demo_candidates(run_ids: Sequence[str]) -> list[CheckpointCandidate]:
    """Synthetic shortlist-shaped candidates for dry CLI smoke (no train tree)."""

    out: list[CheckpointCandidate] = []
    seed = 20260730
    for rid in run_ids:
        for epoch in (60, 120):
            out.append(
                CheckpointCandidate(
                    checkpoint_id=f"{rid}_seed_{seed}_epoch_{epoch:04d}",
                    run_id=rid,
                    seed=seed,
                    epoch=epoch,
                    weight_path=None,
                )
            )
    return out


def synthetic_teacher_references(
    root_ids: Sequence[str] | None = None,
) -> list[TeacherEndpointReference]:
    """Minimal geometry refs for dry pre-screen without teacher frames on disk."""

    roots = list(root_ids) if root_ids else list(VALIDATION_ROOTS[:1])
    refs: list[TeacherEndpointReference] = []
    coords = (
        (0.0, 0.0, 0.0),
        (1.4, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    forces = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    elements = ("C", "N", "H")
    for root_id in roots:
        for endpoint, charge in (("cation", 1), ("neutral", 0)):
            refs.append(
                TeacherEndpointReference(
                    root_id=root_id,
                    endpoint=endpoint,
                    elements=elements,
                    start_coordinates_angstrom=coords,
                    reference_coordinates_angstrom=coords,
                    reference_forces_ev_per_a=forces,
                    charge=charge,
                    multiplicity=1,
                    start_frame_index=0,
                    reference_frame_index=1,
                )
            )
    return refs


def resolve_references(
    layout: GenerationLayout,
    *,
    batch_id: str,
    root_ids: Sequence[str],
    allow_synthetic: bool,
) -> list[TeacherEndpointReference]:
    """Load teacher refs; optionally fall back to synthetic for dry smoke."""

    try:
        return load_teacher_references_for_batch(layout, batch_id, root_ids)
    except Exception as exc:  # noqa: BLE001
        if not allow_synthetic:
            raise PreScreenCliError(
                f"teacher references unavailable for batch {batch_id}: {exc}"
            ) from exc
        return synthetic_teacher_references(root_ids)


# ---------------------------------------------------------------------------
# Ablation table (run_id × seed × epoch × pre-screen metrics)
# ---------------------------------------------------------------------------


def load_pre_screen_campaigns(
    layout: GenerationLayout,
    *,
    batch_id: str,
    run_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Load ``pre_screen_g00N/**/screen_campaign.json`` (optionally filter by run_id)."""

    root = layout.pre_screen_batch_dir(batch_id)
    if not root.is_dir():
        return []
    campaigns: list[dict[str, Any]] = []
    want = set(run_ids) if run_ids else None
    for path in sorted(root.rglob(SCREEN_RECEIPT_NAME)):
        payload, _ = load_json_object(path)
        if not isinstance(payload, Mapping):
            continue
        sid = str(payload.get("screen_id") or path.parent.name)
        if want is not None and sid not in want:
            # still accept if any ranked row matches
            ranked = payload.get("ranked") or []
            if not any(
                isinstance(r, Mapping) and str(r.get("run_id")) in want
                for r in ranked
            ):
                continue
        campaigns.append(dict(payload))
        campaigns[-1]["_receipt_path"] = str(path)
    return campaigns


def rows_from_pre_screen_campaigns(
    campaigns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten ranked candidates into table rows."""

    rows: list[dict[str, Any]] = []
    for camp in campaigns:
        shortlist_ids = set(camp.get("shortlist_checkpoint_ids") or [])
        ranked = camp.get("ranked") or []
        if not isinstance(ranked, list):
            continue
        for r in ranked:
            if not isinstance(r, Mapping):
                continue
            ckpt = str(r.get("checkpoint_id") or "")
            rows.append(
                {
                    "run_id": str(r.get("run_id") or camp.get("screen_id") or ""),
                    "seed": r.get("seed"),
                    "epoch": r.get("epoch"),
                    "checkpoint_id": ckpt,
                    "hard_gates_passed": bool(r.get("hard_gates_passed")),
                    "mean_rmsd_to_reference_angstrom": r.get(
                        "mean_rmsd_to_reference_angstrom"
                    ),
                    "mean_aimnet2_steps_to_gau_loose": r.get(
                        "mean_aimnet2_steps_to_gau_loose"
                    ),
                    "mean_force_rmse_at_reference_ev_per_a": r.get(
                        "mean_force_rmse_at_reference_ev_per_a"
                    ),
                    "in_shortlist": ckpt in shortlist_ids,
                    "screen_id": camp.get("screen_id"),
                    "energy_loss_used_for_ranking": bool(
                        r.get("energy_loss_used_for_ranking", False)
                    ),
                    "final_model_selected": False,
                }
            )
    # Stable order: run_id, seed, epoch
    rows.sort(
        key=lambda x: (
            str(x.get("run_id") or ""),
            int(x["seed"]) if type(x.get("seed")) is int else 10**12,
            int(x["epoch"]) if type(x.get("epoch")) is int else 10**12,
            str(x.get("checkpoint_id") or ""),
        )
    )
    return rows


def format_ablation_markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render ``run_id × seed × epoch × pre-screen metrics`` as a markdown table.

    Ranking metrics only (AGENTS T1): RMSD / steps / force RMSE — never energy loss.
    """

    headers = [
        "run_id",
        "seed",
        "epoch",
        "hard_gates",
        "mean_rmsd_A",
        "mean_steps",
        "mean_force_rmse",
        "in_shortlist",
    ]
    lines = [
        "# Ablation pre-screen summary",
        "",
        "Authority: mindmap step 7 pre-screen only — **not** final model selection.",
        "Metrics: hard gates → RMSD → steps → force RMSE (energy loss never ranked).",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    if not rows:
        lines.append("| _(empty)_ |  |  |  |  |  |  |  |")
    else:
        for r in rows:
            hard = "yes" if r.get("hard_gates_passed") else "no"
            short = "yes" if r.get("in_shortlist") else "no"
            lines.append(
                "| {run_id} | {seed} | {epoch} | {hard} | {rmsd} | {steps} | "
                "{force} | {short} |".format(
                    run_id=r.get("run_id", ""),
                    seed=r.get("seed", ""),
                    epoch=r.get("epoch", ""),
                    hard=hard,
                    rmsd=_fmt_num(r.get("mean_rmsd_to_reference_angstrom")),
                    steps=_fmt_num(r.get("mean_aimnet2_steps_to_gau_loose")),
                    force=_fmt_num(r.get("mean_force_rmse_at_reference_ev_per_a")),
                    short=short,
                )
            )
    lines.append("")
    lines.append(
        f"_schema: {TABLE_SCHEMA}; final_model_selected: false; "
        "selection_authority: pre_screen_shortlist_only_not_final_"
    )
    lines.append("")
    return "\n".join(lines)


def _fmt_num(v: Any) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "nan"
    if f == float("inf"):
        return "inf"
    if f == float("-inf"):
        return "-inf"
    return f"{f:.6g}"


def build_ablation_table(
    layout: GenerationLayout,
    *,
    batch_id: str = "g001",
    run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Collect pre-screen receipts and produce markdown + row records."""

    campaigns = load_pre_screen_campaigns(
        layout, batch_id=batch_id, run_ids=run_ids
    )
    rows = rows_from_pre_screen_campaigns(campaigns)
    md = format_ablation_markdown_table(rows)
    return {
        "schema": TABLE_SCHEMA,
        "batch_id": batch_id,
        "generation_id": layout.generation_id,
        "campaign_count": len(campaigns),
        "row_count": len(rows),
        "rows": rows,
        "markdown": md,
        "final_model_selected": False,
        "selection_authority": "pre_screen_shortlist_only_not_final",
        "energy_loss_used_for_ranking": False,
    }


# ---------------------------------------------------------------------------
# Pre-screen runner
# ---------------------------------------------------------------------------


def run_pre_screen_cli(
    *,
    layout: GenerationLayout,
    batch_id: str,
    run_ids: Sequence[str],
    candidates: Sequence[CheckpointCandidate],
    root_ids: Sequence[str],
    dry_run: bool = True,
    shortlist_count: int = 3,
    screen_id: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Wire candidates + refs + engine into :func:`run_pre_screen_campaign`."""

    if not candidates:
        raise PreScreenCliError("candidates must be non-empty")
    refs = resolve_references(
        layout,
        batch_id=batch_id,
        root_ids=root_ids,
        allow_synthetic=dry_run,
    )
    engine = SimulatedPreScreenEngine() if dry_run else None
    if engine is None:
        raise PreScreenCliError(
            "live pre-screen requires an injected AIMNet2 GauLooseEngine "
            "(not available in this CLI skeleton; use dry-run or library API)"
        )
    sid = screen_id
    if sid is None:
        sid = run_ids[0] if len(run_ids) == 1 else "campaign"
    return run_pre_screen_campaign(
        candidates=list(candidates),
        references=refs,
        engine=engine,
        layout=layout,
        batch_id=batch_id,
        screen_id=sid,
        shortlist_count=shortlist_count,
        write=write,
    )


# ---------------------------------------------------------------------------
# Argparse / main
# ---------------------------------------------------------------------------


def build_pre_screen_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nhc0801_pre_screen.py",
        description=(
            "P5.5 zero-DFT pre-screen (mindmap step 7). "
            "Ranks by RMSD/steps/force RMSE — never energy loss; never final-selects."
        ),
    )
    p.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    p.add_argument("--nhc0801-root", type=Path, default=None)
    p.add_argument("--batch-id", default="g001")
    p.add_argument(
        "--run-id",
        action="append",
        default=None,
        dest="run_ids",
        help="Filter/discover train runs (default: phase-1 matrix)",
    )
    p.add_argument(
        "--candidates-json",
        type=Path,
        default=None,
        help="Optional JSON list of candidates (else discover train shortlists)",
    )
    p.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="When discovering train seeds, use all checkpoints not only shortlist",
    )
    p.add_argument(
        "--demo-candidates",
        action="store_true",
        help="Use synthetic candidates (dry smoke without train tree)",
    )
    p.add_argument(
        "--roots",
        default=None,
        help="Comma-separated molecular roots for teacher refs (default: Val roots)",
    )
    p.add_argument("--shortlist-count", type=int, default=3)
    p.add_argument(
        "--screen-id",
        default=None,
        help="Output subdir under pre_screen_g00N/ (default: single run_id or campaign)",
    )
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SimulatedPreScreenEngine (default: true)",
    )
    p.add_argument(
        "--write",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write screen_campaign.json under pre_screen_g00N/",
    )
    return p


def main_pre_screen(
    argv: Sequence[str] | None = None,
    *,
    default_nhc0801_root: Path | None = None,
) -> int:
    parser = build_pre_screen_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = args.nhc0801_root or default_nhc0801_root
    if root is None:
        print(json.dumps({"error": "missing --nhc0801-root"}, indent=2))
        return 2

    try:
        run_ids = parse_run_id_list(args.run_ids)
    except AblationCliError as exc:
        # Allow free-form run_ids for pre-screen discovery (not only matrix)
        if args.run_ids:
            parts: list[str] = []
            for v in args.run_ids:
                parts.extend(p.strip() for p in str(v).split(",") if p.strip())
            run_ids = tuple(parts) if parts else DEFAULT_ABLATION_RUN_IDS
        else:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 2

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=root
    )
    if not layout.generation_meta_path().is_file():
        init_generation(generation_id=args.generation_id, nhc0801_root=root)
    else:
        ensure_generation_tree(layout, exist_ok=True)

    if args.roots:
        root_ids = [r.strip() for r in str(args.roots).split(",") if r.strip()]
    else:
        root_ids = list(VALIDATION_ROOTS)

    try:
        if args.candidates_json is not None:
            candidates = candidates_from_json_file(Path(args.candidates_json))
        elif args.demo_candidates:
            candidates = demo_candidates(run_ids)
        else:
            candidates = collect_candidates_from_train_runs(
                layout,
                batch_id=str(args.batch_id),
                run_ids=run_ids,
                shortlist_only=not bool(args.all_checkpoints),
            )
            if not candidates and args.dry_run:
                candidates = demo_candidates(run_ids)

        campaign = run_pre_screen_cli(
            layout=layout,
            batch_id=str(args.batch_id),
            run_ids=run_ids,
            candidates=candidates,
            root_ids=root_ids,
            dry_run=bool(args.dry_run),
            shortlist_count=int(args.shortlist_count),
            screen_id=args.screen_id,
            write=bool(args.write),
        )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "FAIL"},
                indent=2,
            )
        )
        return 1

    summary = {
        "status": campaign.get("status"),
        "batch_id": campaign.get("batch_id"),
        "screen_id": campaign.get("screen_id"),
        "candidate_count": campaign.get("candidate_count"),
        "hard_gates_passed_count": campaign.get("hard_gates_passed_count"),
        "shortlist_checkpoint_ids": campaign.get("shortlist_checkpoint_ids"),
        "final_model_selected": campaign.get("final_model_selected"),
        "selection_authority": campaign.get("selection_authority"),
        "energy_loss_used_for_ranking": campaign.get(
            "energy_loss_used_for_ranking"
        ),
        "receipt_path": campaign.get("receipt_path"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    st = str(campaign.get("status") or "")
    return 0 if st.startswith("PRE_SCREEN") else 1


def build_ablation_table_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nhc0801_ablation_table.py",
        description=(
            "Summarize pre-screen receipts into a markdown table: "
            "run_id × seed × epoch × RMSD/steps/force (never energy loss)."
        ),
    )
    p.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    p.add_argument("--nhc0801-root", type=Path, default=None)
    p.add_argument("--batch-id", default="g001")
    p.add_argument(
        "--run-id",
        action="append",
        default=None,
        dest="run_ids",
        help="Optional filter (repeatable / comma-separated)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write markdown to this path (also prints to stdout)",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional machine-readable rows JSON path",
    )
    return p


def main_ablation_table(
    argv: Sequence[str] | None = None,
    *,
    default_nhc0801_root: Path | None = None,
) -> int:
    parser = build_ablation_table_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = args.nhc0801_root or default_nhc0801_root
    if root is None:
        print(json.dumps({"error": "missing --nhc0801-root"}, indent=2))
        return 2

    run_ids: tuple[str, ...] | None
    if args.run_ids:
        parts: list[str] = []
        for v in args.run_ids:
            parts.extend(p.strip() for p in str(v).split(",") if p.strip())
        run_ids = tuple(parts) if parts else None
    else:
        run_ids = None

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=root
    )
    # Table is read-only; do not force-create tree, but allow empty
    if layout.generation_meta_path().is_file():
        ensure_generation_tree(layout, exist_ok=True)

    table = build_ablation_table(
        layout, batch_id=str(args.batch_id), run_ids=run_ids
    )
    md = str(table["markdown"])
    print(md)

    if args.output is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
    if args.json_out is not None:
        write_json(
            Path(args.json_out),
            {
                "schema": table["schema"],
                "batch_id": table["batch_id"],
                "generation_id": table["generation_id"],
                "row_count": table["row_count"],
                "rows": table["rows"],
                "final_model_selected": False,
            },
            overwrite=True,
        )
    return 0


# Re-export for discoverability (train roots unused but common with val)
__all__ = [
    "TRAIN_ROOTS",
    "build_ablation_table",
    "build_ablation_table_parser",
    "build_pre_screen_parser",
    "collect_candidates_from_train_runs",
    "format_ablation_markdown_table",
    "main_ablation_table",
    "main_pre_screen",
    "run_pre_screen_cli",
]
