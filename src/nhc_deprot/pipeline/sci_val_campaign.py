"""Mindmap steps 8–9 — scientific Validation campaign over shortlisted checkpoints.

Default is **dry-run** with Simulated* engines. Live chemistry requires
``scientific_validation_live=True`` plus real engines (not Simulated*).

Selection uses frozen NUMERIC_CALIBRATION_V001 via select_after_scientific_validation.
Does **not** open Final Test. Does **not** authorize post-Test reselection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from nhc_deprot.contracts.tvt_gates import validate_numeric_addendum
from nhc_deprot.data.io_util import load_json_object, write_json
from nhc_deprot.data.paths import VALIDATION_ROOTS
from nhc_deprot.generation.layout import GenerationLayout
from nhc_deprot.pipeline.scientific_validation import (
    Aimnet2RouteEngine,
    CheckpointScientificValidation,
    FrozenEndpointGeometry,
    ParentRouteEngine,
    PureReferenceLabel,
    SimulatedAimnet2Engine,
    SimulatedParentEngine,
    run_scientific_validation_for_checkpoint,
    select_after_scientific_validation,
)
from nhc_deprot.pipeline.training_blockers import load_numeric_calibration

SCI_VAL_CAMPAIGN_SCHEMA: Final = "nhc0801-sci-val-campaign-v1"
MINDMAP_STEPS: Final = (8, 9)


class SciValCampaignError(RuntimeError):
    """Scientific Validation campaign failed closed."""


def load_shortlist_candidates(path: Path) -> list[dict[str, Any]]:
    payload, _raw = load_json_object(path)
    cands = payload.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise SciValCampaignError(f"shortlist has no candidates: {path}")
    out: list[dict[str, Any]] = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        if type(c.get("seed")) is not int or type(c.get("epoch")) is not int:
            raise SciValCampaignError(f"invalid candidate: {c}")
        out.append(c)
    if not out:
        raise SciValCampaignError("no valid shortlist candidates")
    return out


def _default_geometries_and_refs(
    roots: Sequence[str],
    *,
    e_cation: float = -100.0,
    e_neutral: float = -99.5,
) -> tuple[list[FrozenEndpointGeometry], dict[str, PureReferenceLabel]]:
    """Synthetic geometries for dry-run only (not production XYZ)."""

    from nhc_deprot.pipeline.epoch0_runner import (
        build_pure_references,
        build_validation_geometries,
    )

    geos = list(build_validation_geometries(list(roots)))
    refs = dict(
        build_pure_references(list(roots), e_cation=e_cation, e_neutral=e_neutral)
    )
    return geos, refs


def run_sci_val_campaign(
    *,
    layout: GenerationLayout,
    shortlist_path: Path | None = None,
    dry_run: bool = True,
    scientific_validation_live: bool = False,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    geometries: Sequence[FrozenEndpointGeometry] | None = None,
    references: Mapping[str, PureReferenceLabel] | None = None,
    aimnet2_factory: Any | None = None,
    parent: ParentRouteEngine | None = None,
    epoch0_baseline: CheckpointScientificValidation | None = None,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    """Run full sci-val route for each shortlisted (seed, epoch); then select."""

    if not dry_run and not scientific_validation_live:
        raise SciValCampaignError(
            "live sci-val requires scientific_validation_live=true (and real engines)"
        )

    if candidates is None:
        sl = shortlist_path or (layout.sci_val_dir / "shortlist_campaign.json")
        cand_list = load_shortlist_candidates(sl)
    else:
        cand_list = [dict(c) for c in candidates]

    if max_candidates is not None:
        cand_list = cand_list[: int(max_candidates)]

    roots = list(VALIDATION_ROOTS)
    # Match Epoch0Config synthetic energies so dry-run label errors ~0
    e_cat, e_neu = -100.0, -99.5
    if geometries is None or references is None:
        geos, refs = _default_geometries_and_refs(roots, e_cation=e_cat, e_neutral=e_neu)
    else:
        geos, refs = list(geometries), dict(references)

    if dry_run:
        parent_eng: ParentRouteEngine = parent or SimulatedParentEngine(
            energy_cation=e_cat,
            energy_neutral=e_neu,
            opt_steps=18,
            scf_cycles=60,
            handoff_pass=True,
        )
        # epoch-0 baseline with slightly worse parent work if not provided
        if epoch0_baseline is None:
            e0_aim = SimulatedAimnet2Engine(converge=True, steps=12)
            epoch0_baseline = run_scientific_validation_for_checkpoint(
                epoch=0,
                checkpoint_id="epoch-0-official",
                checkpoint_sha256="f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28",
                route_kind="epoch_zero",
                geometries=geos,
                references=refs,
                aimnet2=e0_aim,
                parent=SimulatedParentEngine(
                    energy_cation=e_cat,
                    energy_neutral=e_neu,
                    opt_steps=40,
                    scf_cycles=120,
                    handoff_pass=True,
                ),
                scientific_validation_live=False,
                live=False,
            )
    else:
        if parent is None or aimnet2_factory is None:
            raise SciValCampaignError("live sci-val requires parent + aimnet2_factory")
        parent_eng = parent

    results: list[dict[str, Any]] = []
    val_objects: list[CheckpointScientificValidation] = []

    for cand in cand_list:
        seed = int(cand["seed"])
        epoch = int(cand["epoch"])
        ck_id = f"seed_{seed}_epoch_{epoch:04d}"
        # placeholder sha from path or synthetic
        digest = "a" * 64
        if cand.get("weight_path"):
            wp = Path(str(cand["weight_path"]))
            if wp.is_file():
                import hashlib

                h = hashlib.sha256()
                with wp.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                digest = h.hexdigest()
        else:
            # deterministic pseudo-sha for dry-run identity
            import hashlib

            digest = hashlib.sha256(ck_id.encode()).hexdigest()

        if dry_run:
            # slightly better parent work than epoch-0 synthetic for shortlisted
            aim: Aimnet2RouteEngine = SimulatedAimnet2Engine(converge=True, steps=8)
            par: ParentRouteEngine = SimulatedParentEngine(
                energy_cation=e_cat,
                energy_neutral=e_neu,
                opt_steps=12,
                scf_cycles=40,
                handoff_pass=True,
            )
        else:
            aim = aimnet2_factory(cand)
            par = parent_eng

        agg = run_scientific_validation_for_checkpoint(
            epoch=epoch,
            checkpoint_id=ck_id,
            checkpoint_sha256=digest,
            route_kind="finetuned_checkpoint",
            geometries=geos,
            references=refs,
            aimnet2=aim,
            parent=par,
            epoch0_baseline=epoch0_baseline,
            scientific_validation_live=scientific_validation_live and not dry_run,
            live=scientific_validation_live and not dry_run,
        )
        val_objects.append(agg)
        payload = agg.as_dict()
        payload["seed"] = seed
        payload["shortlist_meta"] = {
            "validation_weighted_loss": cand.get("validation_weighted_loss"),
            "weight_present": cand.get("weight_present"),
            "weight_path": cand.get("weight_path"),
        }
        results.append(payload)

        out_dir = layout.sci_val_dir / f"seed_{seed}" / f"epoch_{epoch:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "sci_val_receipt.json", payload, overwrite=True)

    addendum = load_numeric_calibration()
    validate_numeric_addendum(addendum)
    selection = select_after_scientific_validation(val_objects, numeric_addendum=addendum)

    campaign = {
        "schema": SCI_VAL_CAMPAIGN_SCHEMA,
        "mindmap_steps": list(MINDMAP_STEPS),
        "generation_id": layout.generation_id,
        "dry_run": dry_run,
        "scientific_validation_live": bool(scientific_validation_live and not dry_run),
        "status": (
            "DRY_RUN_SCI_VAL_PASS"
            if dry_run
            else (
                "LIVE_SCI_VAL_PASS"
                if selection.get("outcome") == "VALIDATION_SELECTED"
                else "LIVE_SCI_VAL_REJECTED"
            )
        ),
        "candidate_count": len(results),
        "candidate_results": results,
        "selection": selection,
        "final_model_selected": selection.get("outcome") == "VALIDATION_SELECTED",
        "final_test_authorized": False,
        "final_test_payload_read": False,
        "quick_validation_may_select_final_model": False,
        "numeric_addendum_version": addendum.get("version"),
        "notes": [
            "selection is Validation-only; Final Test remains sealed",
            "dry_run uses Simulated engines unless live gate + real engines",
        ],
    }

    layout.sci_val_dir.mkdir(parents=True, exist_ok=True)
    write_json(layout.sci_val_dir / "campaign_receipt.json", campaign, overwrite=True)
    write_json(layout.logs_dir / "sci_val_campaign_receipt.json", campaign, overwrite=True)
    if selection.get("outcome") == "VALIDATION_SELECTED":
        write_json(
            layout.sci_val_dir / "selection_receipt.json",
            {
                **selection,
                "final_test_authorized": False,
                "mindmap_step": 9,
            },
            overwrite=True,
        )
    return campaign
