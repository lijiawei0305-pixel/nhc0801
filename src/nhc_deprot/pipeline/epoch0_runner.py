"""Mindmap step 3 — Epoch-0 full-route baseline (dry-run skeleton).

Scientific identity:

    Validation frozen geometry
      → official unfined-tuned AIMNet2 (_0 only)
      → AIMNet2 GAU_LOOSE
      → identity / topology / finite gates
      → exact-byte handoff
      → full Parent-Level P01 to final GAU
      → parent final single point
      → deprotonation label
      → compare to Pure-PySCF reference on same roots

Default dry_run=True uses SimulatedAimnet2/Parent engines and writes receipts
under ``layout.epoch0_dir`` (canonical: ``epoch0_val_batches/g001/epoch0/`` for
g001 Epoch-0; other batches use ``epoch0_val_batches/g00N/epoch0/``).
Live chemistry requires ``epoch0_execution=True`` and injected non-dry engines.

Never opens Final Test. Never uses historical finetuned checkpoints as epoch-0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from nhc_deprot.contracts.parent_protocol import (
    CATION_CHARGE,
    CATION_MULTIPLICITY,
    NEUTRAL_CHARGE,
    NEUTRAL_MULTIPLICITY,
    PROTOCOL_SHA256,
    deprotonation_electronic_kcal,
)
from nhc_deprot.data.io_util import write_json
from nhc_deprot.data.paths import (
    OFFICIAL_AIMNET2_WEIGHT_SHA256,
    VALIDATION_ROOTS,
)
from nhc_deprot.generation.layout import GenerationLayout
from nhc_deprot.pipeline.parent_handoff import load_gau_loose_profile
from nhc_deprot.pipeline.scientific_validation import (
    Aimnet2RouteEngine,
    FrozenEndpointGeometry,
    ParentRouteEngine,
    PureReferenceLabel,
    SimulatedAimnet2Engine,
    SimulatedParentEngine,
    run_scientific_validation_for_checkpoint,
)

EPOCH0_CAMPAIGN_SCHEMA: Final = "nhc0801-epoch0-campaign-v1"
EPOCH0_ROOT_SCHEMA: Final = "nhc0801-epoch0-root-receipt-v1"
MINDMAP_STEP: Final = 3
CHECKPOINT_ID: Final = "epoch-0-official-aimnet2_wb97m_d3_0"
ENDPOINTS: Final = ("cation", "neutral")


class Epoch0Error(RuntimeError):
    """Epoch-0 runner failed closed."""


@dataclass
class Epoch0Config:
    official_weight_sha256: str = OFFICIAL_AIMNET2_WEIGHT_SHA256
    checkpoint_id: str = CHECKPOINT_ID
    validation_roots: tuple[str, ...] = VALIDATION_ROOTS
    # Dry-run synthetic parent energies
    # (must match SimulatedParentEngine defaults unless overridden)
    ref_energy_cation: float = -100.0
    ref_energy_neutral: float = -99.5
    epoch0_parent_opt_steps: int = 18  # slightly better than a worse baseline for demo metrics
    pure_parent_opt_steps: int = 40


def _stub_geometry(root_id: str, endpoint: str, n_atoms: int = 3) -> FrozenEndpointGeometry:
    elements = tuple(["C"] * n_atoms)
    coords = tuple((float(i), 0.1 if endpoint == "cation" else 0.0, 0.0) for i in range(n_atoms))
    charge = CATION_CHARGE if endpoint == "cation" else NEUTRAL_CHARGE
    mult = CATION_MULTIPLICITY if endpoint == "cation" else NEUTRAL_MULTIPLICITY
    return FrozenEndpointGeometry(
        root_id=root_id,
        endpoint=endpoint,
        elements=elements,
        coordinates=coords,
        charge=charge,
        multiplicity=mult,
        geometry_sha256="dry" + "0" * 61,
    )


def build_validation_geometries(
    root_ids: Sequence[str],
) -> list[FrozenEndpointGeometry]:
    geos: list[FrozenEndpointGeometry] = []
    for root_id in root_ids:
        geos.append(_stub_geometry(root_id, "cation", n_atoms=3))
        geos.append(_stub_geometry(root_id, "neutral", n_atoms=2))
    return geos


def build_pure_references(
    root_ids: Sequence[str],
    *,
    e_cation: float,
    e_neutral: float,
) -> dict[str, PureReferenceLabel]:
    label = deprotonation_electronic_kcal(e_neutral, e_cation)
    return {
        root_id: PureReferenceLabel(
            root_id=root_id,
            e_cation_hartree=e_cation,
            e_neutral_hartree=e_neutral,
            label_kcal=label,
            protocol_sha256=PROTOCOL_SHA256,
        )
        for root_id in root_ids
    }


def run_epoch0_campaign(
    *,
    layout: GenerationLayout,
    config: Epoch0Config | None = None,
    dry_run: bool = True,
    epoch0_execution: bool = False,
    aimnet2: Aimnet2RouteEngine | None = None,
    parent: ParentRouteEngine | None = None,
    pure_parent: ParentRouteEngine | None = None,
    geometries: Sequence[FrozenEndpointGeometry] | None = None,
    references: Mapping[str, PureReferenceLabel] | None = None,
) -> dict[str, Any]:
    """Run epoch-0 baseline on Validation roots and write g001/epoch0 receipts."""

    cfg = config or Epoch0Config()
    roots = list(cfg.validation_roots)
    if not roots:
        raise Epoch0Error("validation root list is empty")

    if not dry_run:
        if not epoch0_execution:
            raise Epoch0Error("live epoch-0 requires epoch0_execution=true")
        if aimnet2 is None or parent is None:
            raise Epoch0Error("live epoch-0 requires injected AIMNet2 and Parent engines")
        if isinstance(aimnet2, SimulatedAimnet2Engine) or isinstance(parent, SimulatedParentEngine):
            raise Epoch0Error("live epoch-0 refuses Simulated* engines")
    else:
        # Dry-run engines
        aimnet2 = aimnet2 or SimulatedAimnet2Engine(converge=True, steps=10)
        parent = parent or SimulatedParentEngine(
            energy_cation=cfg.ref_energy_cation,
            energy_neutral=cfg.ref_energy_neutral,
            opt_steps=cfg.epoch0_parent_opt_steps,
            scf_cycles=60,
            handoff_pass=True,
        )
        pure_parent = pure_parent or SimulatedParentEngine(
            energy_cation=cfg.ref_energy_cation,
            energy_neutral=cfg.ref_energy_neutral,
            opt_steps=cfg.pure_parent_opt_steps,
            scf_cycles=120,
            handoff_pass=True,
        )

    gau = load_gau_loose_profile()
    geos = list(geometries or build_validation_geometries(roots))
    refs = dict(
        references
        or build_pure_references(
            roots,
            e_cation=cfg.ref_energy_cation,
            e_neutral=cfg.ref_energy_neutral,
        )
    )

    # A) Pure-PySCF reference route (no AIMNet2)
    pure_agg = run_scientific_validation_for_checkpoint(
        epoch=0,
        checkpoint_id="pure-pyscf-reference",
        checkpoint_sha256="0" * 64,
        route_kind="pure_pyscf_reference",
        geometries=geos,
        references=refs,
        aimnet2=None,
        parent=pure_parent or parent,
        profile=gau,
        scientific_validation_live=False,
        live=False,
    )

    # B) Epoch-0 official AIMNet2 → full parent
    epoch0_agg = run_scientific_validation_for_checkpoint(
        epoch=0,
        checkpoint_id=cfg.checkpoint_id,
        checkpoint_sha256=cfg.official_weight_sha256,
        route_kind="epoch_zero",
        geometries=geos,
        references=refs,
        aimnet2=aimnet2,
        parent=parent,
        profile=gau,
        scientific_validation_live=False,
        live=False,
    )

    layout.epoch0_dir.mkdir(parents=True, exist_ok=True)
    root_receipts: list[dict[str, Any]] = []

    pure_by_root = {r.root_id: r for r in pure_agg.root_receipts}
    e0_by_root = {r.root_id: r for r in epoch0_agg.root_receipts}

    for root_id in roots:
        pure_r = pure_by_root.get(root_id)
        e0_r = e0_by_root.get(root_id)
        if pure_r is None or e0_r is None:
            raise Epoch0Error(f"missing route receipt for root {root_id}")

        pure_label = pure_r.label_kcal
        e0_label = e0_r.label_kcal
        abs_err = None
        signed_err = None
        if pure_label is not None and e0_label is not None:
            signed_err = e0_label - pure_label
            abs_err = abs(signed_err)

        # Burden proxy: parent opt steps (cation+neutral)
        def _steps(receipt) -> int:
            total = 0
            for ep in (receipt.cation, receipt.neutral):
                if ep is not None:
                    total += int(ep.parent_opt_steps)
            return total

        pure_steps = _steps(pure_r)
        e0_steps = _steps(e0_r)
        step_reduction = None
        if pure_steps > 0:
            step_reduction = (pure_steps - e0_steps) / pure_steps

        root_payload = {
            "schema": EPOCH0_ROOT_SCHEMA,
            "mindmap_step": MINDMAP_STEP,
            "root_id": root_id,
            "dry_run": dry_run,
            "live_chemistry": not dry_run,
            "official_weight_sha256": cfg.official_weight_sha256,
            "checkpoint_id": cfg.checkpoint_id,
            "parent_protocol_sha256": PROTOCOL_SHA256,
            "single_point_only": False,
            "aimnet2_energy_enters_label": False,
            "pure_pyscf_reference": pure_r.as_dict(),
            "epoch0_route": e0_r.as_dict(),
            "comparison": {
                "pure_label_kcal": pure_label,
                "epoch0_label_kcal": e0_label,
                "absolute_label_error_kcal": abs_err,
                "signed_label_error_kcal": signed_err,
                "pure_parent_opt_steps": pure_steps,
                "epoch0_parent_opt_steps": e0_steps,
                "parent_opt_step_reduction_fraction": step_reduction,
            },
            "status": (
                "PASS"
                if (
                    not pure_r.catastrophic_failure
                    and not e0_r.catastrophic_failure
                    and pure_r.all_identity_and_structure_hard_gates
                    and e0_r.all_identity_and_structure_hard_gates
                )
                else "FAILED"
            ),
        }
        out = layout.epoch0_dir / root_id / "epoch0_root_receipt.json"
        write_json(out, root_payload, overwrite=True)
        root_receipts.append(root_payload)

    failed = sum(1 for r in root_receipts if r["status"] != "PASS")
    campaign = {
        "schema": EPOCH0_CAMPAIGN_SCHEMA,
        "mindmap_step": MINDMAP_STEP,
        "generation_id": layout.generation_id,
        "dry_run": dry_run,
        "live_chemistry": not dry_run,
        "epoch0_execution": epoch0_execution,
        "official_weight_sha256": cfg.official_weight_sha256,
        "checkpoint_id": cfg.checkpoint_id,
        "parent_protocol_sha256": PROTOCOL_SHA256,
        "validation_roots": roots,
        "root_count": len(roots),
        "failed_root_count": failed,
        "status": (
            "DRY_RUN_EPOCH0_PASS"
            if dry_run and failed == 0
            else (
                "DRY_RUN_EPOCH0_PARTIAL"
                if dry_run
                else ("LIVE_EPOCH0_PASS" if failed == 0 else "LIVE_EPOCH0_PARTIAL")
            )
        ),
        "pure_aggregate": pure_agg.as_dict(),
        "epoch0_aggregate": epoch0_agg.as_dict(),
        "root_receipts": root_receipts,
        "baseline_metrics": {
            "mean_absolute_label_error_kcal_mol": epoch0_agg.mean_absolute_label_error_kcal_mol,
            "maximum_absolute_label_error_kcal_mol": (
                epoch0_agg.maximum_absolute_label_error_kcal_mol
            ),
            "catastrophic_failure_count": epoch0_agg.catastrophic_failure_count,
            "pyscf_geometry_work_reduction_fraction": (
                epoch0_agg.pyscf_geometry_work_reduction_fraction
            ),
        },
        "notes": [
            "epoch-0 uses official _0 weight identity only",
            "GAU_LOOSE → exact-byte handoff → full parent GAU required",
            "AIMNet2 energy never enters labels",
            "Final Test not accessed",
            "dry_run synthetic engines are not scientific baselines",
        ],
        "final_test_payload_read": False,
        "training_started": False,
    }
    write_json(layout.epoch0_dir / "campaign_receipt.json", campaign, overwrite=True)
    write_json(layout.logs_dir / "epoch0_campaign_receipt.json", campaign, overwrite=True)
    return campaign


def plan_epoch0_paths(
    layout: GenerationLayout, *, validation_roots: Sequence[str] | None = None
) -> dict[str, Any]:
    roots = list(validation_roots or VALIDATION_ROOTS)
    return {
        "mindmap_step": MINDMAP_STEP,
        "generation_id": layout.generation_id,
        "epoch0_dir": str(layout.epoch0_dir),
        "official_weight_sha256": OFFICIAL_AIMNET2_WEIGHT_SHA256,
        "roots": [
            {
                "root_id": r,
                "receipt": str(layout.epoch0_dir / r / "epoch0_root_receipt.json"),
            }
            for r in roots
        ],
        "campaign_receipt": str(layout.epoch0_dir / "campaign_receipt.json"),
    }
