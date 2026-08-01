"""Mindmap step 2 — Pure-PySCF teacher frame runner (skeleton).

Scientific order (per molecular root):

    frozen initial geometry (cation, then neutral)
      → full Parent-Level P01 PySCF/geomeTRIC optimization to final GAU
      → save every step: geometry, energy, forces, charge/mult, protocol SHA, lineage
      → write under generation g001 teacher tree

Default mode is **dry_run**: plan paths, write synthetic frame stubs + receipts,
drive ``worker_pool`` assignment. No PySCF/AIMNet2 process is started.

Live chemistry requires ALL of:
  - dry_run=False
  - teacher_pyscf_authorized=True
  - worker_pool.live_dispatch_enabled=True
  - resource claim PASS (and dual receipt if dual profile)
  - an injected live engine (not implemented here)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from nhc_deprot.contracts.forbidden_stacks import assert_parent_protocol_allowed
from nhc_deprot.contracts.parent_protocol import (
    BASIS,
    CATION_CHARGE,
    CATION_MULTIPLICITY,
    FUNCTIONAL,
    NEUTRAL_CHARGE,
    NEUTRAL_MULTIPLICITY,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
)
from nhc_deprot.data.io_util import canonical_json, sha256_bytes
from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation.layout import GenerationLayout
from nhc_deprot.resources.profiles import ResourceProfile, get_profile
from nhc_deprot.resources.worker_pool import (
    assert_ready_for_live_dispatch,
    build_pool,
    claim_next_root,
    complete_root,
    progress_summary,
)

FRAME_SCHEMA: Final = "nhc0801-parent-level-training-frame-v1"
ENDPOINT_MANIFEST_SCHEMA: Final = "nhc0801-teacher-endpoint-manifest-v1"
ROOT_RECEIPT_SCHEMA: Final = "nhc0801-teacher-root-receipt-v1"
CAMPAIGN_SCHEMA: Final = "nhc0801-teacher-campaign-receipt-v1"
ENDPOINTS: Final = ("cation", "neutral")

# Mindmap step identity
MINDMAP_STEP: Final = 2


class TeacherRunnerError(RuntimeError):
    """Teacher runner failed closed."""


class TeacherEngine(Protocol):
    """Optimize one endpoint and yield parent frames (live or dry)."""

    def run_endpoint(
        self,
        *,
        root_id: str,
        endpoint: str,
        charge: int,
        multiplicity: int,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Return frame_count, converged, frame_paths, notes."""
        ...


@dataclass
class EndpointResult:
    root_id: str
    endpoint: str
    frame_count: int
    converged: bool
    frame_paths: list[str] = field(default_factory=list)
    dry_run: bool = True
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RootTeacherReceipt:
    schema: str = ROOT_RECEIPT_SCHEMA
    mindmap_step: int = MINDMAP_STEP
    root_id: str = ""
    parent_protocol_id: str = PROTOCOL_ID
    parent_protocol_sha256: str = PROTOCOL_SHA256
    functional: str = FUNCTIONAL
    basis: str = BASIS
    dry_run: bool = True
    live_chemistry: bool = False
    endpoints: dict[str, dict[str, Any]] = field(default_factory=dict)
    status: str = "PASS"  # PASS | FAILED
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TeacherCampaignReceipt:
    schema: str = CAMPAIGN_SCHEMA
    mindmap_step: int = MINDMAP_STEP
    generation_id: str = ""
    profile_id: str = ""
    dry_run: bool = True
    live_chemistry: bool = False
    teacher_pyscf_authorized: bool = False
    root_order: list[str] = field(default_factory=list)
    pool_progress: dict[str, int] = field(default_factory=dict)
    root_receipts: list[dict[str, Any]] = field(default_factory=list)
    status: str = "DRY_RUN_COMPLETE"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def endpoint_charge_mult(endpoint: str) -> tuple[int, int]:
    if endpoint == "cation":
        return CATION_CHARGE, CATION_MULTIPLICITY
    if endpoint == "neutral":
        return NEUTRAL_CHARGE, NEUTRAL_MULTIPLICITY
    raise TeacherRunnerError(f"invalid endpoint: {endpoint}")


def default_pilot_root_queue() -> tuple[str, ...]:
    """Scope C: development train + validation roots only (no Final Test)."""

    return tuple(TRAIN_ROOTS) + tuple(VALIDATION_ROOTS)


def _assert_no_final_test_roots(root_ids: Sequence[str]) -> None:
    # Soft guard: known sealed FT is opaque; reject obvious test keys if ever passed
    forbidden_tokens = ("final_test", "FINAL_TEST")
    for root_id in root_ids:
        lower = root_id.lower()
        if any(tok.lower() in lower for tok in forbidden_tokens):
            raise TeacherRunnerError(f"refusing Final Test-like root id: {root_id}")


def write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(dict(payload))
    if path.exists() and not overwrite:
        existing = path.read_bytes()
        if existing != raw:
            raise TeacherRunnerError(f"refusing overwrite of divergent file: {path}")
        return {"path": str(path), "bytes": len(raw), "sha256": sha256_bytes(raw), "wrote": False}
    path.write_bytes(raw)
    return {"path": str(path), "bytes": len(raw), "sha256": sha256_bytes(raw), "wrote": True}


@dataclass
class DryRunTeacherEngine:
    """Synthetic parent frames for path/layout contracts (not chemistry)."""

    frames_per_endpoint: int = 2

    def run_endpoint(
        self,
        *,
        root_id: str,
        endpoint: str,
        charge: int,
        multiplicity: int,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        if self.frames_per_endpoint <= 0:
            raise TeacherRunnerError("frames_per_endpoint must be positive")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        elements = ("C", "N", "H") if endpoint == "cation" else ("C", "N")
        n_atoms = len(elements)
        for index in range(self.frames_per_endpoint):
            coords = [[float(i) + 0.01 * index, 0.0, 0.0] for i in range(n_atoms)]
            # Fake small gradient / energy trajectory (never for real labels)
            energy = -100.0 - 0.001 * index
            gradient = [[1.0e-3 * (index + 1), 0.0, 0.0] for _ in range(n_atoms)]
            frame = {
                "schema": FRAME_SCHEMA,
                "dry_run": True,
                "live_chemistry": False,
                "root_id": root_id,
                "endpoint": endpoint,
                "frame_index": index,
                "parent_protocol_id": PROTOCOL_ID,
                "parent_protocol_sha256": PROTOCOL_SHA256,
                "functional": FUNCTIONAL,
                "basis": BASIS,
                "charge": charge,
                "multiplicity": multiplicity,
                "elements": list(elements),
                "coordinates_angstrom": coords,
                "energy_hartree": energy,
                "gradient_hartree_per_bohr": gradient,
                "forces_hartree_per_bohr": [[-g[0], -g[1], -g[2]] for g in gradient],
                "optimizer_step": index,
                "is_terminal": index == self.frames_per_endpoint - 1,
                "lineage": {
                    "mindmap_step": MINDMAP_STEP,
                    "engine": "DryRunTeacherEngine",
                    "single_point_only": False,
                },
            }
            path = output_dir / f"frame_{index:04d}.json"
            write_json(path, frame, overwrite=False)
            paths.append(str(path))
        manifest = {
            "schema": ENDPOINT_MANIFEST_SCHEMA,
            "root_id": root_id,
            "endpoint": endpoint,
            "frame_count": self.frames_per_endpoint,
            "complete_geometry_optimization": True,
            "dry_run": True,
            "parent_protocol_sha256": PROTOCOL_SHA256,
            "frames": [
                {"frame_index": i, "path": f"frame_{i:04d}.json"}
                for i in range(self.frames_per_endpoint)
            ],
        }
        write_json(output_dir / "manifest.json", manifest, overwrite=False)
        return {
            "frame_count": self.frames_per_endpoint,
            "converged": True,
            "frame_paths": paths,
            "notes": ["dry_run synthetic frames; not scientific labels"],
        }


def run_root_teacher(
    *,
    layout: GenerationLayout,
    root_id: str,
    engine: TeacherEngine,
    dry_run: bool = True,
) -> RootTeacherReceipt:
    """Run cation then neutral for one root (mindmap: both endpoints, not split)."""

    assert_parent_protocol_allowed(
        {"functional": FUNCTIONAL, "basis": BASIS, "protocol_sha256": PROTOCOL_SHA256}
    )
    receipt = RootTeacherReceipt(root_id=root_id, dry_run=dry_run, live_chemistry=not dry_run)
    try:
        for endpoint in ENDPOINTS:
            charge, mult = endpoint_charge_mult(endpoint)
            out_dir = layout.teacher_endpoint_dir(root_id, endpoint)
            result = engine.run_endpoint(
                root_id=root_id,
                endpoint=endpoint,
                charge=charge,
                multiplicity=mult,
                output_dir=out_dir,
            )
            ep = EndpointResult(
                root_id=root_id,
                endpoint=endpoint,
                frame_count=int(result.get("frame_count") or 0),
                converged=bool(result.get("converged")),
                frame_paths=list(result.get("frame_paths") or []),
                dry_run=dry_run,
                notes=list(result.get("notes") or []),
            )
            if ep.frame_count <= 0 or not ep.converged:
                raise TeacherRunnerError(
                    f"{root_id}/{endpoint}: incomplete teacher trajectory"
                )
            receipt.endpoints[endpoint] = ep.as_dict()
        # both endpoints required
        if set(receipt.endpoints) != set(ENDPOINTS):
            raise TeacherRunnerError(f"{root_id}: missing endpoint")
        receipt.status = "PASS"
        write_json(
            layout.teacher_root_dir(root_id) / "root_receipt.json",
            receipt.as_dict(),
            overwrite=False,
        )
    except Exception as exc:  # noqa: BLE001
        receipt.status = "FAILED"
        receipt.failure_reason = f"{type(exc).__name__}: {exc}"
        write_json(
            layout.teacher_root_dir(root_id) / "root_receipt.json",
            receipt.as_dict(),
            overwrite=True,
        )
        raise
    return receipt


def run_teacher_campaign(
    *,
    layout: GenerationLayout,
    root_ids: Sequence[str] | None = None,
    profile: ResourceProfile | None = None,
    profile_id: str = "single_27_physical_v1",
    engine: TeacherEngine | None = None,
    dry_run: bool = True,
    teacher_pyscf_authorized: bool = False,
    claim_pass: bool = False,
    selection_receipt_present: bool = False,
    live_dispatch_enabled: bool = False,
) -> TeacherCampaignReceipt:
    """Drive worker_pool over roots and write teacher products under g001.

    dry_run=True: always allowed (no chemistry). Live path is fail-closed.
    """

    roots = list(root_ids or default_pilot_root_queue())
    _assert_no_final_test_roots(roots)
    prof = profile or get_profile(profile_id)
    eng: TeacherEngine = engine or DryRunTeacherEngine()

    if not dry_run:
        if not teacher_pyscf_authorized:
            raise TeacherRunnerError(
                "live teacher requires teacher_pyscf_authorized=true"
            )
        if engine is None or isinstance(engine, DryRunTeacherEngine):
            raise TeacherRunnerError(
                "live teacher requires an injected non-dry TeacherEngine"
            )
        pool_probe = build_pool(
            prof,
            roots,
            claim_pass=claim_pass,
            selection_receipt_present=selection_receipt_present,
        )
        pool_probe.live_dispatch_enabled = live_dispatch_enabled
        assert_ready_for_live_dispatch(pool_probe, prof)

    # Dry-run may proceed without claim; live already gated above
    pool = build_pool(
        prof,
        roots,
        claim_pass=claim_pass if not dry_run else True,
        selection_receipt_present=selection_receipt_present if not dry_run else True,
    )
    if not dry_run:
        pool.live_dispatch_enabled = live_dispatch_enabled

    campaign = TeacherCampaignReceipt(
        generation_id=layout.generation_id,
        profile_id=prof.profile_id,
        dry_run=dry_run,
        live_chemistry=not dry_run,
        teacher_pyscf_authorized=teacher_pyscf_authorized,
        root_order=list(roots),
        notes=[
            f"mindmap_step={MINDMAP_STEP}",
            "endpoint order within root: cation then neutral",
            "parallel unit: molecular root via worker_pool",
            "single_point_only=false",
        ],
    )

    # Round-robin workers until queue empty (in-process; no OS processes)
    idle_rounds = 0
    max_idle = max(2, prof.worker_count * 2)
    while True:
        prog = progress_summary(pool)
        if prog["ready"] == 0 and prog["claimed"] == 0:
            break
        progressed = False
        for worker_id in range(prof.worker_count):
            slot = pool.slots[worker_id]
            if slot.status != "idle":
                continue
            root_id = claim_next_root(pool, worker_id)
            if root_id is None:
                continue
            progressed = True
            try:
                root_receipt = run_root_teacher(
                    layout=layout,
                    root_id=root_id,
                    engine=eng,
                    dry_run=dry_run,
                )
                complete_root(pool, root_id, success=True)
                campaign.root_receipts.append(root_receipt.as_dict())
            except Exception as exc:  # noqa: BLE001
                complete_root(pool, root_id, success=False, reason=str(exc))
                campaign.root_receipts.append(
                    {
                        "schema": ROOT_RECEIPT_SCHEMA,
                        "root_id": root_id,
                        "status": "FAILED",
                        "failure_reason": str(exc),
                        "dry_run": dry_run,
                    }
                )
        if not progressed:
            idle_rounds += 1
            if idle_rounds >= max_idle:
                # claimed slots without completion would deadlock; should not happen
                raise TeacherRunnerError("teacher campaign worker deadlock")
        else:
            idle_rounds = 0

    campaign.pool_progress = progress_summary(pool)
    failed = campaign.pool_progress.get("failed", 0)
    if dry_run:
        campaign.status = (
            "DRY_RUN_COMPLETE" if failed == 0 else "DRY_RUN_PARTIAL_FAILURE"
        )
    else:
        campaign.status = "LIVE_COMPLETE" if failed == 0 else "LIVE_PARTIAL_FAILURE"

    out = layout.logs_dir / "teacher_campaign_receipt.json"
    write_json(out, campaign.as_dict(), overwrite=True)
    # also under teacher/
    write_json(
        layout.teacher_dir / "campaign_receipt.json",
        campaign.as_dict(),
        overwrite=True,
    )
    return campaign


def plan_teacher_paths(
    layout: GenerationLayout, root_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Path-only plan (no writes) for mindmap step 2 dry inspection."""

    roots = list(root_ids or default_pilot_root_queue())
    _assert_no_final_test_roots(roots)
    plan = {
        "mindmap_step": MINDMAP_STEP,
        "generation_id": layout.generation_id,
        "teacher_dir": str(layout.teacher_dir),
        "roots": [],
    }
    for root_id in roots:
        plan["roots"].append(
            {
                "root_id": root_id,
                "cation_dir": str(layout.teacher_endpoint_dir(root_id, "cation")),
                "neutral_dir": str(layout.teacher_endpoint_dir(root_id, "neutral")),
                "root_receipt": str(layout.teacher_root_dir(root_id) / "root_receipt.json"),
            }
        )
    return plan
