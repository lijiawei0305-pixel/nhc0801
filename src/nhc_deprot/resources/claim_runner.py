"""Orchestrate two-sample resource claims and durable receipts (no chemistry)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from nhc_deprot.data.io_util import write_json
from nhc_deprot.generation.layout import GenerationLayout
from nhc_deprot.resources.claim import (
    ClaimResult,
    HostSnapshot,
    evaluate_claim,
    gates_from_catalog,
)
from nhc_deprot.resources.host_sampler import (
    ProbeRequest,
    snapshot_to_dict,
    take_two_samples,
)
from nhc_deprot.resources.profiles import get_profile, load_profile_catalog

CLAIM_RECEIPT_SCHEMA: Final = "nhc0801-resource-claim-receipt-v1"


def run_resource_claim(
    *,
    layout: GenerationLayout,
    profile_id: str = "single_27_physical_v1",
    mode: str = "local",
    ssh_alias: str | None = None,
    disk_path: str = "/",
    interval_s: float = 5.0,
    claim_id: str | None = None,
    chemistry_authorized: bool = False,
) -> dict[str, Any]:
    """Sample twice, evaluate gates, write receipt under generation resources/.

    ``chemistry_authorized`` is always recorded false unless caller explicitly
    passes True **and** claim PASSes — still does not start chemistry.
    """

    profile = get_profile(profile_id)
    catalog = load_profile_catalog()
    gates = gates_from_catalog(catalog)
    cpu_list = profile.cpu_lists[0] if profile.worker_count == 1 else ",".join(profile.cpu_lists)

    # For dual, require entire union idle
    if profile.worker_count > 1:
        cpu_list = ",".join(profile.cpu_lists)

    request = ProbeRequest(
        cpu_list=cpu_list,
        disk_path=disk_path,
        mode=mode,
        ssh_alias=ssh_alias,
    )
    s0, s1, sample_meta = take_two_samples(request, interval_s=interval_s)
    result = evaluate_claim(samples=(s0, s1), profile=profile, gates=gates)

    # Never imply chemistry is live-runnable without explicit user gate flag
    chemistry_ok = bool(
        result.chemistry_permitted and chemistry_authorized and result.status.endswith("PASS")
    )

    import secrets

    cid = claim_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(3)
    )
    receipt = {
        "schema": CLAIM_RECEIPT_SCHEMA,
        "claim_id": cid,
        "generation_id": layout.generation_id,
        "profile_id": profile.profile_id,
        "mode": mode,
        "status": result.status,
        "sample_count": result.sample_count,
        "reasons": result.reasons,
        "chemistry_permitted_by_resources": result.chemistry_permitted,
        "chemistry_authorized_by_user": chemistry_authorized,
        "chemistry_run_allowed": chemistry_ok,
        "dual_escalation_permitted": result.dual_escalation_permitted,
        "samples": [snapshot_to_dict(s0), snapshot_to_dict(s1)],
        "sample_meta": sample_meta,
        "gates": asdict(gates),
        "notes": list(result.notes)
        + [
            "read-only sampling; no PySCF/AIMNet2/train started",
            "PASS claim does not open teacher_pyscf_authorized / epoch0_execution",
        ],
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    layout.resources_dir.mkdir(parents=True, exist_ok=True)
    path = layout.resource_claim_path(cid)
    write_json(path, receipt, overwrite=False)
    # latest pointer (overwrite ok)
    write_json(layout.resources_dir / "claim_latest.json", receipt, overwrite=True)
    write_json(layout.logs_dir / "resource_claim_latest.json", receipt, overwrite=True)

    return {
        "receipt_path": str(path),
        "status": result.status,
        "chemistry_run_allowed": chemistry_ok,
        "reasons": result.reasons,
        "profile_id": profile.profile_id,
        "claim_id": cid,
        "receipt": receipt,
    }


def evaluate_injected_samples(
    *,
    layout: GenerationLayout,
    samples: tuple[HostSnapshot, HostSnapshot],
    profile_id: str = "single_27_physical_v1",
    claim_id: str = "injected",
) -> dict[str, Any]:
    """Unit-test / offline path: evaluate without host access."""

    profile = get_profile(profile_id)
    catalog = load_profile_catalog()
    gates = gates_from_catalog(catalog)
    result: ClaimResult = evaluate_claim(samples=samples, profile=profile, gates=gates)
    receipt = {
        "schema": CLAIM_RECEIPT_SCHEMA,
        "claim_id": claim_id,
        "generation_id": layout.generation_id,
        "profile_id": profile_id,
        "mode": "injected",
        "status": result.status,
        "sample_count": result.sample_count,
        "reasons": result.reasons,
        "chemistry_permitted_by_resources": result.chemistry_permitted,
        "chemistry_authorized_by_user": False,
        "chemistry_run_allowed": False,
        "dual_escalation_permitted": result.dual_escalation_permitted,
        "samples": [snapshot_to_dict(s) for s in samples],
        "notes": ["injected samples; no host access"],
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    layout.resources_dir.mkdir(parents=True, exist_ok=True)
    write_json(layout.resource_claim_path(claim_id), receipt, overwrite=True)
    return {"status": result.status, "reasons": result.reasons, "receipt": receipt}
