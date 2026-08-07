"""TVT whole-package resplit (scheme A): stratified draw + identity profiles.

Authority: docs/plans/20260806_tvt_resplit_proposal.md (user-locked 150/16/3).
Does not open Final Test identities in development-visible artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from nhc_deprot.contracts.tvt_gates import audit_split_registry, canonical_json, sha256_bytes

FRAG_FIELDS: Final = ("n1_frag", "n3_frag", "c4_frag", "c5_frag")
SCAFFOLD_ID: Final = "nhc_rigid_small_v1"
SCHEMA_REGISTRY: Final = "nhc0801-tvt-resplit-registry-v001"
SCHEMA_DEVELOPMENT: Final = "nhc0801-tvt-resplit-development-split-v001"
DEFAULT_SEED: Final = 20260806
DEFAULT_TRAIN_N: Final = 150
DEFAULT_VAL_N: Final = 16
DEFAULT_FT_N: Final = 3
LOCKED_VAL: Final = (
    "KZYKDQNIIMATMJ-UHFFFAOYSA-N",
    "RMEQTBVGGNKAEQ-UHFFFAOYSA-N",
)


class TvtResplitError(RuntimeError):
    """Resplit draw failed closed."""


@dataclass(frozen=True, slots=True)
class PoolRow:
    inchikey: str
    n1_frag: str
    n3_frag: str
    c4_frag: str
    c5_frag: str
    smiles_cation: str
    smiles_neutral: str

    @property
    def frag_signature(self) -> str:
        return "|".join(getattr(self, f) for f in FRAG_FIELDS)


def load_pool_csv(path: Path) -> dict[str, PoolRow]:
    out: dict[str, PoolRow] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            ik = str(raw["inchikey"]).strip()
            out[ik] = PoolRow(
                inchikey=ik,
                n1_frag=str(raw["n1_frag"]).strip(),
                n3_frag=str(raw["n3_frag"]).strip(),
                c4_frag=str(raw["c4_frag"]).strip(),
                c5_frag=str(raw["c5_frag"]).strip(),
                smiles_cation=str(raw["smiles_cation"]).strip(),
                smiles_neutral=str(raw["smiles_neutral"]).strip(),
            )
    if not out:
        raise TvtResplitError(f"empty pool csv: {path}")
    return out


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frag_near_dup_id(row: PoolRow) -> str:
    """d=0 signature cluster id (exact four-frag match only)."""

    return _sha256_text("frag_sig_v1|" + row.frag_signature)


def build_identity_profile(row: PoolRow) -> dict[str, Any]:
    """Seven-dimension profile for audit_split_registry.

    Note: ``audit_split_registry`` treats each identity string as
    **globally unique across splits** (first owner wins). Therefore
    ``scaffold_id`` / ``lineage_root_id`` cannot be a pool-wide constant;
    they are root-scoped labels that still carry the common scaffold prefix.
    ``near_duplicate_cluster_id`` is the d=0 frag signature hash — roots that
    share it must not be placed in different splits.
    """

    return {
        "candidate": row.inchikey,
        "root_id": row.inchikey,
        "canonical_identity": row.smiles_neutral or row.inchikey,
        # Root-scoped so train/val/ft do not collide on a shared constant.
        "scaffold_id": f"{SCAFFOLD_ID}::{row.inchikey}",
        "lineage_root_id": f"{SCAFFOLD_ID}::{row.inchikey}",
        "near_duplicate_cluster_id": frag_near_dup_id(row),
        "cation_sha256": _sha256_text(row.smiles_cation),
        "neutral_sha256": _sha256_text(row.smiles_neutral),
        "n1_frag": row.n1_frag,
        "n3_frag": row.n3_frag,
        "c4_frag": row.c4_frag,
        "c5_frag": row.c5_frag,
        "smiles_cation": row.smiles_cation,
        "smiles_neutral": row.smiles_neutral,
        "scaffold_family": SCAFFOLD_ID,
    }


def _stratum_key(row: PoolRow) -> tuple[str, str]:
    return (row.n1_frag, row.n3_frag)


def stratified_sample(
    candidates: Sequence[str],
    *,
    pool: Mapping[str, PoolRow],
    k: int,
    rng: random.Random,
) -> list[str]:
    """Sample k roots with round-robin across (n1, n3) strata."""

    if k < 0:
        raise TvtResplitError(f"k must be >= 0, got {k}")
    if k == 0:
        return []
    avail = [c for c in candidates if c in pool]
    if k > len(avail):
        raise TvtResplitError(
            f"need {k} samples but only {len(avail)} candidates available"
        )
    by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in avail:
        by_stratum[_stratum_key(pool[c])].append(c)
    for bucket in by_stratum.values():
        rng.shuffle(bucket)
    keys = sorted(by_stratum.keys())
    rng.shuffle(keys)
    picked: list[str] = []
    # round-robin
    while len(picked) < k:
        progress = False
        for key in keys:
            bucket = by_stratum[key]
            if not bucket:
                continue
            picked.append(bucket.pop())
            progress = True
            if len(picked) >= k:
                break
        if not progress:
            break
    if len(picked) < k:
        raise TvtResplitError("stratified sample exhausted early")
    return sorted(picked)


def draw_tvt_resplit(
    *,
    pool: Mapping[str, PoolRow],
    usable_teacher_roots: Sequence[str],
    all_teacher_roots: Sequence[str],
    train_n: int = DEFAULT_TRAIN_N,
    val_n: int = DEFAULT_VAL_N,
    ft_n: int = DEFAULT_FT_N,
    locked_val: Sequence[str] = LOCKED_VAL,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Draw Train/Val/FT lists; returns registry payload (includes FT identities)."""

    if val_n < len(locked_val):
        raise TvtResplitError("val_n smaller than locked_val")
    locked = list(locked_val)
    for r in locked:
        if r not in pool:
            raise TvtResplitError(f"locked val root missing from pool: {r}")
        if r not in set(usable_teacher_roots):
            raise TvtResplitError(f"locked val root not in usable teacher set: {r}")

    rng = random.Random(int(seed))
    usable = sorted(set(usable_teacher_roots) & set(pool))
    teacher_seen = set(all_teacher_roots)
    untouched = sorted(ik for ik in pool if ik not in teacher_seen)

    # Val: locked + stratified fill from usable excluding locked
    val_need = val_n - len(locked)
    val_pool = [u for u in usable if u not in set(locked)]
    val_extra = stratified_sample(val_pool, pool=pool, k=val_need, rng=rng)
    val_roots = sorted(set(locked) | set(val_extra))
    if len(val_roots) != val_n:
        raise TvtResplitError(f"val size {len(val_roots)} != {val_n}")

    # Train: prefer usable not in val, then untouched
    train_prefer = [u for u in usable if u not in set(val_roots)]
    if len(train_prefer) >= train_n:
        train_roots = stratified_sample(
            train_prefer, pool=pool, k=train_n, rng=rng
        )
    else:
        need = train_n - len(train_prefer)
        train_fill = stratified_sample(
            [u for u in untouched if u not in set(val_roots)],
            pool=pool,
            k=need,
            rng=rng,
        )
        train_roots = sorted(set(train_prefer) | set(train_fill))
    if len(train_roots) != train_n:
        raise TvtResplitError(f"train size {len(train_roots)} != {train_n}")

    # FT: from untouched, not in train/val
    blocked = set(train_roots) | set(val_roots)
    ft_pool = [u for u in untouched if u not in blocked]
    ft_roots = stratified_sample(ft_pool, pool=pool, k=ft_n, rng=rng)

    # Disjoint check
    t_set, v_set, f_set = set(train_roots), set(val_roots), set(ft_roots)
    if t_set & v_set or t_set & f_set or v_set & f_set:
        raise TvtResplitError("split overlap after draw")

    # near_dup (frag signature) must not cross splits
    def _nd(r: str) -> str:
        return frag_near_dup_id(pool[r])

    owners: dict[str, str] = {}
    for split_name, roots in (
        ("train", train_roots),
        ("validation", val_roots),
        ("final_test", ft_roots),
    ):
        for r in roots:
            nd = _nd(r)
            prev = owners.get(nd)
            if prev is not None and prev != split_name:
                raise TvtResplitError(
                    f"near_dup signature crosses splits ({prev} vs {split_name}) "
                    f"for roots sharing frag vector (example root {r})"
                )
            owners[nd] = split_name

    train_prof = [build_identity_profile(pool[r]) for r in train_roots]
    val_prof = [build_identity_profile(pool[r]) for r in val_roots]
    ft_prof = [build_identity_profile(pool[r]) for r in ft_roots]

    registry = {
        "schema": SCHEMA_REGISTRY,
        "mindmap_step": 0,
        "split_unit": "molecular_root",
        "seed": int(seed),
        "near_dup_mode": "frag_signature_exact",
        "scaffold_id": SCAFFOLD_ID,
        "counts": {
            "train": len(train_roots),
            "validation": len(val_roots),
            "final_test": len(ft_roots),
        },
        "locked_validation_roots": list(locked),
        "train": train_prof,
        "validation": val_prof,
        "final_test": ft_prof,
        "train_roots": train_roots,
        "validation_roots": val_roots,
        "final_test_roots": ft_roots,
        "authority": "user_20260806_scheme_a_150_16_3",
        "notes": [
            "whole-package TVT resplit; prior pilot 3+2 and sealed FT superseded",
            "Train may include roots without teacher yet",
            "FT drawn from pool minus any teacher-touched root",
        ],
    }

    audit = audit_split_registry(
        {
            "train": train_prof,
            "validation": val_prof,
            "final_test": ft_prof,
        }
    )
    registry["audit_split_registry"] = audit
    if audit.get("status") != "PASS":
        raise TvtResplitError(
            f"audit_split_registry BLOCKED: missing={audit.get('missing')[:20]} "
            f"overlaps={audit.get('overlaps')[:10]}"
        )

    # Sealed FT commitment: hash sorted FT root list only (no training surface)
    ft_payload = {
        "schema": "nhc0801-sealed-final-test-v001",
        "root_count": len(ft_roots),
        "roots_sorted": sorted(ft_roots),
    }
    ft_commit_sha = sha256_bytes(canonical_json(ft_payload))
    registry["sealed_final_test_commitment"] = {
        "sha256": ft_commit_sha,
        "root_count": len(ft_roots),
    }
    registry["sealed_final_test_payload_for_hash_only"] = ft_payload

    return registry


def development_split_from_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Public development split: train+val profiles + sealed FT commitment only."""

    return {
        "schema": SCHEMA_DEVELOPMENT,
        "status": "TVT_RESPLIT_V001_DRAWN",
        "selection_authority": registry.get("authority"),
        "split_unit": "molecular_root",
        "seed": registry.get("seed"),
        "near_dup_mode": registry.get("near_dup_mode"),
        "train": list(registry["train"]),
        "validation": list(registry["validation"]),
        "sealed_final_test_commitment": dict(
            registry["sealed_final_test_commitment"]
        ),
        "sealed_final_test_commitment_metadata": {
            "hash_algorithm": "sha256",
            "canonicalization": "sorted-key compact JSON plus LF",
            "payload_visible_to_training": False,
            "root_count": registry["sealed_final_test_commitment"]["root_count"],
        },
        "locked_validation_roots": list(registry.get("locked_validation_roots") or []),
        "counts": {
            "train": len(registry["train"]),
            "validation": len(registry["validation"]),
        },
        "notes": list(registry.get("notes") or []),
    }


def write_resplit_artifacts(
    registry: Mapping[str, Any],
    *,
    out_dir: Path,
) -> dict[str, Path]:
    """Write development-visible + private FT + full registry (private)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    dev_path = out_dir / "nhc0801_g001_tvt_resplit_v001_development.json"
    reg_path = out_dir / "nhc0801_g001_tvt_resplit_v001_registry_private.json"
    ft_path = out_dir / "nhc0801_g001_tvt_resplit_v001_final_test_private.json"

    dev = development_split_from_registry(registry)
    # Bind sha of development document (without the sha field itself)
    dev["split_document_sha256"] = sha256_bytes(canonical_json(dev))

    reg = dict(registry)
    # Keep private registry complete for audit; not for training loaders
    ft_private = {
        "schema": "nhc0801-final-test-private-v001",
        "warning": "DO NOT load into training or publish; sealed commitment only publicly",
        "sealed_final_test_commitment": registry["sealed_final_test_commitment"],
        "final_test_roots": list(registry["final_test_roots"]),
        "final_test": list(registry["final_test"]),
        "hash_payload": registry["sealed_final_test_payload_for_hash_only"],
    }

    dev_path.write_text(
        json.dumps(dev, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reg_path.write_text(
        json.dumps(reg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ft_path.write_text(
        json.dumps(ft_private, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "development": dev_path,
        "registry_private": reg_path,
        "final_test_private": ft_path,
    }
