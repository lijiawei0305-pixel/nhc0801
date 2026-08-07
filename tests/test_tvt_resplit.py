"""Unit tests for TVT resplit stratified draw (synthetic pool)."""

from __future__ import annotations

from pathlib import Path

from nhc_deprot.contracts.tvt_gates import audit_split_registry
from nhc_deprot.pipeline.tvt_resplit import (
    PoolRow,
    build_identity_profile,
    development_split_from_registry,
    draw_tvt_resplit,
    write_resplit_artifacts,
)


def _row(ik: str, n1: str, n3: str, c4: str = "H", c5: str = "H") -> PoolRow:
    return PoolRow(
        inchikey=ik,
        n1_frag=n1,
        n3_frag=n3,
        c4_frag=c4,
        c5_frag=c5,
        smiles_cation=f"C[{ik}]/+",
        smiles_neutral=f"C[{ik}]",
    )


def _synth_pool(n: int = 40) -> dict[str, PoolRow]:
    # Locked val roots with distinct frags; every root unique frag signature
    # so near_dup cannot cross splits under d=0 policy.
    pool = {
        "KZYKDQNIIMATMJ-UHFFFAOYSA-N": _row(
            "KZYKDQNIIMATMJ-UHFFFAOYSA-N", "Et", "iPr", "F", "OMe"
        ),
        "RMEQTBVGGNKAEQ-UHFFFAOYSA-N": _row(
            "RMEQTBVGGNKAEQ-UHFFFAOYSA-N", "Me", "Et", "NH2", "NH2"
        ),
    }
    n1s = ["Me", "Et", "nBu", "tBu", "Vinyl", "Ethynyl"]
    n3s = ["Ph", "Vinyl", "Allyl", "iPr", "Propargyl", "tBu", "Ethynyl", "Et", "Me"]
    c4s = ["H", "F", "CN", "CF3", "NO2", "NH2", "OMe"]
    i = 0
    while len(pool) < n:
        ik = f"SYNTH{i:04d}-UHFFFAOYSA-N"
        # unique c5 ensures unique near_dup signatures under d=0 policy
        pool[ik] = _row(
            ik,
            n1s[i % len(n1s)],
            n3s[i % len(n3s)],
            c4s[i % len(c4s)],
            f"Ux{i}",
        )
        i += 1
    return pool


def test_draw_sizes_disjoint_and_audit_pass() -> None:
    pool = _synth_pool(40)
    usable = list(pool.keys())[:30]  # includes both locked
    # ensure locked in usable
    assert "KZYKDQNIIMATMJ-UHFFFAOYSA-N" in usable
    all_teacher = list(usable)
    untouched = [k for k in pool if k not in set(all_teacher)]
    assert len(untouched) >= 3

    reg = draw_tvt_resplit(
        pool=pool,
        usable_teacher_roots=usable,
        all_teacher_roots=all_teacher,
        train_n=20,
        val_n=4,
        ft_n=3,
        seed=20260806,
    )
    assert reg["counts"] == {"train": 20, "validation": 4, "final_test": 3}
    assert "KZYKDQNIIMATMJ-UHFFFAOYSA-N" in reg["validation_roots"]
    assert "RMEQTBVGGNKAEQ-UHFFFAOYSA-N" in reg["validation_roots"]
    t, v, f = (
        set(reg["train_roots"]),
        set(reg["validation_roots"]),
        set(reg["final_test_roots"]),
    )
    assert not (t & v or t & f or v & f)
    # FT not teacher-touched
    assert f.isdisjoint(set(all_teacher))
    assert reg["audit_split_registry"]["status"] == "PASS"
    assert reg["sealed_final_test_commitment"]["root_count"] == 3
    assert len(reg["sealed_final_test_commitment"]["sha256"]) == 64


def test_development_split_hides_ft_identities() -> None:
    pool = _synth_pool(40)
    usable = list(pool.keys())[:30]
    reg = draw_tvt_resplit(
        pool=pool,
        usable_teacher_roots=usable,
        all_teacher_roots=usable,
        train_n=15,
        val_n=4,
        ft_n=3,
        seed=1,
    )
    dev = development_split_from_registry(reg)
    blob = str(dev)
    for r in reg["final_test_roots"]:
        assert r not in blob
    assert "final_test" not in dev
    assert dev["sealed_final_test_commitment"]["root_count"] == 3


def test_write_artifacts(tmp_path: Path) -> None:
    pool = _synth_pool(40)
    usable = list(pool.keys())[:30]
    reg = draw_tvt_resplit(
        pool=pool,
        usable_teacher_roots=usable,
        all_teacher_roots=usable,
        train_n=12,
        val_n=4,
        ft_n=3,
        seed=2,
    )
    paths = write_resplit_artifacts(reg, out_dir=tmp_path)
    assert paths["development"].is_file()
    assert paths["final_test_private"].is_file()
    audit = audit_split_registry(
        {
            "train": reg["train"],
            "validation": reg["validation"],
            "final_test": reg["final_test"],
        }
    )
    assert audit["status"] == "PASS"


def test_profile_has_seven_dimensions() -> None:
    row = _row("ABC-UHFFFAOYSA-N", "Me", "Ph")
    prof = build_identity_profile(row)
    for k in (
        "root_id",
        "canonical_identity",
        "scaffold_id",
        "lineage_root_id",
        "near_duplicate_cluster_id",
        "cation_sha256",
        "neutral_sha256",
    ):
        assert isinstance(prof[k], str) and prof[k]
    assert len(prof["cation_sha256"]) == 64
