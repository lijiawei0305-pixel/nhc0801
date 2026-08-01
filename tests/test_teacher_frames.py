"""Teacher frame path convention tests (synthetic filesystem only)."""

from __future__ import annotations

from pathlib import Path

from nhc_deprot.data.paths import autofill_run_dir, frame_path
from nhc_deprot.data.teacher_frames import (
    d3_receipt_path,
    expected_frame_path,
    inventory_candidates,
    list_candidate_frame_refs,
)


def test_frame_path_convention() -> None:
    runs = Path("/tmp/fake_runs")
    candidate = "ACGCNTKELWXJPN-UHFFFAOYSA-N"
    p = frame_path(runs, candidate, "cation", 3)
    assert p.name == "frame_0003.json"
    assert "autofill_acgcntkelwxjpn-uhfffaoysa-n_v001" in str(p)
    assert p.parts[-2] == "cation"
    assert expected_frame_path(runs, candidate, "cation", 3) == p


def test_inventory_counts_from_filesystem(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    candidate = "FAKE-ROOT-A"
    for endpoint, n in (("cation", 2), ("neutral", 3)):
        d = autofill_run_dir(runs, candidate) / "training_data" / endpoint
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"frame_{i:04d}.json").write_text("{}", encoding="utf-8")
    refs = list_candidate_frame_refs(runs, candidate)
    assert len(refs) == 5
    inv = inventory_candidates(runs, [candidate])
    assert inv[candidate]["frame_count"] == 5
    assert inv[candidate]["frame_count_by_endpoint"] == {"cation": 2, "neutral": 3}
    assert inv[candidate]["run_dir_exists"] is True


def test_d3_receipt_layout() -> None:
    root = Path("/proj")
    p = d3_receipt_path(root, "ROOT", "neutral", 12)
    assert p == root / "ROOT" / "neutral" / "frame_0012.json"
