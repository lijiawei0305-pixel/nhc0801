"""M2: live_teacher consumes geomeTRIC trajectory JSONL (no live PySCF)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nhc_deprot.pipeline.live_teacher import LiveParentTeacherEngine
from nhc_deprot.pipeline.teacher_runner import FRAME_SCHEMA


def _write_gold_xyz(gold_dir: Path, root_id: str, endpoint: str) -> Path:
    gold_dir.mkdir(parents=True, exist_ok=True)
    path = gold_dir / f"{root_id}_{endpoint}.xyz"
    path.write_text(
        "3\n"
        "synthetic\n"
        "C 0.0 0.0 0.0\n"
        "H 1.0 0.0 0.0\n"
        "H 0.0 1.0 0.0\n",
        encoding="utf-8",
    )
    return path


class FakeParentEngine:
    """Injectable parent: writes fixed JSONL + returns final_gradient (no second SCF)."""

    def __init__(
        self,
        *,
        n_eval: int = 3,
        write_trajectory: bool = True,
        include_final_gradient: bool = True,
    ) -> None:
        self.n_eval = n_eval
        self.write_trajectory = write_trajectory
        self.include_final_gradient = include_final_gradient
        self.first_gradient_calls = 0
        self.optimize_calls = 0
        self.last_trajectory_out_path: str | None = None

    def first_gradient(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        self.first_gradient_calls += 1
        return {
            "scf_converged": True,
            "energy_hartree": -10.0,
            "gradient_hartree_per_bohr": [
                [0.1, 0.0, 0.0],
                [-0.05, 0.0, 0.0],
                [-0.05, 0.0, 0.0],
            ],
        }

    def optimize_to_final_gau(
        self,
        *,
        trajectory_out_path: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self.optimize_calls += 1
        self.last_trajectory_out_path = trajectory_out_path
        if trajectory_out_path and self.write_trajectory:
            path = Path(trajectory_out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                for cycle in range(1, self.n_eval + 1):
                    row = {
                        "cycle": cycle,
                        "energy_hartree": -10.0 - 0.1 * cycle,
                        "coordinates_angstrom": [
                            [0.0, 0.0, 0.01 * cycle],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        "gradient_hartree_per_bohr": [
                            [0.01 / cycle, 0.0, 0.0],
                            [-0.005 / cycle, 0.0, 0.0],
                            [-0.005 / cycle, 0.0, 0.0],
                        ],
                    }
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")

        out: dict[str, Any] = {
            "geometry_converged": True,
            "final_single_point_converged": True,
            "energy_hartree": -10.5,
            "opt_steps": self.n_eval if trajectory_out_path else 100,
            "opt_steps_is_maxcap": trajectory_out_path is None,
            "final_grad_max_eh_bohr": 1.0e-5,
            "final_grad_rms_eh_bohr": 5.0e-6,
            "grad_gate_max": 4.5e-4,
            "grad_gate_rms": 3.0e-4,
            "coordinates": [
                [0.0, 0.0, 0.03],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        }
        if trajectory_out_path:
            out["trajectory_frame_count"] = self.n_eval if self.write_trajectory else 0
            out["trajectory_path"] = trajectory_out_path
            if self.include_final_gradient:
                out["final_gradient_hartree_per_bohr"] = [
                    [1.0e-5, 0.0, 0.0],
                    [-5.0e-6, 0.0, 0.0],
                    [-5.0e-6, 0.0, 0.0],
                ]
        return out


def test_capture_trajectory_writes_variable_frames(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    out = tmp_path / "endpoint"
    _write_gold_xyz(gold, "ROOTA", "cation")

    engine = LiveParentTeacherEngine(gold_xyz_dir=gold, capture_trajectory=True)
    fake = FakeParentEngine(n_eval=4)
    engine.parent = fake  # type: ignore[assignment]

    result = engine.run_endpoint(
        root_id="ROOTA",
        endpoint="cation",
        charge=1,
        multiplicity=1,
        output_dir=out,
    )

    # frame_0000 + 4 trajectory evaluations
    assert result["frame_count"] == 5
    assert result["trajectory_captured"] is True
    assert result["evaluation_count"] == 4
    assert fake.first_gradient_calls == 1  # no second first_gradient (T2)
    assert fake.optimize_calls == 1
    assert fake.last_trajectory_out_path == str(out / "trajectory.jsonl")
    assert (out / "trajectory.jsonl").is_file()

    terminals = []
    for i in range(5):
        frame = json.loads((out / f"frame_{i:04d}.json").read_text(encoding="utf-8"))
        assert frame["schema"] == FRAME_SCHEMA
        assert frame["frame_index"] == i
        terminals.append(frame["is_terminal"])
        if i == 0:
            assert frame["optimizer_step"] == 0
            assert frame["lineage"]["source"] == "first_gradient"
            assert frame["is_terminal"] is False
        else:
            assert frame["optimizer_step"] == i  # cycle 1..4
            assert frame["lineage"]["source"] == "geometric_callback"

    assert terminals.count(True) == 1
    assert terminals[-1] is True
    assert all(t is False for t in terminals[:-1])

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 5
    assert manifest["trajectory_captured"] is True
    assert manifest["evaluation_count"] == 4
    assert any("full geomeTRIC evaluation dump" in n for n in manifest["notes"])
    assert all("not full geomeTRIC step dump" not in n for n in manifest["notes"])


def test_capture_trajectory_false_keeps_legacy_second_gradient(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    out = tmp_path / "endpoint"
    _write_gold_xyz(gold, "ROOTB", "neutral")

    engine = LiveParentTeacherEngine(gold_xyz_dir=gold, capture_trajectory=False)
    fake = FakeParentEngine(n_eval=3)
    engine.parent = fake  # type: ignore[assignment]

    result = engine.run_endpoint(
        root_id="ROOTB",
        endpoint="neutral",
        charge=0,
        multiplicity=1,
        output_dir=out,
    )

    assert result["frame_count"] == 2
    assert result["trajectory_captured"] is False
    assert fake.first_gradient_calls == 2  # initial + final (legacy)
    assert fake.last_trajectory_out_path is None
    assert not (out / "trajectory.jsonl").exists()

    f0 = json.loads((out / "frame_0000.json").read_text(encoding="utf-8"))
    f1 = json.loads((out / "frame_0001.json").read_text(encoding="utf-8"))
    assert f0["is_terminal"] is False
    assert f1["is_terminal"] is True
    assert f0["lineage"]["source"] == "first_gradient"
    assert f1["lineage"]["source"] == "first_gradient"

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert any("not full geomeTRIC step dump" in n for n in manifest["notes"])


def test_empty_trajectory_uses_final_gradient_once(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    out = tmp_path / "endpoint"
    _write_gold_xyz(gold, "ROOTC", "cation")

    engine = LiveParentTeacherEngine(gold_xyz_dir=gold, capture_trajectory=True)
    fake = FakeParentEngine(n_eval=0, write_trajectory=False, include_final_gradient=True)
    engine.parent = fake  # type: ignore[assignment]

    result = engine.run_endpoint(
        root_id="ROOTC",
        endpoint="cation",
        charge=1,
        multiplicity=1,
        output_dir=out,
    )

    assert result["frame_count"] == 2
    assert fake.first_gradient_calls == 1
    f1 = json.loads((out / "frame_0001.json").read_text(encoding="utf-8"))
    assert f1["is_terminal"] is True
    assert f1["lineage"]["source"] == "geometric_callback"
    assert f1["gradient_hartree_per_bohr"][0][0] == pytest.approx(1.0e-5)


def test_empty_trajectory_without_final_gradient_raises(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    out = tmp_path / "endpoint"
    _write_gold_xyz(gold, "ROOTD", "cation")

    engine = LiveParentTeacherEngine(gold_xyz_dir=gold, capture_trajectory=True)
    fake = FakeParentEngine(
        n_eval=0, write_trajectory=False, include_final_gradient=False
    )
    engine.parent = fake  # type: ignore[assignment]

    with pytest.raises(Exception, match="final_gradient"):
        engine.run_endpoint(
            root_id="ROOTD",
            endpoint="cation",
            charge=1,
            multiplicity=1,
            output_dir=out,
        )
    assert fake.first_gradient_calls == 1
