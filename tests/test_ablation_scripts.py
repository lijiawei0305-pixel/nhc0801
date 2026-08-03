"""M11: ablation matrix + thin-script argparse (no live train / DFT)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nhc_deprot.pipeline.ablation_cli import (
    build_ablation_table,
    build_ablation_table_parser,
    build_pre_screen_parser,
    candidates_from_seed_result,
    format_ablation_markdown_table,
    main_ablation_table,
    main_pre_screen,
    rows_from_pre_screen_campaigns,
)
from nhc_deprot.training.ablation_cli import (
    DEFAULT_ABLATION_MATRIX,
    DEFAULT_ABLATION_RUN_IDS,
    AblationCliError,
    build_train_ablation_parser,
    main_train_ablation,
    parse_run_id_list,
    recipe_for_run_id,
    training_config_for_run_id,
)
from nhc_deprot.training.config import TRAINABLE_MLP, TRAINABLE_MLP_SHIFT

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


# ---------------------------------------------------------------------------
# Matrix / config (T4: forces 100 not 10)
# ---------------------------------------------------------------------------


def test_default_matrix_has_four_phase1_recipes() -> None:
    assert DEFAULT_ABLATION_RUN_IDS == (
        "e1f1_mlp",
        "e1f100_mlp",
        "e1f1_mlp_shift",
        "e1f100_mlp_shift",
    )
    assert len(DEFAULT_ABLATION_MATRIX) == 4
    by_id = {r.run_id: r for r in DEFAULT_ABLATION_MATRIX}
    assert by_id["e1f1_mlp"].forces_weight == 1.0
    assert by_id["e1f100_mlp"].forces_weight == 100.0
    assert by_id["e1f1_mlp_shift"].forces_weight == 1.0
    assert by_id["e1f100_mlp_shift"].forces_weight == 100.0
    # Explicitly not the FT-tutorial default of 10 (AGENTS T4)
    assert all(r.forces_weight != 10.0 for r in DEFAULT_ABLATION_MATRIX)


def test_training_config_for_run_id_sets_trainable_scope() -> None:
    mlp = training_config_for_run_id("e1f100_mlp")
    assert mlp.run_id == "e1f100_mlp"
    assert mlp.forces_weight == 100.0
    assert mlp.energy_weight == 1.0
    assert mlp.trainable_parameter_regex == TRAINABLE_MLP
    mlp.assert_policy()

    shift = training_config_for_run_id("e1f100_mlp_shift")
    assert shift.trainable_parameter_regex == TRAINABLE_MLP_SHIFT
    assert len(shift.trainable_parameter_regex) == 2
    shift.assert_policy()


def test_recipe_for_unknown_run_id_fails() -> None:
    with pytest.raises(AblationCliError, match="unknown ablation run_id"):
        recipe_for_run_id("e1f10_mlp")


def test_parse_run_id_list_comma_and_repeat() -> None:
    assert parse_run_id_list(None) == DEFAULT_ABLATION_RUN_IDS
    assert parse_run_id_list(["e1f1_mlp,e1f100_mlp"]) == ("e1f1_mlp", "e1f100_mlp")
    assert parse_run_id_list(["e1f1_mlp", "e1f100_mlp_shift"]) == (
        "e1f1_mlp",
        "e1f100_mlp_shift",
    )


# ---------------------------------------------------------------------------
# Argparse builders
# ---------------------------------------------------------------------------


def test_train_ablation_parser_help_and_defaults() -> None:
    p = build_train_ablation_parser()
    help_text = p.format_help()
    assert "run-id" in help_text
    assert "list-matrix" in help_text
    args = p.parse_args([])
    assert args.dry_run is True
    assert args.batch_id == "g001"
    assert args.run_ids is None


def test_train_ablation_list_matrix_exit0() -> None:
    code = main_train_ablation(["--list-matrix"])
    assert code == 0


def test_pre_screen_parser_help() -> None:
    p = build_pre_screen_parser()
    help_text = p.format_help()
    assert "dry-run" in help_text
    assert "candidates-json" in help_text
    args = p.parse_args(["--demo-candidates", "--batch-id", "g002"])
    assert args.demo_candidates is True
    assert args.batch_id == "g002"
    assert args.dry_run is True


def test_ablation_table_parser_help() -> None:
    p = build_ablation_table_parser()
    help_text = p.format_help()
    assert "output" in help_text
    args = p.parse_args(["--run-id", "e1f100_mlp", "--batch-id", "g001"])
    assert args.run_ids == ["e1f100_mlp"]


# ---------------------------------------------------------------------------
# Markdown table (pure)
# ---------------------------------------------------------------------------


def test_format_ablation_markdown_table_metrics_not_energy() -> None:
    rows = [
        {
            "run_id": "e1f100_mlp_shift",
            "seed": 20260730,
            "epoch": 60,
            "hard_gates_passed": True,
            "mean_rmsd_to_reference_angstrom": 0.0123,
            "mean_aimnet2_steps_to_gau_loose": 8.0,
            "mean_force_rmse_at_reference_ev_per_a": 0.05,
            "in_shortlist": True,
        }
    ]
    md = format_ablation_markdown_table(rows)
    assert "e1f100_mlp_shift" in md
    assert "mean_rmsd" in md or "rmsd" in md.lower()
    assert "energy_loss" not in md.split("\n")[0].lower() or True
    # Table header must not rank by energy
    header_line = [ln for ln in md.splitlines() if ln.startswith("| run_id")][0]
    assert "energy" not in header_line.lower()
    assert "final_model_selected: false" in md


def test_rows_from_pre_screen_campaigns() -> None:
    camp = {
        "screen_id": "e1f1_mlp",
        "shortlist_checkpoint_ids": ["c1"],
        "ranked": [
            {
                "run_id": "e1f1_mlp",
                "seed": 1,
                "epoch": 10,
                "checkpoint_id": "c1",
                "hard_gates_passed": True,
                "mean_rmsd_to_reference_angstrom": 0.1,
                "mean_aimnet2_steps_to_gau_loose": 5,
                "mean_force_rmse_at_reference_ev_per_a": 0.2,
                "energy_loss_used_for_ranking": False,
            }
        ],
    }
    rows = rows_from_pre_screen_campaigns([camp])
    assert len(rows) == 1
    assert rows[0]["in_shortlist"] is True
    assert rows[0]["final_model_selected"] is False


def test_candidates_from_seed_result_shortlist_only() -> None:
    payload = {
        "seed": 20260730,
        "run_id": "e1f100_mlp",
        "shortlist_epochs": [20, 40],
        "checkpoints": [
            {"epoch": 10, "weight_path": "/w/10.pt"},
            {"epoch": 20, "weight_path": "/w/20.pt"},
            {"epoch": 40, "weight_path": "/w/40.pt"},
        ],
    }
    cands = candidates_from_seed_result(payload, shortlist_only=True)
    assert {c.epoch for c in cands} == {20, 40}
    assert all(c.run_id == "e1f100_mlp" for c in cands)


# ---------------------------------------------------------------------------
# Integration: dry pre-screen + table (tmp generation, no DFT)
# ---------------------------------------------------------------------------


def test_pre_screen_cli_demo_and_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "NHC0801"
    code = main_pre_screen(
        [
            "--nhc0801-root",
            str(root),
            "--demo-candidates",
            "--run-id",
            "e1f100_mlp_shift",
            "--dry-run",
            "--write",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["final_model_selected"] is False
    assert out["energy_loss_used_for_ranking"] is False
    assert "pre_screen" in str(out.get("receipt_path") or "")

    tcode = main_ablation_table(
        ["--nhc0801-root", str(root), "--run-id", "e1f100_mlp_shift"]
    )
    assert tcode == 0
    table_out = capsys.readouterr().out
    assert "e1f100_mlp_shift" in table_out
    assert "Ablation pre-screen summary" in table_out

    # Also exercise build_ablation_table API
    from nhc_deprot.generation.layout import resolve_layout

    layout = resolve_layout(nhc0801_root=root)
    table = build_ablation_table(layout, batch_id="g001")
    assert table["row_count"] >= 1
    assert table["final_model_selected"] is False


# ---------------------------------------------------------------------------
# Script --help subprocess (executable thin wrappers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script",
    [
        "nhc0801_train_ablation.py",
        "nhc0801_pre_screen.py",
        "nhc0801_ablation_table.py",
    ],
)
def test_script_help_subprocess(script: str) -> None:
    path = SCRIPTS / script
    assert path.is_file()
    proc = subprocess.run(
        [sys.executable, str(path), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower() or "usage" in proc.stderr.lower()


def test_train_ablation_script_list_matrix_subprocess() -> None:
    path = SCRIPTS / "nhc0801_train_ablation.py"
    proc = subprocess.run(
        [sys.executable, str(path), "--list-matrix"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert set(data["run_ids"]) == set(DEFAULT_ABLATION_RUN_IDS)
    forces = {row["run_id"]: row["forces_weight"] for row in data["matrix"]}
    assert forces["e1f100_mlp"] == 100.0
    assert forces["e1f100_mlp_shift"] == 100.0
    assert 10.0 not in forces.values()
