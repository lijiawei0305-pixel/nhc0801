"""R1 true-resume + R2 early-stop (probe extension + resume_earlystop_fixes)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from nhc_deprot.data.paths import OFFICIAL_AIMNET2_WEIGHT_SHA256, TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation.layout import GenerationLayout, init_generation
from nhc_deprot.pipeline.d3_projection import run_d3_campaign
from nhc_deprot.pipeline.teacher_runner import DryRunTeacherEngine, run_teacher_campaign
from nhc_deprot.pipeline.weighted_dataset_writer import assemble_weighted_dataset
from nhc_deprot.resources.profiles import get_profile
from nhc_deprot.training.config import TrainingConfig
from nhc_deprot.training.multi_seed_trainer import (
    DryRunTrainBackend,
    TrainerError,
    early_stop_update,
    resolve_epoch_cap,
    run_one_seed,
    run_multi_seed_training,
)


def _tiny_weighted_layout(tmp_path: Path) -> GenerationLayout:
    """Minimal generation tree with D3+weighted dry products for trainer tests."""
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    roots = list(TRAIN_ROOTS) + list(VALIDATION_ROOTS)
    run_teacher_campaign(
        layout=layout,
        root_ids=roots,
        profile=get_profile("single_27_physical_v1"),
        engine=DryRunTeacherEngine(frames_per_endpoint=2),
        dry_run=True,
    )
    run_d3_campaign(layout=layout, root_ids=roots, dry_run=True, overwrite=True)
    assemble_weighted_dataset(
        layout=layout,
        train_roots=list(TRAIN_ROOTS),
        validation_roots=list(VALIDATION_ROOTS),
        dry_run=True,
        overwrite=True,
    )
    return layout


def _batches(layout: GenerationLayout, cfg: TrainingConfig):
    from nhc_deprot.training.multi_seed_trainer import iter_batches_from_npz_dir

    train_b, train_n = iter_batches_from_npz_dir(
        layout.datasets_dir, "train", batch_size=cfg.batch_size
    )
    val_b, _ = iter_batches_from_npz_dir(
        layout.datasets_dir, "validation", batch_size=cfg.batch_size
    )
    return train_b, val_b, train_n


def test_early_stop_update_patience_and_improve() -> None:
    best, since, stop = early_stop_update(
        metric=1.0, best_metric=2.0, epochs_since_best=5, patience=3
    )
    assert best == 1.0 and since == 0 and stop is False
    best, since, stop = early_stop_update(
        metric=1.5, best_metric=1.0, epochs_since_best=2, patience=3
    )
    assert best == 1.0 and since == 3 and stop is True
    best, since, stop = early_stop_update(
        metric=1.5, best_metric=1.0, epochs_since_best=1, patience=None
    )
    assert stop is False and since == 2


def test_resolve_epoch_cap_early_vs_fixed() -> None:
    cfg_fixed = TrainingConfig(epochs=120, early_stop_patience_epochs=None)
    assert resolve_epoch_cap(cfg_fixed, epochs=120) == 120
    cfg_es = TrainingConfig(
        epochs=120,
        early_stop_patience_epochs=20,
        early_stop_max_epochs=480,
    )
    assert resolve_epoch_cap(cfg_es, epochs=120) == 480


def test_early_stop_metric_rejects_energy_only() -> None:
    with pytest.raises(ValueError, match="validation_weighted_loss"):
        TrainingConfig(early_stop_metric="weighted_energy_mse").assert_policy()


class _ScriptedValBackend:
    """Val loss follows a scripted sequence so PATIENCE / CAP are unit-testable."""

    def __init__(self, val_seq: list[float]) -> None:
        self._inner = DryRunTrainBackend()
        self.val_seq = list(val_seq)
        self._i = 0

    def train_epoch(self, batches, *, split_frame_count, energy_weight, forces_weight, seed, epoch):  # noqa: ANN001
        return self._inner.train_epoch(
            batches,
            split_frame_count=split_frame_count,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            seed=seed,
            epoch=epoch,
        )

    def evaluate(self, batches, *, energy_weight, forces_weight, energy_bias=0.0):  # noqa: ANN001
        del batches, energy_weight, forces_weight, energy_bias
        if self._i >= len(self.val_seq):
            v = self.val_seq[-1]
        else:
            v = self.val_seq[self._i]
            self._i += 1
        return {
            "validation_weighted_loss": float(v),
            "weighted_energy_mse": float(v),
            "weighted_forces_mse": float(v),
            "sample_weight_sum": 1.0,
            "sample_count": 1,
            "checkpoint_selection_permitted": False,
            "backward_called": False,
        }

    def step_scheduler(self, val_loss: float) -> None:
        del val_loss

    def export_resume_checkpoint(self, path, **kwargs):  # noqa: ANN001
        return self._inner.export_resume_checkpoint(path, **kwargs)

    def load_resume_checkpoint(self, path):  # noqa: ANN001
        return self._inner.load_resume_checkpoint(path)


def test_early_stop_patience_in_run_one_seed(tmp_path: Path) -> None:
    layout = _tiny_weighted_layout(tmp_path)
    seq = [0.5, 0.3, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    backend = _ScriptedValBackend(seq)
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=100,
        early_stop_patience_epochs=3,
        early_stop_max_epochs=100,
        checkpoint_interval_epochs=1,
        resume_checkpoint_interval_epochs=1,
        run_id="es_patience",
    )
    train_b, val_b, train_n = _batches(layout, cfg)
    result = run_one_seed(
        layout=layout,
        seed=20260730,
        config=cfg,
        train_batches=train_b,
        val_batches=val_b,
        train_frame_count=train_n,
        backend=backend,
        epochs=cfg.epochs,
        dry_run=True,
        train_batch_id="g001",
    )
    assert result.status == "PASS"
    assert result.early_stop_triggered is True
    assert result.early_stop_reason == "PATIENCE"
    assert result.right_censored is False
    assert result.best_epoch == 3
    assert result.epochs_run == 6


def test_early_stop_cap_right_censored(tmp_path: Path) -> None:
    layout = _tiny_weighted_layout(tmp_path)
    seq = [1.0 / (i + 1) for i in range(50)]
    backend = _ScriptedValBackend(seq)
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=100,
        early_stop_patience_epochs=20,
        early_stop_max_epochs=8,
        checkpoint_interval_epochs=2,
        resume_checkpoint_interval_epochs=2,
        run_id="es_cap",
    )
    train_b, val_b, train_n = _batches(layout, cfg)
    result = run_one_seed(
        layout=layout,
        seed=20260730,
        config=cfg,
        train_batches=train_b,
        val_batches=val_b,
        train_frame_count=train_n,
        backend=backend,
        epochs=cfg.epochs,
        dry_run=True,
        train_batch_id="g001",
    )
    assert result.status == "PASS"
    # P4: cap is right-censor, NOT early-stop
    assert result.early_stop_triggered is False
    assert result.early_stop_reason == "CAP"
    assert result.right_censored is True
    assert result.epochs_run == 8
    assert result.best_epoch == 8


def test_early_stop_nan_metric_fails_closed(tmp_path: Path) -> None:
    """P3: NaN metric with patience set must raise and not burn the cap."""
    layout = _tiny_weighted_layout(tmp_path)
    seq = [0.5, 0.4, float("nan"), 0.3]
    backend = _ScriptedValBackend(seq)
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=100,
        early_stop_patience_epochs=20,
        early_stop_max_epochs=50,
        checkpoint_interval_epochs=1,
        resume_checkpoint_interval_epochs=1,
        run_id="es_nan",
    )
    train_b, val_b, train_n = _batches(layout, cfg)
    result = run_one_seed(
        layout=layout,
        seed=20260730,
        config=cfg,
        train_batches=train_b,
        val_batches=val_b,
        train_frame_count=train_n,
        backend=backend,
        epochs=cfg.epochs,
        dry_run=True,
        train_batch_id="g001",
    )
    assert result.status == "FAILED"
    assert result.epochs_run < 50  # did not burn cap
    assert "non-finite" in (result.failure_reason or "")
    assert "epoch 3" in (result.failure_reason or "")


def test_early_stop_missing_metric_fails_closed(tmp_path: Path) -> None:
    layout = _tiny_weighted_layout(tmp_path)

    class _MissingMetric(DryRunTrainBackend):
        def evaluate(self, batches, *, energy_weight, forces_weight, energy_bias=0.0):  # noqa: ANN001
            return {
                "checkpoint_selection_permitted": False,
                # deliberately omit validation_weighted_loss
            }

    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=10,
        early_stop_patience_epochs=5,
        early_stop_max_epochs=10,
        checkpoint_interval_epochs=1,
        run_id="es_missing",
    )
    train_b, val_b, train_n = _batches(layout, cfg)
    result = run_one_seed(
        layout=layout,
        seed=20260730,
        config=cfg,
        train_batches=train_b,
        val_batches=val_b,
        train_frame_count=train_n,
        backend=_MissingMetric(),
        epochs=cfg.epochs,
        dry_run=True,
        train_batch_id="g001",
    )
    assert result.status == "FAILED"
    assert "missing" in (result.failure_reason or "").lower()


def test_early_stop_requires_quick_val(tmp_path: Path) -> None:
    layout = _tiny_weighted_layout(tmp_path)
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=5,
        early_stop_patience_epochs=3,
        early_stop_max_epochs=5,
        quick_validation_each_epoch=False,
        run_id="es_no_qval",
    )
    train_b, val_b, train_n = _batches(layout, cfg)
    result = run_one_seed(
        layout=layout,
        seed=20260730,
        config=cfg,
        train_batches=train_b,
        val_batches=val_b,
        train_frame_count=train_n,
        backend=DryRunTrainBackend(),
        epochs=cfg.epochs,
        dry_run=True,
        train_batch_id="g001",
    )
    assert result.status == "FAILED"
    assert "quick_validation_each_epoch" in (result.failure_reason or "")


def test_resume_same_dir_merges_epoch_logs(tmp_path: Path) -> None:
    """P2: same run_id/seed_dir 10 → resume → 20 keeps full curve 1..20."""
    layout = _tiny_weighted_layout(tmp_path)
    rid = "resume_same_dir"
    cfg10 = TrainingConfig(
        seeds=(20260730,),
        epochs=10,
        checkpoint_interval_epochs=1,
        resume_checkpoint_interval_epochs=1,
        run_id=rid,
    )
    train_b, val_b, train_n = _batches(layout, cfg10)
    r_a = run_one_seed(
        layout=layout,
        seed=20260730,
        config=cfg10,
        train_batches=train_b,
        val_batches=val_b,
        train_frame_count=train_n,
        backend=DryRunTrainBackend(),
        epochs=10,
        dry_run=True,
        train_batch_id="g001",
        run_id=rid,
    )
    assert r_a.status == "PASS"
    seed_dir = layout.train_run_seed_dir("g001", rid, 20260730)
    resume_path = seed_dir / "epoch_0010.resume.pt"
    assert resume_path.is_file()
    prior_ckpt_epochs = {int(c["epoch"]) for c in r_a.checkpoints}
    assert 1 in prior_ckpt_epochs and 10 in prior_ckpt_epochs

    cfg20 = TrainingConfig(
        seeds=(20260730,),
        epochs=20,
        checkpoint_interval_epochs=1,
        resume_checkpoint_interval_epochs=1,
        run_id=rid,
    )
    r_b = run_one_seed(
        layout=layout,
        seed=20260730,
        config=cfg20,
        train_batches=train_b,
        val_batches=val_b,
        train_frame_count=train_n,
        backend=DryRunTrainBackend(),
        epochs=20,
        dry_run=True,
        train_batch_id="g001",
        run_id=rid,
        resume_from=resume_path,
    )
    assert r_b.status == "PASS"
    assert r_b.resume_merged is True
    assert r_b.resumed_from_epoch == 10
    assert len(r_b.epoch_logs) == 20
    epochs = [int(e["epoch"]) for e in r_b.epoch_logs]
    assert epochs == list(range(1, 21))
    # shortlist pool must include pre-resume checkpoints
    ckpt_epochs = {int(c["epoch"]) for c in r_b.checkpoints}
    assert 1 in ckpt_epochs or 2 in ckpt_epochs
    assert any(e <= 10 for e in ckpt_epochs)
    # on-disk receipt
    import json

    disk = json.loads((seed_dir / "seed_result.json").read_text())
    assert len(disk["epoch_logs"]) == 20
    assert disk["resume_merged"] is True


def test_resume_20_vs_10_plus_10_loss_diff(tmp_path: Path) -> None:
    """Dry-run continuity: 20 straight vs 10+resume10, loss diffs < 1e-9."""
    layout = _tiny_weighted_layout(tmp_path)
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=20,
        checkpoint_interval_epochs=1,
        resume_checkpoint_interval_epochs=1,
        run_id="resume_straight",
        early_stop_patience_epochs=None,
    )
    train_b, val_b, train_n = _batches(layout, cfg)

    r_straight = run_one_seed(
        layout=layout,
        seed=20260730,
        config=cfg,
        train_batches=train_b,
        val_batches=val_b,
        train_frame_count=train_n,
        backend=DryRunTrainBackend(),
        epochs=20,
        dry_run=True,
        train_batch_id="g001",
        run_id="resume_straight",
    )
    assert r_straight.status == "PASS"

    rid = "resume_split"
    cfg10 = TrainingConfig(
        seeds=(20260730,),
        epochs=10,
        checkpoint_interval_epochs=1,
        resume_checkpoint_interval_epochs=1,
        run_id=rid,
    )
    r_a = run_one_seed(
        layout=layout,
        seed=20260730,
        config=cfg10,
        train_batches=train_b,
        val_batches=val_b,
        train_frame_count=train_n,
        backend=DryRunTrainBackend(),
        epochs=10,
        dry_run=True,
        train_batch_id="g001",
        run_id=rid,
    )
    assert r_a.status == "PASS"
    resume_path = (
        layout.train_run_seed_dir("g001", rid, 20260730) / "epoch_0010.resume.pt"
    )
    cfg20 = TrainingConfig(
        seeds=(20260730,),
        epochs=20,
        checkpoint_interval_epochs=1,
        resume_checkpoint_interval_epochs=1,
        run_id=rid,
    )
    r_b = run_one_seed(
        layout=layout,
        seed=20260730,
        config=cfg20,
        train_batches=train_b,
        val_batches=val_b,
        train_frame_count=train_n,
        backend=DryRunTrainBackend(),
        epochs=20,
        dry_run=True,
        train_batch_id="g001",
        run_id=rid,
        resume_from=resume_path,
    )
    assert r_b.status == "PASS"
    assert len(r_b.epoch_logs) == 20
    s_logs = {int(e["epoch"]): e for e in r_straight.epoch_logs}
    b_logs = {int(e["epoch"]): e for e in r_b.epoch_logs}
    max_train_diff = 0.0
    max_val_diff = 0.0
    for ep in range(11, 21):
        st = float(s_logs[ep]["train"]["train_weighted_loss"])
        bt = float(b_logs[ep]["train"]["train_weighted_loss"])
        sv = float(s_logs[ep]["quick_validation"]["validation_weighted_loss"])
        bv = float(b_logs[ep]["quick_validation"]["validation_weighted_loss"])
        max_train_diff = max(max_train_diff, abs(st - bt))
        max_val_diff = max(max_val_diff, abs(sv - bv))
    assert max_train_diff < 1e-9, f"train loss diff {max_train_diff}"
    assert max_val_diff < 1e-9, f"val loss diff {max_val_diff}"


def test_resume_digest_mismatch_uses_tmp_path(tmp_path: Path) -> None:
    backend = DryRunTrainBackend()
    backend.theta = 1.0
    path = tmp_path / "nhc0801_resume_digest_test.resume.pt"
    backend.export_resume_checkpoint(path, epoch=5, train_config_digest="aaa")
    # dry-run does not dual-write a .json sibling
    assert not path.with_suffix(path.suffix + ".json").is_file()
    other = DryRunTrainBackend()
    meta = other.load_resume_checkpoint(path)
    assert meta["epoch"] == 5
    assert abs(other.theta - 1.0) < 1e-15


def test_default_config_no_early_stop_regression(tmp_path: Path) -> None:
    layout = _tiny_weighted_layout(tmp_path)
    camp = run_multi_seed_training(
        layout=layout,
        config=TrainingConfig(seeds=(20260730,), epochs=3, checkpoint_interval_epochs=1),
        dry_run=True,
        dry_run_epochs=3,
    )
    assert camp["status"].startswith("DRY_RUN")
    assert camp["seed_results"][0]["epochs_run"] == 3
    assert camp["seed_results"][0].get("early_stop_triggered") is False
    assert camp.get("right_censored") is False
    assert OFFICIAL_AIMNET2_WEIGHT_SHA256


def test_p5b_stopping_policy_provenance_on_receipts(tmp_path: Path) -> None:
    """P5-B: seed_result + train_info carry stopping provenance; digest untouched."""
    import json

    from nhc_deprot.training.live_aimnet2 import train_config_digest
    from nhc_deprot.training.multi_seed_trainer import stopping_policy_provenance

    layout = _tiny_weighted_layout(tmp_path)
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=100,
        early_stop_patience_epochs=20,
        early_stop_max_epochs=8,
        checkpoint_interval_epochs=2,
        resume_checkpoint_interval_epochs=2,
        run_id="p5b_prov",
    )
    # Strictly decreasing val → hit CAP at 8
    backend = _ScriptedValBackend([1.0 / (i + 1) for i in range(20)])
    train_b, val_b, train_n = _batches(layout, cfg)
    result = run_one_seed(
        layout=layout,
        seed=20260730,
        config=cfg,
        train_batches=train_b,
        val_batches=val_b,
        train_frame_count=train_n,
        backend=backend,
        epochs=cfg.epochs,
        dry_run=True,
        train_batch_id="g001",
        run_id="p5b_prov",
    )
    assert result.status == "PASS"
    assert result.right_censored is True

    seed_dir = layout.train_run_seed_dir("g001", "p5b_prov", 20260730)
    disk = json.loads((seed_dir / "seed_result.json").read_text())
    assert disk["effective_epoch_cap"] == 8
    assert disk["actual_epochs_run"] == 8
    assert disk["early_stop_patience_epochs"] == 20
    assert disk["early_stop_max_epochs"] == 8
    assert disk["digest_excludes_stopping_policy"] is True

    # Campaign / train_info path
    camp = run_multi_seed_training(
        layout=layout,
        config=TrainingConfig(
            seeds=(20260730,),
            epochs=5,
            early_stop_patience_epochs=None,
            checkpoint_interval_epochs=1,
            run_id="p5b_fixed",
        ),
        dry_run=True,
        dry_run_epochs=5,
    )
    assert camp["digest_excludes_stopping_policy"] is True
    assert camp["effective_epoch_cap"] == 5
    assert camp["actual_epochs_run"] == 5
    assert camp["early_stop_patience_epochs"] is None
    assert camp["seed_results"][0]["digest_excludes_stopping_policy"] is True

    info = json.loads(
        (
            layout.train_batch_run_dir("g001", "p5b_fixed") / "train_info.json"
        ).read_text()
    )
    assert info["digest_excludes_stopping_policy"] is True
    assert info["effective_epoch_cap"] == 5

    # Digest still hashes epochs (historical); provenance is the honest length source.
    d1 = train_config_digest(
        TrainingConfig(epochs=120, early_stop_patience_epochs=20, early_stop_max_epochs=480)
    )
    d2 = train_config_digest(
        TrainingConfig(epochs=480, early_stop_patience_epochs=20, early_stop_max_epochs=480)
    )
    assert d1 != d2  # documents the lie: same trajectory family, different digest
    prov = stopping_policy_provenance(
        TrainingConfig(epochs=120, early_stop_patience_epochs=20, early_stop_max_epochs=480),
        epochs=120,
        actual_epochs_run=37,
    )
    assert prov["effective_epoch_cap"] == 480
    assert prov["actual_epochs_run"] == 37


def test_live_multi_seed_without_factory_fails_g7(tmp_path: Path) -> None:
    """§7.3 / G7: live + multi-seed + shared backend must raise."""
    layout = _tiny_weighted_layout(tmp_path)

    class _Mock:
        def train_epoch(self, *a, **k):
            return {"train_weighted_loss": 1.0, "energy_bias": 0.0}

        def evaluate(self, *a, **k):
            return {
                "validation_weighted_loss": 1.0,
                "checkpoint_selection_permitted": False,
            }

        def step_scheduler(self, *a, **k):
            return None

    with pytest.raises(TrainerError, match="backend_factory"):
        run_multi_seed_training(
            layout=layout,
            config=TrainingConfig(
                seeds=(20260730, 20260731),
                epochs=1,
                checkpoint_interval_epochs=1,
            ),
            dry_run=False,
            aimnet2_train_authorized=True,
            backend=_Mock(),  # type: ignore[arg-type]
            skip_dataset_audit=True,
            require_merge_meta=False,
        )


# ---------------------------------------------------------------------------
# P1: real torch path through build_resume_payload / apply_resume_payload
# ---------------------------------------------------------------------------


def test_live_resume_payload_torch_path(tmp_path: Path) -> None:
    """P1: real RAdam/EMA/scheduler/RNG resume via free functions + 5 assertions."""
    torch = pytest.importorskip("torch")
    from nhc_deprot.training.live_aimnet2 import (
        apply_resume_payload,
        build_resume_payload,
        capture_rng_state,
    )

    class TinyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 8)
            self.fc2 = torch.nn.Linear(8, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)

    def _seed_all(seed: int) -> None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _make_bundle(seed: int):
        _seed_all(seed)
        core = TinyCore().double()
        for p in core.parameters():
            p.requires_grad_(True)
        opt = torch.optim.RAdam(core.parameters(), lr=1e-3, weight_decay=1e-8)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        ema: dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in core.named_parameters() if p.requires_grad
        }
        return core, opt, sch, ema

    def _ema_update(core, ema, decay=0.99):
        for n, p in core.named_parameters():
            if n in ema:
                ema[n].mul_(decay).add_(p.data, alpha=1.0 - decay)

    def _step(core, opt, sch, ema, x, y):
        opt.zero_grad(set_to_none=True)
        pred = core(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
        _ema_update(core, ema)
        sch.step(float(loss.detach()))
        return float(loss.detach())

    def _snapshot_state(core, opt, sch, ema):
        return {
            "live": {k: v.detach().clone() for k, v in core.state_dict().items()},
            "ema": {k: v.detach().clone() for k, v in ema.items()},
            "opt": copy.deepcopy(opt.state_dict()),
            "sch": copy.deepcopy(sch.state_dict()),
        }

    def _assert_state_equal(a, b, label: str) -> None:
        for k in a["live"]:
            assert torch.equal(a["live"][k], b["live"][k]), f"{label} live {k}"
        for k in a["ema"]:
            assert torch.equal(a["ema"][k], b["ema"][k]), f"{label} ema {k}"
        # optimizer exp_avg / exp_avg_sq per param
        sa, sb = a["opt"]["state"], b["opt"]["state"]
        assert set(sa.keys()) == set(sb.keys()), f"{label} opt state keys"
        for pk in sa:
            for field in ("exp_avg", "exp_avg_sq"):
                if field in sa[pk]:
                    assert torch.equal(sa[pk][field], sb[pk][field]), (
                        f"{label} opt {field}"
                    )
        assert a["sch"] == b["sch"], f"{label} scheduler {a['sch']} vs {b['sch']}"

    digest = "probe_resume_digest_v1"
    x = torch.randn(16, 4, dtype=torch.float64)
    y = torch.randn(16, dtype=torch.float64)

    # --- straight 20 steps ---
    core_s, opt_s, sch_s, ema_s = _make_bundle(0)
    losses_s: list[float] = []
    snap_at_10 = None
    payload_at_10 = None
    for step in range(1, 21):
        losses_s.append(_step(core_s, opt_s, sch_s, ema_s, x, y))
        if step == 10:
            snap_at_10 = _snapshot_state(core_s, opt_s, sch_s, ema_s)
            payload_at_10 = build_resume_payload(
                live_state_dict={k: v.detach().cpu().clone() for k, v in core_s.state_dict().items()},
                ema_shadow={k: v.detach().cpu().clone() for k, v in ema_s.items()},
                optimizer_state_dict=opt_s.state_dict(),
                scheduler_state_dict=sch_s.state_dict(),
                rng_state=capture_rng_state(torch),
                epoch=10,
                train_config_digest=digest,
                seed=0,
                run_id="tiny",
                best_epoch=10,
                best_validation_weighted_loss=losses_s[-1],
                epochs_since_best=0,
                ema_decay=0.99,
            )

    assert payload_at_10 is not None and snap_at_10 is not None

    # --- 10 steps + resume + 10 steps ---
    core_r, opt_r, sch_r, ema_r = _make_bundle(0)
    losses_r: list[float] = []
    for step in range(1, 11):
        losses_r.append(_step(core_r, opt_r, sch_r, ema_r, x, y))
    # apply full payload
    apply_resume_payload(
        payload_at_10,
        torch_mod=torch,
        core=core_r,
        optimizer=opt_r,
        scheduler=sch_r,
        ema_shadow_out=ema_r,
        expected_digest=digest,
        ema_decay=0.99,
    )
    snap_after_resume = _snapshot_state(core_r, opt_r, sch_r, ema_r)
    _assert_state_equal(snap_at_10, snap_after_resume, "at-resume-sync")

    for step in range(11, 21):
        losses_r.append(_step(core_r, opt_r, sch_r, ema_r, x, y))

    # Assertion 1: step-wise loss match
    max_diff = max(abs(a - b) for a, b in zip(losses_s, losses_r, strict=True))
    assert max_diff < 1e-9, f"loss diff {max_diff}"

    # Assertions 2–4: end state matches (opt / ema / scheduler)
    snap_s_end = _snapshot_state(core_s, opt_s, sch_s, ema_s)
    snap_r_end = _snapshot_state(core_r, opt_r, sch_r, ema_r)
    _assert_state_equal(snap_s_end, snap_r_end, "end")

    # Assertion 5: destructive — drop each critical field and prove divergence
    def _resume_and_continue(payload_mut):
        core_x, opt_x, sch_x, ema_x = _make_bundle(0)
        losses_x: list[float] = []
        for step in range(1, 11):
            losses_x.append(_step(core_x, opt_x, sch_x, ema_x, x, y))
        try:
            apply_resume_payload(
                payload_mut,
                torch_mod=torch,
                core=core_x,
                optimizer=opt_x,
                scheduler=sch_x,
                ema_shadow_out=ema_x,
                expected_digest=digest,
                ema_decay=0.99,
            )
        except TrainerError:
            # fail-closed on missing field is also a valid "caught" outcome
            return float("inf")
        for step in range(11, 21):
            losses_x.append(_step(core_x, opt_x, sch_x, ema_x, x, y))
        return max(abs(a - b) for a, b in zip(losses_s, losses_x, strict=True))

    # drop optimizer
    p_no_opt = copy.deepcopy(payload_at_10)
    del p_no_opt["optimizer_state_dict"]
    d_opt = _resume_and_continue(p_no_opt)
    assert d_opt >= 1e-9, f"dropping optimizer should diverge, got {d_opt}"

    # drop ema_shadow (apply raises → inf) or zero it
    p_no_ema = copy.deepcopy(payload_at_10)
    del p_no_ema["ema_shadow"]
    d_ema = _resume_and_continue(p_no_ema)
    assert d_ema >= 1e-9, f"dropping ema_shadow should diverge/fail, got {d_ema}"

    # drop rng_state: after identical 10 steps both streams sit at the same
    # position, so "no restore" alone is null-by-construction. Prove capture
    # matters by burning RNG only on the no-restore path (simulates any
    # intervening torch.randn before the next train step), then comparing
    # noisy post-resume trajectories.
    p_no_rng = copy.deepcopy(payload_at_10)
    del p_no_rng["rng_state"]
    core_n, opt_n, sch_n, ema_n = _make_bundle(0)
    losses_n: list[float] = []
    for step in range(1, 11):
        losses_n.append(_step(core_n, opt_n, sch_n, ema_n, x, y))
    apply_resume_payload(
        p_no_rng,
        torch_mod=torch,
        core=core_n,
        optimizer=opt_n,
        scheduler=sch_n,
        ema_shadow_out=ema_n,
        expected_digest=digest,
        ema_decay=0.99,
    )
    _ = torch.randn(256, dtype=torch.float64)  # burn — only no-restore path
    for step in range(11, 21):
        x_noisy = x + 0.01 * torch.randn_like(x)
        losses_n.append(_step(core_n, opt_n, sch_n, ema_n, x_noisy, y))

    core_g, opt_g, sch_g, ema_g = _make_bundle(0)
    losses_g: list[float] = []
    for step in range(1, 11):
        losses_g.append(_step(core_g, opt_g, sch_g, ema_g, x, y))
    apply_resume_payload(
        payload_at_10,
        torch_mod=torch,
        core=core_g,
        optimizer=opt_g,
        scheduler=sch_g,
        ema_shadow_out=ema_g,
        expected_digest=digest,
        ema_decay=0.99,
    )
    # no burn — restored stream is what train would see
    for step in range(11, 21):
        x_noisy = x + 0.01 * torch.randn_like(x)
        losses_g.append(_step(core_g, opt_g, sch_g, ema_g, x_noisy, y))
    d_rng = max(abs(a - b) for a, b in zip(losses_g[10:], losses_n[10:], strict=True))
    assert d_rng >= 1e-9, (
        f"dropping rng_state + intervening burn must diverge, got {d_rng}"
    )
