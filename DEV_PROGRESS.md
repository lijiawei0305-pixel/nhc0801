# NHC0801 自动化开发进度

> 本文件由 **Master Agent** 独占读写。科学/算力进度在 `progress.md`，两者互不覆盖。
> 起始 Prompt：`docs/prompt.md`。任务定义：`docs/plans/20260803_teacher_trajectory_and_finetune_v02_plan.md` §3。

更新：2026-08-03T15:40:00Z
当前轮次：1

---

## 质量门

| 检查 | 命令 | 最近结果 | 轮次 |
| --- | --- | --- | ---: |
| 单元测试 | `PYTHONPATH=src python -m pytest -q` | 92 passed, 1 skipped | 基线 |
| 静态类型 | `mypy src tests scripts` | **30 errors / 17 files**（未配置；prompt 记 12 为旧数） | 基线 |
| 代码规范 | `ruff check src tests scripts` | **31 errors**（28 可自动修） | 基线 |

**验收定义**：三项全绿（0 error）才允许把任务标 DONE。

---

## 任务表

状态：`TODO` → `DOING` → `REVIEW`（Worker 交付，等 Master 跑门） → `DONE` / `FAILED` / `BLOCKED`

| ID | 模块 | 独占文件 | 依赖 | 状态 | Worker | 轮次 | 备注 |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| **M0** | 静态检查基线清理 | `pyproject.toml` + 存量违规文件 | — | DOING | worker-m0-r1 | 1 | 清 mypy 30 + ruff 31；配置固化 |
| **M1** | parent worker 全轨迹 callback | `scripts/nhc0801_pyscf_parent_worker.py`, `tests/test_parent_worker_trajectory.py` | M0 | TODO | — | — | **最高优先级**，数据持续流失 |
| **M2** | live_teacher 消费轨迹 | `src/nhc_deprot/pipeline/live_teacher.py`, `tests/test_live_teacher_trajectory.py` | M1 | TODO | — | — | 含删除冗余 `first_gradient` |
| **M3** | teacher_runner 变长帧 | `src/nhc_deprot/pipeline/teacher_runner.py`, `tests/test_teacher_runner_variable_frames.py` | M2 | TODO | — | — | `teacher_frames.py` **不改**（legacy 只读） |
| **M4** | d3_projection live 接线 | `src/nhc_deprot/pipeline/d3_projection.py`, `tests/test_d3_projection_live.py` | M0 | TODO | — | — | 阻塞项：新帧进不了训练集 |
| **M5** | training/config 新旋钮 | `src/nhc_deprot/training/config.py`, `tests/test_training_config.py` | M0 | TODO | — | — | run_id / EMA / batch 8 / epochs 120 |
| **M6** | weighted_loss 单样本修复 | `src/nhc_deprot/training/weighted_loss.py`, `tests/test_weighted_loss.py` | M0 | TODO | — | — | B1，差 3 倍 |
| **M7** | live_aimnet2 多 regex + EMA | `src/nhc_deprot/training/live_aimnet2.py`, `tests/test_live_aimnet2_config.py` | M5, M6 | TODO | — | — | 含 B2/B3 修复 |
| **M8** | multi_seed_trainer run 子目录 | `src/nhc_deprot/training/multi_seed_trainer.py`, `tests/test_multi_seed_trainer_runs.py` | M7, M9 | TODO | — | — | — |
| **M9** | layout run/pre_screen 路径 | `src/nhc_deprot/generation/layout.py`, `tests/test_layout_run_dirs.py` | M0 | TODO | — | — | 旧路径保留只读 fallback |
| **M10** | pre_screen 新模块 | `src/nhc_deprot/pipeline/pre_screen.py`, `tests/test_pre_screen.py` | M9 | TODO | — | — | 零 DFT；含 Kabsch |
| **M11** | scripts 薄封装 | `scripts/nhc0801_train_ablation.py`, `scripts/nhc0801_pre_screen.py`, `scripts/nhc0801_ablation_table.py` | M8, M10 | TODO | — | — | 逻辑必须在 `src/` |
| **M12** | 集成验收 | — | M1–M11 | TODO | — | — | 全量回归 + dry-run 端到端 |
| **M13** | 部署 + P-1v live 验证 | — | M12 | TODO | — | — | 1 个 endpoint，验 `trajectory_frame_count > 2` |
| **M14** | P1 live 消融 + 预筛 | — | M13 | TODO | — | — | 4 run × 3 seed；`aimnet2_train_authorized` 已开 |

**并行组**（依赖满足后可同轮派发）：
`{M1, M4, M5, M6, M9}` → `{M2, M7, M10}` → `{M3, M8}` → `{M11}` → `M12` → `M13` → `M14`

---

## 变更记录

| 轮次 | 时间 | 事件 |
| ---: | --- | --- |
| 0 | 2026-08-03T15:20:00Z | 初始化。基线：pytest 92 passed / mypy 12 err / ruff 31 err |
| 1 | 2026-08-03T15:40:00Z | Master 开工。实测基线 mypy **30**/17（非 prompt 的 12）。派 Worker M0 |
