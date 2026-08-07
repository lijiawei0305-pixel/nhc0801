# NHC0801 自动化开发进度

更新：2026-08-04（sci-val 仪表作废 + P0/P1）

**必读**：
- sci-val 作废与重跑：`docs/plans/20260804_scival_instrumentation_void_and_rerun_plan.md`
- 微调定稿：`docs/plans/20260804_finetune_recipe_v1_operational.md`
- 晨间结果：`docs/plans/20260804_morning_results.md`
- 交接：`docs/plans/20260804_morning_handoff.md`

---

## 质量门（本地）

| 检查 | 结果 |
| --- | --- |
| pytest | **206 passed**, 1 skipped |
| mypy | （未重跑本轮） |
| ruff | （未重跑本轮） |

---

## 任务表

| ID | 状态 | 证据 |
| --- | --- | --- |
| M0–M12 | **DONE** | commits `455c51f`…`7cbabb4` |
| M13 | **DONE** | P-1v：frame_count=13，E_delta=0，legacy 2 帧可读 |
| M14 | **DONE** | LIVE_ABLATION_PASS；4 run × 144 `.pt`；log `m14_ablation_v2.out` |
| M14fix-pt | **DONE** | `3133350` 修复 live 写 `.pt` |
| P5.5 live 引擎 | **DONE** | `f5c0226`；live 预筛已跑 |
| live 预筛 e1f100 | **DONE** | `PRE_SCREEN_EMPTY_SHORTLIST`（neutral 全未 GAU_LOOSE） |
| Epoch-0 g001 | **BLOCKED** | Val neutral 官方 AIMNet2 亦未收敛 |
| 阶段二 300–500 root | **NOT YET** | 现 ~176 root；等全轨迹+扩标签 |

---

## Live 关键数字

| 项 | 值 |
| --- | --- |
| P-1v frame_count | 13 |
| P-1v traj lines | 12 |
| P-1v 墙钟 | ~52 min |
| M14 F_mse @f100 | ~0.34 |
| M14 F_mse @f1 | ~0.40 |
| 旧 pilot F 改善 | 仅 3.4%（0.42→0.406） |

---

## 配方结论（T1–T9）

1. **forces_weight=100** 方向正确（力误差相对 f1 明显下降）  
2. 禁止 forces_weight=10 当默认  
3. 禁止 energy loss 选模  
4. pilot 3 root 仍可能 sci-val 无增益 → T9  
5. 下一优先：真 GAU_LOOSE 预筛 + teacher 出量  

---

## 变更记录（节选）

| 时间 | 事件 |
| --- | --- |
| 08-03 | M0–M12 代码全绿 |
| 08-04 00:xx | P-1v 启动（gpu4pyscf） |
| 08-04 | M14 v1 meta-only 失败 → 修 export → v2 重跑 |
| 08-04 | P-1v PASS；M14 LIVE_ABLATION_PASS；定稿+结果文档 |
| 08-04 | sci-val 收据 void（仪表 bug）；P0 e0 预筛 rank 8/49；P1 仪表/容差/失败关闭 + 5 tests |
