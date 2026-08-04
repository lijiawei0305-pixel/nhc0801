# NHC0801 AIMNet2 微调操作方案 v1（可执行定稿）

**日期**: 2026-08-04  
**读者**: 明早验收 / 后续 live 训练与选模  
**权威顺序**: `mindmap.md` → `AGENTS.md`（T1–T9）→ `docs/prompt.md` → `docs/plans/20260803_teacher_trajectory_and_finetune_v02_plan.md` → **本文**  
**文献**: arXiv:2506.21935 *Fine-Tuning Universal Machine-Learned Interatomic Potentials*（下称 FT 教程）

本文是 **对照代码现状 + 实测证据 + 论文可行区间** 后的操作配方，**不重新推导** T1–T9。  
工程模块 M0–M12 已落地（本地质量门 175 passed / mypy 0 / ruff 0）。M13/M14 为 live 验证与消融。

---

## 0. 一句话结论

| 问题 | 答案 |
| --- | --- |
| AIMNet2 在本项目里是什么？ | **几何预优化器**（GAU_LOOSE），**不是** ΔE_deprot 能量预测器（T1 / mindmap 8–9） |
| 现在该怎么训？ | 阶段一 4 配方：`e1f1_mlp` / `e1f100_mlp` / `e1f1_mlp_shift` / `e1f100_mlp_shift` |
| 力权重为什么是 100 不是 10？ | 标签未做 E0 对齐，实测 E:F 量级 ~30:1；`forces_weight=10` 仍能量主导（T4） |
| 用什么选模型？ | **禁止**帧级 energy loss / quick-val 终选；预筛看 RMSD/步数/力误差；终选看 sci-val（T1、mindmap 6–9） |
| pilot 3 train root 期望？ | 很可能 **无增益 vs Epoch-0**；诚实报告数据不足，等 teacher 全轨迹出量（T9） |

---

## 1. 科学边界（不可破）

来自 `mindmap.md` + `AGENTS.md`：

1. 反应：`NHC-H+ → NHC + H+`；cation (+1)/neutral (0)；root 不跨 split。  
2. Parent = **P01 only**：`wb97m-d3bj` / `def2-TZVPP`，grid=4，SCF 1e-9。  
3. AIMNet2 停点：**GAU_LOOSE**（ASE fmax **0.10** eV/Å，max 250）——不是 0.05。  
4. 标签只认 parent DFT；**AIMNet2 能量永不进标签**。  
5. 训练目标：冻结 D3 残差 E/F（`P01_total − frozen two-body D3(BJ)`）。  
6. Final Test **封存**；只写 `$WJW/NHC0801`。  
7. 历史 `frame_count == 2` teacher endpoint **只读**。

---

## 2. 实测证据（已冻结，勿重推）

来源：2026-08-03 `campaign_receipt_live.json` + teacher 扫描（v02 plan §1）。

| 事实 | 数 |
| --- | --- |
| pilot 训练 | 3 seed × 200 ep ≈ **5 min / V100** |
| val 触底 | epoch **60–70**，之后 LR 空转 |
| train/val loss | ~0.085 vs ~4.48（**两数量级**）→ 记忆分子常数偏置 |
| 力几乎没学到 | F_mse 全程改善仅 **~3.4%** |
| E_mse : F_mse | ~12.4 : 0.42 ≈ **30:1** |
| teacher 中位墙钟 | **0.72 h**/endpoint；旧路径只存 2 帧 → ~90% 标注流失 |
| 训练 vs DFT 成本 | 训练极便宜，sci-val / teacher 极贵 → **消融要宽、DFT 要窄**（T2） |

**与 FT 教程的对应关系**（采用其可行区间，按本项目实测改权重）：

| 教程建议 | 本项目采用 | 原因 |
| --- | --- | --- |
| lr 1e-4–1e-3 | **1e-4** 不动 | 官方 AIMNet2 口径；一次只动一类变量（T7） |
| batch 2–20（>20 伤泛化） | **8** | 原 32 超限；小数据每 epoch 步数太少（T7） |
| epochs 视收敛 | **120** | val 60–70 触底（T7） |
| EMA 0.98–0.99999 | **0.99** | 教程默认带；可 `None` 关（T7） |
| E/F 权重 1–20 | **forces_weight ∈ {1, 100}** | 教程默认 10 **对本标签无效**（T4） |
| 必须自己算 E0 | 解冻 **`atomic_shift`** | 标签未对齐绝对总能（T3） |
| 不改基座 cutoff | **不改** | 教程硬警告 |

---

## 3. 阶段划分与成功判据

### 3.1 阶段一（现在 · pilot 数据 · 代码路径 + 信号方向）

**目的**: 打通 `train_g001/runs/<run_id>/` + 预筛；判断「力主导 / E0 对齐」是否朝正确方向动。  
**不是**最终生产配方。

**共享冻结超参**（写进每个 `train_info.json` / `train_config_digest`）：

```text
base_weight:     aimnet2_wb97m_d3_0.pt（官方 SHA 校验）
optimizer:       RAdam
lr:              1e-4
weight_decay:    1e-8
grad_clip:       0.4
scheduler:       ReduceLROnPlateau（每 epoch 显式 step，不在 evaluate 内）
batch_size:      8
epochs:          120
ema_decay:       0.99
seeds:           (20260730, 20260731, 20260732)
energy_weight:   1.0
quick_val_final: false   # 永远 false
```

**消融矩阵（唯一合法 4 行）**:

| run_id | energy_w | forces_w | 可训练 |
| --- | ---: | ---: | --- |
| `e1f1_mlp` | 1 | 1 | `^outputs\.energy_mlp\.` |
| `e1f100_mlp` | 1 | **100** | 同上 |
| `e1f1_mlp_shift` | 1 | 1 | energy_mlp + `^outputs\.atomic_shift` |
| `e1f100_mlp_shift` | 1 | **100** | 同上 |

实现入口：`src/nhc_deprot/training/ablation_cli.py` → `DEFAULT_ABLATION_MATRIX`  
CLI：`scripts/nhc0801_train_ablation.py`

**阶段一判据（§6.2，按优先级）——不看 val loss**:

1. 硬门：identity / topology / gau_loose_converged  
2. `rmsd_to_reference_angstrom`（重原子 Kabsch）↓  
3. `aimnet2_steps_to_gau_loose` ↓  
4. `force_rmse_at_reference_ev_per_a` ↓  
5. （诊断用，不终选）`atomic_shift` 开后 val E_mse ↓ 且 train/val 差收窄  

预筛模块：`pipeline/pre_screen.py`，收据强制  
`final_model_selected: false`  
`selection_authority: "pre_screen_shortlist_only_not_final"`

### 3.2 阶段二（数据 300–500 train root 后 · 定配方）

同一两轴，力权重扩到 `{1, 30, 100, 300}` 全因子 8 run（~2 GPU-h）。  
**明确不做**: 同时扫 lr/epochs/optimizer；一次解冻全网；换 backbone。

### 3.3 终选（mindmap 8–9 · 永远）

硬门通过后字典序：

1. ΔE_deprot 误差 vs Pure-DFT  
2. handoff / 拓扑通过率  
3. 力 RMSE + 相对 Epoch-0 的 parent 步数比  
4. 墙钟比  

**帧级 energy loss 在任何阶段都不得作为选择依据（T1）。**

### 3.4 诚实失败（T9）

若阶段一全部不优于 Epoch-0：

- 结论写：**数据不足，等全轨迹 teacher 出量**  
- **禁止**自行扩大解冻范围或扫 lr  
- 若仍登记 `models/v0.1`，`info.json` / card 必须注明 `no gain vs epoch-0`

---

## 4. 数据管线（teacher → 训练集）

```text
冻结几何
  → parent PySCF/geomeTRIC（gpu4pyscf worker + callback）
  → trajectory.jsonl（每次 E+梯度求值；含线搜索拒绝步）
  → frame_0000…frame_NNNN（变长；仅末帧 is_terminal）
  → D3(BJ) 两体投影（纯几何，零额外 DFT；Dftd3Projector）
  → weighted NPZ（endpoint 等质量，帧内均分）
  → train_g00N/runs/<run_id>/seed_*/epoch_NNNN.pt
  → pre_screen_g00N/<run_id>/  （零 DFT 短名单）
  → sci-val（贵；仅 2–3 候选）
  → models/v0.N/model.pt
```

| 规则 | 路径 / 行为 |
| --- | --- |
| teacher | `teacher_gpu_g00N/` |
| 训练过程 | `train_g00N/runs/<run_id>/seed_*/` |
| 发布 | `models/v0.N/model.pt`（g001→v0.1） |
| 历史 2 帧 | 只读，不重算 |
| 新帧 | 默认全轨迹；`trajectory_stride` 仅在实测步数爆炸后启用 |

**M13 验收（P-1v）**:

1. `trajectory_frame_count > 2` 且 = 求值次数  
2. 末帧能量 vs `manifest.final_energy_hartree` ≤ 1e-9 Eh  
3. 旧 `frame_count==2` 仍可读  

当前 live 验证目录：  
`runs/nhc0801-g001/teacher_p1v_m13/CTCBQPXBHHWUIV-UHFFFAOYSA-N/cation/`

---

## 5. 操作清单（服务器 · 可复制）

**环境**:

```bash
# 训练 / 编排
source /home/plab/test/WJW/env/envs/mlff.sh   # 或 conda activate mlff
export PYTHONPATH=/home/plab/test/WJW/NHC0801/src
export NHC=/home/plab/test/WJW/NHC0801

# Parent DFT worker 由代码拉起 gpupyscf，不要混栈
```

**部署（无 --delete）**:

```bash
rsync -avz --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' \
  --exclude 'runs/' --exclude 'docs/' \
  -e 'ssh -o BatchMode=yes' \
  /Users/cc/nhc-deprot/ nhc614:/home/plab/test/WJW/NHC0801/
```

**阶段一训练（占 1 张空闲卡 ~30 min）**:

```bash
# 先 claim / 确认 GPU 空闲（勿占 VASP 所在 2/3/4/6 若他人作业）
cd $NHC
# 基座权重路径按机器实际 aimnet2_wb97m_d3_0.pt
python -u scripts/nhc0801_train_ablation.py \
  --nhc0801-root $NHC \
  --generation-id nhc0801-g001 \
  --batch-id g001 \
  --live --aimnet2-train-authorized \
  --base-weight /path/to/aimnet2_wb97m_d3_0.pt
# 产物: runs/nhc0801-g001/train_g001/runs/<run_id>/
```

**预筛（零 DFT）**:

```bash
python -u scripts/nhc0801_pre_screen.py \
  --nhc0801-root $NHC \
  --generation-id nhc0801-g001 \
  --batch-id g001
# 产物: pre_screen_g001/<run_id>/screen_campaign.json
```

**汇总表**:

```bash
python -u scripts/nhc0801_ablation_table.py \
  --nhc0801-root $NHC \
  --generation-id nhc0801-g001 \
  --batch-id g001 \
  -o runs/nhc0801-g001/logs/ablation_table_phase1.md
```

**sci-val（仅短名单 2–3 个；需 Epoch-0 收尾 + live 授权）**:  
按既有 `sci_val` campaign 路径；**不得**用 quick-val 或 energy MSE 挑最终模型。

---

## 6. 工程落地状态（对照 prompt M0–M14）

| ID | 状态 | 说明 |
| --- | --- | --- |
| M0 | **DONE** | ruff/mypy 配置 + 基线清零 |
| M1 | **DONE** | worker `trajectory_out_path`；缺省兼容 |
| M2 | **DONE** | live_teacher 消费 JSONL；默认不二次 first_gradient |
| M3 | **DONE** | 变长帧 + `trajectory_stride` |
| M4 | **DONE** | injectable `Dftd3Projector` |
| M5 | **DONE** | run_id / EMA / batch8 / epochs120 |
| M6 | **DONE** | B=1 力项缩放修复 |
| M7 | **DONE** | 多 regex + EMA + step_scheduler + digest |
| M8 | **DONE** | `train_g00N/runs/<run_id>/` |
| M9 | **DONE** | layout helpers |
| M10 | **DONE** | 零 DFT pre_screen |
| M11 | **DONE** | 三脚本薄封装 |
| M12 | **DONE** | e2e dry-run 集成测 + RETRO/PHASE |
| M13 | **进行中** | 已 rsync；P-1v 单 endpoint 在 GPU0 跑轨迹 |
| M14 | **待跑** | 4×3×120 live 消融 + 预筛 |

**已知偏差（写进偏差表，不写错代码去「对齐」过时计划）**:

| 项 | 处理 |
| --- | --- |
| 计划写改 `teacher_frames.py` | 不改（legacy）；变长在 teacher_runner |
| `forces_weight=10` | **永不作为本项目默认** |
| shortlist 默认扫旧 `seed_*` | 预筛/训练已用 `runs/<run_id>`；短名单若扫旧路径需传 run 目录 |
| live_epoch0 无 trajectory kwarg | M2 经 `_call` 注入 worker payload |
| ablation live backend | 须注入 `LiveAimnet2TrainBackend`（接线中） |

---

## 7. 推荐阅读顺序（明早 10 分钟）

1. 本文 §0–§3（配方与判据）  
2. `AGENTS.md` →「模型训练注意事项」T1–T9  
3. `DEV_PROGRESS.md`（任务勾选 + P-1v 数字）  
4. 若 P-1v 完成：`teacher_p1v_m13/.../manifest.json` + `trajectory.jsonl` 行数  
5. 若 M14 完成：`logs/ablation_table_phase1.md`  

---

## 8. 明确禁止清单

1. 用 val energy loss / quick-val 选最终模型  
2. 照搬 FT 教程 `forces_weight=10` 并得出「力权重无用」  
3. 为省训练时间只跑 1–2 个配方（成本模型倒置）  
4. 未做 E0/shift 就先加正则、减 epoch、解冻全网  
5. 改 Parent P01 / GAU_LOOSE / 基座 cutoff / Final Test  
6. 重算或改写已完成的 `frame_count==2` teacher  
7. kill daemon / 写 `$WJW` 非 NHC0801 / `--delete` rsync  

---

## 9. 交付检查表（明早）

- [ ] 本地 `pytest` / `mypy` / `ruff` 全绿  
- [ ] P-1v：`frame_count>2`、能量一致、legacy 2 帧可读  
- [ ] （尽量）阶段一 4 run 落盘 `train_g001/runs/*`  
- [ ] （尽量）预筛表 + 相对 Epoch-0 的一页结论  
- [ ] 若无增益：书面 T9 结论，不刷版本号装成功  

**计划状态**: 操作定稿 v1（2026-08-04）。  
**与 v02 plan 关系**: 实现与判据以 v02 + AGENTS T1–T9 为准；本文补「对照代码现状的可执行步骤」与阶段交付检查。
