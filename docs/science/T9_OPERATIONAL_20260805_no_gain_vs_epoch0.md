# Operational T9 · g001：微调在 T1 指标上不优于 Epoch-0

**日期**: 2026-08-05
**范围**: 训练批 `g001`（3 个 train root）· 阶段一 2×2 消融 · 零-DFT 预筛
**对应 mindmap**: 7（预筛）——**不是** 8–9（sci-val 终选）
**权威**: `mindmap.md` → `docs/contracts/*` → `AGENTS.md`（T1/T2/T9）→ 本文
**性质**: **操作级结论（operational T9）**，用于决定「要不要继续调配方」。**不是** sci-val 印章，**不构成**发布或选模决定。

---

## 1. 结论

在 g001 的 **3 个 train root** 规模上，阶段一 2×2 消融的 **144 个 checkpoint**，在 T1 认可的指标上**没有可复现的、优于 Epoch-0 官方基座的增益**；且在唯一可跨运行比较的连续量上，**微调从第一个 checkpoint 起就单调变差**。

**处置**：
- 停止在 g001 上继续调配方（力权重 / 解冻范围 / epochs / EMA / SWA）
- 不以「有增益」为由发布 `models/v0.1`；若仍要登记，`info.json` 与 card **必须**写 `no gain vs epoch-0 (pre_screen T1 metrics, g001)`
- 回到 **T9 主线：等 teacher 出量**（阶段二闸门 G1 = Train 标签根数 ≥ 300）

**这条结论的依据是「无可复现增益」，不是「e0 在三键上全面获胜」。** 后者不成立，见 §3.4 —— 这个区别决定了 §5 里哪些事**不能**顺带宣布。

---

## 2. 证据

### 2.1 产物

| 产物 | 内容 |
| --- | --- |
| `train_g001/runs/{e1f1,e1f100}_{mlp,mlp_shift}/` | 4 配方 × 3 seed × 12 epoch = **144 个 `.pt`** |
| `pre_screen_g001/live_phase1_v002/` | 48 候选（GPU，250 步） |
| `pre_screen_g001/e0_baseline_v001_gpu/` | e0 官方基座，**同设备同参考** |
| `pre_screen_g001/live_seed730_epoch_axis_v1/` | seed 730 全 epoch 轴 48 候选（CPU，250 步）+ `epoch_curve.json` + `paired_recipe_contrast.json` |

参考几何：`_archive_teacher_maxsteps100_frame2_20260804T144613Z/teacher_gpu_g001` 的 4 个 Val endpoint（2 root × cation/neutral），`is_terminal` 终点几何。规范 `teacher_gpu_g001/` 正在被 m250 重算，**两套参考不可混用**。

### 2.2 力误差：唯一可跨运行比较的量，且微调单调变差

`mean_force_rmse_at_reference_ev_per_a` 在参考几何上单点求值，不经弛豫，跨设备重跑偏差仅 2–8% 且无簇跳变（§3.1）。

seed 730 全 epoch 轴（CPU）：

| run_id | ep10 | ep20 | ep30 | ep50 | ep70 | ep90 | ep120 | argmin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `e1f100_mlp` | **0.0653** | 0.1571 | 0.2904 | 0.3642 | 0.3699 | 0.3743 | 0.3768 | **ep10** |
| `e1f100_mlp_shift` | **0.0653** | 0.1571 | 0.2903 | 0.3641 | 0.3699 | 0.3743 | 0.3769 | **ep10** |
| `e1f1_mlp` | **0.0738** | 0.1118 | 0.1472 | 0.1519 | 0.1827 | 0.2041 | 0.2182 | **ep10** |
| `e1f1_mlp_shift` | **0.0738** | 0.1117 | 0.1475 | 0.1516 | 0.1795 | 0.1999 | 0.2133 | **ep10** |

- **4/4 配方 argmin 在最早采样点**；`f100` 两条严格单调递增
- 相对 e0（0.06094，同参考）：**ep10 ≈ 1.05–1.19×（噪声量级）；ep20 起 2.6×；ep120 达 3.6–6.2×**
- ep10 最好只因为它是最早采样点 → 外推最优点在 **epoch 0**，即不微调

### 2.3 配方之间：差别是退化速率，不是质量

配对（同 `seed × epoch` 内比，消掉 seed block 效应）—— `paired_recipe_contrast.json`：

| 对比 | RMSD | force RMSE | seed 一致 |
| --- | ---: | ---: | :---: |
| `e1f1_mlp` vs `e1f100_mlp` | f1 好 27% | f1 好 54% | 3/3 |
| `e1f1_mlp` vs `e1f1_mlp_shift` | +0.12%（无） | +1.96% | 3/3 |

- **`forces_weight=100` 退化更快**：ep10 时 f100 反而略好（0.0653 vs 0.0738），ep30 就到 0.29 而 f1 到 ep120 才 0.218。力权重放大力梯度 → 更快离开基座 → 更快变坏
- **`atomic_shift` 解冻无效**（0.12%）。已核对 `trainable_parameter_count`，是真解冻、真没用，不是 T8 的正则静默失效
- **seed 级独立单位 n=3**，双侧符号检验下限 **p=0.25** —— 3 个 seed 无论多一致都到不了 p<0.05。看方向与效应量，不看 p

---

## 3. 这次同时暴露的口径问题（影响结论的写法）

### 3.1 `mean_rmsd` 是 basin 量化的，不可跨设备复现

同权重、同参考、同 250 步预算，仅 CUDA→CPU：**16 个重叠候选中 4 个（25%）RMSD 跳变约 0.05 Å（相对 ~40%）**，`e1f100_mlp ep10` 与 `e1f1_mlp ep10` 直接互换。seed 730 的 64 个观测聚成 **3 个离散簇**（0.122–0.128 / 0.171–0.193 / 0.203–0.231）。

RMSD 是 LBFGS 弛豫**之后**量的，落进哪个局部极小是离散选择。**它是 basin 标签，不是连续质量度量。**

跨设备偏移（CPU−GPU，16 重叠候选）：

| 指标 | 均值 | 中位 | 范围 |
| --- | ---: | ---: | --- |
| RMSD | +0.0096 | +0.0038 | −0.048 … +0.058 |
| steps | +4.64 | +3.25 | −15.75 … +17.00 |
| force RMSE | +0.0060 | +0.0014 | −0.0026 … +0.0164 |

### 3.2 连带订正

此前「seed 方差 >> 配方方差、前 9 名全是 seed 730」的读数，实为 **basin 归属分层**，不是平滑的配方效应。`live_phase1_v002` 的 RMSD 排名约 1/4 是设备相关的。

### 3.3 `mean_aimnet2_steps` 的复现性最差

范围 −15.75…+17（值本身 ~100–128，即 ±13%），且有 +4.6 的系统性设备偏移。**跨运行比较 steps 前必须同设备。**

### 3.4 e0 **没有**在三键上全面获胜——所以不能照抄旧触发条件

同设备（GPU）、同参考，e0 vs `live_phase1_v002` 的 48 个 FT：

| T1 键 | e0 | FT 最好 | 优于 e0 的 FT 数 |
| --- | ---: | ---: | ---: |
| RMSD | 0.16894 | 0.12181 | **7 / 48** |
| steps | 124.00 | 101.50 | **48 / 48** |
| force RMSE | 0.06094 | 0.06385 | **0 / 48** |

`20260804_scival_instrumentation_void_and_rerun_plan.md` §5 给的触发条件是「**e0 在预筛 T1 三键上不劣于全部 FT 候选**」。按字面：**不成立**（steps 全输、RMSD 输 7 个）。

本文因此**不使用**那条触发条件，改用 §1 的表述（「无可复现增益 + 单调退化」），理由：

1. **RMSD 不能当排序键**（§3.1），赢它的 7 个是 basin 归属
2. **steps 的优势与质量反向**：FT 用 ~101–128 步到 GAU_LOOSE（e0 用 124），但力误差差 3–6 倍——**更快收敛到更差的点不是增益**；且 steps 复现性最差（§3.3），22.5 步的差距只有复现范围的 1.3 倍
3. **force RMSE 是唯一站得住的连续量**，e0 在 48 个里排第一；但对 ep10 的领先（0.0029）落在设备噪声量级，**真正无歧义的是 ep≥20 的 2.6–6.2 倍**

---

## 4. 这份结论**没有**证明什么

| 没有证明 | 为什么 |
| --- | --- |
| FT 在完整 sci-val 下失败 | 预筛量不到 **parent DFT 步数比 / ΔE_deprot 标签误差**，那是步 8–9 |
| e0 应当发布为 v0.1 | 选模权威在 sci-val，预筛不发印章 |
| `forces_weight=100` 在任何数据规模上都差 | 只证明在 **3 train root** 上退化更快 |
| `atomic_shift` 不该解冻 | 只证明 3 root 上分不出效果；T3 的科学理由未被推翻 |
| EMA / SWA 无用 | 本轮未做 SWA；EMA 在所有 run 上都开着，是共同条件不是变量 |
| P2 的 DFT 可以省 | §3.4 的触发条件不成立 → **不能**据此宣布省 DFT |

---

## 5. 处置

### 5.1 立即生效

1. **冻结 g001 的训练配方**：不再扫力权重 / 解冻范围 / epochs / lr
2. **不开阶段二**（G1 未达标：Train 根数 3 << 300）
3. `models/v0.1` 若登记，`info.json` + card 写 `no gain vs epoch-0 (pre_screen T1 metrics, g001)`
4. 主线回到 **teacher 全轨迹出量**

### 5.2 需要用户拍板（本文不擅自执行）

| 事项 | 说明 |
| --- | --- |
| **预筛排序规则** | 现为 `RMSD → steps → force RMSE`（`pre_screen.rank_candidates`）。§3.1/§3.3 说前两键不可跨运行比较。建议改为 **force RMSE 主序 + RMSD 只报簇归属**。这改的是 `NUMERIC_CALIBRATION_V001` 口径 → **需授权 + 版本号** |
| **P2 DFT 是否继续** | §3.4 不支持「省 DFT」。是否继续跑 e0 + sci-val 由你定 |
| **T4 措辞** | `AGENTS.md` T4 与多份 plan 写「forces_weight=100 优于 1」。本轮证据：**帧级成立、T1 几何/力指标上反向**。改措辞属科学口径 |

### 5.3 重新审视本结论的条件

任一条满足即应重跑判读：

- Train 标签根数 ≥ 300（G1）
- Val 参考换成 m250 全轨迹终点（当前规范 `teacher_gpu_g001/` 重算完成后）——**换参考集则本文全部数值作废，须重跑**
- 预筛排序规则按 §5.2 改版
- 出现 ep < 10 的 checkpoint（当前最早采样点就是 ep10，最优点在其左侧未被采样）

---

## 6. 复现命令

```bash
# 1) seed 730 全 epoch 轴（CPU，零 GPU 占用；~26 min）
ARCH=$NHC/runs/nhc0801-g001/_archive_teacher_maxsteps100_frame2_20260804T144613Z/teacher_gpu_g001
OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  $WJW/env/conda/mlff/bin/python scripts/nhc0801_pre_screen.py --live \
  --candidates-json runs/nhc0801-g001/pre_screen_g001/seed730_full_epoch_axis.json \
  --device cpu --screen-id live_seed730_epoch_axis_v1 \
  --nhc0801-root $NHC --teacher-batch-dir $ARCH

# 2) 配对对比 + epoch 曲线（零 GPU）
PYTHONPATH=src python3 scripts/nhc0801_paired_contrast.py \
  runs/nhc0801-g001/pre_screen_g001/live_seed730_epoch_axis_v1/screen_campaign.json
```

**复现校验**：跨设备重跑必须先比对重叠候选。RMSD 若有簇跳变属已知现象（`RETRO.md` → `R-prescreen-rmsd-basin-quantized`），力误差应在 2–8% 内一致。

---

## 7. 相关

- `RETRO.md` → `R-prescreen-rmsd-basin-quantized`（口径）、`R-ema-export-cpu-alias`（导出审计）
- `docs/agent/training_t1_t9.md` → T1 / T2 / T9
- `docs/plans/20260805_paired_recipe_contrast_and_epoch_sweep_plan.md` → 本轮计划
- `docs/plans/20260804_scival_instrumentation_void_and_rerun_plan.md` §5 → 旧触发条件（本文 §3.4 说明为何不采用）
- 代码：`pipeline/paired_recipe_contrast.py`、`pipeline/pre_screen.py`（`teacher_batch_dir` 只读钉参考）
