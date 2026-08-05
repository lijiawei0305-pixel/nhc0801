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

参考几何：T9 原文数值基于 `_archive_teacher_maxsteps100_frame2_20260804T144613Z/teacher_gpu_g001` 的 4 个 Val endpoint（`is_terminal`）。规范 `teacher_gpu_g001/` 已于 2026-08-05 用 m250 全轨迹重算齐（batch `g001_val_reference_m250`）；**两套参考目录仍须分开引用，不可混用**。m250 对照见 §3.1 追加段与 `refcmp_{fc2,m250}_v1`。

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

**2026-08-05 独立验证（m250 全轨迹参考）**：曾怀疑三簇结构是归档 `frame_count==2` 贫瘠参考的假象。对照实验（同 48 候选、同 CPU、同 250 步预算，**唯一变量 = 参考集**）：

| 臂 | 参考 | 产物 |
| --- | --- | --- |
| A | `_archive_teacher_maxsteps100_frame2_…`（`reference_frame_index=1`） | `pre_screen_g001/refcmp_fc2_v1/` |
| B | 规范 `teacher_gpu_g001/` m250 全轨迹终点（`frame_index` 15/26/32/39） | `pre_screen_g001/refcmp_m250_v1/` |

结果：**B 仍为 3 簇**（0.122–0.128 / 0.172–0.193 / 0.203–0.231；簇内 span 0.006–0.029 ≪ 簇间间隔）；A→B 逐候选跨簇 **0/48**；ΔRMSD 均值 **+2e−5 Å**，Δsteps **0**，Δforce **−1e−5**。原因：两套参考的 **parent 终点几何几乎重合**（端点间 Kabsch RMSD 5e−5…3.6e−4 Å），并非预筛读错路径（B 的 `reference_frame_index` 已不同于 A）。  
**⚠️ 本对照的检验力（2026-08-05 复核订正）**：它**没有**检验 basin 假说，实验设计有缺陷。

basin 归属由 AIMNet2 从**起始几何**出发的 LBFGS 弛豫决定，参考集里的终点几何只是 RMSD 的**度量靶**。而两套参考的起始几何（`frame_0000`）经核对是 **bit-identical（RMSD = 0.000e+00 Å，4/4 端点）**——两次 teacher 都从同一份 gold xyz 起算。所以：

- 弛豫输入完全相同 → 轨迹逐候选完全相同（**Δsteps = 0，48/48**，这是决定性证据）
- 落进哪个 basin **被构造性地锁死**，不可能改变
- 观测到的 ΔRMSD +2e−5 Å 只是靶点移了 1e−4 Å 造成的度量位移

**所以「三簇复现」是自动成立的，不构成对 basin 假说的支持。** 之前写的「替代解释被排除」是**过度声称**，已撤回。

本对照**实际证明**的（仍有价值，只是不是原目标）：

1. m250 全轨迹与旧 maxsteps=100 对这两个 Val 根**收敛到同一 parent 极小**（终点 Kabsch RMSD 5e−5…3.6e−4 Å）——teacher 流水线跨 maxsteps 变更的一致性验证
2. 归档 fc=2 的终点几何**没有失效**，T9 原数值的参考基础可靠
3. 规范路径下现有完整 m250 Val 参考（含全轨迹），孤儿帧陷阱已清除

**basin 假说仍未检验。** 正确的检验是**多起点扰动**：对起始几何加不同量级的随机位移后重新弛豫，看落点在多小的扰动下开始跨簇。零 DFT、纯 CPU。在该测试出结果之前，§3.1「RMSD 是 basin 标签、不可跨运行比较」**仍以跨设备实测（4/16 跳变）为唯一依据**——那条证据本身没有被推翻，故**不回调** `force → steps → rmsd` 排序键。

详情见 `docs/plans/20260805_grok_task_val_reference_m250_and_rescreen.md` 交付与本节复核。

**2026-08-05 多起点扰动检验（正确对照）**：`docs/plans/20260805_grok_task_basin_perturbation_test.md`。

| 项 | 内容 |
| --- | --- |
| 设计 | 扰动**起始几何**（逐原子高斯，RMS 位移 = ε）；参考终点固定；同 CPU、m250 Val 参考 |
| 规模 | 8 候选（4 已知跨设备脆弱 + 4 已知稳健，均 seed 20260730）× ε∈{0, 1e−5, 1e−4, 1e−3, 1e−2} × 6 rng（ε=0 仅 1 次）= **200** 次候选级筛 |
| 耗时 | 墙钟 **~600 s**（4 worker × OMP12）；单次均值 ~12 s |
| 产物 | `pre_screen_g001/basin_perturb_v1/results.jsonl`；代码 `pipeline/basin_perturbation.py` |

**内部对照**：同一候选全部复本的 `mean_force_rmse_at_reference_ev_per_a` **spread = 0**（8/8 候选）。力在固定参考几何上单点求值、与起点扰动无关——与预期一致，**模型前向无非确定性**。

**翻转阈值**（跨簇率首次 >0 的最小 ε；簇间隔 >0.01 Å，以 ε=0 所在簇为基准）：

| 组 | 候选 | 翻转 ε |
| --- | --- | --- |
| fragile | e1f100_mlp ep10 | **1e−5** |
| fragile | e1f1_mlp ep10 | **1e−5**（6/6 复本跨簇） |
| fragile | e1f1_mlp_shift ep70 | **1e−5** |
| fragile | e1f1_mlp ep70 | **1e−2** |
| robust | e1f100_mlp_shift ep10 | **1e−5** |
| robust | e1f1_mlp_shift ep10 | **1e−4** |
| robust | e1f1_mlp ep30 | **无**（至 1e−2 仍 0 跨簇） |
| robust | e1f1_mlp_shift ep30 | **无**（至 1e−2） |

**四问摘要**：

1. **阈值**：脆弱组 3/4 在 **1e−5 Å** 即跨簇；稳健组 2/4 也在 ≤1e−4 跨簇，但 **ep30 两条在 1e−2 仍稳健**。
2. **两组不完全分开**：跨设备「脆弱/稳健」标签与扰动敏感性**弱相关**——标签 robust 的 `e1f100_mlp_shift ep10` 在 1e−5 就跨簇；标签 fragile 的 `e1f1_mlp ep70` 要到 1e−2。
3. **相对浮点噪声**：1e−5 Å 远大于典型单次前向坐标噪声量级（~1e−6 相对误差累积仍远小于 0.01 Å 簇间距）。**部分候选坐在 basin 边界上**可解释「极小扰动即跨簇」；但**不能**单独用「全体 basin 边界」解释全部 4/16 跨设备跳变——因为有的候选到 1e−2 仍不跨。
4. **steps / force**：steps 复本散布大（spread 14–29 步）；force 完全不变（见上）。

**判定：basin 假说部分成立。**  
- 成立：对若干 checkpoint，ε ≈ 1e−5–1e−4 Å 即可跨簇 → RMSD 对这些候选**确实是离散 basin 标签**，不宜作主排序键。  
- 不成立为「全员脆弱」：ep30 类候选在 1e−2 Å 下仍稳。  
- 跨设备 4/16 跳变与扰动阈值**未一一对应** → 可能还有设备路径差异；**不据此回调** `force → steps → rmsd`（人拍板），也**不**宣称已穷尽 4/16 机制。

---

**⚠️ 判读订正（2026-08-06 复核）：「翻转阈值」模型不成立，但结论更强了。**

上表用的「跨簇率随 ε 递增、存在翻转阈值」这个模型**与数据不符**。按 ε 分解落入高簇 basin B 的比例：

| checkpoint | 组 | ε=0 | 1e−5 | 1e−4 | 1e−3 | 1e−2 | p(B) |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `e1f100_mlp` ep10 | fragile | 0/1 | 1/6 | 1/6 | 1/6 | 1/6 | 0.17 |
| `e1f100_mlp_shift` ep10 | robust | 0/1 | 2/6 | 2/6 | 3/6 | 1/6 | 0.33 |
| `e1f1_mlp` ep10 | fragile | **1/1** | 0/6 | 0/6 | 1/6 | 1/6 | **0.08** |
| `e1f1_mlp` ep30 | robust | 0/1 | 0/6 | 0/6 | 0/6 | 0/6 | **0.00** |
| `e1f1_mlp` ep70 | fragile | 1/1 | 6/6 | 6/6 | 6/6 | 5/6 | **0.96** |
| `e1f1_mlp_shift` ep10 | robust | 0/1 | 0/6 | 2/6 | 1/6 | 1/6 | 0.17 |
| `e1f1_mlp_shift` ep30 | robust | 0/1 | 0/6 | 0/6 | 0/6 | 0/6 | **0.00** |
| `e1f1_mlp_shift` ep70 | fragile | 1/1 | 3/6 | 2/6 | 4/6 | 4/6 | **0.54** |

**跨 ε 三个数量级，比例基本是平的，不递增。** 所以不存在「阈值」——**最小的 1e−5 Å 就已经把落点完全随机化**。正确的模型是：

> 落点是一次 **Bernoulli 抽样**，成功率 p(B) 由 checkpoint 决定，在 ε ∈ [1e−5, 1e−2] Å 上**与扰动幅度无关**。

三条推论：

1. **单次 RMSD 不是测量值，是一个样本。** `e1f1_mlp` ep10 的 ε=0 基准落在 B，而 24 个扰动复本里只有 2 个落 B（p=0.08）——**未扰动那次恰好抽到了小概率侧**。用单次结果排序，对 p 远离 0/1 的候选就是在按抛硬币排序。
2. **确实「部分成立」，但轴是 p 不是阈值。** 两个 ep30 候选 **24/24 复本全部落 A**（p=0.00），是真确定性的；`e1f1_mlp_shift` ep70 的 p=0.54，接近纯抛硬币。
3. **这解释了大部分跨设备跳变，但不是全部。** `e1f1_mlp` ep70 在 CPU 上 p(B)=0.96（近乎必然落 B），而 GPU 那次落在了 A——一个 CPU 扰动几乎复现不出的结果（p=0.04）。**这一例仍指向真实的设备路径差异，未被本实验解释。** Grok 原文「不宣称已穷尽 4/16 机制」的保留是对的。

（注：本实验的 8 个候选只触及 **2 个**簇；§3.1 开头的第三簇 0.203–0.231 没有候选落入。另：`fragile`/`robust` 分组是**按跨设备结果选的**，用它反过来验证跨设备相关性属于对结果条件化，相关系数 r=+0.25 (n=8) 不应作强解读。）

**对排序键的影响**：**不回调**，`force → steps → rmsd` 保持。理由比之前更硬——不只是「跨设备不复现」，而是**单次 RMSD 对相当一部分候选在数学上就是 Bernoulli 抽样**。

**若将来要把 RMSD 收回作可用指标**，路径已明确且便宜（单复本约 12 s）：对每个候选跑 N 个扰动复本，报 **p(basin) 或多数簇**，而不是单次距离值。这条**需要人拍板**（改的是预筛的测量口径），本文不擅自执行。

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

| 事项 | 说明 | 处理状态 |
| --- | --- | --- |
| **预筛排序规则** | 现为 `RMSD → steps → force RMSE`（`pre_screen.rank_candidates`）。§3.1/§3.3 说前两键不可跨运行比较。建议改为 **force RMSE 主序 + RMSD 只报簇归属**。排序键在代码里，**不在** `NUMERIC_CALIBRATION_V001`——本轮核对后确认无需 bump 合同 | **已处理（2026-08-05）**：排序改为 `hard → force RMSE ↑ → steps ↑ → mean RMSD ↑ → tiebreak`。**未**做 RMSD 分簇；**未**改任何合同数值。连带实现 e0 排除，见下行 |
| **e0 占短名单席位**（新，排序改动的连带后果） | 新规则下 e0 的 force RMSE 全场最低（0.060942），会排第 1 并吃掉一个短名单席位。但合同 `epoch_zero_non_regression_rule` 要的是拿 e0 作**基线**对比，不是让它当**候选**竞争 | **已处理（2026-08-05）**：实现前序计划搁置的 P0-2——`CheckpointCandidate.route_kind`（默认 `finetuned_checkpoint`；官方基座传 `epoch_zero`）。e0 照常参与排名、照常出现在 `ranked`，但不进短名单；收据新增 `epoch_zero_baseline`（含真实名次）与 `epoch_zero_excluded_from_shortlist` |
| **P2 DFT 是否继续** | §3.4 不支持「省 DFT」。是否继续跑 e0 + sci-val 由你定 | （未动） |
| **T4 措辞** | `AGENTS.md` T4 与多份 plan 写「forces_weight=100 优于 1」。本轮证据：**帧级成立、T1 几何/力指标上反向**。改措辞属科学口径 | **已处理（2026-08-05）**：历史 plan 原文保留，追加「2026-08-05 订正」块（`20260804_morning_results.md`、`20260804_ablation_phase1_force_table.md`）；`ablation_cli` 加并列说明「力主导 ≠ T1 更好」。**未**改 T4 方法规则本身 |

### 5.3 重新审视本结论的条件

任一条满足即应重跑判读：

- Train 标签根数 ≥ 300（G1）
- Val 参考换成 m250 全轨迹终点（当前规范 `teacher_gpu_g001/` 重算完成后）——**换参考集则本文全部数值作废，须重跑**  
  **（2026-08-05 已执行对照，见 §3.1 订正）**：`refcmp_fc2_v1` vs `refcmp_m250_v1` 显示 T1 三键在噪声内重合（ΔRMSD +2e−5 Å、Δsteps 0/48）。原因是两套参考的**起始几何 bit-identical**、终点几何仅差 1e−4 Å，弛豫轨迹逐候选相同。**本轮不触发「全部数值作废」**；§1/§2 主结论保持。
  ⚠️ 该对照**不能**当作 basin 假说的验证（§3.1 订正）。若未来参考几何发生 **Å 级**变化，仍须重跑。
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
