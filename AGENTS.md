# AGENTS.md — NHC0801 / nhc-deprot

Mindmap-driven AIMNet2 fine-tune on **Parent-Level P01** teacher frames.  
Local: `/Users/cc/nhc-deprot`. Server writes: **only** `$WJW/NHC0801`.

Not a continuation of `nhc-deprot-ranker` Phase 9B. Not production `two_endpoint`.

---

## Before any code change

1. Read **`mindmap.md`** + this file + **`PHASE_STATUS.md`**.
2. If the task touches **CPU/GPU 波次、10 endpoints、teacher/e0 调度、双代数据归属**：read **`docs/contracts/COMPUTE_DISPATCH_V001.md`** (mandatory).
3. If the task touches **核数/绑核/内存 profile**：read **`docs/contracts/RESOURCE_SCHEDULING_V001.md`** + **`docs/contracts/RESOURCE_PROFILES_V002.yaml`**.
3b. If the task touches **微调超参 / 损失权重 / 解冻范围 / 选模排序 / teacher 落帧**：read **「模型训练注意事项」T1–T9**（本文件）。那里是实测结论，**不要重新推导**。
4. Skim **`RETRO.md`** for similar past failures (same category).
5. Map work to a **mindmap step 0–12**. Do not skip gates.
6. If the task is non-trivial: write a short plan under **`docs/plans/`** *before* vibe coding.
7. Put new files only where **Where to put files** allows.
8. Run `PYTHONPATH=src python -m pytest -q` after code changes.
9. Live chemistry / training / Final Test only with **explicit user authorization**.

Conflict order: **mindmap.md** → **`docs/contracts/COMPUTE_DISPATCH_V001.md`** (compute dispatch) → other frozen YAML/JSON contracts → this file → `RETRO.md` lessons → pilot evidence.

---

## Where to put files（落盘硬规则）

| 内容 | 目录 | 命名 / 说明 |
| --- | --- | --- |
| **可 import 库代码** | `src/nhc_deprot/` | 包内模块；按域：`contracts/` `data/` `pipeline/` `training/` `resources/` |
| **单元测试** | `tests/` | `test_<topic>.py`；合成 fixture，不依赖 HPC |
| **CLI / 作业入口** | `scripts/` | 薄封装 only；逻辑在 `src/`。见 `scripts/README.md` |
| **动手前的规划** | `docs/plans/` | `YYYYMMDD_<topic>_plan.md`。见 `docs/plans/README.md` |
| **冻结合同** | `docs/contracts/` | 版本化 YAML/JSON/MD（GAU_LOOSE、数值标定、**COMPUTE_DISPATCH**、RESOURCE_* 等） |
| **科学讨论 / 阻塞说明** | `docs/science/` | 非合同正文 |
| **pilot 证据（改名后）** | `docs/evidence/` | 只读绑定；勿再堆 phase9b 文件名 |
| **禁止栈 / 历史干扰** | `docs/archive/` | 只读隔离，不当协议源 |
| **阶段勾选** | `PHASE_STATUS.md` | 根目录；完成一步更新一行 |
| **工程踩坑复盘** | `RETRO.md` | 根目录；格式见该文件维护规则 |
| **科学流水线真值** | `mindmap.md` | 根目录；改口径先改它 |
| **协作者入口** | `README.md` | 中文学术说明；勿替代 mindmap |
| **私人配置** | `configs/*.local.yaml`, `private/` | **永不提交** |
| **运行产物** | `runs/`（服务器 `$WJW/NHC0801/runs/<generation>/`） | generation 用 `nhc0801-g001`；**分子组产物**见下节 **Experimental data naming** |
| **资源档案** | `docs/contracts/RESOURCE_PROFILES_V002.yaml` | 默认 trial `auto_fill_112_t10_r12_v1`（见 SCHEDULING）；V001 保留兼容 |
| **计算调度合同** | `docs/contracts/COMPUTE_DISPATCH_V001.md` | 10-ep 波次、分子组 g00N、路径隔离 |
| **命名词典** | `docs/NHC0801_命名与进度指南.md` | 对人说法 + 查进度；与本节一致 |
| **临时草稿** | **禁止长期放仓库根** | 用完删，或迁入 `docs/plans/` / `RETRO.md` |

**禁止：**

- 在仓库根目录乱放 `tmp_*.py`、未命名 `plan.md`、大段实验脚本  
- 把库逻辑只写在 `scripts/` 而不进 `src/`  
- 把规划写进 `src/` 或把合同写进 `RETRO.md`  
- 修改 ranker / science-pilot 源树或 `$WJW` 非 NHC0801 路径  
- **实验数据乱命名**（见下节）：禁止新写 `teacher/`、`teacher_gpu_side/`、顶层特殊 `epoch0/`、对外称「Autofill 批」  

---

## Experimental data naming（实验计算数据命名 · 硬规则）

**原则：** 每一次实验计算产物必须 **组号清晰、目录与组号同构、对人说法与磁盘一致**。  
**实现入口：** `src/nhc_deprot/generation/layout.py`（`teacher_batch_dir` / `epoch0_batch_dir` / `train_batch_dir`）。  
**词典：** `docs/NHC0801_命名与进度指南.md`（本地；可能未上传 GitHub）。

### 1) 分子组 `g00N`（唯一用户可见批名）

| 项 | 规则 |
| --- | --- |
| 组名 | **`g001`、`g002`、`g003`、…** 顺序编号，禁止自造别名当产品名 |
| 规模 | **5 molecular roots** = **3 Train + 2 Val**（InChIKey 字典序：前 3 训、后 2 验） |
| 端点 | **10** = 5 × (cation + neutral) |
| 对人说 | **「g003 teacher」** / **「g003 Epoch-0」** / **「g003 train」** — 禁止说「Autofill 第 3 批」「扩展批」「侧线」当正式名 |

### 2) 产物目录（规范 · 禁止例外）

| 工作 | 标准名（对人） | **唯一规范磁盘路径**（在 `runs/<generation>/` 下） |
| --- | --- | --- |
| 老师帧 | **g00N teacher** | **`teacher_gpu_g00N/`** |
| Epoch-0 基线 | **g00N Epoch-0** | **`epoch0_val_batches/g00N/`**（其下可有 `epoch0/`、`logs/`） |
| AIMNet2 训练过程 | **g00N train** | **`train_g00N/runs/<run_id>/`**（按配方分 run，其下再按 seed/epoch） |
| 零-DFT 预筛（步骤 7 加强） | **g00N 预筛** | **`pre_screen_g00N/<run_id>/`** |
| **发布模型版本** | **v0.1 / v0.2 …** | **`models/v0.1/`**：`model.pt` + `info.json` + **`card.svg`/`card.json`** |
| generation 总目录 | — | **`runs/nhc0801-g001/`**（勿简写成 `runs/g001/`） |

**强制同构示例（小白一眼能对上号）：**

```text
teacher_gpu_g001/     ← 第 1 组的老师数据
train_g001/           ← 第 1 组训练过程（很多中间 .pt）
models/
  v0.1/               ← 发布版 0.1（选模后的正式模型）
    model.pt
    info.json
  v0.2/
    model.pt
    info.json
```

**两层不要混：**

| 层 | 是什么 | 路径 |
| --- | --- | --- |
| 训练过程 | 某个 seed、某轮 epoch 的草稿权重 | `train_g001/seed_…/epoch_0200.pt` |
| **发布版本** | 对外/对下游使用的正式模型 | **`models/v0.1/model.pt`** |

### 训练组 → 发布版本（固定顺序 · 必记）

| 训练过程 | 发布版本 | 磁盘 |
| --- | --- | --- |
| **`train_g001/`** | **v0.1** | `models/v0.1/model.pt` |
| **`train_g002/`** | **v0.2** | `models/v0.2/model.pt` |
| **`train_g00N/`** | **v0.N** | `models/v0.N/model.pt` |

- g001 训完 → 发 **v0.1**；g002 训完 → 发 **v0.2**；依此类推（N = 组号）。  
- **禁止** g001 发成 v0.2，或 g002 发成 v0.1。  
- 代码：`default_model_version_for_train_batch("g001") == "v0.1"`；`register_model_version` 默认按此校验。

对人只说 **v0.1**，不要说一长串英文文件名。  
`info.json` 里记下：来自哪个 `train_g00N` / seed / epoch、权重 sha256。
**训练过程目录（`train_g00N/`）：**

```text
train_g001/
  runs/
    e1f100_mlp_shift/          ← 配方 run_id（必须有；禁止裸 seed_*）
      train_info.json          ← run_id + train_config_digest + 数据 manifest SHA
      train_result.json
      logs/train_g001.out
      seed_20260730/
        seed_result.json
        epoch_0005.pt
        epoch_0005.meta.json
        epoch_0120.pt
        epoch_0120.meta.json
    e1f1_mlp/
      ...
```

**`run_id` 命名**：`e<energy_weight>f<forces_weight>_<可训练范围>`，如 `e1f100_mlp_shift`。
同一 `train_g00N` 下多个 run 并存是**正常**的；`models/v0.N/model.pt` 只能来自其中**胜出 run 的唯一 (seed, epoch)**。

**发布版本目录（强制短名）：**

```text
models/v0.1/
  model.pt       ← 唯一权重名
  info.json
  card.json      ← 特征数据（必出）
  card.svg       ← 发布特征图（必出；每发一版更新）
models/v0.2/
  model.pt
  info.json
  card.json
  card.svg
```

**每发布一版必须生成特征卡片**（`model_card.write_model_card` / `scripts/nhc0801_render_model_card.py`）。  
卡上特征：身份、化学/DFT 参考、训练来源、帧级 E/F（仅筛选）、科学路线指标（ΔE_deprot、相对 e0 步数/墙钟、handoff/拓扑通过率）、SHA 前缀。
| 正确 | 错误 |
| --- | --- |
| `models/v0.1/model.pt` | `aimnet2_wb97m_ft_g001_seed_…_epoch_0200.pt` |
| 版本号 `v0.1` / `0.1` | `best.pt`、`latest.pt`、`final_finetuned.pt` |
| `train_g001/…/epoch_0200.pt`（过程） | 把过程文件直接当发布名到处拷 |

旧 pilot 的 **`train/`**、**`train_batches/`**：**只读**；新训练写 **`train_g00N/`**；新发布写 **`models/vX.Y/`**。

### 3) 目录 / 收据内部字段

- 训练收据必须可定位组号：含 **`batch_id`**（`g00N`）和/或写在规范路径下。  
- 发布 `info.json` 必须含 **`version`**（`v0.1`）及来源 checkpoint 路径。  
- 禁止同一科学对象多套「官方」路径并存写；兼容旧路径时 **只允许只读 fallback**。  
- 日志/回执文件名优先带组号；发布模型只用短版本号，避免 `02c` / `side` 当唯一身份。
### 4) 工程模块名 vs 产品名（必须分清）

| 工程标识（可留在代码/state） | 对外 / 产品命名（必须用 g00N） |
| --- | --- |
| 模块 `gpu_autofill`、目录 `gpu_autofill/`、脚本 `*_gpu_autofill_daemon.py` | **g00N teacher** / 切组队列 |
| 资源 profile 名 `auto_fill_112_*` | 资源档案名，**不是**分子组名 |
| 历史 `$WJW/data/runs/autofill_*_v001` | **只读旧 pilot 帧**，新 NHC0801 产物禁止新写该布局 |

### 5) 废除名（不得再作为规范产品路径或对人主称）

| 废除 | 改用 |
| --- | --- |
| `teacher/`（单独） | `teacher_gpu_g001/` |
| `teacher_gpu_side/` | `teacher_gpu_g002/` |
| `teacher_cpu/`、`teacher_gpu/`（无组号） | `teacher_gpu_g00N/` |
| 顶层 `epoch0/` 作为 g001 专用规范位 | `epoch0_val_batches/g001/` |
| 顶层 `train/` 或 `train_batches/` 作为新训练规范位 | `train_g00N/runs/<run_id>/` |
| `train_g00N/seed_*/`（无 `run_id` 的裸 seed 目录） | `train_g00N/runs/<run_id>/seed_*/`（旧路径只读兼容） |
| 过程权重：`best.pt` / `latest.pt` | `epoch_NNNN.pt`（四位轮次） |
| 发布权重：长英文后缀文件名 | **`models/v0.1/model.pt`** |
| 「Autofill 批 / 扩展批 / 侧线 e0」 | **g00N** / **g00N Epoch-0** / **g00N train** / **v0.1** |
| `runs/g001/` | `runs/nhc0801-g001/` |

### 6) 写代码 / 写文档时的自检

1. 训练过程是否在 **`train_g00N/`**？发布是否在 **`models/vX.Y/model.pt`**？  
2. 对人是否只说 **v0.1**，而不是一长串英文文件名？  
3. 过程 checkpoint 是否为 **`seed_<数字>/epoch_NNNN.pt`**？  
4. teacher / Epoch-0 是否仍用 **`teacher_gpu_g00N/`**、**`epoch0_val_batches/g00N/`**？  
5. 是否零 Autofill / 零 side / 零把 `best.pt` 当发布名？

---

## Documentation rules（写文档规矩）

| 文档 | 何时写 / 更新 |
| --- | --- |
| `docs/plans/*` | **写代码前**（多步任务、改科学口径、新模块） |
| `RETRO.md` | 踩坑后追加；同类问题先查再改 |
| `PHASE_STATUS.md` | 阶段完成或阻塞变化时 |
| `docs/contracts/*` | 冻结/升版合同时（需用户确认若改科学阈值）；计算调度见 `COMPUTE_DISPATCH_V001.md` |
| `README.md` | 对外说明 mindmap 与用法；保持学术中文主叙述 |
| `AGENTS.md` | 仅增删**可执行规则**；不贴长文交接 |

写文档要求：完整句子、中文可用；**短而可执行**；不复制 README 全文进 AGENTS。  
合同类用版本后缀（`_V001`）；升版不静默改旧文件语义。

---

## Mindmap steps（实现对照）

| Step | What | Modules / status |
| ---: | --- | --- |
| **0** | Freeze molecular roots | `data/paths`, `data/development_split` |
| **1** | Split by molecular_root；FT sealed | `contracts/tvt_gates`, `data/development_split` |
| **2** | Pure-PySCF teacher frames | `data/teacher_frames`, `data/weighted_dataset` |
| **3** | Epoch-0 full route | `pipeline/parent_handoff` — exec NOT_RUN |
| **4** | Train on Train frames only | `training/*` — no live train |
| **5** | Multi-epoch checkpoints | trainer loop **missing** |
| **6** | Quick val（非终选） | `weighted_loss` accumulator |
| **7** | Shortlist | `tvt_gates.quick_checkpoint_shortlist` |
| **8** | Full scientific Validation | `pipeline/scientific_validation` — writer ready, live gated |
| **9** | Val selects checkpoint | sci-val + `NUMERIC_CALIBRATION_V001.yaml` |
| **10** | Freeze identities | readiness gates |
| **11** | Final Test once | sealed only |
| **12** | No post-Test shopping | policy |

Routing: `src/nhc_deprot/mindmap_steps.py`  
Preflight: `PYTHONPATH=src python -m nhc_deprot.pipeline.mindmap_orchestrator`

---

## Science non-negotiables

- Reaction `NHC-H+ → NHC + H+`；cation (+1,s) / neutral (0,s)；root 不跨 split。
- **Parent = P01 only**: `wb97m-d3bj` / `def2-TZVPP`，grid=4，SCF 1e-9。  
  SHA256 `227c22a527e567bc4de873ab743fe9f493779eccbb1a698d2913c87695ebf87a`。
- **GAU_LOOSE**: 五准则 + ASE fmax **0.10** eV/Å；步数预算 **max 250**（唯一合同 `docs/contracts/GAU_LOOSE_V001.yaml` / 包内同名）。非 fmax **0.05**。禁止并行第二份 GAU_LOOSE 步数合同。
- 路线：freeze → AIMNet2 GAU_LOOSE → gates → exact-byte handoff → **完整** parent GAU → SP → label。  
  `single_point_only=false`；**AIMNet2 能量永不进标签**。
- 训练目标：冻结 D3 残差 E/F；禁止静默重算 D3。
- Quick-val **不得**最终选模。

| Banned | Why |
| --- | --- |
| `two_endpoint` B3LYP/def2-SVP | Wrong parent |
| fmax=0.05 as parent stop | Wrong AIMNet2 contract |
| Best-by-quick-val finetune | Violates mindmap 6–9 |
| Open Final Test for “dev convenience” | Leak / bias |
| Write outside `$WJW/NHC0801` | Boundary |

---

## 模型训练注意事项（微调硬规则 · 每次动训练前必读）

> 来源：2026-08-03 对 `runs/server_g001_snapshot/train/campaign_receipt_live.json`（3 seed × 200 ep 真实曲线）与服务器 teacher 产物的实测复盘；文献对照 arXiv:2506.21935。
> 全文推导见 `docs/plans/20260803_teacher_trajectory_and_finetune_v02_plan.md` §1。**不要重新推导，直接用。**

### T1 · AIMNet2 在本流水线里是「预优化器」，不是能量预测器

ΔE_deprot 标签**只认 parent DFT**，AIMNet2 能量永不进标签。它唯一的职责是把几何预优化到 GAU_LOOSE、让 parent PySCF 少走几步。

**推论（硬规则）**：

- **帧级 energy loss 在任何阶段都不得作为选择依据** —— 不作为终选、不作为短名单主排序、不作为"同分破并列"。它测的是不进标签的绝对偏置。
- 与预优化器职责直接对应的量是：**力误差、到 GAU_LOOSE 的步数、与 parent 参考终点几何的 RMSD、相对 Epoch-0 的 parent 步数比**。排序时它们优先于任何能量项。

### T2 · 训练是免费的，DFT 才是瓶颈——不要按"训练跑数"做预算

实测：3 seed × 200 epoch ≈ **5 分钟**（1 张 V100）。而 sci-val 每个候选要跑完整 parent GAU 优化；teacher 单 endpoint 中位 **0.72 h**。

**推论（硬规则）**：

- 消融要**跑宽**（多配方并行，成本可忽略），sci-val 要**收窄**（只送 2–3 个候选）。
- 中间必须有**零-DFT 预筛**（`pipeline/pre_screen.py` → `pre_screen_g00N/`）：只跑 AIMNet2 自身 GAU_LOOSE 弛豫，与已有的 parent 参考终点几何比 RMSD / 步数 / 力误差。它属于 mindmap 第 7 步「初步筛选」，**不需要** `scientific_validation_live` 门，收据必须写 `final_model_selected: false`。
- 任何"为了省训练时间所以只跑 A/B 两个配方"的方案都是**成本模型倒置**，直接驳回。

### T3 · 标签是未对齐的绝对总能——先解决 E0，再谈超参

训练目标是 `P01_total_energy_minus_frozen_two_body_D3_BJ`（eV，绝对值），**没有做 per-element 参考能对齐**。实测后果：train loss 掉 8.4 倍时 val energy MSE 只掉 31% 并在 epoch 50 压死，train/val 差两个数量级——模型在记忆每个分子的常数偏置，学不到迁移。

**推论（硬规则）**：

- `outputs.atomic_shift`（per-element E0，十几个参数）应当**可训练**。这正面对应 FT 教程的第一条警告"必须自己算 E0"。只放 `^outputs\.energy_mlp\.` 会逼 MLP 用环境相关容量去吸收常数偏置。
- 看到 train/val 差一个数量级以上时，**先怀疑参考能对齐，不要先加正则或减 epoch**。

### T4 · 损失权重必须按**实测量级**定，不能照抄论文默认值

实测 val `E_mse ≈ 12.4` vs `F_mse ≈ 0.42`，量级比 **≈ 30:1**。`scaled_training_loss` 按 `w/(w_e+w_f)` 归一，所以 `1:1` 的有效配比就是 30:1 偏能量。

| forces_weight | 有效 E:F | 结论 |
| ---: | --- | --- |
| 1 | 30 : 1 | 能量完全主导（当前基线） |
| **10**（FT 教程默认） | 3 : 1 | **仍是能量主导——照抄会得出"力权重无用"的错误结论** |
| 100 | 1 : 3.4 | 力主导 |

**硬规则**：改力权重前，先从最近一份 `campaign_receipt_live.json` 里读出 `weighted_energy_mse` / `weighted_forces_mse` 的实际量级，再定网格。教程给的 `1.0–20.0` 是针对已做参考能对齐的体系。

### T5 · teacher 必须落**全轨迹**帧

geomeTRIC 每步都算了 parent-level 能量+梯度。`geometric_optimize(mf, maxsteps=...)` **不传 `callback` 就等于把 ~90% 已付费的标注数据扔掉**。

**硬规则**：

- 任何 parent 优化的调用点都必须传 `callback`（PySCF 2.13.1 已确认支持；callback 收到 `calc_new` 的 `locals()`，可用 `coords`(Bohr) / `energy` / `gradients` / `self.cycle`）。
- 捕获到的是**所有被求值的几何**（含线搜索被拒绝的试探步），不是"被接受的优化路径"。必须落 `cycle`，且**不得**当作轨迹顺序解释。这些点是合法的 parent 标注，且覆盖非平衡区，对预优化器**更有价值**。
- 下游一律按 `manifest.frame_count` 动态读取，**禁止硬编码 2**。历史上 `frame_count == 2` 的 endpoint **只读，不重算不改写**。
- D3(BJ) 两体项是纯几何解析函数，**任意帧都能零 DFT 成本补投影** —— 不要因为"要重算 D3"而放弃中间帧。

### T6 · 权重策略在变长轨迹下自动成立

`assign_candidate_endpoint_weights` 是「等 candidate 质量 → 等 endpoint 质量 → 端点内均分」。**端点总质量与帧数无关**，长轨迹只是每帧权重更小。所以帧数变长不需要改权重策略——但每次改数据链后必须跑 `audit_split_weight_sums` 复核。

### T7 · 超参默认值的已知偏差

| 项 | 现状 | 应为 | 依据 |
| --- | --- | --- | --- |
| `batch_size` | 32 | **≤ 20，建议 8** | FT 教程明确 >20 伤泛化；且小数据下 32 会让一个 epoch 只剩个位数优化步 |
| `epochs` | 200 | **~120** | 实测 val 在 epoch 60–70 触底，之后 LR 衰到 ~4e-7 空转，shortlist 会选到过拟合点 |
| EMA | 无 | **0.99** | FT 教程推荐 0.98–0.99999 |
| lr / optimizer / cutoff | 1e-4 / RAdam / 基座值 | **不动** | 官方 AIMNet2 口径；FT 教程明确警告不要改基座 cutoff |

**一次只动一类变量**：禁止同时扫 lr、epochs、trainable 范围。

### T8 · 每个训练配方必须自证身份

- 落盘走 `train_g00N/runs/<run_id>/`；`run_id` 格式 `e<E权重>f<F权重>_<可训练范围>`。
- `train_info.json` 与每个 `.pt` 的 meta 必须含 `train_config_digest`（`run_id` / E,F 权重 / `trainable_parameter_regex` / `ema_decay` / `batch_size` / `epochs` 的 canonical SHA）。
- 注意 `trainable_parameter_regex` 是 **tuple**；实现若只取 `[0]` 则多模式配方静默失效——改可训练范围时先确认全部模式都被应用。

### T9 · 诚实失败优先于刷版本号

当前 pilot 只有 **3 个 train root**。在这个规模上，任何超参都救不了泛化。若消融在 T1 列出的指标上都不优于 Epoch-0：结论是**数据不足，等 teacher 出量**，不是扩大解冻范围或扫 lr。`models/v0.N` 若仍要登记，`info.json` 与 card 必须注明 `no gain vs epoch-0`。

---

## Gates

**2026-08-03 用户授权**（自动化开发阶段，见 `docs/prompt.md`）：

```text
OPEN :  teacher_pyscf_authorized   epoch0_execution
        aimnet2_train_authorized   scientific_validation_live
        rsync 部署进 NHC0801（无 --delete）
CLOSED: final_test_open            modify_wjw_outside_NHC0801
        scheduler_submission
```

**红线（即使在「全开」授权下也不得越过）**：

1. 只写 `$WJW/NHC0801`。这是共享机器（12 个用户、GPU 2/3/4/6 上是别人的作业）。
2. **不得** kill / 干扰在跑的 `nhc0801_gpu_teacher_daemon.py`、`e0_val_only`、`nhc0801_compute_steward.py`。
3. **Final Test 保持封存**。mindmap 11–12：Test 一旦开封，选模流程整体作废且不可逆。开封需要用户**单独**授权。
4. 已完成的 `frame_count == 2` teacher endpoint **只读**，不重算不改写。
5. 起训练前必须过 GPU claim；不得抢占他人显存。
6. 不改 `mindmap.md` 科学口径 / Parent P01 SHA / GAU_LOOSE **五准则与 fmax** / TVT 密封规则。步数预算变更须显式改唯一的 `GAU_LOOSE_V001.yaml`（并同步 `src`/`docs` 镜像），禁止另立第二份并行合同。

---

## Compute dispatch（短规则 · 细节见合同）

权威全文：**`docs/contracts/COMPUTE_DISPATCH_V001.md`**。此处仅可执行摘要：

| 规则 | 内容 |
| --- | --- |
| 波次大小 | **固定 10 endpoints** = 5 roots × (cation+neutral) |
| CPU 波 | **g001 pilot** Train3+Val2；正线 teacher / g001 训练只认 **Train3** |
| 分子组 | **g00N** = 5 roots = 3 Train + 2 Val；见 **Experimental data naming** |
| teacher 产物 | **`teacher_gpu_g00N/`** 唯一规范（含 g001/g002） |
| Epoch-0 产物 | **`epoch0_val_batches/g00N/`**；Parent/handoff **默认 GPU**（gpu4pyscf，与 AIMNet2 同卡、无 VASP） |
| 归属 | g002+ 来自池切组；**禁止**并入 g001 训练标签 |
| g003+ 切组队列 | 守护 **`scripts/nhc0801_gpu_teacher_daemon.py`**（旧名 `*_gpu_autofill_daemon` 仅兼容包装）；状态 **`gpu_teacher_queue/state.json`**；日志标签 **`[gpu-teacher]`**；对外只报 **g00N** |
| 并存 | CPU+GPU 默认可并存，但 **每次须用户确认**；路径与 GPU 卡启动前检查 |
| 训练 | AIMNet2 **仅 CUDA 正线**；只认 g001 **Train3** 老师帧（当前正线） |
| xc | 必须 `wb97m-d3bj` |

### 遇问题读哪个文件

| 问题 | 必读 |
| --- | --- |
| 步骤能否做 / 科学口径 | `mindmap.md` → `COMPUTE_DISPATCH_V001.md` §5 → `PHASE_STATUS.md` |
| CPU/GPU、10-ep、双波、g001 vs g002 数据 | **`COMPUTE_DISPATCH_V001.md`** |
| 核/内存/profile | `RESOURCE_SCHEDULING_V001.md` + `RESOURCE_PROFILES_V002.yaml` |
| 踩坑 | `RETRO.md` |
| 选模阈值 | `NUMERIC_CALIBRATION_V001.yaml` |
| **改微调超参 / 损失权重 / 解冻范围** | **本文件「模型训练注意事项」T1–T9**（必读，勿重推） |
| **teacher 落帧 / 轨迹捕获** | 本文件 **T5** + `docs/plans/20260803_teacher_trajectory_and_finetune_v02_plan.md` §3 M1–M4 |

---

## Server / env

| | |
| --- | --- |
| SSH | `configs/server.local.yaml`（勿提交） |
| Write root | `/home/plab/test/WJW/NHC0801` |
| ML | `source $WJW/env/envs/mlff.sh` |
| PySCF | `source $WJW/env/envs/molenv.sh`（禁混栈） |
| Epoch-0 weight | `aimnet2_wb97m_d3_0.pt` only |

```bash
rsync -avz --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' \
  -e 'ssh -o BatchMode=yes' \
  /Users/cc/nhc-deprot/ nhc614:/home/plab/test/WJW/NHC0801/
```

---

## RETRO.md 义务

- 新踩坑：**追加**到 `RETRO.md` 对应分类末尾（现象 / 根因 / `[已解决|未解决]` / 方案）。
- 已固化进本文件的条目，在 RETRO 标 **「已升级为规则」** 并指向本节。
- 不要把 RETRO 当第二 mindmap；科学冲突仍听 `mindmap.md`。

---

## Work style

- 一次只动一个会改变科学口径 / split / 授权 的点。
- 单测用合成 fixture；`PYTHONPATH=src`。
- 不 `git reset/clean` science-pilot；不改 ranker 生产树。
- 优先补齐 mindmap 缺口模块，而非扩大范围。

**Next engineering gap:** teacher 全轨迹落帧（步骤 2；见「模型训练注意事项」T5 — 每小时都在流失已付费的标注数据），其次是零-DFT 预筛 `pipeline/pre_screen.py`（步骤 7）。永不靠 quick-val 或帧级 energy loss 终选。
