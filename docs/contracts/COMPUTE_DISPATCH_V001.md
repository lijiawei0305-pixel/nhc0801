# NHC0801 计算调度与 Mindmap 执行合同 V001

**状态**: 权威合同（用户 2026-08-02 互动确认）  
**文件**: `docs/contracts/COMPUTE_DISPATCH_V001.md`  
**配套**: `mindmap.md` · `RESOURCE_SCHEDULING_V001.md` · `RESOURCE_PROFILES_V002.yaml` · `AGENTS.md`  
**主机**: `nhc614`  
**写根**: 仅 `$WJW/NHC0801`（记 `$NHC0801`）  
**当前 generation**: `nhc0801-g001`（正线）  
**升版规则**: 改语义必须新建 `COMPUTE_DISPATCH_V002`（或更高），禁止静默改本文件含义  

冲突优先级（更严者优先；科学与本调度冲突时先问用户）：

```text
mindmap.md
  → 本文件 COMPUTE_DISPATCH_V001（计算波次 / 后端 / 路径 / 双代归属）
  → RESOURCE_SCHEDULING_V001 + RESOURCE_PROFILES_V002（核/内存/profile 数字）
  → 其它 docs/contracts/*（GAU_LOOSE、NUMERIC_CALIBRATION、parent protocol…）
  → AGENTS.md
  → RETRO.md / pilot evidence
```

> **相对 `RESOURCE_SCHEDULING_V001` 的明确覆盖**  
> 调度文档中「Parent CPU-only」适用于 **未指定后端** 的默认 CPU 波。  
> **本文件允许** Parent 使用 **gpu4pyscf**（用户指定 GPU 波或 e0 parent=GPU 时）。  
> 数字资源（t=10、池 0–99、预留 100–111、8 GiB/endpoint）仍以 RESOURCE_* 为准，除非本文件另写。

---

## 0. 一句话总则（用户已确认）

| # | 规则 |
| ---: | --- |
| 1 | 每波 **固定 10 endpoints** = **5 molecular roots × (cation + neutral)** |
| 2 | **CPU 波** = **g001 正线 pilot**（Train3+Val2），写 **CPU/正线路径** |
| 3 | **GPU 波** = **g002 全套 5 roots**（3 Train + 2 Val 草案；与 pilot **不同 root**），写 **GPU 分目录** |
| 4 | CPU 与 GPU **默认可并存**，但 **每次启动并存须用户当场确认**；启动前做路径与 GPU 占用检查 |
| 5 | g001 **训练只认 pilot Train3**；GPU 帧 **禁止** 进入 g001 Train/加权集/正线 freeze |
| 6 | 当前冻结的 GPU 5 roots = **g002 全部 TVT 开发集**（**3 Train + 2 Val**，字典序划分；**无**本批 FT）；正式 generation 开张时写入 g002 meta |
| 7 | AIMNet2 **训练** = **CUDA 正线**；不提供「纯 CPU 训练」正线路 |
| 8 | Epoch-0 的 **Parent 段默认 CPU**；**仅当用户指定** 时可用 gpu4pyscf Parent |
| 9 | Final Test **默认密封**；永不自动打开 |
| 10 | 代理切换后端：用户说 **用 GPU** → gpu4pyscf 波；说 **用 CPU** → CPU PySCF 波；**10 endpoints 不改** |

---

## 1. 冻结分子名单

### 1.1 g001 pilot（正线 · 10 endpoints）

**Train（3）** — 唯一允许进入 g001 训练标签的 root：

| # | molecular_root (InChIKey) |
| ---: | --- |
| T1 | `ACGCNTKELWXJPN-UHFFFAOYSA-N` |
| T2 | `PDIYCCLDBKWBTK-UHFFFAOYSA-N` |
| T3 | `VNYHGZAUUQMMDL-UHFFFAOYSA-N` |

**Validation（2）** — e0 / quick-val / sci-val；**禁止训练**：

| # | molecular_root (InChIKey) |
| ---: | --- |
| V1 | `KZYKDQNIIMATMJ-UHFFFAOYSA-N` |
| V2 | `RMEQTBVGGNKAEQ-UHFFFAOYSA-N` |

**Final Test（2）** — 仅 commitment 密封；身份不在此文件展开；**禁止**自动计算、禁止用 GPU/CPU 波「顺便」打开。

每个 root 必须同时具备 **cation** 与 **neutral** 两端点；**禁止**把同一 root 拆到不同 split。

### 1.2 g002 第一批（GPU 波 · 仅此 5 roots · 合同冻结草案）

**规模（用户 2026-08-02 确认）：** g002 **只有** 下列 5 个 molecular root；划分方法与 g001 pilot **相同结构** → **3 Train + 2 Val**；本批 **不设 Final Test 身份**（g002 FT 另议/另封）。

**划分规则：** InChIKey **字典序**；前 3 → Train，后 2 → Val（可复现，禁止 silently 重排）。

#### g002 Train（3）— 未来 g002 唯一训练标签源（草案）

| # | molecular_root (InChIKey) |
| ---: | --- |
| T1 | `CLXFIGGGSODORK-UHFFFAOYSA-N` |
| T2 | `CRPRBFHOCLDMMB-UHFFFAOYSA-N` |
| T3 | `HFQMBFOQLKGXEV-UHFFFAOYSA-N` |

#### g002 Validation（2）— 未来 g002 e0 / sci-val；禁止训练（草案）

| # | molecular_root (InChIKey) |
| ---: | --- |
| V1 | `HVVRUQBMAZRKPJ-UHFFFAOYSA-N` |
| V2 | `IPMZWBRHUWBMSP-UHFFFAOYSA-N` |

**权威别名：** `g002_roots_batch1` = 上表 5 根并集；`g002_train_roots_batch1` / `g002_val_roots_batch1` 如上。

**硬约束：**

- 这 5 个 root **不是** g001 Train / Val / FT。  
- 其 teacher 产物（**`teacher_gpu_g002/`**）**不得** 并入 g001 `datasets/`、**不得** 用于 g001 微调标签。  
- 开 **g002** generation 时：必须按上表写入 generation meta / split commitment（3 Train + 2 Val）；**不得**把 Val 两根误标为 Train 或并进 g001。  
- GPU 日常波次默认队列 = 这 5 根 × 两端点 = **10 endpoints**（与 CPU pilot 波对称）。  
- 增删 root 或改 Train/Val 归属 → **升版合同** 或用户书面确认后改本表并记修订记录。

### 1.2b g003 第二批（GPU · 同样取法 · 合同冻结草案）

**取法：** gold 完整 pair 字典序，**排除** g001 pilot 五根 ∪ g002 batch1 五根，再取 5 根 → **3 Train + 2 Val**（前三训、后两验）。  
**产物目录：** **`teacher_gpu_g003/`**（与 g001/g002 同模式，仅批号不同）。  
**归属：** 未来 generation 开发集扩展；**禁止**并入 g001 训练。本批无 FT。

#### g003 Train（3）

| # | molecular_root (InChIKey) |
| ---: | --- |
| T1 | `IRCJCNRXJWPRQT-UHFFFAOYSA-N` |
| T2 | `KDNXGTHUHICKOJ-UHFFFAOYSA-N` |
| T3 | `OOGXHYSEXNJMLD-UHFFFAOYSA-N` |

#### g003 Validation（2）

| # | molecular_root (InChIKey) |
| ---: | --- |
| V1 | `PKQGMOQCRAMPKT-UHFFFAOYSA-N` |
| V2 | `PMPUSEWFQVHUNO-UHFFFAOYSA-N` |

**调度：** g002 未完成时，g003 仅占用**空闲 GPU**（避免与 g002 worker 钉卡冲突）；g002 整波结束后可用满 8 卡续跑/补跑。  
**名单文件：** `runs/nhc0801-g001/logs/g003_roots_frozen.json`。

### 1.3 每波 endpoint 展开式

```text
10 endpoints =
  for root in {5 roots of this wave}:
    (root, cation), (root, neutral)
```

- CPU 正线波：5 roots = Train3∪Val2。  
- GPU 波：5 roots = `g002_roots_batch1`（Train3∪Val2 草案；**g002 仅此一批**，除非升版增根）。  
- **不足 10**：拒绝静默短波；须问用户。  
- **超过 10**：按 10 一切批；下一批须新回执，不混进上一 campaign。

---

## 2. 后端与角色

### 2.1 Parent DFT 后端

| 后端关键字 | 引擎 | 环境要点 |
| --- | --- | --- |
| `cpu` | `pyscf.dft.RKS`（gpupyscf 栈上的 CPU 路径） | `CUDA_VISIBLE_DEVICES=` 空；`OMP/MKL/OPENBLAS=t`（正线 t=10） |
| `gpu` | `gpu4pyscf.dft.RKS` | **每进程钉 1 张物理 GPU**；`host OMP` 宜 2；`xc=wb97m-d3bj` |

**共同科学硬约束（与 mindmap / P01 一致）：**

- `xc = wb97m-d3bj`（禁止 plain `wb97m`）  
- basis `def2-TZVPP`；grid=4；SCF conv 1e-9  
- Parent protocol SHA：`227c22a527e567bc4de873ab743fe9f493779eccbb1a698d2913c87695ebf87a`  
- 老师终点 = parent 最终 GAU；帧至少 **initial + final**（逐步全轨迹可后补，须升版说明）  
- **AIMNet2 能量永不写入最终标签**

### 2.2 AIMNet2

| 用途 | 设备 | 说明 |
| --- | --- | --- |
| 训练（步骤 4–5） | **CUDA（正线）** | 不提供纯 CPU 训练正线路 |
| GAU_LOOSE（e0 / 推理预优） | **CUDA** | 与 parent GPU 波 **错开卡号** |
| 权重 epoch-0 | 官方 `aimnet2_wb97m_d3_0` | SHA `f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28` |

### 2.3 用户口令 → 代理行为

| 用户意图 | 代理动作 |
| --- | --- |
| 「用 GPU / gpu4pyscf / GPU 波」 | 启动 **g002 batch1** 10-endpoint GPU teacher（3 Train + 2 Val 草案五根）；`backend=gpu` |
| 「用 CPU / 只 CPU」 | 启动 **g001 pilot** 10-endpoint CPU teacher；`backend=cpu` |
| 「两波一起 / 并存」 | **必须再确认** 路径不冲突、GPU 卡与 e0/train 不撞；然后 CPU pilot + GPU 候选可同时跑 |
| 「e0 parent 用 GPU」 | 仅 g001 Val parent 段 `backend=gpu`；默认未说则 **CPU parent** |
| 未指定后端 | Parent **默认 cpu** |

---

## 3. 路径隔离（backend 分目录）

根：`$NHC0801/runs/nhc0801-g001/`（下一代换 `nhc0801-g002/`）。

| 角色 | 目录 | 回执 |
| --- | --- | --- |
| g001 teacher（正线 pilot） | **`teacher_gpu_g001/`** | `teacher_gpu_g001/campaign_receipt_live.json` |
| g002 teacher | **`teacher_gpu_g002/`** | `teacher_gpu_g002/campaign_receipt_live_gpu.json` |
| g00N teacher（N≥3） | **`teacher_gpu_g00N/`** | 同目录下 campaign 收据 |
| g001 Epoch-0 | **`epoch0_val_batches/g001/`** | `epoch0_val_batches/g001/epoch0/campaign_receipt.json`（**禁止** dry-run 冒充 live） |
| g00N Epoch-0 | **`epoch0_val_batches/g00N/`** | 同结构 |
| g001 训练 | `train/` | `train/campaign_receipt_live.json` |
| g001 shortlist / sci-val | `sci_val/` | shortlist / sci-val receipts |
| 日志 | `logs/` | 文件名带 `cpu` / `gpu` / `02c` 等后缀 |

**禁止：**

- GPU 组帧写入 **`teacher_gpu_g001/`** 覆盖 pilot 目录  

- CPU pilot 与 GPU 候选 **并发写同一 endpoint 目录**  
- 把 **`teacher_gpu_g00N/`（N≥2）** 链进 g001 weighted dataset 而不经独立 freeze  


实现脚本（现行，可随代码演进，以本目录语义为准）：

- CPU 正线 teacher 波：`scripts/nhc0801_teacher_wave_02c.py`（pilot 队列）  
- GPU 候选波：`scripts/nhc0801_teacher_wave_gpu_02c.py`  
- Parent worker：`scripts/nhc0801_pyscf_parent_worker.py`（`backend=cpu|gpu`）

---

## 4. 资源与启动前检查

### 4.1 CPU 正线波（t=10 · 池 0–99）

见 `RESOURCE_SCHEDULING_V001.md` + profile `auto_fill_112_t10_r12_v1`：

- 10 endpoints × 10 threads → 占用池内 100 逻辑核量级  
- 预留 **100–111** 不绑 teacher CPU 池  
- 内存预算 8 GiB/endpoint；主机预留 40 GiB  

### 4.2 GPU 候选波

- 并发 ≤ **空闲物理 GPU 数**（本机 8×V100 → 默认 max_parallel≤8；第 9–10 endpoint 排队）  
- **一进程一卡**；禁止单进程多卡  
- `host_threads` 默认 **2**（避免与 CPU 波抢 0–99）  
- 启动前：`nvidia-smi` 确认目标卡空闲；与 e0/train 指定卡冲突则 **拒绝启动并询问用户**

### 4.3 并存检查清单（每次）

```text
[ ] 用户已确认并存
[ ] CPU 波 roots ⊆ pilot；GPU 波 roots ⊆ g002_roots_batch1（§1.2）
[ ] 写路径：`teacher_gpu_g001/` vs `teacher_gpu_g002/`… 组号不重叠
[ ] GPU 卡列表与 AIMNet2/e0/train 卡不冲突
[ ] OMP：CPU 波 t=10；GPU host t≤2（除非用户改）
[ ] xc=wb97m-d3bj；无 Final Test
[ ] g001 训练数据源仍仅 pilot Train3
```

---

## 5. Mindmap 0–12 执行细则

下列与 `mindmap.md` 阶段名对齐。模块名随仓库演进，以 **产物与门控** 为准。

### 0. 冻结全部原始分子

| 项 | 内容 |
| --- | --- |
| 输入 | gold XYZ（`mol_gold/xyz/<root>_{cation,neutral}.xyz`） |
| 动作 | 冻结原子序、坐标、charge、multiplicity、结构身份 |
| 输出 | generation meta / 结构 SHA 绑定 |
| 后端 | 无 DFT 波次 |
| 门控 | 未冻结不得 live 化学 |

### 1. 按 molecular root 划分

| 项 | 内容 |
| --- | --- |
| g001 | Train3 / Val2 / FT sealed（§1.1） |
| g002（未来） | **仅** §1.2 五根；**3 Train + 2 Val**（字典序）；开 gen 时写入 meta |
| 输出 | split commitment；`TRAIN_ROOTS` / `VALIDATION_ROOTS` 与合同一致 |
| 禁止 | root 跨 split；Train∩Val∩Test≠∅ |

### 2. 用 Pure PySCF 生成老师答案

| 项 | CPU 正线波 | GPU 候选波 |
| --- | --- | --- |
| roots | pilot 5 | g002 batch1 5（Train3+Val2 草案） |
| endpoints | 10 | 10 |
| backend | `cpu` | `gpu`（gpu4pyscf） |
| 路径 | **`teacher_gpu_g001/`** | **`teacher_gpu_g002/`** |
| 归属 | g001 步骤 2 | **囤给 g002**（Train 帧将来训；Val 帧将来只评估） |
| 收口 | `campaign_receipt_live.json` · `live_chemistry=true` | `campaign_receipt_live_gpu.json` |
| 失败策略 | continue_queue（单端点 FAIL 不默认杀整波） | 同左 |

每 endpoint 流水：冻结几何 → parent 完整优化至 GAU → 存 frame（≥初+终）+ manifest → 协议 SHA。

### 3. 训练前 Epoch-0 基线

**Epoch-0 命名与 Val-only 硬规则（用户 2026-08-02/03 确认）：**

| # | 规则 |
| ---: | --- |
| E0-1 | **只算 Validation roots**；**禁止**对任何 Train root 跑 e0（无用功） |
| E0-2 | **标准名**：**g001 Epoch-0**、**g002 Epoch-0**、**g00N Epoch-0**（按批号）；**废止**「扩展 Val e0」说法 |
| E0-3 | **统一落盘**：`epoch0_val_batches/<batch_id>/`（**含 g001**，不再用顶层特殊 `epoch0/`） |
| E0-4 | **g001 Epoch-0** = pilot 两个 Val（KZYKDQ…、RMEQTB…）→ `epoch0_val_batches/g001/` |
| E0-5 | **g00N Epoch-0** = 该批 **val_roots**（通常 2 根）；train_roots **永不**进队 → `epoch0_val_batches/g00N/` |
| E0-6 | AIMNet2 GAU 仅占 **无 VASP** 的 GPU；可与本项目 **gpu4pyscf** 共卡；**禁止**与 vasp_std 共卡 |
| E0-7 | Parent/handoff：**默认 GPU gpu4pyscf**（与 AIMNet2 同物理卡、无 VASP）；用户 2026-08-03 因 CPU 过慢确认。CPU 仅作回退 |
| E0-8 | 守护：`scripts/nhc0801_e0_val_queue_daemon.py` + `pipeline/e0_val_only.py`（`--skip-g001` 时 **g001 Epoch-0** 由 live_orchestrate / 侧路 `e0_val_only` 负责） |


| 段 | 默认 | 可选 |
| --- | --- | --- |
| AIMNet2 GAU_LOOSE | CUDA（指定空闲卡） | — |
| Parent 精修 | **CPU** | 用户指定时 **GPU parent** |
| 分子 | **仅 g001 Val2**（4 endpoints；若合同坚持「10」仅适用于 teacher 波，e0 以 Val 集合为准） | 禁止 FT |
| 输出 | **`epoch0_val_batches/g001/`**（及 g00N）live 回执；**禁止** dry-run 冒充 LIVE_EPOCH0_PASS |
| 比较 | vs pure-PySCF reference；标签 MAE 等 |

说明：teacher 波固定 10 endpoints；**e0 分子集合 = Validation roots**，不必凑满 10。文档不把 e0 强行扩成 10。

### 4. 正式训练 AIMNet2

| 项 | 内容 |
| --- | --- |
| 数据 | **仅 g001 Train3** 老师帧（+ 已冻结 D3 残差 / weighted NPZ） |
| 设备 | **CUDA** |
| 禁止 | 读 Val/FT；读 **`teacher_gpu_g00N/`（N≥2）** 进 g001 训练；AIMNet2 能进标签 |
| 现状 | g001 已可有 live 3×seed×200；重训须授权 |

### 5. 训练多个 Epoch 并保存 Checkpoint

| 项 | 内容 |
| --- | --- |
| 要求 | 多 epoch 落盘；shortlist 所需 epoch 应有 `.pt` |
| 已知缺口 | 仅最后 epoch 导出时须在计划中补导出/重训 |

### 6. 训练中的快速 Validation

| 项 | 内容 |
| --- | --- |
| 数据 | Val 固定帧标签 |
| 禁止 | 新 PySCF；反传；**用 quick-val 定最终模型** |

### 7. 初步筛选少数候选 Checkpoint

| 项 | 内容 |
| --- | --- |
| 输出 | shortlist campaign；记录 `weights_present` |
| 注意 | 无 `.pt` 的候选不得进入 live sci-val |

### 8. 完整科学 Validation

| 项 | 内容 |
| --- | --- |
| 路线 | 候选 ckpt → AIMNet2 GAU_LOOSE → handoff → Parent GAU → 标签 |
| 门控 | **须用户 live 授权**；默认关闭 |
| Parent 后端 | 默认 CPU；用户指定可 GPU |
| 分子 | g001 Val only |

### 9. Validation 负责最终模型选择

| 项 | 内容 |
| --- | --- |
| 比较 | Pure-PySCF · epoch-0 · finetuned |
| 标定 | `NUMERIC_CALIBRATION_V001.yaml` |
| 禁止 | quick-val 终选 |

### 10. 全部内容正式冻结

| 项 | 内容 |
| --- | --- |
| 内容 | ckpt SHA、split、GAU_LOOSE、P01、numeric cal、git commit 等 |
| g001 | 不得把 g002 候选 root 写进 g001 Train freeze |
| 状态 | 选模前 PROVISIONAL；选模后 hard freeze |

### 11. Final Test

| 项 | 内容 |
| --- | --- |
| 默认 | **SEALED** |
| 开启 | 仅用户二次确认 + 独立审计 |
| 禁止 | GPU/CPU 日常波次触碰 FT |

### 12. Test 后不允许继续选择

| 项 | 内容 |
| --- | --- |
| 禁止 | 换 ckpt、改阈值、继续训、删失败分子、把 Test 并回 Train 再考 |

---

## 6. 双代数据归属（防弄错）

```text
                    ┌─────────────────────────┐
  gold XYZ ────────►│  g001 teacher · 10 ep   │──► teacher_gpu_g001/ ──► g001 Train3 训练
                    │  (正线 pilot)           │──► g001 Val2 只评估
                    └─────────────────────────┘
  gold XYZ ────────►│  g002 teacher · 10 ep   │──► teacher_gpu_g002/ ──► 【囤，不进 g001 训】
                    │  3 Train + 2 Val        │         │
                    └─────────────────────────┘         ▼
                                              另开 generation 时：Train3 可训 / Val2 只评估
```

| 错误 | 正确 |
| --- | --- |
| 用 g002 五根微调 g001 | 只用 g001 pilot Train3 微调 g001 |
| 把 g002 Val 两根当 Train 训 | 字典序后两根 **只** 评估 |
| 开 g002 时忘掉五根或重排 split | 严格按 §1.2 三训两验 |
| CPU/GPU 写同一目录 | backend 分目录 |

---

## 7. 代理问题 → 必读文件

| 问题类型 | 必读（按序） |
| --- | --- |
| 科学步骤卡点 / 能否跳步 | `mindmap.md` → 本文件 §5 → `PHASE_STATUS.md` |
| CPU/GPU 用哪个、10 端点、双波并存 | **本文件** → `RESOURCE_SCHEDULING_V001.md` → `RESOURCE_PROFILES_V002.yaml` |
| 核/绑核/t/内存 N | `RESOURCE_SCHEDULING_V001.md` → `RESOURCE_PROFILES_V002.yaml` → `src/nhc_deprot/resources/*` |
| Parent 协议 / xc / GAU | `mindmap.md` → parent protocol 合同 → `GAU_LOOSE_V001.yaml` |
| 标签 / D3 / 训练目标 | `mindmap.md` → D3/weighted 相关合同与 `RETRO.md` |
| 选模阈值 | `NUMERIC_CALIBRATION_V001.yaml` → mindmap 8–9 |
| 进程挂死 / 误杀 / 路径 | `RETRO.md` → 本文件 §3–4 → `PHASE_STATUS.md` |
| 写权限 / 服务器边界 | `AGENTS.md` → 本文件页眉 |
| 代码入口 | `scripts/nhc0801_teacher_wave_*.py` · `pipeline/live_teacher.py` · `live_epoch0.py` |

**开工前最低阅读集：**

1. `mindmap.md`  
2. **本文件** `COMPUTE_DISPATCH_V001.md`  
3. `AGENTS.md`  
4. `PHASE_STATUS.md`  
5. 若动资源数字：`RESOURCE_SCHEDULING_V001.md` + `RESOURCE_PROFILES_V002.yaml`

---

## 8. 现行 CLI 锚点（服务器）

```bash
ssh nhc614
source /home/plab/test/WJW/env/envs/mlff.sh
export PYTHONPATH=/home/plab/test/WJW/NHC0801/src
export PYTHONUNBUFFERED=1
cd /home/plab/test/WJW/NHC0801

# CPU 正线 teacher（pilot 10）— 示例
python3 -u scripts/nhc0801_teacher_wave_02c.py \
  --profile auto_fill_112_t10_r12_v1 --roots train+val \
  --max-parallel 10 --threads 10 --force-n 10

# GPU 候选 teacher（g002 batch1 10）— 示例
python3 -u scripts/nhc0801_teacher_wave_gpu_02c.py \
  --n-endpoints 10 --max-parallel 8 --gpu-ids 0,1,2,3,4,5,6,7 \
  --host-threads 2 --out-subdir teacher_gpu_g002
```

日志建议：

- `logs/teacher_wave_02c.out`（CPU）  
- `logs/teacher_gpu_wave_02c.out`（GPU）  
- `logs/live_epoch0_02c.out`（e0）

---

## 9. 验收清单（调度层）

### CPU 正线 teacher

- [ ] 恰好 pilot 10 endpoints 入队  
- [ ] `backend` 为 cpu；绑核 ∈ 0–99  
- [ ] 产物在正线 teacher 路径；`live_chemistry=true`  
- [ ] 未写 FT；未读 FT 身份  

### GPU 候选 teacher

- [ ] roots = `g002_roots_batch1`（§1.2 五根；3 Train + 2 Val 草案）  
- [ ] 与 g001 pilot **无交集**  
- [ ] 一卡一进程；回执含 `gpu_index` / `backend=gpu4pyscf`  
- [ ] 产物仅在 **`teacher_gpu_g002/`**（或对应 `teacher_gpu_g00N/`）
- [ ] **未**并入 g001 train 数据；回执可标注各 root 的 g002 split 角色（train/val）  

### 并存

- [ ] 用户当场确认记录在日志  
- [ ] 路径与 GPU 卡检查通过  

### g001 训练

- [ ] 标签源 ⊆ Train3  
- [ ] 设备 CUDA  
- [ ] 无 GPU 候选帧  

---

## 10. 修订记录

| 版本 | 日期 | 内容 |
| --- | --- | --- |
| V001 | 2026-08-02 | 初版：用户互动确认。10-ep 固定；CPU=g001 pilot；GPU=g002 batch1；backend 分目录；并存需确认；e0 parent 默认 CPU、指定才 GPU；训练仅 CUDA；AGENTS 指针配套 |
| V001 修订 | 2026-08-02 | g002 **仅** batch1 五根；**3 Train + 2 Val**（InChIKey 字典序：前三训、后两验）；本批无 FT |
| V001 修订 | 2026-08-02 | 增加 **g003 batch1** 五根（字典序排除 pilot∪g002）；3 Train + 2 Val；GPU 产物 `teacher_gpu_g003/`；g002 完成后可自动/并行开 g003 |
| V001 修订 | 2026-08-02 | **GPU 自动填充流水线**：池 `RIGID_SMALL_NHC_POOL_V001`（401k 库闭集 frag、禁 nOct，~1993）；全局队列+动态占卡；批 5 roots/`teacher_gpu_g00N`；池尽停止；守护 `nhc0801_gpu_autofill_daemon.py` |
| V001 修订 | 2026-08-02 | **e0 Val-only**：永不 train e0；g001 双 Val + 扩展批 val_roots；无 VASP 卡可共 gpu4pyscf；守护 `e0_val_queue_daemon` / `e0_val_only` |
| V001 修订 | 2026-08-03 | Epoch-0 **标准名** = **g00N Epoch-0**（废止「扩展 Val e0」）；磁盘 `epoch0_val_batches/g00N/` 仅为 g00N Epoch-0 落点 |
| V001 修订 | 2026-08-03 | **g001 Epoch-0** 落盘与 g002 对齐：`epoch0_val_batches/g001/`（废止顶层特殊 `epoch0/`） |
| V001 修订 | 2026-08-03 | Epoch-0 **Parent/handoff 默认 GPU**（gpu4pyscf，与 AIMNet2 同卡）；CPU 过慢，用户确认改默认 |
| V001 修订 | 2026-08-03 | Teacher 产物目录统一 **`teacher_gpu_g00N/`**（含 g001/g002）；废止「autofill」作对外称呼、废止 `teacher/` / `teacher_gpu_side/` 规范名 |

---

## 11. 未决（须再问用户，禁止代理自作主张）

- g002 generation id 命名与 **何时**开训 / 是否新建 `nhc0801-g002`  
- g002 是否另封 Final Test（本批五根 **不含** FT）  
- live sci-val 授权时机（g001 / g002 分别）  
- teacher 是否升级为逐步全轨迹帧  

（以上 **不** 在 V001 默认自动执行。**g002 的 3+2 名单已冻结于 §1.2。**）
