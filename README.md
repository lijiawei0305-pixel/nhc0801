# Fine-Tuning AIMNet2 on ωB97M-D3(BJ)/def2-TZVPP Reference Trajectories for N-Heterocyclic Carbene Deprotonation

氮杂环卡宾（NHC）脱质子：

```math
\mathrm{NHC{-}H^{+}} \rightarrow \mathrm{NHC} + \mathrm{H^{+}}
```

本项目在 **ωB97M-D3(BJ)/def2-TZVPP** 参考轨迹上微调 **AIMNet2**：用 PySCF 生成老师帧（几何、能量、力），只在 Train 上做有监督训练，再经完整科学 Validation 选出唯一 checkpoint，最后在密封的 **Final Test** 上只评估一次。

---

## 反应与标签

每个 **molecular root** 同时包含两端点；划分、训练与评估都以 root 为单位，同一分子的 cation / neutral 及其全部轨迹不会拆到不同 split。

| 端点 | charge | multiplicity |
| --- | ---: | ---: |
| NHC–H⁺（cation） | +1 | 单重态 |
| NHC（neutral） | 0 | 单重态 |

脱质子电子能标签（kcal·mol⁻¹，越低越好）：

```math
\Delta E_{\mathrm{elec}} = (E_{\mathrm{neutral}} - E_{\mathrm{cation}}) \times 627.509474
```

```math
\Delta E_{\mathrm{deprot}} = \Delta E_{\mathrm{elec}} - 6.28
```

6.28 为项目冻结的质子相关常数。  
最终标签只来自 **ωB97M-D3(BJ)/def2-TZVPP** 能量；**AIMNet2 能量不进入标签**。

---

## 电子结构方法与几何收敛

老师轨迹，以及 handoff 后的完整优化，均在 **ωB97M-D3(BJ)/def2-TZVPP** 下完成（气相闭壳层 RKS；PySCF + geomeTRIC）：

| 项 | 设定 |
| --- | --- |
| 泛函 / 基组 | ωB97M-D3(BJ) / def2-TZVPP |
| 积分格点 | grid level 4 |
| SCF 收敛 | 1×10⁻⁹ |

AIMNet2 在自身势能面上优化时，采用 geomeTRIC 的 **GAU_LOOSE** 收敛阈值（五条同时满足），并配合 ASE LBFGS：

| 准则 | 阈值 |
| --- | --- |
| 能量变化 | \|ΔE\| ≤ 1×10⁻⁶ Eh |
| 梯度 RMS | ≤ 1.7×10⁻³ Eh/Bohr |
| 梯度最大值 | ≤ 2.5×10⁻³ Eh/Bohr |
| 位移 RMS | ≤ 6.7×10⁻³ Å |
| 位移最大值 | ≤ 1.0×10⁻² Å |
| ASE LBFGS | fmax = 0.10 eV/Å，最多 100 步 |

收敛后检查电荷、多重度、质子身份与拓扑，再将几何 **按字节原样** 交给同一 DFT 级别做完整优化与最终单点（不是只做单点）。

Epoch-0、科学 Validation 与 Final Test 共用：

```text
冻结几何
  → AIMNet2 优化至 GAU_LOOSE
  → 身份 / 拓扑检查
  → 几何 handoff
  → ωB97M-D3(BJ)/def2-TZVPP 完整优化
  → 最终单点与脱质子标签
```

对照路线为：直接在同一 DFT 级别上完整优化得到的参考轨迹与标签。

训练目标是扣除冻结两体 D3(BJ) 后的短程残差；推理时加回同一外部 D3：

```math
E_{\mathrm{short}} = E_{\mathrm{DFT,total}} - E_{\mathrm{D3}}, \quad
\mathbf{F}_{\mathrm{short}} = \mathbf{F}_{\mathrm{DFT,total}} - \mathbf{F}_{\mathrm{D3}}
```

式中 DFT 总能量与力均为 ωB97M-D3(BJ)/def2-TZVPP 结果。

---

## 方法总览

从冻结分子与 DFT 参考轨迹，到 AIMNet2 微调、完整几何路线选模，再到密封测试。细节见 [`mindmap.md`](mindmap.md)。

```mermaid
flowchart TB
  classDef data fill:#e8f4fc,stroke:#1d6fa5,stroke-width:1.5px,color:#0b2e4a
  classDef model fill:#f3e8ff,stroke:#7c3aed,stroke-width:1.5px,color:#3b0764
  classDef gate fill:#fff7ed,stroke:#c2410c,stroke-width:1.5px,color:#7c2d12
  classDef seal fill:#fef2f2,stroke:#b91c1c,stroke-width:2px,color:#7f1d1d

  subgraph P1["① 参考数据"]
    direction TB
    A["冻结 molecular root<br/>两端点 · 电荷 · 多重度 · 结构哈希"]
    B["按 root 划分<br/>Train / Val / Test 互不相交"]
    C["ωB97M-D3(BJ)/def2-TZVPP<br/>完整优化 · 轨迹 · 能量与力"]
    A --> B --> C
  end

  C --> Train["Train 帧<br/>仅用于训练"]
  C --> Val["Validation<br/>基线与选模"]
  C --> Test["Final Test<br/>密封至最后"]

  subgraph P2["② Epoch-0 基线"]
    direction TB
    D["官方 AIMNet2（未微调）"]
    E["GAU_LOOSE → handoff<br/>→ DFT 完整优化 → 标签"]
    D --> E
  end

  subgraph P3["③ 有监督微调"]
    direction TB
    F["仅 Train：拟合短程残差 E/F"]
    G["多 epoch · 多种子 checkpoint"]
    H["快速验证：只筛候选<br/>不定终选"]
    F --> G --> H
  end

  subgraph P4["④ 科学选模"]
    direction TB
    I["短名单完整路线<br/>AIMNet2 → handoff → DFT"]
    J["对照 DFT 参考与 Epoch-0<br/>选定唯一模型"]
    K["冻结模型 · 划分 · 协议"]
    I --> J --> K
  end

  subgraph P5["⑤ 密封 Final Test"]
    direction TB
    L["一次性评估 · 考后不改模型"]
  end

  Val --> D
  Train --> F
  Val --> I
  E --> J
  H --> I
  K --> L
  Test --> L

  class A,B,C,Train,Val,Test data
  class D,F,G model
  class E,H,I,J gate
  class K,L seal
```

| 阶段 | 要点 |
| --- | --- |
| ① 参考数据 | 冻结 root、互不交叉划分、生成 DFT 老师轨迹 |
| ② Epoch-0 | 未微调模型走完整路线，作微调前对照 |
| ③ 微调 | 只学 Train；快速验证只筛候选 |
| ④ 选模 | 完整几何路线比标签，不靠帧 loss 定终选 |
| ⑤ Final Test | 密封集只考一次 |

**共用几何评估路线**（Epoch-0 / 选模 / Final Test）：

```text
冻结几何 → AIMNet2（GAU_LOOSE）→ exact-byte handoff
        → ωB97M-D3(BJ)/def2-TZVPP 完整优化 → 单点与 ΔE_deprot
```

规模化按 **`g00N`**（5 root = 3 Train + 2 Val）：老师帧 `teacher_gpu_g00N/`，Epoch-0 `epoch0_val_batches/g00N/`，训练过程 `train_g00N/`。进度见 [`progress.md`](progress.md)。

---

## 模型发布：train_g00N → v0.N

训练过程与正式发布模型分开命名：

| 训练过程 | 发布版本 | 内容 |
| --- | --- | --- |
| `train_g001/` | **v0.1** → `models/v0.1/` | `model.pt` + `info.json` + **特征卡片** |
| `train_g002/` | **v0.2** → `models/v0.2/` | 同上 |
| `train_g00N/` | **v0.N** | 组号 N 对应小版本号 N |

```text
models/v0.1/
  model.pt      # 权重（固定短名）
  info.json     # 来源 seed/epoch、sha256
  card.json     # 特征数据（可机读）
  card.svg      # 发布图（给人看）
```

**每次发布新版本必须生成一张特征卡片**（`card.svg`）。  
卡片要同时覆盖：领域里 ML 力场常用指标，以及本项目科学路线上的结果——**不能只有帧级 loss**。

### 卡片上展示哪些特征（设计说明）

结合 AIMNet2 / MLFF 文献里常见的 **energy–force（EF）精度**，以及本仓库 mindmap 的 **完整几何 + 脱质子标签** 口径，每张卡分六块：

| 区块 | 特征 | 为何需要 |
| --- | --- | --- |
| **Identity** | 版本 `v0.N`、标题、基座 `aimnet2_wb97m_d3_0`、路径 `models/v0.N/model.pt` | 一眼认出是哪一版 |
| **Chemistry & reference** | 反应 `NHC-H+ → NHC + H+`；参考 DFT ωB97M-D3(BJ)/def2-TZVPP；学短程残差 E/F；标签只用 DFT | 化学任务与老师级别写死，避免和别的栈混淆 |
| **Training** | 来自 `train_g00N`、root/帧数、seed、epoch、**由科学 Validation 选模** | 对应固定顺序 g001→v0.1；强调不是 quick-val 终选 |
| **Frame metrics（仅筛选用）** | 能量 MAE/RMSE、力 MAE/RMSE | MLFF 社区标配；**只作监测/短名单，不写「最终冠军」** |
| **Scientific route（选模依据）** | ΔE_deprot 相对 DFT 参考的误差；相对 **Epoch-0** 的 parent 步数/墙钟比；handoff 通过率；拓扑/身份通过率；评估了几个 Val root | 本项目真正关心：handoff 后完整 DFT 路线是否更省、标签是否站得住 |
| **Provenance** | 权重 SHA256 前缀、备注 | 可核对、可复现 |

实现：`src/nhc_deprot/training/model_card.py`；登记发布时默认写出卡片。本地可演示：

```bash
PYTHONPATH=src python scripts/nhc0801_render_model_card.py --demo --version v0.1
# → runs/.../models/v0.1/card.svg
```

---

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `mindmap.md` | 科学流程真值（方框图） |
| `AGENTS.md` | 协作与命名约定 |
| `PHASE_STATUS.md` | 阶段状态 |
| `progress.md` | 各组 teacher / Epoch-0 进度 |
| `src/nhc_deprot/` | 库代码 |
| `scripts/` | CLI / 作业入口 |
| `tests/` | 合成 fixture 单测 |

科学口径以 `mindmap.md` 为准。
---

## 本地开发

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src python -m pytest -q
```

不跑化学的预检：

```bash
PYTHONPATH=src python -m nhc_deprot.pipeline.mindmap_orchestrator
```

AIMNet2 与量子化学计算使用**相互独立的软件环境**，不要混用同一环境。