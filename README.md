# Fine-Tuning AIMNet2 on ωB97M-D3(BJ)/def2-TZVPP Reference Trajectories for N-Heterocyclic Carbene Deprotonation

氮杂环卡宾（NHC）脱质子：

```math
\mathrm{NHC{-}H^{+}} \rightarrow \mathrm{NHC} + \mathrm{H^{+}}
```

在 **ωB97M-D3(BJ)/def2-TZVPP** 参考轨迹上微调 **AIMNet2**（PySCF 老师帧：几何 / 能量 / 力）。Train 上训练，完整科学 Validation 选模，密封 Final Test 只评一次。标签只认 DFT，AIMNet2 能量不进 $\Delta E_{\mathrm{deprot}}$。

---

## 反应与标签

| | charge | multiplicity |
| --- | ---: | ---: |
| NHC–H⁺ | +1 | 单重态 |
| NHC | 0 | 单重态 |

```math
\Delta E_{\mathrm{elec}} = (E_{\mathrm{neutral}} - E_{\mathrm{cation}}) \times 627.509474
```

```math
\Delta E_{\mathrm{deprot}} = \Delta E_{\mathrm{elec}} - 6.28
```

（kcal·mol⁻¹，越低越好；6.28 为冻结质子常数。）

---

## 计算设定

| | |
| --- | --- |
| 参考 DFT | ωB97M-D3(BJ) / def2-TZVPP，grid 4，SCF $10^{-9}$ |
| AIMNet2 停点 | GAU_LOOSE（五准则 + ASE fmax 0.10 eV/Å） |
| 训练目标 | 冻结 D3 后的短程残差 $E,F$ |
| 评估路线 | AIMNet2 → handoff → 完整 DFT → 标签 |

细节与步骤图见 [`mindmap.md`](mindmap.md)。

---

## 方法总览

```mermaid
flowchart TB
  classDef data fill:#e8f4fc,stroke:#1d6fa5,color:#0b2e4a
  classDef model fill:#f3e8ff,stroke:#7c3aed,color:#3b0764
  classDef gate fill:#fff7ed,stroke:#c2410c,color:#7c2d12
  classDef seal fill:#fef2f2,stroke:#b91c1c,color:#7f1d1d

  A[冻结 root · 划分 Train/Val/Test] --> B[ωB97M-D3(BJ)/def2-TZVPP 老师轨迹]
  B --> C[Train 帧]
  B --> D[Val]
  B --> E[Test 密封]
  D --> F[Epoch-0 基线]
  C --> G[微调 AIMNet2]
  G --> H[科学 Validation 选模]
  F --> H
  D --> H
  H --> I[发布 models/v0.N]
  I --> J[Final Test 一次]
  E --> J

  class A,B,C,D,E data
  class F,G model
  class H,I gate
  class J seal
```

数据按组 **g00N**（3 Train + 2 Val）：`teacher_gpu_g00N/` · `epoch0_val_batches/g00N/` · `train_g00N/`。进度 [`progress.md`](progress.md)。

---

## 模型发布

| 训练 | 发布 |
| --- | --- |
| `train_g001/` | **v0.1** → `models/v0.1/` |
| `train_g002/` | **v0.2** → `models/v0.2/` |
| `train_g00N/` | **v0.N** |

```text
models/v0.1/
  model.pt   info.json   card.json   card.svg
```

每发一版生成 **card.svg**（柱状示意：误差 / 通过率 / 相对 Epoch-0 成本）。示例：

![v0.1 model card](assets/model_card_example.svg)

---

## 仓库与本地

| | |
| --- | --- |
| `mindmap.md` | 科学步骤 |
| `AGENTS.md` | 命名与协作 |
| `progress.md` | 计算进度 |
| `assets/` | README 配图 |
| `src/nhc_deprot/` | 代码 |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src python -m pytest -q
```

AIMNet2 与 PySCF 分环境使用。
