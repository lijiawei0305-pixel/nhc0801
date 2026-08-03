# NHC0801 / nhc-deprot

氮杂环卡宾（NHC）脱质子：

```math
\mathrm{NHC{-}H^{+}} \rightarrow \mathrm{NHC} + \mathrm{H^{+}}
```

**NHC0801** 要做的是：用 **Pure-PySCF** 在 **Parent-Level P01** 协议下生成老师帧，在这些帧上微调 **AIMNet2**，再按完整科学 Validation 选出唯一 checkpoint，最后在密封的 **Final Test** 上只评估一次。

本地仓库名 `nhc-deprot`。服务器写入只允许 `$WJW/NHC0801`。  
这不是 `nhc-deprot-ranker` 的续作，也不是生产栈 `two_endpoint`（B3LYP/def2-SVP）。

科学步骤 0–12 的真值在 **`mindmap.md`**（方框 + 箭头）。协作规则在 **`AGENTS.md`**。

---

## 反应与标签

每个 **molecular root** 两端点：

| 端点 | charge | multiplicity |
| --- | ---: | ---: |
| NHC–H⁺（cation） | +1 | 单重态 |
| NHC（neutral） | 0 | 单重态 |

脱质子电子能标签（kcal·mol⁻¹，`lower_is_better`）：

```math
\Delta E_{\mathrm{elec}} = (E_{\mathrm{neutral}} - E_{\mathrm{cation}}) \times 627.509474
```

```math
\Delta E_{\mathrm{deprot}} = \Delta E_{\mathrm{elec}} - 6.28
```

6.28 为项目冻结的质子相关常数。  
**最终标签只来自 Parent-Level P01 的 Pure-PySCF 能量；AIMNet2 能量永不进标签。**

---

## Parent-Level P01 与 GAU_LOOSE

老师数据与 handoff 之后的完整优化，协议固定为 **Parent-Level P01**（`AGENTS.md`：Parent = P01 only）：

| 项 | 设定 |
| --- | --- |
| 体系 | 气相闭壳层 RKS |
| 泛函 / 基组 | `wb97m-d3bj` / `def2-TZVPP`（ωB97M-D3(BJ)） |
| 格点 | grid = 4 |
| SCF | `1e-9` |
| 协议 SHA256 | `227c22a527e567bc4de873ab743fe9f493779eccbb1a698d2913c87695ebf87a` |

mindmap 里写的 **parent-level PySCF/geomeTRIC**、**parent-level GAU**，指的就是这条 P01 完整优化，而不是生产栈或别的 DFT 级别。

AIMNet2 在自身势能面上的停点是 **GAU_LOOSE**（合同 `GAU_LOOSE_V001`）：

- geomeTRIC 五准则全满足  
- ASE LBFGS：`fmax = 0.10` eV/Å，`max_steps = 100`  
- **不是** 生产栈常用的 `fmax = 0.05`

完整科学路线（Epoch-0 / sci-val / Final Test 共用骨架）：

```text
冻结几何
  → AIMNet2 优化到 GAU_LOOSE
  → 身份 / 拓扑门
  → exact-byte handoff
  → 完整 Parent-Level P01 优化到最终 GAU（single_point_only=false）
  → parent-level 最终单点与脱质子标签
```

再与 **Pure-PySCF reference**（mindmap 步骤 2：直接 parent-level 优化得到的老师参考）对照。

训练目标是冻结两体 D3(BJ) 后的短程残差；推理时加回同一外部 D3，禁止静默重算 D3：

```math
E_{\mathrm{short}} = E_{\mathrm{P01,total}} - E_{\mathrm{D3}}, \quad
\mathbf{F}_{\mathrm{short}} = \mathbf{F}_{\mathrm{P01,total}} - \mathbf{F}_{\mathrm{D3}}
```

---

## 流水线（对照 mindmap 0–12）

| 步骤 | 内容 |
| ---: | --- |
| 0 | 冻结全部原始分子（两端点坐标、charge、multiplicity、结构 SHA256） |
| 1 | 按 **molecular root** 划分 Train / Validation / Final Test（互不相交；轨迹不跨 split） |
| 2 | **Pure-PySCF** 生成老师答案：parent-level 完整优化，逐步存几何、能量、力、lineage、protocol SHA |
| 3 | **Epoch-0**：官方未微调 AIMNet2（`aimnet2_wb97m_d3_0.pt`）走完整路线，得到微调前基线 |
| 4–5 | 只用 Train 老师帧训练；多 epoch、多种子存 checkpoint |
| 6–7 | 快速 Validation（固定 Val 帧 loss）筛短名单；**不得最终选模** |
| 8–9 | 完整科学 Validation + 冻结数值标定选出唯一 checkpoint |
| 10–12 | 冻结身份 → Final Test 一次 → 禁止考后换模型 / 改规则 |

当前规模化按分子组 **`g00N`**（每组 5 root = 字典序 3 Train + 2 Val）：

| 工作 | 对人说法 | 磁盘路径（在 `runs/nhc0801-g001/` 下） |
| --- | --- | --- |
| 老师帧 | g00N teacher | `teacher_gpu_g00N/` |
| Epoch-0 | g00N Epoch-0 | `epoch0_val_batches/g00N/` |

进度见 **`progress.md`**。命名词典见 `docs/NHC0801_命名与进度指南.md`。

---

## 硬规矩（与 AGENTS 一致）

- Parent **仅** P01；禁止把 `two_endpoint` B3LYP/def2-SVP 当 parent。  
- AIMNet2 停点 **仅** GAU_LOOSE（fmax 0.10），不是 0.05。  
- 划分单位是 molecular root，不是单帧。  
- Quick-val 不得最终选模。  
- Final Test 开发期只留密封 commitment；打开后只考一次。  
- 服务器新写仅 `$WJW/NHC0801`。  
- teacher / 训练 / Epoch-0 / live sci-val / Final Test 默认门禁关闭，需显式授权。

口径冲突优先级：`mindmap.md` → 冻结合同 → `AGENTS.md`。

---

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `mindmap.md` | 科学流水线真值 |
| `AGENTS.md` | 落盘与协作规则 |
| `PHASE_STATUS.md` | 阶段状态 |
| `progress.md` | g00N teacher / Epoch-0 进度 |
| `docs/contracts/` | GAU_LOOSE、数值标定、调度合同 |
| `docs/evidence/` | pilot 证据（只读） |
| `src/nhc_deprot/` | 库代码 |
| `scripts/` | 薄 CLI / 作业入口 |
| `tests/` | 合成 fixture 单测 |

---

## 本地开发

```bash
cd /path/to/nhc-deprot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src python -m pytest -q
```

不跑化学的预检：

```bash
PYTHONPATH=src python -m nhc_deprot.pipeline.mindmap_orchestrator
```

服务器环境不要混栈：

```bash
source $WJW/env/envs/mlff.sh    # AIMNet2
source $WJW/env/envs/molenv.sh  # PySCF
```

同步代码（示例）：

```bash
rsync -avz --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' \
  -e 'ssh -o BatchMode=yes' \
  /Users/cc/nhc-deprot/ nhc614:/home/plab/test/WJW/NHC0801/
```

---

## 相关文档

| 问题 | 文档 |
| --- | --- |
| 步骤细节 | `mindmap.md` |
| 写哪里、能不能实算 | `AGENTS.md` |
| 算到哪了 | `progress.md`、`PHASE_STATUS.md` |
| g00N / Epoch-0 命名 | `docs/NHC0801_命名与进度指南.md` |
| 选模数值规则 | `docs/contracts/NUMERIC_CALIBRATION_V001.yaml` |

内部科研代码。改科学口径先改 mindmap 与合同，再动实现。
