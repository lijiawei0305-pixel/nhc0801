# NHC0801：基于 Parent-Level 电子结构老师数据的 AIMNet2 微调与科学验证

**项目代号**：NHC0801（本地仓库 `nhc-deprot`）  
**科学主题**：氮杂环卡宾（NHC）脱质子反应  
\[
\mathrm{NHC\text{-}H^{+}} \;\rightarrow\; \mathrm{NHC} + \mathrm{H^{+}}
\]  
在 **Train / Validation / Final Test** 严格按分子根（molecular root）隔离的前提下，以 **Pure-PySCF 父级（Parent-Level Protocol P01）** 轨迹为老师数据，对 **AIMNet2** 进行有监督微调，并经由完整科学 Validation 选定唯一 checkpoint，最终在密封的 Final Test 上一次性评估。

本仓库实现的是 **mindmap 所规定的科学流水线的工程化**，而非既有生产排序器（ranker）执行链的延续。

---

## 1. 研究动机与定位

低保真力场或半经验方法可对大规模 NHC 候选库进行粗筛，但脱质子相关的高保真电子能与几何响应仍依赖量子化学。将 **AIMNet2** 作为几何预条件器并在 **父级 DFT 表面**上完成最终收敛与标签计算，可在保持标签可审计性的同时，降低后续 PySCF 几何优化负担。

本项目坚持三点原则：

1. **标签真理只来自父级 Pure-PySCF**，AIMNet2 能量不得写入最终脱质子标签；  
2. **数据划分以 molecular root 为原子**，阳离子、中性体及其全部轨迹不得跨 Train / Validation / Final Test 拆分；  
3. **快速 Validation（帧级 loss）仅用于筛选候选 checkpoint**，最终模型必须由完整科学 Validation 决定。

---

## 2. 理论与协议（不可混用）

### 2.1 父级电子结构：Parent-Level Protocol P01

| 项目 | 设定 |
| --- | --- |
| 体系 | 气相闭壳层 RKS |
| 泛函 / 基组 | \(\omega\)B97M-D3(BJ) / def2-TZVPP |
| 积分格点 | grid level 4 |
| SCF 收敛 | \(10^{-9}\) |
| 色散 | 两体 D3(BJ)；ATM = false；VV10 = false |
| 协议指纹 | SHA256 `227c22a5…95ebf87a` |

脱质子电子能标签（kcal·mol⁻¹）：

\[
\Delta E_{\mathrm{elec}} = \big(E_{\mathrm{neutral}} - E_{\mathrm{cation}}\big)\times 627.509474
\]
\[
\Delta E_{\mathrm{deprot}} = \Delta E_{\mathrm{elec}} - 6.28
\]

其中 \(6.28\) 为项目冻结的质子相关常数；**越低越好**（`lower_is_better = true`）。

### 2.2 AIMNet2 停点：GAU_LOOSE

AIMNet2 / ASE LBFGS 在 **自身势能面** 上须同时满足五条 GAU_LOOSE 准则（能量变化、梯度 RMS/最大值、位移 RMS/最大值），并配合 ASE `fmax = 0.10` eV·Å⁻¹、`max_steps = 100`。  
**禁止**将生产栈 `fmax = 0.05` eV·Å⁻¹ 或 B3LYP/def2-SVP 两终点协议当作父级默认。

### 2.3 训练目标：短程残差

在冻结的两体 D3(BJ) 投影收据上构造：

\[
E_{\mathrm{short}} = E_{\mathrm{parent,total}} - E_{\mathrm{D3}},\quad
\mathbf{F}_{\mathrm{short}} = \mathbf{F}_{\mathrm{parent,total}} - \mathbf{F}_{\mathrm{D3}}
\]

推理时须加回同一外部 D3。样本权重策略为  
`equal_candidate_then_equal_endpoint_then_uniform_frames`（键名 `sample_weight`），各 split 权重和为 1。

### 2.4 明确禁止的历史栈

| 栈 | 说明 |
| --- | --- |
| 生产 `two_endpoint` B3LYP-D3(BJ)/def2-SVP | **不得**作为本项目 parent |
| 以 quick-val loss 直接选定最终模型 | 与 mindmap 冲突 |
| 将历史 `$WJW/checkpoints/*.pt` 当作 epoch-0 | 存在金标污染风险 |

相关历史文档已隔离于 `docs/archive/forbidden_b3lyp_stack/`，仅供溯源，不可作为计算协议源。

---

## 3. 完整 Mindmap 科学流水线（步骤 0–12）

下列流程是本项目的 **科学真值**（权威文件：`mindmap.md`）。

```text
0  冻结全部原始分子（cation + neutral；坐标、电荷、多重度、结构 SHA256）
1  按 molecular root 划分 Train / Validation / Final Test（三者互不相交）
2  Pure-PySCF 生成老师帧（逐步几何、能量、力、lineage、protocol SHA）
3  Epoch-0 基线：官方未微调 AIMNet2 → GAU_LOOSE → handoff → 完整 parent GAU
4  仅用 Train 帧训练 AIMNet2（能量与力）
5  多 epoch 保存 checkpoint（多种子结果全部保留）
6  快速 Validation：固定 Val 帧上的加权 loss（不跑新 DFT，不反传）
7  初步短名单：由帧 loss 确定性筛选少量候选（非最终选择）
8  完整科学 Validation：对短名单执行全路线（见下节）
9  Validation 按冻结数值标定规则选出唯一 checkpoint
10 冻结模型、划分、合同、源码 commit 与运行时指纹
11 Final Test 一次性评估（训练与选模阶段不可见其身份）
12 禁止 Test 后改阈值、换模型或事后挑选
```

### 3.1 步骤 0–1：冻结与划分

每个 molecular root 必须同时包含：

- **阳离子** NHC–H⁺：电荷 \(+1\)，单重态；  
- **中性体** NHC：电荷 \(0\)，单重态。

划分单位为 root，而非单帧。Final Test 在开发阶段仅保留 **密封 commitment（SHA256 + root 数）**，身份不对训练暴露。

当前 pilot 开发可见划分（证据见 `docs/evidence/pilot_day1/`）：

| Split | Root 数（pilot） |
| --- | ---: |
| Train | 3 |
| Validation | 2 |
| Final Test | 2（密封，不暴露） |

### 3.2 步骤 2：老师数据

对每个 endpoint 执行 **完整** parent-level PySCF/geomeTRIC 优化至最终 GAU，并逐步保存几何、电子能与原子力。  
Pilot 服务器上已有通过审计的加权发展集证据（235 帧量级）；规模化生成仍需单独授权。

### 3.3 步骤 3：Epoch-0 基线

使用官方权重 `aimnet2_wb97m_d3_0.pt`（仅 `_0` 成员作为当前已知可用 epoch-0），在 Validation 冻结几何上走通：

```text
官方 AIMNet2 → GAU_LOOSE → exact-byte handoff → 完整 P01 优化至 GAU → 父级单点 → 标签
```

得到“微调前”对照基线。**执行默认关闭**，需 `epoch0_execution` 授权。

### 3.4 步骤 4–5：训练与检查点

- 仅读取 Train split 中的父级帧；  
- 加权多任务 loss（能量 / 力）；  
- 多种子、多 epoch；**全部 outcome 保留**；  
- 快速 Validation 每个 epoch 可运行，但  
  **`quick_validation_may_select_final_model = false`**。

### 3.5 步骤 6–7：快验与短名单

在固定 Validation 帧上计算全局加权 loss，用于过拟合监测与确定性短名单（如每 seed 最多 4 个），**不构成最终选模**。

### 3.6 步骤 8：完整科学 Validation（本仓库核心实现）

对短名单中每个 checkpoint、每个 Validation root 的两端点执行：

```text
冻结初始几何
  → 候选 AIMNet2 优化至 GAU_LOOSE
  → 身份 / 拓扑 / 有限性门
  → exact-byte handoff
  → 完整 Parent-Level P01 优化至最终 GAU
  → 父级最终单点
  → 脱质子电子能标签
  → 与 Pure-PySCF 参考标签及 epoch-0 对照
```

Handoff 标定语义：

| 状态 | 含义 | 是否继续完整 parent 优化 |
| --- | --- | --- |
| `HANDOFF_CALIBRATION_PASS` | 首梯度进入 GAU_LOOSE 门 | 是 |
| `HANDOFF_CALIBRATION_MISS` | SCF/身份合法但梯度未进门 | 是 |
| `FAILED_PARENT_HANDOFF` | SCF / 非有限 / 身份 / 拓扑失败 | **否** |
| `FINAL_PARENT_GAU_CONVERGED` | 几何 GAU + 最终单点完成 | — |

实现模块：`src/nhc_deprot/pipeline/scientific_validation.py`  
默认使用可注入引擎；**单元测试采用仿真后端**，不触发真实量子化学。实算需 `scientific_validation_live=true` 并接入服务器 AIMNet2 / PySCF 引擎。

### 3.7 步骤 9：Validation 选模

在冻结数值标定 `docs/contracts/NUMERIC_CALIBRATION_V001.yaml` 下，按 mindmap 优先级选择：

1. 结构、拓扑、电荷、多重度全部正确；  
2. 无碎裂或错误成键（灾难失败预算）；  
3. 父级标签误差在阈值内；  
4. 相对 epoch-0 不退化；  
5. 再比较 PySCF 几何工作量、SCF 循环、墙钟时间等负担指标。

接口：`select_after_scientific_validation(...)`。

### 3.8 步骤 10–12：冻结、Final Test、禁事后挑选

冻结 checkpoint SHA、各 split、GAU_LOOSE、P01、数值标定、源码 commit 与 runtime 指纹后，**仅一次**打开 Final Test。Test 后禁止改阈值、换模型或删除失败分子。

---

## 4. 仓库结构（摘要）

```text
mindmap.md                 # 科学流水线真值
AGENTS.md                  # 代理 / 协作者约束
PHASE_STATUS.md            # 阶段状态
docs/NHC0801_命名与进度指南.md  # g001 / e0 / Val-only 等命名词典 + 如何读进度
docs/contracts/            # GAU_LOOSE、数值标定等冻结合同
docs/evidence/pilot_day1/  # 改名后的 pilot 证据（split / 加权集结果等）
docs/archive/              # 禁止栈历史文档隔离区
src/nhc_deprot/
  contracts/               # P01、TVT 门、禁止栈
  data/                    # split、路径、加权 NPZ 审计
  pipeline/                # handoff、科学 Validation、编排与阻塞诊断
  training/                # 加权 loss、trainer adapter（无 live 入口）
tests/                     # 合成 fixture；不依赖 HPC
```

---

## 5. 环境与计算边界

| 项目 | 约定 |
| --- | --- |
| 本地 | 编辑、审计、单元测试 |
| 服务器写入根 | **仅** `/home/plab/test/WJW/NHC0801` |
| AIMNet2 / ML | `source $WJW/env/envs/mlff.sh` |
| PySCF | `source $WJW/env/envs/molenv.sh`（**禁止与 ML 混栈**） |
| Epoch-0 权重 | `~/.cache/aimnet/aimnet2_wb97m_d3_0.pt` |

默认门禁（未显式授权前均为关闭）：

```text
teacher_pyscf_authorized
aimnet2_train_authorized
epoch0_execution
scientific_validation_live
final_test_open
modify_wjw_outside_NHC0801
scheduler_submission
```

---

## 6. 快速开始（开发者）

### 6.1 安装与测试

```bash
cd /path/to/nhc-deprot   # 或克隆本仓库
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src python -m pytest -q
```

### 6.2 Mindmap 预检编排（不跑化学）

```bash
PYTHONPATH=src python -m nhc_deprot.pipeline.mindmap_orchestrator
```

### 6.3 训练阻塞诊断

```bash
PYTHONPATH=src python -c "
from nhc_deprot.pipeline.training_blockers import *
print(format_readiness_report(assess_training_readiness()))
"
```

### 6.4 科学 Validation 合同摘要

```bash
PYTHONPATH=src python -c "
from nhc_deprot.pipeline.scientific_validation import route_contract_summary
import json; print(json.dumps(route_contract_summary(), indent=2))
"
```

---

## 7. 当前状态与仍开放的阻塞

| 能力 | 状态 |
| --- | --- |
| 开发 split / 加权数据集读取与审计 | 已实现 |
| 数值标定预注册 | 已冻结 |
| 科学 Validation **writer**（步骤 8–9 合同与聚合） | **已实现**（live 引擎接入另授权） |
| 多 seed 训练循环 | 未完成 |
| Epoch-0 / 实算 Validation / 实训练 | 默认禁止，待授权与资源 |

典型仍开放硬阻塞：源码 commit 未冻结、epoch-0 基线未跑、live 训练未授权、HPC 资源 claim 可能失败。详见 `docs/science/TRAINING_BLOCKERS_AND_SOLUTIONS.md`。

---

## 8. 与相关仓库的关系

| 来源 | 关系 |
| --- | --- |
| `nhc-deprot-ranker` | 只读借用科学合同与产品表；**禁止回写** |
| science-pilot V004 | pilot 证据与可移植纯逻辑；已改名纳入 `docs/evidence/pilot_day1/` |
| `$WJW` 生产树 | 环境与历史数据只读；**新写入仅限 `NHC0801`** |

---

## 9. 引用与可重复性建议

发表或归档结果时，建议同时固定：

- Parent 协议 SHA256 与 GAU_LOOSE 合同文本；  
- Train / Validation / Final Test 的 root 列表或密封 commitment；  
- 数值标定文件版本（`NUMERIC_CALIBRATION_V001`）；  
- 代码 commit、运行时环境脚本哈希、checkpoint SHA256；  
- 科学 Validation 与 Final Test 的 receipt 目录。

---

## 10. 许可证与贡献

内部科研代码库。贡献前请阅读 `AGENTS.md` 与 `mindmap.md`，任何改变科学口径、数据划分或服务器写边界的修改须事先确认。实算步骤必须分项授权，禁止在未通过预检时启动训练或打开 Final Test。

---

*文档与代码以 mindmap 为准；摘要冲突时以 `mindmap.md` 及冻结 YAML/JSON 绑定为最高优先级。*
