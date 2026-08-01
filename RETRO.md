# RETRO.md — 工程复盘日志（NHC0801）

> **用途**: 记录开发过程中反复出现的错误、vibe coding 中遇到的典型困难、以及解决方案。  
> 当类似问题再次出现时，**先查本文**，避免重复踩坑。
>
> **维护规则**:
>
> 1. 每条记录必须包含：**现象、根因、是否已解决、解决方案（或规避方式）**。
> 2. 未解决的问题必须标注 `[未解决]`，已解决的标注 `[已解决]`。
> 3. 新条目**追加在对应分类末尾**，不要打乱已有顺序。
> 4. 若某条经验已固化为 `AGENTS.md` 硬约束或棘轮条目，标注 **「已升级为规则」** 并注明位置。
> 5. 科学口径变更仍以 `mindmap.md` 为准；本文件只记工程/协作坑，不另立科学真理。

---

## 0. 索引（按主题）

| 分类 | 说明 |
| --- | --- |
| A. 科学栈与协议混淆 | two_endpoint / B3LYP vs P01 / GAU_LOOSE |
| B. Train–Val–Test 纪律 | 泄漏、密封 FT、235 硬编码 |
| C. 训练阻塞与门禁 | BLOCKED_BEFORE_TRAINING 误清障 |
| D. 仓库布局与 vibe 落盘 | 脚本/规划/文档放错目录 |
| E. 环境与服务器边界 | 混栈、越权写 `$WJW` |
| F. Git / 协作 | source freeze、脏树、推送 |
| G. 工具与测试 | pytest、仿真引擎、import 路径 |

---

## A. 科学栈与协议混淆

### A1. `[已解决]` 把 production two_endpoint 当 parent 默认

- **现象**: 文档/旧代码大量出现 B3LYP/def2-SVP、`fmax=0.05`，与 mindmap 父级协议冲突，agent 易抄错。
- **根因**: ranker 生产栈与 V004/P01 科学栈并存；mtime 新文件不一定是权威。
- **解决方案**: Parent 仅 P01（ωB97M-D3(BJ)/def2-TZVPP）；AIMNet2 停点仅 GAU_LOOSE（fmax 0.10）。禁止栈写入 `contracts/forbidden_stacks.py`；干扰文档隔离至 `docs/archive/forbidden_b3lyp_stack/`。
- **已升级为规则**: `AGENTS.md` → Science non-negotiables / Forbidden；`docs/archive/forbidden_b3lyp_stack/README.md`。

### A2. `[已解决]` 用 quick-val loss 选最终模型

- **现象**: 历史 `finetune.py` 以验证集帧 loss 取 best checkpoint。
- **根因**: 与 mindmap 步骤 6–9 冲突（快验只短名单，科学 Validation 才选模）。
- **解决方案**: `quick_validation_may_select_final_model=false`；选模走 `select_after_scientific_validation` + `NUMERIC_CALIBRATION_V001.yaml`。
- **已升级为规则**: `AGENTS.md`；`contracts/forbidden_stacks.assert_quick_val_not_final_selector`。

### A3. `[已解决]` AIMNet2 能量写进脱质子标签

- **现象**: 标签混入 ML 预测能。
- **根因**: 预条件器与标签源未分离。
- **解决方案**: 标签只来自 parent 电子能公式；handoff 显式 `aimnet2_energy_enters_label=false`。
- **已升级为规则**: `AGENTS.md`；`pipeline/scientific_validation.py`。

---

## B. Train–Val–Test 纪律

### B1. `[已解决]` 为了开发方便打开 Final Test 身份

- **现象**: split JSON 含 `final_test` 列表或加载 Test 几何。
- **根因**: 想“凑齐三路”调试。
- **解决方案**: 开发期仅密封 commitment（sha256 + root_count）；`development_split` 拒绝 `final_test`/`test` 键。
- **已升级为规则**: `AGENTS.md` gates；`data/development_split.py`。

### B2. `[已解决]` 把 pilot「235 帧」写死为全局常量

- **现象**: writer/审计硬编码 235/123/112，换规模即失效。
- **根因**: V004 pilot 证据数被当成科学真理。
- **解决方案**: 帧数从 split/manifest/NPZ 派生；可选 `expected_*` 仅作绑定审计。
- **已升级为规则**: `AGENTS.md` 落盘约定；`data/weighted_dataset.py`。

### B3. `[已解决]` 静默重算 D3

- **现象**: 训练前再跑一遍 D3 投影，字节/协议漂移。
- **根因**: 图省事未消费冻结收据。
- **解决方案**: `d3_recomputation_performed` 必须为 false；只读 frozen receipt。
- **已升级为规则**: `AGENTS.md`；加权集审计。

---

## C. 训练阻塞与门禁

### C1. `[已解决]` 以为「实现了 writer」就可以 live 训练

- **现象**: sci-val writer 落地后仍想直接开训。
- **根因**: 混淆「结构实现」与「live 授权 + epoch-0 收据」。
- **解决方案**: 阻塞矩阵分项；writer RESOLVED ≠ `aimnet2_train_authorized`。查 `training_blockers.assess_training_readiness()`。
- **已升级为规则**: `docs/science/TRAINING_BLOCKERS_AND_SOLUTIONS.md`；`AGENTS.md` Gates。

### C2. `[未解决]` Epoch-0 全路线基线未跑

- **现象**: `EPOCH_ZERO_FULL_ROUTE_BASELINE_NOT_AVAILABLE`。
- **根因**: 需空闲 CPU、官方 `_0` 权重、`epoch0_execution` 授权；尚未 live 执行。
- **解决方案（规避）**: 不在本机跑；等资源 claim 后在 `$WJW/NHC0801/runs/` 写收据。勿用历史 checkpoints 冒充 epoch-0。

### C3. `[未解决]` 多 seed 训练循环缺失

- **现象**: mindmap 4–5 无完整 trainer epoch 循环。
- **根因**: 有意先做数据层与 sci-val 合同。
- **解决方案（规划）**: 实现于 `src/nhc_deprot/training/`，CLI 薄封装放 `scripts/`；禁止 quick-val 终选。

### C4. `[已解决]` SOURCE_COMMIT 未冻结

- **现象**: readiness 报 no .git / dirty worktree。
- **根因**: 冷启动无仓库或未提交。
- **解决方案**: `git init` + 干净 commit 并推送 GitHub；脏树不算冻结。
- **已升级为规则**: `AGENTS.md` Training blockers；`training_blockers._git_source_frozen`。

---

## D. 仓库布局与 vibe 落盘

### D1. `[已解决]` 脚本/库代码/规划文档乱放根目录

- **现象**: 根目录堆 `tmp_*.py`、`plan.md`、长 JSON，难审计。
- **根因**: vibe coding 未约定落盘路径。
- **解决方案**: 见 `AGENTS.md` → **Where to put files**（库代码 `src/`，CLI `scripts/`，规划 `docs/plans/`，复盘 `RETRO.md`）。
- **已升级为规则**: `AGENTS.md` § Where to put files。

### D2. `[已解决]` phase9b/V004 文件名干扰注意力

- **现象**: agent 以旧文件名当权威。
- **根因**: 交接副本未改名。
- **解决方案**: 规范名在 `docs/evidence/pilot_day1/`；`docs/extracted/v004/` 仅 redirect。
- **已升级为规则**: `AGENTS.md` Key paths。

---

## E. 环境与服务器边界

### E1. `[已解决]` ML 与 PySCF 混栈 PYTHONPATH

- **现象**: import 冲突或 silent wrong libs。
- **根因**: 同一 shell 混 `mlff.sh` 与 `molenv.sh`。
- **解决方案**: 分 shell；脚本只 `source $WJW/env/envs/<stack>.sh`，禁止 `source ~/.bashrc`。
- **已升级为规则**: `AGENTS.md` Server / env。

### E2. `[已解决]` 写入 `$WJW` 非 NHC0801 路径

- **现象**: 污染生产 data/checkpoints。
- **根因**: 沿用 ranker 路径习惯。
- **解决方案**: 新写入唯一根 `$WJW/NHC0801`；rsync 无 `--delete`。
- **已升级为规则**: `AGENTS.md` Gates / Server。

---

## F. Git / 协作

### F1. `[已解决]` 对 science-pilot 脏工作树 `git reset/clean`

- **现象**: 可能毁掉未跟踪 V004 产物。
- **根因**: 想「弄干净」再抄代码。
- **解决方案**: pilot 只读；适配写入本仓库；禁止 reset pilot。
- **已升级为规则**: `AGENTS.md` Work style。

### F2. `[已解决]` 提交 `configs/*.local.yaml` 或私钥

- **现象**: 凭据进 git。
- **根因**: gitignore 未盯紧。
- **解决方案**: 仅 `configs/server.example.yaml` 入库；`*.local.yaml` 与 `private/` 忽略。
- **已升级为规则**: `.gitignore`；`AGENTS.md` Server。

---

## G. 工具与测试

### G1. `[已解决]` 单测依赖 HPC / 真实 PySCF

- **现象**: 无服务器则 CI 全红。
- **根因**: 测试绑死 live 引擎。
- **解决方案**: 仿真 `SimulatedAimnet2Engine` / `SimulatedParentEngine`；`live=True` 需门禁。
- **已升级为规则**: `AGENTS.md` Work style；`tests/test_scientific_validation.py`。

### G2. `[已解决]` `PYTHONPATH` 未设导致 import 失败

- **现象**: `ModuleNotFoundError: nhc_deprot`。
- **根因**: 包在 `src/` 布局。
- **解决方案**: `PYTHONPATH=src python -m pytest -q` 或 `pip install -e .`。
- **已升级为规则**: `AGENTS.md` Before any code change；`pyproject.toml`。

---

## 变更记录（本文件自身）

| 日期 | 说明 |
| --- | --- |
| 2026-08-01 | 初创：从 NHC0801 冷启动～sci-val writer 会话提炼 A–G 条目 |
