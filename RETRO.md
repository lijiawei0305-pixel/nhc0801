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

### A4. `[已解决]` 照抄 FT 教程 `forces_weight=10` 会误判「力权重无用」

- **现象**: 本数据 val `E_mse≈12.4` vs `F_mse≈0.42`（≈30:1）。`scaled_training_loss` 按 `w/(w_e+w_f)` 归一后，`forces_weight=10` 有效 E:F 仍约 **3:1 能量主导**。
- **根因**: 教程默认面向已做参考能对齐的体系；本项目标签是未对齐绝对总能（T3），量级比不同。
- **解决方案**: 改力权重前先读 `campaign_receipt_live.json` 实测 E/F 量级；网格应覆盖到 ~100 才可能力主导。**禁止**未经量级审计就把 10 当「力主导默认」。
- **已升级为规则**: `AGENTS.md` → **模型训练注意事项 T4**（勿重推）。

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

### G3. `[已解决]` M0 mypy 基数偏差（计划 12 vs 实测 30）

- **现象**: `docs/prompt.md` / 计划写「mypy 12 errors in 9 files」；M0 开工实测约 **30** errors（文件面更大）。
- **根因**: 计划基数在代码演进后过时；只按 12 会以为基线已清或漏修。
- **解决方案**: 以 `mypy src tests scripts` **当前输出为真**；清零后固化 `pyproject.toml` `[tool.mypy]`；后续任务以 0 error 棘轮，不以历史数字为准。
- **状态**: M0 后 0 errors；M12 全量门仍绿。

### G4. `[已解决]` M1 计划「无条件改 opt_steps」vs 缺省兼容

- **现象**: 计划草稿像要求凡 optimize 就改 `opt_steps` / 关 `opt_steps_is_maxcap`；服务器上已有 teacher 进程不能被行为漂移。
- **根因**: 全轨迹是**新字段**能力，不是替换旧 contract。
- **解决方案**: 仅当请求体带 `trajectory_out_path` 时启用 JSONL + 真实求值次数；**缺省路径与旧输出逐字段兼容**。单测覆盖「无 trajectory_out_path ≈ 旧版」。
- **已升级为规则**: `AGENTS.md` T5；M1 验收「缺省时行为与现在逐字节一致」。

### G5. `[已解决]` M2 `trajectory_out_path` 经 TypeError 注入

- **现象**: `live_teacher` / live_epoch0 调用层未把 `trajectory_out_path` 做成稳定公开 kwargs；硬塞会 TypeError。
- **根因**: 引擎 `_call` / worker 适配面与新可选字段不同步。
- **解决方案**: 优先走支持的 kwargs；遇 `TypeError` 回退到经 `_call` 注入请求体字段（兼容旧签名）。勿假设所有调用点已升级 API。
- **状态**: 单测 mock 覆盖；live 仍待 M13 验证。

### G6. `[已解决]` M12 shortlist 默认扫不到 `runs/<run_id>/seed_*`

- **现象**: `run_shortlist_campaign` 在 `train_g00N/` 下 `glob("seed_*")` 失败，因 M8 产物在 `train_g00N/runs/<run_id>/seed_*`。
- **根因**: shortlist 写于 legacy 扁平 seed 布局；未随 run 子目录自动改写默认扫描。
- **解决方案（规避）**: 集成侧显式 `train_dir=layout.train_batch_run_dir(batch_id, run_id)`。长期可让 shortlist 递归 `runs/*/seed_*`（跨模块，非 M12 范围）。
- **状态**: e2e `tests/test_m12_e2e_dry_run.py` 已按规避串通。

---

## H. Generation / 并行资源

### H1. `[已解决]` 并行策略未写进代码导致 vibe 时乱开进程

- **现象**: 想多分子并行但无档案/claim 模型。
- **根因**: 只有 handoff 叙述，无 NHC0801 模块。
- **解决方案**: `RESOURCE_PROFILES_V001.yaml` + `resources/{profiles,claim,worker_pool}.py`；策略 S（single→dual）；`live_dispatch_enabled=false`。
- **已升级为规则**: `AGENTS.md` 落盘表；`docs/plans/20260801_clean_restart_and_parallel_compute_plan.md`。

### H2. `[未解决]` 服务器 live claim 仍可能 CPU busy

- **现象**: handoff V002 REJECTED。
- **根因**: 共享节点 0,2-27 被占。
- **解决方案（规避）**: 代码可评估快照；真采样与 chemistry 待资源空闲 + 用户授权。勿用 112 逻辑核硬开。

---

## 变更记录（本文件自身）

| 日期 | 说明 |
| --- | --- |
| 2026-08-01 | 初创：从 NHC0801 冷启动～sci-val writer 会话提炼 A–G 条目 |
| 2026-08-01 | 追加 H：generation + resource 模型（C/S，无 live） |
| 2026-08-03 | M12：追加 A4（forces_weight=10）、G3–G6（mypy 基数 / M1 兼容 / M2 TypeError / shortlist 路径） |

### R-gpu-autofill-zombie-reap. `[已解决]` 空闲 GPU 不补位

- **现象**: 作业已 `JOB_EXIT PASS`，nvidia-smi 卡空，但 autofill 长时间 `free_gpus=[]` 不 CLAIM。
- **根因**: `running` 用 `os.kill(pid,0)` 判活；僵尸进程 Z 仍返回成功，假占用 GPU。
- **方案**: 读 `/proc/pid/status` 将 Z 视为结束；日志有 `JOB_EXIT` 即 reap；`compute_steward.py` 周期强制 reap + 重启守护。
- **权威**: `docs/contracts/COMPUTE_DISPATCH_V001.md`；守护 `scripts/nhc0801_gpu_autofill_daemon.py` / `nhc0801_compute_steward.py`。

### R-e0-handoff-grad-key. `[已解决]` g002 Epoch-0 Val 全 FAILED_PARENT_HANDOFF

- **现象**: g002 Epoch-0 两根 Val、四个 endpoint 全 `FAILED_PARENT_HANDOFF` + `ANALYTIC_GRADIENT_UNAVAILABLE`；pure-PySCF 参考路线 PASS；AIMNet2 预优化也做过。
- **根因**: **字段名不一致**，不是分子算不动。  
  - worker `nhc0801_pyscf_parent_worker.py` → `gradient_hartree_per_bohr`  
  - sci-val 读 `first.get("gradient_hartree_bohr")`（仿真引擎键名）  
  → 梯度实际有，被当成 None → 误判 handoff 失败。pure 路径不跑 first_gradient 检查，故纯 Parent 能过。
- **方案**: `scientific_validation` 双键兼容；`LiveParentP01Engine.first_gradient` 归一化为 `gradient_hartree_bohr`。单测覆盖。**已跑完的 g002 收据不会自动变好**，需用修后代码重跑 g002 Epoch-0。g003+ 若在修前已进入 first_grad 同样会中招，rsync 后重启队列。
- **权威**: `src/nhc_deprot/pipeline/scientific_validation.py` · `live_epoch0.py`。

### R-ema-export-cpu-alias. `[已解决]` EMA 导出在 CPU 上会静默变成 raw 权重

- **现象**: 无（**隐患**，非现行故障）。生产训练走 CUDA，落盘的 `epoch_*.pt` 一直是正确的 EMA 权重。
- **根因**: `live_aimnet2.export_checkpoint` 原本在 `_use_ema_weights()` 窗口内做
  `state = {k: v.detach().cpu() for ...}`，而 `torch.save` 在窗口**之外**。
  `Tensor.cpu()` 在张量已在 CPU 上时**返回自身**（不拷贝），`.detach()` 又共享 storage，
  于是 `state` 别名活参数；`_use_ema_weights` 退出时 `param.data.copy_(saved)` 是**原地**写，
  把 raw 值写回同一块内存 → `torch.save` 落盘的是 raw，而 meta 里 `ema_decay` / `ema_enabled` 照样为真。
  CUDA 上 `.cpu()` 是真拷贝，所以只在 `device="cpu"` 时触发。
- **关键教训**: **`ema_enabled` 只是配置回声，不是证据。** 任何「读 checkpoint meta 校验 EMA」的审计脚本
  在这个失效模式下都会报 PASS。唯一能证伪的做法是**把导出的 `.pt` 读回来**与活参数比 L2。
- **解决方案**:
  - `_core_state_snapshot()` 加 `.clone()`，快照与设备无关；
  - `_checkpoint_payload(..., weight_kind)` 让每个 `.pt` 自述 `"ema"` / `"raw"`；
  - `export_raw_audit_sibling()` 在**末 epoch** 写 `epoch_NNNN.raw.pt`，并重读已导出的 `.pt` 比对；
  - `ema_export_audit()` 双向 fail-closed：EMA 开却零差异 → `EMA_EXPORT_IS_RAW`；
    EMA 关却有差异 → `EMA_EXPORT_UNEXPECTED_DIVERGENCE`；
  - `multi_seed_trainer` 仅在 `epoch == epochs` 且非 dry-run 时接线，审计写进 `epoch_NNNN.meta.json`。
- **不需要重跑**: 现有 144 个 `.pt` 由 CUDA 路径产出，EMA 正确；只是缺 `.raw.pt` 对照与 `weight_kind`。
- **测试**: `tests/test_ema_export_audit.py`（16）+ `tests/test_multi_seed_trainer_runs.py`（3）。
  注意 `temporary_array_swap` **重绑定** dict 项，不是原地写，**不能**用来复现这个 bug；
  测试里另写了 `np.copyto` 版的 in-place 夹具来镜像 `param.data.copy_()` 的语义。

### R-prescreen-rmsd-basin-quantized. `[已解决/口径]` 预筛 RMSD 换设备就换 basin，不可复现

- **现象**: 同一批权重、同一套参考几何（钉在归档 fc=2）、同一 250 步预算，只把
  CUDA 换成 CPU 重跑 seed 20260730 的 16 个重叠候选 → **4/16（25%）的 `mean_rmsd`
  跳变约 0.05 Å（相对 ~40%）**。`e1f100_mlp ep10` 与 `e1f1_mlp ep10` 直接**互换**。
- **根因**: `mean_rmsd_to_reference_angstrom` 是 ASE LBFGS 从固定起点弛豫**之后**量的。
  落进哪个局部极小是**离散选择**，浮点级扰动就能翻。seed 730 的 64 个观测聚成
  **3 个簇**（0.122–0.128 / 0.171–0.193 / 0.203–0.231），簇内跨度 << 簇间间隔。
  这不是「噪声大」，是**指标近似离散标签**。
- **连带**: 这解释了此前「seed 方差 >> 配方方差、前 9 名全是 seed 730」——那是 basin
  归属分层，不是平滑的配方效应。`live_phase1_v002` 的 RMSD 排名有约 1/4 是设备相关的。
- **相对稳定的指标**: `mean_force_rmse_at_reference_ev_per_a` 在参考几何上单点求值，
  不经弛豫，重跑偏差仅 ~1e-3–1.6e-2（2–8%），**无换簇**。`mean_aimnet2_steps` 居中。
- **口径（已升级为规则）**:
  1. **RMSD 单值不得作为配方/epoch 的排序依据**；只能作 basin 归属的离散读数
     （报「落在哪个簇」而非小数点后四位）。
  2. 跨设备/跨环境比较预筛结果前，必须先跑重叠候选的复现校验。
  3. T1 三键里，**force RMSE 是目前唯一可跨运行比较的连续量**。
- **产物**: `pre_screen_g001/live_seed730_epoch_axis_v1/`（48 候选全 epoch 轴 + `epoch_curve.json`
  + `paired_recipe_contrast.json`）；代码 `pipeline/paired_recipe_contrast.py`、
  `pre_screen.load_teacher_references_for_batch(teacher_batch_dir=...)` 只读钉参考集。
- **注意**: 归档 `_archive_teacher_maxsteps100_frame2_*` 是当前唯一有 `is_terminal` 的
  Val 参考；规范 `teacher_gpu_g001/` 正在被 m250 重算（无 manifest）。两套参考**不可混用**。
