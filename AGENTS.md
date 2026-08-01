# AGENTS.md — NHC0801 / nhc-deprot

Mindmap-driven AIMNet2 fine-tune on **Parent-Level P01** teacher frames.  
Local: `/Users/cc/nhc-deprot`. Server writes: **only** `$WJW/NHC0801`.

Not a continuation of `nhc-deprot-ranker` Phase 9B. Not production `two_endpoint`.

---

## Before any code change

1. Read **`mindmap.md`** + this file + **`PHASE_STATUS.md`**.
2. Skim **`RETRO.md`** for similar past failures (same category).
3. Map work to a **mindmap step 0–12**. Do not skip gates.
4. If the task is non-trivial: write a short plan under **`docs/plans/`** *before* vibe coding.
5. Put new files only where **Where to put files** allows.
6. Run `PYTHONPATH=src python -m pytest -q` after code changes.
7. Live chemistry / training / Final Test only with **explicit user authorization**.

Conflict order: **mindmap.md** → frozen YAML/JSON → this file → `RETRO.md` lessons → pilot evidence.

---

## Where to put files（落盘硬规则）

| 内容 | 目录 | 命名 / 说明 |
| --- | --- | --- |
| **可 import 库代码** | `src/nhc_deprot/` | 包内模块；按域：`contracts/` `data/` `pipeline/` `training/` `resources/` |
| **单元测试** | `tests/` | `test_<topic>.py`；合成 fixture，不依赖 HPC |
| **CLI / 作业入口** | `scripts/` | 薄封装 only；逻辑在 `src/`。见 `scripts/README.md` |
| **动手前的规划** | `docs/plans/` | `YYYYMMDD_<topic>_plan.md`。见 `docs/plans/README.md` |
| **冻结合同** | `docs/contracts/` | 版本化 YAML/JSON（GAU_LOOSE、数值标定等） |
| **科学讨论 / 阻塞说明** | `docs/science/` | 非合同正文 |
| **pilot 证据（改名后）** | `docs/evidence/` | 只读绑定；勿再堆 phase9b 文件名 |
| **禁止栈 / 历史干扰** | `docs/archive/` | 只读隔离，不当协议源 |
| **阶段勾选** | `PHASE_STATUS.md` | 根目录；完成一步更新一行 |
| **工程踩坑复盘** | `RETRO.md` | 根目录；格式见该文件维护规则 |
| **科学流水线真值** | `mindmap.md` | 根目录；改口径先改它 |
| **协作者入口** | `README.md` | 中文学术说明；勿替代 mindmap |
| **私人配置** | `configs/*.local.yaml`, `private/` | **永不提交** |
| **运行产物** | `runs/`, `reports/`, `models/checkpoints/` | 默认 gitignore；服务器写 `$WJW/NHC0801/...` |
| **临时草稿** | **禁止长期放仓库根** | 用完删，或迁入 `docs/plans/` / `RETRO.md` |

**禁止：**

- 在仓库根目录乱放 `tmp_*.py`、未命名 `plan.md`、大段实验脚本  
- 把库逻辑只写在 `scripts/` 而不进 `src/`  
- 把规划写进 `src/` 或把合同写进 `RETRO.md`  
- 修改 ranker / science-pilot 源树或 `$WJW` 非 NHC0801 路径  

---

## Documentation rules（写文档规矩）

| 文档 | 何时写 / 更新 |
| --- | --- |
| `docs/plans/*` | **写代码前**（多步任务、改科学口径、新模块） |
| `RETRO.md` | 踩坑后追加；同类问题先查再改 |
| `PHASE_STATUS.md` | 阶段完成或阻塞变化时 |
| `docs/contracts/*` | 冻结/升版合同时（需用户确认若改科学阈值） |
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
- **GAU_LOOSE**: 五准则 + ASE fmax **0.10** eV/Å，max 100。非 fmax **0.05**。
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

## Gates（默认全关）

```text
teacher_pyscf_authorized / aimnet2_train_authorized / epoch0_execution
scientific_validation_live / final_test_open
modify_wjw_outside_NHC0801 / scheduler_submission
```

允许：本地代码·测试·文档；只读 SSH；rsync **进** NHC0801（无 `--delete`）。

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

**Next engineering gap:** multi-seed trainer loop（步骤 4–5），且永不靠 quick-val 终选。
