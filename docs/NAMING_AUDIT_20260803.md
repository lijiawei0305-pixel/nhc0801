# 实验数据命名审计（2026-08-03）

**触发：** 用户要求 AGENTS 固化命名规则，并多路排查乱命名。  
**标准：** `AGENTS.md` § Experimental data naming · `docs/NHC0801_命名与进度指南.md` · `layout.py`。

## 规范（摘要）

| 对人 | 磁盘 |
| --- | --- |
| g00N teacher | `teacher_gpu_g00N/` |
| g00N Epoch-0 | `epoch0_val_batches/g00N/` |
| generation | `runs/nhc0801-g001/` |

废除：`teacher/`、`teacher_gpu_side/`、顶层规范 `epoch0/`、对外「Autofill 批」。

## 多代理排查结论

### 已对齐（新写路径）

- `layout.teacher_dir` → `teacher_gpu_g001/`
- `gpu_autofill` 产品目录 → `teacher_gpu_{batch_id}`
- `e0_val_only` → `epoch0_val_batches/<bid>/`
- 服务器：`teacher`→`teacher_gpu_g001`，`teacher_gpu_side`→`teacher_gpu_g002`（symlink 兼容）

### P0 合同半改（已在本轮修补正文多处）

`COMPUTE_DISPATCH_V001.md` 曾自相矛盾：修订记录已统一，正文仍残留 `teacher/` / `teacher_gpu_side` / 顶层 `epoch0/`。  
**本轮已改：** 路径表、E0 输出、训练禁止项、检查清单、CLI 示例、双代图等。

### P1 仍待后续（非阻断）

| 位置 | 问题 | 建议 |
| --- | --- | --- |
| 命名指南 §4 快照 | 仍有「GPU autofill」字样 | 改称 g003+ teacher |
| `scripts/*_02c.py` | 试验戳 02c / side-wave 文案 | 稳定脚本名 + 组号日志 |
| `live_epoch0_02c.out` vs `live_epoch0.out` | 日志 basename 不统一 | 统一为 `live_epoch0_g001.out` |
| `pipeline_status.py` | 只读 legacy `teacher/`、`epoch0/` | 迁移完成后可删 fallback |
| `data/paths.py` `autofill_*` | V004 旧帧布局 | 保留只读；禁止新写 |
| `preopt_gau_loose/` 等 ad-hoc | 非 GENERATION_SUBDIRS | 并入规范或文档标注 |

### P2 历史可保留

- `docs/evidence/`、`docs/REUSE_AUDIT_*`、`inventory/*` 中的 `$WJW/data/runs/autofill_*`（旧 pilot 绑定）
- RETRO 条目标题含 autofill 守护名（工程事故名）

## 工程名 vs 产品名

| 可保留工程标识 | 对外必须 |
| --- | --- |
| `gpu_autofill/` state、daemon 文件名 | **g00N teacher** |
| profile `auto_fill_112_*` | 资源名，非分子组 |

## 代理自检（已写入 AGENTS.md）

新写路径前：组号？`teacher_gpu_g00N`？`epoch0_val_batches/g00N`？零 Autofill 主称？

---

## P1 如何解决（方案 + 状态）

### 1) 脚本名 / 日志带 `02c` / side

| 策略 | 说明 |
| --- | --- |
| **不立刻硬改名在跑进程** | steward / pgrep 仍匹配 `teacher_wave_02c.py`、`gpu_autofill_daemon.py`；硬改会断守护 |
| **规范新日志 basename** | 模块 `generation/artifact_names.py`：`live_epoch0_g001.out`、`teacher_wave_g001.out` |
| **读多写一** | 新写只用规范名；读路径同时认 `*_02c.out` / `live_epoch0.out`（steward、overnight、e0 队列已接） |
| **稳定入口 shim** | `scripts/nhc0801_teacher_wave.py`、`scripts/nhc0801_gpu_teacher_daemon.py` 包装旧脚本 |
| **可选后续** | 全部作业退出后，删试验文件名、只留 shim 或反向 shim |

### 2) 日志标题 `[autofill]`

| 策略 | 状态 |
| --- | --- |
| 守护 stdout 改为 **`[gpu-teacher]`** | **已改** `nhc0801_gpu_autofill_daemon.py` |
| steward 文案改为 g00N teacher / gpu-teacher | **已改** |
| 目录名 `gpu_autofill/` | **保留**（工程 state；对外不叫产品名） |

### 3) V004 `autofill_*` 路径（`data/paths.py`）

| 策略 | 状态 |
| --- | --- |
| 标明 **LEGACY READ-ONLY** | **已改** 注释 + `legacy_v004_teacher_run_dir()` |
| 保留 `autofill_run_dir` 别名 | 兼容旧测试/import |
| 新 teacher **禁止**写该布局 | AGENTS 硬规则；写走 `teacher_gpu_g00N/` |

### 4) `pipeline_status` 旧路径探测

| 策略 | 状态 |
| --- | --- |
| **规范路径优先**（`teacher_batch_dir` / `epoch0_batch_dir`） | **已改** |
| 旧 `teacher/`、`epoch0/` 仅 LEGACY_READONLY 末位 | **保留到 symlink 退役** |
| 退役条件 | 服务器确认无顶层 `epoch0/` 产物且无需 `teacher` symlink 后删除 fallback |

### 执行顺序（运维）

1. rsync 本轮脚本/库（不杀在跑 e0 / teacher）  
2. **新**守护重启时才吃到 `[gpu-teacher]` 与新 e0 日志名  
3. 旧作业跑完后再删 `*_02c` 文件名依赖  

