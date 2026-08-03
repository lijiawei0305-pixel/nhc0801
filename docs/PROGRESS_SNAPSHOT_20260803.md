# NHC0801 进度快照（2026-08-03）

**仓库：** 代码与合同（本提交）。**运行产物**在服务器 `$WJW/NHC0801/runs/nhc0801-g001/`（不进 git）。

## 命名规范（已固化）

| 对人 | 磁盘 |
| --- | --- |
| g00N teacher | `teacher_gpu_g00N/` |
| g00N Epoch-0 | `epoch0_val_batches/g00N/` |
| 切组队列状态 | `gpu_teacher_queue/`（日志 `[gpu-teacher]`） |

详见 `AGENTS.md` § Experimental data naming · `docs/NHC0801_命名与进度指南.md`。

## Teacher（mindmap 步骤 2）

| 组 | 状态 | 目录 |
| --- | --- | --- |
| g001 | **DONE**（10/10 endpoints） | `teacher_gpu_g001/` |
| g002 | **DONE** | `teacher_gpu_g002/` |
| g003–g032 | **PASS**（约 30 组） | `teacher_gpu_g00N/` |
| g033+ | 队列继续（g00N teacher 守护） | `gpu_teacher_queue/` |

- 每组：5 roots = 3 Train + 2 Val = 10 endpoints。  
- g001 正线训练 **只认** g001 Train3；g002+ **禁止**并入 g001 训练标签。

## Epoch-0（mindmap 步骤 3 · Val only）

| 组 | 状态（重启后） | 备注 |
| --- | --- | --- |
| g001–g004 Epoch-0 | **RUNNING** | Parent/handoff = **GPU**（gpu4pyscf）；无 VASP 卡 |
| g005+ | 排队 | e0_val_queue 守护 |

- **永不**对 Train 跑 e0。  
- 修复：handoff 梯度字段名（`gradient_hartree_per_bohr`）；旧 CPU Parent 过慢已改 GPU。

## 其它

| 项 | 状态 |
| --- | --- |
| Final Test | **密封** |
| 正式多 seed 训练 / live sci-val | 门禁默认关；需用户授权 |
| 合同 | `COMPUTE_DISPATCH_V001` · `RESOURCE_*` · `RIGID_SMALL_NHC_POOL_V001` |

## 服务器运维入口（不入库的数据）

```text
$WJW/NHC0801/runs/nhc0801-g001/
  teacher_gpu_g00N/
  epoch0_val_batches/g00N/
  gpu_teacher_queue/state.json
  epoch0_val_queue/state.json
  logs/
```

守护：`nhc0801_gpu_teacher_daemon.py` · `nhc0801_e0_val_queue_daemon.py` · `nhc0801_compute_steward.py`。
