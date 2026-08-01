# 运维记录：停止 Codex phase9b 作业并复用 pilot 数据（2026-08-02）

## 停止的进程（用户授权）

| PID | 角色 | 分子 | CPU |
| ---: | --- | --- | --- |
| 2471578 | gtho continuation supervisor | — | — |
| 2662375–2662378 | pure_pyscf parent-worker 树 | VPAFDQIFHJWCBK… | 0,2-27 |
| 2733685–2733689 | pure_pyscf parent-worker 树 | RBKFFSUUCLDQER… | 28-55 |

- 信号：TERM → 确认死亡；无需 KILL。  
- **未**杀：loky 杂进程、sing-box、Claude remote、mlff 历史 wait 脚本。  
- 上述分子 **不在** NHC0801 scope C 的 3+2 开发集内（RBKF 为 not_admitted 未完成根）。

## 复用决策

| 资产 | 路径 | 判定 |
| --- | --- | --- |
| 老师帧 235 | `$WJW/data/runs/autofill_*`（5 roots） | **复用** |
| D3 投影 | `phase9b_aimnet2_v004_d3_projection_v001` | **复用**（不重算） |
| 加权 NPZ | `phase9b_aimnet2_v004_weighted_dataset_v001` | **复用**（audit PASS 235） |
| Codex 进行中的 VPAF/RBKF 轨迹 | 部分/未进集 | **不**并入 Train；仅保留原路径 |

绑定：symlink → `$WJW/NHC0801/runs/nhc0801-g001/datasets/weighted` 与 `d3`  
收据：`meta/pilot_dataset_binding.json`

## 随后在 NHC0801 上执行

1. Resource claim（local on server）：**LIVE_RESOURCE_CLAIM_PASS**  
2. Train dry-run（5 epochs × 3 seeds）：**DRY_RUN_TRAIN_PASS**  
3. Epoch-0 dry-run：**DRY_RUN_EPOCH0_PASS**  

## 明确未做（无 live 引擎）

- 真 PySCF 老师重算 / 真 AIMNet2 微调 / 真 epoch-0 化学  
- 打开 Final Test  

下一步真算需：`mlff`/`molenv` live 引擎接线 + 用户分项授权。
