# NHC0801 进度

**更新：** 2026-08-05（服务器时间约 14:46 CST / inventory 快照）  
**范围：** `nhc0801-g001` · 只写 `$WJW/NHC0801`  
**权威：** `mindmap.md` → 合同 → `AGENTS.md` → 本文（操作进度，非科学合同）

---

## 0. 一句话

- 试点 **3 Train root 不够**（operational T9：预筛无稳定增益）。
- **不再**按「一组 3 Train」做正式大训；HPC **继续** m250 teacher。
- 目标：**一个大 Train 池 ≥ 150 双端 PASS root**，再锁 split → NPZ → 训。
- Val：**固定 15–20**（约 10%，清单默认 18）；Final Test **仍 sealed**。

---

## 1. 当前策略（已拍板）

| 项 | 决定 |
| --- | --- |
| Train 规模 | **≥ 150** 个 **双端 m250 PASS** root 才锁名单、打 NPZ、开训 |
| Val | **A：固定 15–20**（~10%）；清单 `target_val=18` |
| 小 3 根一组 | **仅历史/legacy**；正式训练改为 **一个大 Train 池** |
| 磁盘 `teacher_gpu_g00N/` | **可继续按批存储**；科学 split 不按 3 根一组训 |
| Final Test | **sealed**，身份不暴露、不进 Train |
| HPC | **不停** `gpu_teacher_daemon` m250 |

### 阶段

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| **A** | teacher 继续；只读清单 `eligible_full_pairs`；看距 150 | **进行中** |
| **B** | 满门槛后锁 `split`（Train≥150 + Val 15–20）+ receipt | 未开始 |
| **C** | 大 Train D3 + 一个 weighted NPZ | 未开始（**禁止**未锁 150 就正式大训） |
| **D** | 少量配方重训 → 预筛 → sci-val | 未开始 |

---

## 2. 数量快照（Phase A inventory）

**命令：**

```bash
PYTHONPATH=src $WJW/env/conda/mlff/bin/python scripts/nhc0801_eligible_full_pairs.py \
  --nhc0801-root $WJW/NHC0801 --generation-id nhc0801-g001
```

**产物：**

- `runs/nhc0801-g001/logs/eligible_full_pairs/eligible_full_pairs.json`
- `.../eligible_full_pairs_status.json`

**最近一次扫描（约 2026-08-05）：**

| 指标 | 值 |
| --- | ---: |
| 双端 PASS `n_full_pairs` | **56** |
| 可进扩大 Train（排除 Val） | **56** |
| **距 Train 锁 150** | **还差 94** |
| 建议锁定前总双端（150+18） | 目标 168，还差 ~**112** |
| incomplete | 11 |
| 队列待做 endpoint | ~**3578** |
| 正在跑 | **8** |
| `train_lock_ready` | **false** |
| teacher daemon m250 | **alive** |

### Legacy TVT（generation 仍写死，未改合同）

| 角色 | 数量 | root |
| --- | ---: | --- |
| Train | 3 | `ACGCNT…` / `PDIY…` / `VNYH…`（双端已齐） |
| Val | 2 | `KZYK…` / `RMEQ…`（e0/sci-val 对照；**不进**扩大 Train 名单） |
| Final Test | 2 | sealed commitment only |

---

## 3. 已完成的科学/工程节点（摘要）

| 项 | 状态 | 备注 |
| --- | --- | --- |
| Mindmap 试点 0–9 环 | 走过 | e0 live + 训 + shortlist + sci-val live（旧 3-root 数据） |
| Operational T9 | **有结论** | `docs/science/T9_OPERATIONAL_20260805_no_gain_vs_epoch0.md`：3-root 上预筛无稳定优于 e0；停扫参 |
| Teacher m250 | **进行中** | 大池扫；Train3 已齐；池子 ~50+ 双端 PASS |
| 大 Train 清单脚本 | **DONE** | `scripts/nhc0801_eligible_full_pairs.py` + `pipeline/eligible_full_pairs.py` |
| Path A m250 小宽重训 | 曾跑 | 仍 3-root 数据；**不替代** 150 门槛后的大训 |
| 正式大 Train NPZ / 锁 split | **未开** | 等 `n_eligible ≥ 150` |

---

## 4. 标签口径（清单字段）

| 字段 | 含义 |
| --- | --- |
| `full_pair_pass` | cation+neutral 均 `endpoint_done_ok` |
| `eligible_for_expanded_train` | 双端 PASS 且 **非** Val |
| `in_train` / `in_val` | 当前 legacy 3+2 |
| `incomplete` | 有产物/在跑但未双端齐 |
| `excluded_test` | Final Test 身份不列出；可见 root 不标 test |

---

## 5. 下一步（只做这些）

1. **每天**跑清单，盯 `gap_to_train_lock_150`。  
2. teacher **继续** m250，不人为停大池（除非抢 GPU 做已授权任务）。  
3. **`n_eligible ≥ 150` 之前：不锁 split、不打正式大 NPZ、不开正式大训。**  
4. 达标后进入阶段 B：固定 Val 15–20 + Train≥150 写 split receipt，再 D3/NPZ/训。

---

## 6. 相关文档

| 文档 | 用途 |
| --- | --- |
| `docs/science/T9_OPERATIONAL_20260805_no_gain_vs_epoch0.md` | 3-root 为何不够 |
| `docs/plans/path_a_diagnosis_charter_20260805.md` | 诊断路径 A（历史） |
| `docs/agent/training_t1_t9.md` | 微调硬规则 T1–T9 |
| `AGENTS.md` | Always / Ask / Never |
| `logs/eligible_full_pairs/README.md`（服务器） | 清单刷新说明 |

---

## 7. 历史备注

- 旧版 `progress.md`（2026-08-03）按 **每组 3 Train + 2 Val** 列表 g001–g372，已被本策略取代。  
- 存储批次名 `g00N` 仍可出现在路径里；**训练不再「一组 3 个 Train」定生死**。
