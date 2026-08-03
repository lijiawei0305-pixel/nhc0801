# NHC0801 计算进度（Teacher + Epoch-0）

**最后更新（服务器 UTC）：** 2026-08-03T12:40:31Z  
**数据根：** `$WJW/NHC0801/runs/nhc0801-g001/`  
**本文件用途：** 一眼看 teacher / Epoch-0 进度。运行产物在服务器，不进 git 大目录。

> **命名：** 分子组 `g00N` = 5 roots（3 Train + 2 Val）= 10 endpoints。  
> **Teacher** = 老师帧（Train+Val 都算）→ `teacher_gpu_g00N/`  
> **Epoch-0** = 微调前基线（**只算 Val**）→ `epoch0_val_batches/g00N/`

---

## 总览

| 组 | Teacher | Epoch-0 | 说明 |
| --- | --- | --- | --- |
| **g001** | **完成** `LIVE_TEACHER_PASS` | **进行中** | 正线 pilot；e0 只跑 2 个 Val |
| **g002** | **完成** `LIVE_TEACHER_GPU_PASS` | **进行中** | 合同第一批 GPU teacher |
| **g003** | **PASS**（teacher） | **进行中**（e0） | 与 g004 同批 e0 并行 |
| **g004** | **PASS**（teacher） | **进行中**（e0） | Parent = GPU |
| **g005–g032** | **PASS**（teacher） | 排队等 e0 | teacher 已齐，e0 未开 |
| **g033+** | 队列中 | — | g00N teacher 守护继续切组 |

**Epoch-0 队列（当前）：** g001 · g002 · g003 · g004 **RUNNING**（parent=gpu）  
**g003+ teacher 队列：** PASS ≈ **30** · RUNNING · PENDING 余下池子

---

## g001（正线 pilot）

### Teacher — **完成**

| 项 | 内容 |
| --- | --- |
| 目录 | `teacher_gpu_g001/` |
| 收据 | `LIVE_TEACHER_PASS` |
| 进度 | **5/5 roots · 10/10 endpoints** |

| 角色 | InChIKey |
| --- | --- |
| Train | `ACGCNTKELWXJPN-UHFFFAOYSA-N` |
| Train | `PDIYCCLDBKWBTK-UHFFFAOYSA-N` |
| Train | `VNYHGZAUUQMMDL-UHFFFAOYSA-N` |
| Val | `KZYKDQNIIMATMJ-UHFFFAOYSA-N` |
| Val | `RMEQTBVGGNKAEQ-UHFFFAOYSA-N` |

### Epoch-0 — **进行中**

| 项 | 内容 |
| --- | --- |
| 目录 | `epoch0_val_batches/g001/` |
| 分子 | **仅 Val2**（上表两根 · 4 endpoints） |
| 资源 | GPU **5** · parent=**gpu** |
| 启动 | 2026-08-03T12:33:06Z |
| 收据 | 尚无 `*_epoch0_val_receipt.json` / campaign |

---

## g002

### Teacher — **完成**

| 项 | 内容 |
| --- | --- |
| 目录 | `teacher_gpu_g002/` |
| 收据 | `LIVE_TEACHER_GPU_PASS` |
| 进度 | **5/5 roots · 10/10 endpoints** |

| 角色 | InChIKey |
| --- | --- |
| Train | `CLXFIGGGSODORK-UHFFFAOYSA-N` |
| Train | `CRPRBFHOCLDMMB-UHFFFAOYSA-N` |
| Train | `HFQMBFOQLKGXEV-UHFFFAOYSA-N` |
| Val | `HVVRUQBMAZRKPJ-UHFFFAOYSA-N` |
| Val | `IPMZWBRHUWBMSP-UHFFFAOYSA-N` |

### Epoch-0 — **进行中**

| 项 | 内容 |
| --- | --- |
| 目录 | `epoch0_val_batches/g002/` |
| 分子 | **仅 Val2**（HVVRUQ… · IPMZWB… · 4 endpoints） |
| 资源 | GPU **7** · parent=**gpu** |
| 启动 | 2026-08-03T12:33:07Z |
| 收据 | 尚无 |

---

## g003 / g004（对照）

| 组 | Teacher | Epoch-0 |
| --- | --- | --- |
| **g003** | PASS · `teacher_gpu_g003/` | **RUNNING** GPU1 · Val=`AAULNF…`,`ABDQQZ…` |
| **g004** | PASS · `teacher_gpu_g004/` | **RUNNING** GPU0 · Val=`ADPCUI…`,`AETSDH…` |

---

## g003+ Teacher 组（池切组）

| 状态 | 组号（约） | 目录模式 |
| --- | --- | --- |
| **PASS** | g003 – g032（约 30 组） | `teacher_gpu_g00N/` |
| **RUNNING / PENDING** | g033 起 | 队列 `gpu_teacher_queue/state.json` |

每组结构相同：**3 Train + 2 Val**。  
**注意：** 这些 teacher 帧 **不进 g001 正线训练**；仅 g001 Train3 用于当前 generation 微调标签。

---

## 怎么读状态（完成判据）

| 工作 | 完成标志 |
| --- | --- |
| **g00N teacher** | `teacher_gpu_g00N/` 下 5 root 两端点有帧 + `campaign_receipt*.json` 为 PASS |
| **g00N Epoch-0** | `epoch0_val_batches/g00N_epoch0_val_receipt.json` 或 `…/g00N/epoch0/campaign_receipt.json` + 日志 `E0_VAL_EXIT` |

### 服务器速查

```bash
G=$WJW/NHC0801/runs/nhc0801-g001   # 或实际 NHC0801 路径

# teacher
ls $G/teacher_gpu_g001 $G/teacher_gpu_g002

# e0 队列
python3 -c "import json;s=json.load(open('$G/epoch0_val_queue/state.json'));print('run',s.get('running'));print('done',s.get('completed'))"

# e0 收据
ls $G/epoch0_val_batches/*_epoch0_val_receipt.json 2>/dev/null
```

---

## 维护说明

- **本文件在仓库根目录** `progress.md`，便于打开即看。  
- 进度会变：由协作者/代理在查服务器后 **更新本文件顶部时间戳与表格**。  
- 不把完整 `runs/` 提交进 git；这里只记 **可复述的进度表**。  
- 命名规则：`AGENTS.md` § Experimental data naming · `docs/NHC0801_命名与进度指南.md`。
