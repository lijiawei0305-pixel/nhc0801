# NHC0801 全 mindmap 自动化 + 只读 TUI 监控 — 设计文档

**日期**: 2026-08-02  
**状态**: 设计冻结（本轮 **不写实现代码**；实现需另授权）  
**约束**: 服务器内网；**不常驻**、项目结束 **可干净删除**；Final Test 不自动开  

相关权威：

- 科学：`mindmap.md`，`AGENTS.md`
- 资源：`docs/contracts/RESOURCE_SCHEDULING_V001.md`，`RESOURCE_PROFILES_V002.yaml`
- 数值选模：`docs/contracts/NUMERIC_CALIBRATION_V001.yaml`

---

## 0. 已确认需求摘要

| 主题 | 确认内容 |
| --- | --- |
| 自动化范围 | mindmap **0–10 可编排自动**；**11–12 Final Test 永远人工二次确认** |
| CPU | 每端点 **8 逻辑线程**；池 **0–111**；**无 N_cap**；\(N=\min(N_{cpu},N_{mem})\) |
| 内存 | 每端点 **8 GiB**（2026-08-02b；实测 HWM~4.5），主机预留 **40 GiB** |
| 端点并行 | **同一 root 的 cation/neutral 可同时跑** |
| GPU | 训练 **1 卡**；GAU_LOOSE **另一空闲卡**；Parent **CPU-only** |
| 共存 | GPU 训练 + CPU PySCF **允许同时** |
| 监控 | **SSH 终端 TUI**；登录 nhc614 后在 NHC0801 **临时只读**跑 |
| 刷新 | **30 s**；**纯只读不写盘** |
| Bug 界定 | 以 generation 下 **receipt `status`** 为准 |
| 提升指标 | **标签 MAE 为主** + quick-val loss + 结构门禁通过率 |
| 速度 | **端到端 wall**（冻结几何 → 最终 parent GAU 标签） |
| 卸载 | 无 systemd/cron；产物仅在 NHC0801；一键清理文档化 |

---

## 1. 自动化总编排（mindmap 0–10）

### 1.1 入口（拟名，实现阶段）

```text
scripts/nhc0801_mindmap_pipeline.py   # 唯一编排入口（拟）
  --generation-id nhc0801-g00x
  --from-step 0 --to-step 10
  --dry-run | --live  (live 仍要分项 gate)
  --profile auto_fill_112_t8_v1
```

**禁止**：默认 `--to-step 12`；任何路径静默打开 Final Test。

### 1.2 步骤状态机

每个步骤写入/更新：

```text
runs/<gen>/pipeline/step_XX_<name>.json
runs/<gen>/pipeline/pipeline_status.json   # 汇总；TUI 主数据源之一
```

| step | 名称 | 成功 status 示例 | 失败 |
| --- | --- | --- | --- |
| 0 | freeze_roots | `PASS` | `FAIL` |
| 1 | tvt_split | `PASS` | `FAIL` |
| 2 | teacher_pyscf | `LIVE_TEACHER_PASS` / dry | `FAIL` / `PARTIAL` |
| 3 | epoch0 | `LIVE_EPOCH0_PASS` | `FAIL` |
| 4–5 | train_checkpoints | `LIVE_TRAIN_PASS` | `FAIL` |
| 6 | quick_val (in train) | 嵌入 seed receipts | — |
| 7 | shortlist | `SHORTLIST_PASS` | `FAIL` |
| 8 | sci_val | `LIVE_SCI_VAL_PASS` | `FAIL` / `REJECTED` |
| 9 | select | `VALIDATION_SELECTED` | `VALIDATION_REJECTED` |
| 10 | freeze | `FROZEN` / `PROVISIONAL` | `FAIL` |
| 11 | final_test | **仅人工** `ARMED`→`RAN_ONCE` | — |
| 12 | no_post_select | 策略断言 | 违规 `POLICY_VIOLATION` |

**RUNNING**：步骤已启动、尚无终态 receipt 时，`pipeline_status.json` 可标 `RUNNING`（由编排器在启动时写入；若用户坚持 TUI 纯只读且编排器也不写，则 TUI 仅用「进程+最新 receipt」推断 — **实现前若冲突再确认**）。  

> 已确认 TUI **不写盘**。编排器本身 **需要写 receipt**（科学与审计必需）。  
> 「不遗留任务」= 无常驻 daemon，不是「不写 runs/ 产物」。

### 1.3 步骤 2/3/8 与资源调度钩子

```text
claim(profile=auto_fill_112_t8_v1)
  → 计算 N
  → 切分不相交 cpu 集合（每块 8 逻辑核）
  → 队列：待跑 endpoint 列表（root, endpoint）可乱序/可同 root 两端点并行
  → 每完成一端点：释放核与内存配额，再调度下一批
  → Parent: CUDA_VISIBLE_DEVICES="" + OMP=8
```

训练步骤（4–5）：

```text
选 GPU_train
  → mlff 环境
  → 与 CPU 队列并行（不抢同一 GPU 给 GAU）
```

### 1.4 Gate 矩阵（live）

| Gate | 挡住的步骤 |
| --- | --- |
| `teacher_pyscf_authorized` | 2 live |
| `epoch0_execution` | 3 live |
| `aimnet2_train_authorized` | 4–5 live |
| `scientific_validation_live` | 8 live |
| `final_test_human_confirm` | 11（另需 readiness） |

Dry-run 可在无 gate 下跑通 0–10 骨架（模拟引擎）。

### 1.5 失败策略

- 单 endpoint `FAIL`：记入 receipt；默认 **不删分子、不换阈值**（mindmap 纪律）。  
- 编排器：`on_endpoint_fail: continue_queue | abort_step`（**实现前需你再选一次默认**，见 §5 未决）。  
- 步骤失败 → `pipeline_status` 红；不自动跳到 Final Test。

---

## 2. TUI 监控（SSH 终端）

### 2.1 运行方式

```bash
ssh nhc614
cd $WJW/NHC0801
source env/envs/mlff.sh   # 或最小 python；实现时定
python -m nhc_deprot.dashboard.tui --generation-id nhc0801-g001
# 或 scripts/nhc0801_tui.py
# Ctrl+C 退出；无后台、无端口
```

- **刷新**: 30 s（可 `--interval`）  
- **IO**: **只读** `runs/<gen>/**` 与可选 `ps`/`nvidia-smi`（只读）  
- **不写** snapshot、不写 lock  

### 2.2 屏幕信息架构（主屏）

```text
┌─ NHC0801 | gen=nhc0801-g001 | refresh 30s | host nhc614 ─────────────┐
│ Pipeline:  [0]ok [1]ok [2]RUN [3]… [4]ok … [10]prov | FT: SEALED      │
│ Resources: CPU idle 80/112 | N_cap_formula=5 | GPU0 train GPU1 free   │
├─ Train ───────────────────────────────────────────────────────────────┤
│ seed 20260730 ep 200/200 loss_tr=… loss_val=… status=PASS             │
│ seed …                                                                │
├─ Scientific metrics (default sort: label MAE) ────────────────────────┤
│ route          MAE↓   max|AE|  gates_ok  e2e_wall_s  vs_e0            │
│ pure_pyscf     0.00   …        yes       1200        —                │
│ epoch0         1.2    …        yes        800        —                │
│ finetuned@e60  0.8    …        yes        500        +0.4 MAE / 1.6×  │
├─ Quick-val (frame loss, not final) ───────────────────────────────────┤
│ …                                                                     │
├─ Problems (receipt status ≠ PASS) ────────────────────────────────────┤
│ step2 teacher RMEQ… neutral FAIL: …                                   │
└───────────────────────────────────────────────────────────────────────┘
```

### 2.3 指标定义（写入 TUI + 未来 JSON 导出可选）

#### A. 相对官方 AIMNet2（epoch-0）与 Pure-PySCF

| 指标 | 定义 | 数据来源 |
| --- | --- | --- |
| **主：标签 MAE** | mean \|label_route − label_pure\| kcal/mol over Val roots | epoch0 / sci_val root receipts |
| max \|AE\| | 同左 max | 同上 |
| 结构门禁通过率 | 硬门全过的 endpoint 比例 | `all_identity_and_structure_hard_gates` |
| Quick-val weighted loss | 帧级 energy+force（**不选 final**） | train seed `epoch_logs` |

**提升展示**（主屏默认 MAE）：

- \(\Delta\mathrm{MAE} = \mathrm{MAE}_{epoch0} - \mathrm{MAE}_{finetuned}\)（**正 = 变好**）  
- 相对：\(\mathrm{MAE}_{finetuned}/\mathrm{MAE}_{epoch0}\)

#### B. 速度（端到端 wall）

对每条路线 **A pure / B epoch0 / C finetuned**，每个 Val root（或 endpoint 聚合）：

\[
T_{\mathrm{e2e}} = t_{\mathrm{AIMNet2\ GAU\_LOOSE}} + t_{\mathrm{handoff}} + t_{\mathrm{parent\ opt+SP}}
\]

（pure 路线无 AIMNet2 段，\(t_{\mathrm{AIMNet2}}=0\)。）

展示：

- 绝对秒数  
- 加速比：\(T_{\mathrm{pure}}/T_{\mathrm{ft}}\)，\(T_{\mathrm{e0}}/T_{\mathrm{ft}}\)

**实现要求**：live 路径必须在 receipt 写入真实 `wall_seconds`（当前部分字段为 0 占位 — **实现 live 时必修**）。

### 2.4 Bug / 问题界定（已确认）

| status 类 | TUI |
| --- | --- |
| `*PASS` / `SELECTED` / `FROZEN` | 绿 |
| `RUNNING` / 缺终态但进程在 | 黄 |
| `FAIL` / `FAILED` / `REJECTED` / `PARTIAL` | 红 + 摘 `error`/`reasons` |
| 缺 receipt 且无进程 | 灰 `NOT_STARTED` |

**不以**人工便签为准（本版不写可写状态库）。

---

## 3. 数据流

```text
                    ┌──────────────┐
  mindmap pipeline  │  write-only  │  receipts under runs/<gen>/
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │  runs/<gen>/**.json     │
              │  (teacher/epoch0/train/ │
              │   sci_val/freeze/…)     │
              └────────────┬────────────┘
                           │ read-only 30s
                    ┌──────▼───────┐
                    │  TUI (ssh)   │
                    └──────────────┘
```

无公网、无长期端口、无 Grafana 依赖。

---

## 4. 安装与干净删除

### 4.1 允许存在的路径

```text
$WJW/NHC0801/
  docs/contracts/RESOURCE_SCHEDULING_V001.md
  docs/contracts/RESOURCE_PROFILES_V002.yaml
  docs/plans/20260802_automation_tui_design.md
  src/nhc_deprot/...          # 实现后
  scripts/nhc0801_*.py
  runs/<gen>/                 # 科学产物 + pipeline 状态
```

### 4.2 禁止

- `systemd --user` / root 服务  
- `crontab` 常驻  
- `/var/`、home 下隐藏 daemon  
- 监听 `0.0.0.0` 的长期 Web（本版 TUI 不监听端口）

### 4.3 项目结束清理（文档级 checklist）

```bash
# 1) 停所有 NHC0801 相关前台/后台作业（人工确认 PID）
pkill -f 'NHC0801|nhc0801_'   # 慎用：先 pgrep 再删

# 2) 可选：归档后删除 generation
# mv $WJW/NHC0801/runs/$GEN /path/to/archive/

# 3) 若整项目下线
# 仅删除 $WJW/NHC0801（禁止碰 $WJW 其它树）
```

本机 Mac 仓库可保留 git 历史；服务器以目录删除为准。

---

## 5. 未决与已补确认

### 5.1 已补确认（2026-08-02 第二轮）

| 项 | 确认 |
| --- | --- |
| 单 endpoint 失败 | **`continue_queue`**：记 FAIL，继续其它端点；步骤可 `PARTIAL` |
| `pipeline_status.json` | **允许编排器写**（含 `RUNNING`）；**TUI 只读** |

### 5.2 实现前仍建议确认（未猜）

1. **空闲 CPU 利用率阈值**：文档草案 15% 是否接受？  
2. **live wall_seconds=0 占位**：是否在实现 TUI 前先修 worker 计时？  
3. **shortlist 中无 `.pt` 的 epoch**：live sci-val 是跳过、补导出、还是只评有权重的？

---

## 6. 建议实现顺序（授权后）

1. 资源：实现 V002 claim + 按块绑 8 核 + 内存门（单测用注入 snapshot）  
2. receipt：统一 `status` 枚举 + wall 字段  
3. `pipeline_status` 汇总器（只读扫描也可先做）  
4. TUI v1：步骤灯 + train + 问题列表  
5. TUI v2：MAE / e2e 表（依赖 epoch0+sci_val 收据）  
6. 编排器 dry-run 0–10  
7. 分项开 live gate  

---

## 7. 修订记录

| 日期 | 内容 |
| --- | --- |
| 2026-08-02 | 首版设计；互动问答锁定资源/TUI/自动化边界 |
