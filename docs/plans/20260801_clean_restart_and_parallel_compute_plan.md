# NHC0801 清乱重启与多分子并行计算规划

**日期**: 2026-08-01  
**对应 mindmap**: 0–12（优先 0–3 老师数据与 epoch-0；4–9 训练与科学 Validation）  
**权威**: `mindmap.md` > 冻结合同 > `AGENTS.md` > pilot handoff（只读）  
**本计划不授权**: 真实 PySCF / AIMNet2 训练 / Final Test 打开  

---

## 0. 你现在面对的两套“乱”

| 来源 | 症状 | 是否应继续当主战场 |
| --- | --- | --- |
| **Codex science-pilot**（`nhc-deprot-ranker-science-pilot`） | 脏工作树、`phase9b_*` / `V004` 海量脚本与 JSON、permit/控制面债务、未跟踪产物 | **否**。只读考古，禁止 reset/clean，禁止当执行根 |
| **NHC0801**（`/Users/cc/nhc-deprot` + `$WJW/NHC0801`） | 已有 mindmap 代码与改名证据，但 docs 仍有 archive/extracted 噪音；服务器上仍引用旧 `phase9b_aimnet2_v004_*` 路径 | **是**。以后只在这里写代码与算 |

**结论**: 不要在 pilot 上“继续收拾 Codex”。在 NHC0801 做 **一代命名清理 + 新 generation 重算**；pilot 与 `$WJW` 生产只读借用。

Handoff 里仍成立、必须继承的科学点：

- AIMNet2 = 预条件器；GAU_LOOSE → exact-byte handoff → **完整** Parent P01  
- 非 B3LYP/SVP two_endpoint；`single_point_only=false`  
- 划分单位 = molecular root；Final Test 密封  
- 官方算力档案曾冻结为 **27 物理核**（SMT 更慢）；双 worker 仅为 14+13 校准候选，**从未**授权“112 逻辑核随便开”  
- dual-worker live claim V001/V002 均 **REJECTED**（CPU 0,2-27 忙）

NHC0801 相对 handoff 已多完成：数值标定冻结、sci-val writer、参数化数据层、GitHub 源码冻结。

---

## 1. 目标：重新计算，但不要科学失忆

### 1.1 建议策略（推荐）

**「干净 generation 重启」**，不是删历史：

```text
保留: mindmap 协议、P01 SHA、GAU_LOOSE、数值标定、TVT 纪律、库代码
归档: pilot 证据与 Codex 文件名（只读）
新建: generation_id = nhc0801-g001-YYYYMMDD
新写: 仅 $WJW/NHC0801/runs/<generation>/...
重算: 老师帧 / D3 / 加权集 / epoch-0 / 训练（分项授权）
```

### 1.2 不建议

| 做法 | 原因 |
| --- | --- |
| 在 pilot 脏树上继续 phase9b 脚本链 | 名称乱、控制面乱、易混生产 |
| `rm -rf` 服务器旧 autofill / D3 产品 | 可能误伤共用数据；先只读引用或复制进 NHC0801 |
| 未 claim 资源就开几十进程 PySCF | OOM、与 VASP 抢核、数值不可复现 |
| 为加速打开 Final Test 身份 | 选模污染，mindmap 禁止 |
| 用历史 `$WJW/checkpoints/*.pt` 当 epoch-0 | 金标污染风险 |

---

## 2. 命名与目录：如何让项目“看起来不乱”

### 2.1 统一命名空间（NHC0801）

| 旧（Codex / pilot） | 新（本项目） | 说明 |
| --- | --- | --- |
| `phase9b-aimnet2-…-v004` | `nhc0801-g001` | generation 短名 |
| `PHASE9B_AIMNET2_*` | 已迁 `docs/evidence/pilot_day1/` | 只读证据，勿再当合同主文件 |
| `scripts/phase9b_*.py` | **不移植整包**；逻辑进 `src/nhc_deprot/`，CLI 进 `scripts/nhc0801_*.py` | 薄入口 |
| `$WJW/data/runs/phase9b_aimnet2_v004_*` | `$WJW/NHC0801/runs/g001/...` | 新产物只进 NHC0801 |
| `autofill_{key}_v001`（共用） | 可只读引用；重算则 `NHC0801/runs/g001/teacher/{root}/` | 避免写爆共用 runs |

### 2.2 推荐本地树（稳态）

```text
/Users/cc/nhc-deprot/
  mindmap.md AGENTS.md RETRO.md PHASE_STATUS.md README.md
  src/nhc_deprot/          # 唯一库代码
  tests/
  scripts/                 # nhc0801_* CLI only
  docs/
    contracts/             # 冻结合同（GAU_LOOSE、数值标定、资源档案）
    plans/                 # 本文件类规划
    science/               # 讨论/阻塞
    evidence/pilot_day1/   # 只读 pilot（可整夹只读）
    archive/               # 禁止栈与噪音
  configs/
  runs/                    # 本地小收据；大算在服务器
```

### 2.3 推荐服务器树（重算唯一写入）

```text
$WJW/NHC0801/
  repo/                    # rsync 代码镜像（可选）
  runs/
    g001/
      meta/generation.json           # generation_id, protocol SHA, split SHA
      resources/claim_*.json         # 资源 claim 收据
      teacher/{root}/{cation|neutral}/frame_*.json
      d3/{root}/...
      datasets/weighted/{train,validation}/*.npz
      epoch0/
      train/seed_*/ckpt_*.pt
      sci_val/{ckpt}/...
      freeze/
  logs/
```

### 2.4 清理动作清单（安全顺序）

1. **冻结清单**: 列出 NHC0801 中“权威 / 只读证据 / 可删噪音”三类（本计划 §2.5）。  
2. **不删 pilot 源树**；最多在 NHC0801 内删除重复副本（`docs/extracted/v004` 已仅 redirect）。  
3. **服务器**: 新建 `runs/g001/`；旧 pilot 路径保持只读。  
4. **代码**: 路径常量逐步改为 `nhc0801` 前缀；旧 `phase9b` schema 字符串仅作“兼容读 pilot 证据”的别名。  
5. **Git**: 大清理用一次 commit 说明 “namespace cleanup”，不 rewrite 科学合同内容。

### 2.5 权威 vs 噪音（决策表）

| 路径 | 角色 |
| --- | --- |
| `mindmap.md`, `docs/contracts/*`, `src/nhc_deprot/**` | **权威** |
| `docs/evidence/pilot_day1/*` | 只读 pilot 证据 |
| `docs/archive/**` | 噪音隔离，勿读协议 |
| `docs/extracted/**` | 历史提取；可继续瘦身 |
| science-pilot 全树 | 外部只读，**不** rsync 进 NHC0801 根 |

---

## 3. 重新计算：科学流水线（mindmap 顺序）

重算必须 **按步关门**，不能“一口气训练”。

```text
[资源 claim PASS] ─────────────────────────────────────────┐
                                                           ▼
0–1 冻结 roots + TVT 划分（可沿用 pilot 3+2+密封 FT，或重新冻结更大集）
        │
        ▼
2  多分子并行 Pure-PySCF 老师帧 → 写入 NHC0801/runs/g001/teacher/
        │
        ▼
2b 冻结 D3 投影（一次，禁止静默重算）→ g001/d3/
        │
        ▼
2c 加权 NPZ 数据集 → g001/datasets/
        │
        ▼
3  Epoch-0 全路线（官方 _0 权重）→ g001/epoch0/
        │
        ▼
4–5 多 seed 训练 + 全量 ckpt 保留 → g001/train/
        │
        ▼
6–7 快验短名单（不终选）
        │
        ▼
8–9 科学 Validation 选模（writer 已有；接 live 引擎）
        │
        ▼
10–12 冻结 → Final Test 一次
```

**若“重新开始”仅指老师数据规模扩大**：  
从步骤 0–1 重新冻结更大 root 集；**不要**打开 Final Test；不完整 root 进 `not_admitted`，不删不换不挪。

**若沿用 pilot 5 根 + 235 帧**：  
可跳过大规模步骤 2，只读引用旧帧/D3/加权集做训练链；仍须在 NHC0801 记 binding SHA（证据在 pilot 路径）。

---

## 4. 前提设置：资源充足时的多分子并行

### 4.1 核心原则（来自 handoff + 物理现实）

1. **并行单位 = molecular root（或 endpoint）**，不是“同一几何开 112 线程”。  
2. **物理核不相交**；默认 **关闭 SMT**（54 线程曾比 27 物理核慢 5.28%）。  
3. **每 worker 内存预算硬限制**；PySCF `max_memory` 只是提示，RSS 才是墙。  
4. **共享节点**：先 live resource claim（双样本），再开工；与 VASP 共存时 fail-closed。  
5. **不重试/不替换**已 claim 的失败任务（审计友好）；失败记收据，人工决定是否新 generation。  
6. **数值身份**：并行不得改变 P01/GAU_LOOSE；校准模式与生产吞吐模式分离。

### 4.2 官方已有档案（太保守，但是安全默认）

| Profile | 物理核 | Worker 数 | 含义 |
| --- | --- | --- | --- |
| `single_27_physical_v1` | 27（亲和 0,2-27） | **1 root** | handoff 官方；每任务 64 GB 提示 |
| `dual_14_13_physical_v1` | 14+13 | **2 roots** | 仅校准候选；需 claim + ≥5% 吞吐提升收据 |

Handoff 里 **root concurrency = 1** 是当时 epoch-0/校准的冻结值，**不是**你最终吞吐的上限。上限应由 **可用物理核 × 每任务内存** 决定，并用 `ISOLATED_BENCHMARK` 出收据。

### 4.3 推荐：吞吐档案族（NHC0801 新建，需标定）

设可用 **物理核** \(C\)（从 online ∩ cgroup ∩ affinity 求交，勿信 `nproc=112`），  
每任务预算内存 \(M\) GB，主机预留 \(R\) GB，可用内存 \(A\) GB：

\[
N_{\mathrm{mem}} = \left\lfloor \frac{A - R}{M} \right\rfloor,\quad
N_{\mathrm{cpu}} = \left\lfloor \frac{C}{t} \right\rfloor,\quad
N = \min(N_{\mathrm{mem}}, N_{\mathrm{cpu}}, N_{\mathrm{cap}})
\]

其中 \(t\) = 每 worker 线程数（建议先 8–14，而不是 27 打满单任务），  
\(N_{\mathrm{cap}}\) = 管理上限（建议先 4–8，标定后再提）。

**示例（需实测改写，不可当真理）**：

| 场景 | 假设 | 粗算 \(N\) |
| --- | --- | --- |
| 节点空闲、~230 GB 可用、预留 40 GB、每任务 48 GB、27 物理核、每 worker 9 线程 | \(N_{\mathrm{mem}}=3\), \(N_{\mathrm{cpu}}=3\) | **3 分子并行** |
| 同上但每任务 32 GB、每 worker 6 线程 | \(N_{\mathrm{mem}}=5\), \(N_{\mathrm{cpu}}=4\) | **4**（受核限制） |
| 仅 27 核且必须每任务 27 线程 | \(N_{\mathrm{cpu}}=1\) | **1**（旧官方） |

**P01/def2-TZVPP 完整优化** 吃内存与墙钟；盲目 \(N=20\) 通常先 OOM 再变慢。

### 4.4 并行调度模型（要实现时）

```text
队列: ready roots（Train 或 Val 列表；永不含 Final Test 身份）
Worker pool: N 个进程/容器
  每个 worker:
    - cpuset 物理不相交 + NUMA local
    - OMP_NUM_THREADS = t
    - 环境: 单独 shell source molenv.sh（PySCF）或 mlff.sh（AIMNet2）
    - 领取一个 root（原子 claim 文件）
    - 顺序算 cation → neutral（或两端点再并行，若内存允许且核仍不相交）
    - 写 NHC0801/runs/g001/teacher/{root}/... + receipt
    - 释放 claim；失败写 FAILED receipt，不自动 retry
协调器:
  - 资源 PSI / MemAvailable 低于阈值 → 暂停派发（backpressure）
  - 进度看板: accepted roots / hour, 核·时效率
```

**两端点并行**: 仅当 `2N` 个核槽与内存仍满足；否则 **root 内串行 endpoint，root 间并行**（通常更稳）。

**AIMNet2 预优化**: 可 GPU 单卡串行或多进程抢 GPU；与 CPU-bound parent 流水线可重叠（stage pipeline），但别默认多进程打满同一 GPU 无标定。

### 4.5 标定协议（开高并行前的前提）

在 `docs/contracts/` 增加（实现阶段）：

1. `RESOURCE_PROFILE_CATALOG_V001.yaml` — 候选档案列表  
2. `PARALLEL_THROUGHPUT_CALIBRATION_PLAN_V001.json` — 冻结 workload（2–4 个固定 root 的 parent 单点或短 opt）  
3. Live claim 双样本 PASS  
4. 跑 `ISOLATED_BENCHMARK`：比较 `N=1,2,3,4…` 的 **accepted-root/wall-hour** 与数值一致性  
5. 选出官方 `THROUGHPUT_COLLECTION` 档案，写 **selection receipt**  
6. 此后大规模老师数据 **只许** 使用该档案  

未出收据前：默认 `N=1`（single_27）或经授权的 `N=2`（dual）。

### 4.6 与“同一时间多个分子更快”的关系

- **是的**：在内存与核允许下，\(N>1\) 通常显著提高 **root/墙钟小时**。  
- **但是**：单分子墙钟不一定变短；总进度取决于 \(N \times\) 效率。  
- **错误加速**: 砍 parent 优化、改 B3LYP、开 SMT 打满 112 逻辑核、部分帧入库 — 科学上不允许。

---

## 5. 实施阶段（工程，非 live）

| 阶段 | 内容 | 产出目录 | 授权 |
| --- | --- | --- | --- |
| **P-clean** | 命名空间文档化；路径常量 `nhc0801`；generation schema | 代码 + `docs/contracts` | 仅代码 |
| **P-res** | 资源 claim 工具（移植/重写进 `src/nhc_deprot/resources/`） | `scripts/nhc0801_resource_claim.py` | 只读 claim |
| **P-cal** | 并行度标定计划 + 收据格式 | `docs/contracts` + `runs/g001/resources/` | 标定授权 |
| **P-teacher** | 多 root 并行老师帧 runner | `pipeline/teacher_runner.py` + scripts | `teacher_pyscf_authorized` |
| **P-d3-ds** | D3 + 加权集（消费冻结收据） | `runs/g001/d3`, `datasets` | 单独授权 |
| **P-e0** | Epoch-0 live | `runs/g001/epoch0` | `epoch0_execution` |
| **P-train** | 多 seed trainer | `training/` | `aimnet2_train_authorized` |
| **P-sval** | live 接 sci-val writer | 已有 writer | `scientific_validation_live` |

当前 NHC0801 已有：数据读层、handoff、sci-val writer、阻塞诊断。  
**缺的并行关键路径**: `resources/` claim、worker 池、teacher runner、generation 布局。

---

## 6. 决策点（需要你拍板）

请在开工前明确（一次一个）：

1. **重算范围**  
   - (A) 仅沿用 pilot 5 根 / 235 帧，清命名后走训练链  
   - (B) 扩大 root 集（从 401k 池抽更多完整 root）→ 必须并行老师 DFT  
   - (C) A 先通全链路，再 B 扩规模  

2. **并行激进程度**  
   - (S) 安全：先 dual 标定，再谈 N>2  
   - (T) 吞吐：允许设计 N=4–8 档案，但必须标定收据  

3. **是否废弃服务器上旧 phase9b 路径的“权威”身份**  
   - 建议：权威改为 `NHC0801/runs/g001`；旧路径降为 `legacy_readonly_binding`

**推荐默认**: **(C) + (T 的工程设计，S 的执行顺序）** — 先打通小集全链路，并行栈按标定爬坡。

---

## 7. 风险与回滚

| 风险 | 缓解 |
| --- | --- |
| 并行 OOM 杀进程 | claim 看 MemAvailable；动态降 N；每 worker cgroup 内存上限 |
| 与 VASP 抢核 | claim 查 busy CPU；亲和集不相交；PSI 背压 |
| 科学栈串味 | forbidden_stacks；作业脚本钉死 P01 |
| 文件名再次混乱 | 禁止新 `phase9b_*` 文件名进入 NHC0801 权威路径 |
| 误删 pilot | 永不改 pilot 树；只增 NHC0801 |

回滚：停止 worker；保留 `runs/g001` 收据；不覆盖已有 versioned 目录（需 `--overwrite` 才许）。

---

## 8. 成功标准

- [ ] 权威路径无新的 `phase9b` 执行入口（证据区除外）  
- [ ] `runs/g001/meta/generation.json` 绑定 protocol/split/commit  
- [ ] 资源 claim +（若 N>1）并行 selection receipt 存在  
- [ ] 老师帧按 root 完整双端点入库，无 partial admit  
- [ ] D3 只消费冻结收据  
- [ ] Epoch-0 / 训练 / sci-val / FT 仍分项授权  
- [ ] `pytest` 绿；RETRO 记录并行相关坑  

---

## 9. 建议的「唯一下一步」（实现顺序）

1. **你确认 §6 决策**（范围 A/B/C 与并行 S/T）。  
2. 实现 **generation 布局 + 路径常量**（无 live）。  
3. 实现 **resource claim + worker 槽位模型**（只读探测 + 本地单测）。  
4. 资源空闲后：claim →（可选）并行标定 → **授权** teacher 并行。  

未确认 §6 前，不启动任何服务器化学。

---

## 10. 参考

- Pilot handoff: `/Users/cc/nhc-deprot-ranker-science-pilot/docs/HANDOFF_PHASE9B_V004_FOR_GROK.md`  
- Server performance contract: pilot `.codex/skills/.../server-performance-contract.md`  
- 本仓库: `mindmap.md`, `AGENTS.md`, `RETRO.md`, `pipeline/scientific_validation.py`  
