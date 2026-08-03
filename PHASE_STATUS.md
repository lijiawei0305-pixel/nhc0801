# Phase Status — NHC0801

Updated: 2026-08-03

**看不懂 g001 / e0 / Val-only？** → 读 **`docs/NHC0801_命名与进度指南.md`**（命名词典 + 查进度命令 + 快照）。

| Phase | Status | Notes |
| --- | --- | --- |
| P0 Bootstrap | Complete | Local tree + server NHC0801 |
| P0.1–P0.4 | Complete | Handoff port, dataset reader, numeric cal, orchestrator |
| P0.5 Sci-Val writer | Complete | `pipeline/scientific_validation.py` (live engines gated) |
| P1 Freeze roots + split | Partial | Pilot 3+2 + sealed FT |
| P2 Teacher Pure-PySCF | Pilot bound | 235-frame weighted pilot reused under g001 |
| P3 Epoch-0 baseline | **LIVE RUNNING** | Parent P01 `wb97m-d3bj`; **must audit receipts on finish** |
| P4 AIMNet2 training | **LIVE_TRAIN_PASS** | 3 seeds × 200 ep; `.pt` under `g001/train/seed_*` |
| P5 Sci Validation / select | **Dry-run ready** | shortlist campaign + sci_val dry; live gated |
| P6 Freeze + Final Test | **PROVISIONAL freeze** | `freeze_manifest`; Final Test still sealed |

## Live dual-path (2026-08-02)

| Track | Resource | Status |
| --- | --- | --- |
| Multi-seed train | GPU0, mlff, OMP=4 | **PASS** (~5 min wall for 3×200) |
| Epoch-0 | GPU1 (AIMNet2) + CPU OMP=12 gpupyscf | **RUNNING** (parent DFT slow) |
| Claim | single_27_physical_v1 | PASS before e0 relaunch |

Artifacts:

- `runs/nhc0801-g001/train/campaign_receipt_live.json` → `LIVE_TRAIN_PASS`
- `runs/nhc0801-g001/train/seed_{20260730,20260731,20260732}/epoch_0200.pt`
- `runs/nhc0801-g001/sci_val/shortlist_campaign.json` → 12 candidates (weights_present=3 last-epoch)
- `runs/nhc0801-g001/freeze/freeze_manifest.json` → `PROVISIONAL`
- `runs/nhc0801-g001/logs/live_epoch0.out` (monitor)
- **On epoch0 exit:** `python scripts/nhc0801_check_epoch0_receipts.py` → campaign + root receipts
- Worker: `scripts/nhc0801_pyscf_parent_worker.py` (must use `wb97m-d3bj`, not plain `wb97m`)

## Resource + automation design (docs only, 2026-08-02)

| Doc | Role |
| --- | --- |
| `docs/contracts/RESOURCE_SCHEDULING_V001.md` | **02c trial**: t=10、预留 12 核(100–111)、8 GiB/端点、10 端点一波 |
| `docs/contracts/RESOURCE_PROFILES_V002.yaml` | `auto_fill_112_t10_r12_v1` rev **2026-08-02c** |
| `docs/plans/20260802_automation_tui_design.md` | mindmap 0–10 编排 + SSH 只读 TUI 30s |

**Implemented (2026-08-02):**

| Module | Role |
| --- | --- |
| `resources/profiles.py` | V001+V002 load; `auto_fill_112_t8_v1` |
| `resources/auto_fill.py` | \(N=\min(N_{cpu},N_{mem})\) + 8-CPU endpoint slots |
| `pipeline/pipeline_status.py` | scan/write `pipeline_status.json` |
| `dashboard/tui.py` | SSH read-only TUI (30s / `--once`) |
| CLIs | `nhc0801_tui.py`, `nhc0801_auto_fill_plan.py`, `nhc0801_pipeline_status.py` |

```bash
# on nhc614
PYTHONPATH=src python3 scripts/nhc0801_tui.py --once
PYTHONPATH=src python3 scripts/nhc0801_auto_fill_plan.py --idle-cpus 0-111 --mem-gib 230 --roots A,B
PYTHONPATH=src python3 scripts/nhc0801_pipeline_status.py --nhc0801-root $PWD --write
```

## Training readiness

| Blocker | Status |
| --- | --- |
| NUMERIC_CALIBRATION_… | RESOLVED |
| FULL_SCIENTIFIC_VALIDATION_WRITER_… | **RESOLVED** (writer implemented; live gated) |
| WEIGHTED_DEVELOPMENT_DATASET | RESOLVED (pilot evidence) |
| SOURCE_COMMIT_NOT_FROZEN | **RESOLVED** at `fb5116a4dc6dc8cac55575f2eadde556a02d24c1` |
| EPOCH_ZERO_… | **IN PROGRESS** (live parent P01) |
| LIVE_TRAINING_NOT_AUTHORIZED | **pilot live train done** (g001; not Final Test) |
| LIVE_RESOURCE_CLAIM | PASS when CPUs idle |

## Gates still closed

teacher DFT / epoch-0 / AIMNet2 train / sci-val **live** / Final Test / write outside NHC0801

## Process hygiene (2026-08-01)

- `AGENTS.md`: 落盘硬规则 + 写文档规矩
- `RETRO.md`: 工程复盘日志（踩坑先查）
- `docs/plans/`: vibe coding 前规划
- `scripts/`: CLI 薄封装 only

## Generation + resources (scope C / parallel S)

| Item | Status |
| --- | --- |
| Decision | C (pilot first) + S (single→dual after claim) + code only, no live |
| Generation layout | `src/nhc_deprot/generation/` → `runs/<id>/…` |
| Profiles | `docs/contracts/RESOURCE_PROFILES_V001.yaml` |
| Claim eval | `resources/claim.py` (injected snapshots; no SSH) |
| Worker slots | `resources/worker_pool.py` (no process spawn) |
| CLI | `scripts/nhc0801_init_generation.py`, `nhc0801_resource_claim_eval.py` |
| Live chemistry | **still closed** |

## Teacher runner (mindmap 2)

| Item | Status |
| --- | --- |
| Module | `pipeline/teacher_runner.py` |
| CLI | `scripts/nhc0801_teacher_runner.py` |
| Mode | **dry-run default** → synthetic frames + receipts under `teacher_gpu_g001/` |
| Pool | `worker_pool` root claims; cation→neutral per root |
| Live | Closed (`teacher_pyscf_authorized` + non-dry engine required) |

```bash
PYTHONPATH=src python scripts/nhc0801_teacher_runner.py --plan-only
PYTHONPATH=src python scripts/nhc0801_teacher_runner.py --frames-per-endpoint 2
```

## D3 + weighted dataset (mindmap residual path, dry-run)

| Item | Status |
| --- | --- |
| D3 module | `pipeline/d3_projection.py` → `g001/d3/` |
| Weighted writer | `pipeline/weighted_dataset_writer.py` → `g001/datasets/weighted/` |
| CLI chain | `scripts/nhc0801_d3_weighted_dry_run.py` |
| Audit | reuses `data/weighted_dataset.audit_weighted_dataset` |
| Live D3/PySCF | **closed** |

```bash
PYTHONPATH=src python scripts/nhc0801_d3_weighted_dry_run.py \
  --nhc0801-root runs/local_nhc0801 --frames-per-endpoint 2
```

## Epoch-0 (mindmap 3, dry-run)

| Item | Status |
| --- | --- |
| Module | `pipeline/epoch0_runner.py` |
| CLI | `scripts/nhc0801_epoch0_dry_run.py` |
| Scope | Validation roots only (pilot 2) |
| Routes | Pure-PySCF reference **and** official `_0` AIMNet2 → GAU_LOOSE → handoff → parent GAU |
| Weight identity | `OFFICIAL_AIMNET2_WEIGHT_SHA256` (`aimnet2_wb97m_d3_0`) |
| Live | Closed (`epoch0_execution` + non-simulated engines) |

```bash
PYTHONPATH=src python scripts/nhc0801_epoch0_dry_run.py --plan-only
PYTHONPATH=src python scripts/nhc0801_epoch0_dry_run.py \
  --nhc0801-root runs/local_nhc0801
```

## Multi-seed trainer (mindmap 4–5, dry-run)

| Item | Status |
| --- | --- |
| Config | `training/config.py` (seeds/epochs/lr frozen defaults) |
| Loop | `training/multi_seed_trainer.py` |
| CLI | `scripts/nhc0801_train_dry_run.py` |
| Data | reads `g001/datasets/weighted` |
| Policy | quick-val **never** final select; retain all seed/epoch outcomes |
| Live | Closed (`aimnet2_train_authorized` + non-dry backend) |

```bash
PYTHONPATH=src python scripts/nhc0801_train_dry_run.py \
  --nhc0801-root runs/local_nhc0801 --bootstrap-data --epochs 5
```

## Ops 2026-08-02

- Stopped Codex phase9b pure_pyscf workers (VPAF on 0,2-27; RBKF on 28-55) + supervisor  
- Reused pilot 235-frame weighted+D3 into `NHC0801/runs/nhc0801-g001` (symlink bind)  
- Server claim **PASS**; train dry-run **PASS**; epoch0 dry-run **PASS**  
- See `docs/science/OPS_20260802_STOP_CODEX_AND_REUSE.md`  
- Live DFT/AIMNet2 still not wired  

## Resource claim sampler (read-only)

| Item | Status |
| --- | --- |
| Probe | `resources/host_sampler.py` (local or SSH BatchMode) |
| Orchestration | `resources/claim_runner.py` → `g001/resources/claim_*.json` |
| CLI | `scripts/nhc0801_resource_claim.py` |
| Chemistry | **never started**; PASS ≠ open teacher/epoch0/train gates |

```bash
# Remote two-sample claim (uses configs/server.local.yaml ssh_alias)
PYTHONPATH=src python scripts/nhc0801_resource_claim.py --mode ssh --interval-s 5

# Local machine only (dev / non-HPC)
PYTHONPATH=src python scripts/nhc0801_resource_claim.py --mode local
```

## Next unique engineering step

科学顺序 live：claim PASS → 授权 teacher → 真 D3 → epoch-0 → train。  
工程可选：PySCF/AIMNet2 live 引擎接线，或端到端 dry-run 编排脚本。
