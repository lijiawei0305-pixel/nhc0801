# Phase Status — NHC0801

Updated: 2026-08-01 (scientific Validation writer + README + GitHub)

| Phase | Status | Notes |
| --- | --- | --- |
| P0 Bootstrap | Complete | Local tree + server NHC0801 |
| P0.1–P0.4 | Complete | Handoff port, dataset reader, numeric cal, orchestrator |
| P0.5 Sci-Val writer | Complete | `pipeline/scientific_validation.py` (live engines gated) |
| P1 Freeze roots + split | Partial | Pilot 3+2 + sealed FT |
| P2 Teacher Pure-PySCF | Not started (pilot data exists) | Live gen needs auth |
| P3 Epoch-0 baseline | Not started | Contract ready; NOT_RUN |
| P4 AIMNet2 training | Blocked | Needs source freeze + epoch-0 + auth |
| P5 Sci Validation / select | Writer ready | Live route needs engines + gate |
| P6 Freeze + Final Test | Not started | Sealed |

## Training readiness

| Blocker | Status |
| --- | --- |
| NUMERIC_CALIBRATION_… | RESOLVED |
| FULL_SCIENTIFIC_VALIDATION_WRITER_… | **RESOLVED** (writer implemented; live gated) |
| WEIGHTED_DEVELOPMENT_DATASET | RESOLVED (pilot evidence) |
| SOURCE_COMMIT_NOT_FROZEN | **RESOLVED** at `fb5116a4dc6dc8cac55575f2eadde556a02d24c1` |
| EPOCH_ZERO_… | OPEN |
| LIVE_TRAINING_NOT_AUTHORIZED | OPEN (default) |
| LIVE_RESOURCE_CLAIM | OPEN if CPUs busy |

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
| Mode | **dry-run default** → synthetic frames + receipts under `g001/teacher/` |
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

## Next unique engineering step

Mindmap **步骤 3**：epoch-0 全路线骨架（dry-run / 接 handoff + sci-val 合同），  
或 **步骤 4–5** multi-seed trainer 循环。Live 仍需 claim + 分项授权。
