# AGENTS.md — NHC0801 / nhc-deprot

Mindmap-driven AIMNet2 fine-tune on **Parent-Level P01** teacher frames.
Local: `/Users/cc/nhc-deprot`. Server writes: **only** `$WJW/NHC0801`.

Not a continuation of `nhc-deprot-ranker` Phase 9B. Not production `two_endpoint`.

---

## Before any code change

1. Read **`mindmap.md`** (science truth) + this file + **`PHASE_STATUS.md`**.
2. Map work to a **mindmap step 0–12** (table below). Do not skip gates.
3. Prefer `src/nhc_deprot/` modules; never write outside `$WJW/NHC0801` on the server.
4. Run `PYTHONPATH=src python -m pytest -q` after code changes.
5. Live chemistry / training / Final Test only with **explicit user authorization**.

Conflict order: **mindmap.md** → frozen YAML/JSON SHA bindings → this file → pilot evidence.

---

## Mindmap steps (implement against these)

| Step | What | NHC0801 modules / status |
| ---: | --- | --- |
| **0** | Freeze molecular roots (cation+neutral, SHA, charge/mult) | `data/paths`, `data/development_split` — pilot 5 roots frozen |
| **1** | Split by **molecular_root**: Train ∩ Val ∩ Test = ∅ | `contracts/tvt_gates`, `data/development_split` — FT **sealed only** |
| **2** | Pure-PySCF teacher frames (geom/E/F every step) | `data/teacher_frames`, `data/weighted_dataset` — pilot frames on server; scale gen not ported |
| **3** | Epoch-0: official AIMNet2 → **GAU_LOOSE** → handoff → full parent GAU | `pipeline/parent_handoff` — contract OK; **execution NOT_RUN** |
| **4** | Train AIMNet2 on **Train frames only** (residual E/F after frozen D3) | `training/weighted_loss`, `data/weighted_dataset` — **no live train** |
| **5** | Multi-epoch checkpoints (retain all seeds/outcomes) | trainer loop **missing** |
| **6** | Quick Validation on fixed Val frames (no new DFT; **not** final select) | `WeightedEvaluationAccumulator` ready |
| **7** | Shortlist few checkpoints | `tvt_gates.quick_checkpoint_shortlist` |
| **8** | Full scientific Validation route per shortlist | **largest gap** — writer not implemented |
| **9** | Val selects one checkpoint (numeric addendum) | `tvt_gates.select_scientific_checkpoint` + `NUMERIC_CALIBRATION_V001.yaml` |
| **10** | Freeze identities (ckpt SHA, splits, protocols, commit) | readiness gates |
| **11** | Final Test **once** | sealed commitment only; identities closed |
| **12** | No post-Test model shopping | policy fail-closed |

Orchestrator (preflight only):  
`PYTHONPATH=src python -m nhc_deprot.pipeline.mindmap_orchestrator`

Routing table: `src/nhc_deprot/mindmap_steps.py`.

---

## Science non-negotiables

- Reaction: `NHC-H+ → NHC + H+`. Endpoints: cation (+1,s), neutral (0,s).
- **Parent = P01 only**: gas RKS, `wb97m-d3bj` / `def2-TZVPP`, grid=4, SCF 1e-9.  
  SHA256 `227c22a527e567bc4de873ab743fe9f493779eccbb1a698d2913c87695ebf87a`.
- **GAU_LOOSE** (AIMNet2 stop): 5 criteria + ASE LBFGS fmax **0.10** eV/Å, max 100.  
  Not VASP “0.1”. Not production fmax **0.05**.
- Route: freeze geom → AIMNet2 to GAU_LOOSE → gates → exact-byte handoff → **full** parent opt to GAU → SP → label. `single_point_only` always false.
- Labels: `(E_n - E_c)*627.509474 - 6.28` kcal/mol; **AIMNet2 energy never in labels**.
- Train targets: residual short-range E/F after frozen two-body D3(BJ); no silent D3 recompute.
- Quick-val **must not** choose the final model.

### Forbidden (attention + code)

| Banned | Why |
| --- | --- |
| Production `two_endpoint` B3LYP/def2-SVP | Wrong parent theory |
| fmax=0.05 as parent stop | Wrong AIMNet2 contract |
| Historical finetune “best by quick-val loss” | Violates mindmap 6–9 |
| Opening Final Test identities for “dev convenience” | Leak / selection bias |
| Writing outside `$WJW/NHC0801` | Boundary |

Archived B3LYP materials: `docs/archive/forbidden_b3lyp_stack/` (do not use).  
Ban helpers: `contracts/forbidden_stacks.py`.

---

## Gates (default all closed)

```text
teacher_pyscf_authorized: false
aimnet2_train_authorized: false
epoch0_execution: false
final_test_open: false
scientific_validation_live: false
modify_wjw_outside_NHC0801: false
scheduler_submission: false
```

Allowed without new auth: local code/tests/docs; read-only SSH inventory; rsync **into** NHC0801 (no `--delete`).

---

## Server / env

| | |
| --- | --- |
| SSH | alias in `configs/server.local.yaml` (default `nhc614`) — never commit secrets |
| Write root | `/home/plab/test/WJW/NHC0801` |
| ML env | `source $WJW/env/envs/mlff.sh` |
| PySCF env | `source $WJW/env/envs/molenv.sh` (never mix stacks) |
| Epoch-0 weight | `~/.cache/aimnet/aimnet2_wb97m_d3_0.pt` only (not `$WJW/checkpoints/*.pt`) |
| Pilot frames | `$WJW/data/runs/autofill_*` + weighted NPZ product (read-only) |

```bash
rsync -avz --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' \
  -e 'ssh -o BatchMode=yes' \
  /Users/cc/nhc-deprot/ nhc614:/home/plab/test/WJW/NHC0801/
```

---

## Key paths

| Path | Role |
| --- | --- |
| `mindmap.md` | Science pipeline truth |
| `docs/contracts/GAU_LOOSE_V001.yaml` | AIMNet2 stop contract |
| `docs/contracts/NUMERIC_CALIBRATION_V001.yaml` | Val selection thresholds (frozen) |
| `docs/evidence/pilot_day1/` | Renamed pilot split/results (use these) |
| `docs/archive/forbidden_b3lyp_stack/` | Quarantined wrong stack |
| `src/nhc_deprot/` | All new code |

---

## Training blockers (current)

| Code | How to clear |
| --- | --- |
| `SOURCE_COMMIT_NOT_FROZEN` | `git init` + clean commit; record SHA |
| `EPOCH_ZERO_…_NOT_AVAILABLE` | Authorized epoch-0 full route + receipt |
| `NUMERIC_CALIBRATION_…` | **Done** — `NUMERIC_CALIBRATION_V001.yaml` |
| `FULL_SCIENTIFIC_VALIDATION_WRITER_…` | Implement mindmap 8 writer (next hard gap) |
| `LIVE_RESOURCE_CLAIM_…` | Wait for free CPUs; re-claim (V002 was REJECTED) |
| `LIVE_TRAINING_NOT_AUTHORIZED` | User must open `aimnet2_train_authorized` |

Diagnose: `PYTHONPATH=src python -c "from nhc_deprot.pipeline.training_blockers import *; print(format_readiness_report(assess_training_readiness()))"`

---

## Work style

- One science/split/auth change at a time; confirm with user if it widens scope.
- Synthetic fixtures in tests; no HPC required for unit tests.
- Do not `git reset/clean` science-pilot or modify ranker production trees.
- Prefer implementing the next **missing mindmap module** over expanding scope.

**Next hard engineering gap:** full scientific Validation writer (steps 8–9), then multi-seed trainer loop that never final-selects on quick-val.
