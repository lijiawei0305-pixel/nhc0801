# Training blockers — diagnosis and solutions (NHC0801)

State machine target: leave `BLOCKED_BEFORE_TRAINING` only when **all hard blockers**
are cleared **and** the user opens `aimnet2_train_authorized`.

Machine-readable check:

```bash
PYTHONPATH=src python -c \
  "from nhc_deprot.pipeline.training_blockers import *; \
   print(format_readiness_report(assess_training_readiness()))"
```

---

## Blocker matrix

| Code | Severity | Status after this session | Solution |
| --- | --- | --- | --- |
| `NUMERIC_CALIBRATION_RULE_NOT_PREREGISTERED` | hard | **RESOLVED** | `docs/contracts/NUMERIC_CALIBRATION_V001.yaml` frozen; validated by `tvt_gates.validate_numeric_addendum` |
| `SOURCE_COMMIT_NOT_FROZEN` | hard | OPEN (no clean git freeze) | `git init` if needed → commit all NHC0801 code → record SHA in `PHASE_STATUS.md` / generation config. Dirty tree ≠ frozen. |
| `EPOCH_ZERO_FULL_ROUTE_BASELINE_NOT_AVAILABLE` | hard | OPEN | Needs **authorized** live run: official `_0` weight only → GAU_LOOSE → handoff → full parent P01 GAU on Validation roots → write receipt under `$WJW/NHC0801/runs/`. |
| `FULL_SCIENTIFIC_VALIDATION_WRITER_NOT_IMPLEMENTED` | hard | OPEN | Implement mindmap steps 8–9 writer (largest code gap). Must use P01 + GAU_LOOSE; never two_endpoint. |
| `LIVE_RESOURCE_CLAIM_REJECTED_OR_UNAVAILABLE` | hard if live | OPEN | Dual-worker V002 REJECTED (CPUs busy). Wait; re-claim; prefer `single_27_physical_v1` until dual-worker calibration PASSes. |
| `LIVE_TRAINING_NOT_AUTHORIZED` | hard | OPEN (by design) | User must explicitly authorize after other hard blockers clear. |
| `WEIGHTED_DEVELOPMENT_DATASET_NOT_READY` | hard | **RESOLVED** (pilot evidence) | Split + public weighted result OK; server NPZ still read-only at V004 product path. |

---

## What “automation” is allowed now

| Layer | Status |
| --- | --- |
| Preflight / plan (`mindmap_orchestrator`) | **Yes** — dry-run default |
| Parameterized dataset read / weight audit | **Yes** — local fixtures + server paths |
| Live epoch-0 / train / parent DFT | **No** until gates open |
| Final Test open | **No** |

Full auto train script shape:

```text
preflight (split + TVT + forbidden stacks + blockers)
  → refuse if open_hard
  → (auth) epoch-0 baseline
  → (auth) multi-seed train + ckpt retain
  → quick-val shortlist only
  → (auth) full sci Validation
  → select via numeric addendum
  → freeze
  → (auth) Final Test once
```

Orchestrator implements the control plane; live stages remain stubs until authorized.

---

## Forbidden stacks (do not “solve” blockers by switching science)

- Production **B3LYP/def2-SVP** `two_endpoint` — archived under `docs/archive/forbidden_b3lyp_stack/`
- Selecting final model by quick-val loss — banned in `forbidden_stacks.py`
- Using `$WJW/checkpoints/*.pt` as epoch-0 — gold contamination risk

---

## Recommended unlock order

1. Freeze source commit (local git).
2. Implement scientific Validation writer (code only).
3. Resource claim when CPUs free.
4. User authorizes **epoch-0 only** → run baseline → freeze receipt.
5. User authorizes **one-shot multi-seed train**.
6. Sci Validation → select → freeze → Final Test once.

Never open Final Test to “debug” training blockers.
