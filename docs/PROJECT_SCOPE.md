# Project Scope — NHC0801

## Goal

Implement the pipeline in `mindmap.md`:

1. Freeze molecular roots (cation + neutral, order, coords, charge, mult, SHA256)
2. Split by molecular root into Train / Validation / Final Test (disjoint)
3. Generate Pure-PySCF teacher frames (geometry, energy, forces)
4. Epoch-0 baseline with official AIMNet2 → handoff → PySCF
5. Train AIMNet2 on Train frames only
6–9. Validate, shortlist checkpoints, full scientific validation, select one
10. Freeze model + splits + contracts
11–12. One-shot Final Test; no post-hoc selection

## In scope

- Document-first design for splits, GAU / GAU_LOOSE, teacher protocol
- Local packaging and server workspace under `$WJW/NHC0801`
- Reuse of **small** audited labels and contracts from prior work
- Future: teacher DFT generation, fine-tune, evaluation (each step needs auth)

## Out of scope (unless re-authorized)

- Continuing `nhc-deprot-ranker` Phase 9B execution gates / v10 runner remediation
- Writing into `$WJW` outside `NHC0801`
- Using legacy fine-tuned checkpoints in `$WJW/checkpoints/` as production defaults
  without a contamination audit (many were trained on overlapping gold sets)
- Downloading AIMNet2 ensemble members `_1`–`_3`
- Ranking the full 401,856 library (that remains the ranker project’s B0 line)
- Hessian / ZPE / free-energy labels

## Scientific inheritance (kept)

- Deprotonation electronic-energy definition and 6.28 kcal/mol constant
- InChIKey primary key
- Endpoint charges/multiplicities
- Exact-byte handoff idea between AIMNet2 geometry and PySCF
- Fail-closed structural / charge checks

## Scientific change vs ranker Phase 9

| Ranker Phase 9 | This project |
| --- | --- |
| Official AIMNet2 as **preoptimizer only** | AIMNet2 **fine-tuned** on PySCF frames |
| Single-candidate paired smoke | Multi-root Train/Val/Test |
| Do not train | Explicit training + checkpoint selection |
| Labels always PySCF (unchanged) | Labels still PySCF; forces train the ML model |

## Initial data available after bootstrap

- 71-label product (`data/labels/labels.parquet` + membership)
- Server gold tables (`gold_labels.csv`, `dft_gold.csv`, …)
- Blind round CSVs from legacy reports
- Contract docs under `docs/extracted/`
- **No** complete teacher trajectory dataset yet
- **No** frozen Train/Val/Test split yet
