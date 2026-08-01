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

## Next unique engineering step

Multi-seed trainer loop (mindmap 4–5) that retains all outcomes and never final-selects on quick-val; then authorized epoch-0 execution.
