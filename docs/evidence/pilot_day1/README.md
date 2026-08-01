# Pilot day-1 evidence (NHC0801 names)

Canonical **read-only** pilot bindings from science-pilot V004, renamed for this project.

| File | Role |
| --- | --- |
| `DEVELOPMENT_SPLIT.json` | Train 3 + Val 2 roots; sealed Final Test commitment only |
| `WEIGHTED_DATASET_RESULT.json` | Weighted NPZ product public result (235 frames evidence) |
| `D3_PROJECTION_RESULT.json` | Frozen D3 projection public result |
| `GENERATION_CONFIG.json` | Frozen generation config snapshot (historical phase9b id) |
| `GAU_LOOSE_SOURCE.yaml` | Source of GAU_LOOSE (also under `docs/contracts/`) |
| `PILOT_HANDOFF.md` | Codex V004 handoff (context only; mindmap + this repo win) |

Server paths (still under `$WJW`, read-only):

```text
data/runs/phase9b_aimnet2_v004_weighted_dataset_v001
data/runs/phase9b_aimnet2_v004_d3_projection_v001
data/runs/autofill_{key_lower}_v001/training_data/
```

Do **not** treat filenames containing `phase9b` / `V004` as alternate science authority.
Parent stack is **P01 only**. B3LYP/SVP materials live under `docs/archive/forbidden_b3lyp_stack/`.
