# Mindmap → Asset Map

How each `mindmap.md` stage maps to material available after bootstrap.

| Step | Mindmap | Available now | Gap / next work |
| --- | --- | --- | --- |
| 0 | Freeze molecular roots | Pilot 3+2 roots in `data/paths` + generation meta | Larger root freeze when scale-up |
| 1 | Train/Val/Test by root | `development_split` + sealed FT commitment | Keep FT sealed; no identity open |
| 2 | Pure-PySCF teacher frames | `teacher_runner` dry-run + pilot 235-frame bind | Live teacher gen when authorized |
| 3 | Epoch-0 baseline | Dry-run + **live parent worker** (`wb97m-d3bj`); `check_epoch0_receipts` | After live finish: audit campaign + root receipts |
| 4–5 | Train AIMNet2 + checkpoints | **Live train PASS** g001 3×200; `.pt` last-epoch | Export shortlist `.pt`s if sci-val needs mid epochs |
| 6–7 | Fast val + shortlist | Trainer quick-val + `checkpoint_shortlist` campaign | Run shortlist on server g001 train |
| 8–9 | Full scientific val + selection | Writer + `sci_val_campaign` dry-run; numeric cal frozen | Live sci-val after e0 + shortlist weights |
| 10 | Freeze | `freeze_package` (PROVISIONAL until live selection) | Hard freeze only after live VALIDATION_SELECTED |
| 11–12 | Final Test sealed | Commitment only; readiness gate | One-shot; not authorized |

## Recommended directory layout (server + local)

```text
NHC0801/
  mindmap.md
  AGENTS.md
  configs/
  docs/
  data/
    labels/           # small tables (present)
    roots/            # frozen molecular roots (TODO)
    splits/           # train/val/test key lists (TODO)
    teacher_gpu_g001/ # pure-pyscf frames (g001 teacher; g00N → teacher_gpu_g00N/)
    geometries/       # initial XYZ (TODO)
  models/
    official/         # pointers / copies of epoch-0 weight metadata
    checkpoints/      # fine-tunes produced by THIS project only
  runs/               # versioned attempts
  reports/
  src/
  scripts/
```

## Contaminating assets (handle carefully)

`$WJW/checkpoints/nhc_final_model_500.pt` and siblings may have been trained on
the same gold molecules used as labels. For mindmap compliance:

- Treat as **historical experiments**, not as the official epoch-0 baseline.
- Epoch-0 must be the **official** `aimnet2_wb97m_d3_0` weight.
- If any historical ckpt is re-evaluated, document train-set overlap first.
