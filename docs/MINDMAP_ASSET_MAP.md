# Mindmap → Asset Map

How each `mindmap.md` stage maps to material available after bootstrap.

| Step | Mindmap | Available now | Gap / next work |
| --- | --- | --- | --- |
| 0 | Freeze molecular roots | Partial: labels + gold tables identify molecules; few validated XYZ (ranker Phase 7 smoke only, not bulk) | Define root schema; generate/freeze initial geometries for chosen roots |
| 1 | Train/Val/Test by root | **Not defined** | Write split protocol; freeze key lists |
| 2 | Pure-PySCF teacher frames | Contracts in `docs/extracted/ranker/PYSCF_*`, `SCIENCE_SCOPE` | Implement teacher runner under NHC0801; no frames yet |
| 3 | Epoch-0 baseline | Official weight hash known; handoff contract docs | Run assisted path after teacher refs exist |
| 4–5 | Train AIMNet2 + checkpoints | Legacy training YAML extract; historical ckpts on server (read-only) | New training code + versioned checkpoint dir **inside NHC0801** |
| 6–7 | Fast val + shortlist | — | Val frame set from step 2 |
| 8–9 | Full scientific val + selection | Promotion-gate ideas in ranker `AIMNET2_PROMOTION_GATES.md` (adapt, do not copy ranks) | Preregister selection rules **before** looking at Test |
| 10 | Freeze | — | Manifests + SHA256 |
| 11–12 | Final Test sealed | — | One-shot only |

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
    teacher/          # pure-pyscf frames (TODO)
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
