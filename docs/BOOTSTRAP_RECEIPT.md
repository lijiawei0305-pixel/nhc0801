# Bootstrap Receipt — NHC0801

Date: 2026-08-01  
Authorization: user explicit permission to create `$WJW/NHC0801` and seed the new project.

## Actions performed

1. Read-only SSH preflight to `nhc614`; confirmed `NHC0801` absent.
2. Built local project at `/Users/cc/nhc-deprot` from `mindmap.md`.
3. Extracted small labels + contracts from:
   - `/Users/cc/nhc-deprot-ranker`
   - `/Users/cc/nhc614` (legacy)
   - `$WJW/data/runs/mol_gold/*` (server CSV only)
4. Created **new** directory only:
   - `/home/plab/test/WJW/NHC0801`
5. `rsync -avz` (no `--delete`) of project files into that directory only.
6. Did **not** modify `$WJW/env`, production `data/`, `checkpoints/`, or other siblings.

## Verification

| Check | Result |
| --- | --- |
| Remote root exists | yes |
| `mindmap.md` / `AGENTS.md` present | yes |
| Labels present | `labels.parquet`, `gold_labels.csv`, … |
| Official weight SHA (in place) | `f0f7c054…4e28` |
| Writes outside NHC0801 | none intended; only `mkdir NHC0801` + rsync into it |
| Teacher DFT / training | not run |
| Remote file count (approx) | 53 files, ~452K |

## Closed gates after bootstrap

```text
teacher_pyscf_authorized: false
aimnet2_train_authorized: false
scheduler_submission_authorized: false
modify_wjw_outside_project: false
```
