# Source Inventory

Record of what was extracted into NHC0801 during bootstrap (2026-08-01).
Paths are relative to the local project root unless noted.

## A. Connection / server facts (from legacy `AGENTS.md`)

| Fact | Value |
| --- | --- |
| SSH alias | `nhc614` (see `configs/server.local.yaml`) |
| HPC | WHUT internal; campus direct or SOCKS5 `127.0.0.1:11080` |
| Production root `$WJW` | `/home/plab/test/WJW` (**read-only for this project**) |
| This project root | `/home/plab/test/WJW/NHC0801` |
| Molecular env script | `$WJW/env/envs/molenv.sh` |
| MLFF / AIMNet2 env script | `$WJW/env/envs/mlff.sh` (also `aimnet2.sh`) |
| Official weight `_0` | `/home/plab/.cache/aimnet/aimnet2_wb97m_d3_0.pt` |
| Weight SHA256 | `f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28` |
| Size | 8836941 bytes |

Legacy full text preserved at `docs/extracted/legacy/AGENTS.md`.

## B. From `nhc-deprot-ranker` (local)

### Labels (small products only)

| File | Role |
| --- | --- |
| `data/labels/labels.parquet` | 71 high-fidelity labels product |
| `data/labels/label_source_membership.csv` | source membership |
| `data/labels/protocol_manifest.json` | protocol identity |
| `data/labels/source_manifest.json` | source identity |
| `data/labels/data_quality.json` | quality summary |

**Not copied:** `candidates.parquet` (~23 MB full library) — remains in ranker;
index only if needed later.

### Contracts / science docs → `docs/extracted/ranker/`

- `SCIENCE_SCOPE.md`, `DATA_CONTRACT.md`
- All `AIMNET2_*.md`, `PYSCF_RESIDUAL_OPTIMIZATION_CONTRACT.md`
- `PHASE9A_AIMNET2_PLAN.md`, `PHASE9A_I_REPORT.md`, `PHASE9A_I_MODEL_WEIGHT_CLOSURE.md`
- `NEXT_PHASE_AUTHORIZATION.md`, `MODEL_CARD.md`, `FAMILY_DEFINITION.md`, `LEGACY_AUDIT.md`

## C. From local legacy tree `/Users/cc/nhc614`

| File | Dest |
| --- | --- |
| `AGENTS.md` | `docs/extracted/legacy/AGENTS.md` |
| `doc/claude/commands.md` | `docs/extracted/legacy/` |
| `doc/claude/science-decisions.md` | `docs/extracted/legacy/` |
| `doc/claude/calc-params.md` | `docs/extracted/legacy/` |
| `doc/claude/deployment-notes.md` | `docs/extracted/legacy/` |
| `config/mlff/aimnet2_dftd3_b3lyp.yaml` | `docs/extracted/legacy/` |
| blind / deltaE CSVs | `data/labels/` |

## D. From server `$WJW` (read-only copy of small tables)

| Remote | Local |
| --- | --- |
| `$WJW/data/runs/mol_gold/gold_labels.csv` | `data/labels/gold_labels.csv` |
| `$WJW/data/runs/mol_gold/dft_gold.csv` | `data/labels/dft_gold.csv` |
| `$WJW/data/runs/mol_gold/gold24_lib.csv` | `data/labels/gold24_lib.csv` |

### Server assets **referenced, not copied** (large / shared)

| Asset | Path | Note |
| --- | --- | --- |
| Historical fine-tune checkpoints | `$WJW/checkpoints/*.pt` (~153 MB total) | Contamination risk if used as default; inventory only |
| Official AIMNet2 weight (wb97m) | `~/.cache/aimnet/aimnet2_wb97m_d3_0.pt` | Shared cache; `_0` only; SHA `f0f7c054…4e28` |
| AIMNet2-2025 ensemble (4 members) | `$WJW/env/model-cache/aimnet2-2025/aimnet2_2025_b973c_d3_{0..3}.pt` | Full ensemble on disk; hashes in `inventory/wjw_aimnet_weights.txt`; **not yet chosen** as NHC0801 epoch-0 |
| Conda stacks | `$WJW/env/envs/*.sh` | Use in place; do not clone env into NHC0801 yet |
| Related experiment tree | `$WJW/nhc-cu-mlip` (~75M) | Separate Cu-MLIP project; do not merge blindly |

## E. Explicitly not transferred

- Ranker Phase 8B private bundles, permits, consumed latches
- Full 401,856 candidate table
- Any PySCF trajectory teacher dataset (does not exist yet for this design)
- Server private wheelhouses / failed unified-env prefixes from Phase 9B-U*
