# Code Reuse Audit V001 — science-pilot → NHC0801

Date: 2026-08-01  
Sources audited: `/Users/cc/nhc-deprot-ranker-science-pilot` (V004 dirty worktree),  
`/Users/cc/nhc-deprot-ranker`, server `$WJW`, target `/Users/cc/nhc-deprot`.

Primary science authority: **`mindmap.md`**.  
Operational evidence: `docs/extracted/v004/HANDOFF_PHASE9B_V004_FOR_GROK.md`.

---

## 1. Current V004 state (from handoff)

| Item | Value |
| --- | --- |
| Generation | `phase9b-aimnet2-nhc-p01-tvt-20260801-v001` |
| State | `BLOCKED_BEFORE_TRAINING` |
| Frames | 235/235 admitted (123 train + 112 val) |
| D3 projection | PASS (server `phase9b_aimnet2_v004_d3_projection_v001`) |
| Weighted dataset | PASS (server `phase9b_aimnet2_v004_weighted_dataset_v001`) |
| Live training | NOT_RUN |
| Epoch-0 full route | writer static OK, execution NOT_RUN |
| Full scientific Validation writer | **not implemented** |
| Dual-worker live claim | V002 REJECTED (CPUs busy) |
| Production labels | still 71; unchanged |

Hard blockers:

```text
SOURCE_COMMIT_NOT_FROZEN
EPOCH_ZERO_FULL_ROUTE_BASELINE_NOT_AVAILABLE
NUMERIC_CALIBRATION_RULE_NOT_PREREGISTERED
FULL_SCIENTIFIC_VALIDATION_WRITER_NOT_IMPLEMENTED
```

---

## 2. Two incompatible science stacks (do not mix)

| Stack | AIMNet2 stop | Parent DFT | Use for mindmap? |
| --- | --- | --- | --- |
| **V004 / Parent-Level P01** | 5-criteria GAU_LOOSE + ASE fmax 0.10 / 250 steps | **ωB97M-D3(BJ)/def2-TZVPP** grid 4, geomeTRIC `GAU` | **YES** |
| **Production Phase 9B two_endpoint** | fmax **0.05** eV/Å / 200 steps | **B3LYP-D3(BJ)/def2-SVP** grid 3 | **NO** as parent |

Ported NHC0801 code uses **P01 + GAU_LOOSE only**.

---

## 3. Reuse matrix

| Source | Verdict | NHC0801 destination |
| --- | --- | --- |
| `phase9b_aimnet2_tvt_contract.py` | **copy** (rename) | `src/nhc_deprot/contracts/tvt_gates.py` |
| `phase9b_aimnet2_v004_weighted_loss.py` | **copy** | `src/nhc_deprot/training/weighted_loss.py` |
| `phase9b_aimnet2_parent_handoff.py` | **adapt** | `src/nhc_deprot/pipeline/parent_handoff.py` |
| `PHASE9B_AIMNET2_GAU_LOOSE_V001.yaml` | **adapt** | `docs/contracts/GAU_LOOSE_V001.yaml` |
| `phase9b_aimnet2_finetune_v004.py` | **adapt** (adapter only) | `src/nhc_deprot/training/trainer_adapter.py` |
| `phase9b_aimnet2_training_dataset*.py` | **adapted (reader only)** | `data/development_split.py`, `data/weight_policy.py`, `data/weighted_dataset.py` — **no 235 hardcode**; counts from split/manifest |
| `phase9b_aimnet2_v004_weighted_dataset.py` | **adapted (audit/reader)** | `data/weighted_dataset.py` audit + NPZ key set; no writer/live assemble |
| `phase9b_aimnet2_v004_d3_projection.py` | **adapt later** | needs PySCF env; NHC0801 forbids silent D3 recompute |
| Historical `phase9b_aimnet2_finetune.py` | **do not reuse selection** | selects best by quick-val loss — violates mindmap |
| `quantum/two_endpoint.py` | **infra patterns only** | wrong locked protocol |
| A1/A2 campaign/permit stack | **do not port** | control-plane debt |
| 401k xTB parquet/CSV | **reference OK** | ranker + `$WJW` crude CSV |

---

## 4. Mindmap vs V004 differences

| Topic | mindmap.md | V004 / handoff | NHC0801 decision |
| --- | --- | --- | --- |
| AIMNet2 role | train on teacher frames; also GAU_LOOSE preopt route | preconditioner + fine-tune residual E/F | **both**: train on P01 frames; routes use GAU_LOOSE→full parent |
| Teacher theory | Pure PySCF parent | P01 wb97m-d3bj/TZVPP | **P01** |
| D3 residual targets | not named | E_short = E_total − E_D3 | **keep** (required for base AIMNet2) |
| Quick val selects model? | no | no | **no** |
| Final Test | sealed | commitment `834f9739…` | **never open in training** |
| Parent after handoff MISS | continue | continue | **continue** |
| Historical B3LYP smoke labels | n/a | v001–v006 history | **not** parent truth |

---

## 5. Server assets (read-only outside NHC0801)

| Asset | Path |
| --- | --- |
| Teacher frames | `$WJW/data/runs/autofill_{key_lower}_v001/training_data/{cation\|neutral}/frame_*.json` |
| D3 receipts | `$WJW/data/runs/phase9b_aimnet2_v004_d3_projection_v001` |
| Weighted NPZ | `$WJW/data/runs/phase9b_aimnet2_v004_weighted_dataset_v001` |
| Gold XYZ | `$WJW/data/runs/mol_gold/xyz/` |
| xTB ~401856 | `$WJW/results/calculations/20260628/imid_v4_crude/imid_full_v4menu_crude_0618_method.csv` (138M) |
| Local ranker product | `/Users/cc/nhc-deprot-ranker/data/processed/v001/candidates.parquet` |
| Official weight | `/home/plab/.cache/aimnet/aimnet2_wb97m_d3_0.pt` SHA `f0f7c054…` |
| Env | `$WJW/env/envs/{mlff,molenv,aimnet2}.sh` |

Total autofill frames on server ≈ 387 (includes incomplete roots); **admitted development = 235**.

---

## 6. What was created under NHC0801

```text
src/nhc_deprot/
  contracts/  parent_protocol, tvt_gates, GAU_LOOSE_V001.yaml
  pipeline/   parent_handoff
  training/   weighted_loss, trainer_adapter
  data/       paths, development_split, teacher_frames,
              weight_policy, weighted_dataset, io_util, errors
  mindmap_steps.py
docs/REUSE_AUDIT_V001.md
docs/MINDMAP_VS_V004.md
docs/extracted/v004/   (handoff + evidence JSON references)
tests/test_weighted_loss.py
tests/test_parent_protocol_and_gau.py
tests/test_development_split.py
tests/test_weight_policy.py
tests/test_weighted_dataset.py
tests/test_teacher_frames.py
```

---

## 7. Explicit non-actions (still forbidden without new auth)

- Live AIMNet2 training or epoch-0 chemistry
- Opening Final Test identities
- Writing outside `$WJW/NHC0801`
- Using production B3LYP/SVP as parent
- Using historical `$WJW/checkpoints/*.pt` as epoch-0
- Selecting final model by quick-validation frame loss
- Silent D3 recomputation (must consume frozen receipts)

---

## 8. Recommended next implementation order (code only first)

1. ~~Port parameterized dataset reader (no 235 hardcode) + NPZ loader~~ **DONE (P0.2)**  
2. Preregister numeric-calibration procedure (docs freeze)  
3. Implement full scientific Validation route writer (GAU_LOOSE → parent GAU)  
4. Implement multi-seed trainer loop that **never** final-selects on quick val  
5. Epoch-0 execution only after resource claim + authority  

---

## Appendix A — P0.2 parameterized dataset reader (2026-08-01)

### A.1 Modules

| Module | Role |
| --- | --- |
| `nhc_deprot.data.development_split` | Load day1 split JSON; reject Final Test identity keys; opaque commitment only |
| `nhc_deprot.data.teacher_frames` | Autofill path inventory; D3 receipt path layout; no JSON chemistry parse |
| `nhc_deprot.data.weight_policy` | `equal_candidate_then_equal_endpoint_then_uniform_frames`; audit sum=1 |
| `nhc_deprot.data.weighted_dataset` | Manifest schema + NPZ `REQUIRED_ARRAYS`; energy/force reconstruction; optional expected counts |
| `nhc_deprot.data.paths` | `$WJW` conventions; V004 product relative paths; sealed FT commitment constants |

### A.2 Hardcode policy (mindmap-first)

| Quantity | V004 pilot evidence | Code policy |
| --- | --- | --- |
| 235 frames | day1 admitted total | **Not** a global constant; derived from NPZ/manifest or public result JSON |
| 123 / 112 | train / val frames | Same — optional `expected_*` args for binding audits only |
| 3+2 roots | day1 split | Loaded from split file; pilot binding via `require_v004_pilot_roots` |
| Parent protocol SHA | P01 | Constant in `parent_protocol.py` / `paths.py` (science identity) |
| Sealed FT commitment | `834f9739…` | Constant for binding; identities never loaded |

### A.3 Server read-only targets

```text
$WJW/data/runs/phase9b_aimnet2_v004_weighted_dataset_v001   # weighted NPZ + manifest
$WJW/data/runs/phase9b_aimnet2_v004_d3_projection_v001      # frozen D3 receipts
$WJW/data/runs/autofill_{key_lower}_v001/training_data/     # teacher frames
```

Local package entry: `default_v004_weighted_dataset_root(wjw)`.  
Full server NPZ audit requires SSH-mounted or remote run; local tests use synthetic fixtures.

### A.4 Conflicts resolved

| Topic | Decision |
| --- | --- |
| V004 writer hardcodes 235 in audit | NHC0801 auditor is parameterized; pilot counts only via explicit expected args or evidence JSON fields |
| Production B3LYP/SVP frames | Not accepted as parent teacher truth |
| D3 recompute | Forbidden in reader; `d3_recomputation_performed` must be false |
| Final Test | Commitment only; reject `final_test`/`test` keys and dirs |
