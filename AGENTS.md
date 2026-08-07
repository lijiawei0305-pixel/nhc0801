# AGENTS.md — NHC0801 / nhc-deprot

AIMNet2 fine-tune on **Parent-Level P01** teacher frames.  
Local `/Users/cc/nhc-deprot` · write **only** `$WJW/NHC0801`. Not ranker Phase 9B / not `two_endpoint`.

**L0 only** (every session). Detail → `docs/agent/*` on demand. No day-to-day GPU logs here（use `PHASE_STATUS.md`）.  
After editing L0/L1 or rsync: **`docs/agent/REGRESSION_CHECKLIST.md`**.

**Conflict:** `mindmap.md` → `docs/contracts/COMPUTE_DISPATCH_V001.md` → other contracts → this file → `RETRO.md` → pilot evidence.

---

## Workflow

1. Read `mindmap.md` + this file + `PHASE_STATUS.md`.
2. Open matching **Read when** row; skim `RETRO.md` for same class of failure.
3. Map to mindmap step 0–12（`src/nhc_deprot/mindmap_steps.py`）. Non-trivial → `docs/plans/` first.
4. Code placement / product names → `docs/agent/naming.md`. After edits: `PYTHONPATH=src python -m pytest -q`.
5. Live chem / train / Final Test need **explicit user auth**. Obsolete params → drop queues（`docs/agent/compute_priority.md`）.

| Task touches… | Read first |
| --- | --- |
| paths / `g00N` / train vs publish / code tree | **`docs/agent/naming.md`** |
| finetune / loss / trainable / teacher frames / pre-screen | **`docs/agent/training_t1_t9.md`**（T1–T9；**勿重推**） |
| waves / teacher·e0 / priority / abandon queues / **Train 合并 g101+** | **`docs/agent/compute_priority.md`** + **`COMPUTE_DISPATCH_V001.md`**（§1.4） |
| cores / memory | `RESOURCE_SCHEDULING_V001.md` + `RESOURCE_PROFILES_V002.yaml` |
| selection numbers | `NUMERIC_CALIBRATION_V001.yaml` |
| GAU stop / maxsteps | **`GAU_LOOSE_V001.yaml` only**（no second contract） |

**Docs:** plans before multi-step work · contracts on freeze/bump（science needs user OK）· long rules in `docs/agent/` · L0 = boundaries + commands + index only. Version contracts `_V001`；never silent semantic change.

**Layout one-liner:** `src/` · `tests/` · `scripts/` thin · `docs/{plans,contracts,agent,science}` · products under `runs/nhc0801-g001/` as `teacher_gpu_g00N/` · `epoch0_val_batches/g00N/` · `train_g00N/runs/<run_id>/` · `models/v0.N/model.pt` — never `best.pt` as release；never product names `teacher/` / `side` / bare `epoch0/` /「Autofill 批」.

---

## Science（non-negotiable）

- `NHC-H+ → NHC + H+`；cation (+1,s) / neutral (0,s)；root 不跨 split.
- **Parent P01 only:** `wb97m-d3bj` / `def2-TZVPP`，grid=4，SCF 1e-9；SHA256 `227c22a527e567bc4de873ab743fe9f493779eccbb1a698d2913c87695ebf87a`.
- **GAU_LOOSE:** 五准则 + fmax **0.10** eV/Å；maxsteps **250**（only `GAU_LOOSE_V001.yaml`）. Not fmax 0.05.
- Route: freeze → AIMNet2 GAU_LOOSE → gates → exact-byte handoff → **full** parent GAU → SP → label；`single_point_only=false`；**AIMNet2 energy never labels**；frozen D3 residual only.
- Quick-val / frame **energy loss** must not final-select（T1）. Steps 0–12: `mindmap.md`. Preflight: `PYTHONPATH=src python -m nhc_deprot.pipeline.mindmap_orchestrator`.
- Finetune（full T1–T9 in L1）: energy never ranks；train `atomic_shift`；forces weight from measured E/F；teacher trajectory `callback`；no gain vs e0 → data shortage.
- **Train/Val 合并（协议）**：Train∪Train→Train；**Val∪Val→Val 评估池 only**；**Val 永不并进 Train**；FT 不入并。合并可接位仍叫 g001（旧 g001 须归档）。权威：`COMPUTE_DISPATCH_V001` §1.4。

| Don't | Do |
| --- | --- |
| B3LYP `two_endpoint` / fmax 0.05 parent stop | P01 + GAU_LOOSE 0.10 |
| Best-by-quick-val | sci-val + `NUMERIC_CALIBRATION_V001` |
| Open Final Test for convenience | sealed until explicit auth |
| Write outside `$WJW/NHC0801` | NHC0801 only |
| **Val 根并进 Train / 当训练标签** | **Val 只进 Val 池；Train 只并 Train** |

---

## Gates · Always / Ask / Never

Auth（`docs/prompt.md`，2026-08-03 automation）:

```text
OPEN : teacher_pyscf_authorized  epoch0_execution  aimnet2_train_authorized
       scientific_validation_live  rsync→NHC0801（no --delete）
CLOSED: final_test_open  modify_wjw_outside_NHC0801  scheduler_submission
```

| | |
| --- | --- |
| **Always** | Write only `$WJW/NHC0801`. Val e0 = **2 roots × cation/neutral 分开算、4 卡并行**（`nhc0801_e0_val_4gpu.py` / `--endpoint`）. GPU claim before train. `pytest` after code. Wave = 10 endpoints；teacher via `nhc0801_gpu_teacher_daemon.py`；train CUDA-only on g001 Train3 labels. |
| **Ask** | CPU+GPU concurrent waves；kill/interrupt any job；bump science contracts / GAU maxsteps；open Final Test；rsync with destructive flags. |
| **Never** | Interfere with `gpu_teacher_daemon` / live `e0_val_only` / `compute_steward` or others' GPUs. Rewrite `frame_count==2` teacher endpoints. Self-open Final Test. Silent change mindmap / P01 SHA / GAU criteria / TVT. **Resume queues built under obsolete params**（→ `compute_priority.md`）. Edit ranker / science-pilot trees. |

**P0 now:** g001 Val e0 baseline（KZYK+RMEQ，cation/neutral 四路并行，maxsteps 250）before sci-val. **P1:** teacher. **P2:** e0 expand **paused**. Detail → `docs/agent/compute_priority.md`.

---

## Server · style

| | |
| --- | --- |
| SSH | `configs/server.local.yaml`（never commit） |
| Root | `/home/plab/test/WJW/NHC0801` |
| ML / PySCF | `source $WJW/env/envs/mlff.sh` · `molenv.sh`（no mix） |
| e0 weight | `aimnet2_wb97m_d3_0.pt` only |

```bash
rsync -avz --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' \
  -e 'ssh -o BatchMode=yes' /Users/cc/nhc-deprot/ nhc614:/home/plab/test/WJW/NHC0801/
```

- One science knob per change；synthetic fixtures；no `git reset/clean` on science-pilot.  
- Footgun → append `RETRO.md`；promoted rules mark **「已升级为规则」** + path. RETRO ≠ second mindmap.  
- **Next:** T5 full-trajectory frames → pre-screen step 7. Live: finish g001 e0 baseline，then sci-val；do not resume obsolete expand queues.
