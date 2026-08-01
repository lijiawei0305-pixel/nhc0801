# Phase 9B V004 complete handoff for Grok

Date: 2026-08-01

Purpose: transfer the current non-production AIMNet2 Train–Validation–Test
science-pilot work to Grok without losing scientific meaning, split isolation,
resource safety, or the dirty-worktree state.

This cold-start guide does not replace `AGENT.md`, `PHASE_STATUS.md`, frozen
contracts, or the exact JSON/YAML evidence cited below. If a summary conflicts
with primary evidence, stop and follow the higher-ranked primary evidence.

## 1. Mandatory cold-start order

Start with:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -1 --format=fuller
```

Then read, in order:

1. `AGENT.md`
2. `PHASE_STATUS.md`
3. `.codex/skills/plan-nhc-aimnet2-workflow/SKILL.md`
4. `.codex/skills/plan-nhc-aimnet2-workflow/references/evidence-routing.md`
5. `.codex/skills/plan-nhc-aimnet2-workflow/references/workflow-contract.md`
6. `.codex/skills/plan-nhc-aimnet2-workflow/references/tvt-execution-contract.md`
7. `.codex/skills/plan-nhc-aimnet2-workflow/references/server-performance-contract.md`
8. `docs/PHASE9B_AIMNET2_MODEL_GENERATION_CONFIG_V004.json`
9. Only the primary artifacts referenced by the applicable V004 section.

Never choose authority by mtime or whichever filename looks newest. Follow
explicit path/SHA256 bindings.

## 2. Git and worktree identity

```text
repository: lijiawei0305-pixel/nhc-deprot-ranker
worktree: science-pilot worktree
branch: agent/phase9b-science-pilot
HEAD: 2e19d2704eca76ef04de13b7a300adf0a6accaef
HEAD subject: Fix GAU_LOOSE parent handoff calibration
worktree: DIRTY
```

The active V004 work is not fully represented by HEAD. The worktree contains
modified skill/contracts/status documents and many untracked V003/V004 source,
test, JSON, YAML, and report files. They are intentional current work.

Do not run `git reset`, `git checkout --`, `git clean`, or `git stash`. Do not
delete or overwrite untracked V004 artifacts. Do not claim an untracked file is
part of HEAD. The old `docs/HANDOFF_PHASE9B_FOR_NEXT_AGENT.md` is historical
control-plane context, not the active V004 next-action guide.

## 3. Current high-level state

```text
active generation: phase9b-aimnet2-nhc-p01-tvt-20260801-v001
generation schema: phase9b-aimnet2-model-generation-config-v004
state: BLOCKED_BEFORE_TRAINING
science_pilot_only: true
production_accepted: false
production_label_inserted: false
single_point_only_eligible: false
training_attempt_consumed: false
retry: false
```

No real V004 training has started. No V004 epoch-0 AIMNet2/PySCF Validation
route has run. No full scientific checkpoint Validation has run. Final Test has
not been read or consumed. Production gates and 71 production labels remain
unchanged.

V004 currently reports these hard blockers:

```text
SOURCE_COMMIT_NOT_FROZEN
EPOCH_ZERO_FULL_ROUTE_BASELINE_NOT_AVAILABLE
NUMERIC_CALIBRATION_RULE_NOT_PREREGISTERED
FULL_SCIENTIFIC_VALIDATION_WRITER_NOT_IMPLEMENTED
```

Dual-worker calibration is an additional performance-readiness branch. It does
not remove those scientific blockers and cannot authorize training.

## 4. Non-negotiable scientific meaning

### 4.1 Model role

AIMNet2 is only a geometry preconditioner:

```text
frozen initial geometry
-> AIMNet2 / ASE LBFGS to AIMNet2 GAU_LOOSE
-> identity, topology, collision, and finite-coordinate gates
-> exact-byte handoff
-> complete Parent-Level P01 PySCF/geomeTRIC optimization to final GAU
-> parent-level final single point
-> deprotonation electronic-energy label
```

`single_point_only` is always `false`. AIMNet2 convergence does not mean
parent-level convergence and never permits skipping parent optimization.

### 4.2 AIMNet2 GAU_LOOSE

This is one indivisible profile on the AIMNet2 surface:

```yaml
energy_change_Eh: 1.0e-6
gradient_rms_Eh_Bohr: 1.7e-3
gradient_max_Eh_Bohr: 2.5e-3
displacement_rms_A: 6.7e-3
displacement_max_A: 1.0e-2
require_all: true
optimizer: ASE_LBFGS
max_steps: 100
```

The active contract also retains the reviewed AIMNet2-side force cap. Do not
rename this profile “true 0.1” and do not introduce VASP terminology.

### 4.3 Parent handoff calibration

The first successful parent energy and analytic gradient must come from the
continuing full PySCF/geomeTRIC optimization, not a duplicate static job. It is
only `PARENT_GAU_LOOSE_GRADIENT_CHECK`, never complete parent `GAU_LOOSE`.

```text
HANDOFF_CALIBRATION_PASS:
  GRMS <= 1.7e-3 Eh/Bohr AND Gmax <= 2.5e-3 Eh/Bohr

HANDOFF_CALIBRATION_MISS:
  valid SCF/gradient/identity/topology, but either gradient gate misses

FAILED_PARENT_HANDOFF:
  SCF failure, unavailable/nonfinite gradient, invalid geometry or identity,
  charge/multiplicity error, fragmentation, or unexpected bonding
```

Both PASS and MISS continue the same full parent optimization to
`FINAL_PARENT_GAU_CONVERGED`. Only `FAILED_PARENT_HANDOFF` stops.

### 4.4 Parent protocol and label

```text
gas phase, closed-shell RKS
functional = wb97m-d3bj
basis = def2-TZVPP
grid = 4
SCF conv_tol = 1e-9
VV10 = disabled
dispersion = explicit two-body D3(BJ)
ATM = false
protocol SHA256 = 227c22a527e567bc4de873ab743fe9f493779eccbb1a698d2913c87695ebf87a
```

```text
electronic_difference_kcal =
    (E_neutral_parent - E_cation_parent) * 627.509474
dft_deprot_electronic_kcal = electronic_difference_kcal - 6.28
lower_is_better = true
```

AIMNet2 energy never enters the label. This is a gas-phase electronic-energy
result, not Gibbs free energy, pKa, solution acidity, experimental enthalpy, or
coupled-cluster truth.

## 5. Immutable science-pilot history

| Attempt | Terminal | Essential result |
| --- | --- | --- |
| v001 | `INCONCLUSIVE` | validator wrongly assumed proton host N1/N3; validator bug, not science failure |
| v002 | frozen-gate `FAIL` | neutral N1–C2–N3 changed 12.366075 degrees; historical 10-degree gate unchanged |
| v004 | `PASS` | corrected signed-dihedral convention; `SAME_BASIN_LIKELY`; two B3LYP-D3(BJ)/def2-SVP single points converged; label `238.8477388721244 kcal/mol` |
| v005 | `PASS` | frozen-initial single-point control; direct label `265.5095713697973 kcal/mol`; assisted-minus-direct `-26.66183249767289 kcal/mol`; not a geometry-optimization speedup |
| v006 | `PARTIAL_PASS` | assisted route `235.90 s`; pure-PySCF neutral timed out at 7190 s/Step 17; speedup only `>30.479271x` lower bound for one candidate |

The anchor was `LBNPGYISTSLAHY-UHFFFAOYSA-N`. Nothing was inserted into
production.

## 6. Parent-Level P01 history

R1 discovered 112 logical CPUs and 56 physical cores on a shared node. A
54-thread SMT calibration was 5.28% slower than 27 physical threads. The safe
profile became physical CPUs `0,2-27`, 27 PySCF threads, SMT disabled, and
64,000 MB PySCF max memory.

Grid 4 converged from the bound grid-3 density:

```text
energy = -1409.4738305457154 Eh
SCF cycles = 2
grid points = 679168
gradient RMS = 0.001643055513 Eh/Bohr
gradient max = 0.006979380254 Eh/Bohr
D3 = -0.04286372842069901 Eh
```

R1 Group A stopped before its first AIMNet2 frame because NVRTC rejected an
overlong temporary path. R2 proved short-cache propagation but its smoke helper
failed before binding elements. R3 corrected the helper and completed one real
CUDA energy/force evaluation, then formal Group A stopped before model load
because the offline verifier compared the short cache against the long attempt
root. The verifier was statically corrected afterward.

There is no completed P01-R4 result in this worktree. Do not fabricate one or
infer Group A/Group B completion. Later V004 TVT priority superseded automatic
continuation; reviving P01 requires separate authority.

## 7. V004 split and Final Test isolation

The indivisible split unit is `molecular_root`: both endpoints, conformers,
trajectories, restarts, augmentations, AIMNet2/PySCF descendants, and
serialization variants remain together.

Development-visible Train roots:

| Root | Electrons | Cation/neutral atoms |
| --- | ---: | --- |
| `ACGCNTKELWXJPN-UHFFFAOYSA-N` | 72 | 19 / 18 |
| `PDIYCCLDBKWBTK-UHFFFAOYSA-N` | 100 | 29 / 28 |
| `VNYHGZAUUQMMDL-UHFFFAOYSA-N` | 68 | 16 / 15 |

Development-visible Validation roots:

| Root | Electrons | Cation/neutral atoms |
| --- | ---: | --- |
| `KZYKDQNIIMATMJ-UHFFFAOYSA-N` | 100 | 29 / 28 |
| `RMEQTBVGGNKAEQ-UHFFFAOYSA-N` | 76 | 23 / 22 |

Two incomplete Train roots are excluded without deletion, replacement, retry,
or split movement: `CLXFIGGGSODORK-UHFFFAOYSA-N` and
`RBKFFSUUCLDQER-UHFFFAOYSA-N`. No partial frames are admitted.

Final Test is opaque to training:

```text
sealed commitment SHA256:
  834f973954064565aa857e8d8c563d110d0f6256c99e54fc3283dc428efa6975
sealed root count: 2
identities exposed to training: false
```

Do not open the full split registry or Final Test payload to improve this
handoff. Final Test requires an independent unopened receipt after model,
threshold, code, runtime, and protocol freeze. It is one-time and no-retry.

## 8. Dataset and trainer completion matrix

| Component | Current state |
| --- | --- |
| root identity | PASS for five development roots |
| frame admission | 235/235 PASS |
| two-body D3(BJ) projection | 235/235 PASS; generated once |
| weighted dataset | PASS: 123 Train + 112 Validation frames |
| weighting | equal candidate, equal cation/neutral endpoint, uniform frames within endpoint |
| stored weight key | `sample_weight`; each split sums to 1.0 |
| historical V002 trainer | immutable and ineligible; did not consume weights |
| V004 weighted loss | implemented and tested |
| training estimator | `N/B * sum(w_i * per_sample_loss_i)` |
| quick Validation reduction | global weighted sum / global weight sum |
| V004 training adapter | loader/training/quick-Validation simulation PASS |
| real model training | `NOT_RUN` |

Short-range training targets are:

```text
E_short = E_parent_total - E_D3_two_body_BJ
F_short = F_parent_total - F_D3_two_body_BJ
```

Restore the identical external D3 definition at inference. A writer rerun must
consume frozen D3 receipts, not silently recompute D3.

## 9. Frozen proposed training configuration

```yaml
seeds: [20260730, 20260731, 20260732]
epochs: 200
optimizer: torch.optim.RAdam
learning_rate: 1.0e-4
weight_decay: 1.0e-8
batch_size: 32
batch_mode: molecules
batches_per_epoch: -1
gradient_clip_value: 0.4
trainable_parameter_regex: ['^outputs\.energy_mlp\.']
loss:
  energy_weight: 1.0
  forces_weight: 1.0
  energy_normalization: sqrt_atom_count
  force_normalization: per_atom
scheduler:
  type: ReduceLROnPlateau
  factor: 0.5
  patience_epochs: 15
  minimum_learning_rate: 1.0e-7
checkpoint_interval_epochs: 10
quick_validation_each_epoch: true
```

Retain all seeds, checkpoints, failures, and stop reasons. Quick stored-frame
Validation creates only a deterministic shortlist; it cannot select the final
model.

## 10. Epoch-0 and scientific Validation

The epoch-0 full-route writer exists and its static audit passed. It projects
only the two Validation roots/four endpoints and requires:

```text
official unchanged AIMNet2
-> AIMNet2 GAU_LOOSE
-> exact-byte handoff
-> first parent gradient classification
-> complete parent optimization to GAU
-> final parent single point
-> pure-parent comparison
```

The epoch-0 execution resource config is frozen, but execution is `NOT_RUN`.
Epoch 0 must close before the single training attempt is consumed.

The full scientific Validation writer for shortlisted checkpoints is still not
implemented. The deterministic numeric-calibration procedure is also not
preregistered. Do not invent label/bias tolerances, catastrophic failure,
allowed Validation failures, epoch-0 non-regression, or minimum burden
reduction after seeing Validation. Freeze the procedure before training;
Validation may instantiate values later; Final Test may not.

## 11. Resource profile and dual-worker calibration

Official profile:

```text
single_27_physical_v1
CPU affinity 0,2-27
27 physical cores / 27 threads / no SMT
PySCF max_memory 64000 MB
root concurrency 1 / endpoint concurrency 1
retry false / fallback false
```

The shared node's 112 logical CPUs are not automatically owned or faster.

The preregistered `ISOLATED_BENCHMARK` compares single-27 with
`dual_14_13_physical_v1`. Both use aggregate 27 physical cores and 64,000 MB.
Dual lanes are `0,2-14` and `15-27`, physically disjoint and NUMA-local. The
workload is two frozen Validation-cation parent energy-plus-analytic-gradient
tasks, not geometry optimization, AIMNet2, training, frame admission, or label
generation.

The order is ABBA with two repetitions/profile. Dual is selected only when all
four tasks/profile are accepted, numerical identity passes, and accepted-task
throughput improves by at least 5%. Otherwise retain single. A selection
receipt is mandatory, and calibration never auto-starts chemistry.

Durable claims:

```text
V001: LIVE_RESOURCE_CLAIM_REJECTED
  selected CPUs 27/27 busy
  memory PSI avg10 8.42% (limit 1%)
  disk free about 49.5 GB
  two-sample closure unavailable

V002 after user disk cleanup: LIVE_RESOURCE_CLAIM_REJECTED
  selected CPUs 27/27 busy in both samples
  memory PSI avg10 0.00%
  I/O PSI avg10 0.00%
  MemAvailable >= 234,967,650,304 bytes
  disk free >= 172,664,582,144 bytes
  two-sample closure PASS
  cgroup memory limit unavailable
  free inode observation unavailable
```

V002 is the latest durable claim. Calibration is `NOT_RUN`, no selection
receipt exists, and single remains official.

An explicitly requested later read-only watcher observed at least 203 samples
from 2026-08-01T13:04:44Z through 2026-08-01T14:49:58Z. The compute process
remained alive and the bundle never had two consecutive fully released
samples. Memory PSI stayed 0 and disk headroom stayed near 247 GiB. This was
diagnostic process state, not a durable third claim. The local SSH watcher was
terminated during this handoff without signalling remote compute.

Future claim collector fixes:

1. Resolve the cgroup v2 scope from `/proc/self/cgroup` and read controller
   files under the effective scope/parent, not only `/sys/fs/cgroup` root.
2. Read free inodes with standalone `df -Pi`; do not mix mutually exclusive
   `df -i` and `--output` options.

Never overwrite V001/V002. Create append-only V003 only after the selected
bundle clears and every frozen gate can be measured twice.

## 12. Ordered next work

Each mutation or live execution needs applicable separate authority.

1. If calibration remains desired, wait for `0,2-27` to clear, obtain a new
   corrected read-only two-sample claim, then fail closed if any gate misses.
2. With a passing claim and execution-capable authority, execute the frozen
   isolated calibration once and create a profile-selection receipt.
3. Preregister and statically audit the Validation-only numeric-calibration
   procedure before training.
4. Implement and audit the full scientific Validation writer/reader.
5. Freeze source/resource/output identities and run epoch-0 only under an
   execution-capable authority.
6. When all readiness gates pass, consume the one three-seed training attempt.
7. Quick Validation shortlists; full scientific Validation selects and freezes
   the checkpoint/numeric addendum.
8. Freeze checkpoint, data/splits, code, runtime, and protocol; obtain an
   unopened receipt; consume Final Test once in a separate process.

Even an accepted Final Test leaves `single_point_only_eligible=false`.

## 13. Safe local validation commands

```bash
python -m json.tool \
  docs/PHASE9B_AIMNET2_MODEL_GENERATION_CONFIG_V004.json >/dev/null
python -m json.tool \
  docs/PHASE9B_AIMNET2_V004_DUAL_WORKER_LIVE_RESOURCE_CLAIM_V002.json >/dev/null

PYTHONPATH=src python -m pytest -q \
  tests/test_phase9b_aimnet2_v004_live_resource_claim.py \
  tests/test_phase9b_aimnet2_v004_dual_worker_calibration.py \
  tests/test_phase9b_aimnet2_v004_epoch0_resources.py \
  tests/test_phase9b_aimnet2_tvt_contract.py

ruff check \
  scripts/phase9b_aimnet2_v004_live_resource_claim.py \
  tests/test_phase9b_aimnet2_v004_live_resource_claim.py \
  tests/test_phase9b_aimnet2_tvt_contract.py

mypy --strict scripts/phase9b_aimnet2_v004_live_resource_claim.py
python -m compileall -q scripts/phase9b_aimnet2_v004_live_resource_claim.py
git diff --check
```

The most recent related run passed 30 targeted tests plus Ruff, strict mypy,
compileall, JSON validation, and `git diff --check`. Run the full suite before a
future commit. Do not weaken contracts to make tests pass.

## 14. Server and execution boundary

Private connection authority lives in ignored local configuration. Resolve it
there. Never copy SSH aliases, usernames, hosts, IPs, private absolute paths,
or credentials into tracked artifacts.

The current planning skill permits bounded read-only observation only. It does
not permit server writes, scheduler submission, process control, AIMNet2
inference/training, PySCF/geomeTRIC execution, Final Test access, or a live
permit. Execution requires a separately authorized execution-capable workflow.

## 15. Files and systems that remain untouched

- production runner and runner v9;
- guardian, permit, campaign, Postflight, and public gates;
- production table of 71 labels;
- historical v001–v006 and P01/R1/R2/R3 evidence;
- unrelated VASP `WAVECAR` and `CHGCAR`;
- incomplete/failed candidate evidence;
- sealed Final Test identities and payloads;
- old roots and consumed attempts.

No xTB, GFN0/1/2-xTB, DFTB, MMFF, UFF, retry, fallback, second candidate,
extension, batch, production permit, or production label insertion is allowed
under the current V004 work.

## 16. Primary evidence index

| Purpose | Path | SHA256 |
| --- | --- | --- |
| V004 authority | `docs/PHASE9B_AIMNET2_MODEL_GENERATION_CONFIG_V004.json` | recompute before source freeze |
| TVT workflow | `docs/PHASE9B_AIMNET2_TVT_WORKFLOW_V001.yaml` | `97d5c545d55f68219c30f5184591bda9410d1602272b424e9b0216be1afd4f3a` |
| development split | `docs/PHASE9B_AIMNET2_TVT_DAY1_DEVELOPMENT_SPLIT_V001.json` | `e3489ec29caa02695d3927e94bc47698ecf57bb189f6084fbeab6384bd04a30c` |
| root contract | `docs/PHASE9B_AIMNET2_V004_ROOT_IDENTITY_CONTRACT.yaml` | `771f5da2fa590d7f2c5c21c3b1c03dbe25a2a85025e795d7e7eed7688874d818` |
| root re-audit | `docs/PHASE9B_AIMNET2_V004_ROOT_IDENTITY_REAUDIT_RESULT.json` | `935331b0bd02d8196b60e402544251cbaffecc29561e1cb558a4b514ffea975d` |
| D3 projection | `docs/PHASE9B_AIMNET2_V004_D3_PROJECTION_RESULT.json` | `f476043f048e6a967d9d652a18e4c9d2b6f9374cb7a7d0f974f7c65d4be885e3` |
| weighted dataset | `docs/PHASE9B_AIMNET2_V004_WEIGHTED_DATASET_RESULT.json` | `3428d1159c280c0c44904f03206de3515ecaae149598c4dec1c0d22897bd3913` |
| trainer audit | `docs/PHASE9B_AIMNET2_V004_TRAINER_WEIGHT_AUDIT_RESULT.json` | `def701bf46f803290e8ee36389d045ae193fb4ce923ed6868184e6d41dd7e948` |
| training simulation | `docs/PHASE9B_AIMNET2_V004_TRAINING_WRITER_SIMULATION_RESULT.json` | `69d0d0f0864a6f29046cee6a188680b18526b8b8c34e2d6df9abb79a15d99713` |
| epoch-0 plan | `docs/PHASE9B_AIMNET2_V004_EPOCH0_VALIDATION_PLAN.json` | `42bac1358fabd67edd0bfb9b80fc074464dd44e6febe95fc62795ca0768a4721` |
| epoch-0 writer audit | `docs/PHASE9B_AIMNET2_V004_EPOCH0_WRITER_AUDIT_RESULT.json` | `501d599e7c75ff17472dd02a6b4169ec7954b6061edb2d33db16ca902d2a6b23` |
| epoch-0 resources | `docs/PHASE9B_AIMNET2_V004_EPOCH0_EXECUTION_CONFIG.json` | `a9006be56041068a0af023723245ac52cde2b45846032ab6c3707f2fefce0f34` |
| epoch-0 resource audit | `docs/PHASE9B_AIMNET2_V004_EPOCH0_RESOURCE_AUDIT_RESULT.json` | `e5726466aca2da9352ef073ac9ff091a3a3e91b6e4ea4967ec21780156e96def` |
| calibration plan | `docs/PHASE9B_AIMNET2_V004_DUAL_WORKER_CALIBRATION_PLAN.json` | `e14ee4db1f83f98e98ad9899d6192ee7279693942e1a164f445775ca10250ff1` |
| calibration audit | `docs/PHASE9B_AIMNET2_V004_DUAL_WORKER_CALIBRATION_AUDIT_RESULT.json` | `fdca2f5c1fc198d79c118cf436bb6ad05ffc2df6426557c10e6d70001d936e1a` |
| live claim V001 | `docs/PHASE9B_AIMNET2_V004_DUAL_WORKER_LIVE_RESOURCE_CLAIM.json` | `c2c265f96de038003b6b474f4bd454b29b123aa6813a361807b14eb08ce35700` |
| live claim V002 | `docs/PHASE9B_AIMNET2_V004_DUAL_WORKER_LIVE_RESOURCE_CLAIM_V002.json` | `56b59957db247f7a8edeec143a4b733dff687d5baa3685ae62af5f76c311bf64` |
| claim validator | `scripts/phase9b_aimnet2_v004_live_resource_claim.py` | `c19fd9223e17230827e4d4342a0a0f852677d427e0f2100da4c304871b2fc328` |
| Parent P01 lock | protocol identity | `227c22a527e567bc4de873ab743fe9f493779eccbb1a698d2913c87695ebf87a` |
| base AIMNet2 weight | `aimnet2_wb97m_d3_0.pt` identity | `f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28` |

Recompute hashes before freezing a source commit. A bound path is authoritative
only while its bytes match.

## 17. Known inconsistencies and cautions

1. `PHASE9B_AIMNET2_TVT_WORKFLOW_V001.yaml` still names V003 as active while
   V004 is the current successor. Do not silently rewrite it. Decide whether it
   is historical or needs a versioned V004 successor, then update bindings and
   tests honestly.
2. The initial dataset audit report records root/D3/weighting blockers. Later
   immutable re-audit/results close them. Preserve the old report and follow
   later V004 bindings.
3. Historical JSON `next_permitted_action` fields may be superseded. Use V004
   and `PHASE_STATUS.md` for current authority.
4. A prior full suite exposed one unrelated timing-sensitive Phase 8A
   supervisor test, which passed isolated and full reruns. Do not weaken it.
5. The dirty worktree means source hashes bind bytes, not necessarily commits.

## 18. Grok acceptance checklist

- Correct branch/worktree and separate HEAD from dirty bytes.
- Preserve all historical attempts and untracked V004 artifacts.
- AIMNet2 remains a preconditioner followed by full parent optimization.
- Handoff PASS and MISS both continue to final parent GAU.
- Never read Final Test identities or payloads during development/training.
- No training before epoch 0 and numeric-calibration closure.
- No dual-worker readiness claim from occupancy alone.
- No use of all 112 logical CPUs without a valid selection receipt.
- No unrelated VASP or production control-plane mutation.
- Report unavailable observations as unavailable, never zero.

## 19. Current progress handoff

```text
当前generation: phase9b-aimnet2-nhc-p01-tvt-20260801-v001
当前阶段: DUAL_WORKER_CALIBRATION_RESOURCE_CLAIM
状态: BLOCKED
完成进度: 3/13
已完成: root/frame/D3/weighting闭合，weighted trainer模拟通过，epoch-0 writer与资源配置静态冻结；两次live claim均拒绝
正在进行: not_run
阻塞点: SELECTED_CPU_BUNDLE_BUSY, CGROUP_MEMORY_LIMIT_UNAVAILABLE, FILESYSTEM_INODE_OBSERVATION_UNAVAILABLE, SOURCE_COMMIT_NOT_FROZEN, EPOCH_ZERO_FULL_ROUTE_BASELINE_NOT_AVAILABLE, NUMERIC_CALIBRATION_RULE_NOT_PREREGISTERED, FULL_SCIENTIFIC_VALIDATION_WRITER_NOT_IMPLEMENTED
唯一下一步: 等0,2-27释放后，修正cgroup/inode只读采集并取得新的双样本live resource claim
自动继续: not_authorized
下一步提示词: 使用当前skill在0,2-27自然释放后修正cgroup与inode只读采集并生成V004双worker live resource claim V003；不得启动AIMNet2、PySCF、训练或读取Final Test。
```
