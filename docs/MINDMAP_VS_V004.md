# mindmap.md ↔ V004 alignment

## Pipeline identity (aligned)

```text
frozen initial geometry
  -> AIMNet2 ASE LBFGS to AIMNet2 GAU_LOOSE
  -> identity / topology / collision gates
  -> exact-byte handoff
  -> full Parent-Level P01 PySCF/geomeTRIC to final GAU
  -> parent final single point
  -> deprotonation electronic-energy label
```

`single_point_only` is always false. AIMNet2 energy never enters the label.

## Step-by-step

| Mindmap | V004 status | NHC0801 code status |
| ---: | --- | --- |
| 0 Freeze roots | 5 dev roots + sealed FT | paths + split JSON extract |
| 1 Split by root | day1 split frozen | `tvt_gates` |
| 2 Teacher frames | 235 P01 frames on server | parameterized reader ready (`data/*`); generator not ported |
| 3 Epoch-0 baseline | writer static; NOT_RUN | GAU_LOOSE + handoff ready |
| 4 Train | adapter simulated; NOT_RUN | `weighted_loss` + adapter |
| 5 Checkpoints | config frozen | **missing loop** |
| 6 Quick val | loss defined | accumulator ready |
| 7 Shortlist | contract ready | `quick_checkpoint_shortlist` |
| 8 Full scientific val | **not implemented** | **gap** |
| 9 Select on Val | needs numeric addendum | selection gates ready |
| 10 Freeze | source commit not frozen | readiness gates |
| 11 Final Test | sealed | commitment only |
| 12 No post-Test selection | policy | gates |

## Parent protocol (must not drift)

```text
wb97m-d3bj / def2-TZVPP / grid 4 / SCF 1e-9
D3(BJ) two-body, ATM=false, VV10=false
protocol SHA256 = 227c22a527e567bc4de873ab743fe9f493779eccbb1a698d2913c87695ebf87a
```

## GAU_LOOSE (must not rename to VASP 0.1)

```text
E: 1e-6 Eh
GRMS: 1.7e-3 Eh/Bohr
Gmax: 2.5e-3 Eh/Bohr
disp RMS: 6.7e-3 Å
disp max: 1.0e-2 Å
ASE fmax: 0.10 eV/Å
max_steps: 100
require all five
```

Handoff PASS and MISS both continue full parent optimization.
