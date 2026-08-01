# ARCHIVE — forbidden parent stack (do not use for NHC0801 calculations)

These documents describe the **production two_endpoint** science stack:

- AIMNet2 preopt stop ≈ **fmax 0.05 eV/Å**
- Parent DFT ≈ **B3LYP-D3(BJ) / def2-SVP**

That stack is **banned** as parent / training truth for mindmap NHC0801.

## Use instead

| Role | NHC0801 authority |
| --- | --- |
| Parent DFT | Parent-Level **P01**: ωB97M-D3(BJ)/def2-TZVPP (`contracts/parent_protocol.py`) |
| AIMNet2 stop | **GAU_LOOSE** 5-criteria + ASE fmax **0.10** / 100 steps |
| Labels | Parent electronic energy only; AIMNet2 energy never enters labels |
| Model selection | Full scientific Validation (not quick-val frame loss) |

## Why kept (not deleted)

Historical 71-label / ranker context for forensics only. Agents must **not** copy
protocols from this tree into new runners. Prefer deleting references over importing.

## Deleting ranker production code

Do **not** `rm` files under `/Users/cc/nhc-deprot-ranker` or `$WJW` production from
this project. Isolation is: never import, never rsync, never set as parent default.
