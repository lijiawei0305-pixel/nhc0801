# 环境与命令速查（在 HPC 上）

> 从 `CLAUDE.md §5` 拆出。根级索引见 `CLAUDE.md`；部署硬约束见 `CLAUDE.md §部署硬约束` 与 `doc/detailed-design.md §0.9`。

```bash
# 激活环境（绝不 source ~/.bashrc）
source $WJW/env/envs/molenv.sh     # 分子栈
source $WJW/env/envs/xtb.sh        # 单软件栈（vasp.sh/cp2k.sh/multiwfn.sh 同理，会先 unset LD_LIBRARY_PATH）
# 入口
nhc [--config config/fragments.yaml] [--asymmetric] [--no-pubchem]   # 枚举
mq                                  # 交互菜单（SSH 后）
mq selfcheck env                    # 11 项环境自检
# 本地（macOS）只做：编辑 + rsync 同步 + pytest/ruff/mypy；不要装/跑计算
```

## 分子链批量运行机制（`mq` 选项2 = 顺序链，非并行 worker）

`mq` 选项2「批量队列」实际是顺序执行 M2→M3→M4→M5→M6（`scripts/mq:_run_steps`，逐步全跑），**不是** `doc/PRODUCT.md §2.3.2` 画的多 worker 并行界面。各步脚本与所需栈：

| 步 | 脚本 | 需 source 的栈 | 关键产出 |
| --- | --- | --- | --- |
| M2 | `scripts/mol/gen_3d.py` | `molenv.sh` | `<cand>/xyz/<key>_{cation,neutral}.xyz` + `_atom_map.json` |
| M3 | `scripts/m03_batch_runner.py` | `molenv.sh` + `xtb.sh` | `xtb_screen.csv`（含 `pass_filter`） |
| M4 | `scripts/mol/dft_batch.py` | `molenv.sh` | `dft_mol.csv` + `runs/mol/<key>/`（molden 等） |
| M5 | `scripts/mol/run_fukui.py` | `molenv.sh` + `multiwfn.sh` | `fukui_summary.csv` |
| M6 | `scripts/mol/mol_assembly.py` | `molenv.sh` | `results/mol_properties.parquet` + 报表 |

**硬性约定（踩过坑，见 skill: xtb-homo-threshold-not-dft-scale）**：
- **一次一栈（C5）**：用子shell逐步 source，别一把全 source；`xtb.sh` 会设 `OMP_NUM_THREADS=8`。
- **Multiwfn**：noGUI 构建，可执行名 **`Multiwfn_noGUI`**，经 `multiwfn.sh` 导出的 `$MULTIWFN_BIN` 调用；**没有名为 `Multiwfn` 的命令**（`command -v Multiwfn` 必失败，不代表缺失）。
- **import 根**：`export PYTHONPATH=$WJW` 且 `cd $WJW`。`scripts/mol/*` 用 `from scripts.mol…`（需 `$WJW` 在 path；`dft_batch`/`run_fukui` 另自插 root 兜底）；`m03_batch_runner.py` 用同级 `import m03_*`（靠"运行脚本时脚本目录自动入 `sys.path`"）。
- **控并行**：`mq` 调 M3/M4 **不传 `--parallel`** → 默认 auto=`cpu//threads_per_job` 填满整机（多核）。共享节点（与 ZJH 等同跑）要留余量须**直接跑底层脚本**加 `--parallel N --threads-per-job 8`（如 12×8=96 核）。
- **预筛参数（已外置）**：M3 的计算设置 + 预筛阈值集中在 **`config/xtb/xtb_config.yaml`**（`calculation:` + `prescreen:` 两段，代码会读）。`mq` 分子链 M3 步**已显式传 `--config` 指向该文件**，`m03_batch_runner.py` 无 `--config` 时也默认加载它——**改 yaml 即生效，无需动代码**。注意它**不是** `config/thresholds.yaml`（那是 M6/M10 终筛的 DFT 尺度 `e_homo.*`/`delta_delta_g.*`，与预筛键名不同、互不影响）。兼容入口 `--thresholds <扁平 yaml>` 仍可单独覆盖 `prescreen:` 段。
  - **现行标准（v3，2026-06-19，两段式 two-band）**：`pass = gap_min_ev≥1.0 AND (delta_e_deprot≤max_deprot_energy_kcal=80 OR homo≥donor_rescue_homo_ev=−9.1)` → 5,795/15,130 进 DFT。精度标签 GFN2/crude/气相/skip-hessian（标准见 `detailed-design §5.3.1.2a`）。
  - 绝对 HOMO 窗口 `homo_min_ev/homo_max_ev` 是 DFT 尺度、会误杀 GFN2-xTB HOMO（~−9 eV），故默认 `null` 关闭（skill: xtb-homo-threshold-not-dft-scale）；HOMO 在 xTB 段只做"σ-给体救援"。相对能/虚频过滤同样默认 `null` 关闭。

底层等价命令（隔离工作目录 `$W`、控并行 + 后台示例）：

M3 的 `--reference-energy` 必须来自 `config/fragments.yaml` 中 `reference_cation` 对应阳离子的 xTB 总能量；`scripts/run_full_pipeline.sh` 会自动查找该参考分子并计算。手动运行时不要使用 `0.0` 或任意占位值，否则 `delta_e_kcal` 的相对能量口径会错误。

```bash
W=$WJW/data/runs/<run>; CAND=$W/candidates; export PYTHONPATH=$WJW; cd $WJW
( source env/envs/molenv.sh && python scripts/mol/gen_3d.py --input <cands.csv> --output $CAND )
( source env/envs/molenv.sh && source env/envs/xtb.sh && python scripts/m03_batch_runner.py \
    --input $CAND/xyz --output $CAND/xtb_screen.csv --run-dir $W/runs/xtb \
    --reference-energy <reference_cation_energy_Eh> --skip-hessian --parallel 12 --threads-per-job 8 [--thresholds <yaml>] )
( source env/envs/molenv.sh && python scripts/mol/dft_batch.py \
    --screen-csv $CAND/xtb_screen.csv --xyz-dir $CAND/xyz --runs-dir $W/runs/mol \
    --summary-csv $CAND/dft_mol.csv --maxsteps 100 --parallel 12 --threads-per-job 8 )
( source env/envs/molenv.sh && source env/envs/multiwfn.sh && python scripts/mol/run_fukui.py \
    --dft-csv $CAND/dft_mol.csv --data-root $W --output $CAND/fukui_summary.csv --parallel 8 )
( source env/envs/molenv.sh && python scripts/mol/mol_assembly.py --config config/thresholds.yaml \
    --dft $CAND/dft_mol.csv --fukui $CAND/fukui_summary.csv \
    --output results/<out>.parquet --report results/<out>.txt )
# 后台多天任务：setsid bash run.sh > $W/run.log 2>&1 < /dev/null &
```
