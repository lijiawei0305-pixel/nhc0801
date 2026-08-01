# NHC614 计算参数总表

> **这是什么**：本项目所有计算环节的权威参数口径速查（给人看）。由 `CLAUDE.md` 经 `@doc/claude/calc-params.md` 每会话自动加载。
> **维护铁律**：任一计算环节切换泛函/基组/赝势/截断/收敛/几何口径，**YOU MUST 同步改本表**（配工作纪律 9 文献支撑）。**色散全项目统一 D3(BJ)**。细节见 `detailed-design`（附录 A）/`science-decisions`。

## Part 1 · 分子：C2 脱质子能力（ΔE_deprot）

| 阶段 | 软件/方法 | 泛函 / 基组 | 关键参数 |
| --- | --- | --- | --- |
| 低精度预筛 | xtb GFN2-xTB | 半经验 | 气相、`opt crude`、`--skip-hessian`、timeout 120s |
| 高精度 opt+freq | PySCF+geomeTRIC | **B3LYP-D3(BJ) / def2-SVP** | 气相、298.15K 热化学、RKS |
| 高精度 单点+性质 | PySCF | **ωB97X-D / def2-TZVP** | HOMO/LUMO/Molden；**开密度拟合**（auxbasis `def2-tzvp-jkfit`）、grids.level 4 |
| **MLFF 训练标签**（2026-07-14 建 🔵） | **GPU4PySCF**（主）/ PySCF（CPU 备）**单点 + 解析梯度** | **B3LYP-D3(BJ) / def2-SVP**（**与 opt 级同口径**） | grids.level 3、`mf.disp='d3bj'`（E 与 grad 都含 D3）、**不开密度拟合**、conv_tol 1e-9、max_memory 8000MB；**DM 接力**（`dm0=dm_prev`）；输出 (E, F) → AIMNet2 HDF5（Å/eV/eV·Å⁻¹）<br>⚡ **GPU 后端与 CPU 逐位一致**（2026-07-14 实测 ΔE = **0.00 μHa**、Δ力 4e-4 eV/Å）→ **口径零变更**，只是快 ~12×（8 卡并发实测）。⚠️ 必须钉死 `CUDA_VISIBLE_DEVICES` 单卡、**绝不调 `.density_fit()`**（DF 会引入 ~2 meV 偏差，而且不需要）——见 skill `gpu4pyscf-multigpu-kernel-fail-pin-single-device` |
| **MLFF 模型**（2026-07-14 建 🔵） | **AIMNet2** 微调（`aimnetcentral`, MIT） | 底座 ωB97M-D3/def2-TZVPP → **微调到 B3LYP-D3(BJ)/def2-SVP** | 化学域=imidazolium **阳离子(+1,单) + 中性卡宾(0,单)**，**不含开壳层**；训练前必跑 `calc_sae`（换理论级别重拟原子参考能）；**从 MTLoss 移除 charges 分量**（AIMNet2 的 charges 是 NQE 学出的，非 Mulliken/Hirshfeld，我们只要 PES） |
| 电荷/自旋口径 | — | — | cation(+1,单,RKS) / neutral(0,单,RKS) / radical(+1,doublet,UKS) |

## Part 2 · 周期表面：NHC–Cu(111) 结合能（E_bind）

| 精度层 | 软件/方法 | 泛函 / 基组 | 关键参数 |
| --- | --- | --- | --- |
| **低精度**（66 标签 ✅） | CP2K GFN1-xTB (GPW) | 半经验 | 5×5×4 slab、底 2 层固定、C2-down top；SMEAR Fermi 2000K、Broyden α0.03/NBUFFER5、EPS_SCF 1e-3、CHECK_ATOMIC_CHARGES off；min-收敛收割 |
| **中精度**（团簇代理 🔄，2026-07 建） | CP2K DFT (GPW) | **PBE-D3(BJ)** / DZVP-MOLOPT(-SR)-GTH + GTH-PBE | 平面 Cu₁₀ top（切自 xTB 几何）、vertical；CUTOFF 600 / REL_CUTOFF 100、POISSON NONE+WAVELET；金属 SMEAR Fermi 1000K+Broyden α0.1+对角化(EPS 1e-4/MAX 200)；分子 OT DIIS(EPS 1e-5/MAX 50)；⚠️ 待补 counterpoise BSSE + CUTOFF 收敛测试 |
| **高精度**（VASP 锚点） | VASP | **PBE-D3(BJ)**（IVDW=12） | ENCUT 520、ISMEAR=1/SIGMA=0.1、p(4×4)/4–5 层底 2 固定、k≈0.03 Å⁻¹(5×5×1)、真空≥15Å+偶极；三体系同胞同 k 同 ENCUT；锚点 EDIFFG=−0.03~−0.05 |

> **多保真桥**：团簇 PBE-D3(BJ) ↔ VASP PBE-D3(BJ)，同泛函同色散 → Δ-learning 残差干净。团簇代理 2026-07-10 从零阻尼 D3 改齐 D3(BJ)。
