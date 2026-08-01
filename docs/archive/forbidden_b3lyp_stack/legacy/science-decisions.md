# 锁定的科学/规模决策

> 从 `CLAUDE.md §3` 拆出（来自 `proposal §0.6`，已校正）。根级索引见 `CLAUDE.md`。

- C2 始终保 H（经典 C2-卡宾）；脱质子位点恒为 C2–H。
- 分子 DFT 分层：**opt+freq = B3LYP-D3(BJ)/def2-SVP；sp+性质 = ωB97X-D/def2-TZVP（PySCF+geomeTRIC）**。
- 自旋/电荷：cation `(+1, 单重态)`；neutral NHC `(0, 单重态)`；**Fukui 的 N−1 体系 = `(+1, doublet/UKS)`**（`xtb --chrg 1 --uhf 1`）。
  ⚠️ `detailed-design §0.6` 把它写成 `+2 dication` 是**错误**（中性闭壳层去 1 电子=+1 奇电子 doublet，+2 是偶电子无法 doublet，且 `gto.M(charge=2,spin=1)` 会报错）。**以 +1 为准**，详见审计 **F-C3**（需连带改 `tasks/m04、m05`）。
- ΔG_deprot **仅气相**（决策#6，勿加隐式溶剂）；绝对值用气相 H⁺ 自由能 −6.28 kcal/mol；ΔΔG 相对**1,3-二甲基咪唑鎓**（正确 SMILES `C[n+]1ccn(C)c1`）。
- 枚举首期**仅对称**（N1=N3 且 C4=C5）；阴离子对/环境因素(O₂/Cl⁻/H₂O)/导电性系统化 = 阶段 3-5 后续。
- 周期首期 **top-10** free NHC 进 Cu(111)，VASP+CP2K 双路线对比；金属 slab 参数 `ISMEAR=1,SIGMA=0.1`、`IVDW=12`、4-5 层/底 2 固定、p(3×3)/(4×4)、k≈0.03 Å⁻¹、真空≥15 Å+偶极校正、ENCUT 520。
- 排序：硬阈值 + Pareto，目标 = **ΔΔG_deprot、E_HOMO、f⁻(C2)、E_bind(NHC–Cu)**，**无主观权重**（决策#17）。
- ML代理模型（M15/M16）：目标A(脱质子化能力)用xTB全量`delta_e_deprot_kcal`(imid v4, 401,856行)做方法学验证+SHAP可解释性，非真外推；目标B(Cu结合能E_bind，M16)待15-20个VASP训练标签就绪后独立立项，放弃MLIP原子势函数路线(MACE-MP0/MACE-OMAT0/GemNet-OC均已证伪)，改用VASP小样本+描述符Stacking回归；两模块均只做imid骨架，benz待imid验证通过后再接。
- 立体描述符（%Vbur，M16 预备）：M15 目标A验证发现现有特征几乎全是电子结构量，取代基身份只有 presence 编码、不量化"多大"；补一版标准 %Vbur（球半径 3.5 Å、金属-卡宾距离 2.0 Å、Bondi半径×1.17、排除H——Clavier & Nolan 2010 *Chem. Commun.*, "Percent buried volume for phosphine and N-heterocyclic carbene ligands"标准口径），用 `morfeus-ml` 在已有 xTB 优化几何上、沿 C2 孤对方向放虚拟 Cu 原子计算（`scripts/surrogate/steric_features.py`）。2026-07-03 已对 `structures_full/xyz/` 全部 15,130 个几何算完（100%成功，`results/surrogate/steric/vbur_v3.csv`）；但按 InChIKey 核对后发现该目录实际来自已被标记过时的 `imid_lib_v3_full.csv`（非 M15 用的"正确"`imid_lib_v3menu_graph_full.csv`），与 v4 crude 目标群体（401,856）交集仅 **7,230 个（1.8%）**，非最初估计的 15,130（3.8%）——已加 `in_v4_crude_population` 列标注、真正可用子集内 %Vbur 均值34.3%±4.27%（更贴文献NHC-Cu ~30-58%量级）。v4 新增候选无持久化 3D 构型，是否补构象扩大覆盖留给 M16 立项时再定（见 skill `structures-full-xyz-v3-scope-not-v4-full`）。
