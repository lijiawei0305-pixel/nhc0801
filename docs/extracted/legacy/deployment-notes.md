# 部署补充说明（已知违规 + VASP 运行细节）

> 从 `CLAUDE.md §1` 拆出。硬约束 C1–C6 主表见 `CLAUDE.md`；完整详解见 `doc/detailed-design.md §0.9`。

## 历史违规（均已修复，2026-06-16 复核）

- ~~`{mq,nhc}_wrapper.sh` 用 `conda activate`~~ → **已改** `source $WJW/env/envs/molenv.sh`（合规 **C5**）。
- ~~`§3.1 install_hpc.sh` 硬编码 GitHub~~ → **已改** 清华/USTC/南大多源镜像回退（合规 **C6**；`detailed-design §3.1` 与 `env/install_hpc.sh` 一致，F-H3 / skill: github-cdn-unreachable-mirror-fallback）。

## HPC 长任务必须脱离本地（硬原则，2026-06-20 用户确认）

> 本地 macOS 只编辑（C2），真实计算全在 HPC（C1），任务动辄数十小时（M4 DFT ~48h、VASP 天级）。用户经常外出/断网。**凡给 HPC 写的长任务，必须保证用户断网/关机/SSH 断都不影响 HPC 上的运行。** 记忆见 `memory/hpc-jobs-decouple-from-local.md`。

- **启动即脱离**：`setsid`（非 nohup）+ `</dev/null` + 显式日志重定向，彻底脱 TTY/SSH（skill: `setsid-not-nohup-for-bg-logs`）。
- **自包含在服务器侧**：env 激活、CPU 封顶 watchdog（skill: `runtime-cpu-cap-whole-process-tree-watchdog`）、resume 标记（如 `freq.json`）全部跑在 HPC，不靠本地驱动。
- **本地只做只读监控**：进度查询可选、只读、对断网容错；任务推进绝不依赖本地轮询。
- **resume 安全** + **离开前给快照**（进程存活 + 资源约束生效 + 已在出活 + 落 `reports/.../PRODUCTION_STATUS.md`）。

## VASP 运行细节（C1/C5 配套）

- 仅 `vasp_std` 可用（无 `vasp_gam`/`vasp_ncl`，Gamma-only 也用 `vasp_std`）。
- 必加 `export NCCL_P2P_DISABLE=1; NCCL_DEBUG=WARN`。
- 显式 `export CUDA_VISIBLE_DEVICES`。
