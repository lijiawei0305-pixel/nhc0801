# AGENTS.md — NHC614 计算筛选系统 AI 工作指引

> 根级索引（每会话载入）：路由 + 硬约束 + 工作纪律；细节在 `doc/claude/`，完整原版备查 `doc/archive/CLAUDE-full.md`。
> **项目定位**：HPC（`10.66.62.183`，8×V100）预测**咪唑鎓型 NHC 前驱体在 Cu(111) 脱质子化能力**并排序——分子漏斗（RDKit→xTB→PySCF→Multiwfn）+ 周期表面（VASP/CP2K）+ Pareto。本地 macOS 仅编辑、不跑计算。

## 路由（动手前按需读）

| 文档 | 何时读 / 内容 |
| --- | --- |
| `doc/proposal.md` | 改业务逻辑前；需求**真值**（性质、技术栈、阶段范围） |
| `doc/design-consistency-audit.md` | **读/改 `detailed-design.md` 前必读**；已知偏差清单（F-C1…） |
| `doc/detailed-design.md` | 深入模块时；§0–§3（C1–C6 详解 §0.9）、§5.4、§6.2.4–6.7、§7、§8.2.5 可信；附录 A=已验证计算参数/方法；§6.1–6.2.3 旧分子晶体作废；§8.3/8.4 见 `tasks/m10` |
| `doc/tasks/mXX-*.md` | 改某模块前优先读（比 detailed-design 新、更贴 proposal） |
| `.codex/skills/` | **Codex 疑似踩过的坑先查这里**：每条踩坑=一个 project skill，按症状自动命中；索引/旧#N↔skill 映射见 `.codex/skills/README.md`。Claude Code 镜像在 `.claude/skills/`，互写时见下方双栈规范。旧 `RETRO.md` 已瘦身为存根（仅留未解决项 #7 与 E_bind 行动清单） |
| `doc/claude/modules.md` | 模块地图 M1–M14 + 文件接口契约 |
| `doc/claude/science-decisions.md` | 锁定的科学/规模决策 |
| `doc/claude/commands.md` | HPC 环境激活与 `nhc`/`mq` 速查 |
| `doc/claude/deployment-notes.md` | 已知违规待修 + VASP 运行细节 |
| `doc/PRODUCT.md` | nhc/mq 交互(UX)基准 |

> **唯一权威设计文档** = `doc/detailed-design.md`；历史/已整合文档在 `doc/archive/`，仅备查。**`doc/` 结构已冻结，不新增文档**（见工作纪律 2）。

## 部署硬约束（C1–C8，违反即错）

| ID | 约束 | 级别 |
| --- | --- | --- |
| C1 | 唯一计算节点 HPC `10.66.62.183`（plab）；计算经 SSH 提交 | Critical |
| C2 | 本地 macOS 只编辑/看结果，不装计算依赖、不跑计算；可装 pytest/ruff/mypy | Critical |
| C3 | 项目根 `$WJW = /home/plab/test/WJW`，勿迁移/勿用 symlink 伪装 | High |
| C4 | conda 双环境：`molecular`（rdkit/pyscf/geometric/xtb）+ `periodic`（ase/pymatgen/spglib） | High |
| C5 | 作业脚本禁 `source ~/.bashrc`，须显式 `source $WJW/env/envs/<软件>.sh`（一次一栈） | Critical |
| C6 | GitHub CDN 不可达，外部下载用国内镜像（清华/USTC/南大 + 多源回退） | High |
| C7 | HPC 是 WHUT 内网服务器：连接 WHUT 校园网时可直连；在外部网络或出现 SSH 连接失败/超时/被关闭时，必须强制走本机 SS/SOCKS5 代理 `127.0.0.1:11080`（详见 `/Users/cc/Atrust/README.md`），再判断服务器状态 | Critical |
| C8 | 大超胞 VASP（尤其 `5×5 nhc_slab`/共吸附）会吃满系统 RAM；启动或重启前必须查 `free -h`、`ps --sort=-rss` 与并发 M4/PySCF。若 `vasp.out` 出现 `signal 9 (Killed)`/脚本 `rc=137`，先按 OOM/内存竞争处理，不要原样重启；等 M4 降并发/结束或改资源口径后再跑 | Critical |

> 违规修复 + VASP 细节（仅 `vasp_std`、`NCCL_P2P_DISABLE=1`、显式 `CUDA_VISIBLE_DEVICES`）见 `doc/claude/deployment-notes.md`。
> 资源口径：`5×5 nhc_slab/step1` 生产档曾显示 `total amount of memory used by VASP MPI-rank0 25640492 kBytes`，8 rank 约 195 GiB；若同时跑 `dft_batch --parallel 10 --threads-per-job 8`（PySCF 子进程约数十 GiB），251 GiB 节点会触发 OOM kill。见 skill: `vasp-large-supercell-oom-concurrent-dft`。
> 代理 SSH 示例：`ssh -o ProxyCommand='nc -x 127.0.0.1:11080 -X 5 %h %p' nhc614`；若本机 DNS/路由出现 `198.18.x.x`，按 fake-ip/TUN 路径处理，不可据此判定公网直连可用。
> 若本地 SS/SOCKS5 登录状态过期或 `127.0.0.1:11080` 未监听：先 `launchctl kickstart -k "gui/$(id -u)/com.cc.zju-connect-whut"` 并用 `lsof -nP -iTCP:11080 -sTCP:LISTEN` 复查；若仍失败，运行 `~/.local/bin/zju-connect-whut -config "$HOME/Library/Application Support/EZ4Connect-WHUT/config.toml"` 前台重新登录，按验证码页面完成验证，看到 `Password-based authentication succeeded` 与 `SOCKS5 server listening on 127.0.0.1:11080` 后按 `Control+C` 退出，再执行 `launchctl kickstart -k "gui/$(id -u)/com.cc.zju-connect-whut"` 恢复后台服务。

## 分子链(mq)运行机制（防幻觉；细节 `doc/claude/commands.md`）

- `mq` 选项2「批量队列」= **顺序链 M2→M3→M4→M5→M6**（非并行 worker）；M4 PySCF DFT 是瓶颈、纯 CPU。
- **每步只 source 所需栈（C5）**：全程 `molenv.sh`，M3 加 `xtb.sh`、M5 加 `multiwfn.sh`。Multiwfn 可执行是 `Multiwfn_noGUI`（经 `$MULTIWFN_BIN`），无 `Multiwfn` 命令。
- **运行根**：`PYTHONPATH=$WJW` + `cd $WJW`。
- **控并行**：`mq` 固定 auto 填满节点；留余量须直接跑 `m03_batch_runner.py`/`mol/dft_batch.py` 加 `--parallel N --threads-per-job 8`。

## 双栈 Skill 书写规范（Codex / Claude Code）

- **本端优先**：Codex 写/查 `.codex/skills/`；Claude Code 写/查 `.claude/skills/`。互写对方 skill 前，先读目标目录 `README.md` 和相邻样例，不能裸复制另一端目录。
- **Codex skill 格式**：目录名必须等于 frontmatter `name`，只用小写字母、数字、连字符（禁止 `.`、`_`、大写）；`SKILL.md` frontmatter 只保留 `name` 与 `description`；`description` 控制在 120 字以内，只写症状/报错 + 模块 + 关键词，长根因和命令放正文六段式里。`agents/openai.yaml` 只属于 Codex UI，不要复制到 `.claude/skills/`。
- **Claude Code skill 格式**：保持本项目六段式（触发场景/根因/解决方案/验证/关联文件&路径/来源），更新 `.claude/skills/README.md` 与 `RETRO.md` 指针；不要写 Codex 专用 `agents/openai.yaml`。新 skill 名也尽量采用 Codex 兼容的小写连字符，旧名改名必须同步索引和引用。
- **双向同步**：从 Claude 迁到 Codex 时先规范名称（如 `1.5x`→`1-5x`、`deltaE`→`deltae`），再压缩 Codex `description`；从 Codex 迁到 Claude 时补足六段式上下文。若 Codex 启动提示 `Skill descriptions were shortened`，先检查 skill 数量和 description 总长度，通常不是 YAML 损坏。

## 工作纪律（本项目硬性）

1. **先问不猜**：不要猜测用户意图。架构、数据流、模块划分、计算口径、资源调度、技术路线或执行边界不明确、存在歧义或需要技术决策时，必须先向用户提问，直至确认完毕，再动手。以后所有对话均遵循此原则。
2. **doc/ 冻结 + 不擅自重写大文档**：现阶段不新增独立文档，新约束/决策并入既有文档（需求/性质/阶段→`proposal.md`，模块/算法→`detailed-design.md`，AI 规则→本文件，踩坑→`.codex/skills/`，必要时同步 `.claude/skills/`；每条一坑一个 skill，索引 `README.md`)，偏差→`design-consistency-audit.md`）；`detailed-design.md`(151KB) 等未经确认不整体重写。
3. **改代码必同步文档**：改 `scripts/` 同步 `detailed-design.md` 对应章节；触及需求/性质/技术栈/阶段再同步 `proposal.md`。
4. **低耦合·高内聚·可独测**：模块经文件契约（CSV/Parquet/XYZ/JSON）解耦、单写者；M1/M6/M10/M13 必可桩测。
5. **遇坑先查 `.codex/skills/`**（按症状命中对应 skill；索引 `.codex/skills/README.md`），别重复踩。**遇到/修复任何踩坑后，无需用户提醒、主动且立即**把它写成新 Codex skill（`.codex/skills/<name>/SKILL.md`，格式见“双栈 Skill 书写规范”），并按需同步 `.claude/skills/`、更新对应 `README.md` 索引与计数，在 `RETRO.md` 留一行指针（skill 名 + 一句根因）。这是默认纪律，不是用户要求才做。
6. **改超 30 行或新模块/安全相关**：跑 `ccg` 质量门（`/verify-change`、`/verify-quality`、`/verify-security`、`/verify-module`；仅 Critical/High 先修）。
7. **代码风格随上下文**。
8. **真实计算结束必出检测报告**：本地与服务器同步建 `reports/audit-YYYY-MM-DD-<project>/`（Markdown 主报告 + 命令/日志摘要 + CSV/JSON 检查清单；核对候选集、参数口径、空值、抽样回溯、本地/服务器一致性）。
