# docs/plans/ — 实现前规划（vibe coding 之前）

本目录存放 **动手写代码之前** 的计划、设计草案与任务拆解。

## 命名

```text
YYYYMMDD_<short_topic>_plan.md
```

示例：`20260801_multi_seed_trainer_plan.md`

## 一篇计划至少包含

1. 对应 **mindmap 步骤**（0–12）
2. 目标与非目标（明确不做什么）
3. 拟改动的路径（遵守 `AGENTS.md` 落盘表）
4. 测试计划（优先合成 fixture）
5. 是否需要新门禁/授权
6. 风险与回滚

## 不要放这里

| 内容 | 应去位置 |
| --- | --- |
| 冻结合同 / 数值标定 | `docs/contracts/` |
| 可复用库代码 | `src/nhc_deprot/` |
| CLI 入口 | `scripts/` |
| 踩坑复盘 | 根目录 `RETRO.md` |
| 阶段状态勾选 | `PHASE_STATUS.md` |
