# scripts/ — CLI 与作业入口（薄封装）

- 可执行入口、批处理、rsync/预检包装放这里。
- **业务逻辑**放在 `src/nhc_deprot/`，本目录只做 `argparse` / 环境 source / 调用库。
- 禁止在此复制 production `two_endpoint` 或历史 finetune 选模逻辑。
- 服务器作业：只 `source $WJW/env/envs/<stack>.sh`，不 `source ~/.bashrc`。

## Val e0 四卡并行（硬约束）

```bash
# 自动：2 Val roots × 2 endpoints → 4 张无 VASP / 尽量空闲的 GPU
PYTHONPATH=src python scripts/nhc0801_e0_val_4gpu.py \
  --nhc0801-root $WJW/NHC0801 --batch-id g001

# 只看计划不启动
PYTHONPATH=src python scripts/nhc0801_e0_val_4gpu.py \
  --nhc0801-root $WJW/NHC0801 --batch-id g001 --dry-run

# 严格只要完全空闲卡
PYTHONPATH=src python scripts/nhc0801_e0_val_4gpu.py \
  --nhc0801-root $WJW/NHC0801 --batch-id g001 --require-free
```

逻辑：`src/nhc_deprot/pipeline/e0_val_dispatch.py` + `resources/gpu_inventory.py`。
