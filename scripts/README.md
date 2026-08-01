# scripts/ — CLI 与作业入口（薄封装）

- 可执行入口、批处理、rsync/预检包装放这里。
- **业务逻辑**放在 `src/nhc_deprot/`，本目录只做 `argparse` / 环境 source / 调用库。
- 禁止在此复制 production `two_endpoint` 或历史 finetune 选模逻辑。
- 服务器作业：只 `source $WJW/env/envs/<stack>.sh`，不 `source ~/.bashrc`。
