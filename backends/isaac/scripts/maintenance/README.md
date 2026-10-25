# 项目维护

- `freeze_project_sources.py`：生成完整源码快照，验证文件、权限与实际进程导入来源。
- `derive_project_sources.py`：从冻结父快照和显式变更清单派生新快照，保留来源证据。
- `rebase_runtime_contract.py`：验证字节与来源后，重定位运行契约中的绝对路径。
- `derive_runtime_contract.py`：仅处理 service.py 单文件替换；其他改动须重新生成完整运行契约。
- `background_job.py`：显式启动后台进程，记录 PID、启动时间和退出状态。
- `compare_physics_recordings.py`：只读比较命令、状态与原始物理轨迹。
- `task_status.py`：从任务目录生成 README 任务状态表。
- `parallel_pytest.py`：每个测试文件一个 pytest 进程并行运行全套 CPU 测试，并汇总结果。
- `check_module_dependencies.py`：静态检查 `bench/`、`scenes/` 不依赖 `agent`。

用法与验证边界见 [源码快照](../../docs/agent_bench/source_snapshots.md)。这些工具不证明任务成功或物理有效。
实验输出放在 `outputs/`，不留在源码目录。数据包的生成与校验见 `scripts/data/package.py` 和 [data/README.md](../../data/README.md)。
