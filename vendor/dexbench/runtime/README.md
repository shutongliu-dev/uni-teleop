# runtime/：本机仿真器与 Python 环境

这里只放指向本机环境的链接，不进 Git。每个人在自己的 checkout 里执行一次：

```bash
runtime/setup.sh --isaacsim <Isaac Sim 6.1 目录> --venv <Python 3.12 环境目录>
```

| 链接 | 指向 | 用途 |
|---|---|---|
| `runtime/isaacsim` | Isaac Sim 6.1 安装目录，内含官方 `python.sh`（或自备的 `python_compat.sh`） | `scripts/isaac_python.sh` 启动原生 worker 和原生验证 |
| `runtime/venv` | Python 3.12 环境，按 `pyproject.toml` 安装依赖 | HTTP 服务、Agent 客户端、CPU 测试 |
| `.venv` | `runtime/venv` | 文档中统一使用 `.venv/bin/python` |

`ISAAC_PYTHON_LAUNCHER` 环境变量优先于 `runtime/isaacsim`。代码和数据都不记录仿真器的安装位置：Isaac Sim 内置材质（如 `OmniGlass.mdl`）只按名字引用，由本机安装解析。

新建 CPU 环境示例：

```bash
uv venv --python 3.12 ~/.venvs/dexbench
uv pip install --python ~/.venvs/dexbench/bin/python -e '.[asset-build]' pytest==9.1.1
```

原生运行需要能渲染的 NVIDIA GPU。多卡机器上请按 UUID 或 PCI 地址选卡（`CUDA_VISIBLE_DEVICES=<GPU UUID>`），不要按序号；不同工具对显卡的编号可能不一致。

如果 worker 日志出现 `The currently installed NVIDIA graphics driver is unsupported`，相机返回空图，说明本机驱动没有通过 Isaac Sim 的 RTX 版本校验。确认该驱动能正常渲染后，可设 `ISAAC_ALLOW_UNVALIDATED_DRIVER=1` 跳过这项校验。
