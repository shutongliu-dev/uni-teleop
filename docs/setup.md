# 安装与本机配置

统一 CLI 仅使用 Python 标准库，推荐 Python 3.10+。在仓库根运行 `./teleop`，无需安装
全局 Python 包。运行环境相互隔离，不把 Isaac 的 USD/Python 与 MuJoCo 环境混装。

## Isaac / DexBench

1. 安装 Isaac Sim 6.1 的 Linux 运行环境。`ISAAC_PYTHON_LAUNCHER` 指向其
   `python.sh`，或已有兼容包装器 `python_compat.sh`。后者必须保留该环境的 USD/Python。
2. 创建 CPU 环境并安装源码依赖：

   ```bash
   python3.12 -m venv local/dexbench-env
   local/dexbench-env/bin/python -m pip install -e vendor/dexbench pytest
   ```

   将 `DEXBENCH_PYTHON` 指向这个环境。已有工作站可以直接引用现成 CPU 环境。
3. 从 [DexBench releases](https://github.com/VVVVME50/dexbench/releases)
   取 **data-v3** 数据包，按上游 SHA256SUMS 校验并解压；`DEXBENCH_DATA` 指向包含
   `MANIFEST.json` 的目录。原始机器人与任务资产约数 GB，未重复提交到本仓库。
4. `CUDA_VISIBLE_DEVICES` 填目标 NVIDIA GPU 的 UUID（`nvidia-smi -L` 可查）。

`vendor/dexbench/scripts/isaac_python.sh` 会把冻结后的源码目录传给 Isaac。
每次交互运行都在 `runs/isaac/` 保存独立源码快照，避免边改源码边运行。

## FACTR 主臂

- `FACTR_PORT` 使用 `/dev/serial/by-id/...` 稳定串口路径。
- `FACTR_CALIBRATION` 指向本机已测量的七轴参考、符号和偏置文件。
- 模板：[arm_calibration.template.json](../vendor/dexbench/scripts/teleop/config/arm_calibration.template.json)。
  模板不能直接当成已完成标定。复制到 `local/` 后通过现场标定流程生成实际内容。
- 串口权限与 USB 延迟按本机配置。切换任务不重记零点，不共享其他设备的序列号与标定。

## MANUS + SHARPA

使用官方 MANUS SDK 3.2.0 或 3.2.1、对应 Python 环境和有效许可。

| 设置 | 内容 |
|---|---|
| `MANUS_SDK_PYTHON` | SDK 的 Python 目录 |
| `MANUS_PYTHON` | 能加载该 SDK 的隔离 Python |
| `MANUS_SDK_VERSION` | `3.2.0` 或 `3.2.1` |
| `MANUS_GLOVE_ID` | 所选手套十进制 ID |
| `MANUS_CALIBRATION` | 官方 Core 保存的标定 JSON |
| `SHARPA_RETARGET_ROOT` | 官方 `retargeting_alg_release_V4.0` 目录 |
| `SHARPA_RETARGET_PYTHON` | 已安装该优化器依赖的 Python |

优化器来源：[sharpa-manus-sdk](https://github.com/sharpa-robotics/sharpa-manus-sdk)。
许可与设备接收器是不同检查项；日志可能包含授权字段，不上传原始 SDK 日志。
本仓库保留官方优化器进程适配层，不复制商业 SDK 或本机许可文件。

## PICO

Isaac 路线使用 ctypes 原生库，`PICO_SDK_LIBRARY` 指向 `libPXREARobotSDK.so`。
SONIC 路线使用 XR Python 绑定，`XR_SDK_PATH` 指向包含 `xrobotoolkit_sdk*.so`
与 `lib/libPXREARobotSDK.so` 的目录。对应 PC Service 和头显客户端需另外安装。
两条路线均保留已有视频协议；切换视频服务后在头显端重新点击 Listen。

## MuJoCo / SONIC

```bash
cd vendor/sonic-mujoco
uv sync --extra sonic --extra teleop --extra recording
```

把 `SONIC_PYTHON` 指向该目录 `.venv/bin/python`；`SONIC_POLICY_DIR` 指向包含
`model_encoder.onnx`、`model_decoder.onnx` 的目录。基础 `mujoco-g1/h2` 自检不需权重。
视频回传所需 GStreamer 系统组件及录制参数见原 SONIC README。

SuperDex 另需其资产包；设置 `SUPERDEX_ASSETS_PATH`。在两个终端分别运行：

```bash
./teleop run mujoco-superdex --config local/host.json
./teleop run pico-superdex-source --config local/host.json
```

SIMPLE 使用独立安装和环境，设置 `SIMPLE_ROOT`；`mujoco-simple` 负责调用其原入口，
不会自动下载全部 HSSD 场景。原工程的扫描重建、策略评测工具保留于 vendor 中，
不作为已验证的统一遥操组合。

`lab_scan` 依赖额外的现场重建碰撞 manifest 和它引用的资产；这些不属于 Git 中的
基础模型。需要运行该场景或完整原测试时，显式设置
`SONIC_RECONSTRUCTION_COLLISION=/path/to/lab_scan_collision.json`。该环境变量由启动器
透传；不要把他人的现场扫描数据当作通用演示资产上传。
