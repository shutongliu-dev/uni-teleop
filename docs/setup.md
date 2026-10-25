# Installation

从 checkout 使用 `./teleop`；它只使用 Python 标准库解析配置。仿真和 SDK 各在自己的 Python 环境中运行，**不要把 pip 安装的 Isaac、MuJoCo 和 MANUS SDK 混装到 Isaac 的 Python 中**。

## 1. MuJoCo + Franka

```bash
python3.12 -m venv .venv-mujoco
.venv-mujoco/bin/pip install -e '.[mujoco]'
```

设置 `SONIC_PYTHON` 为该环境的绝对 Python 路径。Franka 模型需要 DexBench v3 数据，设置 `DEXBENCH_DATA` 为含 `assets/` 的目录。模型读取 `assets/robots/assemblies/fr3_sharpa/isaacsim_6_1_v2/fr3_sharpa.urdf`；不改原数据。

MuJoCo 当前只支持 `--sandbox` 便携工作台，recipe 已自动带上。不要通过 `--family/--task` 请求 DexBench 原场景；该任务迁移尚未实现。

## 2. Isaac + Franka

安装 Isaac Sim 6.1，设置 `ISAAC_PYTHON_LAUNCHER` 为其 Python 启动脚本。单独建立 CPU 环境安装 [DexBench Python 依赖](../backends/isaac/pyproject.toml)，设置 `DEXBENCH_PYTHON`。填写同一份 `DEXBENCH_DATA` 和通过 `nvidia-smi -L` 查询到的 GPU UUID。

启动器会把本次公共模块和原生后端冻结到 `runs/isaac/<session>/source`，随后从快照运行。默认使用 CPU 刚体和关闭评分的练习配置。它与正式 benchmark 评分不是一回事。

## 3. Devices

先生成配置，按组合填字段：

```bash
./teleop configure isaac-factr-manus --output local/isaac.json
./teleop configure mujoco-factr-manus --output local/mujoco.json
```

| 设备 | 需要提供 | 检查重点 |
|---|---|---|
| FACTR | `FACTR_PORT`、`FACTR_CALIBRATION` | 使用 `/dev/serial/by-id/`；电机顺序、符号、零位必须与标定一致 |
| PICO Franka | `PICO_SDK_LIBRARY` | XRoboToolkit 原生 SDK，沿用现有 PICO 客户端；同机 UDP 接收 |
| PICO SONIC | `XR_SDK_PATH` | 可导入 XRoboToolkit Python 模块，机器人服务与客户端可用 |
| MANUS | SDK 路径、SDK Python、手套 ID、官方标定文件 | Core/Integrated 授权、USB 权限、正确左右手；SDK 与 Core 不争抢设备 |
| MANUS→SHARPA | 官方 V4 retarget 目录及其独立 Python | 版本与手部关节模型匹配；不会用闭合预设冒充手套数据 |

MANUS 推荐先用 Core 验证并保存标定，再关闭占用设备的 Core 采集，启动遥操。设备开机灯亮不等于已有新鲜数据。

## 4. SONIC / G1

MuJoCo 使用 Python 3.10+：

```bash
.venv-mujoco/bin/pip install -e '.[sonic]' -e 'backends/mujoco[teleop]'
./teleop configure mujoco-sonic --output local/sonic.json
```

填写 encoder/decoder 所在的 `SONIC_POLICY_DIR` 和 `XR_SDK_PATH`。`isaac-sonic` 使用 Isaac 自带 Python 和 MJCF importer，通过 PhysX 执行同一份 PD 命令。该环境需要可导入 `onnxruntime`、`zmq` 和 XRoboToolkit；先用外部 ZMQ 源也可以绕开本进程的 PICO SDK。

Isaac 版本的 G1 策略迁移是实验性：已验证模型/状态/命令通路，尚未完成实机跟随、平衡、接触与策略效果验证。原有复杂 MuJoCo 场景请用 `mujoco-pico`，完整相机/录像流程仍保留在那里。

## 5. Run

```bash
./teleop doctor local/mujoco.json
./teleop run local/mujoco.json --dry-run
./teleop run local/mujoco.json
```

`doctor` 检查路径，不会连接或使能电机，也不能证明手套已配对。额外运行参数放在 `--` 后；例如 Isaac 选择任务：

```bash
./teleop run local/isaac.json -- --family mechanical_assembly --task peg_insertion
```

`local/`、`runs/`、SDK、权重均不上传。配置中路径允许有空格，启动器直接传 argv，不经过 shell 求值。
