# uni-teleop

统一管理 **FACTR 主从臂、MANUS 手套、PICO VR** 与 **Isaac Sim / MuJoCo** 的遥操作工程。
整合当前 DexBench 遥操分支和 SONIC-MuJoCo 工作树，提供统一启动入口、本机配置、
来源快照和操作文档。原项目仍保留各自的控制实现与 Python 环境。

## 已纳入的控制方式

| 入口 | 输入 → 输出 | 说明 |
|---|---|---|
| `isaac-factr` | FACTR 七轴 → Franka；扳机 → 手 | 固定关节零点与方向映射 |
| `isaac-factr-manus` | FACTR 七轴 + MANUS → Franka + SHARPA | 主臂和手指独立采集 |
| `isaac-pico-controllers` | PICO 手柄 → Franka + SHARPA | 手腕/TCP 映射、IK、离合 |
| `isaac-pico-hands` | PICO 裸手 → Franka + SHARPA | 手腕位姿 + 手指重定向 |
| `isaac-pico-manus` | PICO 手柄 + MANUS → Franka + SHARPA | 手柄控制手臂，手套控制手指 |
| `isaac-manus` | MANUS → SHARPA | 单独测试手指，机械臂保持 |
| `mujoco-pico` | PICO 全身/头部/双手 → G1 | SONIC ONNX、视频、录制与手柄触觉 |
| `mujoco-superdex` | ZMQ 手指目标 → SuperDex | 双手仿真订阅端 |
| `pico-superdex-source` | PICO 手指 → ZMQ | 与 SuperDex 订阅端配合运行 |
| `mujoco-simple` | PICO → SIMPLE 场景 | 外部 SIMPLE 安装的启动适配器 |
| `mujoco-g1` / `mujoco-h2` | 固定目标 → 仿真 | 基础模型自检；H2 未接全身遥操策略 |

支持不等于所有组合都已完成实机验收。MANUS→MuJoCo、FACTR→MuJoCo 尚无统一适配；
不把它们列成已实现功能。原工作站的 `sonic-keyboard` 属于独立的
GR00T-WholeBodyControl 部署项目，不在 SONIC-MuJoCo 源码内，本次未将其冒充为已集成入口。

## 快速开始

```bash
git clone https://github.com/shutongliu-dev/uni-teleop.git
cd uni-teleop
./teleop list
mkdir -p local
cp docs/host.example.json local/host.json
# 按 docs/setup.md 填入所选模式需要的路径。
./teleop doctor isaac-factr-manus --config local/host.json
./teleop run isaac-factr-manus --config local/host.json --dry-run
```

运行 FACTR + MANUS，选择机械装配的皮带任务：

```bash
./teleop run isaac-factr-manus --config local/host.json -- \
  --family mechanical_assembly --task nist_belt_drive_assembly \
  --rear-camera --side-view
```

启动后点击 **开始 / 恢复跟随**。使用固定标定文件，普通 Start 不重新定义零点。
`--` 后面的参数原样传给后端；例如 `--allow-provisional-calibration` 仅在确实使用
临时标定练习时显式添加。实体托举默认关闭，见 [FACTR 与反馈](docs/operation.md)。

PICO 与 MuJoCo 示例：

```bash
./teleop run isaac-pico-hands --config local/host.json -- --sandbox
./teleop run isaac-pico-manus --config local/host.json -- --family kitchen --task breakfast_serving
./teleop run mujoco-pico --config local/host.json -- --scene sweep
./teleop run mujoco-g1 --config local/host.json -- --headless --steps 100
```

`doctor` 只检查已配置路径，不启动仿真、不使能电机，也不代表手套已配对或授权有效。
`--dry-run` 只显示命令。运行后端参数帮助可在 `--` 后加 `--help`；Isaac 外层启动器仍会
准备源码快照和采集进程，因此离线查参数优先阅读下方原始文档。

## 结构

```text
uni_teleop/          不依赖仿真器的统一 CLI
profiles/            12 个可读 JSON 启动配置
docs/                安装、操作、集成边界和验证记录
vendor/dexbench/     Isaac、FACTR、MANUS、PICO、场景与原测试
vendor/sonic-mujoco/ MuJoCo、G1/H2、SONIC、PICO、SuperDex、录制与原测试
sources.lock.json   导入时源提交及逐文件 SHA-256
tests/               统一入口测试
local/               本机路径与标定引用（Git 忽略）
runs/                Isaac 冻结源码与会话输出（Git 忽略）
```

Isaac 入口默认采用工作台精简版、CPU 刚体、无实时评分、穿透仅报警的练习配置。
液体/粒子任务需要显式 `--physics-profile scene`；依赖称重/电视反馈的任务可能需要
`--keep-task-observer`。数据包和原发布规则保持独立，练习运行不等于正式任务通过。

## 文档与测试

- [环境安装](docs/setup.md) · [操作与标定](docs/operation.md)
- [架构、来源与集成改动](docs/integration.md) · [验证范围](docs/validation.md)
- [原 Isaac/PICO 文档](vendor/dexbench/docs/pico_teleop.md)
- [原 FACTR/MANUS 文档](vendor/dexbench/docs/master_slave_manus.md)
- [原 SONIC-MuJoCo 文档](vendor/sonic-mujoco/README.md)
- [第三方来源说明](THIRD_PARTY.md)

```bash
python3 -B -m unittest discover -s tests -v
python3 -B tools/verify_sources.py
```

本仓库不包含 MANUS 授权、商业 SDK、Isaac 安装包、SONIC 权重、个人标定或遥操录像。
各上游代码与模型保留原有归属，详见第三方说明。
