# Architecture and extension

## Ownership

`devices/` 只采集并时间戳输入；`mapping/` 把输入映射到机器人命名关节；`backends/` 执行物理并返回实际状态。`runtime/` 负责把它们组合起来、关闭自己启动的进程以及记录会话。物理反馈在 `feedback/`，不混进输入采集或 launcher。

从两个原项目抽取的设备、映射、录像和视频模块有实际代码迁移，旧 import 已更新。完整任务引擎与模型保留在 `backends/isaac` 和 `backends/mujoco`；这部分并不是凭空重写的轻量包。纯 CPU Franka 运动学暂复用 `backends/isaac/bench/control`，它不加载 Isaac SDK。

## Franka contract

- 29 个命名关节：7 个 FR3 + 22 个 SHARPA；单位弧度。
- 共享 `Retargeter`、`JointRetargeter`、`ManusJointRetargeter`、`PicoManusRetargeter`。
- 同一标定和 `isaacsim_6_1_v2` 装配坐标约定；按名称索引，不能假定原生 joint 数组顺序。
- 每次映射读取实际关节，执行后 acknowledge 已发送目标。输入采集与模拟步进分开。
- MuJoCo `FrankaWorkbench` 从固定 URDF 导入，执行位置驱动和仿真重力补偿；包含碰撞网格桌面练习场景。它不复刻 Isaac 评分、接触证据或正式任务资产。

## SONIC contract

`core/robot.py` 定义 `RobotState` 与 PD `RobotCommand`，`core/g1.py` 定义 29 轴身体顺序。SONIC encoder/decoder 不导入原生仿真器。

MuJoCo 和 Isaac G1 均提供 `get_robot_state()`、`step(command, steps)`、`render()`、`reset()`、`close()`。Isaac 使用 NVIDIA MJCF 转换器和 PhysX articulation tensors，PD 每个物理子步更新。身体 29 轴、双手 40 轴、颈部 2 轴分别按名称解析。源模型努力上限用于 torque clipping。没有把每帧 qpos 瞬移当成动力学控制。

MJCF→PhysX 转换存在接触和惯量表达差异，当前关闭 G1 自碰撞。合成模型/接口检查不能代替策略迁移与站立稳定性验证。复杂场景、完整 RGB-D/触觉记录仍由原 MuJoCo 路线提供。

## Add a route

1. 新设备实现最新样本读取、身份和时间戳；通过合成/录制输入测试丢帧与恢复。
2. 新机器人映射声明关节顺序、坐标系、标定身份和支持的动作。
3. 新 backend 实现命名关节读写，先核对 FK，再检查原生实际运动；只加载模型不算完成。
4. 在 `configs/components.json` 注册依赖片段，在 `configs/recipes.json` 组合它们。`./teleop configure` 自动生成必需字段。
5. 文档区分代码已实现、原生仿真已测、实机已测。禁止把未匹配机器人/策略的组合静默路由到其他控制器。

## Tests

```bash
python -m unittest discover -s tests
# Native MuJoCo checks need the model data and the MuJoCo Python environment:
PYTHONPATH="$PWD:$PWD/backends/isaac:$PWD/backends/isaac/tests" \
  DEXBENCH_DATA=/path/to/data .venv-mujoco/bin/python -m pytest tests/native -q
python tools/verify_sources.py
```

原后端测试从各自目录运行，并把仓库根目录与该后端加入 PYTHONPATH。Isaac launcher 在启动前冻结公共模块和引擎资源；历史 baseline failures 见 validation，不能通过修改断言掩盖。
