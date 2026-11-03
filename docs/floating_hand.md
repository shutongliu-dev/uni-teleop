# PICO + MANUS：独立 SHARPA 手

只加载一只右 SHARPA、桌面和方块，没有 Franka，也不使用机械臂 IK。
右 PICO 手柄控制手掌的六自由度位姿，右 MANUS 手套通过现有官方 SHARPA V4
优化器控制 22 个手指关节。当前不包含左手。

## 启动

```bash
./teleop configure mujoco-pico-manus-sharpa-hand --output local/hand-mujoco.json
./teleop configure isaac-pico-manus-sharpa-hand --output local/hand-isaac.json
# 填写生成文件中的设备、SDK、数据与 Python 路径
./teleop doctor local/hand-mujoco.json
./teleop run local/hand-mujoco.json
./teleop run local/hand-isaac.json
```

已有 host 配置也可以直接复用：

```bash
./teleop run mujoco-pico-manus-sharpa-hand --config local/host.json
./teleop run isaac-pico-manus-sharpa-hand --config local/host.json
```

Isaac 大零件分类练习场景（独立手，复用 Bench 的零件、托盘和桌面）：

```bash
./teleop run isaac-pico-manus-sharpa-hand --config local/host.json -- --scene parts-sorting
```

该模式读取本机 `parts_sorting/desktop_v1` 的冻结场景并校验场景及所用资产，省略
Franka、机械臂立柱和桌面装饰；沿用独立手 CPU 物理配置，不启用原任务评分。
运行目录保存 `scene_receipt.json` 和 `task_assets.json`，原始资产不修改。
Isaac 的 Bench 场景保留主视角，并显示独立的腕部、后方桌面窗口。
腕部沿用已有遥操的手内侧安装参数，跟随物理引擎测得的手掌姿态；
后方窗口沿用工作台后视相机规则。电脑保留三个窗口；PICO 显示双目主视角和
两眼共用的腕部小窗，后方视角只在电脑上显示。

Isaac 使用 6.1 原生 URDF 导入和 PhysX；MuJoCo 使用 3.x URDF 导入。
两者分别在自己的 Python 环境运行，Isaac 不需要安装 MuJoCo。
SHARPA 模型从 `DEXBENCH_DATA` 的 FR3/SHARPA 资产中提取手掌以下子树，生成文件
写入本次 `runs/floating_hand/`，不会改写资产或加载 Franka。
MuJoCo 环境需要 numpy、scipy、mujoco 和 Pillow；SDK、官方优化器仍使用独立 Python。

## 操作

- 启动时暂停。两个后端均可按空格开始/暂停；Isaac 也可点击 **Start / Pause**。
  Isaac 空格只控制遥操，不切换物理时间线；若从编辑器暂停物理，暂停期间不提交控制力。
- 默认按桌面视角对齐：开始时把头显朝向你操作时的正前方，系统将这个水平朝向
  对齐到当前相机看向桌面的方向；向右对应画面右侧，向前对应桌面深处，上下保持竖直。
  对齐后转头、松开 grip 都不会改变方向；转动桌面相机或换站位后按 R / Recenter 重新对齐。
  需要有效头显朝向才能完成对齐；头显接近竖直朝上/下时等待。`--motion-frame world` 保留固定世界坐标模式。
- 按住右手柄 grip 移动手掌，松开保持。重新按住时从当前实际手姿态重新锚定。
- MANUS 手指独立跟随，不要求一直按 grip。
- 按 R 或 Isaac 的 **Recenter** 重新锚定手柄，不重置桌面物体。
- PICO 数据过期只保持手腕，MANUS 数据过期只保持手指。PICO 恢复时重新锚定，
  第一帧不跳变；手指收到有效数据后恢复。手动暂停需要手动开始。
- `--scale 0.7` 可缩小平移比例；旋转仍为 1:1。

PICO 使用 XRoboToolkit 原有客户端和本机 SDK。运行时自行启动输入桥，
若已启动兼容桥可传 `--external-pico`，默认 UDP 端口 5557。
这条路线不打开 FACTR 串口，不输出实体电机力矩。
Isaac 配方默认开启 PICO 视频：沿用 XRoboToolkit 的 Listen 连接方式，控制端口 13579，
每眼 640×480、腕部小窗 256×192，固定主视角，不随转头改变相机。
H.264/H.265 由客户端原协议选择；重启后需要重新点 Listen。
`--no-video` 可关闭回传；MuJoCo 独立手目前仍只有电脑窗口。
运行目录中的 `pico_preview.png` 是双目发送帧预览，`pico_video.json` 记录发布数量；
发布帧不等于头显已收到，连接日志中的 Streaming 才表明建立了传输连接。

## 物理与记录

手掌使用自由根关节和六自由度 PD 力/力矩跟踪，不通过逐帧设置位姿移动。
指关节使用原 URDF 限位和力矩上限。虚拟手有重力支持；物体正常受重力影响。
保留手与桌面、物体接触，关闭手内部自碰撞。两个后端采用各自求解器，
接触结果不保证逐帧一致。

每次运行保存 `frames.jsonl`（目标/实际位姿、手指关节、输入健康状态）、
`report.json` 和 SDK 日志；`--screenshot` 在退出前生成 1280×960 预览。
无设备预览（不连接 SDK）：

```bash
./teleop run mujoco-pico-manus-sharpa-hand --config local/host.json -- --demo --steps 200 --screenshot
./teleop run isaac-pico-manus-sharpa-hand --config local/host.json -- --demo --steps 200 --screenshot
```

也可直接在对应仿真 Python 中运行 `python -m uni_teleop.runtime.floating_hand
--backend mujoco --data /path/to/dexbench/data --demo --steps 200`，不需要硬件配置。

设计参考与验证边界见 [来源说明](provenance/floating-hand.md) 和 [验证记录](validation.md)。
