# sonic_mujoco

`sonic_mujoco` 是面向 Unitree G1 的轻量 MuJoCo 全身遥操作与数据采集工程，
同时提供基于 Unitree 官方资产的 H2 仿真基础。G1 链路在一个进程内完成 PICO 全身
姿态接收、SONIC 推理、仿真控制、第一视角回传、接触触觉反馈和 LeRobot 风格
episode 录制。

项目当前提供空场景、Sweep 扫桌、Chair lean 椅背倚靠，以及刚性与柔性物体
搬运任务。实现强调清晰的数据流和较少的运行时组件，适合仿真遥操作、示教
数据采集及 sim-to-real 实验。

## 功能

- MuJoCo 中的 29-DoF G1、双侧 40-DoF 五指灵巧手、PD 控制与 SONIC
  encoder/decoder 推理；
- 官方 31-DoF Unitree H2 模型、固定版本的 actuator 顺序、原生两轴头部、
  头部双目与双腕相机，以及独立的状态和 PD 控制接口；
- PICO 24 关节全身追踪和 50 Hz 真人到 G1 动作映射；
- PICO 26 点手部追踪到五指灵巧手的实时 retargeting；
- PICO 头部姿态到两轴仿真颈部和第一视角相机的实时跟随；
- G1 头部相机到 PICO 的低延迟 H.264/H.265 画面回传；
- G1 双目第一视角、双腕 RGB，以及同步轴向深度、深度导出的表面法线和相机位姿；
- 同一进程内连续录制多条 episode，并显示 `REC` 与录制时长；
- 机器人—场景接触的力、冲量和接触部位记录；
- 覆盖胸背与双臂的 624 点逻辑触觉皮肤，以及与真机一致的三路
  `uint8[256]` 输出；
- 按碰撞、按压和滑动状态生成左右手柄触觉反馈；
- 接近真实布置、带物理随机化的任务场景和原生 MuJoCo 柔性体。

## 环境要求

- Ubuntu 22.04 或 24.04；
- Python 3.10 及以上；
- `uv` Python 包管理器；
- 支持 OpenGL/EGL 的 NVIDIA 驱动；
- PICO 头显与手柄；
- XRRobotKit PC Service 和 Python SDK；
- SONIC encoder/decoder ONNX 权重。

项目不提交 SONIC 权重和 XRRobotKit 二进制。准备以下文件：

```text
models/model_encoder.onnx
models/model_decoder.onnx
<XR_SDK>/xrobotoolkit_sdk*.so
<XR_SDK>/lib/libPXREARobotSDK.so
```

XRRobotKit PC Service 需要安装在 `/opt/apps/roboticsservice`。PICO 客户端的独立
构建说明见 [`pico_client/README.md`](pico_client/README.md)。

## 安装

```bash
git clone https://github.com/shutongliu-dev/sonic-mujoco.git
cd sonic-mujoco

mkdir -p models
# 将 model_encoder.onnx 和 model_decoder.onnx 放入 models/
uv sync --extra sonic --extra teleop --extra recording
.venv/bin/python scripts/setup_xrobotoolkit.py <XR_SDK>
```

视频回传依赖系统 GStreamer：

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly
```

先运行无窗口自检：

```bash
.venv/bin/python scripts/run_sim.py --headless --steps 100
```

H2 自检使用：

```bash
.venv/bin/python scripts/run_sim.py --robot h2 --headless --steps 100
```

## H2 基础支持

H2 入口使用 Unitree 官方 MuJoCo 模型的 31 个电机，并严格按照仓库固定版本
MJCF 的 actuator 顺序返回关节位置、速度和力矩。最后两个主关节就是 H2 原生的
`head_pitch` 与 `head_yaw`，不同于 G1 上额外安装的两轴仿真颈部。模型还定义了
头部单目、64 mm 双目和左右腕部相机。打开交互 viewer：

```bash
.venv/bin/python scripts/run_sim.py --robot h2
```

当前 H2 是可加载、可观测、可按固定 31 维顺序控制的仿真基础，只开放空场景。
Unitree 不同版本的真机示例曾使用不同的腰、腕和踝关节槽位顺序；连接真机前
必须绑定固件版本并做只读关节回读校验，不能只凭向量长度直接下发。
仓库尚未包含与这份 H2 plant 匹配的 SONIC 或 TWIST2 权重，因此不会把 G1 权重
补零后伪装成 H2 policy；H2 也不会冒用 G1 的 DexHand 和矩侨皮肤字段。新增的
相机安装位姿是明确记录的仿真约定，不是真机外参标定。当前入口只是姿态保持
自检；其 PD 控制器没有主动平衡能力，长时间站立与行走需要匹配 H2 的全身
运动策略。
当前固定的官方 H2 MJCF 到左右腕部为止，没有独立可控的手掌或手指关节；
H2 Plus 的双手属于另一套 embodiment，后续必须用单独的模型与关节映射接入。

## GR00T Sweep 仿真评测

先启动监听 `localhost:5550` 的 GR00T PolicyServer，再运行：

```bash
uv sync --extra sonic --extra gr00t
MUJOCO_GL=egl .venv/bin/python scripts/run_gr00t_sweep.py --seed 0
```

脚本使用双目第一视角、SONIC decoder 和 Sweep 成功判定执行 20 秒闭环评测，
结果、首末帧和 `rollout.mp4` 保存在 `results/gr00t_sweep/`。视频左侧是第三人称
视角，右侧是 Policy 的左眼画面。Policy 的 `vest`、`left_arm`、`right_arm`
三个触觉输入直接来自 MuJoCo 逻辑皮肤；未显式加载 profile 时，没有接触会保持
真机同格式的零值基线。
需要模拟真机的异步采样、低幅底噪和设备差异时，可显式加载版本化 profile：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/run_gr00t_sweep.py \
  --seed 0 \
  --tactile-profile configs/tactile/juqiao_g1_sim2real_provisional_v2.json
```

也可以通过 `SONIC_POLICY_DIR` 改变默认模型目录，或用 `--encoder`、`--decoder`
分别指定模型文件。

### OpenPI Sweep 对比

OpenPI SONIC 环境准备完成后，可以用同一场景依次评测 step 0、5000 和 10000：

```bash
./scripts/run_openpi_sweep.sh 0
./scripts/run_openpi_sweep.sh 5000
./scripts/run_openpi_sweep.sh 10000
```

脚本会临时启动 OpenPI websocket 服务和 SONIC 兼容桥，评测结束后自动关闭两者。
结果与视频分别保存在 `results/openpi_step0/`、`results/openpi_step5000/` 和
`results/openpi_step10000/`。

## PICO 遥操作

1. 将 PICO 与 Ubuntu 主机接入同一局域网。
2. 在 PICO 客户端中连接 Ubuntu 主机 IP。
3. 在 PICO 客户端打开 `Head`、`Controller` 和 `Send`。只有全身遥操时才需要
   额外打开 `Body`。
4. 在主机上启动 Sweep 场景：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene sweep \
  --pico-device TestDevice
```

`--pico-device` 使用 XRRobotKit 显示的设备名，并用于发送手柄触觉。也可以通过
环境变量 `SONIC_PICO_DEVICE` 设置。只在 PICO 中看画面时可使用 EGL：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/run_pico_teleop.py \
  --scene sweep --headless --pico-device TestDevice
```

`--headless` 只关闭主机 viewer，不会关闭 PICO 画面。`--no-pico-video` 用于关闭
实时回传；`--no-record-video` 关闭数据集 RGB、深度和法线录制，两者互不影响。

Chair lean 场景使用相同操作流程：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene chair_lean \
  --pico-device TestDevice
```

Bucket carry 场景也使用相同操作流程：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene bucket_carry \
  --pico-device TestDevice
```

动态装载与重心感知场景使用：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene basket_loading \
  --pico-device TestDevice
```

录制开始后，六件质量、摩擦和装载顺序随机的物体会每隔 5 秒依次装入篮筐，
用于产生逐步增载、左右偏载和重心迁移。场景 reset 会同步恢复空篮筐与待装物体。

柔性玩偶搬运场景使用：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene plush_carry \
  --pico-device TestDevice
```

肘部或肩部开门场景使用：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene door_elbow \
  --pico-device TestDevice
```

### 手柄操作

| 操作 | 功能 |
| --- | --- |
| `A+B+X+Y` | 启动或停止控制 |
| `A+X` | 在待机和全身 POSE 遥操之间切换 |
| 按住左菜单键 | 暂停动作跟随，松开恢复 |
| 左 `grip+A` | 开始或结束当前 episode |
| 左 `grip+B` | 放弃当前 episode |
| 左 `grip+X` | 重置场景、SONIC history 和控制器 |

### 五指灵巧手遥操

`run_pico_teleop.py` 默认同时启用 G1 双侧五指灵巧手。手臂和全身仍由 SONIC 的
29-DoF 动作控制，双手使用独立的 40-DoF 目标和 PD 控制，因此不会改变 SONIC
模型的输入输出维度。

- PICO 裸手追踪可用时，程序读取 OpenXR 的 26 个手部关节，将腕部、拇指、
  食指、中指、无名指和小指完整映射到每侧 20 个关节；
- retargeting 使用 0.2 的低通系数，减少指尖接触时由追踪噪声引起的抖动；
- 使用手柄或某只手暂时丢失光学追踪时，扳机控制食指，`grip` 控制中指，拇指
  自动配合形成捏取或包络抓握；
- 如需关闭手指控制，可加 `--no-dexhand`。

外部 ZMQ 遥操同样支持五指灵巧手。protocol-v3 消息可以直接增加
`hand_joint_pos`（`N × 40`，顺序见 `DEXHAND_JOINT_NAMES`），也可以同时增加
`left_hand_joints` 和 `right_hand_joints`（各为 `N × 25 × 3/7` 或
`N × 26 × 3/7`），由接收端完成相同的 retargeting。旧消息不含这些字段时仍可
正常解码。

### 颈部与视角跟随

PICO 遥操默认启用独立的颈部 yaw/pitch 控制。左右转头和抬头、低头会驱动
MuJoCo 中的两个颈部关节，`head_camera` 及左右目相机固定在可动头部上，因此
主机画面和 PICO 第一视角会同步转动。颈部目标不进入 SONIC 的 29-DoF 动作，
不会改变原策略模型；如需固定视角，可加 `--no-neck`。颈部几何、串联刚体和
惯量来自 [open_2DOF_robot_head](https://github.com/co1one/open_2DOF_robot_head)，
不再使用两个关节共用同一刚体的示意结构。

### Project SuperDex DG5F 后端

项目提供可选的 [Project SuperDex](https://projectsuperdex.com/) 后端，将同一组
PICO 五指目标映射到官方 Tesollo DG5F long 左右手。SuperDex 1.0 固定使用
Python 3.12，因此安装脚本会建立独立环境，不会改变当前 MuJoCo/PICO 环境：

```bash
.venv/bin/python scripts/setup_superdex.py
```

先在一个终端启动 SuperDex 双手。`--demo` 会在尚未收到 PICO 数据时循环做抓握
动作，便于确认模型、关节和控制器均正常：

```bash
.venv-superdex/bin/python scripts/run_superdex_hands.py --demo
```

然后选择一种输入方式。只测试 PICO 裸手/手柄到 DG5F 的链路：

```bash
.venv/bin/python scripts/run_pico_superdex_hands.py
```

或者在正常 MuJoCo G1 遥操的同时镜像双手目标到 SuperDex：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene sweep \
  --pico-device TestDevice \
  --superdex-hand-endpoint tcp://127.0.0.1:5570
```

两个进程使用只保留最新帧的本机 ZMQ 协议解耦，MuJoCo/PICO 控制保持 50 Hz，
SuperDex DG5F 接触仿真保持 200 Hz；任意一边退出都不会破坏另一边的运行环境。
当前后端先覆盖官方 DG5F 双手与手势控制，完整 G1 刚体仍由 MuJoCo 负责。DG5F
资产沿用 Project SuperDex 仓库中的 Tesollo 仿真/研究许可，分发或发布结果时需
保留其版权与引用信息。

### SIMPLE 任务与丰富场景

项目可以通过隔离桥接直接使用相邻的 [SIMPLE](https://github.com/shutongliu-dev/SIMPLE)
仓库。SIMPLE 保持自己的 `.venv`、Isaac Sim 和资产目录；本仓库只负责解析任务、
构造官方命令和保存输出，因此不会修改 SIMPLE，也不会与当前 MuJoCo 环境的依赖
发生冲突。默认目录是 `../SIMPLE`，也可用 `--simple-root` 或 `SIMPLE_ROOT` 指定。

先查看可用的 G1 遥操任务以及 50 个 HSSD 场景的下载状态：

```bash
.venv/bin/python scripts/run_simple_teleop.py --list
```

例如启动弯腰抓取、开烤箱或推办公椅：

```bash
.venv/bin/python scripts/run_simple_teleop.py bend_pick
.venv/bin/python scripts/run_simple_teleop.py open_oven
.venv/bin/python scripts/run_simple_teleop.py push_chair
```

SIMPLE 的实时遥操使用它自己的 G1、Dex3、PICO 与 whole-body controller。启动后
等待机器人落地，同时按左手柄 Menu 和右手柄食指扳机进入控制；左摇杆移动、右
摇杆转向，食指扳机控制手，`X/Y` 控制蹲下/站起，两侧中指扳机一起按下可重置。
这条入口目前不复用本仓库新增的 40-DoF 五指手和虚拟颈部。

需要采集数据时增加 `--record`，默认输出到 `records/simple/`：

```bash
.venv/bin/python scripts/run_simple_teleop.py bend_pick \
  --record --num-episodes 10
```

SIMPLE 的实时 MuJoCo 入口使用任务对应的简化物理场景；完整 HSSD 室内 USD 场景
用于 Isaac Sim 回放与渲染，并不是当前实时遥操窗口里的物理房间。场景可以按需
下载，录制完成后再用官方 MuJoCo/Isaac 双仿真入口恢复同一 episode 的物体、位姿
和房间配置：

```bash
.venv/bin/python scripts/manage_simple_scenes.py --download scene4
.venv/bin/python scripts/render_simple_episode.py records/simple/<dataset> \
  --task bend_pick
```

增加 `--record` 会把 Isaac 渲染图像与原始状态/动作重新导出到
`records/simple_isaac/`。HSSD 和任务资产可能有各自许可证，发布数据或图像前应按
SIMPLE 的资产清单逐项确认。

## 真实实验室扫描

实验性的扫描场景入口采用与 VLK 相同的职责划分：MuJoCo 只运行 G1、任务物体、
碰撞和控制，Nerfstudio 进程只从训练好的 3D Gaussian Splatting 重建中渲染相机
画面。两者通过机器人运动和 PICO 头显的相对 6DoF 位姿连接，不会用手写 XML
代替实验室外观。

先在 Nerfstudio 环境启动扫描渲染器：

```bash
../nerfstudio/.pixi/envs/default/bin/python scripts/run_scan_renderer.py \
  --config ../lab-reconstruction/outputs/lab_first_pass/splatfacto/2026-08-29/config.yml \
  --geometry-summary ../lab-reconstruction/geometry_from_scan/gs_depth/lab_scan_points.json \
  --anchor-image frame_00138.jpg
```

再启动带五指手、虚拟颈部和扫描画面的 PICO 遥操：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene lab_scan \
  --scan-collision-manifest sonic_mujoco/assets/reconstruction/lab_scan_collision.json \
  --scan-renderer-endpoint tcp://127.0.0.1:8765
```

当前工作区也可以用一个入口同时管理两个进程：

```bash
.venv/bin/python scripts/run_lab_scan_teleop.py
```

真实实验室的 manifest 和凸包位于本机
`sonic_mujoco/assets/reconstruction/`，该目录下的 JSON/OBJ 被 Git 忽略，不会随
公开仓库上传。也可以用 `--collision-manifest` 或环境变量
`SONIC_RECONSTRUCTION_COLLISION` 指向仓库外的私有资产目录。代码测试使用独立的
合成 fixture，不依赖或泄露真实实验室布局。

要试用从扫描表面生成的实验性碰撞网格，显式增加：

```bash
.venv/bin/python scripts/run_lab_scan_teleop.py --scan-collision mesh
```

默认 `auto` 目前仍使用较保守的 `boxes`，避免在没有人工走查前悄悄改变已有实验；
`mesh` 会严格加载 243 个独立凸块，任一文件、坐标约定或哈希不一致都会直接拒绝
启动。`--scan-collision boxes` 可明确固定旧后端。

启动前可先做动态视角自检：

```bash
.venv/bin/python scripts/run_lab_scan_teleop.py --check
```

该检查会真实渲染左右眼以及侧移 10 cm 后的视点；只有三张图存在足够像素差异时
才会通过，固定照片或重复左右眼不会再被误判为可用的 3D 重建。

该入口向 PICO 发送 2560×720 side-by-side 画面：左右眼分别按 64 mm 基线渲染
3DGS，并各自完成 MuJoCo 机器人前景的深度遮挡。头显转动、侧移和前后探头都会
改变两眼相机，机器人行走时视点基座也会同步移动。默认打开的
`Reconstructed Lab Camera` 桌面窗口显示 PICO 左眼；原生 MuJoCo viewer 只用于
检查透明碰撞体，本身不会显示 3DGS。需要同时检查碰撞几何时可增加
`--show-physics-viewer`；完全关闭桌面合成窗口可增加 `--no-desktop-video`。
如果程序在 5 秒内没有收到头显姿态，两眼画面会显示橙色 `HEAD OFF`，终端也会
明确提示在 XRRobotKit 中打开 `Head` 和 `Send`；该诊断标记不会写入训练图像。

当前手机 RGB 视频没有提供 LiDAR 公制深度，`--scan-units-per-meter` 因此默认使用
场景元数据中的估计值 `0.198`；它只影响机器人移动与扫描坐标的比例，不改变扫描
外观。该估计来自扫描相机与仿真相机的离地高度对齐，配置同时明确记录
`estimated` 状态、COLMAP 稀疏 SfM 深度来源以及没有实测公制参照。命令行参数只
用于有意覆盖场景值，避免物理碰撞和视觉渲染各自维护一份比例。当前前景
合成适合验证第一视角和手部遥操。实验室地面和 88 个透明静态碰撞盒已经从
3DGS 深度与 Gaussian 中心自动提取；另外提供了从 140 视角过滤 TSDF 表面生成的
243 个 5 cm 占据凸块，作为实验性 `mesh` 后端。独立验收确认 G1 初始位置与全部
140 个训练/评估相机中心均在自由空间，1000 个仿真步数值稳定。它的表面覆盖率
仍只有约 43.6%，自由空间侵入率约 4.75%，因此不能视为测量级完整碰撞。
两种碰撞后端都只是透明物理代理，并不是实验室的视觉网格；真实外观始终由
3DGS 新视角渲染器生成。
动态任务物体还需要独立的视觉网格与碰撞网格，不能直接把扫描中烘焙的静态物体
当成可抓取物体。

本机的实验性凸块可用同一 Nerfstudio 环境复现：

```bash
../nerfstudio/.pixi/envs/default/bin/python scripts/build_scan_collision_mesh.py \
  --input ../lab-reconstruction/geometry_from_scan/tsdf_experiments/filtered140/lab_scan_tsdf.ply \
  --geometry-summary ../lab-reconstruction/geometry_from_scan/tsdf_experiments/filtered140/lab_scan_tsdf.json \
  --floor-plane -0.10184300253341096 -0.08447168429490631 \
  0.991207615682689 0.2586876453643408 \
  --output sonic_mujoco/assets/reconstruction/lab_scan_collision_mesh_v2.json \
  --voxel-size-m 0.05 --maximum-hulls 256 \
  --maximum-hulls-per-partition 1 --partition-fraction 0.95 \
  --anchor-clearance-m 0.75 --overwrite
```

输出 manifest 会记录估算尺度来源、构建参数、结构验证结果以及每个 OBJ 的哈希；
主场景 manifest 只以相对路径引用它，因此稳定的 boxes 回退不会被覆盖。

扫描场景的数据链路借鉴 VLK/LEGS 的物理—视觉解耦方式。MuJoCo 状态、动作和
触觉以 50 Hz 写入数据集。录制相机独立于 PICO 显示，默认以 10 Hz 同步采集四路
RGB、深度和法线；两次采集之间保持最近一组图像，同时记录采集时间、帧龄和索引。
物理接触仍按子步汇总。每路保存与图像配对的 MuJoCo 相机位姿和 3DGS 外观位姿，
便于重放。PICO 自由头部观察位姿及 `REC` 提示不会混进机器人训练视角。

`grip` 指手柄中间由中指扣动的握持扳机。一次常用的连续采集流程是：

1. 连接主机后按 `A+B+X+Y` 解锁，再按 `A+X` 进入 POSE 遥操；
2. 调整好人与 G1 的位置，按左 `grip+A` 开始录制；
3. 完成动作后再次按左 `grip+A`，保存当前 episode；
4. 按左 `grip+X` 重置场景，再重复步骤 2–3；
5. 当前动作无效时按左 `grip+B`，直接丢弃该 episode。

项目不使用摇杆规划行走。操作者的迈步、转身、下蹲和手臂动作直接构成 G1 的
参考动作。录制中使用 `A+X` 不会结束 episode，但暂停片段通常不适合作为训练
数据。

## 数据采集

录制默认写入 `records/<session>/`：

```text
records/<session>/
├── meta/                 # schema、episode 索引、body 与触觉布局
├── data/chunk-000/       # Parquet 控制帧
├── videos/chunk-000/     # 双目与双腕四路 RGB MP4
├── geometry/            # 按真实采集帧分块的深度、法线与有效掩码
└── previews/             # episode 与接触数据预览
```

每个样本包含 MuJoCo `qpos/qvel/ctrl`、PICO 姿态、SONIC token、策略动作、
五指灵巧手与颈部目标、手柄输入和最多 16 组重要接触。扫描场景还包含与 RGB
严格配对的物理相机位姿和 3DGS 相机位姿；`meta/info.json` 记录重建 checkpoint、
anchor、相机内参、重建比例和视觉来源。接触字段包括机器人部位、场景物体、
世界坐标、法向/切向力、法向冲量和物理采样次数。启用 G1 环境时还会生成
`meta/tactile_layout.json`，并逐帧保存下面两层触觉：

- 真机兼容层：`observation.tactile_vest`、`observation.tactile_left_arm`、
  `observation.tactile_right_arm`，均为 `uint8[256]`；
- 物理侧车：每个有效 taxel 的牛顿力、牛顿秒冲量、接触物体、采样次数和未映射
  接触诊断，供标定、回放和重新生成真机兼容值。

启用 Sim2Real profile 后，每条 `meta/episodes.jsonl` 还会保存该 episode 实际抽到
的 seed、设备采样率和参数哈希；session 级 metadata 只保留静态 profile，避免连续
采集多条 episode 时把第一条的随机化参数误写到后续数据。

一次进程可连续采集多条 episode。结束一条后终端会输出文件位置、帧数和时长；
保存期间 PICO 持续收到最后一帧，画面不会因编码和落盘而断流。

### 录制后的离线检查

不用启动仿真或连接 PICO，就可以读取一行真实保存的观测，检查四路 RGB、深度、
法线与触觉，并扫描整条 episode 的时间戳：

```bash
.venv/bin/python scripts/inspect_episode.py records/<session> \
  --episode 0 --row 17 --output results/episode_inspection
```

输出 `contact_sheet.png` 和 `audit.json`。工具按视频行号读取 RGB，再按
`sample_index` 读取同一时刻的几何样本，保持帧不会被误当作新采集帧。
报告给出时间戳一致性、最大帧龄、实际采集次数、所选帧触觉摘要和任务结果；
旧版 RGB-only 记录会明确显示没有深度/法线。输入数据集只读，输出不能放进它内部。
该工具检查所有行的时序，但只解码选中行的媒体，**不等于整套数据完整性认证或
物理回放**。RGB 是有损 H.264，深度/法线侧车是无损数组。

录制器拒绝重复开始、非有限数值、时间倒退/控制行跳帧，以及 session 中途变更字段或维度。
视频写入发生异常或被中断后，片段不能再作为有效 episode 保存；运行入口会丢弃
当前失败片段，保留此前成功保存的片段。元数据先准备再替换；这提供进程内错误
保护，不宣称能够抵御断电或存储设备损坏。终端会明确提示丢弃，避免误以为保存成功。
如果元数据回滚也失败，错误中会给出保留下来的备份目录；恢复元数据前不要继续使用该数据集。

## 相机与触觉的同步约定

`run_pico_teleop.py` 默认录制 `ego_view_left/right` 和 `wrist_view_left/right`。
四路均为 640×480，使用同一仿真时刻；普通场景的 PICO 左右眼分别渲染，不再复制
同一张单目图。头部视角随 G1 颈部与身体移动，腕部视角随手腕移动。重建场景中，
PICO 的自由 6DoF 观察继续独立于记录的机器人相机。

可用 `--camera-fps 30` 或 `--camera-fps 50` 提高采集频率，也可用
`--camera-width 320 --camera-height 240` 降低负载。这些是仿真时间频率，不保证
同样的墙钟实时速度；四路深度/法线的渲染、压缩和写盘开销明显高于单路 RGB。
默认 10 Hz 是工程折中，不等同于论文的视觉采样频率。

RGB 以数据行频率编码为 MP4；保持的重复帧仍指向同一个几何样本。
`observation.camera.<view>.timestamp/age_s/sample_index/updated` 记录新旧帧关系，
其 `camera_to_world` 属于实际采集时刻而不是保持行的新时刻。
深度与法线保存在独立 NPZ 分块中，不伪装成标准 LeRobot 视频字段：

- `meta/cameras.json`：内参、安装位姿、坐标约定和几何文件索引规则；
- `geometry/episode_000000/chunk_000000.npz`：每块最多 8 次实际采集，
  键如 `ego_view_left.depth_m`、`ego_view_left.normals_camera`；
- 用 `sample_index // 8` 找块，`sample_index % 8` 找块内帧；
- 深度是米制光轴 Z，扫描场景则是估算米制；法线使用光学坐标
  X 向右、Y 向下、Z 向前，朝向相机。无效深度/法线置零并附独立有效掩码。

法线是深度有限差分结果，不是接触法线传感器。双目基线、视场角、腕部安装和
深度噪声尚未做真机标定；当前记录用于联调，不能据此声称消除了视觉 Sim2Real gap。

## 接触触觉

- G1 表面布置 624 个不参与碰撞的逻辑 taxel：背心 112 点、左右独立臂套各
  256 点；taxel 不增加质量或碰撞几何；
- 每个 taxel 按它到实际 G1 碰撞壳的距离使用 3–6 cm 自适应覆盖门限；衣片之外
  的同 body 接触记为未映射，不生成假读数；
- MuJoCo 每个物理子步取得接触力，并按距离核分配到相邻 taxel；分配权重和为 1，
  保持总力与总冲量；
- v2 profile 在接触投影之后、非线性换算之前，以同一衣片内的一环和二环邻接核
  模拟布料传力；空间核仍保持总力，不跨设备或衣片传播；
- 手臂压到胸背等机器人自接触会在相接的两侧衣片分别产生读数；
- 真机兼容层沿用 `vest/left_arm/right_arm` 三设备与原始通道顺序，未接线的
  144 个 vest 槽始终为零；
- 三路设备以相互独立的约 14 Hz 采样，50 Hz 数据集使用 sample-and-hold，模拟
  真机数据中大量相邻重复帧；袖套轴向、翻转和接缝偏移可单独标定；
- 左右手臂接触分别反馈到对应手柄；
- 胸腹、腰和骨盆接触同时反馈到两个手柄；
- 突然碰撞、持续按压和切向滑动使用不同的强度、时长和频率；
- 腿脚接触不触发手柄，避免站立和行走产生持续噪声。

真机目前没有可靠的牛顿到 8-bit 压阻读数标定曲线，因此默认换算只用于联调，
metadata 会明确标记为 provisional。训练可以直接使用与真机同形状的 8-bit 字段，
同时保留物理量侧车；完成砝码或测力台标定后只需替换 gain/offset/gamma，不必
重新仿真。

仓库默认推荐
`configs/tactile/juqiao_g1_sim2real_provisional_v2.json`。它保留三路独立约
14 Hz 采样、100 ms stale、sample-and-hold、8% 通道固定灵敏度差异和每 episode
额外 ±8% gain 域随机化，并增加保守的一环/二环衣片传力、随机 source phase、
AR(1) 相关底噪、带符号的慢速 OU 漂移，以及非零的上升/回落时间常数。这些参数
来自匿名时序与空间统计，只是未配对的 Sim2Real 先验，不是厂商标定结果。

v2 的 `observation_delivery` 有意保持为 `null`。当前真机数据没有同时保存 source
时间、接收时间、序号和 updated 标志，无法把传感器采样与 50 Hz 发布链路可靠
解耦；再拟合一层 hold-gap 会重复计算已经由约 14 Hz source clock 产生的保持。
袖套的轴向、接缝方向和通道偏移也尚未完成真机配准，左右袖套都需要按已知位置
逐点按压后才能确定 mount。N→count 的 gain/offset/gamma、空间扩散和传感器动态
均明确标记为 provisional；完成同步测力与点按标定后，应替换对应参数而不是将
现有任务边际分布当成标定曲线。

旧的 `configs/tactile/juqiao_g1_sim2real_provisional_v1.json` 继续受支持，用于复现
原有无空间扩散、白噪声和固定 phase 的数据链路。PICO 遥操用一个参数启用推荐的
v2 profile：

```bash
.venv/bin/python scripts/run_pico_teleop.py \
  --scene sweep \
  --tactile-profile configs/tactile/juqiao_g1_sim2real_provisional_v2.json
```

可以用真实或仿真 LeRobot 数据集运行匿名分布审计：

```bash
.venv/bin/python scripts/audit_tactile_sim2real.py \
  /path/to/lerobot_dataset \
  --label real_clean \
  --output results/tactile_sim2real/real_clean.json
```

报告包含重复帧率、可观察更新率下界、零包率、有效通道数、raw-count 分位数和
静默包逐通道统计；不包含原始触觉帧或输入绝对路径，并默认跳过 metadata 中标记
为 discarded 的 episode。静默统计会分别报告“包含全零包”和“仅非零包”两组，
因为全零既可能是无接触，也可能是 stale 补零；两组都可能混合衣物预载、基线
漂移和未标注接触，所以只用于配置 count 域噪声，不能用于拟合 N/kPa 到 count
的传递曲线。该曲线需要同步记录测力台读数、接触面积和原始三路触觉包后再标定。
若直接审计没有 `meta/info.json` 的单个 Parquet 文件，必须显式传入 `--fps`。

可用一个沿 G1 胸前滑动的 8 N 小箱子运行独立验收：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/demo_tactile_skin.py
```

结果写入 `results/tactile_skin_mujoco_demo.png` 和 `.gif`。演示会自动检查空载零值、
热点跟随、定位误差、力值和冲量守恒，并确认逻辑 taxel 没有创建碰撞 geom。

## Uni-VLaT 任务场景

参考 [Uni-VLaT 项目](https://ggkiller-air.github.io/Uni-VLaT/) 和
[论文附录 C](https://arxiv.org/html/2609.35450v2#A3)，提供五个独立仿真近似。
论文报告的是真机任务；这里不是作者发布的仿真资产，也不是扫描重建或复现成功率。
尺寸、质量、接触阈值为本地工程估计，人类交互者是有碰撞的运动学代理。

| 场景参数 | 内容 | 已实现的任务检查 |
| --- | --- | --- |
| `univlat_sweep` | 木桌、蓝色分界线、五种物体，桌高随机 0.75–1.05 m | 全部越线、无掉落、过大桌面碰撞、60 s 超时 |
| `univlat_basket` | 开口篮筐、交接支撑、碗/瓶/配重及备料桌 | 双臂接稳后随机装载、掉落/倾覆/位移、归还释放、60 s 超时 |
| `back_tap` | 行走区域、人体代理和背部接触器 | 1–4 次接触/释放、1 s 内启停、30 s 超时 |
| `human_hug` | 有接触顺应性的人体代理 | 双臂接触、保持、退出释放、过大接触、30 s 超时 |
| `composed_cleanup` | 高桌、低桌、垃圾、开口收纳箱、垃圾桶 | 按序扫入/抱起/搬运，放进桶才成功，180 s 超时 |

每个场景沿用 G1 + DEXHAND + 两轴颈部、624 点触觉以及四路视觉。
箱子与篮筐由独立壁面碰撞体组成，内部可装物，不使用填满内部的凸包。
人体代理/装载器只驱动环境，不替操作者控制机器人；不具备生物力学人体或布料精度。
任务状态按 `task.*` 写入控制帧，具体判据保存在 session 的 `task_protocol`。

先看布局，不执行策略、不开 PICO 服务：

```bash
.venv/bin/python scripts/run_sim.py --scene composed_cleanup --paused --seed 4
MUJOCO_GL=egl .venv/bin/python scripts/preview_univlat.py
```

预览生成在 `results/univlat/`，含五场景总览及每个场景的 RGB/深度/法线。
实机 PICO 遥操入口仍然相同，推荐先试扫桌：

```bash
.venv/bin/python scripts/run_pico_teleop.py --scene univlat_sweep \
  --tactile-profile configs/tactile/juqiao_g1_sim2real_provisional_v2.json
```

默认开始录制时才启动任务计时和环境交互，可先进入 POSE 调整姿态而不消耗时限。
想不录制直接试玩交互，可加 `--task-start pose`。重置场景会恢复未开始状态，
或按 `pose` 设置立即重新开始。成功/失败只报告，
不自动保存或关闭 episode。`run_sim.py --start-task` 可启动交互，但其简单姿态保持
不具备全身平衡或任务能力，不能用它的结果评估 SONIC 或 Uni-VLaT。
当前测试验证加载、短程物理步进、接触到触觉、任务判据与录制往返；
还需逐场景 PICO 人工完成任务验收，尤其是抱箱、拥抱和归还篮筐。

旧的 `sweep`、`basket_loading` 场景保持原语义，不被论文近似场景覆盖。

`--scene-seed 42` 让每次重置保持相同的随机布置；省略则重新抽样。Uni-VLaT 环境
会保存实际 seed（包括自动生成的 seed）、初始物体位姿及任务相关随机参数，
连同录制开始时的任务阶段、计时与交互状态一起写入
`meta/episodes.jsonl` 的 `task_context`。`task_outcome` 区分成功、失败和未完成，
不把用户手动结束录制当成任务成功。seed 可复现同版本初始场景；它不替代中途恢复
完整动力学状态所需的速度、控制器历史和接触求解器状态。

## Sweep 场景

Sweep 场景按真实采集桌面布置：木纹桌面由蓝色胶带纵向分区，杯子、齿轮、
白卡、纸板和小鸭位于同一侧。桌子是带质量、摩擦和顺应性的自由刚体，明显
碰撞会产生移动与晃动；任务物体的质量、摩擦和初始位姿会在 reset 时小范围
随机化。

## Chair lean 场景

Chair lean 按参考椅的 56 cm 宽、54 cm 深、35 cm 座高和 66 cm 总高搭建。椅子
是约 11 kg 的自由刚体，脚垫具有较高摩擦，轻微碰撞会产生真实反作用，强碰撞
仍可能移动或倾倒。椅背采用柔顺接触模型；reset 时小范围随机化椅背宽高、倾角、
软硬度、椅子质量与摩擦，以及 G1 的初始距离、朝向和左右偏移。外观使用带织物
与木腿细节的扫描级扶手椅网格，视觉表面与稳定的简化碰撞体相互独立。

场景不自动判定成功或结束 episode，操作者根据稳定倚靠状态控制录制。

## Bucket carry 场景

Bucket carry 使用两张相同的 90 × 60 × 76 cm 折叠桌，水桶从起始桌搬到带蓝色
区域的目标桌。桌子和水桶都是自由刚体，碰撞会产生反作用、晃动或位移。水桶按
常见 18.9 L / 20 L 饮水机桶的比例搭建，外观与由圆柱、椭球组成的稳定碰撞体分离。

reset 会随机化桶的直径、高度、5–10 kg 验证重量、重心偏移、摩擦和初始姿态，
以及两张桌子的质量与摩擦。待遥操作稳定后，可将重量范围校准到满桶约 19–20 kg。
胸部、上臂和前臂的接触会进入数据集，并按左右接触部位映射为 PICO 手柄震动。
场景不包含自动成功判定。

水桶和两张折叠桌使用 CC BY 4.0 的真实网格作为外观；稳定的圆柱/椭球与桌面/桌腿
碰撞体继续负责动力学和触觉采集，因此更换外观不会改变控制接口或数据格式。

## Plush carry 场景

Plush carry 使用两张动态折叠桌和开阔搬运路线，两桌中心间距约 4.92 m，是 Bucket
carry 默认间距的三倍。G1 从桌子长边一侧接近约 74 cm 高的牛油果软玩偶。外观采用
[melsto 的 Avocado Plush Toy](https://sketchfab.com/3d-models/avocado-plush-toy-567c19c1347542c3a63eabc8675257ce)
官方模型（CC BY 4.0）；碰撞主体使用尺寸对齐的 MuJoCo 三维柔性体，115 个表面节点
负责碰撞和局部压缩，形状约束使其在释放后回弹，并避免软体节点长期漂移。

reset 会随机化玩偶 0.9–2.0 kg 的重量、软硬度、表面摩擦和初始姿态。柔性体与
胸部、上臂和前臂的接触会进入同一套 episode 数据和 PICO 手柄反馈。高精度网格只
负责显示，简化柔性体负责稳定接触；场景不自动判定成功。

## Door elbow 场景

Door elbow 使用 90 × 204 cm、约 20–30 kg 的室内木门。门后是可进入的 4.1 × 4.9 m
房间，带木地板、墙体、顶灯和具有碰撞的家具；门口至房间内部保留 1.6 m 宽的通道。
门板通过带阻尼、静摩擦和 100° 限位的真实铰链连接门框；reset 会随机化门重、门轴
阻尼、开启阻力、门面摩擦、1–10° 初始开角，以及 G1 的初始距离、身体夹角和左右
执行侧。肩部、肘部、前臂和上臂接触沿用统一的接触记录与 PICO 触觉反馈，不自动
判定任务成功。

门板使用带木纹、面板与门把手细节的 CC BY 4.0 室内门网格；门轴、碰撞、随机化和
数据格式仍由项目内的简化物理模型负责。第三方视觉资产及转换说明见
[`sonic_mujoco/assets/mujoco/meshes/SOURCES.md`](sonic_mujoco/assets/mujoco/meshes/SOURCES.md)。

## 代码结构

```text
sonic_mujoco/
├── controllers/sonic/        # observation、encoder 和 decoder
├── envs/mujoco/g1/           # G1 环境、PD 控制与任务场景
├── superdex/                 # DG5F 映射、低延迟协议和 SuperDex 运行时
├── teleop/                    # PICO 姿态、控制状态、视频和触觉
├── simple_bridge.py           # 外部 SIMPLE 任务、场景和回放桥接
├── scan.py                    # MuJoCo 与 3DGS 相机对齐和渲染协议
├── cameras.py                 # 同步机器人相机、轴向深度与法线
├── camera_recording.py        # 四路 RGB 和无损几何分块
├── episode_reader.py          # 离线读取观测与时序审计
├── hand.py                   # 与仿真后端无关的五指关节约定
├── contact.py                 # 物理子步接触汇总
├── tactile.py                 # 逻辑 taxel 布局与守恒接触映射
├── tactile_skin.py            # 真机 3×256 皮肤布局、量化与异步采样
└── recording.py               # episode、视频和预览写入
scripts/
├── run_sim.py                 # 最小仿真自检
├── run_pico_teleop.py         # 遥操作与采集入口
├── run_superdex_hands.py      # SuperDex DG5F 物理运行时
├── run_pico_superdex_hands.py # 独立 PICO → SuperDex 入口
├── setup_superdex.py          # 隔离安装 SuperDex 1.0 与 DG5F 资产
├── run_simple_teleop.py       # SIMPLE G1 实时遥操入口
├── manage_simple_scenes.py    # SIMPLE HSSD 场景列表与按需下载
├── render_simple_episode.py   # SIMPLE HSSD/Isaac episode 回放
├── run_scan_renderer.py       # 隔离的 Nerfstudio 3DGS 渲染服务
├── run_lab_scan_teleop.py     # 扫描渲染与 PICO 遥操的一键入口
├── demo_tactile_skin.py       # 箱子按压逻辑皮肤验收与可视化
├── pico_video_bridge.py       # XRRobotKit 视频协议与编码
└── setup_xrobotoolkit.py      # 安装本地 XR SDK
```

四元数统一使用 `wxyz`。硬件/MuJoCo joint order 与 SONIC 内部 order 的转换均在
边界处显式完成。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests
uvx ruff check .
```

测试覆盖 G1 模型、PD 控制、SONIC observation/action、PICO 协议、姿态转换、
视频回传、连续录制、触觉映射、DG5F 关节映射和 SuperDex 传输协议。
依赖外部 C++/TensorRT 参考实现的等价性测试会在未配置相应环境变量时自动
跳过。
