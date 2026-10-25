# PICO 遥操 Isaac 场景

保留裸手与手柄两种模式，沿用现有 XRoboToolkit PICO 客户端。默认运行厨房的
`breakfast_serving` 发布；也可选择 catalog 中已实现、使用 FR3 + SHARPA 的任务。
这是人工操作入口，直接生成完整关节目标并交给原生 Isaac 后端，不需要 HTTP 服务。

## 启动

先按仓库 README 配好 CPU 环境、Isaac 和数据包。电脑启动 XRoboToolkit PC Service，
头显连接同一局域网中的电脑，在客户端开启头部与所选输入模式的追踪。
视频进程使用系统 Python 的 PyGObject、GStreamer 和 x264/x265 编码器。
Ubuntu 对应包为 `python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-plugins-base
gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly`。

在仓库目录运行：

```bash
export CUDA_VISIBLE_DEVICES=<GPU UUID>
export PICO_SDK_LIBRARY=<XR SDK 目录>/lib/libPXREARobotSDK.so
scripts/teleop/run.sh --mode hands
# 或：
scripts/teleop/run.sh --mode controllers
```

SDK 通过独立 CPU 进程加载 C ABI，不需要 sonic_mujoco 的 Python 3.10 扩展。
Isaac 端的接收线程持续读取 UDP，只保留最新有效帧，避免慢仿真期间缓冲区积压、丢掉新数据。
启动器冻结源码到仓库外的 `../dexbench_runs/`，记录 GPU、数据版本、动作与输入。
可以用 `DEXBENCH_RUNS`、`DEXBENCH_DATA`、`ISAAC_PYTHON_LAUNCHER` 指定本机位置。
`PICO_DEVICE` 可固定头显序列号；未设置时绑定第一个合法发送者。

头显的视频接收设置填写电脑 IP、控制端口 **13579**，选择左右并排双目画面。
默认每眼 640×480，合并为 1280×480，渲染目标 30 FPS；客户端应请求相同比例的尺寸。
编码支持普通 H.264/H.265，不支持 MV-HEVC 多视图扩展。渲染帧率是上限，
编码器重复上一帧不能提升实际场景刷新率。头显自己的视频接收端口也需要可从电脑访问。
输入桥仅监听电脑回环地址 UDP 5557；`--port` 可以更改它。

## 操作

先检查接入链路时可运行 `scripts/teleop/run.sh --sandbox --mode hands --profile`。
这是独立的桌边 FR3/SHARPA 工程场景，桌上有现有资产库的瓶子（直径 6.4 cm、高 19 cm、300 g）和罐子（直径 6.8 cm、高 8.4 cm、160 g），
初始腕部比原先低约 20 cm、前移约 9.5 cm，减少向桌面伸手的距离。
保留资产原有质量、惯量与碰撞，不加载任务裁判；
不是厨房任务的简化验收。保留 0.5 ms 物理步长、20 ms 控制周期、原生接触采样与机器人参数。
此场景显式使用 `cpu_rigid_v1`：原生 PhysX CPU 刚体求解、单线程物理与 GPU RTX 渲染。
单个机器人避免了每个小步的 GPU 调度成本；这不是降低物理频率。已发布任务仍使用各自
原有配置，含粒子、布料或软体的场景不能选择此模式。CPU 模式每次采图使用一次同步
渲染更新，首次创建产品仍预热；移动物体的 RGB、深度和实测位置需要通过同步验证。
会话标记为 `teleop_sandbox`，任务和发布字段为空，场景及初态另存用于复现。

刚体标准场景可显式选择操作会话专用的 CPU 配置，例如：

```bash
scripts/teleop/run.sh --family kitchen --task breakfast_serving --physics-profile cpu_rigid_v1 --no-task-scoring --profile
```

默认 `--physics-profile scene` 仍保留场景原配置。覆盖只作用于内存中的会话副本，
不修改发布文件；会话 `runtime_overrides` 和有效 `scene.json` 记录实际选择。
这属于带配置覆盖的人工遥操，不代表原发布配置的资格验证。
CPU 选项在启动前拒绝带粒子、布料或软体的场景，渲染仍使用 GPU。
不同场景采用前须验证原生接触、轨迹与任务状态读取。

`--no-task-scoring` 用于暂时不需要判断任务完成的遥操练习：显式关闭任务观察器，
但继续运行全部物理小步、碰撞、接触采样、固定物体检查和刚体接触安全保护。
不提供任务成功、任务顺序或任务专属失败判定；控制窗口和会话均标注未评分。
带粒子、约化动力学或依赖任务状态的视觉效果时拒绝关闭，避免改变模拟行为。
默认保持评分开启；此选项不修改发布文件，不作为正式评测配置。

- 开始时暂停。把手放在舒适的位置，裸手张开，然后点击 **Start / Pause** 或按空格。
  首帧建立相对锚点，不把机器人瞬移到人的绝对位置。
- **裸手**：腕部位姿控制机械臂；26 个手关节用于手掌标定和五个指尖目标，
  使用 SHARPA 运动学做每帧最多四次的渐进指尖跟随；不可达目标也允许有界的改善，
  不再因为未达到精确端点容差就丢弃整根手指的更新。这是相对手势近似映射，非人体关节角一一复制。
- **手柄**：按住 grip（握把，阈值 0.5）移动腕部，trigger（扳机）从张手插值到现有捏合预设。
  松开握把保持最后目标；再次握住会重新建立相对锚点。
  窗口显示实际 grip 百分比；`ready — hold grip to move` 表示未握住，尚未发送移动目标。
- **Hands / Controllers** 切换模式并暂停；再次 Start 校准。客户端也要开启对应追踪。
  `hands/controllers unavailable` 表示收到数据但所选输入未被追踪；
  `paused — press Start` 表示对应输入已有效，等待开始。仅打开视频 Listen 不会开始控制。
- 默认主画面为固定 `overview` 双目视角，右上角同时显示 `wrist` 腕部近景。
  PICO 显示画中画；电脑主窗口显示无遮挡全景，独立的 **Wrist camera** 窗口可拖动标题栏移动、拖动边缘缩放。
  桌面浮窗复用已有近景像素，不新增相机渲染；移动桌面浮窗不改变 PICO 小窗位置。
  腕部小窗为单目，两眼共用，不能依赖小窗的双目视差判断深度。
  默认每眼 640×480，小窗 256×192，三路相机在同一次渲染更新中读取，不额外推进物理。
  主画面双目间距 64 mm；操作方向按主画面校准，小窗不改变控制映射。
  `--inset-camera none` 关闭小窗；`--camera wrist --inset-camera none` 使用原有腕部双目。
  `--head-tracked` 可让世界主相机随头显移动，腕部相机不能启用此选项。
- **C / Recenter** 暂停并重新标定操作方向（固定或腕部相机不因头动而转动）；**R / Reset scene** 重置整个场景并暂停。
- 输入超过 150 ms 或所绑定的手暂时无效，立即保持上一关节目标，不使用过期数据。
  裸手：最近一次有效控制输入后的 2 s 内，追踪连续稳定 250 ms 后自动恢复，恢复首帧
  只建立新的手腕起点，不补做断流期间的移动。稳定窗口内手腕相对起点不超过 3 cm / 15°，
  每帧都须新鲜，重复缓存数据不能完成计时。超过 2 s 后仍保持目标；当前单右手控制时，
  右手保持稳定，左手先分开拇指/食指，再捏合保持 250 ms 可恢复。左手必须有有效追踪。
  两只手均绑定机器人时不占用操作手做确认，仍用 Start。
  手柄：保留 300 ms 短断自动接续；较长中断通过松开、重新握住 grip 恢复。
  普通暂停、掉追踪和恢复保留手型标定；**Recalibrate fingers** 才清除手型标定并暂停，
  之后张开手、点 Start。Reset scene 和数据源重连也清除手型标定。
  应用失去焦点、数据源重连、主动暂停、模式切换或 Reset/Recenter 不走自动或手势恢复，需 Start。
  物理仿真在保持目标时仍继续运行。
  如果只是 IK 计算导致本次目标过期，而最新追踪仍有效，则丢弃本次目标、保持上一目标，
  显示 `slow IK` 并继续等待下一次计算；不会把计算慢误判为断流。
- 遥操每帧 IK 使用 80 ms 的协作式计算预算，腕部单次最多分配 25 ms，其余时间分给手指。
  腕部精确求解超时保持上一目标，显示 `IK budget exhausted`。手指用阻尼雅可比步进，
  每个候选均投影到原有关节限位与手型约束，只保留指尖误差下降的步骤；到预算即结束。
  每帧仍应用原有关节速度限制，记录 `finger_target_errors_m`，不宣称到达精确指尖目标。
  预算在求解迭代边界检查，不是实时系统的硬时限。基准任务求解默认不设置该预算，
  原有容差、关节限位、速度限制和 150 ms 输入有效期保持不变。
  手部不可用时，窗口和会话诊断会区分应用失焦、未发送手数据、手未激活及关节追踪无效。
- 达到回合时限或触发刚体接触保护后停止发送动作，需 Reset 才能继续。

单手任务默认使用右手；双手任务绑定同名输入。多个具名机器人需要显式指定，例如：

```bash
scripts/teleop/run.sh --family living_room --task tower_extraction \
  --mode controllers --bind front=right --bind rear=left
```

`--scale 0.5` 可缩小腕部位移；没有绑定的机器人保持原目标。
所有目标受原 URDF 关节限位和增量速度限制，求解失败的关节保持原目标。
不改发布数据、物体位姿、摩擦或共用求解器配置。

## 验证与限制

会话目录包含 `frames.jsonl`（输入、动作、实测关节、暂停原因）、`session.json`、
`left.png`、`right.png`；SDK 日志在上一级 `pico_sdk.log`。
原生后端仍在每个物理子步更新发布中的任务观察器（包括裁判状态及显示状态）。
SDK 日志的 `tracking_gap` 记录超过 150 毫秒的有效输入间隔；Isaac 控制台的
`tracking_expired` 记录过期时的输入年龄、拒收数量及是否还有 UDP 待接收，帮助区分
上游断流与本地接收延迟。同一过期帧只记录一次，避免断开后刷屏。这些诊断不改变过期阈值。
人工入口尚未汇总正式评分和完整物理审计：`task_success`、`physics_valid` 为 null，
`formal_evaluation` 为 false。关节运动和接触保护不等于物理验收或任务成功。

加上 `--profile` 可记录每次循环的渲染、输入映射、原生步进墙钟耗时和接触采样覆盖，
并在 `session.json` 汇总，保存 `final_state.npz` 用于同条件物理状态对比。
计时不包含启动和退出；`--no-video` 只关闭网络编码，仍渲染双目画面。
双目通过后端 `render_many` 同批刷新、随后读取；物理状态与原生时钟检查仍保留。

CPU 回归入口：

```bash
.venv/bin/python -B -m pytest tests/test_pico_teleop.py tests/test_pico_video.py
```

`scripts.validation.pico_teleop_smoke` 是持续保留的原生诊断工具，适用于当前
`kitchen/breakfast_serving/v3` 单手发布。它发送合成 UDP 输入、驱动真实 Isaac、
检查运动及丢失追踪后的保持，并检查双目图像；不依赖头显，也不证明真实设备联调通过。
从完整源码快照启动，设置数据、GPU 和 Isaac 环境后运行：

```bash
scripts/isaac_python.sh -B -m scripts.validation.pico_teleop_smoke \
  --mode hands --headless --no-video --output <新的证据目录>
# 加 --sandbox 验证桌边双物品链路。
# 再加 --stress 注入不可达的快速跳变，验证 IK 预算及断流保持（自动开启计时）。
# 改成 --mode controllers 检查手柄；去掉 --headless 检查窗口初始化。
```

2026-10-21 在 RTX 4090 / Isaac 6.1 / data-v3 上，两种模式均通过上述原生检查，
手柄模式也通过窗口运行。双目合批后，同条件裸手 60 步（1.2 秒仿真）的单组对照：

| 墙钟耗时 | 原来逐眼刷新 | 合批刷新 |
| --- | ---: | ---: |
| 双目渲染 | 5.76 s | 2.88 s |
| 原生步进、读回和检查 | 26.20 s | 26.23 s |
| 完整循环 | 32.12 s | 29.26 s |

两次动作、每步实测关节、完整接触采样记录、最终物理数组和任务观察器状态完全一致；
物理步长仍为 0.0005 s，每个控制步采样 40 次。证据索引为
`outputs/pico_performance_comparison.json`（本机输出，不纳入 Git）。
共用 RGB/RGB-D 路径也通过原生移动物体同步检查，包含旧位姿负例拒绝。
这是一组性能对照，整体仍约 2 Hz，尚未达到实时遥操速度；剩余主要耗时在原生步进路径。
真实 PICO 的姿态方向、手型映射和端到端延迟仍需设备联调。
本机重复启动还出现过 Isaac 初始化长时间停滞；超过 60 秒会输出启动堆栈，
用于排查原生运行时问题。启动器的 100 步运行与失败启动日志分别保留，不能据此保证启动稳定性。
追踪有效性依赖客户端报告；客户端若持续发送带新时间戳的缓存有效姿态，电脑无法判定它已冻结。

实现参考本地 `sonic-mujoco` 的 `67c8f82214bf22e4e9712688570b0ed1e7864d9d`：
沿用 XRoboToolkit 接入方式与视频线协议，视频桥由其 `scripts/pico_video_bridge.py` 改编；
机器人控制改用本仓库公开 FR3 + SHARPA 运动学，不复用 MuJoCo 的关节编号。
客户端字段与坐标约定核对了 XRoboToolkit Unity Client 的 `TrackingData.cs` 和
PICO 原生接口；使用原生右手坐标系的 xyz + xyzw。SDK 和 APK 保持外部依赖。

小场景 CPU 物理的依据：[NVIDIA Simulation Performance Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/physics-performance.html)。
发布场景采用 CPU 配置前须另做接触、轨迹和任务观察器验证，不从空场景结果推定所有场景等价。

CPU 刚体路径的移动物体渲染回归也从完整源码快照启动：

```bash
scripts/isaac_python.sh -B -m scripts.validation.validate_render_sync --cpu-rigid --output <新输出目录>
```

该检查用实际自由落体位置验证 RGB、深度与复位后的画面，拒绝使用上一物理位置的图像；
它是渲染/物理接口检查，不代表抓取成功或正式任务资格。

合成输入工具默认注入超过恢复窗口的断流，并验证新数据返回后仍保持暂停；加 `--brief-loss`
则验证短暂缺帧时保持目标、恢复首帧重新锚定，以及后续重新运动。两种模式均需独立验证。
## 仅工作台版本

Add `--workbench-only` to select `teleop_workbench_v1`. Omit it to restore the
original furnished release. This is a separate runtime composition over the
hash-verified assets; it does not overwrite the downloaded data package.

The six room families retain their original work surfaces, floor,
task actors and fixtures. The kitchen cooking task also retains its preparation
counter and hob; laboratory keeps both benches; study keeps the destination
shelves; the remote-operation task retains its television feedback. Walls,
ceiling and unrelated furnishings are deactivated before native
physics registration. Optional kitchen/laboratory task dressing is also removed.
Retained meshes, materials, placements and solver parameters stay unchanged.
The existing warm lighting preset avoids overexposure after removing the roof.
Kitchen bar practice frames the robot and tabletop together; its wrist camera
is aimed at the task area at initialization, then follows the measured wrist.
The session's `scene.json`, `scene_variant` and workroom geometry-selection
record identify this altered practice environment; it is not the original
benchmark scene or a formal qualification result.

For the rigid kitchen practice session:

```bash
scripts/teleop/run.sh --family kitchen --task breakfast_serving \
  --workbench-only --physics-profile cpu_rigid_v1 --no-task-scoring --mode hands --profile
```

The original CPU/observer restrictions still apply: particle, cloth or reduced
dynamics tasks cannot silently switch to a rigid-only, unscored configuration.

## Tracking 故障隔离与恢复

头部、左右手柄和左右手分别解析。某一路姿态格式错误不会丢掉另一条有效输入。
裸手必须有有效手腕；单个手指缺少关节数据时只保持该手指的已确认目标，
其余手指和机械臂继续使用当前有效数据。首次手指校准仍需完整的手指数据；
不完整时保持机器人，等张开手恢复数据后完成校准，无需重复点击 Start。

短暂丢失后仍需新鲜、稳定的手腕才能重新建立映射。恢复期间持续收到有效数据、
但手还在移动，不再计作持续断联；真正长时间缺数据仍需另一只手捏合或 Start。
手柄长时间断联仍需松开再按 grip。150 ms 过期阈值、关节速度限制和失效时保持
目标均未放宽，也不通过外推补造丢失姿态。

接收线程记录每侧输入连续有效的起点，区分慢控制循环与真正的上游间断。
`frames.jsonl` 的 `input_health` 即使在输入过期时也保留接收数、拒绝数、
最近拒绝原因和最新帧年龄；`input.continuous_since_ns` 是接收端统计。
SDK 日志每五秒有效输入记录一次 `input_health`，包含回调数、重复设备时间戳、
队列替换数、最大回调间隔、桥接耗时和各路输入状态。队列替换本身不代表掉线。
这些统计用于定位问题，不将接收端新鲜度等同于头显光学追踪准确度。

无头显可用 `scripts.validation.pico_teleop_smoke --mode hands --partial-hand-loss`
注入无效手指、头部和另一只手数据，再验证真正断联时保持目标。可同时使用
上面的工作台、CPU 和关闭任务评分选项。此检查不验证真实 PICO 追踪质量。

## 全任务人工资产复核

用手柄逐任务检查抓放、机构功能及完整操作，裸手跟踪质量单列。
此流程属于人工操作复核，不替代场景工程验收或正式评分。

先在 CPU 环境创建一次清单：

```bash
python -B -m scripts.teleop.review init
python -B -m scripts.teleop.review status
```

清单由 catalog 生成，固定任务发布路径及 manifest 哈希；重复初始化会拒绝覆盖。
开始一次任务时给常规 PICO 启动器追加 `--review-dir /absolute/path/to/campaign`。
每次切换任务都退出当前实例，再启动新的源码快照，避免上一任务残留状态。
首轮厨房可使用 `--mode controllers --workbench-only --physics-profile cpu_rigid_v1
--no-task-scoring --profile`。其他任务先核对物理和观察器要求，液体、柔体、
布料、任务驱动视觉及双机器人不得直接套用厨房配置。保留场景默认配置是起点。

记录包括原有输入/关节日志、每个控制步已有的刚体/机器人/接触读回、重置分段、
每秒一张主视角（含 wrist 小窗）的异步 JPEG。无额外物理步或相机渲染，
图片队列至多一张；这不是连续录像，也不是可重放的完整柔体数据。
`session.json` 保留源码快照、发布身份、运行覆盖及记录错误；验收时先检查这些信息。

每个物品尝试接近→抓住→离台→搬动→放下→松手，建议连续三次；
固定夹具检查支撑/交互而非抓起。容器另查内腔，机构另查运动和接触。
工作台版后再检查完整房间的遮挡、碰撞及可达性。

观察完成后记一条（session 必须是实际证据目录）：

```bash
python -B -m scripts.teleop.review note --session /absolute/path/to/session \
  --target cup --check grasp_place --outcome pass \
  --detail '操作者报告：第 1–3 次均完成离台、移动、放回并松手；对应时间……'
```

`--check` 可选 grasp_place/mechanism/full_task/room/control，
`--outcome` 可选 pass/needs_fix/retest。记录只追加，不覆盖失败，
物品试抓通过不会自动把整个任务标成通过。`status` 显示各任务观察次数；
没有观察记录的任务始终待测。关闭评分器的完整操作只能记作人工观察，不能声称裁判通过。
问题修复保留旧证据；改发布后创建新清单并复测。资产修改遵守数据打包、哈希及发布流程。

### 手柄扳机闭合范围（sharpa_trigger_pinch_v1）

按住侧面 grip 后，食指 trigger 的前半程从 URDF 张开姿势到原来的捏取预备姿势；
后半程继续收拢拇指和食指。松开 trigger 张开；松开 grip 仍保持当前目标，
因此释放物品时也要按住 grip。窗口显示两个按键的百分比。

原映射的满扳机只到预备姿势，拇指/食指 FK 参考点相隔约 64 mm。
新闭合端点对实际左右手分别拟合，沿两参考点连线向中点靠拢，目标间距 12 mm；
这些数值是机器人参考点距离，不是物品直径或指面净间隙。其他三指保持预备姿势。
端点属于 PICO 输入映射，不更改公开预姿势库，也不引入物体信息或每帧手指 IK。
这是捏取模式，尚无包握模式；满扳机不等于最大夹持力或全部关节限位。
关节速率、驱动增益/力限、接触保护和原有丢追踪保持规则不变。

原生回归工具 `scripts.validation.pico_controller_closure` 在冻结源码中运行：
指定单右手任务，传入 `--headless --no-video --profile` 和所需运行配置。
它通过合成输入完成预备、闭合、张开，检查实际关节跟踪、参考点距离和接触采样，
将结果写入 `closure_check.json`。这是输入映射与原生运动验证，不是物品抓取成功证明。

### 查看全任务初始布置

`scripts.media.render_task_gallery` 从 catalog 当前发布的 seed 0 初始配置生成原生
RTX 图片，始终停止物理时间线。默认使用每个任务声明的 `task` 相机；
`--camera room` 可查看完整场景，`--workbench-only` 使用遥操的同一精简配方。
原版与精简版应写入不同的 `--output` 和 `--run-output` 目录，保留独立的
发布哈希、相机和图片来源记录。使用完整源码快照及显式 GPU UUID 启动。
这些图片展示发布初态，没有经过物理沉降或动作，不属于物理/抓取验收证据。

### 手柄抓取中心与朝向锁定

`--control-point pinch` 将手柄旋转中心放到 SHARPA 捏取预备姿势的拇指、
食指参考点中点。该偏移由当前机器人 FK 求得，固定在腕部坐标系中；扳机
开合不改变它。它是操作参考点，不代表实际接触点，也不保证物体抓稳。
默认 `--control-point wrist` 保留原映射；裸手始终使用原来的腕部映射。
`--lock-orientation` 保持本次锚定时的朝向，只跟随平移；`--scale .3`
可降低平移比例，人的 10 cm 对应目标 3 cm。

桌面控制面板 **Pinch center / Wrist** 切换控制点，**Lock / Free orientation**
或键盘 **L** 切换朝向锁定。切换后的首个有效控制帧只按实测腕部位姿
重新锚定，不跳到旧目标，也不会自动解除暂停。松开 grip 后再次握住仍会重新
锚定。模式仅作用于手柄，面板显示当前设置；会话与逐帧日志记录这些选项。

原生回归：在冻结源码和普通 Isaac 启动环境下运行
`python -B -m scripts.validation.pico_control_point --sandbox --physics-profile cpu_rigid_v1 --headless --no-video`，
再增加 `--lock-orientation` 运行一次。两次各 120 个原生控制步，以合成手柄
旋转输入检查实测固定参考点漂移、朝向变化、完整接触采样与安全触发。
这是控制链路检查，不是 PICO 硬件追踪或抓取成功验证。

手柄模式下，主双目画面和腕部小窗的黄色十字标记实测控制参考点，随选择
显示腕部或固定捏取中心。它经过各相机独立投影，仅为画面叠加，不参与物理；
不表达接触、抓牢或遮挡关系。裸手模式不显示此标记。

### 练习时的接触报警

`--practice-contact-warnings` 只能与 `--no-task-scoring` 或 `--sandbox` 一起使用。
普通遥操与正式验收仍保留原有严格 2 mm 检查。显式开启后，超过 2 mm 的
求解器报告穿入会显示报警、逐帧记录并继续；达到 5 mm，或连续 25 个 20 ms
控制窗口都出现超 2 mm 穿入时，锁存停止并要求 Reset scene。这些是练习模式
的操作阈值，不是经过硬件标定的安全限值。连续窗口表示每个窗口中出现过
超限，不表示整段时间的每个物理小步都超限。

碰撞求解、全部物理小步、接触采样完整性、关节范围和速度限制不变；不改动
资产或正式评测门槛。报警画面保留约 2 秒仿真时间，严重/持续穿入停止显示
具体原因及深度；时间上限结束则单独标注。每帧 `contact_warning` 和
`safety_trigger` 分开记录，发生报警/停止即刷新日志，防止暂停时证据仍在缓冲区。
会话记录策略、阈值和报警窗口数；带报警的练习不能据此宣称物理有效或任务通过。

### 掌内腕部相机

PICO 的 wrist feed 使用运行时专用的 `in_hand_wrist_camera_v1` 外参：相机挂在
`right_hand_C_MC`（SHARPA 掌心坐标系），光心在该坐标系的
[55, 50, 45] mm，位于掌侧表面外、指根后方，向抓取中心倾斜约 11.5 度。
早期 [0, 0, 45] mm 候选埋入掌部实体导致黑屏，已弃用。镜头沿手指之间朝工作区看。它不再用世界坐标目标瞄准，
所以相机随手掌刚性运动，手指会自然出现在视野两侧。场景发布文件不改动；
仅在遥操运行时替换 wrist 相机外参，保留其原有光学参数。会话报告记录相机策略与外参。

这遵循 eye-in-hand 的常见结构：传感器刚性装在末端执行器附近，面向实际
操作区域。Isaac Lab 的 XR 反馈也将 robot POV camera 作为挂在机器人 body
link 上的独立画面；相关文献将末端相机用于改善远程操作的工作区可见性。
实际机械安装仍需用真机的手眼标定替换这组仿真外参。

### Task-driven visuals in practice

Use `--no-task-scoring --keep-task-observer` for practice tasks whose displays depend on native task observation, such as the television remote. This retains native button/contact observation and visual updates; practice still does not report formal task qualification. It may cost more CPU time than disabling the observer. Arm practice overrides remain explicitly marked `unscored_practice`.

`--arm-base-yaw DEG` sets the simulated base heading in unscored practice, without changing its position or the joint calibration. Laboratory sample sorting uses `-90` to face the island worktop. This is a runtime override; released assets remain unchanged.

Practice contact policy: `--practice-contact-warnings` reports penetration, including severe or repeated penetration, without pausing. Collision response and contact receipt validation remain enabled. Strict runs retain their original stop policy.

Gravity support can use `--leader-gravity-calibration PATH` together with `--leader-gravity-urdf`. This measured hardware reference is used only for physical gravity output; `--arm-calibration` continues to control the simulated follower. Keep these references separate when changing simulation joint offsets or heading. Start enables gravity; pause disables it. A new physical reference needs a supported reference-pose check and low-current direction confirmation.

For grasp-only practice, `--no-task-scoring --hide-task-displays` disables task observation and hides its material-bound display meshes. Native loading rejects display meshes containing collision or rigid-body geometry. Particle and reduced-dynamics requirements remain enforced. No live weight or formal task result is available in this mode.

### 多刚体线缆的性能检查

用 `scripts/validation/teleop_chain_benchmark.py` 在冻结源码中比较
`nist_cable_routing`、`nist_belt_drive_assembly`、`nist_wire_harness`。
通过 `scripts/isaac_python.sh -B -m scripts.validation.teleop_chain_benchmark`
运行，指定 `--task`、`--profile cpu_rigid_v1`（或 `scene`）、`--output`。
默认预热20步后记录80步；`--cprofile` 仅用于定位 Python 热点，正式耗时对比应关闭。
输出包含每步物理/渲染耗时、完整40子步采样断言、接触总量和逐帧状态数组。
测试只使用仿真关节目标，不连接实体设备；小幅机器人运动及被动链条测试
不能替代实际抓取、绕障、张紧或任务成功验证。

接触记录按相同点数批量计算，保持各接触组内部 float64 求和顺序、校验和
去重顺序。对外返回的接触字典与向量独立复制，避免通用递归复制的开销。
这些优化不改变物理步长、链条分段、碰撞体、求解器参数或接触采样频率。
