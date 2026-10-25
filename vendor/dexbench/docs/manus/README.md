MANUS 采集接口（2026-10-11）

本接口已实现独立 SDK producer、结构化人体动作归一化、原始数据记录、回放、
多机器人 retarget 回调，以及新 agent bench 的 DirectJointAction / RobotJointCommand
转换。SDK 动态库和真实 CFFI 结构已在本机加载验证；没有接入手套、追踪器或许可证
设备，因此当前数据验证是 synthetic/replay，不能写成真实 MANUS 示教采集成功。

设备与版本依据

官方当前产品是 Metagloves Pro 和 Metagloves Pro Haptic。Quantum 已于 2026-06-23
宣布停产。不能仅由“最新款”推断用户购买的是 Pro 还是 Haptic；producer 从实际
Landscape 记录 DeviceFamilyType、hand、firmware，而不是固定型号字符串。
https://www.manus-meta.com/products/overview
https://www.manus-meta.com/blog/manus-quantum-metagloves-reach-end-of-life

官方当前下载页是 Core / SDK 3.2.1，更新说明列出 Pro 系列固件 5.20；这只是调研
时的最新版本，不代表用户设备已经升级。项目固定 SDK 3.2.1，不自动更新设备或软件。
https://docs.manus-meta.com/latest/Resources/
https://docs.manus-meta.com/latest/ReleaseNotes/

3.2.x 官方包已有 Python CFFI wrapper、C++ 和 ROS2。Linux 支持 Integrated，
也可连接独立 Core；Windows Core 主机可以通过外置 producer 将数据发送到 Linux
采集端。SDK 数据流需要对应 SDK entitlement，普通 Core 许可不能据此推定可读
原始传感器。项目未激活或验证任何许可证。
https://docs.manus-meta.com/latest/Plugins/SDK/Python/getting%20started/
https://www.manus-meta.com/software

官方包保存在共享目录：
  <开发存储>/assets/devices/manus/sdk/3.2.1/
    downloads/MANUS_Core_3.2.1_SDK_Linux.tar.gz
    distribution/              原样保留 LICENSE、Python/C++/ROS2
    provenance.json            来源 URL、包 SHA256、文件数量与日期
    env/                       新建专用 Python 环境，不修改 Isaac/Newton 环境
    compat/                    本机缺失的 libzmq.so.5 的独立解析链接
    python_sdk.sh              仅对子进程增加动态库搜索路径
    environment.lock           cffi / pycparser / pyzmq 固定版本

官方包与 SDK 功能访问权是两回事。库加载探测只检查 ABI，不触发设备发现、固件更新、
标定、许可证写入或 haptics。当前宿主原来缺 libzmq，专用环境使用 PyZMQ wheel 携带
的 ZeroMQ / libsodium，不替换系统库，许可证和来源保留于 environment_provenance.json。

数据与控制结构

  手套 + 外部腕追踪 → 官方 Core / SDK 独立进程 → JSONL acquisition packet
    → CaptureBridge → human_motion_v1 → RetargetRegistry(robot_id, version)
    → RobotJointCommand → DirectJointAction → Isaac bench 服务

SDK 库绝不载入 Kit 所在进程。官方 callbacks 内复制 CFFI 节点/结构到 Python 基本类型，
再放入有界队列；不在 SDK 回调里运行 IK、RL、仿真或文件写入。队列满时明确计数，序号
继续递增，接收端据此保留 gap；不会把丢帧伪装为等间隔完整轨迹。SDK 进程 stderr 的
最终 sdk_queue_drops_total 也应归档。producer stdout 是干净 JSONL；native stdout
转到 stderr。重连或 SDK 重启必须创建新 stream_id 和新 capture，不跨重启恢复幂等。

坐标、密集信息与标定

* acquisition packet 保存完整节点 ID、parent ID、chain/finger-joint 语义、位置、四元数、
  有效性；人体节点数不固定为 21，也不等同于机器人 DOF。
* 官方 SDK 设置 VUH = right-handed / +Z up / XFromViewer / unitScale=1 / world。
  这只确定坐标表达，不证明已经获得可靠世界腕位姿。项目四元数固定 wxyz。
* 通用输入允许 m/cm/mm，父节点局部骨架会逐级复合；归一化输出为 m 和 rad。
  SDK 人体 ergonomics 的 degree 明确转 rad，它们不会直接当机器人关节位置。
  RawSkeleton 是 Core 估计的人体骨架，不是未经估计的 EMF 原始电信号。
  当前 producer 未订阅独立 RawDeviceData sensor stream，不能把该能力标作已采集。
* T_world_capture 与 T_tracker_wrist 是独立外参；数据记录保留配置 SHA256。
  identity 只用于 synthetic 示例，不是物理标定。world_transform_calibrated 标识
  输出坐标是否实际标定，不能把这个标记为 false 的点当成真实世界测量。
* 手套标定由 Core 的用户/手套标定流程完成，保存 .mcal 和散列引用；外部追踪系统的
  world-to-bench、tracker mount-to-wrist，以及 robot flange-to-palm 各自标定。
  机器人安装外参与 retarget 参数属于 mapper 的固定版本，不能在 episode 中偷偷变化。
  CaptureConfig.calibration_artifacts_sha256 记录 .mcal/外参资产散列，
  retarget_implementation_sha256 记录实际映射实现；不具备这些资产时保持空值。
* 独立 wrist tracker 必须同时提供有效位置与方向、tracker ID、tracked body frame，
  并满足时钟、外参和新鲜度要求。IMU orientation-only 绝不补成 XYZ=0。
  无有效腕追踪时可继续采人体手部数据，映射动作保留上一组 arm targets。

官方坐标与 callback/结构依据：
https://docs.manus-meta.com/latest/Plugins/SDK/C%2B%2B/SDK%20Client/
https://docs.manus-meta.com/latest/Plugins/SDK/Python/SDK%20Minimal%20Client/
本地精确 ABI 来自 distribution/C++/SDKClient/ManusSDK/include/ManusSDK*.h。
本地 Python examples/minimal_client.py、manus_wrapper.py 与 generated/_enums.py
也保留；本实现不会猜测 ctypes 布局。SDK producer 使用原生函数的真实名称、字段。

时钟与失效处理

capture_time_ns 是 SDK callback 进入用户代码时的 monotonic 时间，不是传感器曝光。
source_timestamp 保留 ManusTimestamp 的完整 uint64 字符串和编码标记；不会错误地
当 Unix 纳秒。source timestamp 和人体原始坐标在 acquisition.jsonl 中保存。
tracker.lastUpdateTime 改变时才刷新 tracker callback 时间，重复发布旧测量不算更新。
ergonomics 有自己的 callback 时间与 vendor 时间，仅合并 30 ms 内数据。

同机进程共享 host/boot/monotonic clock_id。跨机必须配置实测 offset、uncertainty、
同步方法；不根据两台机器的 monotonic 数值大小猜 offset。未同步的帧仍可记录，但
hand/arm control_valid 为 false。当前实现不提供自动 PTP/NTP 校准，不插值、不预测
补帧。数据集保留测量 uncertainty，之后可以在明确时间域上做离线对齐。

默认最大 callback age = 150 ms，glove/tracker skew = 30 ms，clock uncertainty
= 5 ms。未来时间、tracking lost、标定缺失都会记录 hold 原因。RetargetRegistry.map
必须在每个控制 tick 再用 now_ns 检查 valid_until；输入停更后立即返回 hold_all，
不能沿用最后一帧的旧 valid 布尔值无限驱动。实际机器人急停另由硬件控制器负责。

同一 stream 的 sequence 单调递增；重复同序号同内容幂等，不重复生成 normalized
样本；同序号不同内容、乱序、设备/时钟/拓扑/标定改变均拒绝并记录事件。进程内保存
最近 4096 个摘要；更早的重试安全拒绝，绝不重执行。采集重启创建新目录。

运行（项目根目录 <开发存储>/interface）

1. 无硬件采集和回放检查：

  .venv/bin/python -m bench.adapters.manus --mock 20

  输出是新的 datasets/fr3_sharpa_bench/manus/captures/<uuid>/。
  默认 mock 标定为 false，arm/hand 控制不可用，不伪装为设备已校准。
  将 manifest.configuration 保存为一个配置 JSON 后可执行：

  .venv/bin/python -m bench.adapters.manus --config /path/to/capture-config.json \
    --replay <开发存储>/datasets/.../acquisition.jsonl

  回放原始时间和 source_kind 不改写为“现在”，新 manifest.replay_of 链接父记录；
  过期样本保留但不用于控制。采集器不是仿真物理回放器。

2. 原生 SDK ABI 探测（无设备访问）：

  <开发存储>/assets/devices/manus/sdk/3.2.1/python_sdk.sh \
    -m bench.adapters.manus.sdk_process \
    --sdk-python <开发存储>/assets/devices/manus/sdk/3.2.1/distribution/Python --probe

3. 真设备时使用 bench/adapters/manus/examples/capture.template.json 创建独立配置，
   填写实际 calibration_id、host clock、追踪外参和 mapper 版本。设备序号用实际
   glove ID，不能用示例序号冒充。然后由设备工作站运行 producer，例如：

  set -o pipefail
  <开发存储>/assets/devices/manus/sdk/3.2.1/python_sdk.sh \
    -m bench.adapters.manus.sdk_process \
    --sdk-python <开发存储>/assets/devices/manus/sdk/3.2.1/distribution/Python \
    --mode local --glove-id 123456 --calibration-id calibration_20261011 \
    --duration 60 2><开发存储>/datasets/fr3_sharpa_bench/manus/sdk_capture.stderr \
  | .venv/bin/python -m bench.adapters.manus --stdin --config /path/to/capture-config.json

  123456 和 calibration_20261011 是占位值。remote 模式要求明确主机；发现多台 Core
  时不会自动选择首台。没有腕追踪不传 --tracker-id。连接后读取真实 SDK/Core 兼容性。
  本项目为这一流程提供实现，尚未完成带许可证和真实设备的端到端测试。

与 agent bench / System0 的对接

bench.adapters.manus.agent_adapter.build_command(frame, registry, capabilities, ...)
读取动态 RobotCapabilities。registry 为每种 robot_id + retarget_version 注册映射：
  sharpa_w01_right / allegro_v5_right / leap_v1_right / shadow_hand_e_right。
支持这些 ID 的协议，不表示已提供四只手的标定后 mapper。真实 retargeter 必须由
逐手型标定或优化求解接入；未注册即报错，不使用固定 finger curl 或通用比例替代。
输出关节集合必须完整匹配当前 robot，检查位置/速度限制，腕无效时锁住 arm targets。

调用方式：

  command, mapping = build_command(
      frame, registry, robot_capabilities,
      previous_targets=last_acknowledged_targets,
      now_ns=time.monotonic_ns(), now_clock_id=host_clock_id())
  action = direct_action(command, robot_capabilities,
      action_id="capture_001_tick_42", observation_sequence=observation.sequence)
  # POST /episodes/{episode_id}/actions : action.model_dump(mode="json")

代码返回真正的 bench.protocol.schemas.RobotJointCommand / DirectJointAction。
记录完整 action payload 和 mapping.source_stream_id / source_sequence。
HTTP 重试复用完全相同 action_id 和 payload；新的控制 tick 使用新的 action_id。
只有服务器确认动作应用后才更新 last_acknowledged_targets。不要把发送尝试当成应用。
采集模式应另建 human demonstration episode，不计作 agent 自主完成成绩。

System0 训练时按 (capture_id, stream_id, sequence) 关联 human_motion、mapper 输出、
实际应用 robot actions、仿真时间、对应 RGB/proprioception，以及独立 evaluator 的
object/cloth/contact ground truth；raw 人体动作不是 ground-truth 机器人命令。
原始采集保持不可变，retarget/重采样/技能分段各建 derivation 和配置 hash。
上层语义 intent / dense geometry 可作为人工标注或 policy 条件，不自动从人体姿态
捏造“GRASP 成功”。模拟 ground truth 只入训练/评价通道，不泄露给 vision-only agent。

记录文件与当前边界

manifest.json：配置、SHA256、采集代码 hash、父记录、来源统计、结束原因、文件 hash；硬件真实性
由 SDK/hardware 验证报告证明，采集器不会仅因 source_kind=hardware 就自证已连接。
acquisition.jsonl：每次 schema-valid 原始输入，包含 accepted/duplicate/rejected。
human_motion.jsonl：每次 accepted 的米制/弧度骨架、时钟、wrist、hold 原因、gap。
events.jsonl：顺序/去重/校验错误等采集事件。未通过 schema 的任意内容不会回显保存，
因此凭证字段被拒绝，解析错误日志不泄露原文。项目不接收密码或 license key。

API 采集数据只保存匿名 operator_id。SDK、.mcal 和其他可复用标定资源放 assets；
episode 原始传感器、图像、派生数据和验收记录放 datasets；项目内只放代码与配置模板。

尚未验证：真实手套/追踪器、SDK entitlement、端到端延迟、机器人安装/人体映射标定。
尚未实现：生产用多手型 retarget 优化器、自动跨机时钟同步、haptic 反馈、真实机器人
网络控制。以上不妨碍当前采集接口、回放、动态 joint command 对接的离线验证。
采集适配使用 `bench/adapters/manus/agent_adapter.py` 对接公开机器人动作；主仿真服务为 `bench.runtime.server`。旧 Newton adapter 已归档，不属于当前运行路径。
