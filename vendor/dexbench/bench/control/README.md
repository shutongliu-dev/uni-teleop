# 经典腕部与指尖执行器

`ClassicalFingertipPolicy`（`fingertip.py`）是任务无关的经典控制器，通过 `System0Policy` 协议提供 reset、begin、step、cancel。当前版本为 `classical_fingertip 1.9.0`，容差协议为 `geometry_tolerance_policy_v2`；descriptor 声明 `trained=false`，没有学习权重。它支持 `move_geometry` 与定时 `hold`，控制周期为 20 ms，默认本体为 `sharpa_w01_right`。

双臂 `dual_fr3_sharpa_w01` 由 `bimanual.py` 的 `classical_fingertip_bimanual 1.2.0` 驱动，每只手使用同一单臂 1.9.0 执行器，通过 `hand_geometries` 分手给目标。当前建设范围为单臂与双臂 SHARPA；Allegro V5、LEAP V1、Shadow E3M5 的版本化装配、代码与记录保留，集成暂缓。历史 CPU A/B 证明 v1 装配下 1.9.0 与 1.8.0、以及 SHARPA 的 1.8.0 与 1.7.0 对应序列逐位相同；这个结论不跨装配朝向或物理模型，详见[机器人本体](../../docs/agent_bench/embodiments.md)。

控制器只接收静态机器人描述、基座变换和新鲜关节反馈。Agent 从公开 RGB-D 决定目标；控制器不读取任务名称、物体真值、评分或任务阶段。它不做触觉闭环、力控制或通用避碰规划。

## 输入与坐标

场景工具与 HTTP 服务共用 `bench.control.factory`。输入是所选 release 的场景配置，不从任务名称推断机器人：

```python
from bench.control.factory import (
    robot_configuration, configured_policy, public_robot_geometry,
)

configuration = robot_configuration(scene)
policy = configured_policy(configuration)      # 每次创建独立执行器
geometry = public_robot_geometry(configuration)  # 只含机器人几何
```

工厂保留机器人资产、基座位置／朝向、hand_id、装配版本及双臂 arm_base_poses；支持单臂 SHARPA 的 29 个主动关节与双臂的 58 个。默认键省略保持历史语义：未固定 `robot.assembly_version` 时仍使用对应手的历史默认装配，不自动升级为 v2。机械室几何模型、棋牌室 ready 布局及房间观测／棋子拾取诊断已复用此入口；它不调整摩擦、驱动或其他物理参数。具体接入见[场景交接说明](../../docs/agent_bench/scene_integration.md)。

双臂 1.2.0 允许显式传入 `arm_base_poses={"left": {"position_m": [...], "quaternion_wxyz": [...]}, "right": {...}}`。两侧位姿均相对公共机器人基座，单位为米和 wxyz 单位四元数；省略时保留原装配并排偏移。场景的同名 `robot.arm_base_poses` 同时用于 USD 引用、原生 reset 和公开 FK/IK，并要求 `isaacsim.robot_base_constraint="native_fixed_base_v1"`。每侧保留原始 29 DOF / 43 links，原生启动与 reset 分别核对两侧固定基座、关节归属和世界位姿；这不替代抓取、接触或避碰验收。

`public_geometry(joint_positions_rad)` 用实际 URDF FK 返回世界坐标腕姿，以及 `right_hand_flange` 局部坐标的五指位置。局部坐标随腕部运动；`little` 对应 URDF 的 `right_pinky_fingertip`。这些点是虚拟 frame 原点，不是已标定的接触面中心。`public_finger_geometry` 是服务公开的 `robot_geometry_v2`：另含各指尖 frame 姿态（+z 沿远节指向远端，+x 指向掌侧指垫；原点位于指垫内、掌侧表面下约 2 mm）和每指 URDF 关节原点，便于判断中节/近节是否会先接触。

`move_geometry` 可包含：

- `wrist`：世界坐标位置与 wxyz 单位四元数；省略时采用当前测量腕姿。
- `fingertip_positions_m`：任意非空手指子集，固定 `fingertip_frame_id=right_hand_flange`。
- `hand_posture_rad`：手关节参考；与同指 tip 同时提供时作 IK 冗余参考，不再独立要求其关节到位。

至少提供一种目标。不接受 `contacts` 或 `wrist_trajectory`；增量接口由上层转换为绝对目标。`hold` 不携带 geometry，显式指定正的 20 ms 整数倍时长，且 timeout 不少于保持时长。

公开 `right_hand_flange` 是随手转动的手坐标系 H。`flange_axis_centred_v1` 下它与 `fr3_link8` 重合；当前 SHARPA `isaacsim_6_1_v2` 装配使用 `flange_axis_centred_v2`，相对 `fr3_link8` 绕法兰轴转 180°。这些均为版本化、未实物标定的仿真安装假设。`right_hand_C_MC` 掌根相对 H 另有约 0.0295 m 局部 z 偏移，二者不可混用。

## 跨动作状态

未选手指完整链保留前一次成功接受的目标；首次动作取测量参考。取消时服务暂停推进并保留已确认驱动目标；下一次接受的动作从该参考执行恢复制动，停稳后固定保持。不能把取消响应解释为已额外执行了一段物理制动。新 geometry 只恢复机械臂与被明确选择的手指链；被拒绝的请求不改变已有计划。省略腕部的动作仍恢复腕部跟踪。

正常动作保留已确认参考位置、速度与有界积分；`hold` 维护上一目标。直接关节轨道的交接必须来自 worker 已确认的实际驱动与速度，不因此获得经典控制器的平滑保证。HTTP `direct_joints` 的 `min_jerk_velocity_preflight_v2` 只约束本段参考速度；短时长零步拒绝，不自动延长，也不承诺加速度或跨段连续性，详见[执行边界](../runtime/README.md#控制边界)。

接触可把实测关节推出 URDF 边界。1.7.0 起所有动作在 begin 时都接受不超过 **0.03 rad**（与服务端整回合安全带相同，`MEASURED_JOINT_LIMIT_SAFETY_BAND_RAD`）的实测超限：超限值只在内部裁剪为 IK 初值与锚点，原始位置、速度与 FK 不变，发出的目标仍严格在 URDF 内；超过 1 mrad 时附带 `measured_limit_projection` 诊断。动作执行中若超限继续增大——手部超过 max(5 mrad, begin 时超限 + 2 mrad)、机械臂超过 max(1 mrad, begin 时超限 + 1 mrad)——该动作以 `invalid_robot_state` 失败，但回合继续，下一动作可以从当前状态开始；超过 0.03 rad 才由服务端结束回合。

旧版本对普通几何动作只容许 0.001 rad，导致历史 G28 回合小指 PIP 在接触中实测 −0.00189696 rad（原始速度 −13.78 rad/s）后所有几何动作都被拒绝的死锁；1.6.2 只为 `hold` 增加的 0.005 rad 恢复带已被上述统一规则取代，`measured_limit_recovery` 诊断码仅为读取旧记录保留。

`step` 产生候选命令。worker 执行成功后调用 `commit_pending_step`；未执行时 `discard_pending_step` 回滚候选，避免取消竞态把未执行命令当成下一动作的起点。

## 求解与完成条件

腕部与手指均使用有界 IK，最终以真实 FK 检查目标残差。指尖先以完成容差的一半（0.75 mm）严格搜索；若所有种子都达不到，但最近解残差不超过 **1.0 mm**（可达集边界附近），则执行该最近解，完成仍按原请求以 1.5 mm 判定，并附带 `ik_target_at_reach_boundary` 诊断报告残差。超过 1.0 mm 时拒绝，诊断给出 `best_found_fingertips_m`（找到的最近指尖位置，非全局可达证明）与 `limiting_joint_names`（处于限位的关节）。所选手指目标应用公开的 `sharpa_dip_pip_v1` 姿态先验：食指、中指、无名指和小指 DIP−PIP 不超过 0.5 rad。原始测量不被投影；该先验不等于已测得的人类抓持质量。

默认腕位置容差为 0.003 m，腕旋转容差为 0.04 rad；所选指尖容差为 0.0015 m，只有 posture 目标的关节容差为 0.03 rad。未选手指的维护残差公开，但不阻止所请求腕部动作完成。IK 接受阈值比跟踪容差更严格，以留出跟踪余量。

关节参考采用五次插值与逐步限幅。FR3/手参考速度上限分别为 0.65/1.0 rad/s，加速度上限为 2/3 rad/s²，并遵守 URDF 限位与速度上限。有界 PI 补偿及积分各限制在 ±0.04 rad，存在饱和抑制与有限响应参考。机械臂积分只在名义轨迹静止后累积：运动中的误差主要是刚性驱动的阻尼滞后（Kd/Kp≈0.07 s），对它积分会造成腕部超调（旧版 35–51 mm 移动中位超调约 6–7 mm）。手部积分不变。失败恢复时清除失败任务的积分，但机械臂保留上一次正常完成时的积分（重力/负载补偿），避免下一个零位移动作下沉。这些限制针对命令序列，不是实测接触力或刚体运动的保证。

`move_geometry` 正常完成需要所请求机器人几何达标、参考静止，且测得腕部速度不超过 2 mm/s（到期限时若仍在容差内则照常完成，并在 `robot_execution_state.wrist_settled=false` 中如实报告）。名义轨迹结束后，若 0.3 s 内目标残差改善不足其容差的 5%，且驱动目标（含 PI）变化小于 2 mrad，判定为停滞（通常被接触阻挡）：参考先连续制动到静止，再以 `tracking_stalled` 失败结束，不再空等到超时；已执行的驱动目标保持，因此正在按压物体的手指继续按压。服务端在未达到的手指或腕部确有公开接触时把 `termination_reason` 标为 `contact_stalled`，状态仍为失败。`hold` 完成需要执行完整时长。时长不足时返回 `insufficient_duration` 与 `duration_preflight`，说明最短参考、制动与稳定等待时间，不自动延长动作。

动作完成、任务成功与物理验收是三个独立结果。服务若因任务成功提前结束活动动作，应使用对应 `termination_reason`，不能据此推断原几何目标已到位。

## 可观测性与验证

HTTP `GET /episodes/{id}/robot` 公开所选机器人坐标、关节限位、速度上限、手指链和姿态约束；`POST /episodes/{id}/robot/ik` 从绑定快照的公开实测关节检查几何目标，不调用 policy 的 begin/step，不推进物理或改变执行器。它不检查轨迹、碰撞和执行时长，数值求解失败也不等于不可达证明。手姿态先验是约束，不是经过抓取验收的 open/pinch/envelop 预姿态库。

descriptor 的 `camera_observation_mode=decision_only` 表示决策边界提供 RGB-D；每个控制 tick 使用关节反馈，不反复消费旧图像。`execution_state(observation)` 是纯读取，公开已确认机器人目标与维护状态，不推进控制器，也不暴露物体真值。

provenance 固定 URDF、控制源码、几何与运动学源码、基座变换、NumPy 版本及容差/限速配置。服务另行固定协议与原生 worker 源码。

相关 CPU 检查保留在 `tests/test_classical_fingertip_control.py`、`tests/test_reference_governor.py`、`tests/test_measured_limit_recovery.py` 与手型/交接测试中。超限测试使用带来源 hash 的历史原始 q/dq 与末两条已确认驱动命令，检查安全带、超限增长规则、取消/ACK 回滚及继续拒绝不合法请求目标；它们不模拟或证明实际接触会恢复。1.7.0 的离线复核（CPU 关节 PD 被控对象，无 LLM）：67 个历史腕部移动全部完成，≥20 mm 移动峰值超调中位 0.31 mm、最大 0.67 mm（1.6.2 为 6.2/6.6 mm），完成时腕速 p90 1.9 mm/s；39 个历史指尖 IK 拒绝中 9 个（真实最小残差 0.77–0.99 mm）改为执行，17 个超出可达 1.5 mm 以上的请求全部仍被拒绝；历史 G28 小指超限状态下几何动作可正常完成。CPU 结果只能证明对应软件条件；原生 PhysX 跟踪、接触与任务结果仍需独立记录。历史失败及已冻结源码保持原样，本说明不移植旧版本成绩。

当前安装朝向与双臂组合另由 `tests/test_mount_convention_v2.py`、`tests/test_bimanual.py` 检查。单臂默认 v12 使用刚性指垫，双臂 v13 r2 使用柔顺指垫；这属于发布的物理配置，不由控制器版本自动切换。已有原生脚本正例只证明其固定任务与运行配置，不能据此宣称经典控制器具有通用抓取能力或完整模型评测资格。
