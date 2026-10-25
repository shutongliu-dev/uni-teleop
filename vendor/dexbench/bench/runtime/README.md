# Native Isaac Agent API

唯一服务入口是 `bench.runtime.server:app`，HTTP 协议为 `agent_bench_v1`。前端负责协议和调度，Isaac/PhysX 在独立 worker 主线程执行。每个服务只允许一个活动回合。

服务、客户端、worker 和控制器从显式冻结根启动，步骤见 [bench 使用说明](../../docs/agent_bench/README.md)与[快照契约](../../docs/agent_bench/source_snapshots.md)。部署时显式设置 `BENCH_TASK_RELEASE`、数据目录，并按 UUID/PCI 核对 GPU；不复制旧 runtime pins。单臂默认发布为 `v12_mount_v2_r1/sharpa_w01_right`（刚性指垫），双臂层使用 `v13_pad_compliance_r2/bimanual`（柔顺指垫）。用户于 2026-10-18 授权 task0 的 Micu 模型验证，旧矩阵队列仍停用；已有零模型脚本工程正例，单项成功不代表完整 release 验收或模型成绩。

HTTP 是公开输入边界，不是文件系统权限沙箱。参赛程序不应同时获得 private/datasets 的直接访问。MANUS 采集说明在 [docs/manus](../../docs/manus)，硬件目前未连接。

## 目录、回合和动作

| API | 用途 |
|---|---|
| GET /robots、/tasks | 公开机器人与任务 |
| GET /policies?task_id=…、/interfaces?task_id=… | 选定任务的控制器与接口能力 |
| POST /episodes | 选择 task/robot/track/policy、seed 与预算 |
| POST /episodes/{id}/actions | 提交绑定 observation_sequence 的机器人动作 |
| GET /episodes/{id}/actions/{action_id} | 查询 accepted/running/completed/failed/cancelled |
| POST /episodes/{id}/actions/{action_id}/cancel | 在控制边界取消动作 |
| GET /episodes/{id}/observation | 最近已发布决策快照 |
| GET /episodes/{id}/robot | 当前机器人说明、坐标、关节与控制边界 |
| POST /episodes/{id}/robot/ik | 绑定公开快照的只读机器人几何求解 |
| POST /episodes/{id}/finish | 结束未完成回合并保存 |
| GET /episodes/{id}/result | 获取终局结果；运行中为 409 |

`/policies` 和 `/interfaces` 接受 `task_id`。未知任务返回 422；同一策略在多个任务上有不同 descriptor 时，省略任务也返回 422。只有查询无歧义时才兼容旧的无参数请求，客户端应先选择任务再查询能力。

创建回合只接受公开选择和配置摘要，不接受客户端指定 release 路径、私有 hashes、物体初态或评分器。响应含 episode_id、固定 spec、capabilities 与 sequence=0 的 observation。模型先核对公开能力和实际 spec，再执行动作。

原生初始化与初始 RGB-D 完成后开始 wall 时钟，setup 耗时单列。运行时源码改变会使受影响回合失去资格；改代码后需新快照和新服务。省略预算字段使用默认值；显式 null 关闭相应 sim/wall 总时限，次数限制仍生效。有限 wall 预算包含思考、网络、仿真与后续取图，迟到决定不能继续发动作。

动作返回 202；同 ID、同 payload 重试返回原状态，不再执行，同 ID 换内容拒绝。非法目标、stale、busy 等请求不推进物理，服务端尝试与客户端 primitive 尝试分别记录。取消保留最后已确认驱动目标，不把测量值写回目标。未 ACK 的控制候选不应被当作已执行状态。

原生阻塞不应阻塞整个 HTTP 事件循环。读取公开状态使用已经提交的快照；查询仍不保证 worker 可推进或可被强杀。终止/释放结果以 worker 和服务记录为准。

## 控制边界

`direct_joints` 使用 capabilities 的完整具名主动关节表。目标必须有限、有界、无缺失/未知关节；被动 mimic 坐标和 object-state 字段不接受控制。

原生路径声明 `direct_joint_execution=min_jerk_velocity_preflight_v2`。起点为 worker 最后 ACK 的驱动目标，缺失 ACK 时拒绝，不能用实测关节冒充。每个关节的 `max_velocity_rad_s` 取参考上限（臂 0.65、手 1.0 rad/s）、已固定 URDF 速度上限及可用原生限速回读中的最小值；缺少几何或原生回读时不虚构该来源。最短时长按 min-jerk 峰值系数 1.875 计算并向上取整至 20 ms。

请求过短时，普通 direct 与 `free_tool_agent` 的 direct 分支都在交接和原生步进前失败：`insufficient_duration` 配合 `duration_preflight.skill=direct_joints` 给出请求／最短时长与步数，`termination_reason=begin_rejected`。不自动延长请求。可行请求在本段内生成速度受限参考；不保证加速度、前后轨道的速度连续性、实测速度或接触力。无 ACK 的非原生旧 worker 仅声明 `step_target_v1`，不享有 v2 保证。

默认经典控制器经 `bench.control.factory` 提取静态机器人配置，由服务端按所选任务注册工厂，并在回合创建时实例化。同一 release 可以包含不同机器人摆位的任务，不再把首个任务的布局套给全部任务；descriptor 与实例配置必须一致。任务、策略与配置摘要由受信任部署固定，客户端不能提交任意工厂或源码路径。旧式单实例策略注册仍保留，其生命周期按该插件合同管理。

`system0` 选择公开注册策略。当前单臂默认 `classical_fingertip 1.9.0` 是未训练经典控制器，按 20 ms 机器人反馈输出关节目标。geometry.wrist 是 world 中 hand-flange 位姿，fingertip_positions_m 是 right_hand_flange 局部的 thumb/index/middle/ring/little 目标；可选择手指子集，未选链遵守控制器声明的保持规则。手坐标系随所选装配固定：v12 的 SHARPA 使用绕法兰轴转 180° 的 v2 约定，不能把它与历史 v1 的 `fr3_link8` 坐标直接混用。

双臂 `dual_fr3_sharpa_w01` 使用 `classical_fingertip_bimanual 1.2.0`，通过 `hand_geometries.left/right` 指定各手目标，省略的一侧保持已有目标。公开手几何与触觉也按手区分；两侧任一执行器失败会结束该动作。这不提供自动协调抓取或力控制。

具名多机器人 `fr3_sharpa_instances` 使用相同控制工厂，以配置声明的实例名绑定独立的原始右手装配。客厅双人抽塔的 `front/rear` 通过 `/turn-games` 绑定玩家；具名实例与真正左右手装配分别保留各自几何、关节映射和接触历史版本。

它不读取物体真值生成任务流程，也没有通用抓取、自动避障或力控制。IK 不可达或时长过短可零步失败；duration_preflight 给出参考时长下界，不保证收敛，也不自动替模型重试。

`move_geometry` 的 duration/timeout 是最大执行窗口，到位可提前结束。`hold` 是显式定时保持，支持 0.02–5 s、20 ms 整数倍；零增量运动不能替代 hold。termination_reason 区分 controller_converged、time_hold_completed、direct_duration_completed 和 task_terminated；任务提前终止不等于原目标已收敛或保持足够久。

声明该能力的 `system0`／`free_tool_agent` 支持 `hold(mode="acknowledged_drive")`，显式保留最后 ACK 的静止驱动目标，与默认几何保持具有不同参考语义。`direct_joints` 轨道使用默认 `hold` 按 ACK 目标生成定时关节动作，不接受上述显式模式。等待场景稳定仍须由调用方提交保持动作，服务不会暗中推进物理。

固定四接口与独立自由混用模式见 [modes.md](../../docs/agent_bench/modes.md)。自由切换要求服务公开 ACK handoff 能力，切回几何控制从原生已确认驱动目标衔接。

## 机器人说明与只读 IK

`GET /episodes/{id}/robot` 返回 `bench_robot_guide_v1`：所选本体、SI 单位和 wxyz 约定、坐标系、关节顺序与限位／速度上限、各手链结构和安装约定、姿态先验、执行及接触能力。不返回私有物体状态或源码路径。`hand_presets` 按实际本体返回 open/pinch_prepose 目录和来源／CPU 验证范围；预姿态不代表抓取或碰撞验收。

`POST /episodes/{id}/robot/ik` 接受 `bench/protocol/robot.py` 中的 `IKQuery`（请求 schema 见 `/openapi.json`，公共函数也通过 `query_ik` 暴露）；接口目录的 shared_contract 可发现这两个端点。必须携带当前 `observation_sequence`，单臂给 `geometry`，双臂或具名实例给非空 `hand_geometries`，二者互斥，各手 fingertip_frame_id 必须匹配对应手坐标系。目标可为 wrist、局部 fingertip_positions_m 或 hand_posture_rad；不接受 contacts 和 wrist_trajectory。返回 `bench_ik_query_v1`、对应 sequence／sim_time、每手 solution_found、从实测关节种子求出的候选关节表、FK 核对几何和残差或失败诊断。

查询只使用最新已发布的机器人实测关节，不读取物体真值、不推进物理、不改变控制器。回合结束、忙碌、快照过期或求解期间状态改变返回 409；几何不可用或目标不受支持返回 422。它不预检动作时长，不检查轨迹或碰撞；找到几何解不意味着能安全执行，没找到也不证明全局不可达。实际动作仍需走 actions 的校验和独立执行规划。

Agent 公共函数客户端已接入机器人说明与只读 IK，并校验返回的回合及观测绑定。BenchHTTP 遍历全部公开相机，能够消费腕视角；通用 RGB-D 运算由 SDK 提供。场景接入和剩余边界见[交接说明](../../docs/agent_bench/scene_integration.md)。

## 公开 RGB-D 与接触反馈

每路 CameraObservation 提供 RGB URI/SHA、宽高、sim_time、intrinsics_3x3、world_from_camera。光学坐标 x 右、y 下、z 前，四元数 wxyz。HTTP worker 发布场景解析后的完整 camera_ids，包括声明的 world 和 link／腕相机。`agent_rgbd_v2` 逐路固定 width、height、kind、role、mount_link，保留顶层 rgb 清单和 depth 声明；没有自定义 cameras 的旧场景继续使用 v1 两路格式，旧 v1 记录仍可读取。

内外参来自该 RGB-D 帧同一 render product 的 RTX 矩阵；移动腕相机不读取可能滞后的 USD 位姿。服务核对实际相机清单、逐路尺寸与观测时间，渲染诊断仍留在 private。原生 HTTP 相机证据绑定各自源码、冻结发布和运行配置，见[交接说明](../../docs/agent_bench/scene_integration.md)；旧房间／棋类相机证据只覆盖原记录配置。

depth_uri 指向同帧 float32 NPY，下载先校验完整文件 SHA，再用 `allow_pickle=False` 解码。深度单位米、语义 camera_z、0 为无有效命中；深度与 RGB 对齐并共享标定。不能从 observation sequence 推算 URI 中的 revision。

```text
z = depth[v, u]
x = (u + 0.5 - cx) * z / fx
y = (v + 0.5 - cy) * z / fy
```

通用深度校验／解码及坐标转换位于 [bench SDK](../sdk/README.md)。SDK 只计算公开数据，不扩大传感器权限，也不执行动作。

u/v 为零起始像素索引。深度来自 RTX 图像平面距离，不能以物体位姿伪造。RGB-D 仅在决策边界发布，执行中查询保留上一快照的原 sequence/时间；取图不推进物理。仅白名单中已发布 RGB/depth 可下载，教师中间帧与私有路径不开放。

机器人 proprioception 可含关节、腕部和指尖 FK，不含物体 oracle。单臂 `tactile`、双臂或具名实例的 `hand_tactile` 是最后完整控制窗口的模拟刚体法向接触代理。`action.contact_history` 覆盖动作的已确认窗口，分别支持单臂 v1、左右双臂 v2 和具名实例 v3；实际可用性仍由完整原生映射和采样证据决定。缺失值不能填零；手指闭合、法向接触和实际抓住物体是不同事实，服务没有通用 grasp_detected 判定。详见[触觉协议](../../docs/agent_bench/tactile.md)。

## System0 插件

策略实现 `bench.protocol.policies.System0Policy`：descriptor、reset、begin、step、cancel；step 返回 RobotJointCommand 和状态，不获得环境对象。注册到 PolicyRegistry 后由 create_app(registry=...) 提供。

descriptor 固定实现摘要、技能和输入权限；trained=true 还需权重 hash。当前仅支持 control_period_s=0.02。视觉策略可绑定只读 PNG reader；它不扩大 Agent 下载白名单，不允许反馈泄漏私有状态。已接收的抓握 student 未注册到此服务；独立的 [System0/System1 工程](../../system0-system1/README.md)及其历史结果使用自己的入口和记录。

有限 wall watchdog 可终止回合，但不能强杀可信 Python 线程。旧策略线程退出前禁止开始新回合，持续阻塞仍可能需要部署者结束所属进程。

动作、观测和策略等公共契约的 schema 位于 `configs/agent_bench/json_schema_bundle.json`，可由 `python -m bench.protocol schema --output ...` 生成；实际 Pydantic 还检查时间、标定、唯一相机等字段关系，JSON Schema 不代替关系校验。

场景 catalog 中的 `planned` 任务没有可执行 release。棋牌室 L1/L2 v4、Hex v11 已经由独立 loader 适配；`BENCH_TASK_RELEASE` 选择冻结发布后，通过 `/board-games` 创建两个认证玩家会话。普通 `/episodes` 动作入口拒绝这些双人对弈 episode，媒体也需要任一有效玩家令牌。玩家的公共函数复用同一动作账本；原生子步裁判及结果映射见[棋牌运行边界](../../docs/agent_bench/card_room_rules.md#agent-侧对接与已知操作边界)。Reference 仍留白。

棋牌室单臂 `task0` 使用自己的 `isaacsim-card-room-placement-release-v1` 冻结发布与普通 `/episodes`。省略 `seed` 时生成随机种子，显式传入 `seed` 时复现三个不同目标格；正式评测 slot 省略种子时使用其冻结种子。创建响应的 `spec.seed` 和 `task.episode_seed` 记录实际值，`task.goal` 与 `GET /episodes/{id}/task` 返回本回合目标。`GET /tasks` 仅公布九个候选格及抽样规则，不把发布模板目标当作当前目标。目标由冻结规则绑定，机器人和三枚棋子的物理初态保持固定。操作复用 `EpisodeSession.functions`，任务接入与 CPU/API 检查不代表原生抓放验收。

实验室五项 `sample_sorting`、`vial_racking`、`cap_placement`、`solid_weighing`、`pipette_transfer` 均通过普通 `/episodes` 和共用公开函数运行。部署者按 catalog 选择冻结发布，`get_task` 返回公开规则与粗粒度裁判状态；worker 回传自身加载的公开任务投影，host 在发布初态前核对一致性，缺失或漂移直接拒绝。真实物体位姿、接触、称量库存和液体账本仍只供私有裁判读取。原生初始化、物物验证与机器人完整任务成功分别记录，手物交互由数采验证。版本、启动命令及边界见 [Bench / Agent 检查入口](../../docs/agent_bench/scene_integration.md#bench--agent-检查入口)。

## 结果与正式评测

部署者可在启动冻结服务前设置 `BENCH_DEMO_FPS=10 BENCH_DEMO_CAMERAS=task,overview`，从 reset 开始记录原回合 RTX 画面；默认关闭。采样率必须是 50 的正整数因子，相机必须属于当前场景。原始 PNG、每帧仿真时间／评分／哈希及完整性说明保存在 `private/demo_recording/`，不进入 Agent 观测。逐相机渲染验证物理状态不变及 RGB-D 时间戳一致；捕获故障会明确终止，不默默补帧。

task0 回合完成封存后，使用 `python -m scripts.media.export_task0_demo --episode <episode_dir> --client <client_dir> --source-commit <40位源码SHA> --output <demo/interface/agent_demos/.../task0_2x.mp4> --playback-speed 2` 导出双视角视频。导出器校验原帧哈希、10 Hz 时间网格和终末结果，保留失败／恢复过程；倍率基于仿真时间，省略模型与计算等待。视频附来源 sidecar 和真实终末 PNG；该过程不启动模拟器或回放物体状态。

agent/spec、events、result、artifacts 和源码摘要公开记录协议；private 中保存状态、隐藏评分、policy_steps、原生日志与物理审计，默认 API 不提供读取入口。

终末取图或合法终末观测生成失败仍需封存并报告 infrastructure_error/invalid，不能把旧图改成新时间或补造未 ACK 指令。服务的任务成功必须同时满足 `outcome=success`、`physics_valid=true`；独立评分、接触和归档核验另列，不由模型 stop 或自述决定。

声明连续评分的普通任务还返回 `EpisodeResult.scoring`，包含 `score`、`score_unit`、`score_valid` 和声明过的指标；运行中只公开静态 `scoring_contract`，数值在终局通过结果和 `get_task` 返回。正常取消或预算结束可以保留此前合格分数，不能仅凭 `outcome=cancelled` 将连续成绩算成二元失败；物理或证据失效时保留审计值但撤销评分资格。场景私有中间计算值不随该接口公开。

双人终局也受同一预算边界约束。预算外到达的完整 native ACK 仍更新机器人观测、驱动目标和动作账本；其原始裁判终局与评分保存在 private，但不得产生公开有效输赢。服务从最后预算内裁判收据生成 board `truncated`／turn `interrupted`，移除迟到的胜负或和局，另存 `private/late_game_receipt.json` 绑定原始收据与预算。恰到墙钟上限已属超时；预算内已确认的终局先停止活动时钟，后续归档耗时或重复 finish 不撤销它。结果合同拒绝预算终止与竞争终局组合；归档或物理审计故障仍可保留已接受的原生胜负作为审计记录，但不能计入有效成绩。

普通 POST 是开发模式。正式 `EvaluationPlan` 由部署者预先冻结并提供，slot 创建核对任务、机器人、源码、物理、相机、策略与资源 pins。源码路径或环境改变需重新生成。shared 资源声明不证明 GPU 独占；当前 seed 也不保证随机初态。

在线调用、预算和证据文件见[模型客户端](../../agent/model_client/README.md)。源状态、动作完成、任务成功、物理有效和正式资格应始终分别报告。
