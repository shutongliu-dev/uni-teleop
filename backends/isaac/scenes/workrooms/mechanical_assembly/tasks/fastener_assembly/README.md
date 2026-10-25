# 散装螺栓装配

**2026-10-20 默认任务更新：** `fastener_assembly` 的 catalog `source_release` 改为 loose_bolt_candidate_v10_pgs（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/releases/loose_bolt_candidate_v10_pgs/release_manifest.json`），manifest SHA256 `47e574e81558ab16e7fffcda9846f2ab8c281660b0c85bb9ee507b95c94f262d`。默认操作目标为 **拾取螺栓 → 对准并起牙 → 工具接合 → 旋拧 → 释放**；螺栓与工具从独立桌面供料托出发。v10 仅把 v9 的场景求解器 TGS 改为 PGS，0.5 ms 步长、资产、reset、规则、阈值及共享控制保留，全局 TGS 默认不变。发布仍为 `prepared_unvalidated`、`physics_accepted=false`、`task_policy_success_claimed=false`，尚无完整机器人操作成功。

旧预插入发布 `rules_v1/fastener_assembly`（manifest `6d2c6150c78f9f0cc04743ce39d326d5d0ea5602a3ee529d200358a7b8e35c8a`）及 loose v7／v8／v9 和证据原样保留，作为历史对照。2026-10-20 的 `mechanical_bench_check_20261020/l4_http01` HTTP smoke 绑定该旧预插入发布；另一个 v8 HTTP smoke（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_l4_loose_default_20261020/l4_http01/qualification.json`） 因 reset 穿透失败。两份同名记录均不能授予 v10 资格。

v10 的 PGS 预接合接触报告（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/contact_pgs_native01/result/report.json`）通过 8,800 个原生子步的导程、初末被动稳定、穿透、真实接触链及物性检查，退出 0、零原生错误、零 fallback；导程为 3.064001 mm/圈，误差 +2.1334%。完整运动审计（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/contact_pgs_native01/complete_stream_motion_audit.json`）覆盖 8,760 个连续滚动 20 ms 区间和 80 个阶段边界区间，螺栓原生峰角速 0.513870 rad/s，低于冻结 0.523599 rad/s 上限。v10 HTTP smoke（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/pgs_http01/qualification.json`）通过四路 RGB-D、只读 IK、关节微动／回位／保持，3 个公开动作／0.92 s／1,840 子步，`physics_valid=true`，worker／server 退出 0，`model_calls=0`。冻结复算（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/pgs_replay01.json`）保持零信用与无啮合稳定基线；供料支撑复核（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/pgs_support01.json`）确认末 0.5 s 螺栓／工具各 1,000 子步均有向上承托，连续缺失 0。完整入口见PGS 交付索引（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/qualification_pgs01.json`）。预接合工程 reset 不证明散装拾取、起牙、插刀或机器人旋拧；HTTP smoke 只验证供料初态与基本接口。

历史 v9 原生 HTTP smoke（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_l4_loose_default_20261020/l4_http02/qualification.json`）及其冻结复算（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_l4_loose_default_20261020/replay_audit02.json`）通过初始化／基本接口，但 v9 的 TGS 预接合接触诊断（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/contact_native01/result/report.json`）整体失败：初始被动稳定、最终释放未通过，完整运动审计（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/contact_motion_audit01.json`）还发现加载螺栓原生峰角速 1.251275 rad/s 超限。v10 通过不覆盖这些失败，旧索引（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_l4_loose_default_20261020/qualification.json`）保留原范围。

## 历史预插入短行程任务（非默认）

历史预插入版本使用保留源几何的 SDF／凸体分区、PGS 和共同相位 reset，状态仍为 `prepared_unvalidated`。独立真实工具加载的 `native18/19` 在 0.35 rad/s 目标、0.5／0.25 ms 两步长下通过既定导程、原生速度和释放稳定性检查；半步长更细姿态差分超界及约 9.17° 的末姿态差仍保留，不宣称瞬态收敛。该历史目录使用规则一致的 **0.5 ms** 发布，见[场景验证与限制](../../../../../docs/scenes/mechanical_workshop.md)。控制器和机器人操作由共享 bench/agent 负责，本次交付不附加机器人成功声明。

任务继续使用同一 `mechanical_workshop_20261017_v3` 父环境和 `mechanical_single_arm_v1`，基座为 `(-0.23, 0, 0.78)m`。源 `threaded_sharpa/v3` 的几何、缩放、质量和材料保留，原文件不改写；新碰撞表示、任务专用 PGS 配置和共同相位初态独立冻结。桌面包含独立螺丝刀、被动螺丝和固定底座。预啮合螺丝和预插工具是明确的 reset 简化，不计为 Agent 成绩。正式执行仍只接受机器人动作，无焊接、物体外力或运行中位姿写入；另行授权的工具外力诊断不计分。

冻结任务要求先稳定至少 0.5 s，再握住工具，俯视顺时针慢转约 **30°**，使螺丝按 **3 mm/圈** 的导程额外推进约 **0.250 mm**，最后松手稳定至少 0.5 s。45°／0.375 mm 是此前设计候选，不是当前目标。所有下述阈值是已冻结的工程候选，尚非原生验收容差。

- 成功需要实际螺丝净转角 25–33°、新增轴向推进至少约 0.2083 mm，且全程角度／进给残差不超过 0.050 mm。
- 正向增量只有在同一 20 ms 窗口内，至少一半原生子步同时存在手—工具、工具—螺丝、螺丝—底座的有效接触时才计分。所有反向增量都扣回，保留信用不能超过实际正向净转角，且最终至少保留 25°。无链微动不加分，但允许重抓、回退和纠正。
- 预稳定和释放稳定分别要求连续 0.5 s 的位置变化范围不超过 0.1 mm、姿态变化范围不超过 1°；20 ms 位姿差分速度不超过 0.002 m/s 和 0.02 rad/s。原生瞬时角速度另限于 0.3 rad/s，以区分求解器瞬时抖动与持续姿态变化。
- 脱轴、导程不一致、超速、机器人直接碰螺丝等硬门禁保留。工具可留在孔中；最终不能仍由机器人接触支持。

`SequenceEvaluator.update(observation)` 读取标准 worker 每 20 ms 的真实物体位姿／速度，以及完整的 40 个 0.5 ms 原生子步接触与采样收据。不使用工具角度、接触数量或动作完成状态直接判成功；它也不测量扭矩、不声称拥有所有子步的物体姿态。固定底座模式／位姿、实际对向握持及公共函数执行还需要独立原生记录核对。

## 当前默认：Loose bolt v10 PGS

The task module exposes `build_task_inputs(reset_variant='loose_bolt')` and
`build_loose_task_inputs()` for authoring this reset. The catalog now binds this
independent prepared candidate as the default task: the
bolt starts as a free rigid body on the original table inside its annular nest, and the
driver starts separately on the table. A shallow annular entry funnel is a
task-owned static fixture placed above the authored thread mouth. It provides
coarse centering only; it does not snap, weld, track pose, or create a screw
joint. The public goal requires pickup, thread start and tool insertion, all
uncredited until a settled bolt-fixture/tool contact baseline is recorded.

The candidate asset catalog is
`<开发存储>/assets/task_parts/mechanical_assembly/fastener_assembly/loose_bolt_v5`,
with manifest SHA256 `75c0d7f879b1dd692cd67c391bb90f98b7641186b3729e8efdea2f7cd93b37f9`.
The current default release is
`<开发存储>/datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/releases/loose_bolt_candidate_v10_pgs`
with manifest SHA256 `47e574e81558ab16e7fffcda9846f2ab8c281660b0c85bb9ee507b95c94f262d`.
It retains v9's task-owned bolt supply asset and reset: a dedicated
annular bolt nest retains its 64 ring pieces and omits the two unrelated tool
side rails. The bolt head starts 0.2 mm above the original tabletop, the
specified vertical support through the open nest centre. The nest provides
passive lateral guidance with 0.5 mm source-mesh radial clearance at reset.
The tool rest and entry-guide
geometry are retained. The sole physical change from v9 is the explicit PGS
scene solver selection; the 0.5 ms timestep, shared controls, scoring thresholds,
reset and threaded source assets are retained. The global solver default stays
TGS. v9's TGS stability and motion failures remain frozen; v10 has its own
contact and HTTP evidence and does not inherit manipulation qualification.
It uses the source-preserving threaded collision candidate
`<开发存储>/assets/task_parts/threaded_sharpa/sdf1024_convex_partition_candidate02_20261018`
with manifest SHA256 `d54448d6087144ed991c60b35ef598bfbf096a980d3f2fde4027daddb67c3238`.
The task remains `prepared_unvalidated`; changing the catalog default does not qualify its geometry or manipulation.
Prepared releases are written under the shared datasets tree with
`prepare_task_release(..., reset_variant='loose_bolt')`, and remain
`prepared_unvalidated` until privileged native reset, pickup, guidance and
thread-contact checks are complete. The historical preinserted release and its
rules remain available for explicit comparisons, with their frozen bytes unchanged.

v10 沿用 v9／v8 修订后的 loose 接触规则：尚未建立被动稳定基线前允许手部直接接触螺栓，用于拾取、对准、起牙和插工具；这一阶段没有旋入信用，包括零冲量接触流在内的机器人接触都会重置被动稳定窗口。基线建立后，超过冻结冲量门槛的机器人直接接触螺栓仍触发失败，非手部机器人接触螺栓或工具始终受禁止接触门禁约束。轴偏移／倾角硬门从首次完整啮合窗口起生效，不等待稳定基线。历史 v7（manifest `8daab1293d5872cc62a28602c93c2ea009ad1608232b85f375716ecf1bb4c248`）没有拾取阶段豁免，保留其 `direct_robot_bolt_contact` 阻断及原生报告。旧 preinserted 的禁止接触语义不变。目录切换、规则修复及 CPU 几何检查不代表完整操作链通过；具体合同见[统一规则说明](../../RULES.md)。

历史 v8（manifest `b2c6968e7253107ae1c11d4ee16e8db1be4104341801d229987d8649b02a2d2a`）仅修规则，继承了 v7 `bolt_supply_rest` 中误带的 `tool_side_rail`：轨道内侧为 x=±20 mm，螺栓头半径为 22.5 mm，reset 已有约 2.5 mm 穿透。v8 原生安全回执（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_l4_loose_default_20261020/l4_http01/episodes/7c725e725dc44e73806b39abd6cd5f9e/private/physics_safety_stop.json`）记录首 20 ms 窗口最大报告穿透约 3.009 mm，超过原 2 mm 门槛；裁判为 `safety_stop/reported_penetration_bound`、`physics_valid=false`，尚未建立稳定基线，HTTP smoke 未通过。这是供料几何／reset 失败，不是完整操控尝试；冻结 v8 和失败记录保留，v9 结果须另行绑定。

| 文件 | 职责 |
|---|---|
| `definition.py` | 目录注册、版本指针及未验收状态 |
| `task.py` | 任务自有场景／初态和静态描述，`build_task_inputs(target_turn_degrees=30.)` |
| `rules.py` | 版本化公开规则及工程候选阈值 |
| `evaluator.py` | 标准 `SequenceEvaluator.update` 评分接口 |
| `release.py` | `prepare_task_release(output, target_turn_degrees=30.)`，仅写全新 datasets 目录 |
| `motion_plan.py` | 原先的独立夹柄诊断计划，保留历史范围，不作为当前任务控制入口 |
| `public_motion_plan.py` | 公共函数运动候选；按实际 80% 插值段及公开速度限制计算动作时长 |

历史预插入冻结目录为 rules_v1/fastener_assembly（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_referee_20261018/rules_v1/releases/fastener_assembly/release_manifest.json`），manifest SHA256 为 `6d2c6150c78f9f0cc04743ce39d326d5d0ea5602a3ee529d200358a7b8e35c8a`。该历史版本仅升级规则与裁判：完整中文合同、公开阶段／原因、可恢复条件、零冲量接触的松手判定及无效回执分类；[统一规则说明](../../RULES.md)解释其接口和终局含义。物理场景、初态、资产清单、物理配置和评分数值阈值均保持原共同相位候选 `task_release_sdf1024_convex_partition_pgs_phase_reset_candidate01`（`94bb1be47dc2a9bdd5cfa402bf96162a6c168a168f0766c6fa19196806faa88c`）不变；该次 CPU 检查不增加原生任务资格。标准加载器对应核查（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/fastener/canonical_delivery_review01/report.json`）确认该历史 reset、scene 物理字段和机器人初态与 `native18` 一致，规则步长同为 0.5 ms，无工程探针 reset。旧 v1–v4 发布及所有失败记录保留。

以下为历史候选与失败记录。显式选择 `reset_variant='preinserted'` 时，`build_task_inputs`／`prepare_task_release` 的 `tool_clocking_degrees=-30.` 生成旧 v4 reset，`0.` 生成旧 v3 初态。历史对照须显式加载对应冻结发布；当前默认执行读取上面的 loose v10 PGS。authoring 函数的默认 `reset_variant` 仍保留 preinserted 以便复现历史输入，catalog/runtime 才决定当前默认发布。原预插工具位于六角配合的反转侧，源几何计算得到约 31.75° 的旋入空程。新候选仅在 reset 绕工具自身轴调整角度，使空程约为 1.75°，工具原点／轴线、螺丝、机器人和物理参数不变；该初始化不计任务信用，必须重新验证静置、握持和旋拧。候选已另存为 task_release_v4（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_fastener_functions_20261017/task_release_v4/release_manifest.json`），已在 native04（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_fastener_functions_20261017/native04/launch.json`） 完成新抓握动作测试，尚未 `task_qualified`。源 `threaded_sharpa/v3` 资产、base／ready、physics、rules 和 evaluator 均保持不变；release 版本与共享资产版本相互独立，v3 发布和历史记录保留。

用户后续明确本阶段只核验能否真实旋入，不再追加追满 30°及释放的策略调参。native04 有限旋入证据（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_fastener_functions_20261017/independent_native04_rotation_advance_evidence.json`）确认真实机械手夹柄后发生了螺丝旋转和向下位移：最后有效 18.00 s，相对 1.94 s 静稳基准，转动 10.51594°、推进 0.059783 mm，末 20 ms 的三段接触链同步正压为 38/40 子步。18.02 s 保持阶段因螺丝原生角速度 0.59519 rad/s 超限终止，未释放，完整任务仍失败。

**旋入现象不等于导程准确匹配。**现有轨迹核对（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_fastener_functions_20261017/native04_rotation_advance_consistency.json`）显示：6.904°时理论／实际推进为 0.057533／0.055432 mm；10.516°时为 0.087633／0.059783 mm，末值少 0.027849 mm（约 32%），后段转动继续而下降趋平。落在旧 50 µm 绝对容差内不能表述为相对导程准确。`native05` 已按用户收窄后的范围受控取消，不作为完整通过证据；没有启动 `native06`，新预载与慢转策略未合入主线或运行。

导程偏差专项核查（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_fastener_functions_20261017/lead_mismatch_check01/report.json`）离线复算了原始位姿、接触和源 USD：源几何导程约 3 mm/圈，缩放正确，所查末段没有行程顶死。15.96–18.00 s 再转 3.612°，理论应下进 30.10 µm，实际仅 4.35 µm；更换转角算法或轴线上测量点仍不能解释短缺。同期螺丝倾角从 0.155°增至 0.784°，末态轴原点偏心约 0.312 mm；约 17.02 s 起刀头相对螺丝持续轴向抬升，18.00 s 相对倾角为 0.553°，底面呈边缘承托。偏轴摇摆和接触载荷重分配是现有证据支持的候选解释，但持续抬升晚于下降趋平，不能单独确定为初因，也尚不能分摊摩擦、碰撞离散和求解误差的贡献。

同深度约 9.117 mm 的历史 v3 外部工具施力轨迹，转约 15°下进 0.12288–0.12315 mm，与实际转角对应理论值仅差约 2–3 µm，且偏心不超过 5.8 µm、倾角不超过 0.012°。这支持资产在较好同轴条件下能按导程旋入；历史施力方式、时间步及场景与本次不同，不能替代机器人对照实验。此次专项只读取既有数据，未修改资产、规则或历史结果，未新增原生运行；下降／旋转及倾角曲线（开发存储 `demo/interface/scenes/mechanical_fastener_functions_20261017/lead_mismatch_check01/rotation_advance.png`）保留全部原始 20 ms 采样。

只读目录及加载检查：

```bash
.venv/bin/python -m scenes.workrooms tasks --family mechanical_assembly
.venv/bin/python -m scenes.workrooms task --family mechanical_assembly --task fastener_assembly
```

运行使用 `bench.runtime.service.EpisodeSession` 和其 `session.functions.invoke(...)` 公共函数入口：`query_ik`、`move_wrist`、`set_hand_targets`、`move_joints`、`hand_preset`、`hold` 等均遵守同一动作、预算、确认和记录协议。具体调用参数取当前函数 catalog。canonical worker 通过 `bench.isaacsim.release.task_inputs` 和 `load_evaluator` 读取冻结发布；机械 schema 拒绝 `qualification_probe` 的目标态重置。上述只读 CLI 不启动模拟器。

`scripts.validation.validate_mechanical_functions` 是零模型调用、仅命令机器人关节的工程脚本。默认运行完整旋拧和释放；`--turn-degrees 0 --stop-after-grasp` 仅运行至握持保持，用于快速诊断。短诊断仍要求原生对向压力、完整采样、物理有效性和源码封存检查，但报告固定标为 `grasp_diagnostic_only`、`task_qualified=false`。从另外冻结的接近姿态启动的诊断不能替代原 ready 初态的完整任务验收。

review_v2 索引（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_fastener_functions_20261017/independent_native02_03_review_index.json`）记录源码 `1efb7e2`／release v3 的两轮完整复核。02 保持末 300 ms 对向接触 599/600，通过原握持门槛；13.46 s 第三段旋拧因螺丝原生角速度范数 0.54686 rad/s > 0.52360 rad/s 终止，主要为横向摆动，未执行释放／回位。03 保持末窗为 0/600，未启动旋拧；24.94 s 完成释放及 ready 保持，`open`／`final_hold` 通过，但 `retreat` 最后 100 ms 的关节误差峰值 0.01657 rad > 0.01 rad，仍失败。03 实际闭手命令峰值约 0.148 rad/s，不是严格 0.12 rad/s 对照。这些是公共函数工程实测，未覆盖 HTTP 传输或模型策略，也不证明完整任务成功。

CPU 报告（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_fastener_functions_20261017/task_cpu03/report.json`）记录 62 项正负例与 loader 回归通过。历史 `tool_motion04` 的 456 帧回放可在 0.5 s 建立稳定基线，旧夹柄失败没有被误判为成功；这只是旧记录的 CPU 重评分，不是当前公共函数或新任务原生通过。环境与历史动力学边界见[机械室证据](../../../../../docs/scenes/mechanical_workshop.md)，共用布局和其他任务见[家族说明](../../README.md)。
