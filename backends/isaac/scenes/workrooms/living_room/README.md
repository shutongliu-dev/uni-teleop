# 客厅任务

固定父环境沿用 `scene.py` 的 `living_room_residential_20261018_v7`。房间、空桌、装修和灯光不随任务重建；所有操作物体、初态、规则和评分属于各自任务。编号是任务 ID 对照，不是学习阶段，也不保证不同参数下难度严格单调。

| 编号 | 目录 | 机器人 | 核心任务 | 当前范围 |
|---|---|---|---|---|
| Task1 | `tasks/remote_operation/` | 单臂 | 遥控器开电视、连续开机 10 秒、关机后归架 | 电源实体按键；归架需松手稳定 300 毫秒 |
| Task2 | `tasks/water_pouring/` | 单臂 | 抓壶定量倒水、控制洒漏、放回水壶 | 已注册可运行候选：冻结 250 mL 水、原生粒子观察器与裁判 |
| Task3 | `tasks/tower_building/` | 单臂 | 自由搭高，记录回合内最高稳定高度 | 36 根木条、两侧实体供料托盘；不限塔形、层数、用料数或方向 |
| Task4 | `tasks/tower_extraction/` | 双臂、两方对抗 | 轮流抽一根放入己方托盘，倒塔方输 | 桌面上 12 层 36 根初始塔、双方各 15 格实体收集托盘 |

Task1／Task2 已注册为 `implemented`，`catalog.load_task` 可校验并加载各自冻结发布，仍是 `prepared_unvalidated` 场景候选。Task1 于 2026-10-19 修订为电视开机保持 10 秒、关机后归架任务；当前 v7 的原生入口与独立电视显示检查已通过，完整机器人操作尚未验收。旧 v3 两键任务记录仅作历史证据。Task2 的入口检查与机器人完整倒水操作分别记录。Task3／Task4 分别接入 `/episodes` 和认证 `/turn-games`。2026-10-19 新场景使用显式 `trays_v1`：Task3 木条为 15×4.5×2.5 cm，Task4 为 20×4.5×3 cm，任务发布分别为 v5／v8。实际加载路径和哈希以各 `definition.py` 为准。CPU 合成观测测试和物体静置不代表机器人已经搭高或抽塔成功。

## 目录与接口

```text
living_room/
├── scene.py                 # 原固定环境及哈希
├── task_design.py           # 任务元数据与共同物理约束
├── tower_geometry.py        # 明确尺寸后的层/槽/木条编号
├── tower_assets.py          # Task3/4 木条与显式版本化资产作者
├── tower_trays.py           # 供料/收集托盘布局、实体底板与围边
├── task_scene.py            # 静态展示场景组合，历史图像入口
├── composition.py           # 真实 task_layers、机器人布局与reset
├── release.py               # Task1/2 bundle 冻结发布
├── layer_release.py         # Task3/4 task_layers 冻结发布及nominal reset
├── runtime.py               # 私有native采样和回合授权
├── evaluation.py            # release冻结评分器
├── scene_candidates.py     # 四套初始 USD/manifest 哈希及证据入口
├── dual_robot_layout.py     # 原尺寸 FR3 + 右 SHARPA 两侧布局
└── tasks/
    ├── remote_operation/    # definition.py + task.py + rules.py + evaluator.py + assets.py + README.md
    ├── water_pouring/       # 同上；真实 PBD 粒子和空腔容器
    ├── tower_building/      # 上述文件 + evaluator.py
    └── tower_extraction/    # 上述文件 + evaluator.py
```

使用现有 `scenes.workrooms.catalog` 发现任务，不引入另一套注册表。`definition.py` 的 `task_number` 固定编号，语义目录名保持其他工作间的习惯。Task1／Task2 的 `task.task_configuration(seed=...)` 验证发布 manifest，再加载真实 canonical `scene/task/initial_state`，同时保留顶层 `environment/rules/public_goal` 便于检查；调用本身不启动模拟器。Task2 仅默认 250 mL 初始水、100±10 mL 目标和 5 mL 洒漏预算对应当前发布；任何不同的 `quantities` 都返回 `design_only_proposal`，没有 source_release、没有原生初态且 `native_execution_available=False`。也可显式调用 `water_pouring.task.design_configuration(...)` 研究参数。Task3／Task4 通过目录加载固定尺寸的运行发布，设计接口仍可返回显式尺寸的几何配方。

当前 Task1／Task2 使用 scene 正式发布的 front 单臂，基座 `(0.65,-0.76,0.78)m`；Task3 使用 `robot_layout.py` 单臂 `(0.65,-0.88,0.78)m`；Task4 使用 `dual_robot_layout` front/rear，两基座 `y=±0.76m`。Task4 通过 `fr3_sharpa_instances/namespaced_v1` 组合两套原始右 SHARPA。原装配、模型比例、控制增益不变，历史候选布局及其证据保留。

```bash
.venv/bin/python -m scenes.workrooms tasks --family living_room
```

## Task1／Task2 冻结发布

发布位于 `<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/`。Task1 统一使用电视开关并归架的新目标，旧两键序列任务已从当前入口移除。当前冻结发布如下：

| 任务 | 发布目录 | release manifest SHA256 |
|---|---|---|
| Task1 | `remote_operation/v7` | `b644bc3bf5fc0513db6295aea8785d7649d3b92b816be9e3245bdadecffb664b` |
| Task2 | `water_pouring/v5` | `83e6616a3ef22864139298ccdd36f07345c532935c63ca67fb3cd0ab4eaaffd5` |

两项复用普通 canonical `/episodes` 与公开机器人函数，冻结 `scene.json/task.json/initial_state.json/rules.json`、原生观察器与裁判源码以及父环境／共享资产哈希。`implemented` 表示可加载的完整运行候选；manifest 的 `physics_accepted=false` 和 `task_policy_success_claimed=false` 保留，本轮 native smoke 与实际机器人操作证据另外附加，不能改写冻结发布为已验收。

## 本轮发布验证与使用

Task1 v7 绑定冻结 `source03`（manifest `2e7484fa8b41a577ce084b457396adec30b277aa940ad1656fbf4fabaa70c027`）。原生入口检查（开发存储 `datasets/fr3_sharpa_bench/development/living_task1_tv_20261019/native03/qualification.json`）通过 300 ms／600 子步的公开函数保持、16 次裁判观测和四路 RGB-D，电视保持关闭且零得分，`physics_valid=true`、worker 退出 0。独立显示诊断（开发存储 `datasets/fr3_sharpa_bench/development/living_task1_tv_20261019/display01/display_diagnostic.json`）确认关／开／关三张真实 RTX 图及屏幕深度，实际观察器和物理数组未变，重置后推进物理 0 步、机器人动作和模型调用均为 0，进程退出 0。其状态仅为显示夹具，不构成按键、10 秒等待或归架操作证据。最终 209 项针对性 CPU 回归通过；发布与源码、审阅及后续目录检查统一索引于验证记录（开发存储 `datasets/fr3_sharpa_bench/development/living_task1_tv_20261019/qualification.json`）。完整机器人开机、等待、关机与归架仍未验收。

Task2 v4 的新入口验证（开发存储 `datasets/fr3_sharpa_bench/development/living_water_coupling_20261018/publication/water_pouring01/qualification.json`）已通过同样的 300 ms／600 子步检查，488 粒留壶，裁判无误得分，四路 RGB-D 正常。资产 v2 保持 250 g 真水，改用 8 mm 离散并关闭两容器休眠；独立倒水台架达到 93.75 mL 且无洒漏。满壶水重反作用误差约 +4.4%，该杯量下约 +22.7%，按用户要求作为实用近似，完整机器人倒水未验收。新版实际相机图（开发存储 `demo/interface/scenes/living_water_coupling_20261018_v4/index.html`）。

Task2 v5 沿用 v4 的物理资产，补充水差与洒漏报告：初始水量减壶内、杯内及未决水量得到当前洒漏，累计洒漏仍限 5 mL，洒后回收不退额度。新冻结裁判的原生入口检查（开发存储 `datasets/fr3_sharpa_bench/development/living_water_coupling_20261018/publication/water_pouring02/qualification.json`）通过，报告含完整水量账本；v5 实际相机画面（开发存储 `demo/interface/scenes/living_water_coupling_20261018_v5/index.html`）。

历史 bench 接入检查覆盖 Task2 v5、Task3 v4 等候选：各通过 0.42 s／840 原生子步的公共微动与四路 RGB-D，进程和 worker 均退出 0。Task2 v5 的两次 reset 还验证了 488 个原始粒子 ID 及持续更新的真实位置／速度。检查通过进程内 ASGI HTTP 调用真实独立原生 worker，未调用模型；它们验证接口与读回，完整按键、倒水和搭塔仍未验收。当前及历史结果分别绑定源码与发布，见本轮集成报告（开发存储 `datasets/fr3_sharpa_bench/development/living_bench_integration_20261018/report.json`）。

侧前方任务相机避开待命机械臂对桌面物体的遮挡；旧 v1/v2 发布和失败记录保留。Task2 第二次复位的原生数据缓冲设备兼容问题已修复；运行时遵从实际 `SimulationView.device`，没有修改物理材质、粒子质量或控制器。

```bash
.venv/bin/python -m scenes.workrooms tasks --family living_room
.venv/bin/python -m scenes.workrooms task --family living_room --task remote_operation
.venv/bin/python -m scenes.workrooms task --family living_room --task water_pouring
```

服务通过 `BENCH_TASK_RELEASE` 选择完整冻结路径后，使用现有 `POST /episodes` 和公开机器人函数；`GET /episodes/{episode_id}/task` 返回本回合目标与规则。Task1 当前选择 `remote_operation/v7`，Task2 为 `water_pouring/v5`。Agent 入口采用 `--family living_room --task task1` 或 `task2`，校验目录发布 hash。任务恢复、预算耗尽和观测故障的语义均随 `rules.json` 冻结。

## 自由搭高与对抗抽塔

Task3 的新任务目标是**回合内曾达到的最高稳定高度**。两侧供料托盘各放 18 根木条，3×3 列、每列两根；中央原桌面为空，不设置塔形、目标层数、每层数量、朝向或用料数量条件。计分采用实际定向木条的最高点相对桌面的高度，要求相连搭建有真实支持链落到原桌面、脱手且稳定至少 1 秒；托盘叠料、手持或飞起的木条不计分。已确认的最高高度在后来倒塌后保留，允许恢复继续搭建。余料不必全部用完；无效证据和预算结束独立报告。详见 [Task3 自由搭高](tasks/tower_building/README.md)。

Task4 的初始塔继续直接放在原桌面上，采用 20×4.5×3 cm 木条、12 层共 36 根、相邻层交错 90°，高度 36 cm。同层横向缝隙为 10 mm，每端超出交叉层 22.5 mm；原候选分别为 1 mm 和 6.5 mm。双方各有 3×5 格实体收集托盘，单层容量 15 根，与默认各 15 回合匹配。初始塔与 Task3 的自由搭建得分相互独立。

两项任务的托盘均有 6 mm 厚底板和真实碰撞围边，承托面 `z=0.786m`，边缘顶高 `z=0.798m`，作为明确声明的静态任务夹具。底板和围边采用独立接触身份，围边不能代替底板承托得分。Task4 塔与最近托盘外缘有 64 mm 净距，最低可抽层底面高于托盘边缘 12 mm。几何净距不等于机器人可抓或无碰撞轨迹验收。托盘版本本身保留原房间、桌面、机器人装配与共享控制器；Task4 随后的数值积分候选单独声明如下。

当前资产为 Task3 `tower_building/v6` 和 Task4 `tower_extraction/v3`；任务发布为 `<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/` 下的 `tower_building/v5` 与 `tower_extraction/v8`，精确哈希见下表及各 `definition.py`。省略资产 `profile` 或自由搭高 `scoring_mode` 保留历史作者／规则默认行为。

Task4 采用已经确认的“抽走并收集”，不放回塔顶。底层以及当前最高完整层和其上方受保护。先手由种子交替，双方收集容量、回合额度一致；同一塔采用前后对称的机器人位置。合法动作可以推、拉和重新抓握，不能撑住其他木条。当前木条在己方区释放、双手离塔、剩余塔稳定满足窗口后才交接。缺证据、非当前方物理干涉、无明确操作归因的倒塌均不能随意判输；具体状态机、边界和例外见 [Task4 规则](tasks/tower_extraction/README.md)。

历史 Task3 v4／Task4 v5 位于 `development/living_bench_integration_20261018/qualification_review/releases/`，其资产和失败记录不改写。旧 Task4 front 的原生检查跑满 10 s 后返回 `invalid_reset_not_stable`，未进入机器人微动或抽取；rear 在共同初态问题确认后于 6.62 s 工程取消。`physics_valid=true` 不能代替场景稳定判据，这项失败既不转为通过，也不能被未验证的新几何覆盖。原生证据必须绑定实际源码、资产和发布版本。

## 新发布与原生图片（2026-10-19）

| 任务 | 当前任务发布 | release manifest SHA256 |
|---|---|---|
| Task3 | `tower_building/v5` | `2fe24d4ff529cdbdc60afd7f0fda024110dae559af054cb5cd1f0977500b287f` |
| Task4 | `tower_extraction/v8` | `cc31b454e8a1421bbbc85bbfdd72f615b2a585652150bed3da1ca40141644499` |

本轮托盘与木塔图库（开发存储 `demo/interface/scenes/living_trays_towers_20261019/index.html`）汇集两项近景、全景和图片来源；独立资格记录（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/qualification.json`）为本轮证据入口，各项结论以绑定版本的原始报告为准。出图与 canonical 原生入口检查使用独立冻结源码，精确身份如下：

| 用途 | 源码 manifest | manifest SHA256 |
|---|---|---|
| RTX 布局出图 | source01（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/source01/source_snapshot_manifest.json`） | `341e66bc32ea16ac25cf0cc9f493d4689f42b1fda6906267178451bf213285ca` |
| Task3 v5／Task4 v6 历史 canonical 检查 | source02（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/source02/source_snapshot_manifest.json`） | `d993bf55984a358b24027e16184d2af9bcb725bd3ac17d72ac8a989f89bd079c` |
| Task4 v8 canonical 检查 | 修复 source02（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/source02/source_snapshot_manifest.json`） | `6ad8ababa6b216297772d18dccb239ce37585aeda8a91ba72ef58cd494fe1b7f` |

Task3 桌面近景（开发存储 `demo/interface/scenes/living_trays_towers_20261019/tower_building/tabletop.png`）与 Task4 桌面近景（开发存储 `demo/interface/scenes/living_trays_towers_20261019/tower_extraction/tabletop.png`）均已由真实 RTX 输出，并保留同目录全景和 `provenance.json`。两项各记录 4000 个 0.5 ms 显式物理步，图片均为 1920×1440、实际 128 spp；使用上表 `source01`，动态对象按原生 PhysX 状态读回，物理结束后才导出 USD。

图片中的机器人是静态展示模型；Task3 图片沿用 front 基座 `y=-0.76m`，实际 v5 任务发布使用单臂 `y=-0.88m`，二者不能视作同一机器人运行布置。Task4 图片显示两台原始右手机器人和桌面上的完整初始塔。两项图片验证物件布局与出图，不证明 canonical reset 裁判通过、抓取或整项任务成功。冻结 manifest 均保持 `prepared_unvalidated`、`physics_accepted=false`、`task_policy_success_claimed=false`。

托盘布局首轮 canonical 原生入口检查使用独立 `source02`。Task3 的 native_building01 报告（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/native_building01/report.json`）为 `passed=true`：v5 加载、四路 RGB-D、小幅手腕动作及保持完成，公开仿真时间 1.38 s，关节实际最大变化 0.00569 rad、手腕目标误差约 0.137 mm，`physics_valid=true`，以 `cancelled` 正常关闭。托盘接触复核（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/tray_contact_review01.json`）确认全部 2760 个原生子步中 36 根供料均有托盘底板支撑链；其中 `t>=0.05s` 的 2661 个样本确认 18 根直接落底板、18 根由下层木条承托，搭建高度全程为 0。该检查仅覆盖入口、控制及证据接线，没有取料或搭高成绩，`task_success_qualified=false`、`physical_fidelity_qualified=false`。

Task4 v6 的 native_extraction01 报告（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/native_extraction01/report.json`）为 `passed=false`：front 先手的回合 `5aef9574d0fe4a44ba167c74cd953e4a` 跑满原定 10 s／20000 个原生子步，以 `task_failure` 结束；冻结评分为 `interrupted / invalid_reset_not_stable`，游戏 `abort`，双方完成回合数均为 0、未收集木条、无胜负。正式回合结果记录 `physics_valid=true`，五项必需通用物理审计均通过，采样取得与交付各 20000 次、无重用。smoke 在最后一次 `reset_wait_49` 收到 `episode_terminated` 后报错，未进入手腕微动或抽取；这是场景 reset 未稳定，不是墙钟或动作超时。该 smoke 报告未含 `physics_valid` 或 `worker_exit_code`，不能将缺失字段解释为 `false` 或某个退出码。

Task4 reset 复核（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/task4_reset_review01.json`）每 50 ms 检查一个相邻原生帧区间，共 200 个抽样点：全部有运动门槛失败，每次 24–28 根木条超出稳定运动限制；这些抽样点的支持、原槽位、深接触、双手离塔与倒塌检查均无失败。此诊断不声称逐帧支持或完整稳定窗口通过。该 v6 canonical 结果仍为 reset 失败，空收集盘未验证承载放置；v6 结果与上面的历史 v5 失败分别保留。该轮检查未调整房间、控制器、物理参数、稳定门槛或一秒窗口。

后续 v7 只打开 TGS 每轮外力、仍为 0.5 ms。独立筛查与冻结回放（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/canonical_review01.json`）覆盖基线及 2800 子步，1.4 s smoke 预算内始终 `reset_pending`，最长连续有效端点窗口 7.5 ms；未进入微动。这是预算内未就绪，不是跑满裁判 10 s 后的终止判定，保留为 v7 独立历史。

Task4 当前 v8 候选显式选择 `tower_extraction_tgs_external_forces_v2`：TGS 每轮施加外力，物理步长 0.25 ms；开关与步长作用于整个 PhysicsScene，包含双方机器人。公开控制周期仍为 20 ms，每次 ACK 内的原生步从 40 增至 80；原重力补偿算法仍在每个子步读取并施加力矩，因此它的采样间隔也变为 0.25 ms。机器人几何、质量／惯量、增益、控制算法、材质、迭代数、任务规则和稳定阈值保持原定义，但离散积分已经改变。半步诊断只支持候选选择；下述独立 canonical smoke 通过不覆盖 v5/v6/v7 的既有发布与记录，也不提供抽取、托盘承载放置、换手或完整对局的成功资格。

v8 canonical 报告（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/native_extraction02/report.json`）为 `passed=true`：seed 0、`player_0/front`、回合 `eeef751dc81148f3a370fbd92fa00166`，绑定上表修复 `source02` 和 v8 manifest。公开状态首次在 1.2 s 观察到 `await_intent`；之后完成初始保持、请求上移 2 mm 的手腕微动和最终保持，仿真时间共 1.6 s。实测目标误差 0.224 mm，最大关节变化 0.005476 rad，五路 RGB-D 均实际获取并核对哈希，`physics_valid=true`，worker 退出 0。独立冻结回放（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/canonical_review02.json`）逐帧核对 6401 帧（含基线）、6400 个连续新鲜原生采样和 81 个控制观测；原裁判在 tick 4003／1.00075 s 首次进入 `await_intent`，此前 tick 3–4003 的 4001 个有效端点形成 0.9999999999999178 s 窗口，按原有 1e-12 s 浮点容差通过完整 1 s 稳定条件。控制状态与原生归档逐项一致；这是对既有记录的只读回放。另有 28 项元数据检查全部通过，统一见修复资格记录（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/qualification.json`）。以 `cancelled` 结束，无机器人／木条接触、已完成抽取、收集、回合或胜负；未调用模型。该次 front 微动不代表 rear 先手或完整机器人任务验收。

## Task1、Task2 的设计理由

Task1 的当前目标是用遥控器开电视，连续保持开机至少 10 秒，再用遥控器关机。10 秒按原生仿真时间计量；提前关机后重新开机从零计时，不累加多段时长。翻正可用桌面辅助，电源键仍需同手握持中的真实按压和回弹；`confirm` 不参与目标；关机后须放回收纳架，手松开并稳定 300 毫秒才能完成。完整机器人操作资格独立记录。详见 [Task1 电视控制](tasks/remote_operation/README.md)。

Task2 已按用户要求改为水壶倒水：单臂抓壶、对准杯口、定量倾倒、止流并放回。当前冻结候选为初始 250 mL、目标 100±10 mL、累计洒漏不超过 5 mL。它增加负载下握持、腕部连续姿态和流量控制，符合客厅待客场景。采用实际液体质量计量；接水托盘中的水也记洒漏。候选作者使用真实 GPU PBD 粒子、开放壶嘴、杯腔和接水托盘；液体动力学诊断、已实现的原生评分接入与机器人任务成功分别记录，详见 [Task2 倒水设计](tasks/water_pouring/README.md)。

这四项分别覆盖单件定向与独立手指、持壶倾倒与液体控制、多件堆叠稳定、对抗下的精细抽取。旧杯垫摆放／遥控器归位／玩具分类只是未发布的占位定义，本轮用四项明确任务替代，不增加重复的目录任务；Git 历史保留原规划。随后被倒水替换的眼镜盒提案同样未发布，不保留重复目录。

## 接入与验收边界

Task1／Task2 的冻结 `NativeTaskObserver` 只接收 canonical worker 从原生 tensor、关节锚、接触或原始粒子读出的私有子步状态，普通 20 ms 裁判消费对应 `task_state`；Agent 不能上传自报物理真值作为得分。Task3／Task4 的 `runtime.py` 从同一原生子步刚体与接触缓冲构造私有归一化输入，交给冻结裁判。逻辑通过、场景启动与机器人完整操作分别记录。

Task1–3 使用通用 `/episodes` 和公开函数；Task4 使用认证的 `/turn-games`，玩家 `player_0/front` 与 `player_1/rear` 各控制一套机器人，意图仅改变裁判选择，不写入物体状态。Task1–3 及历史 Task4 v5/v6/v7 使用 0.5 ms 物理步，Task4 v2 数值候选显式改为 0.25 ms；公开控制周期均为 20 ms。旧展示／液体探针的 1/240s 或 1/960s 证据不转移为本配置资格。Task4 reset 稳定前只允许保持原 drive targets，真实支持、释放与稳定窗口通过后才能进入动作回合。完整机器人操作和物理负例仍需独立验收。

任务代码不得替代 Agent 策略或改变 Bench 控制器；共同的 practice／validation／evaluation 学习权限不变。

所有可复用模型仍放共享 `assets/task_parts/`，发布与实验放共享 `datasets/`，效果图放 `demo/interface/scenes/`；不将生成资产写进任务代码目录。

## 历史四套场景与原生图（2026-10-18）

下表记录 2026-10-18 的旧资产与图片，不表示新版托盘布局或自由搭高规则已经完成原生验证。每项 `assets.py:author_task` 只创建任务物体；`task_scene.py` 组合固定客厅 v7 与原尺寸 FR3/右 SHARPA。机器人在照片中为静态待命姿态，任务物体保留真实动态刚体／关节／粒子。`initial.usda` 总是任务初态；另行摆放的展示态不覆盖初态。

| 任务 | 共享物体资产 | 原生记录 | 图像状态 |
|---|---|---|---|
| Task1 | `assets/task_parts/living_room/remote_operation/v5` | `task1_native05` | 正面朝上的按键展示；背面朝上的真实 reset 单独保存 |
| Task2 | `assets/task_parts/living_room/water_pouring/v1` | `task2_native01` | 壶内粒子液体、空杯和接水托盘 |
| Task3 | `assets/task_parts/living_room/tower_building/v5` | `task3_native05` | 左右各 9 根散料、中心空；另保留三层搭建展示图 |
| Task4 | `assets/task_parts/living_room/tower_extraction/v2` | `task4_native02` | 六层 18 根初始塔、两机器人与对称收集区 |

所有相对共享路径均位于 `<开发存储>/` 下。原生记录根目录为 `<开发存储>/datasets/fr3_sharpa_bench/development/living_task_scenes_20261018/`；全景与桌面细节画册（开发存储 `demo/interface/scenes/living_task_scenes_20261018/index.html`） 包含每项两张 1920×1440、128 spp 的真实 RTX 图。各次 source 快照、native 日志、对象状态、USD、图片哈希独立保留，失败尝试不改成通过。

Task1／Task3／Task4 各记录 480 个显式物理步（1/240 s）；Task2 在 1/960 s 下记录 2880 步，图片对应的初末两次粒子账本都是 9259 粒留壶、杯内及壶外为零，另有逐步三秒保水证据。液体的无平滑物理点与渲染等值面分开处理，末态通过官方 Fabric 保存接口回写 USD，并与独立原生刚体状态核对。Task3 旧三层展示图的 USD/native 位置差为 5.41 μm，在其记录的容差内；本轮实际 reset 使用修正后的精确同步路径。

这些历史图像结果覆盖几何与出图；当时运行中的 Fabric USD 回写会影响原生速度，不能把静置图或旧反力记录当作机构验收。后续 Task1 的无运行时回写复位、真实探针按压回弹与归架复核独立保存在 `development/living_task12_validation_20261018/task1/validation_summary.json`，也不代表机器人同手操作。Task2 历史含水／空壶反力差只达到理论水重约 7.51%，该记录与水重定量校准限制保留；当前场景使用、粒子计量裁判与定量耦合资格分开，不从发布候选推导完整机器人倒水成功。历史细节与失败记录见 液体有限范围证据（开发存储 `datasets/fr3_sharpa_bench/development/living_water_20261018/bounded_evidence01.json`）。

重建时先用 `scripts/maintenance/freeze_project_sources.py` 冻结完整源码并显式包含 `bench/apps/workroom.headless.kit`，再从快照使用 `scripts/media/render_living_task.py --task <task_id> --asset-output <新资产版本> --run-output <新实验目录> --output <新媒体目录>`。指定健康 GPU UUID；工具会拒绝覆盖旧目录。液体默认采用 1/960 s、2880 步，其余采用 1/240 s、480 步。

初次四任务版本（Task2 当时为眼镜盒）的 CPU 规则、任务目录和模块依赖检查记录在规则验证记录（开发存储 `datasets/fr3_sharpa_bench/development/living_tasks_20261018/cpu_rules01/validation.json`）。合成规则测试只验证裁判逻辑，不产生原生物理或机器人动作资格。
