# Task3：单臂自由搭高

用户于 2026-10-19 确认本任务只比较搭建高度：**不限定塔形、层数、每层木条数量或摆放方向，记录回合内曾经达到的最高合格稳定高度**。单个 Agent 从两侧托盘取木条，在原桌面自由搭建。计分高度以原桌面 `z=0.78m` 为基准，来自真实刚体几何和接触支持链；机器人举高木条、托盘里的初始叠料和短暂飞起都不计为塔高。

新场景使用 `assets.author_task(..., profile='trays_v1')`，保留 150 × 45 × 25 mm 木条尺寸，将库存增加到 36 根。左右供料托盘各有 3 × 3 列、每列两根，共 18 根；横向净隙 35 mm、端部净隙 50 mm。中央桌面初始为空，没有目标槽或限定塔形的标线。36 是可用库存数量，不是搭建合格所要求的用料数量。

当前物体资产为 `assets/task_parts/living_room/tower_building/v6`，任务发布为 `<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/tower_building/v5`，manifest SHA256 为 `2fe24d4ff529cdbdc60afd7f0fda024110dae559af054cb5cd1f0977500b287f`。目录绑定见 `definition.py`，冻结发布保持 `prepared_unvalidated`、`physics_accepted=false`、`task_policy_success_claimed=false`；旧版原生启动、静置或图片不授予新配置资格。原 `tower_building/v4` 及更早任务发布、资产和运行记录均保留。

新版桌面近景（开发存储 `demo/interface/scenes/living_trays_towers_20261019/tower_building/tabletop.png`）展示两侧真实托盘、36 根供料和空中央桌面；全景（开发存储 `demo/interface/scenes/living_trays_towers_20261019/tower_building/workcell.png`）及同目录 `provenance.json` 保留真实 RTX 相机和源码记录。两图均为 1920×1440、实际 128 spp，来自 `development/living_trays_towers_20261019/source01` 的 4000 个 0.5 ms 任务物体静置步。机器人只作静态展示，基座 `y=-0.76m`；实际任务发布沿用 `y=-0.88m` 的单臂布局，图片不是运行机器人精确布置或操控验收。

独立 `source02` 的原生入口检查报告（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/native_building01/report.json`）返回 `passed=true`：实际隔离 worker 加载 v5、公开四路 RGB-D，并执行小幅手腕动作和保持；1.38 s 时关节最大实际变化为 0.00569 rad，手腕目标误差约 0.137 mm，`physics_valid=true`、worker 正常退出。回合以 `cancelled` 关闭，裁判仍为 `building`；这证明入口、控制和证据接线，不证明取料、搭高或物理保真验收，报告的 `task_success_qualified` 与 `physical_fidelity_qualified` 均为 `false`。

托盘接触复核（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/tray_contact_review01.json`）检查了该回合全部 2760 个原生子步样本（另有未计分初态），每步 36 根供料木条均有带载支撑链落到声明的托盘底板；其中 `t>=0.05s` 的 2661 个样本确认 18 根直接落底板，另外 18 根由下层木条承托。全程没有自由搭建高度，最终 `current_height_m=best_height_m=0`、`score_valid=true`。这项支撑与零误计分证据限于该次 1.38 s 入口检查，不覆盖从托盘取料或搭建。

## 高度计分与恢复

候选搭建必须通过真实接触支持链落到原桌面，脱离机械手和其他外部物体支持，并持续满足释放、速度、漂移和接触几何要求至少 1 秒。规则不要求规则网格或水平姿态；侧立、交错、桥接等形状只要满足同一支持与稳定条件即可参与高度计分。测量采用整根木条的实际定向边界，不用木条中心高度代替最高点。

已确认的最高稳定高度持续保留；后来倒塌或木条落地不会抹去此前成绩。Agent 可以重抓、重搭并继续争取更高记录，每次新高度仍要完成独立稳定窗口。初始托盘叠料和仍留在托盘的余料不能抬高得分，托盘也不能成为计分搭建的支撑根。任何托盘、机器人、地板或外部接触都会使整个相连的搭建部分暂时不合格；与它分离的余料移动或落地不会打断已合格部分。

结果报告 `current_height_m/cm`、`best_height_m/cm`（也提供 `best_stable_height_m/cm`），`score` 以米为单位，`score_unit='m'`。预算结束为 `incomplete` 并保留成绩；缺失或无效证据为终止 `invalid`，既有最高值仍可见，但 `score_valid=False`。此模式没有二元成功阈值，不能把最高高度指标转换成虚构的完整任务成功或对抗判负。

## 托盘、物件与配置

两托盘底板厚 6 mm，底面落在原桌面，内部承托面为 `z=0.786m`；围边厚 6 mm、比内底高 12 mm。底板与四边均为真实静态盒碰撞体，固定是明确的任务夹具假设。每块木条仍为独立动态刚体，没有焊接、驱动或隐藏托架。静态托盘的构造质量来自实体体积与 650 kg/m³ 工程先验，仅用于构造记录，不伪装为动态质量。木条密度、质量、盒惯量、摩擦和源纹理许可继续记录在资产元数据中；这些参数没有实测标定声明。

```python
from scenes.workrooms.living_room.tower_trays import (
    BUILDING_GEOMETRY, building_supply_layout, building_supply_trays,
)
from scenes.workrooms.living_room.tasks.tower_building.task import task_configuration

configuration = task_configuration(
    BUILDING_GEOMETRY, scoring_mode='freeform_height', inventory_count=36,
    supply_trays=building_supply_trays(), supply_layout=building_supply_layout(),
)
```

`target_slots` 在此模式为空；共享几何中的旧层数参数不约束计分搭建。`supply_trays` 声明托盘实体接触身份，底板与围边分开记录；供料下层实际由底板支撑，上层由下层木条支撑。显式 `presentation=True` 只生成说明动作的示意摆放，不能成为 reset 或得分证据。省略 `profile` 的作者和省略 `scoring_mode` 的规则工厂保留历史行为，以便重现旧版本。

任务沿用原单臂 `robot_layout.py`、通用 `/episodes` 和公开函数。父客厅 v7、机器人装配、共享控制器与既有物理参数不随自由搭高改写。

## 当前自由搭高的 CPU 验证

本轮相关 CPU 检查共 256 项通过，执行退出 0；测试日志（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/cpu_tests01.log`）保留通过进度。新增自由搭高、评分接入和托盘接触覆盖分别在 `test_living_tower_freeform.py`、`test_living_height_integration.py` 和 `test_living_tray_contacts.py`；资产与抽取规则覆盖也在既有 `test_living_tower_assets.py`、`test_living_tower_extraction.py` 中扩展。以下命令只列本任务当前自由搭高相关的四个文件，不作为 256 项总数的原始完整选择命令：

```bash
.venv/bin/python -m pytest -q \
  tests/test_living_tower_freeform.py \
  tests/test_living_height_integration.py \
  tests/test_living_tray_contacts.py \
  tests/test_living_tower_assets.py
```

这些检查覆盖自由姿态的实际定向边界高度、完整 1 秒支持窗口、倒塌后保留最高记录、无效证据与预算语义，以及托盘底板／围边身份、物件结构、冻结评分器隔离和公开高度反馈。接触与子步集成测试使用合成输入，不启动原生物理；真实稳定性与机器人操作由绑定版本的独立原生证据说明。旧固定塔的 `tests/test_living_tower_building.py` 保留在下面的历史验证范围内。

## 历史 v4：固定六层塔规则与证据

以下保留原 v4／默认参考配置的规则说明。其十八根、六层、槽位和朝向条件只适用于旧版，不适用于上述 `freeform_height`。

历史 `tower_building/v4` 工程 release 位于 `development/living_bench_integration_20261018/qualification_review/releases/`。原 v5 资产的 150×45×25 mm 木条、18 根分离供料和评分配置被冻结；参数化规则工厂仍可独立用于 CPU 设计，不修改已发布几何。真实接触归一化由场景 `runtime.py` 提供，评分复用冻结的参考算法。其状态为 `prepared_unvalidated`、`physics_accepted=false`，没有机器人搭塔成功结论。

v4 修正冻结适配器的原始 worktop 接触名称，恢复桌面作为支撑链根的语义，并纳入角速度端点上界与首错保留修复；原始参考裁判、规则、初态、机器人及物理参数不变。旧 v2 的短原生记录仅证明接口接通，其桌面曾被误标为 external；离线复放修正此标记不构成搭塔成功。历史 v2/v3 和原记录均保留。

目标默认六层、每层三根同规格长方块，共十八根；相邻层方向相差 90°，同层水平间隙参数为 `in_layer_gap_m`，层间没有竖向空气隙。使用公共 `tower_geometry.validate_geometry` / `tower_slots` / `block_ids`。运行 release 使用 `robot_layout` 单台 FR3 + 右 SHARPA，基座 `(0.65,-0.88,0.78)m`，待命腕 `(0.65,-0.55,1.40)m`。历史场景展示使用 dual front 的 `y=-0.76m`，与本次运行布局分别记录。

### 历史配置与供料

`assets.author_task(stage, root_path='/World/LivingTask', *, presentation=False)`
只在指定新根下添加本任务物件，不添加房间、机器人、灯光或 PhysicsScene。
该历史场景候选为 150 × 45 × 25 mm、6 层 18 根，独立动态刚体路径为
`<root>/Blocks/block_00` 至 `block_17`。每根使用一个等尺寸盒碰撞体，视觉边缘
倒圆 0.4 mm；盒体密度 650 kg/m³、质量约 109.69 g，静／动摩擦 0.45／0.35，
均为未标定工程先验，质量与惯量由名义实体盒推导。复用共享 CC0
`kitchen_wood` 木纹、法线与粗糙度纹理，返回元数据保留来源和文件哈希。
数字 01–18 是随木条移动的平面印记，没有独立碰撞。

默认作者将 18 根平铺在两组 3 × 3 供料区，全部通过现有配置工厂的显式
`supply_layout` 验证，中心目标留空。边框只有零厚度、无碰撞的桌面标线，
不提供隐藏承托或定位夹具。`presentation=True` 另以 9 根摆出中央三层，
其余 9 根仍在供料区；返回 `presentation_only=True`、`is_task_reset=False`，
配置中的真正供料初态保持不变。这是动作意图展示，不是机器人搭建或成功记录。

返回 JSON 包含配置、每根真实初态／边界／质量／惯量、动态 prim 路径、标线、
共享依赖与物理先验。CPU 检查见 `tests/test_living_tower_assets.py`；候选的原生
承托、稳定和机器人操作仍须独立验收；release 的候选尺寸与规则已经冻结。

```python
from scenes.workrooms.living_room.tasks.tower_building.task import task_configuration

# length、width、height 必须来自显式尺寸决定；此处不提供默认值。
configuration = task_configuration(
    {'block_size_m': [length, width, height]}, seed=17,
    budget={'max_sim_time_s': episode_time_budget},
)
```

`task_configuration(geometry, *, seed=0, supply_layout=None, budget=None, stable_duration_s=1., tolerances=None, floor_drop_is_terminal=True)` 返回独立 JSON 配置及其 SHA256，包含规范几何、全部目标槽位、供料初态、规则与单臂选择，并绑定当前客厅 v7 父环境的 USD／描述文件路径与哈希。尺寸变化会生成不同配置身份，不会创建或改写 USD、共享资产、场景或物理参数。`native_execution_available=False`、`source_release=None`；`evaluator.reference_implementation` 单独标识参考实现，不能被解释为可运行的 native task。

自动供料按固定算法生成原桌面上轴向对齐、无重叠、与整座塔目标足迹分开的平铺块，再按 seed 的 SHA256 排序绑定块 ID。同一配置和 seed 可复现。所有块放不下时明确拒绝，须缩小显式尺寸或另行设计并验证供料区域；本工厂不凭空增添供料台。显式 `supply_layout` 也必须在原桌面上、每块一次、平放、无重叠且不侵入任何塔位；不能借显式输入摆好奖励初态。供料几何可容纳不代表机械臂可达或可抓取。

### 历史固定塔的成功与恢复

全部物理块必须与全部合法槽位形成一一对应；同规格块可以交换 ID，沿长轴反向放置的 180° 等价姿态也可接受。槽位同时检查 XY 偏差、Z 偏差、层方向与倾斜，不能只依据数量或塔高。位置、接触间隙、穿透和稳定漂移容差由尺寸推导并显式写入配置；调用方可收紧或在保留独立槽位／表面约束的范围内调整。默认 XY 容差为 `min(6mm, 0.12×块宽)`，Z 容差为 `min(3mm, 0.12×块高)`，偏航 5°、倾斜 3°。这些是待原生校准的工程规则，不是控制器参数。

底层每根都需要真实桌面接触承托；上层每根需要接触支持链穿过立即下一层的实体块，最终到达桌面。正向承托证据须覆盖整个观察区间，接触点须同时处于接触双方的实体范围及对应顶／底面附近。机器人任意接触，以及墙、家具、托盘等外部接触，都会清除整塔成功窗口；零力触碰也不能被当作已经释放。所有接触的过深穿透（包括非承托接触），以及依据长方块位姿得到的实体深重叠，均打断稳定窗口。传入接触不在几何表面附近、隔层承托或仅有零支撑力均不获成功信用。

所有块必须同时保持唯一槽位、支撑链、释放和静止至少 1 秒。默认区间最大平移／角速度分别不超过 `0.01m/s` / `0.05rad/s`，锚点漂移不超过 `min(1mm,0.04×块宽)` / `0.02rad`。每段使用两端最大锚点距离加上 `区间速度上界×区间时长/2` 的保守界检查整段漂移，不能只靠相同端点隐藏区间往返。首个合格样本只建立时间起点，之前未观察到的区间不计时。错放、重抓或未掉地的倒塔可以通过真实机器人动作纠正并重建，整塔稳定窗口重计；禁止对象位置写入、焊接、自动扶正或自动重建。默认任一块接触地板是终止失败；可显式配置为可恢复。显式预算耗尽，或时间／动作计数达到或超过配置上限，优先于同帧成功判定，返回终止 `incomplete`，不是成功或对抗判负。

### 历史固定塔的可信归一化观察契约

`TowerBuildingEvaluator(configuration, trusted_normalized_input=False).update(observation)` 是流式接口，`evaluate_tower(sequence, configuration, trusted_normalized_input=False)` 可重放完整序列。默认拒绝未经信任边界接入的观察；观察 JSON 自填 `trusted=True` 无效。集成方必须先校验实际 native 来源、完整接触覆盖和状态身份，才可在构造参数开启可信输入。此参数不是认证机制，也不能把手写 JSON 变成原生证据。

每条观察必须包括：

- `schema='living_tower_normalized_observation_v1'`、匹配的 `configuration_sha256`、固定非空 `episode_id`、从 0 开始的连续 `sample_index`，以及 `time_s` / `interval_start_s`。首条是时间 0 的未评分供料初态，随后必须严格按规则的 `sample_period_s` 连续提供；默认 20ms。
- `sampling_complete=True`、`contacts_complete=True`、`physics_valid=True`、`robot_only_actions=True`、`object_pose_writes=False`。这些是可信适配器归一化后的前提，参考模块不自行认证。`robot_action_count` 单调递增，`budget_exhausted` 必须显式为布尔值。
- `objects` 恰好包含全部块 ID。每块有世界坐标 `position_m`、单位 `quaternion_wxyz`、`linear_velocity_m_s`、`angular_velocity_rad_s`，及覆盖上一完整区间的 `interval_max_linear_speed_m_s` / `interval_max_angular_speed_rad_s`。区间上界不得小于当前速度或由相邻姿态差推导的平均速度；不能用零速度字段掩盖位置跳变。
- `contacts` 是该区间涉及任务块的完整接触汇总，包含瞬时触碰和持续承托。每行有 `a`、`b`、`point_m`、`separation_m`、`support_for`、`minimum_upward_force_n`、`continuous`。参与者为块 ID、`table`、`floor`、`robot/<link>` 或 `external/<object>`；两端必须不同，至少一端是任务块。`support_for` 为受向上承托的块 ID 或 `None`；正支撑力需由可信适配器从实际有方向的接触量得到，单位 N，不能由几何重叠猜测。`continuous=True` 表示该支撑边及记录的力／间隙界限覆盖整个区间；零时刻 baseline 不给任何稳定信用。

缺字段、NaN、缺块、伪造配置身份、时序断点、不完整证据或与速度界限矛盾的运动会锁定 `invalid`。非法摆位和支撑不足则为可恢复的 `incomplete`。返回的 `success` / `reference_success` 只表达这些可信归一化输入下的规则结论，始终附带 `native_qualified=False`；CPU 测试不验证接触传感器、真实连续时间稳定性、机器人抓取策略或物理搭塔可行性。当前场景适配器对每个 0.5ms 原生步记录六缓冲接触和真实物体状态，并按完整 20ms 区间调用该冻结裁判；启动、采样与操控资格仍由独立证据评定。

运行合成规则测试：`python -m pytest -q tests/test_living_tower_building.py`。测试覆盖合法完整塔、完整 1 秒窗口、支撑链缺失、浮空／隔层／外部承托、重复槽位、错误朝向、速度伪报、断帧、供料初态、尺寸拒绝、恢复与预算语义。

任务通过通用 `/episodes` 和公开函数执行；客厅总报告（开发存储 `datasets/fr3_sharpa_bench/development/living_bench_integration_20261018/report.json`）统一记录本次接入检查。CPU 规则通过、原生启动或空动作等待均不等于完成搭塔。
