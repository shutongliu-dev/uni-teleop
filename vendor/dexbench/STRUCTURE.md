# 代码结构与职责

主工程只有 `bench`、`scenes`、`agent` 三个源码根：bench 提供各任务共用的执行、物理与观测能力，scenes 定义环境与任务，agent 根据公开信息决定动作并管理经验。后端、Kit 应用、数据准备和外部采集适配均收进 `bench/`，不再各占一个顶层目录。

```text
bench/             # 通用基础设施，含 simulation、apps、adapters、data_tools
scenes/            # 固定环境、可变任务、规则与评分
agent/             # 模型决策、工具编排与学习
configs/           # 运行配置与示例；凭据内容不入库
scripts/           # 构建、验证、媒体与维护入口
tests/             # 自动化检查
docs/              # 使用、设计与历史证据说明
data/              # 数据包（房间、机器人、物体、任务发布）；Git 只跟踪 README 与 MANIFEST.json
runtime/           # 本机仿真器与 Python 环境链接，不进 Git
outputs/           # 实验输出与媒体，不进 Git
```

## 模块边界

| 位置 | 职责 |
|---|---|
| `bench/protocol/` | 与 HTTP 无关的动作、观测、策略、预算和记录契约 |
| `bench/runtime/` | HTTP、统一函数分发、回合与动作生命周期、机器人说明及只读 IK、公开观测投影、原生 worker 通信；唯一服务入口 `bench.runtime.server:app` |
| `bench/sdk/` | 公开深度校验与解码、像素反投影、world 位姿增量；不含模型决策、预算或隐藏状态 |
| `bench/control/` | 静态机器人配置工厂、运动学、重定向、经典腕部／指尖 IK、关节反馈和双臂执行；不提供自动任务规划、通用抓取或自动避碰 |
| `bench/sensors/` | 模拟触觉、接触窗口完整性与动作级接触历史 |
| `bench/evaluation/` | 通用物理有效性审计，`cloth/` 集中布料几何与碰撞审计；任务成功条件归任务定义 |
| `bench/isaacsim/` | 原生离线 rollout、复杂任务发布、回放和独立验收 |
| `scenes/workrooms/` | Astra 六个固定环境、每个环境的多个任务及版本选择 |
| `scenes/manipulation/`、`scenes/cloth/` | 刚体／软体操控和布料任务的发布、规则、评分与特权工程策略 |
| `scenes/board_games/` | 可复用棋类规则、玩家协议与物理落子判定；`tabletop/` 是当前三级任务，`hex/` 保留历史契约 |
| `agent/model_client/` | 模型传输、公开观测输入、动作／工具／程序解释、实验预算与记录 |
| `agent/model_client/learning/` | 六场景共用的学习运行时、技能版本、调度和计量源码；生成的技能、经验与日志放共享 datasets，实际能力与限制见[学习模块](agent/model_client/learning/README.md) |
| `bench/simulation/isaacsim_backend/` | Isaac/PhysX 生命周期、关节驱动、相机标定、接触采样和材料绑定 |
| `bench/simulation/assets/` | 资产读取与几何配方；房间 USD 配方在 `workrooms/` |
| `bench/apps/` | 原生 Isaac 的 Kit 启动配置 |
| `bench/adapters/manus/` | MANUS 数据采集与动作适配；硬件未连接 |
| `bench/data_tools/` | OakInk2／MANO 数据准备、参考动作读取、离线动作准备和任务部件生成 |
| `data/`、`runtime/`、`outputs/` | 数据包、本机环境链接和运行输出，见各自 README；路径统一由 `bench/paths.py` 提供 |

`bench/geometry.py`、`convex_geometry.py`、`evaluator.py`、`sequence_evaluator.py` 保留在 bench 根，维持冻结评分器所需的相对依赖结构。任务专属规则仍放在 scenes，不在这些基础模块里新增任务策略。

任务可以选择机器人、初态、相机和控制器；`bench.control.factory` 从场景的静态机器人配置创建几何模型与独立执行器，默认控制器按任务／回合实例化。`/policies?task_id=…` 和 `/interfaces?task_id=…` 返回该任务的配置；多种 descriptor 下不允许省略任务。机器人说明和只读 IK 使用选定实例的公开状态，通用深度与投影计算由 SDK 提供。

HTTP worker 已发布场景声明的全部视角，腕相机采用同帧 RTX 标定；CPU 检查及历史原始帧复核不代替新原生验收。新增查询已通过 Agent 公开函数工具接线，CPU 协议测试不代表模型操控资格；双臂动作级接触历史已提供严格采样检查。棋牌 release 由 `scenes/board_games/runtime_release.py` 校验，`bench/runtime/board_games.py` 提供认证玩家会话，`board_game_control.py` 固定非当前玩家的驱动目标，`board_game_worker.py` 接入原生子步裁判；公开 schema 位于 `bench/protocol/board_games.py`。统一学习阶段调度仍待实现；Agent 函数运行入口与对弈编排状态见[客户端说明](agent/model_client/README.md)，不能用 API 接通代替模型整局验收。详见[通用能力接入](docs/agent_bench/scene_integration.md)与[API 范围](bench/runtime/README.md)。

## 固定环境与可变任务

```text
scenes/workrooms/
├── catalog.py
├── <family>/scene.py
└── <family>/tasks/<task_id>/
    ├── definition.py
    └── task.py             # 仅已实现任务具有执行入口
```

六个 family 是 `mechanical_assembly`、`laboratory`、`kitchen`、`study`、`card_room`、`living_room`。`scene.py` 固定父环境路径、版本和哈希；建筑、家具、空桌、基准灯光及世界相机属于环境。任务定义工作台上的物体与夹具、机器人及其摆位、reset 状态、任务相机覆盖、公开规则和成功／失败判定。更换 task 不复制房间、不改写父环境资产。

Agent 侧的机械装配、实验室、厨房、书房和客厅统一采用任务已知 RSI，复用 `agent/model_client/learning/`，不在各场景下复制学习器。棋牌室 task0 使用普通 `/episodes` 单 agent 回合并复用该流程；task1/task2/task3 分别为井字棋 `tic_tac_toe_3x3`、平面连四 `connect_four_5x5`、Hex `hex7x7`，均使用 `/board-games`。task0/task1 允许练习和 RSI，task2/task3 固定技能版本并禁止 RSI；任务编号不是学习阶段。持久 Python、运行隔离、阶段调度与累计指标仍待实现，函数运行入口不代表完整学习闭环。设计与实现状态见[学习模块](agent/model_client/learning/README.md)。场景规则继续由 scenes 维护，自编任务技能属于实验数据，不写进场景源码。

`agent.model_client.scene_tasks` 只读列出场景任务与 Agent 适配状态；`function_runner --family … --task …` 在场景内解析编号并绑定对应冻结发布。机械 L1–L4 使用普通公开函数回合。客厅 Task1–3 是单 agent，Task4 `tower_extraction` 是两名独立 agent 各控一套右手机器人的轮流抽塔对抗，保留客厅 RSI 政策。客厅 Task1–3 使用普通 `/episodes`；Task4 已接认证 `/turn-games`，`player_0`／`player_1` 分别绑定 front/rear，按裁判当前玩家调度，一轮可有多次操作，确认稳定交接后换手。不能套用棋牌左右手绑定或把它作为普通单人任务。客户端接通不改变原生 reset 阻断及操控资格。详见[Agent 入口](agent/model_client/README.md)。

厨房四项任务复用 `scenes/workrooms/kitchen/{assets,geometry,composition,release,rules,evaluation,native_evidence}.py`，独立任务目录绑定各自冻结发布。资产在共享 `assets/task_parts/kitchen/`，规则/裁判与原生承托证据分开绑定；初始化与承托通过不代表机器人完整端送或抓放成功。厨房共用原 FR3＋右 SHARPA，任务仅选择朝吧台或灶台的基座方向。

一个场景内共用的组装和资产配方放在该 family 根目录，例如棋牌室的 `game_task.py`、`game_assets.py`、`reserve_assets.py` 和 `robot_layout.py`；每个任务目录仍固定自己的 release。跨场景任务规则复用 `scenes/manipulation`、`scenes/cloth` 或 `scenes/board_games`，不进入 Agent 决策模块。

机械加工室的 `scene.py` 固定空桌环境 `mechanical_workshop_20261017_v3`；`robot_layout.py` 定义五项任务共用的 `mechanical_single_arm_v1`，基座为 `(-0.23,0,0.78)m`、支架底脚距桌边 20 cm。`task_design.py` 复用设计约束，`threaded_assets.py` 保留原 v1 校验，`threaded_sharpa_catalog.py` 核验用户选择的 SHARPA v3；`preview.py` 与 `threaded_preview.py` 负责静态展示。任务保持 `tasks/<task_id>/definition.py` 结构，L1–L4 绑定各自冻结候选，由 canonical worker 和公共 functions 执行；发布状态仍为 `prepared_unvalidated`，L5 已绑定当前 inset-layout 冻结候选，受控工具证据与机器人装配资格分开。L1–L3 保留现有发布，L3 为 `shaft_spacer_candidate_v2`；L4 当前默认是散装螺栓装配 `mechanical_scene_delivery_20261020/l4/releases/loose_bolt_candidate_v10_pgs`，需拾取螺栓、起牙、拾取并插入工具，稳定接合后才计旋入与释放。v10 保留 v9 的 `loose_bolt_v5` 供料和 reset，仅把显式场景求解器由 TGS 改为 PGS；0.5 ms 步长、原螺纹资产、机器人和控制器不改，全局 TGS 默认不变。预接合接触、完整运动和初态 HTTP smoke 通过范围受限检查；v9 TGS 稳定／运动失败、v8 原生初态失败和旧预插入版本的证据保留。`referee.py` 定义家族静态合同和裁判状态，`referee_release.py` 在保留原 reset_variant 和物理输入的前提下生成新规则版本；通用 `bench/protocol/task_referee.py` 白名单投影通过既有任务接口发布。历史 authoring 默认仍为 preinserted，运行默认由 catalog 显式发布决定。L4 的 `motion_plan.py` 与旧验证脚本保留各自历史诊断范围，当前 loose 原生输入由 `scripts/validation/validate_mechanical_fastener.py` 生成。详见[机械规则](scenes/workrooms/mechanical_assembly/RULES.md)、[任务设计](scenes/workrooms/mechanical_assembly/README.md)和[环境与证据](docs/scenes/mechanical_workshop.md)。

机械室九项（L1–L5 与 NIST 任务板适配 Task6–9）、厨房六项、实验室五项，棋牌室、书房和客厅各四项，共 32 项 implemented 候选。实验室五项的任务入口在 `scenes/workrooms/laboratory/tasks/`，家族共享组合、规则、裁判与发布逻辑在其根目录，通过普通 `/episodes` 接入 bench；Agent 使用 `--family laboratory --task task1` 至 `task5`。新增 planned 项仍不列入默认可运行清单，加载也拒绝。目录、原生初始化、实际操控和模型成功是不同状态。旧 `scenes/rooms/`、`scenes/catalog.json` 与六房间诊断 registry 保留各自原始范围，不冒充新任务。完整目录见[场景／任务说明](scenes/workrooms/README.md)。

## 执行与信息边界

```text
agent/model_client 读取公开任务与观测，产生动作
  → bench/runtime 校验协议、预算和动作生命周期
  → bench/control 将几何目标转换为机器人关节目标
  → bench/simulation/isaacsim_backend 执行原生物理
  → bench 发布新的公开观测，任务评分器与物理审计独立记录结果
```

模型调用期间暂停物理；控制周期为 20 ms，接触按原生子步采样。动作只能写机器人关节目标，物体只能在 reset 初始化。图像与深度在决策边界发布并携带时间和标定，旧快照不能标为当前状态。公开观测不含隐藏物体真值和评分器内部状态；特权工程策略的资格与模型成绩分别记录。

原生任务成功只覆盖报告绑定的源码、release、初态和物理参数。迁移源码路径不重新证明物理资格，也不改写历史评分。当前已知修复及原生范围见[失真修复](docs/agent_bench/fidelity_repair.md)、[工程基线](docs/agent_bench/engineering_baselines.md)和[房间诊断](docs/agent_bench/rooms.md)。

## 本次源码迁移

| 旧位置 | 当前位置 |
|---|---|
| `interface/model_client/` | `agent/model_client/` |
| `interface/agent/` | `bench/runtime/` |
| `interface.server` | `bench.runtime.server`，旧入口不保留 |
| `bench/agent/` | `bench/protocol/` |
| `bench/scenarios/` | `scenes/` |
| `bench/simple_tasks/`、`bench/cloth_tasks/`、`bench/board_games/` | `scenes/manipulation/`、`scenes/cloth/`、`scenes/board_games/` |
| `interface/agent/tactile.py`、`interface/agent/physics_audit.py` | `bench/sensors/tactile.py`、`bench/evaluation/physics.py` |
| `interface/agent/sensors.py` 中的 USD 相机标定 | `bench/simulation/isaacsim_backend/camera_calibration.py` |
| 历史 Hex 本地模型桥接 | `scripts/validation/hex_model_smoke.py` |
| `simulation/` | `bench/simulation/` |
| `apps/` | `bench/apps/` |
| `interface/manus/` | `bench/adapters/manus/`；根 `interface` 包已删除 |
| `interface/README.md` | `bench/adapters/README.md` |
| `data_tools/` | `bench/data_tools/` |
| `bench/agent_plan.py`、`reference.py`、`prepare_motion.py`、`generate_task_parts.py` | `bench/data_tools/` 下的同名文件 |
| `bench/kinematics.py`、`retarget.py`、`keypoint_retarget.py` | `bench/control/` 下的同名文件 |
| `bench/audit_cloth_surface.py`、`audit_cloth_self_intersection.py`、`cloth_robot_collision.py`、`cloth_self_intersection_precomputed.py` | `bench/evaluation/cloth/` 下的同名文件 |

迁移表供维护者更新导入和理解旧源码记录；旧位置按原名保留在左列。活动代码使用新路径，不提供旧 Python 路径别名。共享资产、模拟器 runtime、冻结 datasets、旧 manifest、原实验路径与 SHA 保持不变；旧运行继续使用自己的冻结源码。独立 `system0-system1` 的源码、配置和冻结依赖不随主工程迁移。此前场景从 `integrations/scence` 移入正式模块的历史也保留，不创建兼容目录或归档代码 fallback。

## 工具、存储与命名

`scripts/` 按 `validation`、`diagnostics`、`media`、`assets`、`maintenance` 分类；根目录只保留单次入口 `run_push.py` 与原生解释器 `isaac_python.sh`。模型实验配置在 `configs/agent_bench/`，任务版本指针跟随 scenes，凭据仅引用快照外的密钥文件。

| 内容 | 存储位置 |
|---|---|
| 代码、配置示例、测试和文档 | 项目仓库 |
| 机器人、网格、纹理、房间与任务发布 | `data/`（`DEXBENCH_DATA`），保留来源、版本与许可 |
| 源码快照、实验数据 | `outputs/`（`DEXBENCH_OUTPUT`） |
| Agent 自编技能、候选版本和学习记录 | `outputs/skill_learning/<experiment_id>/`；默认按任务、learner 和独立实验隔离，身份及阶段写入冻结合同，详见[存储设计](agent/model_client/learning/storage.md) |
| 展示媒体及其 sidecar | `outputs/media/` |

活动模块按职责使用稳定名称；日期、候选版本和一次性脚本进入实验目录。schema、协议及冻结发布保留显式版本。运行使用同一完整封存源码根，封存后验证 launcher 的执行权限与实际导入路径。修改已有实验依赖前先隔离运行，不能用目录别名规避源码清单。

静态依赖检查可在 CPU 上运行，不导入仿真器：

```bash
.venv/bin/python -m scripts.maintenance.check_module_dependencies
```

检查递归覆盖完整 `bench/` 和 `scenes/`，拒绝依赖 `agent` 或已经移除的顶层 `interface`、`simulation`、`data_tools`。它检查普通导入与字面量动态导入；计算得到的模块名不在此静态检查范围，不能把它当运行时沙箱。新快照使用 v6 的三个源码根，历史 v5 五根与 v1–v4 三根继续按原布局校验，详见[快照契约](docs/agent_bench/source_snapshots.md)。
