# 在线模型客户端

## 通用公开函数与场景入口

`python -m agent.model_client.function_runner` 是完整公开函数入口，运行协议为 `public_function_agent.v3`。它按服务 `/tasks` 的合同选择普通 `/episodes`、棋牌 `/board-games` 或轮流操作 `/turn-games` 会话；显式指定 `--family` 时，先由可信宿主从场景目录解析任务编号、实现状态与冻结发布，再校验服务。带公开裁判合同的普通任务每轮 `task` 只含静态目标、规则与评分合同；动态阶段、终局和成绩通过 `get_task` 查询，保留其观测序号，不反复呈现创建时的初始状态。原 `runner` 保留已有固定动作／工具／AST 比较条件；双玩家任务使用 `function_runner`。

`--family` 接受六个场景的 family 名：`mechanical_assembly`、`laboratory`、`kitchen`、`study`、`card_room`、`living_room`。各场景均可使用 canonical task ID 或目录显式编号的 `taskN`，机械装配还支持 `L1`–`L9`。实验室已发布 `task1`–`task5` 五项、厨房已发布 `task1`–`task6` 六项单 agent 候选。未指定 family 的 `task1`／`task2`／`task3` 保留棋牌室旧别名；选择其他场景的同号任务必须显式写 family。`RunConfig.family` 和任务接入身份进入宿主记录，不能只凭相同编号混用任务。

只读查看接入清单不创建回合、不联网或调用模型：

```bash
python -m agent.model_client.scene_tasks --family mechanical_assembly
python -m agent.model_client.scene_tasks --family living_room
python -m agent.model_client.scene_tasks --family kitchen
python -m agent.model_client.scene_tasks --family laboratory
```

清单显示编号、canonical ID、状态、会话类型、冻结发布及 Agent 适配是否就绪和原因。`adapter_available` 只表示 Agent 有对应会话适配器；`runnable` 还要求任务非 planned，两者都不证明原生操控验收。客厅 Task1–3 使用普通 `/episodes` 适配器；Task4 使用认证 `/turn-games` 适配器，四项都绑定各自的冻结发布。`planned` 在创建会话或请求模型前拒绝，留下结构化 `task_setup_error`；CLI 遇到该类任务设置错误返回退出码 2。

显式指定 family 时，`release_binding=catalog_pinned`，三种会话均核对创建响应 spec 的 `task_release_sha256`；服务须使用该任务 definition 选择的冻结 release。省略 family 的已知 canonical ID 仍推断场景、学习权限和可运行状态，但保留 `release_binding=service_selected`，兼容服务选择的历史发布；此时 metadata 的 `source_release` 仅是当前 catalog 引用，实际发布以创建 spec 为准。省略 family 不能绕过 planned 或缺失适配器的拒绝。

### 机械装配与客厅

| 场景／编号 | canonical task ID | Agent 会话语义 | 当前交付 |
|---|---|---|---|
| 机械 L1／task1 | `parts_sorting` | 单 agent `/episodes` | implemented 候选 |
| 机械 L2／task2 | `peg_insertion` | 单 agent `/episodes` | implemented 候选 |
| 机械 L3／task3 | `shaft_spacer_assembly` | 单 agent `/episodes` | implemented 候选 |
| 机械 L4／task4 | `fastener_assembly` | 单 agent `/episodes` | implemented 候选 |
| 机械 L5／task5 | `gear_train_assembly` | 单 agent `/episodes` | implemented 候选；独立工具传动与退出证据不等于机器人装配 |
| 机械 L6／task6 | `nist_connector_insertion` | 单 agent `/episodes` | v5 implemented 候选；USB／BNC 插接 |
| 机械 L7／task7 | `nist_belt_drive_assembly` | 单 agent `/episodes` | v2 implemented 候选；带轮与闭环皮带 |
| 机械 L8／task8 | `nist_cable_routing` | 单 agent `/episodes` | v3 implemented 候选；单线按序导向 |
| 机械 L9／task9 | `nist_wire_harness` | 单 agent `/episodes` | v3 implemented 候选；彩线配对、成束及拆除 |
| 客厅 Task1 | `remote_operation` | 单 agent `/episodes`、原始右手单臂 | v7 implemented 候选：开电视保持 10 s、关机后归架；普通适配可用 |
| 客厅 Task2 | `water_pouring` | 单 agent `/episodes`、原始右手单臂 | v5 implemented 候选；采用实用水耦合 v2 资产 |
| 客厅 Task3 | `tower_building` | 单 agent `/episodes`、原始右手单臂 | v5 implemented 候选：36 根木条自由搭高，按最高稳定高度计分；普通适配可用 |
| 客厅 Task4 | `tower_extraction` | `two_player_adversarial`，认证 `/turn-games` | v8 implemented 候选；front/rear 轮次适配可用，front 先手 canonical reset 与微动通过 |

机械九项共用单 FR3 + 右 SHARPA，均由目录选择对应发布并使用普通回合与公开函数链路，不内置装配解法。当前 release 仍为 `prepared_unvalidated`，有界接触证据和 CPU 接线测试都不等于机器人成功。本轮 Agent 接入检查不启动模型或原生实验；NIST 物物验收与手物数采分开。各任务的 reset、规则、发布和证据范围见[机械任务](../../scenes/workrooms/mechanical_assembly/README.md)、[NIST 任务](../../scenes/workrooms/mechanical_assembly/NIST_TASKS.md)及[特权验证索引](../../docs/scenes/mechanical_privileged_validation.md)。

客厅 Task4 是两名独立 agent 交替抽一块、放入己方收集区，**不重新叠到塔顶，造成倒塌的一方输**。`player_0`／`player_1` 分别绑定机器人 `front`／`rear`，各自都是原始右手 FR3 + SHARPA。客户端使用各自的 Bearer token、关节和 frame 权限，并保留整局独立上下文。调度以公开 `turn_game.current_player_id` 和 `turn_index` 为准：一轮可以包含多次查询、意图和物理动作；只有原生裁判确认收集、释放及稳定交接后才换手。不能按每次函数调用交替，也不能把 release 意图当作交接。共享未决动作收束前不放行另一方；`cancelled/turn_changed` 表示交接，不表示原运动目标已完成。

Task4 使用公开 `TurnIntent`，例如 `{"intent":{"type":"turn_intent","name":"select","arguments":{"block_id":"block_03"}}}`；具体命令和合法目标取当前公开规则与进展，意图本身不移动积木。终局从 `TurnGameResult` 读取：比较 `winner_player_id`／`loser_player_id` 记录胜负，裁判确认的完成且无胜者才记录平局。预算未完成、无效 reset、缺失证据和 API 中断不能据此判某方输或平局；执行健康与已确认终局分别保留。完整规则见[Task4 说明](../../scenes/workrooms/living_room/tasks/tower_extraction/README.md)。

Task4 的 `public_progress.status=reset_pending` 尚不是选块阶段。Agent 此时只向当前玩家提供显式 `hold(mode="acknowledged_drive")` 和只读、动作恢复、结束接口；普通运动、默认 `reference` hold 和选块意图在宿主侧拒绝。模型必须明确提交保持动作推进物理，轮询本身不推进时间；公开裁判进入 `await_intent` 后恢复正常工具。这个过程不额外偷偷执行动作，保持动作仍计入共享预算。

Task3 v5 的公开终局现在通过 `EpisodeResult.scoring` 提供连续成绩、单位、有效性和白名单指标；在线仍只提供粗粒度裁判阶段。正常预算结束或主动停止保留最高稳定高度，Agent 报告 `task_scoring_mode=continuous`、`task_score`、`task_score_unit`、`task_score_valid`，二元 `task_success=null`／`task_outcome=unknown`，原因是 `continuous_scoring`，不制造成功或失败。故障、未决动作和无效证据不能产生可评估成绩，原终局中的数值仍可供审计。创建时的静态 `scoring_contract` 与后续回包绑定；旧二元任务保持原分类。

开发 suite 复核归档时同样核对冻结的公开评分合同，并按任务、单位、优化方向分别汇总有效连续成绩；零分也是有效成绩。连续任务不进入二值成功率、惩罚成本或成功／秒的分母。技能库 `record_validation` 接受宿主记录的有效连续成绩；故障或无效成绩只能保留审计记录，不能作为有效验证选版。这不补齐尚未实现的自动练习调度。

有限回合的资源上限由 Agent 与 Bench 共用 `bench.protocol.allocations`：150 s 仿真、2400 s 墙钟、7000 次动作。机械 L3/L4 当前冻结推荐预算均可提交；这些是接入封顶，不会自动替换 CLI 默认预算，也不改变场景规则或冻结发布。显式关闭时间预算的 `none` 语义保持不变。

Task1–2 使用 front 单臂布局，Task3 保留其 y=-0.88 m 的单臂布局，不能复用其他任务的机器人基座。四项发布保持 `prepared_unvalidated`／`physics_accepted=false`，后续证据单独绑定。当前 Task2 v5 绑定实用 8 mm 粒子 v2 资产：独立支撑检查水重误差为 +4.42%，倒水夹具中杯内水反力仍偏高 +22.68%；历史低水负载传递结果不代表当前版本。Task4 v5／v6 曾在完整 10 s reset 窗口后以 `invalid_reset_not_stable` 中止，v7 在 smoke 预算内未就绪，这些记录保留。当前 v8 显式选择 TGS 每轮外力和 0.25 ms 步长（公开控制周期仍为 20 ms），seed 0 的 `player_0/front` canonical smoke 在 1.2 s 公开看到 `await_intent`，并完成 2 mm 手腕微动；rear 先手未单独验证，两者均未进入抽取。客户端接通不提供机器人开关电视、倒水、搭高、抽塔或整局资格；见[客厅任务](../../scenes/workrooms/living_room/README.md)和[场景接入证据](../../docs/agent_bench/scene_integration.md)。

### 书房与厨房

| 场景／编号 | canonical task ID | Agent 会话语义 | 当前交付 |
|---|---|---|---|
| 书房 Task1 | `stationery_sorting` | 单 agent `/episodes` | v4 implemented 候选 |
| 书房 Task2 | `book_stacking` | 单 agent `/episodes` | v3 implemented 候选 |
| 书房 Task3 | `book_retrieval` | 单 agent `/episodes` | v3 implemented 候选 |
| 书房 Task4 | `book_shelving` | 单 agent `/episodes` | v3 implemented 候选 |
| 厨房 Task1 | `breakfast_serving` | 单 agent `/episodes` | v3 implemented 候选 |
| 厨房 Task2 | `produce_transfer` | 单 agent `/episodes` | v3 implemented 候选 |
| 厨房 Task3 | `utensil_sorting` | 单 agent `/episodes` | v3 implemented 候选 |
| 厨房 Task4 | `pot_lid_placement` | 单 agent `/episodes` | v4 implemented 候选 |
| 厨房 Task5 | `pantry_stocking` | 单 agent `/episodes` | v2 implemented 候选；八件瓶罐与干货分类归位 |
| 厨房 Task6 | `table_clearing` | 单 agent `/episodes` | v2 implemented 候选；两盘、两碗、两杯、勺和叉分类收集 |

两个场景都复用原单臂 FR3 + 右 SHARPA 和普通公开函数回合，`--family study|kitchen --task taskN` 按目录编号解析，并以 `catalog_pinned` 方式核对冻结发布。书房基座与机械室相同；厨房基座居中，Task1–3、Task5–6 朝吧台，Task4 朝灶台，由任务 reset 选定。Agent 已有覆盖两类全部十项任务的 CPU runner 测试，包括编号选择、冻结发布、公开规则及 observe/hold/stop HTTP 链路，使用替身 worker/model，不是模型操控验收。场景原生证据只覆盖初始化、承托和有界检查；尚无完整 Agent 操控运行。学习政策与机械、客厅相同，为任务已知 RSI。规则与发布见[书房任务](../../scenes/workrooms/study/README.md)和[厨房任务](../../scenes/workrooms/kitchen/README.md)。

厨房 Task5/6 启动时显式添加 `--sim-budget none`，例如在通用命令模板中选用 `--family kitchen --task task5 --sim-budget none`；Task6 将 `task5` 改为 `task6`。两项冻结裁判允许 240 s，客户端默认 120 s 不会自动随任务调整，而共享有限请求上限为 150 s，因此不要传 `--sim-budget 240`。`none` 仅关闭客户端／回合仿真时间上限，冻结裁判的 240 s／6000 次动作上限和默认 1800 s 墙钟仍生效。Agent 自身默认另限 256 次动作、128 轮决策，可通过 `--max-actions`／`--max-decisions` 显式调整；实际取较先耗尽的预算。前四项继续使用默认 120 s。

### 实验室

| 编号 | canonical task ID | 当前冻结发布 | 公开任务 |
|---|---|---|---|
| Task1 | `sample_sorting` | v2 | 四个密封样本瓶按颜色分类，直立放稳 |
| Task2 | `vial_racking` | v2 | 三支试管扶正，放入任意三个不同架孔 |
| Task3 | `cap_placement` | v3 | 瓶身入防转座，旋盖、放盖、复盖并归位 |
| Task4 | `solid_weighing` | v3 | 药匙运送颗粒，称取 6±0.6 g，累计洒落不超过 1 g |
| Task5 | `pipette_transfer` | v1 | 活塞吸液后双杯分装，每杯 8–12 mL，累计洒漏不超过 2 mL |

五项均为 `implemented` 候选，使用 `--family laboratory --task task1`–`task5` 或表中 canonical ID，创建普通单 agent `/episodes`。两张台面由同一台原始 FR3 + 右 SHARPA 操作，不创建第二位玩家。服务预先加载对应 catalog release；显式 family 绑定其 manifest hash，发布不符、会话路由不符或重复任务条目均拒绝，创建和终局继续核对同一 episode/release 身份。

`get_task` 返回已绑定的公开规则及粗粒度 `working`／`settling` 裁判阶段；完整结果取 `get_result`／`finish_episode`。模型不读取本地 reset、evaluator、物体真值、原生接触、颗粒身份或液体账本，也没有直接写质量／液量、驱动工具的私有操作。所有任务采用任务已知 RSI 权限，task2/task3 不继承棋牌的冻结比赛限制；当前 JSON 函数基线仍是空技能库，完整 RSI 调度未实现。

检查入口（CPU worker／模型替身，零外部模型调用、零原生仿真）：

```bash
python -m agent.model_client.scene_tasks --family laboratory
python -m pytest tests/test_scene_function_runner.py -q -k laboratory
```

检查覆盖全部五项普通 HTTP 创建、规则读取、observe/hold/stop、单臂身份、错误发布／错误会话拒绝及私有信息隔离。它验证接线，不证明机器人操控成功。已有原生证据仅覆盖场景初始化与物物可行性；Task2 自由投落后仍摇晃，Task5 的有限体积近似未耦合水重或射流反作用，手物交互由数采负责。冻结 manifest 的 `prepared_unvalidated`／`physics_accepted=false` 保持不变。具体规则、证据和场景图见[实验室任务](../../scenes/workrooms/laboratory/README.md)，学习边界见[实验室接入约束](learning/constraints.md#laboratory-adapter)。

机械与客厅均保留任务已知 RSI 政策。客厅 Task4 的两位 learner 分别管理上下文、技能版本和成本；对抗会话与学习权限是独立维度，不因玩法类似棋牌而套用 task2/task3 的禁 RSI 政策。当前函数入口仍为空技能库 JSON 基线，持久 Python、隔离和练习调度尚未实现。

可信宿主读取本地 catalog 仅用于选择与校验；模型的任务目标与规则来自服务公开合同，不读取本地 evaluator、私有 initial_state 或任务发布中的内部评分。公开任务规则缺项不能用本地私有文件补齐。接入约束见[learning](learning/constraints.md#living-room-adapter)。

HTTP 客户端固定普通及双人终局的完整 `EpisodeSpec`，棋牌公开规则／几何也与创建回执绑定。`query_ik` 同时核对请求、结果和上下文的观测序号及仿真时间；旧快照解不能作为当前查询结果返回。当前修复范围与 Agent／Bench 后续分工见[接入修复与待办](../../docs/agent_bench/scene_integration.md#agent-integration-repairs)。

控制器身份按完整 `PolicyDescriptor` 绑定：发现阶段选择的描述必须与创建回执的 `spec.system0`、`capabilities.system0` 一致，包括实现／权重哈希、控制周期和能力语义。同 ID/version 不能替代完整比较；不一致在会话初始化、模型调用和动作之前拒绝，已有回合仍会清理。报告记录 `ready_validated=false` 和 `result_valid_for_scoring=false`，清理结果只供审计，任务或比赛的可评估结果保持未知。

### 棋牌室

| 棋牌任务 | 实际 task ID | 会话 | 学习权限 |
|---|---|---|---|
| task0 | `task0` | 单 agent `/episodes` | 允许练习／RSI |
| task1 | `tic_tac_toe_3x3` | 双玩家 `/board-games` | 允许练习／RSI |
| task2 | `connect_four_5x5` | 双玩家 `/board-games` | 冻结技能，禁止 RSI |
| task3 | `hex7x7` | 双玩家 `/board-games` | 冻结技能，禁止 RSI |

棋牌任务接受 `--family card_room` 加编号或实际 task ID，也保留不指定 family 的 `task1`／`task2`／`task3` 别名。task0 从创建响应的 `task` 读取本回合随机目标；任务列表没有这些目标，不能使用发布模板代替。省略 `--seed` 时 task0 由 bench 选择新 seed，明确传入则复现三个目标格；这不随机化物体初态。

模型获得当前全部 RGB-D、标定、完整机器人 guide、公开状态、函数 JSON schema 及执行反馈。工具覆盖像素投影、坐标变换、只读 IK、open／pinch_prepose、腕姿态、指尖／手关节目标、保持及原动作查询／取消。模型输出 `{"function":"hold","arguments":{"duration_s":0.4}}` 这类 JSON；宿主绑定观测序号和动作 ID。棋牌另可输出 `{"intent":{"type":"placement_intent","piece_id":"black_00","q":0,"r":0}}`；意图本身不移动棋子。`{"stop":true}` 请求结束未完成回合。

棋牌由同一可信宿主串行调度两份独立上下文，按裁判当前玩家换手，整局保留各自的文字历史和最新图像。玩家令牌只留在宿主内存，状态、函数与媒体请求均使用本方认证；模型和记录不含令牌。手部 schema 绑定本方关节及 frame；未知 POST 结果只查询原 ID，未决动作阻止新动作和交接，确认拒绝后允许纠正请求。动作 `cancelled/board_turn_changed` 按真实换手处理，不能当作腕目标完成。

新入口当前是 **空技能库、公开 JSON 函数基线**，固定技能条件身份写入配置。没有生成 Python、技能修改工具或持久执行器；task0/1 的学习权限不表示 RSI 调度已经实现。task2/3 在此入口同样只有只读固定工具条件，尚不执行从学习库选出的 Python bundle；跨局选版、完整比赛计划及跨回合指标仍按 [learning](learning/README.md) 的实现边界管理。函数成功、机器人动作完成、任务完成和棋局胜负分别记录；已经确认的棋局结果与运行故障并存，故障运行不计有效成绩。

使用已完整冻结且与服务匹配的源码，从快照根启动。例如以下为命令模板；服务需预先选择对应 task release：

```bash
cd "$INTERFACE_AGENT_SNAPSHOT"
<开发存储>/interface/.venv/bin/python -B -m agent.model_client.function_runner \
  --project-root "$INTERFACE_AGENT_SNAPSHOT" \
  --bench-url http://127.0.0.1:8000 --task task0 \
  --credentials /absolute/path/to/external-profile.json \
  --output <开发存储>/datasets/fr3_sharpa_bench/agent_bench/NEW_RUN
```

棋牌对局改为 `--family card_room --task task1`、`task2` 或 `task3`，并连接发布该棋种的服务。机械候选使用 `--family mechanical_assembly --task L1`（也可 `task1` 或 `parts_sorting`），L2–L9 同理；例如 `--family mechanical_assembly --task task6`、`--task L6` 和 `--task nist_connector_insertion` 在同一 family 下均选择接头插接。服务须预先绑定相应当前 release；这些是启动模板，本次接线不启动新模型或原生实验。客厅使用 `--family living_room --task task1`、`task2`、`task3` 或 `task4`，服务预先绑定对应冻结发布；前三项创建单人回合，`task4` 创建两玩家轮流对局。先手由服务初态决定，不能固定假定 `player_0` 先行。书房使用 `--family study --task task1`–`task4`，厨房使用 `--family kitchen --task task1`–`task6`（Task5/6 加 `--sim-budget none`），实验室使用 `--family laboratory --task task1`–`task5`（均可用 canonical ID），都创建单人回合。

默认模型 `gpt-6-astra`／`max`，两个独立客户端采用相同显式模型条件；不自动切换供应商。`--max-decisions` 为每名玩家的决定上限，`--max-actions` 为整个物理回合共享的动作尝试上限。CLI 校验快照内容与实际导入路径，启动前保存源码验证记录，拒绝复用输出目录。新增测试使用 CPU worker／模型替身验证 HTTP 与 Agent 循环，不代表模型调用、真实抓放或整局物理验收。

公开函数入口默认 sim=120 s、wall=1800 s、model=300 s、HTTP=60 s、输出上限=4096 tokens；各项可显式传 `none` 关闭，决定／动作次数上限仍生效。Micu 的每次请求及重试事件实时写入 `<single|player_N>/decisions/NNNN/model_api_events.jsonl`，终端同步输出请求开始、失败、等待与完成；错误另存脱敏的 `model_error.json`。事件仅包含调用序号、请求 hash、次数、耗时和类型化错误，不保存响应错误正文或凭据。

`runner.py` 使用公开任务、当前 RGB-D、机器人状态和动作反馈调用模型，再通过公开 HTTP 提交动作/工具/受限程序。它没有任务脚本接管、私有物体位置或训练好的 RL。服务契约见[bench/runtime](../../bench/runtime/README.md)，四种固定入口及自由模式见[比较协议](../../docs/agent_bench/modes.md)。

自编技能与场景练习的 agent 侧模块放在 [learning/](learning/README.md)。空技能库、不可变版本和宿主动作恢复已实现；持久 Python、运行时隔离和自动练习调度仍待实现。现有 runner 接入动作账本和结果三态，`bench/protocol/` 仍负责 bench 的公开契约和校验。

`--max-status-queries` 默认 1200，限制每次等待原动作的状态查询次数，取消时间预算后仍生效。共享 GPU 上的录像 debug 可显式增大此值；它不改变单动作仿真收敛超时，也不会重发动作。查询耗尽时保留未决账本，不能当作动作已停止。

用户于 2026-10-18 授权 task0 使用 Micu `gpt-6-astra`／`max` 进行原生公开函数验证，debug 可取消总时间预算；此授权不恢复旧矩阵队列。已授权的零模型脚本工程基线不由本客户端调用模型，也不产生模型成绩。服务与客户端都使用显式冻结根；启动服务与发布状态见[bench 说明](../../docs/agent_bench/README.md)。单次推块入口为 [scripts/run_push.py](../../scripts/run_push.py)。单臂默认 release 为 `v12_mount_v2_r1/sharpa_w01_right`；双臂 `v13_pad_compliance_r2/bimanual` 需匹配的服务、任务和机器人选择，不与单臂结果合并。

该次验证的 独立证据入口（开发存储 `datasets/fr3_sharpa_bench/development/task0_agent_validation_20261018/qualification.json`） 保留两次运行：attempt01 为切换无时间预算和连续录像而主动取消；attempt02 使用冻结源码 `f418f939`、task0 v3、seed 7，45 次 Micu Astra 调用均返回，模型仅凭公开观测完成三子放置，并自行重新抓放纠正第三枚偏格。64.30 s 仿真、128,600 个原生子步的评分重算与完整接触记录一致，但首次释放第三枚时 `right_index_MP` 与棋子的求解器报告间距达到 −4.030 mm，超过冻结审计的 2 mm 阈值。最终任务评分为 3/3，权威结果为 `safety_stop`、`physics_valid=false`，**不计有效成功，也不构成完整物理资格或 RSI 验证**。运行故障为空，任务结果保持 `unknown / physics_not_validated`。模型请求参数为 `max`，不据此证明上游内部 effort；bench 端模型计数不可用，应读客户端的 45 次调用记录。完整 2 倍速录像（开发存储 `demo/interface/agent_demos/task0_agent_validation_20261018/attempt02/task0_2x.mp4`） 保留纠正过程和无效物理结论，按仿真时间加速，省去模型与计算等待。

## 连接与启动

当前连接配置为 `configs/agent_bench/local_gateway_credentials.json`；默认端点 `http://127.0.0.1:8788/v1`，模型 `gpt-6-astra`、请求 reasoning_effort=max。请求参数会记录，但不能据此证明上游内部推理实现。不会自动换模型或降档。

凭据配置使用 `base_url` 及恰好一个 `api_key_file` 或 `api_key`。推荐外部密钥文件引用；相对路径以配置文件真实目录解析，支持绝对路径和 `~`，不展开环境变量。不把密钥值写入项目、命令行或实验。也可省略 --credentials，使用 DEX_AGENT_BASE_URL/DEX_AGENT_API_KEY。端点元数据脱敏；运行中的客户端不热重载配置。

用户指定的 Micu 使用 `--provider micu --credentials ~/.config/dex-agent-bench/micuapi.json --model gpt-6-astra --reasoning-effort max`（凭据文件放在各自机器上，不进仓库）。客户端读取该外部 profile 的实际地址，通过 Responses 传多轮文字/图像，不替换为本机网关。Micu 必须显式选择，使用 Micu profile 却省略 provider 会拒绝启动。

Micu 对网络中断、请求超时、HTTP 408/429/5xx 默认等待 **10 秒**后重试同一请求，恢复后继续原决定；不另起回合、不丢历史、不重复物理动作。默认不限制重试次数，直到恢复、用户中断或既定 wall budget 耗尽。`--model-timeout` 分别约束每次请求，等待不耗尽下一次请求的超时额度；总 wall budget 仍计入等待时间。可显式用 `--model-retries 0–3` 设置额外请求次数上限；旧 gateway 默认仍为 0。401/403、无效请求、模型不符或输出 token 耗尽会单独报出，不无限重试，也不自动换供应商、模型、推理档或协议。真实连通性检查不代表机器人任务验收。

新 suite 计划冻结 `model_retry_policy=provider_default_v2`；缺少该字段的历史计划按旧策略只读解释，历史 Micu 队列不能按新语义续跑，需要新建计划。`decisions/*/model_api_events.jsonl` 持续保存请求次数、故障类别及实际等待时间，终端同步显示等待事件；缺失的中断请求 usage 保持未知。

下面是未来显式启动时的通用模板，BENCH_ROOT 为新的完整源码快照，BENCH_PYTHON 为已有前端解释器：

```bash
cd "$BENCH_ROOT"
PYTHONPATH="$BENCH_ROOT" "$BENCH_PYTHON" -B -m agent.model_client.runner \
  --bench-url http://127.0.0.1:8000 \
  --credentials <开发存储>/interface/configs/agent_bench/local_gateway_credentials.json \
  --model gpt-6-astra --reasoning-effort max \
  --task 01_cube_push --robot sharpa_w01_right \
  --control-mode fingertip --interaction-mode action \
  --visual-input rgbd --tactile-input on \
  --max-decisions 128 --max-actions 128 \
  --output <开发存储>/datasets/fr3_sharpa_bench/EXPERIMENT/client
```

输出必须不存在。本例未关闭时间/输出上限：CLI 默认 sim=45 s、wall=1800 s、model/http=300 s、program=120 s、max_output_tokens=16384。128 次数是示例开发预算，不保证成功。需要关闭某项才显式传 `none`；矩阵 JSON 使用 null。关闭输出 token 上限时请求省略供应商字段，不发送 null 或伪造大数。

`scripts/run_push.py start` 与上面的通用模板不同：它固定单臂推块、128 次决定/动作，显式关闭 sim/wall/model/http/program 时限及输出 token 上限，并要求外部 key 文件形式的 8788 profile。它不启动服务、后台进程或矩阵；每次必须给出新 `--run` 与明确的 `--project-root`。

开发模板位于 `configs/agent_bench/suites/{rigid_fixed,rigid_free,cloth_free}.example.json`，必须按新服务和 release 固定计划，不复用旧 hashes。suite 使用普通回合 runner，计划阶段拒绝棋牌或轮流对抗发布并提示改用 `function_runner`；后者已有三类会话适配器。单个服务内部串行；并发实验需各自独立服务、回合和输出。

## 当前输入与历史

RGB-D 模式每轮给模型两张 RGB、两张深度预览以及每相机 16×12 公制采样网格。原始 float32 NPY 先验证 SHA、形状、时间与标定后保存。深度是 camera_z（米），无效值 0；网格无效项为 null。预览映射 0.1–4 m，饱和只影响预览，不裁剪原始深度。

client schema 为 `online_visual_agent.v11`（v10 记录保持原样），历史协议为 `dynamic_history_v1`。当前轮仍发送完整 task、capabilities、action_limits、状态和反馈；前两轮文字历史移除重复 task/capabilities/action_limits，保留动态状态与模型回答。旧轮图片不重复附带；新轮实际图像与完整展开消息摘要继续归档。这项输入缩减不等于已测得模型延时或任务成功率改善。

截图下载、深度转换、提示词构造和记录都消耗 wall 预算；模型调用前重新计算剩余时间，用其约束请求。截图阶段已耗尽预算时不再发模型请求；迟到响应也不能继续发动作。

RGB-only 不给深度，也不允许 point_at；tactile off 移除触觉与接触历史，包括工具/代码反馈。私有真值从不进入提示词。

## 动作、工具与程序

action 模式一次返回 joint_delta、move_geometry、hold 或 stop。目标由当前公开状态和具名关节/局部指尖约束构造，不静默裁剪。工具模式一次 JSON 调用 observe、point_at、move_geometry、hold、stop；point_at 仅反投影当前公开深度像素，没有语义定位或抓取方案。

`agent_control_v6` 的 joint_delta 以公开的 `commanded_joint_positions_rad`（已确认的驱动目标，相当于真实机械臂的 q_d）为基准：目标 = 指令目标 + 增量，省略关节和 hold 保持指令目标；未公布指令目标的旧服务仍以实测为基准。历史 v5 及以前以实测位置为基准，曾在无重力补偿的记录中造成每个动作约 9–11 mm 的手臂下沉、未指令手指漂移和动作边界握持预紧丢失；这些是历史失真记录，不能当作当前发布的预期误差。

服务已公开指令目标时，对显式指定增量的关节，只有新目标与实测的距离同时超过配置阈值和原有目标间距，才拒绝该增量。`max_arm_target_gap_rad` / `max_hand_target_gap_rad` 默认分别为 0.10/0.40 rad；已经超过阈值时允许保持或缩小原有间距，拒绝继续拉大间距。实测超出 URDF 限位超过 0.03 rad 时判定状态无效；带内仍保留原始测量，构造增量前将所用基准限制在 URDF 内。服务端按 `capabilities.direct_joint_execution` 执行：`min_jerk_from_acknowledged_target_v1` 从上次确认目标出发，在 80% 的时长内（不足以满足 0.65/1.0 rad/s 时按速度上限延长到最多整段）做 min-jerk 插值后保持终点，终点不变。

`move_geometry` 在 action、tools、code 和 free 模式均可选传有限正数 `position_tolerance_m` 与 `orientation_tolerance_rad`，精确映射既有 System0 腕部目标容差。控制器仍取请求与自身完成阈值的较小值；不传时保留默认值。没有可选的指尖容差参数，也不改变关节、速度或接触限制。该扩展从 `agent_control_v5`／`free_tool_agent_v2` 引入，当前 `agent_control_v6` 保留；微小下压必须明确请求合适精度，动作完成仍不能代替真实接触或物体进展反馈。

code 解析 `agent_program_v2` 有界 AST，支持变量、JSON 下标、有限算术、if、字面量 range、break/continue 与 abs/min/max/len。禁止 import、attribute、函数定义、while、推导式、文件和网络，不使用 exec/eval。变量不跨程序持久化。

代码每次移动仍计 primitive 尝试，默认每程序最多 3 次物理动作、每决定最多 8 次工具调用，另限源长、数值、容器和 fuel。执行反馈保留每个 primitive 的失败；后续成功或查询不覆盖先前失败。动作已产生的物理效果不会因反馈预算超限而回滚或自动重试。

客户端物理尝试采用 `recognized_physical_request_v1`，完整规则见 [modes.md](../../docs/agent_bench/modes.md)。call_status 表示工具调用返回，primitive_status/feedback.status 表示动作结果，任务成功仍读服务终局。model/program/http 与 wall 的有效期限共同约束操作；同步宿主阻塞不能靠 AST 抢占。

## 证据与职责

| 文件 | 内容 |
|---|---|
| agent_config.json | 模型/端点、预算、协议、控制器与源码摘要 |
| episode_ready.json | 服务公开初态、能力与冻结 spec |
| decisions/*/observation.json、PNG、depth.npy | 实际使用的公开快照和深度 |
| decisions/*/model_request.json | 文字、图像引用/SHA、展开消息摘要 |
| decisions/*/model_response.json | 原回答与网关 token/时延元数据 |
| action.json、execution.json、tools/、program_execution.json | 命令、工具与程序反馈 |
| run_report.json、artifact_hashes.json、client_source/ | 终止原因、最终记录与客户端归档 |

`online_visual_agent_run.v7` 分开记录 `run_status`、`task_outcome` 和可空的 `task_success`。无有效任务终局时 success 为 null；API 中断或未确认动作后的清理取消不计任务失败，模型 API 故障期间产生的 watchdog wall_timeout 也保留未知，已确认有效终局仍保留。`run_error`、`finalization_error` 和 `result_source` 保存归因。正常预算耗尽、主动停止及真实任务失败不自动归为基础设施异常。历史 v6 只在汇总时解释，不改旧记录。

普通 runner 的轮询、动作后读取及结束清理均校验 `EpisodeResult`，并将完整 `result.spec` 与创建时保存的独立副本比较。缺少 spec、控制器或其他回合身份漂移会记录为无效响应，任务结果保持未知；后续清理不能用另一个成功结果覆盖这个身份错误。报告通过 `result_validation_error` 保留校验故障及原始运行错误；suite 重读报告时仍拒绝计分。身份有效的终局结果不会因随后发生的网络故障而被抹除。历史实验文件保持不变。

报告仍保存模型尝试、primitive 尝试、提交、完成数与原始终局。Responses 的 input/output token 与旧 Chat 的 prompt/completion token 按协议读取；有中断重试而缺少全量 usage 时总成本保留未知。模型 stop、动作 completed 或预算耗尽不等于任务成功；计划槽位数、有效覆盖和未知原因随成功率一起显示。

每次执行的 `host/action_journal/` 保存提交前的动作身份与内容、回执和恢复事件。恢复只查原 ID，取消确认也必须等终态与对应公开快照。新 InteractionSession 拒绝已有账本，完整回合续跑尚未支持；独立 RobotAPI 可用于收束原回合的未决动作。

`action_recovery_policy` 披露每次等待默认最多 1200 次状态查询操作，独立于可关闭的任务 wall budget；耗尽后仍保持 pending。客户端严格复核公开动作反馈字段，嵌套诊断、机器人执行状态和接触历史均使用已有公开 schema。

runner 负责闭环，transport 负责模型请求，bench_http 只访问公开 API；geometry_actions 根据机器人 FK 构造目标，服务端控制器执行 IK。interaction/program 统一工具与代码预算，capture/visibility 统一输入投影，suite 组织开发计划与归档复核。客户端可远程部署，但必须保留相同的源码与 spec 证据。
