# Agent 技能学习

本目录归属正式模型客户端。已实现空技能库、不可变版本与来源管理、持久宿主事件、公开动作的查询与恢复；现有 runner 已接入动作账本与任务结果三态。持久 Python 执行、隔离挂载和自动练习调度尚未实现，现有 code 模式仍是有界 AST。

六个场景共用本目录中的学习运行时。机械装配、实验室、厨房、书房和客厅统一采用任务已知 RSI：公布任务 → 预算内多次练习和修订 skill/code → 独立验证选版 → 冻结测试。五个场景仅绑定各自的任务、公开能力、规则、初态和预算，不复制五套学习器。`task_conditioned_rsi` 是设计标识，尚无对应 CLI；详见[通用任务 RSI 与中间指标](../../../docs/agent_bench/skill_learning.md#task-rsi-mode)。

实验室 Task1–5 分别为 `sample_sorting`、`vial_racking`、`cap_placement`、`solid_weighing`、`pipette_transfer`，均已接普通单 agent `/episodes`。双台面任务仍由同一个 learner 控制单台 FR3 + 右 SHARPA；五项共用上述任务已知 RSI 权限，实验室 task2/task3 不套用棋牌冻结比赛政策。`--family laboratory` 将任务编号绑定到 catalog 的冻结发布，学习记录继续按 canonical task ID、release、learner 和独立 run 隔离；物料账本、原生接触和评分器中间状态不成为学习输入。规则／粗粒度进展只读 bench 公开合同。CPU 接线、原生初态与物物检查都不代表手物交互或完整 RSI 验收，详见[实验室入口](../README.md#实验室)和[接入约束](constraints.md#laboratory-adapter)。

学习权限与参与者／会话是独立维度。客厅 Task1–3 是单 agent、单臂任务；Task4 `tower_extraction` 是两个独立 agent 各控一臂的轮流对抗，保留客厅既有 RSI 权限，不套用棋牌室 task2/task3 的禁 RSI 政策。双方交替抽出木条放入己方收集区，不重新叠到塔顶，造成倒塌的一方输；`player_0`／`player_1` 分别绑定 `front`／`rear` 两套原始右手机器人。四项均为 `implemented` 工程候选；Task1–3 已接普通 `/episodes` 适配器，Task4 已接认证 `/turn-games` 适配器，按公开裁判当前玩家串行调度独立上下文。一轮可有多次动作，只有稳定交接后换手；一次函数返回不等于交接，换手不清空历史或重置成本。原生 reset 阻断与完整操控资格仍按对应发布单独记录。

中间指标共用任务尝试、动作请求、实际物理交互、只读查询、模型逻辑调用/API 重试、技能修订和资源成本；重点报告首次成功成本、首个达标版本成本及冻结版本成功率/有效覆盖。分场景、任务和阶段汇总；不同任务的物理阶段进度仅消费其公开合同，不统一编造进度百分比。已有单回合计数和 journal 可复用，跨回合汇总与学习曲线仍待实现。

棋牌室按用户 2026-10-18 的澄清区分四项任务。任务编号表示具体任务，学习权限与会话路由分别绑定：

| 用户任务编号 | canonical task ID | 会话与学习权限 |
|---|---|---|
| task0 | `task0` | 普通 `/episodes` 单 agent 回合；允许练习和 RSI |
| task1 / L1 | `tic_tac_toe_3x3` | `/board-games` 双玩家井字棋；允许练习和 RSI |
| task2 / L2 | `connect_four_5x5` | `/board-games` 双玩家平面连四；冻结技能，禁止 RSI |
| task3 / L3 | `hex7x7` | `/board-games` 双玩家 Hex；冻结技能，禁止 RSI |

task0 与其他五个场景共用任务已知 RSI，使用一套 FR3 + 右 SHARPA，将三枚黑子分别放入本回合随机选出的三个不同棋格。每个 learner 使用独立的普通 episode、上下文、工作区、技能库和预算，没有对手或换手机制。目标来自创建回执 `ready['task']` 或 `/episodes/{episode_id}/task`，其中 `goal.target_cells`、`goal.target_positions_m` 和 `public_rules` 已绑定本回合 `episode_seed`；`/tasks` 仅列任务模板，不能充当实际目标。省略 seed 会抽取新目标，固定 seed 可复现；这改变目标，不改变物体 reset 初态。任意一枚指定黑子可填任意一个目标格，不按两列表位置固定配对。

task1 仍使用井字棋的认证玩家、共享棋局和轮次，不因允许 RSI 改成普通单 agent episode。允许在练习预算内写改、验证和选择技能；双方的上下文、工作区、技能版本、开发记录和模型成本分别管理，不自动共享学习经验。task0 的任务成功和 task1 的胜负都由对应 bench 合同判定，RSI 是允许的解题过程；技能数量、修订次数和单次动作连通都不替代完整结果。持久 Python、隔离运行和这些学习阶段调度尚未实现。

task2/task3 在首局前绑定双方各自选定的 bundle、来源和兼容条件，整组比赛不换版，关闭技能写改、候选发布、验证选版、导入和经验写回。允许局面推理、当前动作计算、调用/组合冻结技能和合法恢复；不能借 scratch 或内存覆盖技能、调优可复用参数。每局使用新的独立上下文，换手保留本局状态，结束后不向下一局携带经验。从 task0/task1 导入技能须显式记录来源和成本；其成功不自动证明 task2/task3 的抓放或整局能力。

bench 已发布 task0 的普通 episode 合同，以及三个棋种的 `/board-games` 认证玩家、公开函数、状态、意图和结果路由。客户端必须按任务使用对应会话，不能把 task0 包成比赛或把三棋种送入普通 `/episodes`。Agent transport、函数工具与比赛运行入口的具体实现状态统一见[客户端说明](../README.md)；接线不代表持久 Python、运行隔离、阶段调度或模型物理操控验收已完成。详见[接入约束](constraints.md#card-room-adapter)。

普通任务与比赛共用公开函数 SDK 和动作账本，分别按其会话和结果合同运行。生成代码运行与隔离、共用练习/修订/验证、冻结版本交接及跨回合指标仍待实现。机械装配等普通任务、客厅 Task1–3 及棋牌室 task0 的学习闭环不依赖比赛运行器；客厅 Task4 和棋牌室 task1/task2/task3 需要各自的玩家会话，函数接线不以完整 RSI 调度器作为前置。

统一模式不代表物理任务已通过。机械 L1 `parts_sorting`、L2 `peg_insertion`、L3 `shaft_spacer_assembly`、L4 `fastener_assembly`、L5 `gear_train_assembly` 均注册为 `implemented` 场景候选，发布仍为 `prepared_unvalidated`，有界接触证据不等于机器人任务成功。固定初态多次 reset 只能检查复现，不能称为新初态泛化。客厅四项冻结发布保持 `prepared_unvalidated`／`physics_accepted=false`，客户端接线不等于完整操控资格；Task2 当前采用实用水耦合 profile。Task4 v8 已有 seed 0/front 先手的 canonical reset 和微动证据，旧版 reset 阻断记录保留；尚未验证完整抽取、收集、交接或 rear 先手。Agent 对 reset_pending 提供显式固定驱动保持，不能把复位未就绪当成可选块阶段。任何后续标记为 planned 的任务仍不可运行。机械和客厅的具体边界见[接入范围](../../../docs/agent_bench/skill_learning.md#mechanical-assembly-mode)及[客户端说明](../README.md)。

连续计分任务独立于二元成功指标：客厅 Task3 按 Bench 公开终局的最高稳定高度保留成绩，正常预算结束不记为失败，`task_success` 为 null。报告中的 `task_score_valid` 还检查执行健康和未决动作，不能仅因历史高度数值存在就作为有效评估样本。未来学习曲线应分别消费连续成绩与二元成功，不能将前者的 null 计作零成功率；通用 Bench 汇总已按计分模式分组。

任务规则、允许行为、成功/失败判定和评测协议由 bench 侧统一发布。本目录负责消费该公开合同并实现 agent 运行时约束，不维护另一套任务规则；规则交付与版本绑定要求见下方约束设计。

- [Agent 侧约束](constraints.md)：agent/learner/skill/candidate/attempt 定义、自改进边界、公开函数和棋牌室接入、权限、阶段与证据合同。
- [自进化库设计](storage.md)：共享 datasets 下的目录、读写权限、不可变版本、发布恢复与跨场景导入。
- [整体设计提案](../../../docs/agent_bench/skill_learning.md)：课程、任务资格、验证与迁移评测，以及双方分工。
- [类似方法与 benchmark 的设计取舍](../../../docs/agent_bench/skill_learning.md#design-references)：官方来源、可采用的做法及本项目不继承的假设。

目录边界如下：

| 位置 | 归属 |
|---|---|
| `agent/model_client/learning/` | 六场景共用的学习运行时、技能版本、调度和统计源码及约束；不按场景复制运行时 |
| `agent/model_client/` | 现有模型传输、公开 HTTP 客户端、动作构造与输入投影 |
| `scenes/workrooms/<family>/tasks/<task_id>/` | bench 维护的任务定义、规则、评分与发布引用；不存 agent 自编技能 |
| `docs/agent_bench/skill_learning.md` | 共用学习流程和指标、棋牌室例外及场景接入限制；存储细节统一见本目录 `storage.md` |
| `bench/protocol/` | bench 定义的 Agent 数据契约、动作校验、时钟、策略插件协议、记录与统计 |
| `bench/runtime/` | bench 公开服务、原生 worker 通信与观测发布 |
| `system0-system1/` | 独立 student／私有 RPC 工程；可参考运行时设计，不作为本目录的执行后端 |

共用模块按职责命名。比赛的玩家会话、上下文与轮次由客户端运行器编排；学习调度器后续负责阶段权限、冻结版本交接和跨回合统计，复用公共库与账本，不实现第二套 RSI。不为 planned 场景预建空运行器或任务攻略目录。

本目录不存放运行中生成的技能、轨迹或模型回复。自进化库默认位于 `<开发存储>/datasets/fr3_sharpa_bench/skill_learning/<experiment_id>/`，默认一个任务、一个 learner、一次独立学习 run 一个库，`author/` 保存可编辑草稿，`bundles/<hash>/` 保存已发布版本，证据与选用记录由宿主另存。场景、任务和学习者身份写入冻结合同；详情与命名约定见 [storage.md](storage.md)。项目源代码位置、开发 worktree 与学习模型看到的工作区是三个不同位置，不能直接把整个项目 checkout 挂载给被评测 agent。

## 已实现的宿主接口

| 模块 | 当前能力 |
|---|---|
| `skills.py` | `SkillLibrary.create`、revision 保护的 author 读写、源码静态解析、完整内容 hash、发布与恢复、验证证据、选用指针和显式导入 |
| `journal.py` | 严格 JSON、原子写入、单写者锁、顺序与哈希链校验、事件 ID 幂等 |
| `robot_api.py` | 提交前持久化命令、同 episode/trial/spec/服务绑定、单个 pending 动作、原 ID 查询/取消、终态与公开快照核对 |
| `../outcomes.py` | `run_status` 与 `task_outcome` 分离；缺少有效任务终局时 `task_success=null`；已有有效终局不被后续 API 故障覆盖 |
| `../responses.py` | 显式 Micu Responses 接入，多轮文字/图片、瞬态异常等待 10 秒后同请求重试、无供应商/模型/协议回退 |

新库只有空 catalog，没有抓取或放置程序。所有影响执行与检索的源码、参数、笔记和 catalog 都进入 bundle hash；验证结果另记宿主事件。`record_validation` 和 `select` 是可信宿主接口：库保存 bench 证据与晋级决定，不自行编写评分规则，也不能直接暴露给生成程序调用。

```bash
python -m agent.model_client.learning.skills init <experiment_id> \
  --contract /absolute/path/to/frozen_run_contract.json
python -m agent.model_client.learning.skills status \
  <开发存储>/datasets/fr3_sharpa_bench/skill_learning/<experiment_id>
```

`--learning-root` 可显式配置共享存储根。初始化拒绝覆盖已有 run；恢复使用 `recover <library>`。合同为可信宿主提供的 JSON 对象，`dependencies` 固定 import root 到精确版本，`compatibility` 保存场景/机器人/观测等版本约束；完整学习调度器对 bench 规则合同的接入仍待实现。

宿主通过 `write_author(path, content, expected_revision=...)` 更新草稿，随后调用 `publish(revision, transaction_id=..., reason=...)`。提交返回的 candidate 事件绑定 bundle hash 与来源；丢失回执时用相同事务查询/重试，不重复生成候选。`verify(hash)` 检查完整内容，`recover()` 续接有 prepared 记录的发布，隔离无已提交事务的暂存树。未知来源和损坏版本不获得执行资格。

## 动作恢复与结果

现有 runner 为一次执行保存 `host/action_journal/`。提交前记录 payload 和 hash；POST 不自动重发。查询按原 action ID 进行，只有权威终态与对应 episode、action ID、状态、时间的公开快照一致后，才解除 pending 锁。取消请求的回执本身不证明停止。

每次恢复等待默认最多 1200 次状态查询操作，写入账本绑定和 runner 配置；SDK 构造参数可固定其他正整数上限，单次 wait 只允许进一步缩紧。即使任务 wall budget 关闭，次数上限仍有效；HTTP 读取重试另受其请求上限约束。耗尽只封存未决状态，不发送新动作。公开动作反馈按服务的字段白名单和嵌套 schema 校验，未知字段不会原样转给模型。

`RobotAPI` 可以按原绑定重开账本，执行 `pending`、`reconcile()`、`wait()`、`cancel()`。它恢复的是宿主动作状态，不恢复 Python 栈或模型上下文。`InteractionSession` 拒绝用已有账本开启新会话，防止计数清零后重新获得预算；整回合续跑及 slot/attempt 补跑调度尚未实现。创建 episode/reset 超时仍不盲重试。

Micu 瞬态 API 故障默认等待 10 秒重试，保留同一决定的上下文；单次请求超时与整个回合 wall budget 分开，重试细节见[客户端说明](../README.md)。若重试被中断或耗尽 wall budget 且没有有效终局，报告保留 `infra_error` 和 `task_outcome=unknown`，并记录 pending 状态。故障后清理取消、模型 API 故障期间的 watchdog wall_timeout 不冒充任务失败；已独立确认的有效终局仍保留。输出 token 耗尽、正常本地预算耗尽、主动停止和自身程序错误分开记录，不都归为外部 API 故障。正式补跑和评分仍消费 bench 的规则。

## 当前边界

这些模块由可信宿主调用。文件权限与 advisory lock 约束协作写入者，不提供对同 UID 恶意进程的隔离；`status()` 明确返回 `runtime_isolation_implemented=false`。不应将库管理对象、host 路径或完整项目 checkout 交给生成代码。源码解析不会执行代码，也不证明动态依赖或运行时访问已被隔离。

发布兼容当前 CPFS：文件系统不支持 `RENAME_NOREPLACE` 时，在目标目录的协作锁内检查目标不存在并执行原子 rename；已有版本仍拒绝覆盖。当前共享盘证据覆盖同主机多进程协作，不代表跨主机锁或掉电恢复已验收。

CPU 故障注入检查存储、回执和结果协议；共享存储实测检查当前文件系统操作。不把它们称为掉电一致性、真实抓放能力或完整 RSI 学习验收。对应任务仍需 bench 提供规则、冻结练习初态和物理可操控性证据。

完整学习闭环仍需持久代码/文件工具与运行隔离、共用阶段调度、干净上下文验证及累计指标；它们与普通/比赛函数运行入口分开验收。本文约束不构成已完成的 RSI 或物理操控证据，也不授权新增模型或原生运行。
