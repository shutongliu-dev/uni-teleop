# 机械加工／装配室

**NIST 去重扩展：** 原 Task1–5 后新增 Task6 接头插接、Task7 带轮／皮带装配、Task8 单线布线、Task9 多线配对与拆除。ATB1–4 各保留不同操作，ATB5 与原紧固任务重复，不另加 Task10。目录、规则、物理近似和原生验证范围见 [NIST 任务说明](NIST_TASKS.md)。下面“五任务”的历史记录继续保留其当时范围；实际发布身份以各任务 `definition.py` 为准。

**2026-10-20 默认任务更新：** L4 `fastener_assembly` 改为 loose_bolt_candidate_v10_pgs（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/releases/loose_bolt_candidate_v10_pgs/release_manifest.json`），manifest SHA256 `47e574e81558ab16e7fffcda9846f2ab8c281660b0c85bb9ee507b95c94f262d`，目标是拾取螺栓、对准起牙、工具接合、旋拧和释放。v10 保留 v9 的 `loose_bolt_v5` 供料窝、工具托、reset、资产、规则和阈值，仅显式选择 PGS（0.5 ms）；全局 TGS 默认不变。预接合接触、完整运动审计和初态 HTTP smoke 均为范围受限通过，仍为 `prepared_unvalidated`，没有完整机器人任务成功；旧预插入、v7、v8、v9 发布及证据保留。

历史 v8 HTTP smoke（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_l4_loose_default_20261020/l4_http01/qualification.json`） 因供料窝误带侧轨而失败：轨道内侧 x=±20 mm，小于螺栓头半径 22.5 mm，reset 约 2.5 mm 穿透；首 20 ms 峰值约 3.009 mm，触发 `reported_penetration_bound`，`physics_valid=false`。2026-10-20 `mechanical_bench_check_20261020/l4_http01` 则绑定旧预插入版；两份记录均不能给予 v10 资格。

v10 的 PGS 接触报告（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/contact_pgs_native01/result/report.json`）通过 8,800 个原生子步；完整运动审计（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/contact_pgs_native01/complete_stream_motion_audit.json`）覆盖 8,760 个滚动 20 ms 区间和 80 个阶段边界区间，螺栓峰角速 0.513870 rad/s，低于 0.523599 rad/s 上限。PGS HTTP smoke（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/pgs_http01/qualification.json`）通过四路 RGB-D、只读 IK、微动／回位／保持，`model_calls=0`。这些证据仅覆盖预接合接触和 bench 初态接入，不包含散装拾取、起牙、插刀或完整装配；v9 TGS 初末被动稳定及 1.251275 rad/s 螺栓角速失败保留。

先读[当前交接记录](../../../docs/scenes/mechanical_handoff.md)，包含已整合代码、L3/L4 终局报告、L5 大齿轮桌边修复／新图及未完成项。历史 `l4_native_timed01` 已按用户要求终止，退出 143，属于未完成记录。五任务尚未完成原生操控验收；普通 Agent 验证保持关闭。下面的 2026-10-18 外观记录属于历史 run05；当前组合保留功能供料托，并采用新的 L5 inset 布局。

五个单臂任务共用一个固定房间、工作桌、机器人基座与待命布局。先丰富车间环境，再逐项实现任务。房间保持空操作面；零件、夹具、工具和初态由任务组合。`scene.py` 固定父环境，`robot_layout.py` 提供共用的 `mechanical_single_arm_v1` 布局，任务继续使用 `tasks/<task_id>/definition.py`。

任务操作与裁判行为先读[五任务中文规则](RULES.md)：涵盖各任务成功条件、阶段／原因码、同回合恢复、终局分类，以及公开粗状态与私有精确证据的边界。当前源码为 L1–L3 规则 v2、L4 loose 规则 v5（历史 preinserted 保持 v4）、L5 v3；每次运行仍以其实际冻结发布为准。CPU 规则测试不赋予物理资格；L5 约 47 µm 的 SDF 诊断误差也不是新增任务装配门槛。

当前环境为 `mechanical_workshop_20261017_v3`，新增钻床、小型封闭机床、工程图及量具／齿轮陈设。统一基座后移至 `(-0.23,0,0.78)m`，底脚距桌沿 20 cm。环境、独立机器人／指定工具预览及证据见[场景构建说明](../../../docs/scenes/mechanical_workshop.md)；原生预览图（开发存储 `demo/interface/scenes/mechanical_workshop_20261017_v3/workcell/room.png`）仅检查外观，不代表任务运行。

2026-10-18 的新外观／布局候选由 冻结 source06 中的 renderer（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_visual_upgrade_20261018/source06/scripts/media/render_mechanical_presentation.py`） 组合。每项任务新增 5 件有碰撞、可自由移动的非评分散件（两螺母、两垫圈、一短螺栓），使用共享 `assets/task_parts/mechanical_assembly/table_hardware/v1`；评分目标数量不变。去掉多余供料垫块，L2/L3 供料件和 L5 六个自由零件按真实边界放到原桌面上方 0.2 mm，保留 L1 目标浅盘、功能夹具、齿轮安装板与机器人支座。独立 presentation 层的材质和可视细节保持其父版物理不变；新散件、供料初态和 L5 几何则构成新的物理候选，不继承旧布局的接触／操控资格。

新风格预览位于 mechanical_visual_upgrade_20261018（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_visual_upgrade_20261018`）。`run01` 在 session-layer 前置检查中止，未产出五任务预览；`source02/run02` 与 `assets/task_parts/mechanical_presentation/v2` 的历史记录保留。当前 run05 provenance（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_visual_upgrade_20261018/run05/provenance.json`） 已封存五任务 `preview.usda`、任务图、近景图、逐任务 presentation audit，以及 L4/L5 工具细节图；其 presentation asset 为 `assets/task_parts/mechanical_presentation/v4/worktop.usda`，SHA256 `d414e232807222dcadb9d765ecbed9be72f1d5d02c41e6b8073942277d0653f5`。run05 是独立的 `prepared_unvalidated` 视觉／布局候选，增加五件 task-owned 桌面散件；它尚未替换 catalog 绑定，仍需按对应物理 release 重做 reset、碰撞、native 和 RGB-D 审计。静态预览不代表渲染或动力学通过。下面的已有发布与原生证据保留其历史适用范围。

L4/L5 另有独立 task-owned toolbox v7 manifest（开发存储 `assets/task_parts/mechanical_assembly/toolbox/v7/manifest.json`） 和 CPU report（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261019/toolbox/task_owned_toolbox_v7/report.json`）。箱体、动态盖板与旋钮都有独立视觉／碰撞资产；源几何和四份场景的变换后包围盒检查通过。铰链／锁扣尚未实现，原生承托、开盖、相机观察和机器人可达性仍需验证。候选未绑定 L4/L5 release，固定房间 toolbox 继续作为房间陈设保留。

**L1–L4 已注册为 `implemented` 的可运行场景候选，L5 齿轮仍为 `planned`。** 目录分别绑定分拣分列布局／SDF 环、插销 PGS、轴套 dense SDF 三环和 loose bolt v10 PGS 发布；正式初态、规则、评分器与资产清单均冻结。发布仍为 `prepared_unvalidated`，没有机器人任务成绩；已验证的场景接触条件及限制见[交付对应表](../../../docs/scenes/mechanical_workshop.md)，逐项特权报告和当前失败／未完成边界见[机械装配室特权验证索引](../../../docs/scenes/mechanical_privileged_validation.md)。共享 bench/agent 可从目录加载这些修复，场景交付不以新增任务专用控制器为前提。

L5 后续有界查询、八片 dense 和关闭网格焊接三轮原生对照已完成记录。验证器已区分完整失败与内部校准中断，并强制核验新 cooking profile 在实际场景中的属性。subgrid2／weld0 仍有 26 个整件投影失败点，dense1250 上部采样全零导致八片校准中断；没有取得几何或传动资格。具体证据及下一步容量假设见[修复续报](../../../docs/scenes/mechanical_workshop.md#l5-后续修复有界查询与存储路径对照)。

2026-10-18 的[场景动力学核验](../../../docs/scenes/mechanical_workshop.md)覆盖全部五项。五项任务的实体资产、组合、冻结发布工具及评分器已补齐，当前目录只开放 L1–L4 的已修候选。插销在原步长通过供区、正对、偏置导向和孔外负例；PGS 半步长延至 8 s 后通过最后半秒静置，但原 5 s 失败保留，不能宣称瞬态步长收敛。分拣的保留源面单 SDF 环通过四组初态与定量承托；轴套的 dense SDF 三环通过供区和低位落座，其余原初态补验也通过真实导入／承托及末窗静置，但高位正放／错序超线速限制，偏置例还超瞬态角速限制，均保留失败。螺丝的新共同相位 reset 候选通过四个固定窗口导程，原 0.5 rad/s 工程载荷下峰角速与末窗工具线速仍失败；独立的 0.35 rad/s 对照在 0.5／0.25 ms 均通过原导程、原生速度及初末静置门，但半步长姿态差分超界及约 9.17° 末姿态差保留，不能称瞬态收敛；齿轮原候选传动停转，新单 SDF／八扇区候选也尚未通过完整几何门槛。共同房间通过不能替代各装配接口通过。scene 侧负责几何、碰撞、物理参数、初态、夹具和接触约束；控制器、握持策略和轨迹调参由统一 bench/agent 负责。

L4 `native18` 的独立主报告（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/fastener/native18_independent_review01/result/report.json`）和COM／尾窗接触附录（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/fastener/native18_independent_review01/peak_and_contact_addendum.json`）覆盖完整 8800 子步。相对 17 仅独立工具 COM 工程自转目标 0.5→0.35 rad/s；同 reset、3 s 加载／0.8 s 释放、PGS、0.5 ms、几何、物性、全部增益／限幅及门槛不变。0–33／5–25／5–30／5–33° 导程误差为 −1.0091%／−0.6244%／+0.0666%／+0.3936%，全过原 5% 门槛；全程原生／逐子步四元数差分角速峰分别为 0.504398／0.521379 rad/s，后者距原 0.523599 上限仅 0.424%。末半秒工具／螺丝原生峰线速 1.057889／1.148293 mm/s，COM 差分也低于 2 mm/s；初末静置、全程穿透及封存通过。17 原载荷的峰角速 0.656063 rad/s 与释放工具线速 2.239422 mm/s 两项失败保留，不能被 18 覆盖；偏轴及反转仍属未测试范围，未修改共享控制器，也未取得机器人任务成绩。


L4 `native19` 同条件半步长完整复核（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/fastener/lower_spin_halfdt_qualification01/report.json`）只把步长改为 0.25 ms，17,600 子步完整结束；四个固定窗误差 −0.693734%／−0.092900%／+0.472715%／+0.776687%，原生峰角速 0.509225 rad/s、原 20 ms 位姿门、初末静置及穿透均通过。末半秒工具／螺丝原生峰线速为 1.127898／1.096631 mm/s。但逐子步归一化四元数峰 0.551317 rad/s 有 5 个超界区间，螺丝末窗 COM 差分峰 2.165170 mm/s 有 3 个超界区间；均作为独立诊断保留，不新设硬门或改判原生速度。18/19 在相同 3 s 加载后的实际转角为 54.307943°／45.304942°，共同 20 ms 采样的终态姿态差约 9.17°，不能宣称瞬态收敛。原 0.5 rad/s 失败仍保留，未改载荷／时长寻优，也未适配半步长的正式评分器。 19 独立主审计（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/fastener/native19_independent_review01/result/report.json`）与差分附录（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/fastener/native19_independent_review01/difference_addendum.json`）分别确认原门和诊断差异。

| 级别／目录 | 基础任务 | 降低难度的方法 | 完成依据 |
|---|---|---|---|
| L1 `parts_sorting` | 3 个大零件，按形状分别入 3 个宽浅盘 | 零件约 60–85 mm；不堆叠，不要求朝向；颜色之外保留形状图标 | 所有零件完全进入正确盘中、受盘承托、松手稳定 |
| L2 `peg_insertion` | 2 根粗销分别放入浅孔 | 粗销直径 32 mm；60 mm 导入口、38 mm 孔；短直孔；初始竖直 | 两销实际到达底座并松手；停在漏斗上方不算完成 |
| L3 `shaft_spacer_assembly` | 在同一轴上依次套隔套、宽环、顶环 | 宽环、大入口倒角、固定竖轴；无螺纹、键槽或卡簧 | 顺序正确，肩台→隔套→宽环→顶环构成实际承托链 |
| L4 `fastener_assembly` | 拾取散装螺栓，对准起牙，接合工具后旋拧并释放 | 独立螺栓／工具供料托、浅被动导向口；无吸附、焊接或螺纹约束 | 建立真实啮合的稳定基线后，手握工具传扭，取得导程一致的额外推进并松手稳定 |
| L5 `gear_train_assembly` | 装 3 个齿轮，自主选择并使用工具完成低速传动检查 | 20/32/20 齿，中间轮更大；向上固定轴、导入口；允许小幅试转和回退；只转短弧 | 三齿轮落座，持握工具驱动输入端，实际啮合带动输出标记到目标区 |

L1 基础版仅三件评分目标，六件目标作为后续独立扩展，避免用重复搬运拉长链路；新版另加的五件非评分散件不改变这一目标数。五任务统一使用单 FR3＋右 SHARPA W01；不能为某一任务单独把基座贴近桌沿。共用布局中的腕端可达区域仅是位置规划依据，具体零件、手型、工具柄、导向口与转动轨迹仍要单独验证。齿轮沿世界 Y 排列，核心装配中心候选为 `(0.40, 0.0)m`；取料和工具分布在近侧与两侧，避免把必要物件放到远桌边。

用户确认后，L4 及 L5 的直柄工具统一采用已做接触动力学验收的 `<开发存储>/assets/task_parts/threaded_sharpa/v3`。工具长 211 mm、柄径 44 mm、可握段长 110 mm、质量约 304 g；18.75 mm 对边刀头配 21 mm 六角孔，单边间隙 1.125 mm。螺丝与底座已经包含 1.5 倍源变换，不再重复缩放。历史预插入版通过 reset 预啮合、预插刀降低难度；当前默认要求从独立供料托完成拾取和起牙。工具仍是受重力的独立刚体，手不与工具绑定。历史实测导程约 3 mm/圈，冻结旋拧目标为基线后的 30°／0.250 mm；旧 45°／0.375 mm 设计保留在历史记录。当前候选仍需新任务原生验证。

L4 当前运行入口为标准 `bench.runtime.service.EpisodeSession` 及 `session.functions.invoke(...)`，使用 `move_wrist`、`set_hand_targets`、`move_joints`、`hand_preset`、`query_ik` 等公共函数。冻结发布的路径／哈希、30°目标、净接触推进信用和释放稳定规则详见[任务说明](tasks/fastener_assembly/README.md)。评分只接受实际螺丝的旋转、导程一致的新增推进和释放稳定；无链微动可恢复，所有回退扣回信用，不能反复转进／退回刷分。62 项 CPU 检查与历史失败轨迹重评分不等于公共函数原生通过。

历史验收覆盖有界外力／扭矩驱动下的旋入、反转、释放、负对照及 480/960 Hz 对比，没有机器人抓握。因此它证明其记录范围内 v3 的螺纹／六角接触可工作，不代表 L4 已验收。当前默认要求完成散装拾取、导向起牙、工具接合、真实手柄持握、接触传扭、短弧推进和释放；预插入版保留为显式历史对照。原 `threaded_tools/v1`、原图及实验记录全部保留。

`threaded_assets.py` 和 `threaded_catalog.py` 保留原 v1 的校验与引用；`threaded_sharpa_catalog.py` 核验 v3 完整来源链、原文件和历史验收哈希，生成共享引用目录 `assets/task_parts/mechanical_workshop/threaded_v2`，不复制或修改模型。v3 保留已修复螺纹、底板 SDF 和七块凸包螺丝头。物理参数优先采用源 USD，缺失项显式沿用历史验收配置：工具／螺丝／底座摩擦 0.6/0.01/0.75、求解器 192/16、接触偏移 0.3 mm；不得用统一默认值覆盖。直柄工具为 CC0，上游零件和 NVIDIA 螺纹内容分别保留授权。静态展示不等于预插 reset 或旋拧证据。

L5 原 `v2` 的三只渐开线直齿轮均为模数 5 mm、20 齿、20° 压力角，节径 100 mm、外径 110 mm、相邻轴距 100 mm；该版及其原生记录继续保留。标准几何关系参考 [KHK 齿轮尺寸技术资料](https://khkgears.net/gear-knowledge/gear-technical-reference/calculation-gear-dimensions/)。宽导入口用于帮助初始对准，最终孔轴配合不能随意放大：候选孔径 24 mm、轴径 23.4 mm、齿侧间隙 0.8 mm，需要共同验证。实体齿面传力、孔轴接触和承托面都保留；遇到顶齿允许慢速小幅转动、抬起重放，不使用吸附对齿或隐式齿轮约束。

新共享 v3_4（开发存储 `assets/task_parts/mechanical_assembly/gear_train_assembly/v3_4/manifest.json`） 保持模数与压力角，改成 **20/32/20 齿、130 mm 轴距**；中间轮外径 **170 mm**，按真实新实体重算质量与惯量，孔、轴及小轮不随之缩放。板、目标、供料和预装 reset 同步，理论中间轮／输入轮转角比为 −0.625，输出／输入为 +1。供料点采用输入 `[.21,-.20]`、惰轮 `[.25,.32]`、输出 `[.21,.10]`，避开工作板并保留机器人可达性；原有 CPU 净空范围不覆盖手、偏心或倾斜工具，也不代替原生传动验证。正常齿侧及套筒配合空程保留，空程本身不列为故障。L5 仍为 `planned`。

L5 基础工具为自由摆放的深套筒曲柄、开口扳手和现有直柄六角螺丝刀。设计假设是宽入口套筒曲柄更容易抓握和驱动，开口扳手也可能完成任务；这一工程判断不写进给 Agent 的目标指令。目标只要求“自行选择工具，从输入端驱动，让输出标记进入目标区域”。评分不检查正确工具 ID，而检查持握、工具与输入轮毂接触、两级啮合连续传动及输出运动；仅选工具、仅摆好三个齿轮、直接推输出轮都不算成功。当前候选规则的输出转角为 30°、允许范围 25–35°、低速上限 15°/s。真实工具抓取、套筒接合与短圆弧仍是主要难点，不因为大零件就预先认定可完成。橡胶锤可作为以后有充分空间时的干扰工具，基础版不要求敲击。

```text
mechanical_assembly/
├── RULES.md                # 五任务中文公开规则、恢复和结果语义
├── scene.py
├── robot_layout.py
├── rules.py                # L1–L3 冻结规则
├── evaluation.py           # L1–L3 裁判
├── referee.py              # 五任务静态公开契约与粗状态
├── task_design.py          # 共用设计约束和冻结候选绑定，无执行器
├── threaded_assets.py      # 原版共享资产引用／校验和静态展示
├── threaded_catalog.py     # 原版USD引用目录、质量／碰撞／参数来源审计
├── threaded_sharpa_catalog.py # 当前v3引用、来源链及历史接触配置
├── threaded_preview.py      # 默认v3静态展示，可显式选择历史v1
├── threaded_reset.py        # 已释放末帧仅作reset种子，运行中不回放
├── dynamics.py             # 当前房间的独立落体、待命与reset验收组合
├── preview.py              # 独立静态机器人组合，无物理步进
└── tasks/
    ├── parts_sorting/definition.py
    ├── peg_insertion/definition.py
    ├── shaft_spacer_assembly/definition.py
    ├── fastener_assembly/
    │   ├── definition.py
    │   ├── task.py          # 任务自有初态与静态描述
    │   ├── rules.py         # 冻结工程候选规则
    │   ├── evaluator.py     # canonical SequenceEvaluator
    │   ├── release.py       # 只准备全新版本
    │   ├── README.md
    │   └── motion_plan.py   # 特权关节脚本候选，不是Agent策略或可运行发布
    └── gear_train_assembly/definition.py
```

L1/L2/L3/L5 的任务目录现也包含 `assets.py` 和 `task.py`，配合家族共用规则／评分及冻结发布工具；L5 另有专用规则和传动评分器。上图保留目录定位用途。新独立冻结候选（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/release_preparation01/preparation_report.json`）明确选择 L1 TGS、L2/L3/L5 PGS，均为 `prepared_unvalidated`、`physics_accepted=false`。构建器通过 `solver_type` 显式冻结选择；省略时仍为原有 TGS，旧发布保持原样。

后续 scene 验证继续闭合未通过的运动／传动约束，并明确各工程候选适用的初态与载荷，再固定任务发布。正式任务操作由共享 bench/agent 控制器通过机器人关节动作推进。用户授权的独立工程诊断可在 reset 设置测试初态，并向真实独立工具施加有界载荷；不能对被动螺丝／齿轮施力或写入轨迹，也不生成任务成绩。两种路径都不使用抓取焊接或隐式螺纹／齿轮运动约束。失败后可重新抓取、回退与纠正，预算耗尽保留未完成进度，API／仿真故障单列。可复用任务资产仍放共享 `assets/task_parts`，冻结发布和实验记录放共享 `datasets`，不将模型复制进任务源码目录。

独立原生诊断入口为 `scripts.validation.validate_mechanical_dynamics`，默认 `--asset-profile sharpa_v3`。它使用同一父房间和机器人布局，检查桌面／地板落体、横放工具和头朝下螺丝的承托、完整待命关节、腕端、机器人小幅关节运动、reset 重复性及固定夹具。工具与螺丝尚未啮合。原版 v1 仅供显式选择复核；该诊断不注册为第六个任务，不作为当前任务目录的交付依据，也不构成 L4 任务验收。

`scripts.validation.validate_mechanical_tool_motion` 单独检查 v3 预插组合与真实机器人关节脚本。`threaded_reset.py` 仅取历史释放后末帧的位姿，平移到当前夹具位置，并以零速度初始化；不回放旧轨迹。脚本记录每个原生子步的接触与物体状态，只有手柄对侧持续产生拇指及另一根手指的正压力，才通过预插状态下的对向接触检查；这不证明自由工具的举起或承重握持。运动命令执行完毕不能替代接触、螺纹推进或任务成功。实际结果与限制见[动力学证据说明](../../../docs/scenes/mechanical_workshop.md)。

历史 `tool_motion04` 运动候选在首次实际闭合失败后，仅将食指 MCP 屈曲目标增加 0.02 rad，并把闭合、保持、释放各延长到 1 s；完整关节计划为 16.2 s，诊断另加 0.5 s 静置。保持机械臂／拇指目标、资产与物理参数不变；源网格的目标重叠仅为关节驱动预载，不能当作允许实际穿透。脚本仍须通过原生对向接触、松开、撤回及返回待命检查，且不包含旋拧阶段。

新的静态组合入口 `scripts.assets.prepare_mechanical_preview` 默认 `--threaded-profile sharpa_v3`，可显式使用 `requested_v1` 复核旧版。原展示文件保持原样；每次准备仍须使用新的输出目录。v3 源文件中停用的旧螺丝头保留停用；原生材料张量可能仍包含它，参数审计必须分别记录启用／停用碰撞体和实际绑定，不能删掉不匹配的张量行来通过检查。
