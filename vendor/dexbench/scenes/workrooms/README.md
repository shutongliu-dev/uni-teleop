# 固定场景与可替换任务

六个场景 ID 保持稳定。每个场景使用 `scene.py` 固定环境路径、版本、哈希和共享坐标，任务放在 `tasks/<task_id>/`。建筑、家具、空桌、灯光和世界相机基准属于场景；任务拥有物体、机器人布局、reset 初态、规则、评分及必要的相机覆盖。切换任务无需复制或重建房间。

| 场景 | 任务目录 |
|---|---|
| 机械加工／装配室 `mechanical_assembly` | L1 零件分类 `parts_sorting`、L2 粗销 `peg_insertion`、L3 隔套装配 `shaft_spacer_assembly`、L4 散装螺栓装配 `fastener_assembly`、L5 多齿轮装配及自主选工具 `gear_train_assembly`；Task6 接头插接 `nist_connector_insertion`、Task7 带轮／皮带装配 `nist_belt_drive_assembly`、Task8 单线布线 `nist_cable_routing`、Task9 多线配对与拆除 `nist_wire_harness`（九项已实现候选；任务操作与独立工程验证分别记录） |
| 实验室 `laboratory` | Task1 样本分类 `sample_sorting`、Task2 试管入架 `vial_racking`、Task3 开盖复盖 `cap_placement`、Task4 颗粒称量 `solid_weighing`、Task5 活塞移液 `pipette_transfer`（五项已实现，物物验证与数采手物操作分开） |
| 厨房 `kitchen` | Task1 早餐配盘端送 `breakfast_serving`、Task2 果蔬分类 `produce_transfer`、Task3 餐具收纳 `utensil_sorting`、Task4 冷灶锅盖 `pot_lid_placement`、Task5 食品调料归位 `pantry_stocking`、Task6 餐后台面整理 `table_clearing`（六项已实现候选） |
| 书房 `study` | Task1 文具分类 `stationery_sorting`、Task2 按序叠书 `book_stacking`、Task3 后架取书 `book_retrieval`、Task4 分类上架 `book_shelving`（四项已实现候选） |
| 棋牌室 `card_room` | **Task0 单臂随机三子摆放 `task0`、L1 三子棋 `tic_tac_toe_3x3`、L2 平面四连棋 `connect_four_5x5`、L3 Hex `hex7x7` 均已实现** |
| 客厅 `living_room` | Task1 遥控电视开机保持10秒、关机后归架 `remote_operation`、Task2 单臂水壶倒水 `water_pouring`、Task3 单臂自由搭塔按高度计分 `tower_building`、Task4 两名玩家各控一台单臂轮流抽塔对抗 `tower_extraction`（四项均已实现候选，Task4 使用双玩家 turn-game 入口） |

共 **32 项任务定义，均为已实现候选**；包括棋牌室四项、机械室九项、实验室五项、客厅四项、书房四项与厨房六项，具体冻结发布以各 `definition.py` 为准。可运行不等于原生操控通过。规划项尚无冻结的可运行任务发布；独立工程诊断不改变 `planned` 状态。默认任务清单排除它们，加载会拒绝；目录数量不代表已完成任务数量。棋牌任务的原生预检和操作范围见[任务说明](../../docs/agent_bench/card_room_games.md)及独立证据，不由客厅规则代码继承。

Task0 是 Task1 的单臂简化练习：沿用 3×3 棋盘和三枚黑子，每回合随机指定三个不同目标格，要求实际放入并松手稳定 300 ms，不要求连线或对弈。普通 `/episodes` 和统一公共函数执行；种子、目标与冻结发布可复现。见[Task0 规则和入口](card_room/tasks/task0/README.md)。

厨房当前固定环境为 `kitchen_galley_20261019_v2`，用 0.90 m 高的灶台与 2.8 × 0.8 m 吧台替代通用方桌，中央通道宽 1.12 m。机器人在独立展示层居中组合；场景定义分别声明吧台、灶台和备菜面。六个厨房任务已有独立动态资产、冻结初态、公开规则与原生裁判，通过普通 `/episodes` 使用原单臂机器人。Task1/2/3/5/6朝吧台、第四项朝冷灶；合理容差与恢复语义见[厨房任务](kitchen/README.md)。场景支撑证据与完整机器人操作资格分别记录。高搁板、推车、茶具、粮食罐、沥水架和扫描厨具食材丰富陈设，吧台中央保留 1.2 × 0.6 m 空任务区。详见[厨房布局与资源](../../docs/scenes/kitchen.md)。

书房固定环境为 `study_library_20261019_v7`：2.00 × 0.90 m 胡桃木行政大班台，80 mm 厚台面、双落地柜、六只抽屉和整幅访客挡板，齐平皮革嵌面配缝线和黄铜细线，桌面保持清空；后方开放书架留出四个操作书格。窗帘、原创植物画、档案盒和带中文书名的精装书补充房间细节，阅读椅、边桌、台灯、植物与 PBR 材质复用共享资产。四项书房任务拥有独立物件、冻结规则和逐子步接触裁判，通过普通 `/episodes` 接入；当前 Task1 文具分类为 v4（分类框移近机器人），Task2–4 为 v3，各项原生 300 ms hold 与裁判重放通过。房间图片、原生初始化与完整操控资格分别记录；历史 v6 房间及其 v2 发布保留。见[书房设计](study/README.md)与[环境证据](../../docs/scenes/study_room.md)。

客厅固定环境为 `living_room_residential_20261018_v7`：房间缩为 4.8 × 4.5 m，原 1 × 1 m 空任务桌及共享坐标不变；保留中性日间照明和实体灯槽，窗外改为私人阳台、对面住宅楼与庭院几何，背景采用纯天空 HDR。家族根目录的 `dual_robot_layout.py` / `dual_preview.py` 单独组合两台原尺寸 FR3 + 右 SHARPA，基座 `(0.65,±0.76,0.78)m`、偏航 ±90°，支架底脚距桌边 8 cm；机器人不写入固定房间资产。完整源码冻结的 `cpu03` 验证了 9 个双方分别可达的腕采样点与 40 组分开的对置腕目标，仅属摆放／端点 IK 证据，不代表抓取、无碰撞路径或任务资格。最终双视角 1920 × 1440 原生图片、固定资产哈希和复现命令见[客厅外观证据](../../docs/scenes/living_room.md)；v6 单臂历史保留，当前木条使用明确固定尺寸的候选，不视作用户最终尺寸；Task3 单臂搭塔与 Task4 双臂对抗抽塔已有原生任务层和冻结裁判，分别走 `/episodes` 与认证 `/turn-games`；配置和规则见[客厅任务](living_room/README.md)，Task1 遥控器和 Task2 水壶倒水已冻结规则、裁判和原生观察器，通过普通 `/episodes` 接入，完整机器人策略另行验证。

机械加工室九项设计共用固定空桌环境 `mechanical_workshop_20261017_v3` 和 `mechanical_single_arm_v1` 单臂布局，基座为 `(-0.23,0,0.78)m`、支架底脚距桌边 20 cm。当前工具选用用户确认的 SHARPA v3，静态组合默认 v3，原 v1 展示和记录保留。场景动力学与 L4 关节脚本具有独立验收入口，静态展示、CPU IK 和这些诊断均不等同于完整任务验收。L4 当前默认为散装螺栓 `loose_bolt_candidate_v10_pgs`：拾取独立螺栓并起牙，再拾取并插入独立工具，确立稳定接合后旋入约 30°／0.250 mm 并稳定松手；v10 仅把场景求解器由 TGS 改为 PGS，旧预插入版及 loose v7–v9 证据保留。目标、标准公共函数入口与评分边界见[可运行候选说明](mechanical_assembly/tasks/fastener_assembly/README.md)。设计见[机械家族任务说明](mechanical_assembly/README.md)，构建与证据见[机械加工室](../../docs/scenes/mechanical_workshop.md)。

新增 Task6–9 是去重后的 NIST ATB1–4 适配，走普通 `/episodes` 和共享控制器。Agent 使用 `--family mechanical_assembly --task task6`～`task9`，也接受 `L6`～`L9` 或上述 canonical ID；服务需绑定对应目录冻结发布。Task6/7/8/9 当前 release 分别为 v5/v2/v3/v3，原五项发布不变，ATB5 不另设 Task10。柔性物体采用声明的被动刚体链近似；原生初始化和物物可行性已分别记录，手物操作属于数采范围。详见 [NIST 任务](mechanical_assembly/NIST_TASKS.md)。

```text
card_room/
├── scene.py
├── game_task.py          # 共享任务组合
├── game_assets.py        # 共享棋具生成
├── reserve_assets.py     # 任务所属供棋托盘
├── robot_layout.py       # 对称待命和固定腕镜头
└── tasks/
    ├── task0/           # 单臂三子放置，独立规则、评分与冻结发布
    ├── tic_tac_toe_3x3/  # definition.py + task.py
    ├── connect_four_5x5/ # definition.py + task.py
    └── hex7x7/           # definition.py + task.py；assets.py 保留历史配方
```

`catalog.list_scenes()` 列出固定场景；`catalog.list_tasks()` 默认只列已实现任务，`include_planned=True` 可查看规划。已实现任务的 `release_schema` 显式固定发布类型，未知或不匹配的类型会被拒绝。`catalog.load_task(family, task_id)` 核验冻结 release、文件及资产哈希，并要求父环境与所选场景完全一致；棋牌任务复用运行入口的完整契约校验，包括棋子库存、棋盘几何、玩家归属、机器人初态和 evaluator 源码；机械任务同时核验 canonical loader 的资产目录、依赖清单和冻结评分器。加载不启动 Isaac 或模型，也不授予原生物理资格。Python 定义自动纳入完整源码快照，wheel 同时包含这些命名空间目录。

```bash
.venv/bin/python -m scenes.workrooms scenes
.venv/bin/python -m scenes.workrooms tasks --family card_room
.venv/bin/python -m scenes.workrooms tasks --include-planned
.venv/bin/python -m scenes.workrooms task --family card_room --task hex7x7
```

旧 `list/export/validate` 命令和 `registry.py` 保留 `rooms_20261017_v3` 六房间落体／相机诊断，不作为上述操控任务。历史 `scenes/rooms` 与 `scenes/catalog.json` 也保留原范围与发布。三级几何、规则、协议和物理落子判定使用 `scenes/board_games/tabletop`，旧 `hex` evaluator 保留冻结契约；房间 USD 配方位于 `bench/simulation/assets/workrooms/card_room.py`。

可复用资源存放在 `<开发存储>/assets`，冻结任务与实验在 `<开发存储>/datasets`，展示媒体在 `<开发存储>/demo/interface/scenes`。环境、初态、物理或评分变化均使用新 release；目录元数据不回写旧发布的准备状态或验收结果。
