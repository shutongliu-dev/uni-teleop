# 场景任务采集兼容性验证（2026-11-05）

## 结果概览

32 个任务全部完成配置检查；24 个完成原生仿真、状态落盘、独立进程恢复和三路图像重渲染；8 个不兼容当前配方。已完成的原生测试没有运行失败或超时。**24 项是技术链路通过，不代表都能直接开始人工数采。**

| 场景 | 技术链路通过 | 当前不兼容 |
|---|---:|---:|
| 机械装配 | 9 | 0 |
| 实验室 | 3 | 2 |
| 厨房 | 6 | 0 |
| 书房 | 4 | 0 |
| 棋牌室 | 1 | 3 |
| 客厅 | 1 | 3 |

## 头显视野复核

- `kitchen/pot_lid_placement`：主要预览朝向空吧台，实际锅具位于灶台；原采集视角可见锅具。需要根据任务工作区定位头显相机。
- `card_room/task0`：主要预览可见棋盘，但初始棋子位于画面外的侧方支架；原采集视角可见支架与棋子。需要覆盖取棋子区域。
- 实验室三个可运行任务：物体较小、靠近主画面下沿；腕部预览初始朝向不能保证覆盖操作目标。建议单独调近景。
- 部分机械装配、书房任务：操作区偏画面右侧，部分物体或后续目标区域可能超出视野，需人工操作时复核。
- 本次只检查初始姿态图像，未验证整条运动轨迹的无遮挡和可视范围。

## 当前不兼容项

- **需要任务观察器／状态驱动显示：**实验室 `solid_weighing`、`pipette_transfer`，客厅 `remote_operation`。当前关闭观察器的练习／补渲染配置被显式拒绝；不能简单关掉限制，否则可能丢失称量、液量或显示状态。
- **粒子物理：**客厅 `water_pouring`。当前 CPU 刚体与状态重放格式不支持粒子。
- **多机械臂映射：**棋牌室 `tic_tac_toe_3x3`、`connect_four_5x5`、`hex7x7`，客厅 `tower_extraction`。当前单只 MANUS 右手配方不能直接覆盖多臂机器人；未自动改用其他配方。

## 验证范围与逐项证据
验证配置：PICO 控臂＋MANUS 右手映射；CPU rigid；物理步 4 ms、控制周期 40 ms；工作台练习模式；独立头显后上方＋腕部预览，头显隐藏机械臂；采集状态后独立进程重渲染。

范围：全量 32 个任务做资源／配置／机器人映射预检，兼容项启动原生 Isaac。每项 2 个保持关节目标的控制周期（80 ms），保存初始、中间、终止共 3 帧；再由独立 Isaac 进程恢复，输出左右主视角和腕部无损 PNG。检查恢复状态一致、时间戳关联、图像非空，以及重渲染推进物理步数为 0。

没有接入真实 PICO／MANUS 或踏板，不测试头显网络时延、主动抓取、长时间稳定性或任务成功；任务评分关闭。预检失败表示不兼容当前配方，并不表示任务在其原生配置下不可运行。

统一启动器 CPU 测试：49 项通过。

| 场景 | 任务 | 配置预检 | 原生存状态 | 独立重渲染 | 说明 |
|---|---|---|---|---|---|
| mechanical_assembly | parts_sorting | passed | passed | passed |  |
| mechanical_assembly | peg_insertion | passed | passed | passed |  |
| mechanical_assembly | shaft_spacer_assembly | passed | passed | passed |  |
| mechanical_assembly | fastener_assembly | passed | passed | passed |  |
| mechanical_assembly | gear_train_assembly | passed | passed | passed |  |
| mechanical_assembly | nist_connector_insertion | passed | passed | passed |  |
| mechanical_assembly | nist_belt_drive_assembly | passed | passed | passed |  |
| mechanical_assembly | nist_cable_routing | passed | passed | passed |  |
| mechanical_assembly | nist_wire_harness | passed | passed | passed |  |
| laboratory | sample_sorting | passed | passed | passed |  |
| laboratory | vial_racking | passed | passed | passed |  |
| laboratory | cap_placement | passed | passed | passed |  |
| laboratory | solid_weighing | failed | 未运行 | 未运行 | ValueError: Task observation is required for particles, reduced dynamics or task-driven visuals |
| laboratory | pipette_transfer | failed | 未运行 | 未运行 | ValueError: Task observation is required for particles, reduced dynamics or task-driven visuals |
| kitchen | breakfast_serving | passed | passed | passed |  |
| kitchen | produce_transfer | passed | passed | passed |  |
| kitchen | utensil_sorting | passed | passed | passed |  |
| kitchen | pot_lid_placement | passed | passed | passed |  |
| kitchen | pantry_stocking | passed | passed | passed |  |
| kitchen | table_clearing | passed | passed | passed |  |
| study | stationery_sorting | passed | passed | passed |  |
| study | book_stacking | passed | passed | passed |  |
| study | book_retrieval | passed | passed | passed |  |
| study | book_shelving | passed | passed | passed |  |
| card_room | task0 | passed | passed | passed |  |
| card_room | tic_tac_toe_3x3 | failed | 未运行 | 未运行 | ValueError: Current MANUS right-hand recipe requires a single right-hand robot; multi-arm needs explicit mapping |
| card_room | connect_four_5x5 | failed | 未运行 | 未运行 | ValueError: Current MANUS right-hand recipe requires a single right-hand robot; multi-arm needs explicit mapping |
| card_room | hex7x7 | failed | 未运行 | 未运行 | ValueError: Current MANUS right-hand recipe requires a single right-hand robot; multi-arm needs explicit mapping |
| living_room | remote_operation | failed | 未运行 | 未运行 | ValueError: Task observation is required for particles, reduced dynamics or task-driven visuals |
| living_room | water_pouring | failed | 未运行 | 未运行 | ValueError: CPU rigid profile cannot simulate particles, cloth or soft bodies |
| living_room | tower_building | passed | passed | passed |  |
| living_room | tower_extraction | failed | 未运行 | 未运行 | ValueError: Current MANUS right-hand recipe requires a single right-hand robot; multi-arm needs explicit mapping |

统计：{'passed': 24, 'incompatible': 8}

详细日志、配置、测试状态与 PNG 位于本机忽略目录 `local/validation/task-matrix/`；`results.json` 为机器可读结果，`audit.py`、`native.py`、`run-native.py` 为本次诊断脚本。这些测试数据不属于用户采集 episode。

## 复查入口

- 本机机器可读结果：`local/validation/task-matrix/checked-results.json`。
- 每个 `场景__任务/` 目录包含冻结输入、阶段日志、回执、头显截图、3 帧状态和补渲染图像。
- 头显拼图：`headset-contact-sheet-1.jpg` 到 `headset-contact-sheet-4.jpg`；空格为未执行原生测试的不兼容项，并非黑屏渲染。
- 本次共 24×3=72 帧状态、216 张原始采集相机 PNG，另有 review 合成图和头显截图；测试目录约 65 MB。
- 所有原生测试进程已退出，未启用真实设备或实体电机。

## 建议后续顺序

先修复锅盖与取棋子的主视角目标区，再为实验室补近景机位；随后对目标任务做真实 PICO＋MANUS 张手、握拳、小幅位移与一条完整 episode 的采集／补渲染验收。多臂、粒子和任务动态显示属于独立能力扩展，需要补齐映射或状态格式后重新验证。

## 2026-11-06 头显机位调整复核

已为工作台练习模式的 18 个任务调整独立头显机位：机械装配 9、实验室 3、书房 4、厨房锅盖 1、单臂三子摆放 1。锅盖改为灶台目标区；棋盘兼顾取子架；实验室拉近样本区；机械装配和书房桌面视角居中留余量；取书／上架兼顾书架与桌面。其余任务保留原机位。

53 项 CPU 测试通过（含双眼视锥覆盖关键工作区、不修改输入场景和默认回退）。18 个任务在原生 Isaac 中生成预览并通过采集相机不变检查：左右主采集相机与腕部采集相机的镜头参数完全一致，世界变换与上轮存档在 2e-5 容差内一致。腕部小窗仍使用原有随手运动视角，未修改它的安装位置。

这些结果覆盖 seed=0 初始姿态，不代替完整遥操轨迹的遮挡／可达性验收。未启用 PICO 或 MANUS 读取，未启动人工采集；所有测试进程已退出。未修改任务资产、碰撞、物理配置或原有采集数据。

本机证据：`local/validation/camera-adjustment/results.json`、各任务 `headset.png` 与 `native.log`；调整前后对照图：`local/validation/camera-adjustment/before-after.jpg`。本次更改不改变此前 24 项技术链路通过、8 项不兼容的统计。
