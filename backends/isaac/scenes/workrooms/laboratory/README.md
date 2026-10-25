# 实验室双台面任务

固定房间为 `laboratory_island_20261020_v3`。五项任务共用原 FR3、右 SHARPA 和过道中的原支架，底座为 `(0.70, 1.2875, 0.78) m`。岛台和后台均高 `0.90 m`；任务道具位于预留操作区靠近过道的一侧。固定环境内的显微镜、离心机和仪器仍为陈设，任务自带功能器具。

2026-10-21 桌面细节版本为五个任务加入相同的 17 组、50 件背景物品：记录本、文件夹、笔、护目镜、擦拭纸盒、试剂瓶、备用试管架、冷冻盒、标签卷、培养皿、清洗瓶、备用盖和耗材包装等。它们属于任务层的固定陈设，带独立静态碰撞，不是可抓取目标或有效计分支撑。岛台两端与服务侧放耗材，中央操作通道、后台目标和机器人过道保持空净；原固定房间不重建。

任务器具增加标称质量标签、瓶盖防滑纹、瓶身分类与编号、架孔标记、金属/塑料表面、移液器行程及按内径计算的杯体容量刻度。原任务物体碰撞几何、质量、惯量、关节、初态和评分阈值保留。公开 `laboratory_rules.v2` 的 `object_properties` 列出可操作物品的标称质量；Task4 台秤增加随实际稳定杯内库存更新的净重屏，数值通过相机可见，见 [Task4](tasks/solid_weighing/README.md)。

当前资产版本为 Task1/4/5 v2、Task2/3 v3；对应发布为 Task1 v3、Task2/3/4 v4、Task5 v2。仍使用普通 `/episodes` 与 Agent 的 `--family laboratory --task task1`–`task5`。旧发布与物物操作证据均保留，不能把外观升级或接口微动检查当成完整机器人操作。新版证据位于 `<开发存储>/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/`，原生近景位于 `<开发存储>/demo/interface/scenes/laboratory_desktop_20261021/`。

当前五项发布均通过真实 TCP Bench/Agent 联调：四路 RGB-D、只读 IK、关节微动、回程、保持及动作重放，每项 0.92 s / 1840 原生子步，均取消封存且物理有效。Agent 的创建后/结束前查询保留完整标称质量表。对应独立复核为 `native_review01`（Task1/3/4/5）与 `native_review02`（最终 Task2）；初始 Task2 v3 的记录保留。`balance03` 三个独立初态各运行 0.6 s / 1200 子步，真实观察器分别驱动 `---`、`00.0`、`06.0`，诊断 RTX 近景的逐段像素/深度与原生状态相符。该额外近景用于显示验证，不替代公开四路相机的分辨率与视角限制。新版本已确认初始化和接口；目录没有把旧版本的完整物物操作资格自动迁移过来。

| 任务 | 难度 | 岛台 | 后台 | 物物功能 |
|---|---|---|---|---|
| Task1 `sample_sorting` | 简单 | 4 个密封样本瓶、两色分类托盘 | 空 | 瓶与托盘接触、放稳 |
| Task2 `vial_racking` | 中等 | 3 支横放试管与浅供料架 | 三孔试管架 | 管体穿过真实空孔并落底，一孔一支，不绑定隐藏身份 |
| Task3 `cap_placement` | 中高 | 已闭盖的方形样本瓶、归位区 | 防转瓶座、盖托 | 独立瓶盖沿粗螺旋导槽旋出，放盖、复盖、归还瓶子 |
| Task4 `solid_weighing` | 高 | 24 颗独立颗粒、药匙、称量舟 | 称量平台 | 实际铲取、携带、倾倒；目标 `6±0.6 g`，累计洒落 `≤1 g` |
| Task5 `pipette_transfer` | 高 | 80 mL 源杯、活塞式移液器、工具架 | 两个接收杯 | 实际活塞行程驱动吸排，各 `10±2 mL`，累计洒漏 `≤2 mL` |

验收范围由用户明确为**物体之间交互可行性**。工程探针允许对被操作的器具施加明确记录的工具边界；被动对象的运动来自接触、重力或公开说明的有限体积模型。手物交互、抓握策略和机器人完整任务执行由数采后续处理。放置后的支撑、机构功能、独立物料转移，与机器人任务成功分别记录。

代码结构与其他房间一致：`scene.py` 固定环境；`tasks/<task_id>/definition.py` 元数据；同目录 `assets.py`、`task.py`、`release.py` 为任务入口；公共组合、规则、裁判、发布逻辑在本家族根目录。共享资产在 `<开发存储>/assets/task_parts/laboratory/`，冻结发布在 `<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/`，实验记录在 `development/laboratory_tasks_20261020/`。资产和历史失败记录不覆盖。

规则详见 [RULES.md](RULES.md)。发布清单中的 `prepared_unvalidated`、`physics_accepted=false` 是不可变的准备时记录，后续证据由任务目录独立引用。`manipulation=false` 保留，不将台架结果写成机器人成功。

Task4 使用已知单颗质量计算杯内净样本量，未模拟标定电子天平。Task5 使用原生动态活塞、实际吸嘴/杯口位姿和守恒有限体积模型，包含空吸、压杆排水、重力飞行、杯口捕获、容量和洒漏；未求解粘性、飞溅或液体对器具的重量与射流反作用。它是适合本轮任务设计的实用近似。

2026-10-20 首轮发布为 Task1/2 v2、Task3/4 v3、Task5 v1；资产为 Task3 v2，其余 v1。首轮每项原生整屋检查均通过 1.2 秒 / 2400 子步、四路 RGB-D、实际支撑和不误计初态得分。首轮原生工程称量实现 12 颗 / 6.0 g 入杯，终态器具外颗粒为零；移液为 10.035 mL 和 9.923 mL，洒漏约零。

Task2 的额外自由投落试验确认三支管能穿孔并落底，但冲击后短窗仍有角向摇晃，未通过同次投落后的稳定判定。此前预置入架的实际支撑与稳定证据独立通过；两者不合并成完整投落成功。数采应控制落座，完整手物操作另行验证。Task4 原先额外要求所有球的瞬时总速度低于 0.02 m/s 的探针失败保留；按本任务稳定质量规则对全量原始轨迹的独立复核通过，没有改旧记录。

统一证据索引（开发存储 `datasets/fr3_sharpa_bench/development/laboratory_tasks_20261020/qualification.json`） · Task1 图（开发存储 `demo/interface/scenes/laboratory_tasks_20261020/sample_sorting/task.png`） · Task2 图（开发存储 `demo/interface/scenes/laboratory_tasks_20261020/vial_racking/task.png`） · Task3 图（开发存储 `demo/interface/scenes/laboratory_tasks_20261020/cap_placement/task.png`） · Task4 图（开发存储 `demo/interface/scenes/laboratory_tasks_20261020/solid_weighing/task.png`） · Task5 图（开发存储 `demo/interface/scenes/laboratory_tasks_20261020/pipette_transfer/task.png`）
