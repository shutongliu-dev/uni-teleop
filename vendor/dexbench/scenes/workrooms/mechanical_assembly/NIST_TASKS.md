# NIST 去重改编任务

用户确认按 Task Board 分任务，同时要求不重复已有能力。因此在原 Task1–5 后增加四项：ATB1–4 各保留一项不同的任务，ATB5 不再单独增加。它们是根据原始资料制作的适用性改编，不是完整官方板件、官方评测协议或认证。

| 编号／目录 | 来源与保留操作 | 去重边界 |
|---|---|---|
| Task6 `nist_connector_insertion` | ATB1：USB、BNC 分别插入匹配接口 | 不重复 Task2 插销、Task4 螺母、Task5 齿轮；只保留不同接口的方向与配合 |
| Task7 `nist_belt_drive_assembly` | ATB2：带轮、闭环皮带与可拆张紧轮装配 | 新能力是柔性闭环布置及带轮接触；省去轴环、螺纹紧固 |
| Task8 `nist_cable_routing` | ATB3：单根电缆依次穿过导向区并接入末端接口 | 重点是整根线缆路径，不能只摆对末端 |
| Task9 `nist_wire_harness` | ATB4：两根彩色导线配对、共同导向成束，再整体取出归位 | 新增多线配对和装后拆除的先后关系；不再复制单线任务 |
| 不新增 | ATB5：障碍附近的螺栓／螺母装配 | 与现有 Task4 紧固能力重复，保留来源资料供以后有明确新目标时使用 |

## 目录与接入

每项都使用 `tasks/<task_id>/definition.py`、`task.py`、`rules.py`、`evaluator.py`、`README.md`。家族根目录的 `nist_assets.py`、`nist_composition.py`、`nist_release.py`、`nist_rules.py`、`nist_native_evidence.py` 和 `nist_evaluation.py` 提供共用构建与裁判逻辑。

继续使用固定房间 `mechanical_workshop_20261017_v3`、原 FR3／右 SHARPA 和 `mechanical_single_arm_v1` 支架布局。任务自有零件与工装放在原工作桌上；原 Task1–5 的发布指针与规则不因这些新增任务改变。

通过普通 `/episodes` 接入，Agent 使用 `--family mechanical_assembly --task task6`～`task9`，也可用对应 canonical ID。种子记录回合身份，当前物理初态固定；不会按种子改变原始几何。公开端只返回规则和粗粒度阶段，私有裁判读取原生刚体、接触与被动链条状态。

## 实用规则与裁判

- 允许重新抓取、退出重插、错位纠正和重新布线；一般未完成状态可以继续。
- 必须满足实际目标、配合接触、支撑及释放后连续 300 ms 稳定。初始化和工程探针的预摆目标状态均不计任务分。
- Task6 不要求电气导通或 BNC 锁扣功能。BNC 使用原尺寸接口的部分插入目标；不以扩大孔洞解决源模型配合问题。
- Task7 允许正常松弛，不要求精确张力或传动效率；皮带必须与带轮实际配合，单纯环绕并躺在底板上不应计作装配。
- Task8 检查整条中心线按顺序通过导向区，末端正确但绕过路径不合格。
- Task9 必须先确认颜色配对、成束与释放稳定，然后才能取得拆除归位成绩；从初始供料区直接放进收集盘不能成功。
- 预算耗尽记未完成；掉落地面与明显物理异常分别记录；证据缺失、不连续或无效不能生成成功。

## 资产与物理范围

原始资料来自 [NIST Assembly Performance Metrics and Test Methods](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly)，共享源快照位于 `<开发存储>/assets/nist/assembly_task_boards/20261020_source_snapshot/`。`inventory_20261020_complete.json` 记录原文件、来源、单位及完整清单。原 STL 按 mm→m 转换；保留源尺寸、原文件与归属说明，不能将第三方 CAD 统一宣称为无版权。

任务资产使用 `<开发存储>/assets/task_parts/mechanical_assembly/<task_id>/<version>/`。每版含 USD、metadata、manifest、结构审计和来源说明。源板没有提供的电缆、导线、皮带链节及辅助工装明确标记为程序化近似。柔性件由具有实际重力、碰撞与被动关节的短刚体连接，不声称材料弯曲、摩擦或张力的定量标定。

独立发布位于 `<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/<task_id>/<version>/`。准备期始终保持 `prepared_unvalidated`、`physics_accepted=false`、`task_policy_success_claimed=false`；后续证据独立记录，不改写冻结清单。

## 验证入口

完整源码快照和原生证据集中在 `<开发存储>/datasets/fr3_sharpa_bench/development/mechanical_nist_20261020/`。

- `scripts.validation.mechanical_nist_release`：普通 EpisodeSession、原机器人保持、四路 RGB-D、真实任务零件支撑、私有观测连续性及初始化零得分。
- `scripts.validation.mechanical_nist_interactions`：同尺寸桌面上的物物工程探针。目标姿态只在 reset 前设置，此后由重力、接触及被动关节演化，保留逐子步状态和有符号接触流；不计机器人任务成绩。
- CPU 检查覆盖规则、负例、发布哈希、任务隔离、别名／种子及普通 HTTP 接口。CPU 通过不替代原生物理证据。

原生失败和后续修复版本均保留。此轮负责场景与物物可行性，手物交互交由数采；不运行模型策略、不调整共享控制器，也不宣称完整机器人装配成功。

## 当前发布与验证结果

最终证据索引（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_nist_20261020/qualification.json`） 绑定下列版本、源码、原生报告、独立复核及图片哈希。四项正式初态都通过 1.2 s／2400 原生子步保持、四路 RGB-D、实际承托、连续裁判读回及零初始化得分，worker 均退出 0。

| 任务 | 资产／发布 | 冻结源码 | 物物工程探针 |
|---|---|---|---|
| Task6 | v3／v5 | source05 | `task6/goal03`：USB 与 BNC 实际接口接触、配合及末 300 ms 静稳通过 |
| Task7 | v5／v2 | source05 | `task7/goal01`：三轮就座、实际槽内皮带、两轮接触、被动连接及静稳通过 |
| Task8 | v5／v3 | source06 | `task8/goal02`：整线导向、实际插针配合点、摩擦参与的承托及静稳通过 |
| Task9 | v4／v3 | source06 | `task9/goal02`：两线端子配对、共同导向、四处出线槽及被动连接静稳通过 |

所有目标工程探针各记录 1 s／2000 原生子步，末窗为 601 个连续端点；仅在 reset 前设置目标姿态，不施加后续位姿或外力、不计装配操作成绩。Task9 另有收集盘静态几何见证（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_nist_source_review_20261020/task9_v4_collection_geometry/collection_fit.json`）：两根完整 410 mm 导线和四个源壳体均能放入，最小盘内平面余量约 2.37 mm；这不等于实际拆除、收集或抓取轨迹通过。

最终相关 CPU 检查为 `cpu_checks06.xml` 的 131 项和 `cpu_core06.xml` 的 70 项通过，均无跳过，两组包含重复资产检查，不相加为独立用例数。前一轮 276 项中的 11 个 HTTP 用例在工作树继续合入代码时触发服务的源码变更保护；原失败保留，固定源码后这 11 项已全部复测通过，没有关闭保护。

保留的修复记录包括薄凸包台板首步错误接触、探针 API／浮点时钟检查、Task8 外壳原点随倾斜下垂的误判，以及 Task9 完整端壁阻挡导线。新版本使用同尺寸解析碰撞体、插针上的评分点和真实出线缺口；原始零件尺寸与共享控制器不变。

力学范围仍有限：Task6/7 的 v1 探针记录法向接触，Task8/9 的 v2 另记录原始摩擦和有符号合力。Task8 末窗合力竖直分量与自重之比约 1.387，原始角速度也保留为诊断，不能称定量力标定通过；公开实用规则根据释放后的位姿稳定判定。柔性链、承托与配合可用性和精确材料动力学分别表述。

实际原生图片（未经生成式修改）：Task6（开发存储 `demo/interface/scenes/mechanical_nist_20261020/native01/task6/task.png`）、Task7（开发存储 `demo/interface/scenes/mechanical_nist_20261020/native01/task7/task.png`）、Task8（开发存储 `demo/interface/scenes/mechanical_nist_20261020/native01/task8/task.png`）、Task9（开发存储 `demo/interface/scenes/mechanical_nist_20261020/native01/task9/task.png`）。图片展示正式供料初态与保持，不冒充机器人已完成装配。
