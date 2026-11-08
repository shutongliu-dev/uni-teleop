# NIST 去重改编任务

用户确认按 Task Board 分任务，同时要求不重复已有能力。因此在原 Task1–5 后增加四项：ATB1–4 各保留一项不同的任务，ATB5 不再单独增加。它们是根据原始资料制作的适用性改编，不是完整官方板件、官方评测协议或认证。

| 编号／目录 | 来源与保留操作 | 去重边界 |
|---|---|---|
| Task6 `nist_connector_insertion` | ATB1：USB、BNC 分别插入匹配接口 | 不重复 Task2 插销、Task4 螺母、Task5 齿轮；只保留不同接口的方向与配合 |
| Task7 `nist_belt_drive_assembly` | ATB2：带轮、闭环皮带与可拆张紧轮装配 | 新能力是柔性闭环布置及带轮接触；省去轴环、螺纹紧固 |
| Task8 `nist_cable_routing` | ATB3：单根电缆依次穿过导向区并接入末端接口 | 重点是整根线缆路径，不能只摆对末端 |
| Task9 `nist_wire_harness` | ATB4：两根彩色导线配对、共同导向成束，再整体取出归位 | 新增多线配对和装后拆除的先后关系；不再复制单线任务 |
| 不新增 | ATB5：障碍附近的螺栓／螺母装配 | 与现有 Task4 紧固能力重复，保留来源资料供以后有明确新目标时使用 |

## 当前尺寸与可操作性修正（2026-10-23）

当前四项选择显式 `nist_manipulation_scale_v1`，保留原 CAD／旧发布及来源授权；物理几何以米烘焙，按原声明密度重算质量、重心和惯量。固定房间、桌面、原机器人和原数值稳定阈值不变。每项仍可在开局选单臂、同侧左右双臂、对侧左右双臂，由同一 Agent 控制。

| 任务 | 当前资产 | 实际修正 |
|---|---|---|
| Task6 | `usability_v1` | USB／BNC 公母成对放大 1.6 倍，配套底座和目标偏移同步；BNC 仍只检查槽对齐的部分插入 |
| Task7 | `usability_v3` | 张紧轮增加 24 mm 直径抓持轮毂，延长导轴；12 mm 孔、14 mm 入口及下外缘导角；原两带轮和 675 mm 闭环带保持原尺寸 |
| Task8 | `usability_v2` | 端头放大 1.6 倍，线径 5.25 mm，导向管放大 1.25 倍；完整插头穿管径向余量约 2.20 mm |
| Task9 | `usability_v2` | 端头放大 1.6 倍，线径 4.5 mm，出线槽 9 mm，收集盘 380×140×30 mm；补齐与端子底部等高的连续开放支承条 |

Task8 真实孔径变大后，路线几何门从 9.5 mm 调整为 11.25 mm，约等于孔半径减去线半径再加原有实用余量；这属于目标几何的适配，稳定时间、速度和接触阈值没有放宽。Task9 的支承条解决中段悬空弧垂和滚动，不增加顶盖、卡扣或绑定；64 个动态体、62 个关节与原线长均保留。

Task7 建议先装两只带轮和皮带，再沿轴装入张紧轮，裁判不强制动作顺序。原平面整圈同时跨过三轮外缘需要约 684.76 mm，超过 675 mm 带长；拆去张紧轮后有保持链节原长的双轮套带几何见证。该结果不排除其他倾斜或逐轮操作路径。最终原生导入检查从已捕获导轴、尚未入座的上方 10 mm 释放，真实下降约 10 mm，三轮就座、皮带配合与末 300 ms 释放稳定通过，随后无位姿写入或外力。完全脱轴的 50 mm 释放虽能落座，但整条皮带稳定检查未通过；延长等待还触发异常接触安全终止，这些记录均保留，不能称任意落放都可靠。

Task6／8／9 各完成 1.2 s／2400 原生子步的实际目标配合、承托与末 300 ms 稳定检查；Task7 导入检查完成 1 s／2000 子步。Task9 两根完整导线及端头入盘的静态几何见证通过，最小盘边余量 5.10 mm。工程探针的目标姿态只在 reset 前设置，不计机器人装配或拆除成绩。

四项三构型共 12 组、81 个离散指尖接近样本全部通过；另检查 141 个保存姿态的非相邻 FR3 臂凸包。实际发布中的机器人、reset、台架网格和场景资产与检查记录逐一绑定。这些检查不证明连续路径、抓取或完整任务成功。真实 HTTP／Agent 接口证据按具体构型和冻结源码单独记录，不推广到全部构型。

本轮真实 HTTP／Agent 检查：Task6 单臂、Task7 对侧双臂、Task8 同侧双臂、Task9 单臂，各完成微动、返回和 0.32 s 保持，总计约 1 s／2000 原生子步，均为 0 模型调用。前三项使用 `root/source_runtime01`，Task9 成功重跑使用合入项目最新接口代码的 `root/source_runtime03`；每次的确切源码、发布和进程退出记录均绑定在证据里。Task9 第一次在 0.56 s 遇到固定 120 s 原生 RPC 超时、worker 被关闭，保留为基础设施失败；换用空闲 GPU 后完整重跑通过，未改变物理或数值规则。相关最终 CPU 回归为 `root/regression03.xml`，374 项通过、0 失败／跳过。

当前发布位于 `datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/`：Task6 为 batch01，Task7／8 为 batch02，Task9 为 batch03；双臂对应 dual_batch01／02／03。旧发布保持不可覆盖。原生与 CPU 记录见 本轮证据索引（开发存储 `datasets/fr3_sharpa_bench/development/nist_usability_20261023/qualification.json`），图片见 三构型图集（开发存储 `demo/interface/scenes/task_configurations_20261023/index.html`）。

## Task8／9 公开信息修复（2026-10-21）

该历史修复的 Task8／9 选择 `desktop_v2`，基础规则发布均为 v4，原资产分别保留 v5／v4。Task8 明确从黑色方块端沿线朝音频插头端依次通过绿色、黄色（琥珀色）导向管；规定最终路线，操作先后可自由调整。Task9 同色线的两端等价，允许整根掉头，两个端头分别占用两个同色端子，并共同经过导向区域；必须先释放确认装配，再拆除收集。Task6／7 的合同和发布不变。以下旧版本与原生物物记录保留各自历史范围；本轮 公开规则／裁判／接入证据（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_routing_public_20261021/qualification.json`） 单独记录。

本轮 135 项相关 CPU 回归通过；两项新发布各通过 0.8 s／1600 原生子步的真实 HTTP／Agent 接口、四路 RGB-D 和手腕微动检查。Task9 四种正反组合另有静态配合检查。这些结果不包含完整机器人布线、装配或拆除。

## 目录与接入

每项都使用 `tasks/<task_id>/definition.py`、`task.py`、`rules.py`、`evaluator.py`、`README.md`。家族根目录的 `nist_assets.py`、`nist_composition.py`、`nist_release.py`、`nist_rules.py`、`nist_native_evidence.py` 和 `nist_evaluation.py` 提供共用构建与裁判逻辑。

继续使用固定房间 `mechanical_workshop_20261017_v3`；默认单臂保留原 FR3／右 SHARPA 和 `mechanical_single_arm_v1` 支架，双臂使用已发布的官方左右手构型。任务自有零件与工装放在原工作桌上；原 Task1–5 的发布指针与规则不因这些新增任务改变。

通过普通 `/episodes` 接入，Agent 使用 `--family mechanical_assembly --task task6`～`task9`，也可用对应 canonical ID。种子记录回合身份，当前物理初态固定；不会按种子改变原始几何。公开端只返回规则和粗粒度阶段，私有裁判读取原生刚体、接触与被动链条状态。

## 实用规则与裁判

- 允许重新抓取、退出重插、错位纠正和重新布线；一般未完成状态可以继续。
- 必须满足实际目标、配合接触、支撑及释放后连续 300 ms 稳定。初始化和工程探针的预摆目标状态均不计任务分。
- Task6 不要求电气导通或 BNC 锁扣功能。BNC 使用成对同比例接口的部分插入目标，未单独扩孔或增加锁紧成绩。
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

## 初次 NIST 发布与验证结果（历史）

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
