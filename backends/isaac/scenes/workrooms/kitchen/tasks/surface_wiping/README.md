# Task7 · 海绵擦拭台面并归位

用蓝色浅盘中的黄色海绵擦三片棕色污渍，然后放回浅盘并松手。总体清除至少80%，每片至少60%；允许少量边缘残留、分段重复、重抓和推动，不要求固定路线、顺序或握法。最后海绵在盘内真实受支撑并稳定0.4秒。

任务拥有1个120×78×45mm、79.2g弹性海绵、1个真实回收浅盘、3片无碰撞可见污渍及12件固定非计分背景物。父房间、FR3/right SHARPA、原29关节初态、物理材料与控制器不变。污染由有限可见单元近似；实际有载擦面接触与至少3mm有效擦程累计12mm后消除该单元，材质恢复到原吧台色。海绵具有真实受力、压缩和回弹；没有流体、吸收或化学模拟。

静置、悬空、无接触、倒置和跳变不给清洁信用；正常抬起换位不会清空历史。海绵初态在回收盘内，但污渍尚未擦除，因此不能初态成功。180秒仿真预算，缺失证据返回unknown，预算耗尽为未完成，重试不永久判负。

任务入口为`definition.py`和`task.py`，资产、规则、裁判与发布共用家族实现。擦拭逻辑位于家族的 `wiping_assets.py`、`wiping.py` 和只读 `evaluation.py`。通过普通 `/episodes` 与 `--family kitchen --task task7` 或 `surface_wiping` 选择；Agent建议 `--sim-budget none`，采用任务自身180秒预算。

历史尺寸版资产和release为`surface_wiping/household_visual_v2`，以definition的路径与SHA256为准；2026-10-23证据在尺寸与接入检查（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_prop_scale_20261023/qualification.json`）。新海绵保留原40mm内移布局和平底凸碰撞，平面尺寸增加20%，擦面范围、质量和惯量同步更新；两层孔隙外观不替代体积柔性模型。独立0.4N工程擦拭与回收盘承托、机器人HTTP接口和离散IK分别验证。

## 当前弹性方案

当前资产与release为 `surface_wiping/elastic_cell_lattice_v2`，使用27个真实凸形材料单元、原4mm绿色擦层和63个被动六自由度力弹簧。保留原质量、尺寸、房间、机器人和共享控制器，不用位姿动画驱动物理。多孔皮层按原生部件姿态只读变形；v2仅消除重复皮层元数据，物理USD与v1逐字节相同。

GPU原生冲量、校准承重、原浅盘静置及动态压板测试已通过：承重0.776960N（名义0.776952N），最大压缩1.3686mm并回弹，接触载荷与承托增量误差0.36%，末段28部件原始速度均在原门槛内。19个离散IK／5mm净空点全部通过。普通HTTP／Agent和独立回放均通过：3,281帧完整记录28个部件及63个连接，末801帧承重0.776952N、最大原始线速度0.002431m/s、角速度0.01382rad/s。完整13,201帧擦拭复核实际有载接触及原污渍规则，首片100%、第二片7.35%、第三片0%；这验证物体受力擦拭，不计机器人完成三片任务。原运行的保守包围盒净空失败保留；实际碰撞形状的全帧净空下界为5.0497mm，含1µm数值保护，满足原5mm门槛。最终绑定见当前海绵验证索引（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/sponge_qualification.json`）。

公开合同v6要求完整海绵入盘、绿色擦层真实受支撑、所有物理部件稳定、整个原生碰撞外形离机器人至少5mm，并连续满足0.4秒。内部连接与几何必须完整，刚性底层静止不能掩盖上部运动。清洁阈值、预算及原速度／姿态限值保持原值。材料采用离散弹性工程先验，不声明连续体实物标定或吸水能力。

方案调研（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/research/actionable_review01/report.json`） · 完整原生压缩曲线（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/research/compression_curves01/compression_curves_v2.png`） · v1/v2物理身份独审（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/research/compact_asset_identity01/report.json`）。

## 历史：2026-10-24 刚体动力学与接口复核

正式刚体版本在 `source19` 重新完成真实HTTP／Agent接口、1.6秒归位承托、19个IK／实际几何净空点和6.6秒物体级受力擦拭，过程与独审绑定在第七项最终报告（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023/current_rigid_wiping_plan01/qualification02.json`）。擦拭仅施加每轴不超过0.4N的水平外力，无手部抓持或重置后的状态写入；首块污渍覆盖100%、第二块5.882%、第三块0%，不计完整任务成功。独立台面使用自动contact offsets，未证明与正式台面等价。末0.2秒未输出接触行，但位姿逐bit恒定且速度为0；没有原生休眠标志，不能补零推断承重不足或宣称测得休眠。

## 历史FEM诊断

已研究SPONGE的旧Isaac Gym／FleX方案，并在固定Isaac Sim 6.1上建立两层原生FEM候选。只读可视网格接口修复后，候选可压缩约2.31mm并回弹，没有四面体翻转；但独立校准承重约1.259N，明显超过79.2g对应的0.777N，无接触施力试验也丢失较多水平动量。粗网格和坐标平移均未消除问题，scene速度迭代上限为0的运行与基线逐bit相同，不能据此声称原生FEM确实执行0次迭代。

该阶段曾保留刚体入口；用户要求继续修复后，当前改用上述真实离散弹性方案。历史刚体资格不自动转移给新模型。FEM资产、未通过的动力学记录及原生只读皮层修复保留作显式诊断；未改质量来补偿误差，也未升级运行库或改共享控制器。具体来源与源代码／原生记录绑定见海绵决策报告（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023/sponge_dynamics_decision01/report.json`）和厨房当前复核（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023/qualification.json`）。

## 历史：2026-10-21 无标签内移版本

以下路径、55g海绵及验证属于放大前的资产v4／release v5，保持原记录，不继承为新物品证据。

共享资产路径：`<开发存储>/assets/task_parts/kitchen/surface_wiping/v4`。
冻结发布路径：`<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/kitchen/surface_wiping/v5`。
当前证据目录：`<开发存储>/datasets/fr3_sharpa_bench/development/kitchen_wiping_inset_20261021`。

当前台面没有新增文字、数字、箭头或提示色块。海绵与蓝色归位盘、三片棕色污渍按自然外观识别；公开物品描述使用左侧／中间／右侧，不要求擦拭顺序。12件背景用品不变。Task近景随操作区调整，overview保持原视角。

相对原近侧布局，所有操作物体整体向台内平移4cm（世界坐标y减0.04m）：海绵／盘(.25,-.67)m，三片污渍(.48,-.74)、(.65,-.76)、(.82,-.72)m。实际USD吧台范围为x[-.95,1.85]、y[-1.35,-.55]m；盘外沿距机器人侧桌沿由10mm增至50mm。相同种子、腕高1.12/1.17m及姿态的8个参考端点，在原布局和最终4cm布局均可解；6cm为7/8、8cm为3/8，因此采用保守4cm。失败记录保留，只说明本组参考姿态的收敛范围，不代表更深位置整体不可达，也不证明抓握或无碰撞轨迹。

布局变更审计（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_wiping_inset_20261021/layout_change_audit.json`）核对470个原无标签USD prim：仅海绵／盘的整体平移及204个污渍单元的同量平移，碰撞形状、质量、接触材料和其他属性不变。机器人、控制器、父房间、规则／裁判和其他相机保持原值。原生初始化／支撑、独立海绵接触、HTTP／Agent和当前图像按新v5重新检查；旧位置上的接触报告不直接作为新布局通过证据。

独立实际USD布局复核（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_wiping_inset_20261021/layout_review01.json`）确认海绵、盘、污渍完整处于桌面内，15个碰撞体的形状／属性保持一致；盘离12件背景用品最近508mm，三片污渍加海绵完整外廓的平面包络离背景至少416mm，操作区未发现其他静态碰撞体。新初态1.6秒／3200子步通过，四路RGB-D、海绵盘内真实支撑及零初态清洁信用正常；新接触探针6.6秒／13200子步通过，实际有载接触12786步，第一片100%、第二片2.94%、第三片0%，释放后工具静止且未获整项完成信用。

现有14项相关CPU检查通过；静置与微振负例从当前污渍metadata取真实中心，避免移位后空洞通过。独立物物探针在reset前将海绵置于第一片附近，使用与台面一致的代理，仅施加每轴不超过0.4N的水平工程力；不含机器人、reset后位姿／速度写入或手物策略。手物交互和完整机器人任务由数采验证。准备release保留 `prepared_unvalidated`、`physics_accepted=false`。

当前v5真实TCP HTTP与实际Agent客户端复核通过：冻结合同、四路RGB-D、只读IK、关节微动／回程／hold和幂等均正常，physics_valid=true，server与worker退出0，0模型调用。当前图片已直接检查，没有新增文字／数字标识，操作物体可见。

当前交付索引（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_wiping_inset_20261021/qualification.json`） · 当前操作近景（开发存储 `demo/interface/scenes/kitchen_wiping_inset_20261021/task.png`） · 当前房间全景（开发存储 `demo/interface/scenes/kitchen_wiping_inset_20261021/overview.png`）。图片使用原生PNG原字节，无后期文字。

历史资产v1–v3、release v1–v4及其失败／通过报告全部保留；原物物验证（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_wiping_20261021/qualification.json`）和历史标签候选验证（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_wiping_labels_20261021/qualification.json`）均不替代当前无标签布局证据。
