# Task2：单臂水壶倒水

用户将原眼镜盒提案替换为水壶倒水。任务沿用固定客厅 v7 和 front 单台 FR3＋右 SHARPA，文件沿用其他任务的 `definition.py`、`rules.py`、`task.py`、`evaluator.py` 和 `release.py`，粒子读取在同目录 `native_observer.py`。原眼镜盒仅为未发布设计，保留在 Git 历史中，不再占一个目录任务。

当前任务候选为 **v5**，引用共享水体资产 **v2**：488 个真实 PBD 粒子、8 mm 间距、总水重仍为 250 g，两容器关闭休眠以持续传递和观测液体受力。它采用用户要求的实用近似；满壶水重反作用误差约 +4.4%，杯内 93.75 mL 时约 +22.7%，不宣称精确水动力学标定。v5 在 v4 相同物理配置上补充水差与洒漏反馈；原 v1 资产、v3／v4 任务及以下历史试验均保留。详见文末本轮记录。

流程是抓住壶把、提壶、对准杯口、倾斜倒水、收壶止流、放回并松手。基版采用常温水、一个带把手和壶嘴的水壶、一个宽口杯及接水托盘，考察稳定握持、负载下的腕部姿态、壶嘴对准和流量控制。杯和壶均自由放置，不暗中固定；接水托盘只收集洒水，不免除洒漏计分。

当前冻结发布的初始水量 **250 mL**，向可用容量 **200 mL** 的杯中倒入 **100±10 mL**，累计洒漏上限 **5 mL**；壶可用容量暂按 **500 mL**。这是已冻结的仿真候选，不代表实物标定。`task_configuration(seed=0)` 加载真实发布；覆盖为不同 `quantities` 时仅返回显式设计方案，不修改发布或原生粒子库存。新的水量与容器参数必须创建新版本。

成功需要实际经壶嘴流入杯中的水量落在目标区间，洒漏未超限，杯仍正立，壶正立放回归位区，手离开两容器，且容器、水量和液体运动满足完整 0.3 秒的稳定窗口。未决水量不超过 1 mL，杯量在稳定窗口内变化不超过 2 mL；大量水仍在空中时不能提前完成。允许停倒、补倒、放下换抓；超量不算成功，不能删水或修改液位纠正。超时记未完成；观测或质量账本缺失则不评分。

水量按冻结的粒子物理质量除以参考密度换算，不按渲染表面、蓝色面积或倾倒角度估算。杯内统计使用真实内腔和杯口范围，杯外壁上的水不能算入杯。当前水量按壶内、杯内、飞行中、其他位置互斥分类；累计洒漏另按粒子身份只计一次，回收不会抵扣，避免与当前质量分区重复相加。漏读、删除或消失的粒子是证据故障，不能当成没有洒水。

洒漏按真实粒子轨迹进入冻结的托盘、桌面、地面或容器外壁区域，或离开任务域来认定。空中尚未落定的水单独跟踪；分类需要完整区间覆盖和明确的几何容差。粒子不提供普通刚体接触报告，不能直接套用搭塔的 contact 裁判，也不将缺少 contact report 解释成没有碰撞。

按用户要求，v5 裁判保存的每次结果和 `final_score.json` 明确报告水差：

- `water_difference_ml = initial_water_ml - kettle_water_ml - cup_water_ml`。
- `current_spill_ml = water_difference_ml - unresolved_ml`；水流在空中时暂不算洒漏，落定且未决为零时直接由水差得到洒漏量。
- `cumulative_spill_ml` 保留整回合已洒粒子历史，洒后回收也不减分；`remaining_spill_budget_ml` 为 5 mL 减累计洒漏，最低为零。

例如既有独立台架结果为 `250 - 156.25 - 93.75 = 0 mL`，没有洒漏。
当前粒度下 9 粒约 4.61 mL，允许；10 粒约 5.12 mL，判为洒漏超限。
杯中量即使达到目标，累计洒漏超量仍不能成功。规则和这些跨阈值、空中水、回收水的
CPU 回归均已覆盖。上述数值是保存的裁判结果；当前公开动作接口不新增私有粒子数据。

## 液体实现路线与当前边界

候选路线是当前固定 Isaac／PhysX 的 GPU PBD 液体，外观再用等值面重建。官方 [Omni Physics 粒子说明](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/particles/particles.html)描述了 GPU 粒子液体及与刚体、关节系统的交互；表面后处理只改变渲染。这说明引擎存在实现路线，不等于本项目已经接入或验证。

2026-10-18 只读检查：默认 `scripts/isaac_python.sh` 指向共享 `assets/simulators/isaacsim/6.1.0`，实际 `runtime/VERSION` 为 `6.1.0-rc.26+release.49347.2d230af4.gl`，PhysX 扩展为 `110.3.2+110.3.0`。其 `extscache/omni.physx-110.3.2+110.3.0.lx64.r.cp312.u7f4/omni/physx/scripts/particleUtils.py` 已提供液体粒子创建工具。已安装的 FluidBallEmitter 示例使用高黏参数，不能直接当作水的标定。

通用 `bench/simulation/isaacsim_backend/native_task_bundle.py` 已接入只读原生粒子和容器状态；逐子步计量通过冻结 observer 交给裁判。开启表面平滑时，需核对 `physxParticle:simulationPoints` 等真实状态与渲染点的区别，并验证同步时钟、数量和质量；本地未发现可直接复用的粒子 tensor view。官方[粒子限制](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/guides/current_limitations.html)与[碰撞兼容表](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/rigid_bodies_articulations/rigid_bodies.html#rigid-body-collider-compatibility)也要求单独核对粒子接触报告与 GPU 碰撞形状支持。文档版本不是本机 RC 运行验收。

必须先验证壶腔、壶嘴、杯壁对粒子不漏穿，水可以自然经壶嘴流出，质量账本闭合，并验证液体与动态壶／杯的双向耦合及机器人受载。不能只让壶推动水而不核对液体反作用，也不能将水质量同时算进壶的干质量和粒子质量。碰撞厚度、粒子间距、时间步长与流变参数需要共同校准；不能把空心壶／杯烘成封口凸包。

建议验收顺序：静置保水与账本 → 独立容器倾倒／停止出流 → 动态容器双向受力 → 原机器人实际提壶与倾倒。原生观测与参考裁判已按下述接口实现；发布状态和原生可运行性以实际冻结任务包及其入口验证为准，不能仅凭参考裁判存在宣称机器人成功。当前候选包含容器、真实 PBD 水、规则、裁判和 canonical 回合入口，完整机器人抓壶倾倒尚未验证。验证不升级共享运行时、不改共享控制器，也不把粒子动画称为液体物理通过。

本轮参数约束和目录回归见设计验证记录（开发存储 `datasets/fr3_sharpa_bench/development/living_tasks_20261018/water_design01/validation.json`），仅为 CPU 检查。

## 实际场景资产候选

`assets.author_task(stage, root_path='/World/LivingTask', presentation=False)`
创建玻璃冷水壶、玻璃杯、暖白托盘和真实 PhysX PBD 粒子水，关联调用方的
`/World/PhysicsScene`；不创建房间、机器人、灯光或物理场景。壶和杯均为动态
刚体，壶的定向敞口壶嘴与内腔是同一闭合玻璃实体壳上的真实空腔，使用 SDF
碰撞；把手由单独实体段构成，不能穿过握持空隙。托盘是明确声明的静态承托。
玻璃、钢底圈、暖白把手和真实液体等值面提供外观，presentation 不替换物理水。

原 v1 候选初水为 0.25 kg、9,259 个固定 ID 粒子，间距 3 mm，逐粒质量为
`0.25/9259` kg。壶干质量 0.32 kg，杯干质量 0.105 kg，未重复计算水重。
粒子间距、材质系数、容器干质量和 SDF384 都是待验证候选，不能当实物标定值。
原始粒子位置通过 `read_water_state(stage, metadata)` 读取。当前固定运行时
显式禁用 smoothing／anisotropy 后使用真实 `UsdGeom.Points.points`；不要求只在
渲染平滑分离时才存在的 `physxParticle:simulationPoints`。任何后处理开关、
粒子数量／ID 或质量不符合契约都会拒绝读取，不用渲染表面或初始点补缺。
动态调用方传入 `container_pose_xyzw` 原生世界位姿，直接计算腔体分类，避免
为了更新容器坐标而写回正在仿真的 USD。该资产读取函数返回诊断账本；正式累计洒漏和任务计分由冻结的
`native_observer.py` 与 `evaluator.py` 完成。

`scripts/validation/living_water_probe.py` 仅允许完整冻结源码和显式 GPU UUID，
在独立桌面上检查逐步粒子库存、防漏、动态容器运动及壶底刚体反力，导出真实
`settled_snapshot.usda` 和 `settled_particles.npz`。`--dry-control` 会在初始化前
去掉粒子作为独立空壶对照；其证据不混作含水任务。成功启动或静置保水不能证明
双向负载已定量准确，更不证明机器人抓握、倒水或任务成功。每次试验使用新的
共享 datasets 输出目录，保留失败记录；旧资产及固定环境不变。

2026-10-18 的 bounded_evidence01.json（开发存储 `datasets/fr3_sharpa_bench/development/living_water_20261018/bounded_evidence01.json`）
记录源码 `19bff51`、当前固定运行时和 AB GPU 的独立原生检查：`dt=1/960 s`，
含水与空壶各 2,880 步／3 秒且进程退出 0。含水运行全程 9,259 粒、0.25 kg
保留壶内，逐步 USD／原生刚体位姿一致；最终 USD、NPZ 与最后观测的物理点相同，
原生生成 10,746 顶点等值面。当前候选诊断时间步为 `1/960 s`；原 `1/240 s`
记录在 `native10`，漏出 1 粒约 0.027 mL，明确未通过，未删水或覆盖失败。

**后续复核发现该历史流程每步 `fabric.save_to_usd()` 会将刚体速度清零，
因此这些记录只保留为历史观测，原静稳／动力学资格撤回；不覆盖原文件。**
历史含水壶最后 1 秒地板反力均值 3.321 N，空壶
为 3.137 N，差 0.184 N，远小于 0.25 kg 水的 2.453 N 重量；定量水重反作用
**未通过**。不能从这张真实粒子水图推断提壶负载、自然壶嘴倾倒、机器人操作、
累计洒漏计量或整项任务合格。历史记录不作为当前发布的动力学验收。

## 负载与观测修复验证

`author_task(..., water_mass_kg=.25)` 支持显式水量对照，默认保持原 3 mm 间距、
1000 kg/m³ 参考密度和精确总质量，粒子数按 `round(m/(1000*.003**3))`
生成；超出候选内腔容量会拒绝。125／250／375 g 分别为 4,630／9,259／13,889
粒。壶的干质量仍为 0.32 kg，没有把水重加到壶上，也没有额外补偿水力。

`living_water_probe` 默认 `--sync-mode read_only`：每步只用官方结果读取
更新真实粒子点，以 native rigid view 的位姿分类；校验读取前后刚体速度
未改变。Fabric 仅在最终暂停后导出快照，导出后不再执行任何物理步。
`--sync-mode legacy_save` 只用于复现已失效的历史观测流程，不得用其结果
宣称动力学合格。开启 Fabric 的速度更新开关本身未能修复该反馈。

诊断可用 `--vessel-sleep-threshold 0` 排除休眠，`--particle-iterations`
对比求解次数，`--upward-force-n 4` 在 1.5–1.75 s 向壶的真实质心施加
有限外力。后者是独立台架激励，不是机器人控制，也不是液体补偿。
动态抬升会有意违反原静置位移门限，需结合 `status`、逐步记录和明确的
负载判断读取结果，不能把通用 `passed` 当作机器人或液体受载通过。

最新负载诊断摘要（开发存储 `datasets/fr3_sharpa_bench/development/living_task12_validation_20261018/task2_load/evidence01/result.json`）
保留全部失败与对照，不覆盖历史数据。修复后的 `sync02` 源码为 `e449b3a`，
固定运行时、AB GPU、3 秒原生逐步观测：

- 只读 `dt=1/960 s` 静置全程 9,259 粒保留壶内，逐步读取前后刚体速度差为 0；
  半步 `1/1920 s` 也无腔外粒子。这是该候选的短时保水证据，未作实物标定。
- 原生质量核对（开发存储 `datasets/fr3_sharpa_bench/development/living_task12_validation_20261018/task2_load/mass_analysis01/result.json`）
  从 OVD 的首／中／尾真实 float4 缓冲取出 9,259 个逆质量，全部为 37,036 kg⁻¹，
  合计 0.25 kg；每一行 XYZ 与官方 OVD 转换器输出精确一致。该录制发生于
  历史诊断，后续外力调用失败且用了失效同步；只采信其质量，不采信动力学。
- 同源只读、相同 1.0–1.5 s 窗口：空壶反力 3.139200 N，含水壶 3.320984 N，
  差值为预期水重的 **7.41%**。半步含水壶反力 3.346540 N，仍不足。
- 4 N 向上短脉冲使空壶抬升 10.70 cm，含水壶抬升 7.42 cm。4 N 小于含水总重
  5.5917 N，却能明显抬起，说明负载不足不只是地板反力读取问题。两次均完整
  完成 2,880 步；退出 1 是主动抬升违反静置位移门限，非启动故障。含水脉冲
  曾有 1 粒短暂在腔外，最终全回；不把它冒充累计洒漏计量。

单次边界密度参数候选得到约 3.394 N，未解决负载不足，未用于默认资产；
休眠和粒子迭代次数的早期对照使用旧同步，保留为受限历史数据。没有给壶重复
加水质量或补偿力。**水重耦合列为后续改进，不阻塞候选场景使用**；其不足需
在解释机器人受载时保留，不宣称定量负载已校准。本文不替代自然壶嘴倾倒、
任务规则／裁判或机器人操作的独立验证。

## 已实现的规则、裁判与原生观测接口

沿用棋牌室的 `SequenceEvaluator(task).update(observation)` 和
`inspect_conditions()` 接口，冻结 `rules.py`、`evaluator.py`、
`native_observer.py` 三个文件。裁判本身只依赖标准库和同包规则；原生 observer
用 NumPy 对真实粒子进行向量化计量，不导入可变项目源码、USD 或控制器。

任务工厂调用 `evaluation_configuration(seed=0, physics_dt_s=.0005,
asset_metadata=metadata, quantities=None)` 冻结量纲、粒子身份、内腔、托盘及
规则摘要。`NativeTaskObserver(完整task)` 只接收后端只读 `on_frame(frame)`，
在 `on_observe()` 返回私有 `task_state`。每个原生子步检查固定粒子数量、ID、
质量和时间连续性，累计明确落到托盘（含边沿）、桌面、地面或任务域外的粒子；
当前腔体分区与历史洒漏分开，回收水不会消除洒漏。容器位姿直接来自 native
rigid tensors，仿真期间不调用 Fabric 保存回 USD。

实用成功条件为：真实提壶至少 2 cm，并观察到经前壶嘴区域流出；杯中名义水量
90–110 mL，累计明显洒漏至多 5 mL，未决水至多 1 mL；壶放回初始中心 6 cm
范围、高度误差不超过 1.5 cm，两容器倾斜不超过 15°，具有桌面／托盘支撑，
无活跃机器人接触，低速保持 300 ms，杯量在窗口内变化不超过 2 mL。精确阈值
由 `public_rules()['success']['thresholds']` 发布。休眠容器仅在已观察到真实
支撑、位姿基本不变且低速时保留上次支撑证据，不能从初始位置凭空推定支撑。

首帧只建立未倒水基线，不得凭 reset 得分。漏读、重复帧、配置串用、粒子身份
或总量变化返回 `invalid_observation`；超时返回 incomplete，均不假判成功。
明确超出洒漏预算或杯子翻倒／容器跌落才给出不可恢复原因。任务状态的
`success/progress/reason` 与棋牌室兼容；服务私有水量／粒子数据不由动作 API
提交。名义体积裁判可使用当前候选场景，水重定量标定不作为其发布阻塞。

53 项 CPU 契约与资产测试覆盖正常完成、松手与归位、累计洒漏、未决容差、观测故障、
休眠支撑和独立冻结导入；这些测试不是原机器人完整倒水的成功轨迹。


## 自然出流与洒漏的独立验证

只读原生台架 `development/living_task12_validation_20261018/pouring/aimed05`
使用独立真实铰链与不超过 2 N·m 的外力矩倾壶，完整记录 9 秒／8640 子步；
3559 粒水从前方壶嘴离开，最终杯内 96.096 mL，洒漏和未决水为零，收壶后杯量稳定。
`aimed05_position_audit.json` 独立重读所有原始粒子点，计量一致。该台架没有机器人，
仅用于验证出流、止流和账本，不向任务裁判提交成功。

偏离杯口 9 cm 的 `missed01` 正确记录杯内为零。原账本把停在托盘沿的 3 粒水
留作未决，原失败报告保留；`missed01_rim_review.json` 使用相同 8640 步原始点和
明确托盘沿几何重新分类，最终累计洒漏 110.784 mL、未决为零。这是原始数据复核，
不是新的原生试验。未做额外半步倾倒收敛试验，水重不足的限制仍按上文保留。

## 水重耦合的实用改进：资产 v2／任务 v4

按用户“适当符合逻辑，不要太严格”的要求，这轮交付采用实际粒子液体的工程近似，
保留真实水质量、自然出流、双向受力和逐粒计量；不要求精确压力场或实物标定。
完整本轮证据（开发存储 `datasets/fr3_sharpa_bench/development/living_water_coupling_20261018/evidence01/result.json`）
保留所有对照、失败、源码快照和进程退出码。

新资产将间距从 3 mm 调到 8 mm，库存从 9259 粒变为 488 粒，单粒约
0.5123 mL；水仍为 0.25 kg、参考密度仍为 1000 kg/m³，壶／杯干质量仍为
0.32／0.105 kg。粒子偏移、初始边界留隙和渲染等值面随分辨率一起调整，
所以这是整套离散配置变化，不是只改一个接触参数。两容器 sleep threshold
显式为零，保证受水影响时持续参与求解和接触力读取。未添加额外重量、补偿力、
脚本水流或抓握固定，也没有改共享控制器、运行时和客厅 v7。

全部下列新试验采用 0.5 ms 原生子步、只读观测：

| 检查 | 实测结果 | 解释 |
|---|---|---|
| 满壶静置 3 秒 | 488 粒全留壶，支撑 5.70017 N；理论 5.59170 N | 总重误差 +1.94%，扣空壶后水重误差 +4.42% |
| 同源空壶 | 3.139200 N | 与 0.32 kg 干质量一致 |
| 向上 4 N、持续 0.25 秒 | 满水壶没有离开支撑面 | 修正了旧配置能被小于总重的外力抬起的表现 |
| 75° 独立倾倒台架 | 杯内 93.75 mL，壶内 156.25 mL，洒漏／未决为零 | 达到 90–110 mL 区间，收壶后稳定，无底部漏水 |
| 杯子真实增重 | 26.64 mL 时误差约 +0.12%；93.75 mL 时约 +22.68% | 有实际双向耦合，但存在水量相关误差 |

原 3 mm 在同一步长下只传递约 8.59% 水重；64 次粒子迭代只有约 11.66%，
未被选用。10 mm 配置水重误差约 +10.10%，也未被选用。65° 台架只倒出约
27 mL，其未达目标记录保留。75° 是独立台架的有限外力矩测试，不是机器人策略；
没有为这个结果修改共享控制器或任务成功区间。

杯中 93.75 mL 的约 23% 偏高已经用逐步原始点、固定粒子身份和静稳状态复核，
不能解释为仍在入水，也不能声称定量受力已经准确。当前固定 PhysX 的多接触质量
分配机制仍存在；粗粒子只在这套容器和水量下显著改善效果，流束和杯沿交互更粗。
相关原生内核研究（开发存储 `datasets/fr3_sharpa_bench/development/living_water_coupling_20261018/native_kernel_audit02/README.md`）
及独立 SPH 反力原型（开发存储 `datasets/fr3_sharpa_bench/development/living_water_coupling_20261018/splishsplash03/review.json`）
仅作为后续备选；本次没有升级运行时或接入另一套流体求解器。

资产构建入口保持在本任务目录：从完整冻结源码执行
`python -m scenes.workrooms.living_room.tasks.water_pouring.build_assets --output <新的共享资产版本目录> --particle-spacing .008 --vessel-sleep-threshold 0`。
它只构建初态 USD、metadata 和 manifest，不伪造原生资格；每次输出必须是新目录。
默认 `author_task()` 仍保持原 3 mm 作者行为，原 v1／v3 内容未被改写。

v4 的独立任务入口检查（开发存储 `datasets/fr3_sharpa_bench/development/living_water_coupling_20261018/publication/water_pouring01/qualification.json`）
通过：原机械臂保持 300 ms／600 原生子步，488 粒全部留壶，裁判未误判成功，
四路 RGB-D 正常。查看实际相机画面（开发存储 `demo/interface/scenes/living_water_coupling_20261018_v4/index.html`）。

新增水差结果的 v5 已独立通过同样的原生任务入口检查（开发存储 `datasets/fr3_sharpa_bench/development/living_water_coupling_20261018/publication/water_pouring02/qualification.json`）：600 子步／四路 RGB-D 正常，最终水差为零、剩余洒漏额度 5 mL。发布 manifest 为 `83e6616a3ef22864139298ccdd36f07345c532935c63ca67fb3cd0ab4eaaffd5`。查看v5 实际画面（开发存储 `demo/interface/scenes/living_water_coupling_20261018_v5/index.html`）。
