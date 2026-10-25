# L5 三齿轮装配与工具传动

当前验证对象是 `table_inset_v1` 新布局，使用 `gear_train_assembly/v3_4` 的
20/32/20 齿、5 mm 模数、130 mm 相邻轴距；中间轮外径 170 mm。
三轮和三件工具在原桌面独立供料，没有供料垫块。新版冻结入口由
[definition.py](definition.py) 的 `design.supply_layout_candidate` 指定，
不能用旧布局的原生报告替代。

当前 rules v4 仅把终点净转比容差由 5% 改为 10%，容纳短行程的正常齿隙；
每对相位误差、目标区、真实工具接触、传动信用、速度和释放条件均保留。
新发布与原 inset 发布的 `scene.json`、初态、物理配置和资产清单逐字节一致，
见物理等价核验（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_gear_scene_validation_20261020/practical_rules_v4/physical_equivalence.json`）。

按用户 2026-10-20 的要求，场景验证以实际物理行为和目标完成为主：

- 独立工具真实接触输入轮，两级齿轮接触传动；中间轮反转，输出轮与输入轮同向，转比合理。
- 输出相对自然落座后的实际基线进入 25–35° 目标区，撤掉外部工具加载后能停住。
- 无明显卡死、穿透、飞散。正常套筒空程、齿侧间隙和微小接触抖动作为参考，不追求微米级精度。

[`validate_gear_driven_tool.py`](../../../../../scripts/validation/validate_gear_driven_tool.py)
直接规定真实套筒的运动，验证用户要求的工具—输入轮—中间轮—输出轮响应。
自然落座 1 s 后，只让套筒以 15°/s 绕输入轴转动；根据输出轮的实际角度停止，
随后把工具抬离 8 cm，再让三轮被动停稳 1 s。驱动最多 8 s。
末段要求实际工具至少抬离 6 cm、没有承载接触，齿轮留在轴位、有承托且能停住。
同一物理子步三对接触同时成立仅作参考；正常间歇接触不独立判失败。

这是显式的工程运动边界：仅套筒切换为运动学刚体，以原生 `set_transforms`
每 0.5 ms 写入套筒位姿。三只齿轮始终自由动态，reset 后不写齿轮状态、不加约束或外力。
此模式验证几何接触传动，不证明自由动态工具的力矩、机器人抓握或普通任务完成。
当前 GPU runtime 的 kinematic-target 接口未实现，逐步 USD 动画也未被手动积分路径消费；
对应失败记录保留，不能判为齿轮失败。

[`validate_gear_scene.py`](../../../../../scripts/validation/validate_gear_scene.py)
另提供新版供料、单轮、双轮、三轮诊断，以及 `--goal --practical` 自由动态工具加载诊断。
后者只给套筒有界扭矩，停止后不调用外力 API，其他物体保持被动；
没有更改共享机器人控制器。已有 1.6 N·m 和 2.8 N·m 加载均未达到目标，不能写成通过。

两种实用模式每 20 ms 保存一次实际位姿／速度，保留 backend 每个 0.5 ms 子步的
有向接触汇总及工具输入记录。
它不保存完整原始接触点，不宣称每个物理子步的物体速度最大值都经过独立复核。
所有运行从完整冻结源码启动。
本轮证据位于
`mechanical_gear_scene_validation_20261020`（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_gear_scene_validation_20261020`）：
`source01/input01.json` 对应分阶段诊断，`source02/input02.json` 对应完整记录的目标驱动，
`source03/input03.json` 对应普通控制周期观测的自由动态工具目标驱动，
`source04/input04.json` 对应 2.8 N·m 加载，`source07/input07.json` 对应套筒原生位姿边界。
每次运行的 `invocation.json`、`result/report.json`、`process_receipt.json`
分别记录启动输入、原生结果和进程退出；以实际已生成的结果为准。

`native09` 已完整通过：11,600 个原生子步、290 次实际状态观测，
输入轮转 29.040°、中间轮转 −17.646°、输出轮转 27.314°。
工具实际抬离约 8 cm 后无接触，三轮仍有承托并停稳；正常关闭，原生错误为零。
报告见 native09（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_gear_scene_validation_20261020/native09/result/report.json`），
独立退出核验见 process_receipt.json（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_gear_scene_validation_20261020/native09/process_receipt.json`）。
这是受控工具运动下的场景功能证据，不能把该轨迹的手持、逐窗传动信用或短窗速度当作正式任务已通过。
汇总见 qualification01.json（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_gear_scene_validation_20261020/qualification01.json`）。
新 rules v4 发布的 canonical HTTP 检查（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_gear_scene_validation_20261020/l5_http02/qualification.json`）
也已通过：规则／裁判加载、四路 RGB-D、只读 IK、小幅关节运动及共享 hold 均正常，无模型调用。

后续工具功能验证入口是
[`validate_gear_tool_function.py`](../../../../../scripts/validation/validate_gear_tool_function.py)，
实验位于 mechanical_gear_tools_validation_20261020（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_gear_tools_validation_20261020`）。
`socket_native01` 和 `wrench_native01` 均已完整通过：

| 工具 | 实际接合与转动 | 输出轮响应 | 实际退出 |
| --- | --- | --- | --- |
| 套筒 | 从上方下插 80 mm，再正向转 20° | 9.679° | 向上 80 mm，无接触 |
| 开口扳手 | 从侧面插入 40 mm，再绕输入轴转 20° | 9.506° | 沿工具后方 50 mm，无接触 |

三轮均有原生接触传动、承托并停稳。扳手接合时工具原点在输入轴 −X 方向偏 8 mm，
否则钳口背壁会顶到六角；绕轴转动时这段偏置也跟着转。只有受测工具规定运动，
其余五物保持动态；这两轮不授予手持资格。
18.75 mm 公六角螺丝刀不直接适配 54 mm 外六角，是合理的工具选择负例。

按用户后续澄清，手物交互由数采负责，本次只验收物体传动。
工具传动验收报告（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_gear_tools_validation_20261020/transmission_qualification01.json`）
已通过；手部历史试验另留在实验目录，不作为传动验收门槛，也不继续推进抓握脚本。

预装 reset 是验证场景接触的特权，不计装配、选工具、抓握或机器人任务成绩。
catalog 可运行候选绑定同一 inset 发布；三轮和三件工具仍从桌面独立散件开始，
预装 reset 只存在于特权验证中。发布准备时的 `prepared_unvalidated`、
`physics_accepted=false` 和任务成功标记保持原值；独立验证通过也不授予机器人装配资格。
历史诊断和冻结发布不改写。
