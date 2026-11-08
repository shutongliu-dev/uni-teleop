# Task5 · 开抽屉取物并关回

任务自有的桌上胡桃木抽屉柜位于原书桌近侧，内部放一本绿色工作笔记本，旁边有开放式绿色托盘。固定房间原有六个装修抽屉不变；原 FR3、右 SHARPA、支座及共享控制器不变。

拉开抽屉至少 100 mm，取出笔记本放入托盘，最后把抽屉关回 12 mm 范围内。笔记本在托盘内任意朝向和位置均可，允许重抓、反复开关、桌面掉落后捡回。最终整本进入托盘，底面真实承托，机器人离开抽屉和笔记本，两件动态体共同稳定 300 ms。预算 180 s，动作次数不限；不要求一次成功或指定运动轨迹。

抽屉底板、前后板、侧板和黄铜把手均有真实碰撞；滑轨是 160 mm 行程的原生 prismatic 关节，纯速度阻尼 5 N s/m，无位置弹簧、吸附、抓取焊接或运行时位姿写入。抽屉由轨道约束承载；笔记本评分必须取得真实托盘向上支撑接触，位置接近不代替接触。

冻结裁判逐原生 0.5 ms 读回两件刚体、滑轨坐标与有符号接触。先记录真实开启，再观察笔记本从柜体前方上提脱离，最后检查关回和托盘释放稳定。未打开直接预放目标、未取出、支撑缺失、仍由手托住都不会计分。真实落地为失败；缺帧或读回不一致为裁判无效；预算耗尽或取消为未完成。

资产位于共享 `assets/task_parts/study/drawer_retrieval/<version>`；发布位于 `datasets/fr3_sharpa_bench/isaacsim/releases/study/drawer_retrieval/<version>`；验证记录位于 `datasets/fr3_sharpa_bench/development/study_drawer_20261021`。每份 release 保留原始 prepared 标记，原生场景/物物交互证据独立记录，手物操作由数采验收。

公开入口为普通 `/episodes`，Agent 使用 `--family study --task task5` 或 `drawer_retrieval`；使用 `--sim-budget none` 时由冻结任务的 180 s 裁判预算管理回合。

当前资产 v1，发布 v4。v4 修复顺序追认：开满 100 mm 后必须观察到整本笔记本仍在当前抽屉包络内，再发生取出；先取出后补开、同帧首次开满并取出都不记取出分。提前取出仍可真实放回抽屉后重取，在柜外放低或放进目标盘不能代替放回。规则、场景、初态、物理及资产与 v2 逐字节相同，仅冻结裁判和来源记录变化。新冻结裁判 12 条正反例（开发存储 `datasets/fr3_sharpa_bench/development/all_task_rules_review_20261021/kitchen_study/drawer_v4_frozen_regression.json`）均符合预期；v2 原生证据单独保留。v3 是未采纳的标签候选，继续保留。

v4 的原生接入复验（开发存储 `datasets/fr3_sharpa_bench/development/rule_repairs_native_20261021/qualification.json`）从完整 `source01` 通过实际 HTTP／Agent 校验、单腕微动／返回／保持、幂等和四路 RGB-D；共 0.8 s／1600 原生步，`physics_valid=true`，服务及 worker 退出 0，零模型调用。独立复核（开发存储 `datasets/fr3_sharpa_bench/development/rule_repairs_native_20261021/drawer_review/http01_independent_review.json`）用 v4 冻结裁判逐帧重放 1601 帧，全部 41 个在线分数及最终结果一致。抽屉全程未打开，opened／armed／extracted 均为 false，因此只确认新裁判加载、观测链路及无误计分；完整取物和修复分支的物理操作尚未验证。

历史 v2 只把任务近景移到抽屉可操作侧，其他相机、物理、规则、初态和资产字节与 v1 相同。v2 在冻结 source04 / native04 通过 1 s / 2000 原生步、2001 帧冻结裁判重放、四路 RGB-D，`physics_valid=true`，worker 退出 0，无初态计分。

单独 `slide01` 使用限幅 0.6 N 工程载荷作用在抽屉上，笔记本保持被动：6.8 s / 13600 步内抽出完整160毫米、关闭，并在最后300毫秒真实承托静止，没有越轨或飞出。单独 `tray01` 从显式目标支撑初态运行1.2 s / 2400步，笔记本由真实托盘向上承托并释放稳定。73个实际USD碰撞盒路径采样进一步确认全开后笔记本可上提、越过柜体、进入托盘；这是对象几何通路，未验证手部轨迹。

上述独立台架不进入任务评分，不声称完整机器人取物。121项规则/旧runtime检查、77项发布与证据检查及两个新v2/几何检查通过，组间有重叠。准备时的 `physics_accepted=false` 等字段保持不可变；原 preflight 启动失败和 v1 成功记录均保留。

证据：qualification.json（开发存储 `datasets/fr3_sharpa_bench/development/study_drawer_20261021/qualification.json`）；原生原图：任务近景（开发存储 `demo/interface/scenes/study_drawer_20261021/task.png`）、房间全景（开发存储 `demo/interface/scenes/study_drawer_20261021/room.png`）、图片入口（开发存储 `demo/interface/scenes/study_drawer_20261021/index.html`）。
