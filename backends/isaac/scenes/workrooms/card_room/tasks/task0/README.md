# Task0：单臂随机三子摆放

Task0 是棋牌室 Task1 的简化练习：一套 FR3＋右 SHARPA W01，把三枚黑子分别放入 **3×3 棋盘上随机指定的三个不同格子**。每格一枚，三子可以任意分配到目标格，不要求三子连线，也没有对手或轮次。原有 L1／L2／L3 游戏及 Task1／Task2 的角色保持各自含义。

父环境仍为 `card_room_heritage_20261017_v3`。棋具以现有 `tic_tac_toe_3x3/v3` 为来源，棋格间距 140 mm、黑子底径 56 mm；仅保留三枚黑子。当前 task0 v5 显式选择保留原外轮廓的棋子 SDF 和原盒体表面的棋盘三角网格；棋子质量、质心、惯量及棋具视觉、材质、摆位不变，棋盘使用固定网格的通用 kinematic 适配。左侧支架和黑方西侧托盘仍用原资产。单臂基座在 `(0.65, -0.60, 0.90)m`，待命腕位置为 `(0.65, -0.40, 1.42)m`。机器人使用固定的 v12 单臂装配与刚性指垫。初始化仅设置关节，物体初态来自冻结场景；动作只能驱动机器人关节。

每回合从九格中无放回选三格。版本化 `sha256_ranked_cells_v1` 按种子对格子排序，同种子可复现同目标。创建普通回合时省略 `seed` 会自动抽样并返回实际种子；显式 `seed=0` 等值保持确定。正式评测计划采用已固定的计划种子。目标在回合期间不变，物理初始摆位不随目标种子改变。

成功要求三枚真实棋子分别独占三个目标格，完整投影轮廓距格边至少 1 mm，由棋盘接触承托，机器人松手后同时稳定至少 300 ms。不要求精确对准格心。落错格、仍在接触或被扰动时可继续纠正，稳定时间重新计；预算耗尽表示未完成。完整阈值来自冻结 `rules.json`，任务特有评分器在同一发布中冻结。

评分使用每 20 ms 的真实物体位姿、速度和每 0.5 ms 原生子步的完整接触收据。缺少收据或时间不一致会停止评分；没有支撑或没有松手不会记为成功。该采样范围不能表述为每个子步都测得物体位姿。任务源码实现、原生初始化预检和完成三子抓放是三个独立状态。

只读查询任务：

```bash
.venv/bin/python -m scenes.workrooms tasks --family card_room
.venv/bin/python -m scenes.workrooms task --family card_room --task task0
```

当前冻结发布为 `<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/card_room/task0/v5`，manifest SHA256 为 `024e9e842bcc9d17fe718c22b7bbe4d31fd32c9486ed5dbe1f8df0fa9b2f8108`。v3/v4 及其记录保留。服务配置该发布后使用普通 `POST /episodes`，`task_id="task0"`、`robot_id="sharpa_w01_right"`、`scene_variant="original_v1"`。创建响应的 `task` 和 `GET /episodes/{episode_id}/task` 提供本回合目标格、名义世界坐标及公开规则；`GET /tasks` 提供候选格和抽样规则。使用 `EpisodeSession.functions` 或对应 `/episodes/{id}/call` 执行动作；本任务不走双人 `/board-games`。

四路公开 RGB-D 为 `overview`、`task`、`room`、`wrist`，腕相机挂在 `right_hand_flange`。Agent 可读取目标和公开机器人状态，但任务内部的真实物体状态、评分器和私有接触记录不作为模型工具公开。

`scripts.validation.card_room_task0` 提供零模型调用的原生预检：从完整源码快照运行，经公共函数观察、读取相机、保持 0.5 s，然后检查原始接触与物理记录。它检查单臂 29 DOF、三子初始托盘承托和四路相机；保持不应误判任务成功。初始化通过不代表三子抓放通过。证据入口为 `<开发存储>/datasets/fr3_sharpa_bench/development/card_room_task0_20261017/qualification.json`，冻结 manifest 的准备状态保留为 `prepared_unvalidated`。

`native01` 已通过上述预检：源码 `0022aec`、发布 v3，实际执行 1000 个原生子步，单固定基座 29 DOF／43 links；最后 300 ms 三子托盘承托均有完整接触证据，机器人接触为零。初始与保持后各四路 RGB-D 通过数值检查；人工查看了初始 task／overview／wrist 三张原图，确认单臂、空棋盘和三枚供棋黑子。worker 退出 0，无原生错误；最终放置数为 0、成功为 false，未发生误记分。没有执行抓取或落子，也没有模型调用。原生记录绑定自己的源码快照；后续目录校验整合与文档更改不重写这份证据。

2026-10-18 的公开函数模型验证另见 qualification.json（开发存储 `datasets/fr3_sharpa_bench/development/task0_agent_validation_20261018/qualification.json`）。attempt02 以冻结源码 `f418f939`、同一 v3 发布和 seed 7 运行 Micu `gpt-6-astra`（请求 `max`），空技能库、无总时间预算；模型仅用公开 RGB-D／机器人信息和函数完成 3/3 放置，含第三枚偏格后的自主重新抓放。逐帧评分重算一致，128,600 个原生子步的接触采样完整，原生日志无错误。但第一次释放第三枚时，食指 MP 壳与 `black_02` 的求解器报告间距达到 −4.030 mm，超过审计阈值 2 mm，故最终 `outcome=safety_stop`、`physics_valid=false`，**未通过完整任务物理验收**。该接触事件待 bench 复核，不据此认定控制器缺陷，也不放宽阈值或改写发布状态。全程双视角 2 倍速 demo（开发存储 `demo/interface/agent_demos/task0_agent_validation_20261018/attempt02/task0_2x.mp4`） 按原始回合的仿真时间展示，明确保留无效物理结论；此结果不是 RSI 或多种子成功率验证。

用户指出上述视频约 27 s 的第三枚像瞬移后，另做了 运动连续性复核（开发存储 `datasets/fr3_sharpa_bench/development/task0_agent_validation_20261018/attempt02/third_piece_motion_continuity_review.json`）：仿真 55.06→55.08 s 棋子中心移动 16.309 mm，随后 20 ms 再移动 10.184 mm，第一窗口手腕仅移动 0.205 mm，并包含 55.0605 s 的穿透超限点。食指实测运动同窗突增，目标位置未见同等级突跳。原 RGB 中已有该快速偏移，不能用编码完整或倍速播放解释掉物理异常；20 ms 位姿尚不足以区分子步内高速弹出与求解器位置修正，需 bench 核对接触响应。0.25 倍速局部诊断（开发存储 `demo/interface/agent_demos/task0_agent_validation_20261018/attempt02/third_piece_first_release_025x.mp4`） 保留原始 10 Hz 采样，无插帧或重模拟，不改变原回合无效结论。

bench 后续以原始关节目标做了零模型 GPU 重现，证据见 接触复核报告（开发存储 `datasets/fr3_sharpa_bench/development/task0_contact_review_20261018/report.json`）。完整冻结源码保留原 360 文件，仅加入诊断入口；未改发布或物理参数，reset 后未写入物体状态，省略视频渲染。0→55.52 s 的 2776 个控制步及初态、全部 14 个状态数组与原记录逐元素相同；54→55.52 s 额外记录 3040 个 0.5 ms 子步及完整逐点接触。最大单子步原点位移 0.680 mm、质心位移 0.513 mm，发生在同一 −4.030 mm 报告间距处，速度从 0.069 升至 0.776 m/s；16.309 mm 是多个子步累积的快速弹出，并非本次记录到的单步 16 mm 位姿跳写。该点法向改变约 65.44°，用速度端点积分仍有残差，不能仅凭这一残差唯一归因于几何、CCD 或求解器内部修正。子步诊断图（开发存储 `datasets/fr3_sharpa_bench/development/task0_contact_review_20261018/replay01/plots01/contact_diagnostic.png`）与完整数值分析（开发存储 `datasets/fr3_sharpa_bench/development/task0_contact_review_20261018/replay01/substep_analysis.json`）保留这一区别。

已修复的 bench 缺口是 **2 mm 通用穿透检查原先仅在回合结束时运行**。当前 worker 在完整 20 ms ACK 后锁存超限并拒绝后续步进，service 先保留已执行 ACK，再优先返回 `safety_stop / physics_valid=false`；按原始记录会在 55.08 s 停止，无需继续到 64.3 s。239 项相关 CPU／协议回归通过、2 项跳过，全部 3215 个原始收据离线复核无误拒。此修复不预防该周期内首次峰值，也没有修复上述接触动力学；原回合及该重现仍不构成完整物理验收，不放宽 2 mm 通用门槛或任务自己的 1 mm 稳定条件。

进一步隔离见 原生接触缺陷报告（开发存储 `datasets/fr3_sharpa_bench/development/task0_contact_fix_20261018/root_cause.json`）。OmniPVD 确认旧棋子使用 `PxConvexCoreCylinder`。只保留原生烘焙后的食指 MP 凸包和原棋子，把原事件前一子步的真实状态作为 reset，第一步仍报告约 −4.030 mm；两次重复逐元素相同。端点处实际几何并不存在 4 mm 重叠，错误接触可以在没有机器人控制器、拇指或棋盘的独立系统中复现。该最小系统的冲量与完整关节系统不同，不将两者的响应幅值视为相同。

修复候选采用保留原始底盘和手柄外轮廓的单一闭合 SDF 网格，质量、质心、惯量、视觉材质、摩擦、接触偏移、CCD 和求解器设置均独立核对。512 边周向离散的最大径向弦差为 0.527 μm；共面细分不改变这一多面体边界。不启用全局圆柱近似：该诊断改变了其他形状且丢失原生接触记录，不能用其空记录宣称穿透消失。原始长细三角形的 SDF 烘焙还有约 563 μm 局部误差，失败候选全部保留；有限原生几何见证、法向承托力平衡、原异常姿态及完整机器人运动分别验收，尚未通过的候选不能发布为已修复。

新碰撞资产通过 `scenes.workrooms.card_room.tasks.task0.release` 的 `--token-asset-root`、`--token-asset-id`、`--token-manifest-sha256` 显式选择，三项须同时提供。它只替换三枚黑子的版本化资产引用，冻结清单包含 USD 及全部依赖。旧发布、原随机目标规则、机器人初态和统一控制接口保持自己的冻结身份；几何和接触证据不能替代新的公开函数操控验收。

2026-10-18 收尾证据见 qualification.json（开发存储 `datasets/fr3_sharpa_bench/development/task0_contact_closeout_20261018/qualification.json`）。v4 的单一棋子 SDF 已消除原圆柱最小复现中的错误深接触，但完整轨迹后段仍出现棋盘实际重叠与接触报告不符、随后大幅弹出的现象。v5 在相同棋子 candidate10 上增加棋盘 candidate01：4602 个顶点、9200 个三角形完全保留原棋盘六个平面；原生 OVD 烘焙几何逐点及拓扑一致，实际世界表面边界误差最大 28.6 nm。棋盘材料仍为 0.4/0.4/0，contact offset 1 mm、rest offset 0。新的 `--board-asset-root`、`--board-asset-id`、`--board-manifest-sha256` 同样须成组显式提供。

共享后端仅新增可复用的 `triangle_mesh` 固定道具表示及输入预检，非法顶点、索引、缩放和退化面在创建 USD 前拒绝。原有 `mesh`/`convex_mesh` 行为保留；任务资产选择在 task0 配置中。通用控制器、接口、摩擦、TGS 16/4、0.5 ms 步长、CCD、规则及阈值均保持原值，没有 task0 专用求解分支。

冻结源码 `3a36067` 的全程回归执行原始 3215 个关节命令、64.3 s、128600 个原生子步，退出 0，通用物理审计通过；最大报告穿透为 0.136629 mm。54–64.3 s 独立保存并复核了 20600 份六缓冲接触样本及 20601 帧原生状态；原先第三枚 1.96885 m/s、76.62 rad/s 的大幅弹出未重现，新窗口峰值为 0.34994 m/s、13.3336 rad/s。54–54.5 s 窗口内两枚静止棋子的有符号法向支撑冲量 ΣJz/(mgT) 分别为 0.999986、1.000001，T=0.5 s。全部已记录固定道具位姿零漂移；全程原生接触收据与后段原始缓冲区的采样范围分别保留。

用户明确：修复棋子异常运动后，旧回放的后续纠偏动作不再适合新的物理状态，不能用旧策略是否再次成功作为 bench 修复门槛。冻结裁判在 55.76 s 达到 3/3；继续执行原录制纠偏动作后，64.3 s 为 2/3，第三枚仍与拇指有载接触。保留这一真实结果，未调整控制器、停止时点或规则来凑成功分数。本次是已知接触故障的工程回归，没有新增模型调用或新策略成功率结论。

冻结源码 `3a36067` 的 443 项 CPU／协议回归通过；v5 真实 HTTP 的四路 RGB-D、IK、微小关节运动及返回、共享控制器 hold、动作去重均通过，46 个 ACK／1840 子步完整，server 和 worker 均退出 0。仍保留接触近似误差：有载棋盘接触最大深度漏报 0.051154 mm，底缘附近一个有载点法向偏 Z 82.39°（该时刻整对合力方向偏 Z 约 1.684°）；详见独立运动补充记录（开发存储 `datasets/fr3_sharpa_bench/development/task0_contact_closeout_20261018/full_replay01/motion_independent_review/ADDENDUM.md`）。原生子步位置与终态速度的积分残差及重叠 reset 探针的步长敏感性同样保留；局部法向随棋子转动不能单独证明内部缓存状态。本次不宣称零误差接触、全局数值收敛或其他任务发布已获资格，冻结 manifest 的 `prepared_unvalidated` 状态不改。
