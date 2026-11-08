# Task6：展开阅读支架并放书

将折平的阅读背板展开至约55°，把右侧支撑销滑入，再将绿色硬壳书放到背板和前沿托唇上。允许重抓、重新调角、抽销再插、先放书后调整；最终要求50–60°、支撑销中心在20 mm范围内、书本完整位于阅读面、实际接触承托，并在机器人完全松开后稳定400 ms。没有固定双手分工或一次完成要求。

任务目录与其他书房任务一致；可复用配方、规则和裁判位于书房根目录的 `reading_stand_assets.py`、`reading_stand_rules.py` 和 `reading_stand_evaluation.py`。`task.py` 是本任务入口。父书房、机器人资产和控制器保持原定义。可重复使用的资产放在共享 `assets/task_parts/study/reading_stand/`，运行记录放在 `datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/study/`。

背板通过世界锚定的真实Y轴铰链连接到任务底座；带手柄的横向金属销使用真实Y轴滑轨，只有被动阻尼、没有位置弹簧或运行时锁定开关。销进入后，背板由实际杆面接触承载；把销抽出后，背板可以在重力下折回。铰链和滑轨连接明确属于任务机构，机器人没有隐藏吸附、焊接或物体位姿驱动。背板约320×250 mm，硬壳书140×170×16 mm；质量和材质是工程先验，未标定。

裁判从完整原生刚体姿态、原始速度与带符号接触读取：支撑销必须真实承托背板，书必须真实接触背板／托唇，所有机器人链接接触均阻止释放计时。初态、缺接触、销退出、书悬空、任一只手仍接触或缺失原生帧不能获得成功。预算240秒；桌面暂放可恢复，真实落地为任务失败，证据无效与超时分别处理。

独立动力学验证使用明确声明的有限力／力矩操作机构，随后彻底撤力；工程数据只验证物体交互，不是机器人抓取或完整Agent成功。原生结果、失败尝试、冻结源码及实际进程退出分别保存，最终证据由 `definition.py` 绑定。

## 已验证范围

资产 `reading_stand/v1` 与单臂基础发布 `study/reading_stand/v1` 冻结不变。单臂、同侧双臂与对侧双臂共用同一物件和规则，机器人配置选择及普通 Bench/Agent 接口由统一注册层提供。

物体交互验收（开发存储 `datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/study/qualification.json`）绑定七次原生运行、33.4秒／66,807帧和每次实际进程退出。最终400毫秒均复核原始速度、接触及位姿漂移，最大接触穿透1.35微米以内。正例含已锁带书静置、展开插销放书，以及全程每帧最多操作一个物体的顺序流程；负例含初态、未插销、实际拔销及锁好支架但书仍在桌面。182项相关CPU测试通过，原五项任务当前发布与新增发布均能按原哈希加载。

顺序流程先把背板抬到约78°，3.4秒撤去板上力矩，再单独推销；4.4秒时销进入中心4.1毫米，背板仍约74.7°，随后自然落到销上形成55°承托。8秒后才独立搬书，11.5秒全部撤力。这验证机构允许先抬板再换抓推销；机械臂实际从板柄移到销柄的抓取与换手轨迹仍需数采验证。真实阶段姿态（开发存储 `datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/study/workspace_stages_sequential.json`）供只读离线IK使用，不会写回运行场景。

完整只读复核（开发存储 `datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/study/full_readback01.json`）使用原Isaac Python逐帧重算冻结裁判的物理条件。4001帧原始初态始终零进度；工程操作第一次施力即被普通裁判拒绝，工程终态通过不冒充普通Agent任务成功。失败的第一次抽销控制停在262毫米，销尖仍承托背板，报告明确失败；随后只调整声明的工程控制力并真实退至280毫米。旧失败、CPU构建环境问题、GPU容量拒绝和跨解释器末位浮点差异均保留在进程记录（开发存储 `datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/study/process_receipts.json`）。

原生入口是 `scripts.validation.study_reading_stand_interactions`，只读复核是 `scripts.validation.study_reading_stand_readback`。都从共享datasets下完整冻结源码运行；物理探针使用原Isaac6.1.0启动器、显式GPU UUID与driver override。发布继续保留 `prepared_unvalidated` 和 `physics_accepted=false`；物物验证、接口验证和机器人操作资格分别记录。
