# Task5 · 卡扣收纳盒取物

依次完成真实机构操作：旋开前侧两个金属卡扣、打开盒盖、取出蓝色小盒放入右侧木托盘，再关盖并旋回两个卡扣。卡扣先后顺序不限，允许重抓、调整和借助桌面；一个 Agent 可用单右手或选用左右双臂构型。

盒体约 30×24×11 cm，蓝色物件约 7×5×3.6 cm，托盘约 18×22 cm。盒体、盖、两卡扣和物件均为重力下动态刚体；盖和卡扣以三个被动 RevoluteJoint 连接。卡扣向外旋开，锁闭依靠舌片与盖的真实碰撞；没有状态开关、位姿动画、吸附或暗中固定盒体。

裁判从每个 0.5 ms 原生步只读刚体姿态、速度和接触，要求实际见过开扣与开盖，再确认物件完整在盘内且盘底承托，最后盖角度不超过 5°、两扣不超过 8°、全部脱离 `/World/Robot/` 接触并稳定 300 ms。初态不计分，留扣未关不成功。允许左手、右手和双手参与；普通动作接口没有物体位姿或卡扣状态写口。超时是未完成，不是物件损坏。

文件职责与其他客厅任务相同：`assets.py` 作者化物件，`rules.py` 公开规则，`evaluator.py` 冻结只读观察和裁判，`definition.py` 固定发布，`task.py` 加载。房间、机器人与控制器复用家族共享入口。资产位于 `<开发存储>/assets/task_parts/living_room/storage_box/v2`；发布位于 `<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/storage_box/v3`，精确哈希见 `definition.py`。

验证材料统一放在 `<开发存储>/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/living`，保留所有冻结源码与原始失败。`native_cycle01` 暴露早期右卡扣转向会撞盖把手、被动阻尼量级过大的问题；`native_reset01` 是探针固定场景检查 API 调用错误，未作物理资格。`native_cycle02` 已完成 v2 原生正反例全链。最终 v3 修正预算终局锁存，保持全部物理资产、阈值和 dt 不变，并直接复跑原生验证。

最终 qualification_exact.json（开发存储 `datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/living/qualification_exact.json`） 通过，SHA-256 `e715dd685e6b11d24edae14da13a7193b0b0c100d34649777ee7d4b6261e2de3`。`source_runtime05` 驱动 v3 的 2 秒静置和 13.5 秒完整机构循环；共 31,000 个原生步、31,002 帧，逐帧重放与原记录完全相同。最大穿透 0.6573 mm；锁闭受力时盖最多张开 0.252°，且有真实盖扣承力接触；最终自由释放稳定约 1.2915 秒，未获得机器人任务分数。工程力峰值约 1.305 N、力矩峰值 0.25 N·m，盒体没有固定夹具。

workspace_stages.json（开发存储 `datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/living/native_cycle03/workspace_stages.json`） 保留锁闭、开扣、开盖取物、入盘和重闭锁的全部实测刚体姿态，供离线构型/IK检查；它绑定完整原生流的哈希。审计使用 `source_review08` 与原 Isaac 6.1 Python 解释器，不启动仿真。早期 `qualification.json` 的严格重放失败也保留：项目 `.venv` 与 Isaac 内置 Python 运行环境的 `atan2` 有约 1.8e-15 度末位差；最终改用同一解释器精确重放，数值差为零，没有放宽任何阈值。`source_review07` 的容差草案未运行、未作为资格。

工程验证允许显式、记录在每帧中的有限外力/力矩来验证物物交互；基线以后没有位姿/速度 setter。记录真实自由释放、全部原始接触、阶段实测姿态、源码、发布、GPU 和进程退出码。它不证明灵巧手策略成功，`manipulation`、`physics_accepted` 和 `task_policy_success_claimed` 继续为 false。规则单元测试的合成轨迹也不作为原生证据。最终 Agent 三构型接线和逐阶段 IK 由共享构型发布记录给出。
