# Task10 · 双臂协作插接

同一个 Agent 通过同一个普通 `/episodes` 同时控制两台 FR3，分别装官方左 SHARPA 和右 SHARPA，共 58 DOF。当前修正版采用 `dual_fr3_sharpa_w01 / isaacsim_6_1_v2` 和 `mechanical_same_side_left_right_sharpa_v1`：两臂并排位于工作台的 −X 侧，面向 +X 时，左手基座为 `(-0.10, +0.30, 0.78)` m，右手为 `(-0.10, -0.30, 0.78)` m，两者 yaw 均为 0。左右是机器人面向工作台的自身方位，不按相机画面的左右命名。

公开接口的 `active_sides` 为 `left/right`；同一条 `move_geometry.hand_geometries` 可以同时携带两手目标，分别使用 `left_hand_flange`、`right_hand_flange`。两套独立 articulation 仍属于同一个 Agent，没有玩家分区或轮次。历史 v3 的 `front/rear` 是两只右手的命名空间，单 Agent 控制语义原本正确，但并排复制右手没有形成真实左右手组合；新版本使用已有官方左手模型，未通过旋转右手或负缩放伪造左手。

桌面有独立套筒、独立插销和蓝色收集托盘。套筒开口为 28 mm 方孔，插销为 20 mm 方销，每侧 4 mm 余量；孔底和四壁都是实际碰撞几何，没有吸附、焊接或虚拟固定。一臂扶持可移动套筒，另一臂插销插入，再将装配件放进托盘。

规则允许重抓、退出重插、换手、借助台面辅助承托，不限制尝试次数。裁判要求套筒移动至少 15 mm，两个不同机械手分别实际接触套筒和插销，累计配合至少 80 ms，并在此期间产生至少 8 mm 的相对插入进展。最终插入至少 30 mm，插销真实接触并由套筒承托，套筒由托盘承托；完全松手稳定 300 ms 后成功。不要求精密压装、标定受力或漂亮轨迹。300 秒预算耗尽记未完成；物体实际落地才记任务失败；证据缺失记未知。

原房间、任务物品物理、材料和共享控制器保持固定，选用的官方左右机器人资产也不改写。远端 `x=.86..1.12 m` 加入 31 组固定工具架、螺丝刀、内六角、零件盒及量具，作为 task layer 复用；它们不是操作目标或评分支撑。中央工作面及双臂进场方向留空。

实现入口为 `task.py`、`rules.py`、`evaluator.py`、`definition.py`。资产在共享 `assets/task_parts/mechanical_assembly`；当前外观修正版证据根为 `datasets/fr3_sharpa_bench/development/mechanical_task10_finish_20261021`，图片根为 `demo/interface/scenes/mechanical_task10_finish_20261021`。原实验、图像和冻结发布继续保留在各自历史目录。

CPU 对抗裁判测试区分真实双手配合、同手/机械臂链接接触、零载荷接触、纯终态摆放、缺少托盘承托和未释放；这些测试不宣称手物操作成功。原生初始化与物物探针记录另附，手物策略归数采。

## 当前裁判 v6

当前发布为 `cooperative_insertion/v6`，资产仍为 v4。v6 修复接触空档里的退回未扣减：向内进展仍只在连续两帧有不同手真实配合时记功；退回始终扣除尚未确认的进展，包括松手、单手和重新接触期间。静止重抓可保留有效进展，正常重试仍可完成；50 微米往返不能累计成 8 毫米协作插入。

冻结 v6 正反例（开发存储 `datasets/fr3_sharpa_bench/development/all_task_rules_review_20261021/mechanical/cooperation_ratchet_frozen_v6.json`）确认合成观测中的真实净插入通过、连续接触及接触空档往返都拒绝。公开规则、任务、初态、物理和资产与 v5 逐字节相同，scene 只更新观察器源码哈希。以下 v5 的原生记录保留原版本范围。

v6 的原生接入复验（开发存储 `datasets/fr3_sharpa_bench/development/rule_repairs_native_20261021/qualification.json`）从完整 `source01` 通过实际 HTTP／Agent 校验、同动作双腕微动／返回／保持、幂等和五路 RGB-D；共 0.8 s／1600 原生步、58 DOF，`physics_valid=true`，服务及 worker 退出 0，零模型调用。独立复核（开发存储 `datasets/fr3_sharpa_bench/development/rule_repairs_native_20261021/mechanical_review/native_v6_review.json`）确认冻结评分器的 41 个控制摘要完全重放一致，3200 条原生接触记录中两物体各有末窗 600 步真实桌面承托。位姿仅保存每 20 ms，不能完整重放逐原生步观察器；没有手物接触或协作得分，退回修复分支及完整插接操作尚未物理验证。

## 物品细节 v5（v6 延用相同资产）

历史 v5 选用资产 `cooperative_insertion/v4`，当前 v6 延用该资产，外观 profile 为 `v2`。套筒采用银色加工框架与深青色嵌面，方孔内壁、孔口倒角、齐平紧固件和握持面分层可见；插销保留钢质方杆，头部增加银色车削肩、实际菱形滚花网格与端面同心细纹；蓝色托盘补充折边、角部连接、内底压边和轻微擦痕。没有新增文字标签。

36 个视觉网格配合原创 PBR 材质，30 张贴图及许可／来源清单共 32 项依赖均在共享资产中冻结。原 13 个任务碰撞体、6 个支座碰撞体、质量／惯量／接触材质、reset、左右手机器人及裁判阈值保持一致。可见表面最大外扩约 0.100 mm；逐三角面检查确认 28 mm 方孔无遮挡。材质 normal 仅影响显示，没有几何位移或物理作用。

公共 `task` 镜头移近并采用 1440×1080，另四路相机保留。`scripts.media.render_mechanical_cooperative_details` 从冻结目录生成双臂全景、桌面及三道具独立特写：1920×1440、64 spp，停止物理时间线、原始 reset，仅用于外观展示；HTTP 原生动态与接口记录单独保存。

本轮 v5 原生 HTTP 与 Agent 检查通过：同一动作左右腕微动、返回、保持及幂等重放，0.8 s／1600 步、5 路 RGB-D；独立回放验证末 300 ms 的实际桌面支撑以及原有 58 DOF 驱动、质量和接触参数。进程正常退出，零模型调用、无装配得分。前两次离线审核脚本的元数据／哈希域比较错误与修正均保留，正式通过报告为 `http01_review03.json`。

本轮证据（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_task10_finish_20261021/qualification.json`） · 桌面近景（开发存储 `demo/interface/scenes/mechanical_task10_finish_20261021/render01/task.png`） · 套筒特写（开发存储 `demo/interface/scenes/mechanical_task10_finish_20261021/render01/sleeve.png`） · 插销特写（开发存储 `demo/interface/scenes/mechanical_task10_finish_20261021/render01/peg.png`） · 托盘特写（开发存储 `demo/interface/scenes/mechanical_task10_finish_20261021/render01/tray.png`）。历史资产和发布均保留，完整手物操作仍由数采验证。

## Bench／Agent 驱动入口

当前双方选择同一 release v6（manifest `c125824029e81e4a55ce39149db922ea0ce96bfdb0d5609767dcaa6f576d8cf9`）；本轮已完成 CPU 规则核对和原生接口复验，以下命令使用同一冻结源码：

| 项目 | 当前值 |
|---|---|
| Agent 选择 | `--family mechanical_assembly --task task10`，也可 `L10`／`cooperative_insertion` |
| 会话 | 一个 Agent、普通 `POST /episodes`、`players=[single]` |
| 机器人 | `dual_fr3_sharpa_w01`，装配 `isaacsim_6_1_v2`，58 DOF（左右各 7 臂＋22 手指） |
| 共享驱动 | `classical_fingertip_bimanual` v1.2.0，由冻结任务的 `robot_configuration`／`configured_policy` 实例化 |
| 双手目标 | `move_geometry.hand_geometries.left/right`；分别绑定 `left_hand_flange`／`right_hand_flange` |
| 观察 | `overview/task/room/left_wrist/right_wrist` 五路 RGB-D，配套公开关节、标定、IK 和动作状态 |
| 任务时间 | Agent 使用 `--sim-budget none`，由冻结裁判执行 300 s 上限；墙钟和动作预算独立生效 |

下面使用本轮 v6 复验的完整 `rule_repairs_native_20261021/source01`，将长驻服务与继续编辑的主线隔离。该快照须从已选择 v6 的完整源码通过 `scripts.maintenance.freeze_project_sources create` 生成，启动前先执行下面的 `verify`。历史 `mechanical_task10_finish_20261021/source02` 的 Agent catalog 固定 v5，不能与 v6 服务混用，否则会被 release SHA 校验拒绝。Bench 先在一个终端启动，`TASK10_GPU_UUID` 由运行者设为本次空闲 GPU 的 UUID，`TASK10_PORT` 可改成空闲端口：

```bash
TASK10_SOURCE=<开发存储>/datasets/fr3_sharpa_bench/development/rule_repairs_native_20261021/source01
TASK10_PYTHON=<开发存储>/interface/.venv/bin/python
TASK10_RUN=<开发存储>/datasets/fr3_sharpa_bench/agent_bench/task10/$(date +%Y%m%d_%H%M%S)
TASK10_PORT=8765
: "${TASK10_GPU_UUID:?请设置本次空闲 GPU UUID}"
cd "$TASK10_SOURCE"
PYTHONDONTWRITEBYTECODE=1 "$TASK10_PYTHON" -B \
  -m scripts.maintenance.freeze_project_sources verify "$TASK10_SOURCE"
env -u ISAAC_PYTHON_LAUNCHER \
  PYTHONPATH="$TASK10_SOURCE" PYTHONDONTWRITEBYTECODE=1 \
  BENCH_TASK_RELEASE=<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/cooperative_insertion/v6 \
  BENCH_AGENT_DATA_DIR="$TASK10_RUN/episodes" BENCH_DEVICE=cuda:0 \
  CUDA_VISIBLE_DEVICES="$TASK10_GPU_UUID" ISAAC_BENCH_SINGLE_CUDA_DEVICE=1 \
  ISAAC_ALLOW_UNVALIDATED_DRIVER=1 \
  ISAAC_BENCH_EXPERIENCE="$TASK10_SOURCE/bench/apps/workroom.headless.kit" \
  "$TASK10_PYTHON" -B -m uvicorn bench.runtime.server:app \
  --host 127.0.0.1 --port "$TASK10_PORT"
```

Host 使用项目 CPU Python；`NativeWorker` 自动调用同一快照的 `scripts/isaac_python.sh`，进入共享 Isaac Sim 6.1.0。上述 driver override 与已有验证配置一致。服务收到创建请求才启动原生回合；`GET /health` 只表示 HTTP 存活。

Agent 在另一个终端使用项目现有外部凭据配置；下面是实际模型回合入口，本轮接入核对没有执行模型回合：

```bash
TASK10_SOURCE=<开发存储>/datasets/fr3_sharpa_bench/development/rule_repairs_native_20261021/source01
TASK10_AGENT_RUN=<开发存储>/datasets/fr3_sharpa_bench/agent_runs/task10/$(date +%Y%m%d_%H%M%S)
TASK10_PORT=8765
cd "$TASK10_SOURCE"
PYTHONDONTWRITEBYTECODE=1 <开发存储>/interface/.venv/bin/python -B \
  -m scripts.maintenance.freeze_project_sources verify "$TASK10_SOURCE"
PYTHONPATH="$TASK10_SOURCE" <开发存储>/interface/.venv/bin/python -B \
  -m agent.model_client.function_runner \
  --project-root "$TASK10_SOURCE" --bench-url "http://127.0.0.1:$TASK10_PORT" \
  --credentials <开发存储>/interface/configs/agent_bench/local_gateway_credentials.json \
  --family mechanical_assembly --task task10 --sim-budget none \
  --http-timeout 360 --output "$TASK10_AGENT_RUN"
```

Agent 自动选择机器人和控制器；初始化先核对 release SHA。通过 `get_robot` 读取左右手实际关节顺序、限位和坐标系，通过 `query_ik` 查询，再用同一 `move_geometry` 的两手目标执行。双手共用一个动作状态，等待完成／失败或取消反馈后再发下一动作。手指姿态或指尖目标遵循公开 guide；此处不内置抓取策略。历史 `front/rear` 仅用于 v1–v3 的独立旧服务。

此前 v5 的 Agent 驱动审计（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_task10_driver_20261021/agent_audit01.json`） 的 2 项实际 FunctionRunner／HTTP CPU 检查通过；该审计记录中的 Agent／Bench／controller 与 v5 原生 `source02` 逐字节一致。该 v5 的 原生驱动记录（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_task10_finish_20261021/http01_review03.json`） 已通过双腕同步微动、返回、保持、动作幂等和五路 RGB-D，0.8 s／1600 步。这些记录仅适用于 v5；v6 已有上述独立新复验，不继承旧版原生资格，微动也不代表完整装配。

## 历史左右手修正版 v4

该轮资产为 `cooperative_insertion/v3`，发布为 `cooperative_insertion/v4`。左臂 ready 对原右臂关节 1/3/5/7 作镜像符号变换，官方左手的手指关节沿用相同数值；两侧拇指朝向共同工作区。机械臂仍是两台原始 FR3，不需要另造一体双臂底座。原同侧独立脚座、加工外观、插销和套筒配合、物品初态及宽松规则继续保留。

设计参考 [DexMimicGen 官方示例](https://github.com/NVlabs/dexmimicgen/blob/940e8a1b3ad70eb1925ada6b364b197de6bb2af9/scripts/demo_random_action.py)：其双手灵巧任务将 `PandaDexRH` 与 `PandaDexLH` 放在同一个环境中，由一个环境 action 驱动。这里采用已有 FR3＋SHARPA 的真实左右组合与同一 Agent 动作入口；这项参考不意味着复用了它的机器人资产、控制器或验证成绩。

源码、资产和 FK 审计（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_handedness_review_20261021/single_agent01/review_summary.json`）确认真实手型与命名一致、两侧拇指向内，ready 指尖镜像误差不超过 0.95 µm；12 个单腕接近端点和 2 组同时端点通过 CPU IK 与实际 USD 保守包围盒检查。该检查没有覆盖扫掠路径、自碰撞或抓取。左右手修正不启用柔顺指垫，保留原 `sharpa_official_position_mode_v1` 以及已有接触选项。

v4 已通过自身冻结源码下的真实 TCP HTTP／Agent 检查：同一 Agent 同一 action 驱动左右手，实际上移各约 1.864 mm，返回误差小于 0.097 mm，并完成 hold 和幂等重放。0.8 秒／1600 原生步、58 DOF、五路 RGB-D，`physics_valid=true`，server／worker 正常退出，零模型调用。独立原生审计确认两侧拇指向内、58 个关节的驱动／armature 与旧版映射后一致、70 个碰撞体／10 个指垫且无柔顺参数，两物品最后 300 ms 均实际由桌面承托。没有机器人与物体接触，也没有协作得分。

物品 USD 的全部几何、质量、惯量、接触和位姿与 v2 精确一致，旧孔外入口证据通过此身份链保留其有限适用范围；本轮未重跑机器人插接。当前 manifest SHA 为 `75ba29c5324c65331bf53b65564d140b1530230f8e7c88a365d5d458bcaa79d2`，v1–v4 的 Bench 冻结发布加载检查通过，prepared 接受标记不回写。当次历史 v1–v3 Agent 接入仍需显式指定机器人；2026-10-21 六场景复核已修复此限制：连接独立旧服务时使用不带 `--family` 的 `cooperative_insertion`，客户端从该任务公开的唯一控制器描述推导 `fr3_sharpa_instances`；也可显式指定 `--robot`。有歧义则拒绝自动选择，带 `--family mechanical_assembly` 则继续严格锁定当前目录发布。见 完整验证记录（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_cooperative_handedness_20261021/qualification.json`）、左右手全景（开发存储 `demo/interface/scenes/mechanical_cooperative_handedness_20261021/overview.png`） 和 桌面近景（开发存储 `demo/interface/scenes/mechanical_cooperative_handedness_20261021/task.png`）。完整手物操作仍由数采验证。

## 2026-10-21 场景验证

初版资产 `cooperative_insertion/v1`、release `cooperative_insertion/v1` 保持不可变。`canonical03` 通过 1.2 秒 / 2400 原生子步、双原始 29 DOF 固定机器人、五路真实 RGB-D、实际桌面支撑和释放静止，`physics_valid=true`，没有初始化得分，worker 正常退出。

`drop_insert01` 将插销在 reset 时放到浅插位置，之后完全靠重力完成约 40 mm 插入（深度从 12 mm 到 52 mm），没有仿真开始后的姿态写入、外加力或手物策略。两物体通过 `peg→sleeve→collection_tray` 实际接触承托，终态稳定 300 ms；最大报告穿透为 1.06 µm。这是物物可行性证据，不是完整双臂机器人操作成功。六个各自近侧和共享中央高位的离散腕部 IK 终点通过，早期四个对侧低位局部 IK 失败保留，不据此宣称整个空间不可达。

这轮旧探针已经预插入 12 mm，只能证明孔内继续插入和承托，不能证明从孔外经过入口。完整入口补测单独记录如下。

14 项 CPU 对抗裁判/冻结身份检查通过。公开接口与整体来源绑定见 场景验证记录（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_coop_20261021/qualification.json`）。原始图片见 桌面和双机械臂（开发存储 `demo/interface/scenes/mechanical_coop_20261021/canonical03/task.png`），图片字节与原生输出一致。

### 历史裁判 release v2

历史 release `cooperative_insertion/v2` 保留原 v1 和其所有记录。v2 只修正协作进展历史：仅相邻两帧都存在两只不同机械手真实接触时才计算带方向的插入变化，接触空档中的单手运动不能在再次触碰时补记；微小往复抖动不能只累计正向位移冒充插入。允许接触中断、重抓和重新尝试，不因此判失败。后续 v3、v4 延续这些裁判要求。

16 项 CPU 检查通过。原始物理资产、reset、rules、physics profile 及 asset inventory 字节与 v1 相同，物物探针仍绑定这些原始资产；v2 的原生裁判与真实 TCP 接口检查由整体接入记录单独绑定，不能把 v1 的裁判初始化资格自动转移。见 v2 来源与检查范围（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_coop_20261021/qualification_v2_preflight.json`）。

`cooperative_http02` 已从完整冻结 `source06` 通过 v2 真实 TCP HTTP 和实际 Agent 客户端验证：普通 `/episodes` 的同一个 Agent 在同一 action 控制 `front`、`rear`，双腕各实际上移 1.864 mm，返回误差各 0.096 mm，再完成 hold 和相同 action 重放幂等检查。共 0.8 秒 / 1600 原生子步，58 个关节、五路 RGB-D；两次 Agent 只读检查确认目录 release、公开规则和模型静态上下文一致。裁判 native evidence 完整，没有双手接触、协作进度或成功误判，`physics_valid=true`；结束为 cancelled，零模型调用，server 与 worker 均正常退出。命名双手没有发布 `hand_preset`，按能力目录使用现有 `hand_posture_rad`，不假造可用预设。

完整绑定见 v2 验证记录（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_coop_20261021/qualification_v2.json`），该版原生图见 任务桌面（开发存储 `demo/interface/scenes/mechanical_coop_20261021/http_v2/task.png`）。这些检查支持场景、物物配合及双臂接口交付；手物抓取和完整协作装配由数采验证，prepared release 的接受标记不回写。

### 从孔外进入的补充验证

`mechanical_coop_entry_20261021/native03` 从插销轴向对正、尖端位于实际孔口上方 10 mm 开始。两物体保持独立动态刚体，reset 后没有姿态或速度写入、外加力或机器人。原资产、碰撞、质量、材料、求解器、规则和 release v2 全部保持不变。

6000 个连续原生子步记录实际经过孔口：约 0.045 秒进入，0.0895 秒达到 30 mm，最终深度 52.000 mm，相对行程约 62 mm。独立复核实际 USD 方杆截面在整个入口轨迹中位于孔内，最小侧向余量 1.969 mm，最大报告穿透约 0.055 mm。最后 601 帧 / 300 ms 均有插销底面→套筒底面→托盘底面的真实承托，释放静止检查通过。

首轮 `native01` 缺少单 GPU 环境开关而在仿真启动前拒绝；`native02` 的 1.2 秒测试已完成入口，但末窗插销速度/角速度尚未满足原工程稳定检查。两个失败记录均保留。`native03` 使用相同源码和物理，仅延长到 3 秒后自然落稳，没有放宽阈值。结论是**对正后可以真实插入并放稳**；不代表任意偏角都能插入，也不代表双臂已完成抓取插接。

见 完整入口验证与原始记录（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_coop_entry_20261021/qualification.json`） 和 独立逐帧几何、接触与时钟复核（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_coop_entry_20261021/independent_review01/native03_review.json`）。

### 历史同侧双右手细节版本 v3

该版选用资产 `cooperative_insertion/v2`、release `cooperative_insertion/v3`。两台 FR3 并排位于同一桌边，保留原 ready 关节和两只原始右 SHARPA；脚座距桌沿70mm，两个脚座间200mm。12个离散单腕接近点和2组同时双腕接近点通过 IK 与实际 USD 形状的保守间隙检查；低腕探索中与物品包围盒相交的样本仍保留，不当作抓取路径。

套筒增加银色加工孔口、金属外壳和握持细节，插销增加钢色方轴与黄铜色车削圆头，托盘增加折边和细纹表面。没有新增文字、编号。13个任务碰撞体、孔轴尺寸、质量、惯量、接触材质和原始物品位置完全保留；20个新增视觉网格不带物理 API。外观最大外扩约0.128mm，不封闭真实孔口。旧版从孔外实际插入的证据经明确的碰撞/物理身份比对保留适用范围，没有冒称本轮重跑该探针。

新版本通过真实 HTTP 与实际 Agent 客户端：单个 Agent 在同一 action 同时控制两腕，微动后返回并保持，共0.8秒/1600原生步、五路RGB-D，physics_valid=true，server/worker退出0。独立复核最后300ms的逐步接触，两个道具均真实承托于桌面，无意外机器人接触。结果为cancelled、无模型调用，没有抓取、协作进展或任务成功信用。规则再次明确“同一个Agent同时控制”，阈值未收紧。

本轮证据（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_cooperative_detail_20261021/qualification.json`） · 同侧全景（开发存储 `demo/interface/scenes/mechanical_cooperative_detail_20261021/overview.png`） · 物品近景（开发存储 `demo/interface/scenes/mechanical_cooperative_detail_20261021/task.png`）。旧资产、发布和失败记录保留，prepared manifest的接受状态不回写；完整手物操作交给数采。
