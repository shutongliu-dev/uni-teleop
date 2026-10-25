# DexBench v1

在 Isaac Sim / PhysX 中评测模型用 Franka FR3 + SHARPA 灵巧手完成桌面操作任务。六个固定房间、每个房间若干任务；模型只通过公开 HTTP 函数（观测、相机、只读 IK、机器人动作）交互，裁判在仿真内按原生子步评分。

代码分三层，边界见 [AGENTS.md](AGENTS.md) 与 [STRUCTURE.md](STRUCTURE.md)：

| 目录 | 职责 |
|---|---|
| `bench/` | 通用执行：HTTP 服务、协议、控制器、传感、物理审计、Isaac 后端 |
| `scenes/` | 固定房间与任务：物体、机器人摆位、初态、规则、裁判、任务发布选择 |
| `agent/` | 模型客户端：公开函数工具、会话编排、学习模块 |
| `tests/` | CPU 测试，不启动仿真器 |
| `scripts/` | 资产构建、原生验证、媒体与维护工具 |
| `docs/` | 设计、协议与历史证据说明 |

Git 只同步上面这些源码。以下目录每人在本机准备，内容不进 Git：

| 目录 | 内容 | 说明 |
|---|---|---|
| `data/` | 数据包：房间、机器人、物体与任务发布，约 2 GB，从 GitHub Release 下载；Git 只跟踪 `data/MANIFEST.json` | [data/README.md](data/README.md) |
| `runtime/` | 本机 Isaac Sim 6.1 与 Python 3.12 环境的链接 | [runtime/README.md](runtime/README.md) |
| `outputs/` | 实验输出、渲染媒体等，自动创建 | 可用 `DEXBENCH_OUTPUT` 改到别处 |

代码和数据都不含机器路径：代码只通过 `bench.paths` 找数据根目录（`DEXBENCH_DATA`，默认 `data/`）和输出根目录（`DEXBENCH_OUTPUT`，默认 `outputs/`）；仿真器只通过 `runtime/isaacsim` 或 `ISAAC_PYTHON_LAUNCHER` 找到。

## 快速开始

```bash
git clone https://github.com/VVVVME50/dexbench.git && cd dexbench   # 私有仓库，需先被邀请为协作者
runtime/setup.sh --isaacsim <Isaac Sim 6.1 目录> --venv <Python 3.12 环境>
# 下载并解压数据包到 data/，见 data/README.md
.venv/bin/python -m scripts.data.package check               # 逐文件校验数据包
.venv/bin/python -B -m scripts.maintenance.parallel_pytest   # 全套 CPU 测试，按文件并行
```

全套 CPU 测试约 7950 项：`parallel_pytest` 按文件并行（默认 32 个进程，可用 `--jobs` 加大），仓库在本地磁盘上约 5 分钟；放在网络盘上会慢一倍以上，因为部分测试要复制完整源码快照。直接 `pytest` 单进程超过 1 小时。日常改动可先跑受影响的文件，例如 `.venv/bin/python -B -m pytest tests/test_scene_function_runner.py`，合入前跑全套。5 项测试只在 Isaac 自带的 Python 里运行，在 CPU 环境中会被跳过。

查看任务（不启动仿真）：

```bash
.venv/bin/python -m scenes.workrooms tasks --include-planned
.venv/bin/python -m scenes.workrooms task --family kitchen --task breakfast_serving
.venv/bin/python -m agent.model_client.scene_tasks --family living_room
```

启动服务：每个服务实例只绑定一个任务发布（路径见下方任务表，相对于 `data/`），原生 worker 在创建回合时才启动；回合记录默认写到 `outputs/agent_bench/episodes`。

```bash
BENCH_TASK_RELEASE=$PWD/data/datasets/fr3_sharpa_bench/isaacsim/releases/kitchen/breakfast_serving/v2 \
BENCH_DEVICE=cuda:0 CUDA_VISIBLE_DEVICES=<GPU UUID> ISAAC_BENCH_SINGLE_CUDA_DEVICE=1 \
  .venv/bin/python -B -m uvicorn bench.runtime.server:app --host 127.0.0.1 --port 8000
```

运行 Agent：

```bash
.venv/bin/python -B -m agent.model_client.function_runner \
  --bench-url http://127.0.0.1:8000 --family kitchen --task task1 \
  --credentials <快照外的凭据配置，模板见 configs/agent_bench/> \
  --output $PWD/outputs/<实验名>/client
```

人工使用 PICO 裸手或手柄遥操 Isaac 场景，见 [PICO 遥操](docs/pico_teleop.md)。

## 任务状态

下表由 `python -m scripts.maintenance.task_status --write README.md` 从任务目录生成，请勿手改；`tests/test_repository_status.py` 会拦截不一致。

<!-- task-status:begin -->
| 场景 | 编号 | task_id | 名称 | 状态 | 会话 | 任务发布（相对 data/） |
|---|---|---|---|---|---|---|
| `mechanical_assembly` | L1 | `parts_sorting` | L1 · 大零件分类 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/development/mechanical_referee_20261018/rules_v1/releases/parts_sorting` |
| `mechanical_assembly` | L2 | `peg_insertion` | L2 · 粗销浅孔放置 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/development/mechanical_referee_20261018/rules_v1/releases/peg_insertion` |
| `mechanical_assembly` | L3 | `shaft_spacer_assembly` | L3 · 轴上隔套顺序装配 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261019/shaft_spacer_candidate_v2` |
| `mechanical_assembly` | L4 | `fastener_assembly` | L4 · 散装螺栓装配 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/releases/loose_bolt_candidate_v10_pgs` |
| `mechanical_assembly` | L5 | `gear_train_assembly` | L5 · 多齿轮装配与自主选工具 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/development/mechanical_gear_scene_validation_20261020/practical_rules_v4/release` |
| `mechanical_assembly` | L6 | `nist_connector_insertion` | Task6 · NIST ATB1 · 电气接头插接 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_connector_insertion/v5` |
| `mechanical_assembly` | L7 | `nist_belt_drive_assembly` | Task7 · NIST ATB2 · 滑轮皮带装配 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_belt_drive_assembly/v2` |
| `mechanical_assembly` | L8 | `nist_cable_routing` | Task8 · NIST ATB3 · 线缆绕障布线 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_cable_routing/v3` |
| `mechanical_assembly` | L9 | `nist_wire_harness` | Task9 · NIST ATB4 · 多线束装配与取下 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_wire_harness/v3` |
| `laboratory` | task1 | `sample_sorting` | Task1 · 样本分类 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/sample_sorting/v3` |
| `laboratory` | task2 | `vial_racking` | Task2 · 试管扶正入架 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/vial_racking/v4` |
| `laboratory` | task3 | `cap_placement` | Task3 · 样本瓶开盖与复盖 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/cap_placement/v4` |
| `laboratory` | task4 | `solid_weighing` | Task4 · 药匙定量称取颗粒 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/solid_weighing/v4` |
| `laboratory` | task5 | `pipette_transfer` | Task5 · 跨台定量移液分装 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/pipette_transfer/v2` |
| `kitchen` | task1 | `breakfast_serving` | Task1 · 早餐配盘与整盘端送 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/kitchen/breakfast_serving/v3` |
| `kitchen` | task2 | `produce_transfer` | Task2 · 果蔬分类装盘 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/kitchen/produce_transfer/v3` |
| `kitchen` | task3 | `utensil_sorting` | Task3 · 餐具分类收纳 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/kitchen/utensil_sorting/v3` |
| `kitchen` | task4 | `pot_lid_placement` | Task4 · 冷灶锅具就位与锅盖盖合 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/kitchen/pot_lid_placement/v4` |
| `kitchen` | task5 | `pantry_stocking` | Task5 · 调味品与干货分类归位 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/kitchen/pantry_stocking/v2` |
| `kitchen` | task6 | `table_clearing` | Task6 · 餐后餐桌分类清理 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/kitchen/table_clearing/v2` |
| `study` | task1 | `stationery_sorting` | Task1 · 文具分类 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/study/stationery_sorting/v4` |
| `study` | task2 | `book_stacking` | Task2 · 按序叠书 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/study/book_stacking/v3` |
| `study` | task3 | `book_retrieval` | Task3 · 指定书取出 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/study/book_retrieval/v3` |
| `study` | task4 | `book_shelving` | Task4 · 按类别上架 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/study/book_shelving/v3` |
| `card_room` | task0 | `task0` | Task0 · 单臂随机三子摆放 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/card_room/task0/v5` |
| `card_room` | task1 | `tic_tac_toe_3x3` | L1 · 木质三子棋 3×3 | implemented | `/board-games` | `datasets/fr3_sharpa_bench/isaacsim/releases/board_games/tic_tac_toe_3x3/v4` |
| `card_room` | task2 | `connect_four_5x5` | L2 · 木质平面四连棋 5×5 | implemented | `/board-games` | `datasets/fr3_sharpa_bench/isaacsim/releases/board_games/connect_four_5x5/v4` |
| `card_room` | task3 | `hex7x7` | L3 · 木质 Hex 7×7 | implemented | `/board-games` | `datasets/fr3_sharpa_bench/isaacsim/releases/board_games/hex7x7/v11` |
| `living_room` | task1 | `remote_operation` | Task1 · 遥控电视开关机 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/living_room/remote_operation/v7` |
| `living_room` | task2 | `water_pouring` | Task2 · 单臂水壶倒水 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/living_room/water_pouring/v5` |
| `living_room` | task3 | `tower_building` | Task3 · 单臂搭积木塔 | implemented | `/episodes` | `datasets/fr3_sharpa_bench/isaacsim/releases/living_room/tower_building/v5` |
| `living_room` | task4 | `tower_extraction` | Task4 · 双臂轮流抽塔对抗 | implemented | `/turn-games` | `datasets/fr3_sharpa_bench/isaacsim/releases/living_room/tower_extraction/v8` |
<!-- task-status:end -->

`implemented` 表示有任务发布、规则和裁判，可以创建回合，**不表示机器人已经能完成这个任务**：目前只验证过初始化、承托、相机和短时保持／微动，完整抓放、装配、倒水、端送等还没有跑通。`planned` 任务不可运行。

| 会话 | 用途 |
|---|---|
| `/episodes` | 普通单人任务 |
| `/board-games` | 棋牌室双人对弈，玩家需认证 |
| `/turn-games` | 客厅轮流抽塔，玩家需认证 |

## 文档

| 主题 | 入口 |
|---|---|
| 协作流程与分工 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| HTTP 协议、观测、动作 | [bench/runtime/README.md](bench/runtime/README.md) |
| 场景接入通用能力 | [docs/agent_bench/scene_integration.md](docs/agent_bench/scene_integration.md) |
| 模型客户端与场景入口 | [agent/model_client/README.md](agent/model_client/README.md) |
| 房间与任务目录 | [scenes/workrooms/README.md](scenes/workrooms/README.md) |
| 控制器 | [bench/control/README.md](bench/control/README.md) |
| 来源与历史记录 | [PROVENANCE.md](PROVENANCE.md)、[docs/history/](docs/history/) |
