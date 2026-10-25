# 首次集成验证 · 2026-10-25

## 统一入口

- 7 项入口测试通过，覆盖所有 12 个 profile 的命令构建、缺少配置、带空格路径、
  参数透传、PICO+MANUS 入口选择及旧 Isaac 源码环境清理。
- `doctor` 已检查本机 FACTR+MANUS、PICO+MANUS、SONIC/PICO 所需配置路径存在。
  这不代表设备连接、许可或实际采样已完成验证。
- 1,402 个导入文件通过 SHA-256 来源校验；两处解释器路径适配显式记录。
- 1,196 个 Python 文件语法解析通过；入口 shell 语法检查通过。
- Isaac 完整源码冻结检查通过，生成 719 个快照文件；未在本次打包中另开 Isaac GUI。

## DexBench

PICO、MANUS、主从臂、托举、接触精度与接触采样的定向回归：首轮 **240 通过、1 失败**。
失败为 `test_owner_gil_stall_keeps_output_and_owner_death_disables` 的进程退出时序断言。
新仓库与原仓库分别单独复测该项均通过；新仓库完整反馈测试文件复测 **23 通过**。
保留首轮失败事实，不能据此宣称退出时序在任意系统负载下都稳定。

未重跑 DexBench 全量套件；历史有 12 项既存全量失败，不能将定向结果描述为全量全绿。
此次包装未改变物理和控制函数，也未进行实体电机使能或实机遥操验收。

## SONIC-MuJoCo

通过统一 CLI，从新源码运行 G1 与 H2 各 100 个无窗口仿真步，均正常退出。
G1 推进 0.500 s，H2 推进 0.200 s。这是模型加载/步进检查，不是全身平衡、遥操或任务成功。

原 unittest 套件首次运行 265 项：现场 `lab_scan` 的两个子用例缺少外部碰撞 manifest，
另有 2 项按原测试条件跳过。显式指定已有外部重建 manifest 后，受影响测试文件
5 项全部通过；随后完整重跑 **265 项完成、2 项跳过、无失败**。现场重建资产仍不打包。

本机测试过程输出保留在本机临时日志，不上传 SDK 日志、现场标定或数据集。
没有宣称全部 12 个入口均已在这次集成中完成真实设备测试。

---

# 结构重构与跨后端补全 · 2026-10-25

上述首次快照结果保留为历史。当前代码已将设备/映射抽取为公共模块，不能再用原始文件 hash 不变来说明当前实现未改；原 hash 保存在 `sources.lock.json`，迁移对应关系在 `docs/provenance/module-migration.json`，当前文件清单在 `integration.lock.json`。

| 检查 | 本次结果 | 证明范围 |
|---|---|---|
| 统一 CLI | 9 项通过 | 所有 recipe 命令、最小配置、双端设备矩阵、路径与环境隔离 |
| MuJoCo Franka 原生 | 7 项通过 | 同源 FK、动力学关节驱动，以及六种设备路线的合成输入→目标→原生步进 |
| 共享 PICO/FACTR/MANUS/反馈定向回归 | 194 项通过；1 项旧 mock 路径失败，迁移 mock 后该项单独复测通过 | 公共逻辑、采集/恢复、标定/反馈边界；非完整 DexBench 全量套件 |
| SONIC 原 unittest | 265 项运行，2 跳过，无失败 | 使用外部 lab_scan manifest；未提供它时首轮有 2 个场景子用例报缺文件 |
| Isaac Franka 原生 | 100 步合成控制器检查通过 | 实际最大关节变化 0.1241 rad；断流保持、恢复逻辑、非空双目与视差 |
| Isaac G1 原生 | 20 物理步、71 轴、有限状态；994 维 SONIC observation→29 轴 command 通过 | 原生模型/状态/控制接口；使用零输出合成 policy，非训练策略平衡验收 |
| Python 语法 | 1,224 个文件解析通过 | 语法，不代表运行时全部模块执行过 |
| README 图 | SVG 渲染并检查，MuJoCo 截图原生生成 | 图文布局和来源；非实机验收 |

Franka 模型与公共运动学的最大分量差约 `1.29e-6`（位置以米、旋转矩阵无量纲分别比较）。原生 MuJoCo 位置驱动测试的第一轴实际运动约 `0.0998 rad`。动态接触导致实际关节不必逐帧等于目标；这是驱动仿真，不是逐帧覆盖 qpos。

原生 Isaac 测试从包含公共包的冻结快照启动。G1 导入器生成的静态/运动学相机固定关节已在适配层过滤，避免 PhysX 对两个非动态物体建关节的错误。机器人动态关节保持。

## 可重复执行

```bash
python -m unittest discover -s tests
PYTHONPATH="$PWD:$PWD/backends/isaac:$PWD/backends/isaac/tests" \
  DEXBENCH_DATA=/path/to/data .venv-mujoco/bin/python -m pytest tests/native -q
PYTHONPATH="$PWD:$PWD/backends/mujoco" \
  /path/to/isaac/python.sh tools/check_isaac_g1.py --output /tmp/uni-g1-check
python tools/verify_sources.py
```

## 尚未验证 / 尚未迁移

- 本次未读取实体 FACTR、PICO 或 MANUS，也未使能实体电机。新增跨后端路线仍需真实设备联调。
- Isaac G1 尚未验证真实 SONIC 权重的长期站立、移动、接触和全身遥操稳定性；标为实验性。
- MuJoCo Franka 是工作台练习场景，未迁移 DexBench 正式任务资产、评分、柔性体或完整接触证据链。
- MuJoCo Franka 视频为复制 mono；Isaac 多视角、原 MuJoCo SONIC 场景 RGB-D/录像能力没有虚称为新路线的共同能力。
- 历史 DexBench 12 项基线失败仍保留；本次没有跑全量套件或宣称全量全绿。
