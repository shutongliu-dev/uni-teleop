# 基础操作任务与评分

本页说明基础刚体任务与 v7 接触计时修复的历史契约。当前单臂默认发布为 `v12_mount_v2_r1/sharpa_w01_right`，双臂使用 `v13_pad_compliance_r2/bimanual`；现行范围与入口见[项目说明](../../README.md)。`retained_run.json` 只标识历史保留回合，不选择新的运行或自动续跑队列。

接触计时修复的历史 release 为 `v7_contact_substeps`，由 `contract_release.py` 从 `v6_public_control` 派生。它已有[四任务原生脚本正例](../../README.md)，但冻结 manifest 仍保留生成时的 `prepared_unvalidated`、`physics_accepted=false`。单任务可行性、整套发布验收和模型成绩分别记录；CPU 测试与生成成功不能代替原生验收。旧 release、场景、物理参数和既有结果保持原样，新旧评分结果不能混合统计。

## 目标与静态成功条件

以下描述 v7 基础刚体任务；各派生发布的完整条件以其冻结任务为准。目标资产是 `collab_cube_small`，任务内 ID 保留 `cube`。它实际为长方体，物体本地 X/Y/Z 方向尺寸约 **32 × 40 × 60 mm**，质量约 0.05 kg；这些值来自共享资产目录的版本化 manifest。资源路径为 `<开发存储>/assets/objects/hand_system0/bench_catalog/v1`。v7 公开描述从已验证 manifest 的边界读取尺寸，不改缩放、质量或碰撞几何。

`01_cube_push` 要求将目标沿桌面推向世界坐标 x=0.45、y=0.20 m 附近，并移开整台机器人。终态同时要求：

- XY 距离不超过 0.03 m，高度与公开目标相差不超过 0.006 m。
- 线速度不超过 0.04 m/s，角速度不超过 0.3 rad/s，连续保持 0.30 s。
- 倾角不超过 0.7 rad（约 40.1°）。倾角是**物体本地正 Z 轴与世界正 Z 轴的夹角**；绕竖直轴的转向本身不计入倾角。
- 有求解器报告的桌面支撑，且机器人与目标脱离。
- 累计手部接触至少 0.04 s，分布于至少两个控制窗口；最大位移至少 0.05 m，全回合抬升不超过 0.04 m。
- 机械臂本体接触目标是不可撤销的违规。持续不合规的抓握候选也保留违规；单指推移不要求对指抓握。

这些阈值没有因 v7 接触计时修复而改变。到达目标 XY 但倾倒的物块仍不满足任务。

其他保留的模式为 `lift_hold`、`pick_place`、`grasp_hold`。抬起与搬运需要先建立合规抓持并保持携带阶段的连续抓持；桌面抓稳不要求抬升。各任务准确阈值以冻结 task 的公开 `success_contract` 为准，不根据任务名称推断。

## 接触采样与私有评分

v7 release 冻结的是 `simple_manipulation_contact_substeps_v4`，公开契约为 `simple_manipulation_success_v2`。后续 v8 r2 / 单臂 v12 冻结评分器为 v5，并为 `grasp_hold` 增加承重夹持的 v3 契约；双臂 v13 r2 冻结评分器为 v6。当前源码 [evaluator.py](evaluator.py) 是 v7，增加了软体窗口静止判定，不能据此重标历史运行版本。执行时必须使用所选 release 的冻结评分器与任务 SHA，详见[失真修复](../../docs/agent_bench/fidelity_repair.md)与[软体说明](../../docs/agent_bench/soft_bodies.md)。实际训练或模型客户端不能提交评分证据；可信原生 worker 提供私有观测，评分与公共 Agent 观测分开。

接触时长按“具有正手部法向冲量的唯一物理子步数 × 冻结 physics_dt_s”计算。默认 2 kHz 物理频率下，两个控制窗口中各出现一个 0.5 ms 接触子步，合计为 **1 ms**。同一子步多个手指接触不会增加接触时长；不同接触对的真实冲量可以相加。

原生环境逐次记录实际成功采集次数，包括无接触子步，并记录窗口起止公共时钟、原生时钟和原生步号。评分器要求完整且连续的窗口，拒绝重复接触对、错位时间戳、缺采、观测断档或时钟不一致。缺失证据不补零，也不补成完整窗口。

历史 v1 公开契约仍可投影。旧任务缺少新采样配置时须使用其冻结评分器，不能配新评分器沿用旧的时长估计。

## 候选派生与开发入口

以下只展示候选派生方法，不是当前默认发布的准备步骤，也不会复现历史 v7 的字节；生成器会复制当前源码评分器。必须使用尚不存在的新输出路径，保存源码版本后单独验收：

```bash
.venv/bin/python -B -m scenes.manipulation.contract_release \
  --source <开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/v6_public_control \
  --output <开发存储>/datasets/fr3_sharpa_bench/development/contact_substeps_candidate_example
```

输出已存在时拒绝覆盖。该入口只派生评分和静态公开契约，保留源场景、资产及物理 profile；新评分器需配套具有采样摘要的原生环境。`release.py` 是从基础场景生成任务的另一个入口，其默认输出为 `v7_contact_substeps_base`，不要将它误认为上述保留 curated 场景的 v7 派生 release。

`probe.py` 与 `controller.py` 提供可信工程诊断，控制器可读取私有物体状态，不能作为视觉模型 Agent 的成绩。HTTP 链路诊断入口为 `scripts.validation.validate_simple_bench`；对新评分必须显式指定新 release。单次模型回合入口为项目根目录的 `scripts/run_push.py`，需要独立启动且配置匹配的服务。

已有记录的媒体导出：

```bash
.venv/bin/python -B -m scripts.media.package_simple_bench \
  --run /path/to/completed/run --output <开发存储>/demo/interface/controller_baselines/example/video.mp4
```

媒体保留真实物理时间与关联 sidecar。不要以画面或退出码代替独立物理审计。

轻量回归：

```bash
.venv/bin/python -B -m pytest -p no:cacheprovider \
  tests/test_simple_tasks.py tests/test_success_contract.py \
  tests/test_humanlike_grasp.py tests/test_contact_sampling_contract.py
```

这些测试验证逻辑和采样契约，不启动 Isaac，也不能替代真实原生接触验收。
