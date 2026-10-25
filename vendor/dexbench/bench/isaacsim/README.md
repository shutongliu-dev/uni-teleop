# 原生 Isaac 执行与独立审计

原生后端使用 Isaac Sim 6.1、PhysX GPU 和 RTX。默认机器人为 FR3 七关节与右 SHARPA Wave/W01 二十二关节；当前建设范围还包括双臂 FR3 + SHARPA，其他已登记手型的集成暂缓。HTTP 服务 `bench.runtime.server` 使用同一原生 Isaac 后端。MANUS 硬件未连接，已接收的抓握 student 未注册到此服务；独立的 [System0/System1 工程](../../system0-system1/README.md)有自己的接入记录。

本目录保留顺序任务执行、冻结 release 加载、轨迹记录、视频与独立审计。HTTP 单臂默认发布为 `v12_mount_v2_r1/sharpa_w01_right`（刚性指垫），双臂层为 `v13_pad_compliance_r2/bimanual`（柔顺指垫）；两者显式选择、独立记录。基础任务见[基础任务说明](../../scenes/manipulation/README.md)，当前主流程通过 `bench.runtime.worker` 调用原生后端。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `release.py` | 冻结输入校验、按完整依赖加载独立评分器、读取 task/scene/profile。其 `DEFAULT` 是 `sequential_workcell/v1`，与 HTTP 的 `scenes.manipulation.DEFAULT_TASK_RELEASE` 分属不同任务族。 |
| `run.py`、`suite.py`、`contact_probe.py` | 顺序任务与接触工程诊断的显式 CLI；保留独立的 nominal、reset-only qualification 和接触对照语义。当前单推块入口不调用整套 suite。 |
| `controllers.py` | 原生诊断的关节目标控制与接触摘要；也由基础任务工程控制器引用。 |
| `curated_release.py` | 在新 release 中整理场景实例，逐字保留源 task、评分器和物理 profile；内置 v4→v5 是历史派生关系，不是当前评分默认。 |
| `audit.py` | 回放记录、哈希、原生时钟、接触与几何证据的独立审计；部分公共函数也由 Agent 物理审计调用。 |
| `video.py` | 对真实记录生成画面与视频，不用物体轨迹回写仿真。 |

`sequence_evaluator.py` 文件名是 release 加载接口约定：基础任务 release 同样将自己的评分器冻结为该名称。这个名字不表示基础任务使用了顺序任务的评分逻辑。

## 运行边界

原生解释器为项目根目录的 `scripts/isaac_python.sh`。`.venv/bin/python` 仅供 API、调度与轻量检查；不向共享环境同步 Isaac 依赖，也不依据环境历史目录名判断当前仿真后端。

当前主机启动原生诊断仍需显式 `ISAAC_ALLOW_UNVALIDATED_DRIVER=1`，该值必须记录在元数据中，不能称为厂商支持的驱动配置。按 UUID/PCI 地址选择已核实可渲染且资源足够的 GPU，不使用终端或容器的固定数字编号。使用 `CUDA_VISIBLE_DEVICES=<已核实的完整GPU UUID>` 与 `ISAAC_BENCH_SINGLE_CUDA_DEVICE=1` 时，原生参数 `--device cuda:0` 才表示该唯一可见设备；记录并核对实际 UUID。已知有渲染故障的 BB/DB 设备不用于 Isaac 或 EGL，不终止其他用户进程，也不改共享驱动。

`env.step()` 只接收完整机器人关节目标；物体与布料状态只能在 reset 初始化。当前标准物理 profile 使用 0.5 ms 子步（2 kHz），控制周期为 20 ms；运行证据以所选 profile 和原生元数据为准，两种时钟独立检查。刚体接触来自场景绑定的 PhysX 张量，法向冲量不等于完整接触力或硬件触觉。布料接触力仍不可用；已有布料几何、碰撞与逐子步观察工具保留其证据范围。

原生 rollout、批量数据与正式评测使用 GPU。CPU 几何检查、有限数组或退出码只能说明相应软件检查通过。

## 资源与记录

- 仿真运行时、机器人与物体资产位于 `<开发存储>/assets`。
- release、原始记录、派生数据和完整源码快照位于 `<开发存储>/datasets/fr3_sharpa_bench`。
- 展示视频位于 `<开发存储>/demo/interface`，按 `agent_demos`、`bench_comparisons`、`controller_baselines`、`scenes`、`archive` 归类，sidecar 与视频放在一起。

旧视频路径通过存储迁移清单解释，不修改历史 JSON。长时间运行必须使用完整源码快照，并校验实际导入路径；单独复制几份脚本不能完成隔离。

现有记录的渲染入口为 `scripts.media.render_isaac_demo`；基础任务回合导出入口为 `scripts.media.package_simple_bench`。工程验证入口位于 `scripts/validation/`，资产与静态诊断入口位于 `scripts/assets/`、`scripts/diagnostics/`。所有输出选择新目录。已有零模型特权工程基线；各任务的原生记录及范围见[项目说明](../../README.md)与[机器人本体](../../docs/agent_bench/embodiments.md)。新的模型 Agent 测试仍暂缓，单项成功不等于整套 release 验收，静置或渲染通过也不等于操作成功。

## 保留的离线参考工具

`bench.data_tools.reference` 读取并校验 HOI/OakInk2 数值参考；`bench.data_tools.prepare_motion` 通过 `bench.control.keypoint_retarget` 生成完整的运动学关节目标。它们不写物体状态，运动学可达性也不代表操作成功。

`bench.control.retarget.MotionReferenceController` 保留为 MANO 四元数幅度的轻量替代基线；它与使用已解码关节点的 `bench.control.keypoint_retarget` 不同。`bench.evaluator` 提供显式传入完整轨迹的离线刚体/布料参考指标，不由当前原生 Agent worker 调用，也不具备 v7 的逐子步接触采样证明。重用旧指标时必须保留其输入语义与版本，不能将其布尔接触时长用于宣称当前 bench 的成功。
