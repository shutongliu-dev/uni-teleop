# Task8 · NIST ATB3 线缆绕障布线

沿电缆从黑色方块端向细长音频插头端检查，最终路径必须依次穿过绿色导向管、黄色（琥珀色）导向管，音频插头插入末端金属圆孔接口。黑色方块端是可移动线尾，只定义沿线检查方向，不要求固定在某个位置；任务不额外要求绕两根银色柱子。

这是最终布线路径的顺序，实际穿线、插接和调整的操作先后可以自由选择。允许重抓、退出重插及重新布线，末尾松手稳定 0.3 秒。

新增重点是单根线缆绕障，不重复单独的接头装配考核。 这是根据 NIST 原始资料改编的去重任务，不声称完整复现官方板件或评测协议。原机械 Task1–5 保持独立。

- 场景入口：`task.py`；公开规则：`rules.py`；裁判入口：`evaluator.py`；发布指针：`definition.py`。
- 复用固定机械室、原 FR3＋右 SHARPA 和原支架；物件、工装、初始化与裁判属于本任务。
- 素材存放共享 assets，冻结 release 和实验存放共享 datasets，不在任务源码目录堆放模型。
- 接口：普通 `/episodes`；Agent 选择 `--family mechanical_assembly --task task8`、`L8` 或 `nist_cable_routing`。
- 允许重抓、退出重插、错放纠正及重新布线。达到目标后需要真实支撑、脱离机器人接触并稳定 0.3 秒。
- 预算耗尽记未完成；证据缺失不制造成功或操作失败；初始化不计任务分。

柔性件使用被动关节连接的短段刚体近似，具有真实重力和碰撞；不宣称材料弯曲精度。连接器只评估几何插接与支撑，不宣称电气导通或卡扣锁紧。手物交互交由数采，本轮只承担场景和物物可行性。

当前使用 `usability_v2`：端头放大 1.6 倍、线径 5.25 mm、导管放大 1.25 倍；路线几何门随真实孔径同步为 11.25 mm，原稳定与接触阈值保留。实际目标配合、整线导向及释放稳定通过原生物物检查。 三种构型可在开局选择；完整抓取、布线／拆除仍由数采验证。

各版本的实际原生证据和限制见 [NIST 任务说明](../../NIST_TASKS.md)。

来源：[NIST Assembly Performance Metrics and Test Methods](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly)。原文件、版本、单位和哈希随资产 provenance 保留；具体变换和近似以资产 metadata 为准。

## 公开路线修复

当前选用 `desktop_v2`（由新规则基础版 v4 派生，继续使用资产 v5）。新冻结公开规则在 `instruction_zh` 和 `referee.success_conditions` 中明确起端、颜色顺序及插头目标，经 `/tasks`、创建回包、`get_task` 与 Agent 静态上下文传递。裁判原有路线方向、几何、物理、阈值和机器人保持不变；原 `desktop_v1` 的笼统说明作为历史保留。见 修复及接入证据（开发存储 `datasets/fr3_sharpa_bench/development/mechanical_routing_public_20261021/qualification.json`）。
