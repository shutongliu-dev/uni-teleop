# Task7 · NIST ATB2 滑轮皮带装配

将带轮及独立张紧惰轮装到对应轴上，将柔性闭环皮带套到带轮和张紧区域。按几何包围、合理长度、实际支撑及释放稳定判定；无需精确张力或传动效率。

删除与 Task3、Task4 重复的套环和紧固操作。 这是根据 NIST 原始资料改编的去重任务，不声称完整复现官方板件或评测协议。原机械 Task1–5 保持独立。

- 场景入口：`task.py`；公开规则：`rules.py`；裁判入口：`evaluator.py`；发布指针：`definition.py`。
- 复用固定机械室、原 FR3＋右 SHARPA 和原支架；物件、工装、初始化与裁判属于本任务。
- 素材存放共享 assets，冻结 release 和实验存放共享 datasets，不在任务源码目录堆放模型。
- 接口：普通 `/episodes`；Agent 选择 `--family mechanical_assembly --task task7`、`L7` 或 `nist_belt_drive_assembly`。
- 允许重抓、退出重插、错放纠正及重新布线。达到目标后需要真实支撑、脱离机器人接触并稳定 0.3 秒。
- 预算耗尽记未完成；证据缺失不制造成功或操作失败；初始化不计任务分。

柔性件使用被动关节连接的短段刚体近似，具有真实重力和碰撞；不宣称材料弯曲精度。连接器只评估几何插接与支撑，不宣称电气导通或卡扣锁紧。手物交互交由数采，本轮只承担场景和物物可行性。

各版本的实际原生证据和限制见 [NIST 任务说明](../../NIST_TASKS.md)。

来源：[NIST Assembly Performance Metrics and Test Methods](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly)。原文件、版本、单位和哈希随资产 provenance 保留；具体变换和近似以资产 metadata 为准。
