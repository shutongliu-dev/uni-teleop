# Operator guide

## Start and pause

| 路线 | 开始 | 暂停 / 重置 | 手部 |
|---|---|---|---|
| Franka / Isaac | GUI Start 或空格 | GUI Pause；Reset scene | 依 recipe 来自扳机、PICO 或 MANUS |
| Franka / MuJoCo | 空格 | 空格；R 重置工作台 | 同一套公共映射 |
| G1 / SONIC 共用循环 | A+B+X+Y 进入 READY，A+X 进入 POSE | 同组合切模式；客户端 reset | PICO 双手与头部 |

Franka 默认暂停。FACTR 使用保存的关节映射 `q_robot = q_reference + sign × (q_leader − leader_reference)`；不是末端 IK。PICO 手柄/裸手按腕部位姿解机器人 IK。MANUS 独立控制手指，不能替代腕部空间定位。

MuJoCo 的键盘快捷键需要原生查看器窗口获得焦点。它提供可旋转的主视角；PICO 视频桥目前是主画面复制给双眼的 mono，不是双目深度，不含 Isaac 的腕部 inset。Isaac 的多相机与任务记录流程保持原有入口。

## FACTR support

默认关闭实体电机输出。需要重力托举时，在会话 `arguments` 中显式配置：

```json
["--leader-feedback", "--leader-gravity-urdf", "/path/to/factr.urdf",
 "--leader-gravity-calibration", "/path/to/physical-gravity-calibration.json"]
```

Isaac 用“开启 / 恢复托举”按钮；MuJoCo 先空格开始，再按 F 启用/关闭。暂停和退出关闭输出。仿真目标的限位与映射标定和实体主臂重力零位是不同概念，不能用临时匹配姿态替代真实主臂模型。这里的托举/阻尼也不等于完整接触力反射。

## Diagnose the channel

- **主臂不动**：看 FACTR 读数时间、电机回应、标定身份和映射状态；不要不断重记零点。
- **手不动、臂能动**：看 MANUS SDK/optimizer 日志、手套 ID、标定和授权。两路数据独立，缺少手指数据不应把整个机械臂关掉。
- **PICO 停住**：先确认 grip、客户端输入模式与有效追踪；视频正常并不证明姿态包新鲜。
- **换后端外观/抓取不同**：机器人坐标约定共享，物理接触和模型转换不保证数值等价。MuJoCo 工作台不是 DexBench 任务评分复刻。

Franka 输入过期后不生成新目标；现有短时恢复/手柄离合逻辑在两端复用。会话日志记录实际状态和目标，不能只用“进程还在”判断遥操有效。
