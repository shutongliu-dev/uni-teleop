# Operator guide

## Start and pause

| 路线 | 开始 | 暂停 / 重置 | 手部 |
|---|---|---|---|
| Franka / Isaac | GUI Start 或空格 | GUI Pause；Reset scene | 依 recipe 来自扳机、PICO 或 MANUS |
| Franka / MuJoCo | 空格 | 空格；R 重置工作台 | 同一套公共映射 |
| G1 / SONIC 共用循环 | A+B+X+Y 进入 READY，A+X 进入 POSE | 同组合切模式；客户端 reset | PICO 双手与头部 |

Franka 默认暂停。FACTR 使用保存的关节映射 `q_robot = q_reference + sign × (q_leader − leader_reference)`；不是末端 IK。PICO 手柄/裸手按腕部位姿解机器人 IK。MANUS 独立控制手指，不能替代腕部空间定位。

MuJoCo 的键盘快捷键需要原生查看器窗口获得焦点。它提供可旋转的主视角；PICO 视频桥目前是主画面复制给双眼的 mono，不是双目深度，不含 Isaac 的腕部 inset。Isaac 的多相机与任务记录流程保持原有入口。

Isaac 可追加 `--operator-view rear`：头显使用机器人后上方偏肩侧的独立双目画面，不带腕部小窗；桌面预览复用头显主画面；原场景相机和 review 图像仍走原配置，不记录新增操作者相机。PICO 每次 Start 用当前头显朝向重新标定手柄方向：右/上对应画面右/上，前推对应沿视线向前，旋转增量使用同一坐标系。正对画面按 Start，再握住 Grip 建立末端参考；松开 Grip 可调整持握姿势，再握住不会跳变。`session.json` 的 `operator_view` 记录本次显示选择。

## FACTR support

默认关闭实体电机输出。需要重力托举时，在会话 `arguments` 中显式配置：

```json
["--leader-feedback", "--leader-gravity-urdf", "/path/to/factr.urdf",
 "--leader-gravity-calibration", "/path/to/physical-gravity-calibration.json"]
```

Isaac 用“开启 / 恢复托举”按钮；MuJoCo 先空格开始，再按 F 启用/关闭。暂停和退出关闭输出。仿真目标的限位与映射标定和实体主臂重力零位是不同概念，不能用临时匹配姿态替代真实主臂模型。这里的托举/阻尼也不等于完整接触力反射。

## Diagnose the channel

### PICO 脚踏离合

`isaac-pico-controllers-pedal` 使用本机配置的 `PEDAL_EVENT` evdev 路径，默认识别 F3（Linux key code 61）。运行时独占这一个踏板输入，不依赖桌面窗口焦点。普通手柄 recipe 不打开踏板。

先开启 Head、Controller、Send，并完成视角标定。每踩一下切换暂停/继续；可以一直按住手柄 Grip。暂停后收回手，下一次踩踏时重新建立当前手柄与末端参考，保留视角坐标映射和机器人当前目标，不跳回初始位姿。触发器和手指目标也随暂停保持。踏板不承担重置功能；重置使用窗口 Reset scene 或键盘 R。按住踏板不会连续切换。

追踪无效或尚未完成头部标定时不允许踏板继续；关节限制、碰撞保护和场景时限仍生效。踏板断开或输入事件丢失时结束会话，不自动恢复。踏板是遥操离合，不是硬件急停。

Isaac 后上方视角可追加 `--operator-inset table`，在头显主画面下方的独立区域显示桌面交互特写，完整主画面等比缩小，不被小窗覆盖。相机保持主视角的水平方向，位于桌面上方约 16 cm，向下约 4°，近乎平行桌面；两眼共用特写，不会替换或增加 episode 的记录图像。

- **主臂不动**：看 FACTR 读数时间、电机回应、标定身份和映射状态；不要不断重记零点。
- **手不动、臂能动**：看 MANUS SDK/optimizer 日志、手套 ID、标定和授权。两路数据独立，缺少手指数据不应把整个机械臂关掉。
- **PICO 停住**：先确认 grip、客户端输入模式与有效追踪；视频正常并不证明姿态包新鲜。
- **PICO 数据仍到达但 SDK 无帧**：如果服务同时报告 `device missing`，检查独立心跳超时误判，见 [本地服务修复说明](provenance/xrobotoolkit-liveness.md)。不要关闭姿态过期保护。
- **换后端外观/抓取不同**：机器人坐标约定共享，物理接触和模型转换不保证数值等价。MuJoCo 工作台不是 DexBench 任务评分复刻。

Franka 输入过期后不生成新目标；现有短时恢复/手柄离合逻辑在两端复用。会话日志记录实际状态和目标，不能只用“进程还在”判断遥操有效。

### 按需渲染与采样频率

未启用 `--review-dir` 时，只渲染头显所需相机（后上方双目，以及启用时的桌面特写）。启用 review 记录后，原采集相机只在图像采样到期且上一张已写完时渲染；默认目标 **15 Hz**，可用 `--record-fps 20` 调整（范围大于 0、至多 50）。桌面窗口复用头显图像；显式 `--rear-camera` 仍会增加一台桌面相机。

采样按主机单调时钟调度；渲染或写盘跟不上时跳过采样，不重复旧图补帧。`review_images.jsonl` 保存每张图对应的主机时间与仿真时间，`session.json` 的 `review_image_sampling` 报告目标与实际平均帧率。15 Hz 是目标，不是性能保证；提高采集频率会增加渲染负载。原有 review 记录仍为原 stereo-left 合成图与状态日志，并非新增完整多视角 episode 导出器。

### PICO 物理与控制周期

Isaac PICO 可在非评分练习中追加 `--no-task-scoring --physics-dt 0.004 --control-dt 0.04`：物理子步为 4 ms，每次关节目标更新推进 10 个子步、共 40 ms，控制循环目标为现实时间 25 Hz。实际频率受渲染和求解耗时限制。未指定时保留原场景物理子步和 20 ms 控制周期；MuJoCo 与其他 mapper 不支持自定义控制周期时明确报错。

PICO SDK 使用单帧缓冲，接收线程持续更新最新帧。每轮在渲染后、求解前读取最新可用输入，过期输入保持当前位置；循环超时不会积压动作或补跑旧目标。关节速度限幅使用配置的仿真控制周期，接触检查要求覆盖全部子步。`session.json` 的 `control_timing` 与冻结的 `scene.json` 记录实际周期。该设置没有改变头显视频或 review 图像采样目标，且不代表 RoboDojo VR 输入频率的已验证复刻。

4 ms 子步已通过原生保持姿态与接触采样检查；灵巧手抓取、滑动和插入质量仍需要实际交互验证。

### 低分辨率头显预览与离线补渲染

预配置 recipe 为 `isaac-pico-controllers-pedal-deferred`。Isaac 后上方视角可用 `--deferred-recording --preview-scale 0.75 --video-fps 20`。原 `--width 640 --height 480` 仍是采集双目的分辨率，头显独立相机降为 480×360；桌面特写随预览缩放。采集期间只渲染头显相机；每个控制步保存原采集相机的位姿和内参、机器人根位姿/关节状态、全部已注册刚体状态、固定支撑状态，以及后续动作、episode 编号、主机时间和仿真时间。状态异步写盘，队列满时等待写盘，不静默丢帧。

`replay/manifest.json` 保存相机与实体清单、时间约定和冻结场景文件哈希；`replay/*.npz` 保存数值状态，`replay/frames.jsonl` 将状态与后续动作对应。终态记录的 action 为 null，不能作为常规动作训练样本。按 40 ms 控制周期，每个 episode 的动作状态按 25 Hz 仿真时间采样；现实时间更新频率仍取决于实际运行速度。该模式不使用 `--record-fps` 丢弃状态采样。

停止会话后关闭输入和视频传输，再自动恢复各帧状态、生成 `replay_images/` 中的原双目、原腕部小窗及左眼合成图（无损 PNG）。不记录操作者后上方/桌面特写相机，不重执行动作，不推进物理。渲染失败保留状态；可使用已配置的 Isaac Python 运行 `-m scripts.teleop.render_recording SESSION --output NEW_DIRECTORY` 重新渲染，输出目录必须不存在。资产和运行环境需保持与采集时一致。

当前支持单个机器人 articulation、`cpu_rigid_v1`、关闭任务视觉观察器的刚体场景。布料、软体、粒子和已检测的任务关节场景明确拒绝；这份状态是可视化回放记录，不是可续跑物理的 solver checkpoint。相机位姿随每帧保存，保留腕部随动和头部相对运动。PNG 保留原始相机 RGB，不加入遥操控制点标记。

暂停遥操时停止物理步进和仿真计时，保留头显/桌面预览与输入接收；不写入重复的状态—动作采集帧。120 秒仿真时长上限仅在实际推进物理时消耗。窗口重置会保存前一段终态并递增 episode 编号，重置后计时从零开始。

头显视频只编码新的预览帧；编码前最多缓冲一帧，预览码率上限 6 Mbps。发送失败后尝试重新连接头显接收端，重新发送编码头和关键帧；若头显已关闭接收端，仍需重新开启 Remote Vision。此限制不影响离线采集 PNG。
