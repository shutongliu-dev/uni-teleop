# 公共传感器与坐标 SDK

`bench.sdk` 提供可被不同 Agent 和任务复用的纯数据操作。它只读取调用者提供的公开数据，不读取场景真值、不调用模型、不管理预算，也不提交机器人动作。

| 模块 | 能力 |
|---|---|
| `depth.py` | `validate_metric_depth` 校验非空二维 float32 公制深度；`decode_metric_depth_npy` 限制头部、尺寸和负载长度，以无 pickle 方式解码并校验形状 |
| `geometry.py` | `camera_z_to_world` 按标定反投影像素；`apply_world_pose_delta` 组合 world 平移与旋转向量增量 |

坐标约定为 SI 单位、wxyz 四元数；深度是光学轴 camera_z，像素索引从零开始，计算采用像素中心 `(u + 0.5, v + 0.5)`。调用者负责核对快照身份、时间、图像边界、SHA 和访问权限。SDK 不识别物体、不产生抓取方案，也不承诺目标可达。

模型客户端使用该 SDK 计算公开深度与坐标，并继续负责模型输入预览、历史、预算和动作准入。旧 action 模式不会因此自动获得新的查询工具；腕相机与双臂工具是否可用仍以所选服务实际发布的能力为准。HTTP 契约见 [bench/runtime](../runtime/README.md)。
