# PICO 客户端

`sonic_mujoco` 使用 XRRobotKit 的追踪和视频能力，并增加由 MuJoCo 接触驱动的
PICO 手柄触觉。仓库中的补丁基于 XRoboToolkit Unity Client `c932609`。

## 构建

```bash
git clone https://github.com/XR-Robotics/XRoboToolkit-Unity-Client.git
cd XRoboToolkit-Unity-Client
git checkout c932609
git apply /path/to/sonic-mujoco/pico_client/xrobotoolkit-haptics.patch
```

使用 Unity `2022.3.16f1` 打开工程并构建 Android APK，然后安装：

```bash
adb install -r <apk>
```

补丁包含三项改动：

- 处理服务端发送的 `HapticImpulse` 命令；
- 使用 PICO 原生接口控制左右手柄的强度、时长和频率；
- 使用独立包名 `com.xrobotoolkit.client.sonic`，可与原版客户端同时安装。

追踪、按钮输入和视频回传逻辑保持不变。开发构建使用 Unity 默认签名，不依赖
上游工程中的本机 keystore 路径。
