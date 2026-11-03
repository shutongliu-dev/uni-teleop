# 依赖准备

配套版本、Python 依赖和安装脚本随仓库提供。脚本在 `local/dependencies/` 建立独立环境，
把路径写入 `local/dependencies.json`；不会连接手套、头显或机械臂。

## 快速准备

Linux x86_64 先准备 Python 3.12、Python 3.10 和 venv/pip。
从 MANUS 获取并解压 **3.2.0 Linux SDK**；从 SHARPA 获取获准使用的 **V4 优化器**。
然后执行（把路径换成自己的）：

```bash
python3 tools/setup_dependencies.py --mujoco --pico \
  --manus-sdk /path/to/MANUS-3.2.0 \
  --sharpa-retarget /path/to/sharpa-manus-sdk \
  --data /path/to/dexbench/data
```

脚本从官方固定提交下载 PICO C SDK，并核对 SHA-256，保留原 LICENSE 和头文件。
它只下载 SDK，不安装 PC Service 或头显 APK；这两项按
[XRoboToolkit 官方说明](https://github.com/XR-Robotics)安装。
已有 SDK 可用 `--pico-library /path/to/libPXREARobotSDK.so`，无需再次下载。
当前自动下载仅覆盖独立手/Franka 使用的 C SDK，不包含 SONIC 的 Python 绑定。

使用 Isaac 时单独安装 Isaac Sim 6.1，另加：

```bash
python3 tools/setup_dependencies.py --cpu \
  --isaac-python /path/to/isaac/python.sh
```

Isaac 原生环境仍需项目使用的 numpy、scipy、pydantic 和 Pillow；按
[原生后端依赖](../backends/isaac/pyproject.toml)准备，勿把 MuJoCo 或 MANUS 装入 Isaac。
脚本不会重装 Isaac、安装 GPU 驱动或下载整套 DexBench 资产。
每次调用保留本地配置已有字段，可以分步安装。
`--python` / `--retarget-python` 可以指定现有解释器；V4 固定用 CPython 3.10。

填写 `local/dependencies.json` 中的 `MANUS_GLOVE_ID`、`MANUS_CALIBRATION`；
Isaac 再填写 `CUDA_VISIBLE_DEVICES`。随后检查和启动：

```bash
./teleop doctor mujoco-pico-manus-sharpa-hand --config local/dependencies.json
./teleop run mujoco-pico-manus-sharpa-hand --config local/dependencies.json
# 或改用 isaac-pico-manus-sharpa-hand
```

USB 权限、有效 MANUS 授权和本机手套标定仍需使用者提供；安装脚本不改变系统权限。
`doctor` 只检查路径存在，不证明授权、二进制兼容或设备跟随。

## 包含范围

| 依赖 | 提供方式 |
| --- | --- |
| Python 数值库、MuJoCo | 包依赖声明，由 pip 安装到隔离环境 |
| MANUS Python 辅助依赖 | `manus.txt` 固定版本；SDK 本体从本机授权目录读取 |
| SHARPA V4 辅助依赖 | `sharpa-v4.txt` 固定版本；优化器从本机授权目录读取 |
| PICO C SDK | `sources.json` 固定官方提交和校验值，脚本下载到本机 |
| Isaac、机器人/任务资产 | 使用现有安装及数据目录，未镜像上传 |

SHARPA 随附 `License` 明确要求事先书面许可才能上传公共或共享仓库。
MANUS SDK 的许可包含有条件的 CoreSDK 应用分发条款，不等于允许原样镜像整套 SDK；
因此这里提供[官方获取入口](https://docs.manus-meta.com/3.2.0/Plugins/SDK/getting%20started/)，不重发 SDK 原包。
授权钥匙、用户标定、个人路径、SDK 二进制及生成的虚拟环境不进入本仓库。
