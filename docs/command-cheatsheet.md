# 指令速查

适用于当前 PICO＋Isaac＋踏板的状态采集流程，同时提供统一 CLI 常用命令。所有命令默认在仓库根目录执行，路径中的 `<运行目录>`、`<编号>` 等需替换为实际值。

本文维护操作命令；项目总结文档不随日常操作改动自动更新。

当前本机默认数据源已切换为 **DexBench data-v6**，位于
`local/data/dexbench-data-v6`。`local/dependencies.json`、`local/pico-manus.json`
和后端 `runtime/data` 链接均选择该版本，原有启动命令不变。数据目录不要与旧版混合。
本机 `local/run-pico-manus.sh` 和 `local/run-pico-video.sh` 已默认选择
L 型连接件（`--robot-profile l_mount`）。需要临时切回原版时，在启动命令后
添加 `--robot-profile original`。直接使用统一 CLI 时仍需显式选择机器人构型。
新版目录有 7 个房间、51 个任务，但不代表全部任务都支持当前单臂 CPU 刚体遥操配置。
升级来源与验证范围见 [data-v6 集成记录](provenance/dexbench-data-v6.md)。

启动时会打印 `Loading task` 和 `Task ready in ...`。任务加载在一次校验范围内
复用发布验证和文件哈希，首次仍读取完整内容；范围仅覆盖本次加载，不跨启动缓存。
这段时间是场景准备耗时，与运行时的遥操延迟不同。

## 1. 常用命令一览

| 用途 | 命令／操作 |
|---|---|
| 查看可选组合 | `./teleop list` |
| 启动本机 XR 服务，日志输出到终端 | `bash local/run-xr-service.sh` |
| 启动本机 PICO 状态采集 | `bash local/run-pico-video.sh` |
| 只测试，不保存采集数据 | `bash local/run-pico-video.sh --no-recording` |
| 开始／暂停 | GUI Start / Pause、空格或踩一下踏板 |
| 保存当前条并准备下一条 | GUI Save episode / Next 或 F8 |
| 退出采集 | 先暂停，再在采集终端按一次 Ctrl+C，等待退出 |
| 查看已封存片段 | `cat "$TELEOP_SESSION/episodes.json"` |
| 重渲染一条 | `python3 local/render-pico-recording.py "$TELEOP_EPISODE"` |
| 查看渲染结果 | `cat "$TELEOP_EPISODE/replay_images/result.json"` |
| CPU 测试 | `local/dependencies/cpu/bin/python -m unittest discover -s tests` |

`local/` 中的启动脚本、环境和配置是本机辅助文件，被 Git 忽略；新克隆仓库不会自动带有这些文件。正式 recipe 和后端脚本位于版本管理目录中，其他机器需先按 [安装说明](setup.md) 配置。

## 2. 检查配置与启动参数

已有本机配置时：

```bash
./teleop list
./teleop doctor isaac-pico-controllers-pedal-deferred --config local/dependencies.json
./teleop run isaac-pico-controllers-pedal-deferred --config local/dependencies.json --dry-run
```

`doctor` 检查配置和依赖，不等于头显、踏板或遥操已验证。`--dry-run` 只打印命令，不启动设备或仿真。

新建配置时选择一个新的输出文件，不替换正在使用的配置：

```bash
./teleop configure isaac-pico-controllers-pedal-deferred --output local/pico-new.json
```

按生成的字段填写本机环境和设备信息，然后使用对应文件检查和启动。

## 3. 启动服务和采集

### 终端 A：XR 服务

```bash
bash local/run-xr-service.sh
```

本机脚本加载隔离构建的服务修复库，前台运行并直接输出日志。已存在服务时先复用，不要重复启动。原安装包中的 `runService.sh` 不等同于这个本机入口。

### 头显设置

1. 打开 XRoboToolkit，开启 Head、Controller 和 Send。
2. 追踪连接目标填写主机可达的局域网地址。
3. 开启 Remote Vision，填写主机地址；地址字段只填 IP，不附加端口。
4. 正对操作画面再开始，让控制坐标与视角建立参考。

可用以下命令查看主机地址：

```bash
hostname -I
```

存在多个网卡时选择头显可达的地址，不能直接假定输出的第一项正确。视频连接与追踪连接分别检查。

### 终端 B：本机一键启动

```bash
bash local/run-pico-video.sh
```

该入口使用 `isaac-pico-controllers-pedal-deferred` recipe 和 `local/dependencies.json`，并设置本机 GStreamer 插件环境。默认配置为物理子步 4 ms、控制周期 40 ms、预览缩放 1.0（每眼源图 640×480）、预览目标 20 fps。采集与重渲染已分开。

正式 CLI 的等价 recipe 入口如下；本机视频插件所需环境仍需自行准备，因此当前机器优先使用上面的辅助脚本：

```bash
./teleop run isaac-pico-controllers-pedal-deferred --config local/dependencies.json
```

需要调整预览时，可以通过辅助脚本传参：

```bash
bash local/run-pico-video.sh --preview-scale 1.0 --video-fps 20
```

程序启动会打印当前 `session` 路径。改代码后要重新启动采集进程，已有运行使用冻结的源码快照。

清晰度默认已提高到每眼源图 640×480；原来是 480×360。主画面与小窗合成时会缩放，实际显示细节也受头显编码与显示影响。如需降低渲染负担，可传 `--preview-scale 0.75`。帧率与码率限制保持不变，采集图像分辨率不受此参数影响。

### 遥操位移倍率

当前默认 `--scale 1.0`：手柄相对参考点移动 10 cm，对应末端目标移动 10 cm；旋转不放大。实际运动仍受关节速度、可达范围和控制保护约束。测试和采集均使用此默认值，重启后生效。

```bash
bash local/run-pico-video.sh --no-recording --scale 1.0
# 如需临时放大位移
bash local/run-pico-video.sh --no-recording --scale 1.5
```

### 按任务自动选择头显机位

使用 `--operator-view rear` 和工作台练习模式时，已为以下工作区设置独立头显机位：

- 锅盖任务：瞄准灶台上的锅具与锅盖。
- 单臂三子摆放：同时覆盖棋盘和侧方取子架。
- 实验室样本分类、试管入架、瓶盖：拉近实际样本操作区。
- 机械装配、书房桌面任务：居中并留出边缘余量。
- 取书、书籍上架：同时覆盖书架与桌面。

这些机位只修改头显预览；左右采集相机和腕部相机的位姿、镜头参数不变。腕部小窗继续随手移动。其他任务沿用原机位。配置会在下一次启动时自动生效，无需增加参数；全房间原生任务不自动套用工作台机位。

### 头显主视角隐藏机械臂

当前 PICO 采集 recipe 默认开启 `--operator-hide-arm`。在头显主视角和腕部小窗中隐藏 Franka 机械臂几何，保留灵巧手；机器人状态、驱动和碰撞不变。当前 deferred／只测试模式下，两路头显视角共用一次 `render_many` 调用，不再为腕部单独渲染。离线采集图像仍显示完整机械臂；若同时要求即时采集图像或额外完整机器人桌面视角，则需要另一次渲染。首次渲染产品初始化可能内部预热多次，并非只调用一次 GPU 绘制。尚需头显原生渲染实测。

```bash
# 隐藏机械臂视觉（默认）
bash local/run-pico-video.sh --no-recording --operator-hide-arm

# 恢复显示
bash local/run-pico-video.sh --no-recording --no-operator-hide-arm
```

头显主相机已沿原视线移到基座前方，朝工作区推进最多 25 cm（较小布局按基座到工作区距离缩减），保持原有水平朝向与俯视角。采集相机位置不变；改动需重启生效。

### 头显辅助视角

当前默认 `--operator-inset wrist`：主视角下方只显示原来的腕部相机小窗，不创建或渲染桌面辅助相机。腕部视角仍跟随手的位置与姿态，不改变相机外参。头显对原图做自适应平面旋转：尽量将控制右对齐屏幕右、控制前对齐屏幕上；替代原固定顺时针 90° 显示。角度采用平滑与每秒 90° 的转速限制，投影方向互相矛盾时保持上一角度；初始无法判断时回退顺时针 90°。旋转保留完整图像并适配小窗，可能有留边和缩小。该方法只能近似对齐，不能消除透视或相机随手运动造成的物体反向运动。采集图像不做这项旋转。

头显主视角与腕部小窗都隐藏机械臂、保留灵巧手，在 deferred／只测试模式下共用一次渲染更新。采集相机姿态和图像方向不变，离线图像仍显示完整机械臂。

```bash
# 仅腕部辅助视角（当前默认）
bash local/run-pico-video.sh --no-recording --operator-inset wrist

# 如需恢复桌面近景＋腕部双小窗
bash local/run-pico-video.sh --no-recording --operator-inset table-wrist

# 只显示主视角
bash local/run-pico-video.sh --no-recording --operator-inset none
```

### 只测试，不保存采集数据

```bash
bash local/run-pico-video.sh --no-recording

# 可同时切换场景
bash local/run-pico-video.sh --no-recording --family study --task stationery_sorting
```

`--no-recording` 覆盖 recipe 默认的状态录制与 profile 状态快照：不生成 episode、逐帧动作日志、状态存档或图像。头显、踏板、暂停和 Reset 仍可使用，界面显示 `TEST MODE - NOT RECORDING`，不提供 F8 保存按钮。恢复采集时，退出后不带此参数重新启动。

运行目录仍保存源码快照、场景／初态配置、会话摘要和 SDK 诊断日志，因此不是完全零文件写入。该选项不能与 `--review-dir` 同时使用，目前仅支持 Isaac 遥操入口。

### 切换任务场景

当前通过重启采集并指定 `--family` 与 `--task` 切换，不支持运行中热切换。先暂停并结束当前采集，XR 服务可以保持运行。

```bash
# 厨房：早餐配盘（当前默认）
bash local/run-pico-video.sh --family kitchen --task breakfast_serving

# 厨房：果蔬分类装盘
bash local/run-pico-video.sh --family kitchen --task produce_transfer

# 书房：文具分类
bash local/run-pico-video.sh --family study --task stationery_sorting

# 机械装配：大零件分类
bash local/run-pico-video.sh --family mechanical_assembly --task parts_sorting
```

上面的任务 ID 已在注册表中实现，不表示每个场景均通过当前头显／离线采集实测。现用 recipe 会保留工作台和任务物体、裁剪无关房间布景，继续使用无评分刚体练习和 deferred 采集。完整房间不是当前入口的默认行为。

查看已实现的任务（只读，不启动仿真）：

```bash
PYTHONPATH="$PWD/backends/isaac:$PWD" local/dependencies/cpu/bin/python - <<'PYLIST'
from scenes.workrooms.catalog import list_tasks
for task in list_tasks():
    print(task['family'], task['task_id'], task['title'])
PYLIST
```

列表是任务实现清单，不是 deferred 采集兼容性清单。含软体、粒子、任务关节或多机器人等超出当前回放范围的任务不能直接套用此 recipe。换场景后相机遮挡与手物交互仍需检查；F8 / R 只重置当前场景，不换任务。

## 4. 连续采集操作

| 操作 | 行为 |
|---|---|
| Start / Pause 或空格 | 开始／暂停 |
| 握住 Grip | 移动机械臂；扳机控制手指开合 |
| 踩一下踏板 | 切换暂停／继续；恢复时重新建立手柄参考 |
| Save episode / Next 或 F8 | 保存当前片段终态，封存文件，重置场景并暂停 |
| Reset scene 或 R | 当前 deferred 模式下也会封存当前片段，不是丢弃 |
| C | 暂停并请求重新建立视角参考 |
| L | 切换手柄朝向锁定 |

键盘快捷键需要仿真应用获得焦点；踏板通过设备事件读取，不依赖窗口焦点。踏板长踩重置已移除。

推荐顺序：开始 → 操作 → 踏板暂停 → F8 保存 → 开始下一条。暂停时不推进物理、不累计仿真时间、不生成重复状态帧。空片段不落盘，所以编号可能跳号；以 `episodes.json` 为准。

结束全部采集：暂停后在采集终端按一次 Ctrl+C，等待状态封存和程序退出。退出不会自动重渲染，也不会关闭独立终端 A 中的 XR 服务。需要关闭服务时，在终端 A 按 Ctrl+C。

当前 F8 连续采集逻辑通过 CPU 测试，头显端多条连续流程仍待实测。

## 5. 定位采集数据

先把启动日志中打印的会话路径填入变量：

```bash
TELEOP_SESSION="runs/isaac/<运行目录>/session"
cat "$TELEOP_SESSION/episodes.json"
```

新格式：

```text
session/
  session.json                 # 整体运行信息
  frames.jsonl                 # 控制与事件日志
  episodes.json                # 已封存片段清单
  episodes/
    000000/
      scene.json
      initial_state.json
      session.json
      replay/
        manifest.json
        frames.jsonl
        000000.npz
        ...
```

选择一条已封存片段，以下命令在同一个终端继续使用：

```bash
TELEOP_EPISODE="$TELEOP_SESSION/episodes/000000"
cat "$TELEOP_EPISODE/session.json"
wc -l "$TELEOP_EPISODE/replay/frames.jsonl"
```

旧版数据没有 `episodes/`，直接包含 `session/replay/`，此时设置：

```bash
TELEOP_EPISODE="$TELEOP_SESSION"
```

每条最后一行是没有后续动作的终态，因此状态数通常比动作数多 1。数据在 `runs/` 下，不提交到 Git。

## 6. 独立重渲染

采集进程结束后，在仓库根目录运行：

```bash
python3 local/render-pico-recording.py "$TELEOP_EPISODE"
```

输出默认为该片段的 `replay_images/`。如果已经存在输出目录，换一个新目录，不覆盖旧产物：

```bash
python3 local/render-pico-recording.py "$TELEOP_EPISODE" \
  --output "$TELEOP_EPISODE/replay_images_v2"
```

该辅助脚本读取本机配置，设置 Isaac 解释器、资产与 GPU 环境。底层正式入口为 `scripts.teleop.render_recording`，需要 Isaac Python 和正确的仓库／后端模块路径，不能用普通系统 Python 直接替代。

正常完成后检查：

```bash
cat "$TELEOP_EPISODE/replay_images/result.json"
wc -l "$TELEOP_EPISODE/replay_images/frames.jsonl"
```

结果应包含 `status: complete`、`physics_steps: 0`，图像索引条数应与状态索引一致。原采集相机输出 `pico_left`、`pico_right`、`pico_inset`；`review_left` 是便于查看的拼接图，不是第四台采集相机。单 GPU 默认先采集、后渲染；当前没有自动后台任务队列。

## 7. PNG 导出 MP4

前提：重渲染成功，已安装 `ffmpeg`。以下按当前 40 ms 控制周期使用 **25 fps 仿真时间播放**，不是复原含暂停和卡顿的墙钟时间。若改变控制周期，需相应修改 fps。

```bash
TELEOP_IMAGES="$TELEOP_EPISODE/replay_images"
mkdir -p "$TELEOP_EPISODE/videos"
ffmpeg -hide_banner -nostdin -n -framerate 25 \
  -i "$TELEOP_IMAGES/%06d_review_left.png" \
  -c:v libx264 -threads 4 -preset fast -crf 18 \
  -pix_fmt yuv420p -movflags +faststart \
  "$TELEOP_EPISODE/videos/preview.mp4"
```

`-n` 表示已有同名视频时不覆盖。分别导出三路采集相机：

```bash
for TELEOP_CAMERA in pico_left pico_right pico_inset; do
  ffmpeg -hide_banner -nostdin -n -framerate 25 \
    -i "$TELEOP_IMAGES/%06d_${TELEOP_CAMERA}.png" \
    -c:v libx264 -threads 4 -preset fast -crf 18 \
    -pix_fmt yuv420p -movflags +faststart \
    "$TELEOP_EPISODE/videos/${TELEOP_CAMERA}.mp4"
done
```

查看视频帧数、分辨率和时长：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
  -of json "$TELEOP_EPISODE/videos/preview.mp4"
```

默认相机为双目各 640×480、腕部 256×192。MP4 用于观看，原始无损 PNG 和状态存档保留。

## 8. 快速排错

### 查进程、端口和 GPU

```bash
pgrep -af 'RoboticsService|uni_teleop.runtime.isaac|scripts.teleop.pico_isaac|pico_video.py'
ss -tnp '( sport = :13579 or dport = :12345 )'
nvidia-smi
```

13579 为视频控制监听端口，12345 为默认头显视频接收端口；实际以客户端协商日志为准。连接存在不等于帧在更新。

### 查当前会话日志

```bash
tail -n 5 "$TELEOP_SESSION/frames.jsonl"
cat "$TELEOP_SESSION/session.json"
```

| 症状 | 首先看什么 |
|---|---|
| 一直暂停 | 日志 `status`、Head 是否开启、是否正对画面完成 Start 标定 |
| tracking lost | `input_health` 最新帧年龄、服务日志、头显 Send；必要时头显 Reconnect |
| 灰白或 client not connected | Remote Vision 的主机地址、视频端口；关闭后重新开启 Remote Vision |
| 电脑已重置，头显仍旧画面 | 视频发送超时／重连日志，头显接收端是否仍开启 |
| 踏板无响应 | 本机配置 `PEDAL_EVENT`、设备权限、是否被另一个进程独占 |
| 没有采集 PNG | 新流程先存状态，需要单独重渲染，不是缺帧 |
| 重渲染提示目录存在 | 用 `--output` 指向新目录 |

不要另启动一个 SDK 订阅读取器和正在运行的采集进程抢同一输入。

历史上可能使用过临时 systemd 服务。先查看实际单元，再针对存在的单元查日志；本机一键脚本本身不会创建这些服务：

```bash
systemctl --user list-units --all 'uni-teleop*'
# 将下方占位替换为上一步看到的实际单元名
journalctl --user -u '<实际单元名>' -n 50 --no-pager
```

## 9. 开发检查

```bash
local/dependencies/cpu/bin/python -m unittest discover -s tests
git diff --check
git status --short
```

CPU 测试不代替原生仿真、真实设备和任务成功验证。本页不包含自动清理数据、使能实体电机或推送远程的命令。

## 10. MANUS 手套标定（本机）

保持手套接收器和 license key 都插着，手套开机；先退出占用 MANUS 的读取程序／Core，然后在桌面终端启动：

```bash
bash local/run-manus-calibration.sh
```

这是 SHARPA 配套 Linux CalibrationGUI 的本机启动脚本，不是纯命令行自动标定，也不会启动机器人。选择 Left Glove / Right Glove，点击 Start Calibration（F5），按照界面动作提示完成每步，用 Next Step（F9）继续。左右手分别完成；步骤以当前 SDK 实际提示为准。

启动时打印独立输出目录 `local/manus-calibrations/session-时间-随机后缀/`。成功后分别生成 `Calibration_left.mcal` 和 `Calibration_right.mcal`，不会覆盖其他目录中的旧文件。同一窗口中重新标定同一只手会更新本次目录中的该手文件。完成后关闭窗口再启动其他 MANUS 读取程序。

标定成功保存不代表手形已经正确。接入前应验证同一张手／握拳动作的原始骨架和机器人手形。本机已完成新旧标定对照，当前使用历史默认右手标定；文件与设备选择仅保存在忽略的 `local/pico-manus.json` 中。不要把标定、设备标识或 SDK 二进制提交到仓库。

### PICO 控臂 + MANUS 控手指

先关闭标定工具及其他 MANUS 读取器。手臂仍用 PICO Grip，位移倍率 1.0；手指由 MANUS 控制，PICO Trigger 不再控制手形。

```bash
# 只测试，不保存 episode
bash local/run-pico-manus.sh --no-recording

# 采集状态，结束后另行重渲染
bash local/run-pico-manus.sh
```

启动后保持暂停，由操作者按 Start 或踏板继续。若本次诊断启动的后台服务仍在运行，不要重复启动：

```bash
systemctl --user status uni-teleop-pico-manus.service
tail -f local/logs/pico-manus.log
# 改为终端启动前，先停止已有后台实例
systemctl --user stop uni-teleop-pico-manus.service
```

本机脚本使用注册配方 `isaac-pico-manus-native-pedal-deferred`，通过隔离进程读取 MANUS 原生 SDK，加载明确指定的 `.mcal` 后接入 SHARPA 映射。旧的 Python SDK 配方仍独立保留。
