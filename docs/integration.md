# 集成结构与来源

第一版采用统一入口 + 两套独立后端。这里包含真实实现、模型与原测试，不是仅链接
两个远程仓库；同时没有把两套不同的控制时序、SDK 或 Isaac/MuJoCo 强行合并。

```text
uni_teleop CLI → profile + local host paths
  ├─ DexBench frozen source
  │   ├─ PICO SDK process → wrist/finger mapping → IK → Isaac
  │   ├─ FACTR serial reader → calibrated joint mapping → Isaac
  │   └─ MANUS SDK process → official SHARPA optimizer → fingers
  └─ SONIC-MuJoCo
      ├─ PICO body → SONIC encoder/decoder → G1
      ├─ PICO fingers/head → hand/neck mapping
      ├─ ZMQ → SuperDex hands
      └─ video, recording, contacts, haptics
```

## 快照

- DexBench：`4a2853bd83da14a38b553ca9781a8bbddea9c266` 加导入时工作树修改。
- SONIC-MuJoCo：`67c8f82214bf22e4e9712688570b0ed1e7864d9d` 加导入时工作树修改和
  相关未跟踪源码/模型，例如相机、episode reader、UniVLaT 与颈部模型。
- `sources.lock.json` 记录实际导入字节哈希，提交号不冒充全部文件的精确版本。
- 原 DexBench `pico_isaac.py` 未提交的报告字段删除随工作树一同导入；原仓库未改变。
- 个人 AGENTS 历史、运行输出、SDK、权重和本机配置未导入。

## 本仓库对导入代码的改动

仅在 DexBench 的 `scripts/teleop/run.sh` 和 `run_master_slave.sh` 增加
`DEXBENCH_PYTHON` 解释器覆盖项；未设置时仍回退原来的 `.venv/bin/python`。
统一 CLI 另外设置每次启动的源码路径，清理继承的旧 Isaac experience，保留后端
物理和控制语义。原两个工程的源码、Git 状态和已运行进程不受影响。

`sources.lock.json` 保存导入前哈希，`tools/verify_sources.py` 明确校验这两处适配差异。
后续改变 vendor 实现应更新此说明和校验工具，避免不透明覆盖。

## 可复现与边界

公共 JSON 不含本机路径。路径、设备 ID 和标定文件引用在忽略的 `local/host.json`。
参数数组直接交给进程，不通过 shell 拼接；含空格的路径保持为单个参数。
额外后端参数通过 `--` 透传，统一入口不重新实现底层参数校验。

SIMPLE、官方 SHARPA 优化器与商业 SDK 是外部依赖。MuJoCo H2 只是基础控制模型，
未复用 G1 策略伪装全身遥操。跨后端通用数据协议和录制格式统一属于后续工作；
第一版沿用各自现有格式，避免破坏已使用的数据集。
