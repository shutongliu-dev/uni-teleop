# 来源与权利说明

| 组成 | 来源 | 本仓库处理 |
|---|---|---|
| DexBench + 本地遥操开发 | https://github.com/VVVVME50/dexbench | 任务引擎及公共模块来源，保留 PROVENANCE |
| SONIC-MuJoCo | https://github.com/shutongliu-dev/sonic-mujoco | 场景/模型来源，设备与映射模块已抽取到公共包 |
| SHARPA MANUS retargeting V4 | https://github.com/sharpa-robotics/sharpa-manus-sdk | 仅集成适配代码，优化器单独安装 |
| MANUS / PICO SDK | 官方 SDK | 不分发二进制或许可 |
| Isaac Sim | NVIDIA | 不分发运行时 |
| SONIC ONNX | 对应策略发布方 | 不分发权重 |

各机器人、场景、纹理来源说明与已有 LICENSE/SOURCE 文件保留在 backends 中。
本次导入的两个顶层工程没有统一顶层 LICENSE；本仓库不替上游授予新的开源许可，
也不将商业依赖重新许可。仓库目前私有。具体资产条款以各自来源说明为准。

README 的 MANUS 产品图来源与权利说明见 [图片说明](docs/assets/README.md)。
