# Isaac backend

本目录只维护仿真后端、机器人模型、场景以及后端测试。设备与映射实现在
[`uni_teleop/`](../../uni_teleop/)，请从[项目首页](../../README.md)选择遥操组合。

- 用户入口：仓库根目录 `./teleop list --backend isaac`
- 安装：[环境指南](../../docs/setup.md)
- 原工程说明：[导入时文档](../../docs/provenance/dexbench-README.imported.md)（历史参考，旧启动路径不再作为推荐入口）

不要在此目录启动第二份设备采集器；统一入口负责设置模块路径与会话环境。
