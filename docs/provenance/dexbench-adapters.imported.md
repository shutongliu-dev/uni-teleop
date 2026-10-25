# 外部采集适配

`bench/adapters/` 管理外部采集适配，目前包含 [MANUS 数据采集与重定向](../../docs/manus/README.md)，硬件未连接。`manus/agent_adapter.py` 将采集结果转换为公开机器人动作，不拥有任务规则、评分器或模型决策。

仿真 HTTP 服务位于 [bench/runtime](../runtime/README.md)，唯一入口是 `bench.runtime.server:app`；模型决策和学习位于 [agent/model_client](../../agent/model_client/README.md)。根 `interface` 包已删除，MANUS 的当前入口为 `python -m bench.adapters.manus`。冻结实验继续使用自身快照，活动代码不提供旧路径别名；完整迁移表见 [STRUCTURE.md](../../STRUCTURE.md)。
