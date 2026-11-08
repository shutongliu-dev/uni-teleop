# 记忆翻块 · memory_pairs

两名玩家轮流翻开十二块厚牌，共六对图案，以收集配对数较多者获胜。保留原棋牌室，桌面为 3 行×4 列牌阵和双方收集区。牌块为 64×80×20 mm，reset 时图案朝下、背面一致；唯一几何源为 `game_designs.design_for`。

黑方先手。一个回合顺序翻开两张不同且未被收走的牌；第一张保持正面朝上供双方观察，再翻第二张。

- 两图案相同：必须把这两张牌实际收进本方区域，确认后才记一对；同一玩家继续下一次翻牌。
- 两图案不同：必须让双方获得两次揭示观测，再将两牌实际翻回各自格位；确认二者均背面朝上后换手。
- 在收走或翻回确认前，不得翻第三张。已收走牌不能再次选择，单张不能充当一对。非法请求不改变状态或消耗回合。

六对全部收走后比较对数，3:3 为平局。预算耗尽、取消、API/仿真故障或缺少物理证据均保持未完成，不提前判领先者获胜。

[MemoryPairsRules](../../expansion_rules.py) 的构造函数必须接收**与实际 USD 图案绑定的私有 `card_id→pair_id` 映射**，没有默认猜测或再次洗牌。资产构建可用 `game_designs.memory_pairs_for_seed(seed)` 产生 reset 映射；该映射属于私有冻结初态，不进入公开规则、任务发现或 Agent 上下文。不能运行时重排牌面来迎合模型回答。

真实图案在牌的局部 -Z 面。实际图案面法向的世界向上分量超过 `1e-6`，即产生永久 `card_revealed` 事件，未完全翻平也算揭示。事件序号连续递增，details 只有 `card_id`、可见 `symbol` 和 `expected`，不附未揭示映射；第二次揭示不能覆盖第一次。无意图或第三张牌意外露面时，先公开 `expected=false` 事件，再以 `rule_violation` 中断且无赢家。翻回或取消不能抹去已看到的信息。

逻辑规则与 [ExpansionMatch](../../expansion_referee.py) 分开：实际揭示立即记录，操作提交还须真实己方手接触、原格／收集区目标、正反面朝向、完整释放及新的连续 300 ms 稳定证据。收集时牌须正面朝上、完整投影在本方矩形，可由同玩家已收集的牌及本次匹配牌形成无环实际支撑链，最终由 Deck 承托；翻回则两张各自背面朝上归原格。实际支撑不能从高度或目标坐标推算。非目标牌也逐帧检查。

任务已通过认证 `/board-games` 的独立 `bench_card_game_state_v1` 接入，意图为 `reveal_card(card_id)`、`collect_pair`、`flip_back`。宿主逐 ACK 缓存公开事件，双方可分别认证分页读取，关闭 worker 后仍可读取历史。Agent 将相同事件分别追加到双方独立上下文；不共享对手推理，也不把事件缓存当作模型已看过图像的证明。每次 reveal／collect／flip_back 确认都会递增 `turn_index`，即使当前玩家不变也停止旧动作。

Agent 使用 `--family card_room --task task6`；真实观测、事件分页、epoch 和启动方式见[接入指南](../../../../../docs/agent_bench/card_room_expansion.md)。当前 `scene_v2` 图案固定绑定资产私有 reset 映射，会话 seed 不重新洗牌。公开状态和模型输入不发布该完整映射。

```bash
python -m scenes.workrooms.card_room.tasks.memory_pairs.task
```

入口只显示无私有牌面的静态候选。资产构建统一使用 `scripts.assets.build_card_game_candidates.build_candidate`，保留 seed、实际牌面映射、源码与输入哈希，输出存入共享资源目录。

CPU 状态机、HTTP 生命周期与模型替身检查不运行原生物理；本次接入的原生 HTTP 验证按任务单独封存。场景工程翻转、机器人翻牌和完整模型记忆对局分别验收，历史报告不进入启动依赖。
