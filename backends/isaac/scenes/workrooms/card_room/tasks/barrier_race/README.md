# 路障竞速 · barrier_race

两名玩家在 5×5 棋盘上移动各自棋子并放置路障，先到达对岸任一格者获胜。保留 `card_room_heritage_20261017_v3` 原棋牌室；棋盘、棋子和路障为任务桌面层。棋格节距 75 mm；每方一枚棋子、四块墙。唯一尺寸和初态来源为 `game_designs.design_for`。

格坐标为 `(q,r)`，q 沿世界 +X，r 沿 +Y，均为 0..4。黑方从 `(2,0)` 出发、目标 r=4；白方从 `(2,4)` 出发、目标 r=0；黑先。每个回合选择移动己方棋子或放置一个未使用的己方路障。

移动规则：

- 通常沿没有墙的正交边走一格，不能走对角线。
- 若相邻格是对方棋子，且其后方格和边存在且无墙，必须用直跳到后方格代替走入被占格。
- 只有直跳被墙或棋盘边界阻挡时，才可从对方棋子所在格向两侧选择一条无墙正交边，形成侧跳。不能跳过不相邻的棋子。

墙覆盖相邻的两条棋格边；`anchor(q,r)` 的 q、r 为 0..3。水平墙沿 X，阻断 `(q,r)↔(q,r+1)` 和 `(q+1,r)↔(q+1,r+1)`；竖墙沿 Y，阻断 `(q,r)↔(q+1,r)` 和 `(q,r+1)↔(q+1,r+1)`。不得与已有墙共享阻断边，水平/竖直墙不得在同一 anchor 交叉。放墙后，双方必须各自仍存在一条通向目标行的正交路径；路径检查把对方棋子视为可以移动，不将临时占格当作永久封路。已放墙不能移走或转向。

非法移动、越界、重复用墙、墙重叠/交叉或封死路径均拒绝，不消耗回合。只有实际确认棋子到达对岸才获胜；预算耗尽、取消和运行故障保留未完成且无赢家。已确认的合法终局保持不变。

格心为 `board.center_xy_m + ((q-2)*pitch, (r-2)*pitch)`；墙 anchor 指其中心周围四格的西南格，墙中心为 `board.center_xy_m + ((q-1.5)*pitch, (r-1.5)*pitch)`。棋子偏航不影响逻辑占格，墙平面内旋转 180° 等价。

纯规则在 [BarrierRaceRules](../../expansion_rules.py)，含 BFS 路径检查及完整跳越规则。[ExpansionMatch](../../expansion_referee.py) 另外要求真实己方手接触和移动、棋格／墙位及实际朝向、Deck 支撑、释放和连续 300 ms 稳定，检查其他棋子与墙保持。纯规则不移动物体，逻辑路径可达也不代表机器人无碰撞路径可达。

本任务已通过认证 `/board-games` 的独立 `bench_card_game_state_v1` 接入；意图为 `move_pawn(destination)` 或 `place_barrier(piece_id,anchor,orientation)`。非法意图不耗操作 epoch，每次真实操作确认后更新 epoch。Agent 使用 `--family card_room --task task5`；启动、物理门槛与终局语义见[接入指南](../../../../../docs/agent_bench/card_room_expansion.md)。原三棋种合同不变。

```bash
python -m scenes.workrooms.card_room.tasks.barrier_race.task
```

入口只输出静态候选；共享构建入口为 `scripts.assets.build_card_game_candidates.build_candidate`。CPU 检查覆盖规则、物理证据消费及 HTTP 生命周期；本次接入的原生 HTTP 验证按任务单独封存，场景／物物证据不计完整机器人对局成功。资产、开发证据和运行发布分别保存。
