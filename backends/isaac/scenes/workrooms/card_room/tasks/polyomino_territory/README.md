# 拼板占地 · polyomino_territory

两名玩家轮流在 6×6 棋盘上放入自己的拼块，以最终占据格数比较胜负。保留 `card_room_heritage_20261017_v3` 原棋牌室；棋盘、拼块和供料区属于本任务。棋格节距 52 mm，拼块厚 22 mm，具体尺寸、初态和几何由家族唯一设计源 `game_designs.design_for` 定义。

双方各有完全相同的六块：单格、二连格、三直格、三格 L、四格方块、四格 L，共 17 格。黑方先手。一次合法回合放入一个尚未使用的己方拼块；可旋转 0°、90°、180°、270°，不能翻面镜像，必须平放、完整位于棋盘格内且不与已有拼块重叠。不增加首角、角接或边接限制。

只有己方全部未用拼块在所有位置和旋转下均放不进去，才允许 pass。双方都没有合法放置（含库存耗尽）时结束，比较实际占据的格数；相等为平局。非法选块、重叠、越界或非法 pass 不改变棋盘或消耗回合。超时、步数耗尽、取消和运行故障均为未完成，不依当前面积提前判赢家。

格坐标 q 沿世界 +X、r 沿 +Y，西南格为 `(0,0)`。正旋转绕 +Z，从上方看为逆时针；旋转形状后减去两坐标最小值归一化，再加 `anchor=[q,r]`。anchor 是旋转后包围框的左下格，刚体原点在该包围框中心。格心为 `board.center_xy_m + ((q-2.5)*pitch, (r-2.5)*pitch)`。

纯规则实现是 [PolyominoTerritoryRules](../../expansion_rules.py)，物理提交由 [ExpansionMatch](../../expansion_referee.py) 负责。真实己方手操作后，裁判检查实际盒形碰撞体并集的完整投影、平放方向、Deck 支撑、机器人释放与连续 300 ms 稳定窗口；保留 L 形凹口，不用单个格心代替全轮廓。已放拼块和其他库存必须保持安全布局；目标坐标或模型声明不构成证据。pass 也由下一个安全原生子步确认，意图请求不会直接提交。

本任务通过认证 `/board-games` 和独立 `bench_card_game_state_v1` 接入 Agent；原三棋种的落子合同保持不变。意图为 `place_piece(piece_id,anchor,rotation_deg)`、`pass`，每次确认后更新操作 epoch。Agent 使用 `--family card_room --task task4`；完整命令、阈值及事件语义见[接入指南](../../../../../docs/agent_bench/card_room_expansion.md)。

```bash
python -m scenes.workrooms.card_room.tasks.polyomino_territory.task
```

该入口只输出静态候选描述，不创建回合。候选资产构建入口为 `scripts.assets.build_card_game_candidates.build_candidate`；资产位于共享 assets，冻结场景和实验记录位于共享 datasets，展示媒体位于共享 demo。

纯规则、物理消费器与 HTTP 生命周期已有 CPU 检查，合成帧不算原生资格。本次接入的原生 HTTP 验证按任务单独封存，完整机器人摆放和模型整局能力尚未验收。早期物物动力学与外观记录独立保存，不进入运行依赖或评分。
