# Card-room board games

The fixed card-room environment hosts three independent tasks. Black moves first;
the complete inventory contains only the maximum number of legal alternating
moves, with no automatic replenishment.

| Level | Task | Board | Cell pitch | Black / white pieces |
| --- | --- | --- | --- | --- |
| 1 | `tic_tac_toe_3x3` | 3×3, three in a line | 140 mm | 5 / 4 |
| 2 | `connect_four_5x5` | 5×5, four in a line, planar with no gravity | 115 mm | 13 / 12 |
| 3 | `hex7x7` | 7×7 Hex, opposite-edge connections | 90 mm | 25 / 24 |

Square games accept horizontal, vertical and diagonal lines and permit draws.
Hex black connects the two r edges and white connects the two q edges. Neither
game captures pieces or allows a pass/swap. Competitive comparisons must pair
agent assignments across colors; the design does not remove first-player bias.

`geometry_for(game_id)` supplies metric cells, reserve locations and inventory.
Reserves occupy four task-owned lateral supply trays, two per player, clear of
the space directly in front of each base. Tray surfaces are at 0.800 m; the
unchanged 56 mm pieces have minimum reserve centre spacing of 140/120/100 mm
across the three levels. `reserve_fixture_id()` binds each reserve to its actual
support fixture. Default preflight checks native contact with that fixture,
not merely with any support. These trays do not modify the fixed parent room.
`BoardGameRules` is pure logic. `BoardGameMatch` is the authoritative physical
match: placement intent reserves a move but never changes the board or a body.
Only complete, consecutive native solver evidence can commit a move after the
unchanged 300 ms release/support dwell. Missing contact separation is unavailable
evidence, never inferred zero penetration. Every observation includes the entire
task inventory. Disturbing a committed piece pauses commitment until the current
player restores it to its original cell; it does not forfeit the game. Placement
may be off-centre within the published footprint allowance. Unused reserves may
move on the table. Pending intents can be revised or cancelled without spending
a turn, and identical pending intents are idempotent.

`BoardGameNativeObserver` observes existing contact acquisitions and reads one
batch of body states per native substep; it never advances physics or sets poses.
The default board fixture is `board`. `BoardGamePlayerProtocol` admits only owned
robot joint targets, placement intents and intent cancellation. RGB-D observations exclude privileged
logical board state; the separately labelled protocol-smoke mode may expose it.

New releases freeze all evaluator Python files into `evaluator/tabletop` and
retain `prepared_unvalidated` / `physics_accepted=false` preparation status.
`verify_evaluator_sources` checks frozen hashes and actual imported module paths.
`rules.json` and `task.json.public_rules` publish the same versioned contract;
`verify_public_rule_contract` binds their hashes to the actual rules, geometry and
thresholds. Runtime/API interruptions and budget exhaustion do not award wins.
See [the public bench contract](../../../docs/agent_bench/card_room_rules.md) for
action fields, recoverable feedback and the split from the Agent implementation.
Native acceptance and manipulation evidence must be attached independently.
The legacy `scenes.board_games.hex` package and historical releases remain intact.
