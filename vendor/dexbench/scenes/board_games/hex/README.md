# Robot Hex7x7

This package owns the rules, metric geometry, independent-player contract and
private physical move adjudication. Scene authoring and native execution remain
outside this package. CPU tests establish these contracts; they do not qualify
robot reach, grasp success, contacts, rendering or a completed physical game.

`HexGeometry` is the shared geometry source. The default nearest-neighbour pitch
is 75 mm. The 56 mm circular base is 14 mm tall, with a 40 mm diameter, 40 mm tall
upright handle. These are simulation design priors, not measured SHARPA grasp
specifications. The rigid body origin is at the **base centre**; its centre of
mass generally differs. All poses use metres and quaternion order `wxyz`.
`cell_center(q, r)` returns the desired base-centre world position. Columns
`q=0..6` are labelled A–G; rows `r=0..6` are labelled 1–7. Reserve identities are
`red_00..24` and `blue_00..24`. Each reserve uses three rows at signed local
Y = 300, 367.5 and 435 mm, with X = ±234, ±301, ±368 and ±435 mm, plus one token
at the centre of the innermost row. Explicit world reserve poses may override
these defaults via the 50-entry `reserve_positions_m` tuple.

Red joins `r=0` to `r=6`; blue joins `q=0` to `q=6`. Adjacent cells share an edge
under axial coordinates. Red moves first. Version 1 disables the optional pie
rule so colors, robot ownership and goals stay fixed throughout an episode.
Any later competitive comparison needs games with reversed Agent/color
assignments; equal observations do not erase Hex's first-player advantage.
The game's connection and optional swap rules are described in the
[Hexboard rule guide](https://www.hexboard.com/Hex_Connections_Rule_Guide_2017.pdf).

Create `HexMatch(geometry)` and two `ActuatorOwnership` records, then build a
`HexPlayerProtocol`. Authentication chooses the player identity independently
of model JSON. `admit_action(player_id, action)` accepts only:

- `{"type":"placement_intent","piece_id":"red_00","q":3,"r":3}`
- `{"type":"joint_targets","robot_id":"left","joint_positions":{"fr3_left_joint1":0.1},"duration_s":0.1}`

An intent calls `match.begin_move(...)`, reserving a cell and token. It neither
moves a physical body nor changes the logical board. Joint targets are restricted
to the active player's articulation and named joints. The runtime must enforce
native joint limits, hold the other articulation's previous target, and make no
object-pose writes or grasp welds after reset.

The native runtime privately feeds `match.observe(NativeFrame(...))` once per
integration substep. Each frame contains every token's `PieceEvidence` in the
base-centre frame, actual velocities, named **board** support impulse, contact
impulse against **either** robot, and reported penetration. `contacts_complete`
must reflect actual contact collection, never inferred completeness from an
empty list. A snapshot, commanded target or logical board state is not evidence.
Missing penetration evidence remains `None` and rejects settlement.

`HexNativeObserver(env, match).attach()` installs the private adapter after reset.
It calls the original `_read_rigid_contacts` first and consumes only the rows
added by that acquisition. It performs no extra solver/contact step. A single
`/World/Objects/*` native rigid view maps all 50 tokens by actual prim path; each
substep uses one pose batch and one velocity batch. Actual COM offsets read once
from that view convert COM linear velocity to base-centre linear velocity.
The adapter checks board and both-robot contact-filter coverage, native time,
substep sequence and reset epoch. `close()` restores the original contact reader.
`summary()` is small; optional `frame_jsonl` records private complete frames under
shared datasets. Per-pair `min_separation_m` is actual native evidence; zero with
no reported pair means zero **reported** penetration, not a full mesh certificate.

Only a complete consecutive 300 ms interval can commit the move. The piece must
be upright, released by both robots, supported by the board, slow, at board
height, and wholly within the target cell with the frozen edge clearance.
Incomplete/out-of-order frames reset the interval. Displacing an already
committed token or putting an unrequested token on the board invalidates the
match. Terminal matches admit no actions. `SettlementResult.committed` is the
only physical match signal that advances the turn; use `match.commit_log` for
substep/time provenance. `HexRules.play` exists for pure rule tests, never as a
physical API endpoint.

Both public observations use the same synchronized published RGB-D snapshot,
calibration and rules; only identity/color/owned actuators differ. The default
`rgbd` mode omits exact occupancy and legal-cell lists. Explicit
`protocol_smoke_board_state` mode includes these with a privilege label and must
not be reported as a model benchmark score. No opponent reasoning or private
native object/contact state is published.

`task_specification` includes source SHA256s, dimensions, rules and settlement
thresholds. `prepare_task_release` writes a new directory under shared datasets
with `scene.json`, `task.json`, frozen `evaluator/hex` sources, asset hashes and
`release_manifest.json`. Preparation preserves `physics_accepted=false` and
`status=prepared_unvalidated`; attach later native evidence separately. Runtime
loading must explicitly use the frozen evaluator package when claiming that
release, rather than silently importing the editable project copy.

## Minimal two-Agent model chain

`scripts.validation.hex_model_smoke.HexAgentBridge` accepts the existing player protocol
and a client binding for each player. Its two contexts have separate histories.
The requested profile is explicitly `gpt-6-astra`, reasoning `max`, with an
8192-token output budget (configurable from 4096 to 16384). Responses clients are
caller-owned; the bridge loads no credentials and provides no provider/model
fallback. It never makes model calls until the caller invokes `propose` or
`run_turn`.

```python
bridge = HexAgentBridge(protocol, clients={"player_0": red_client, "player_1": blue_client})
blue_ready = bridge.propose("player_1", public_snapshot, joint_command=blue_small_command,
                            images=(rgb_png, depth_png), readiness=True)
red_turn = bridge.propose("player_0", public_snapshot, joint_command=red_small_command,
                          images=(rgb_png, depth_png))
result = bridge.execute(red_turn, published_snapshot=public_snapshot,
                        execute_joint_targets=execute_one_owned_native_command)
```

`public_snapshot` contains `snapshot_id`, `simulation_time_s`, public `rgb` and
`depth` reference strings, and optical `calibration` with `intrinsics_3x3` and
`world_from_camera`. Images are optional PNG bytes ordered RGB, then a grayscale
depth visualization. The second image requires positive
`calibration.depth_scale_m`: zero is invalid, nonzero pixel values multiply that
scale in metres, and values saturate at 255 times the scale. The original metric
depth reference remains public separately. Snapshot content and attached image
hashes must match for both Agents using the same snapshot ID.

The caller supplies a prevalidated small owned-joint command (or hold). Model
output must be exactly a legal `placement_intent` and an unchanged copy of that
advertised `joint_targets` command. Both parts are validated before reservation;
no raw output prose is retained. The callback receives the admitted player ID,
robot ID, owned joint targets and duration; it must preserve the other robot's
targets and use the ordinary native joint action path. Its return value is not
copied because it could contain private state. Keep physics paused during model
calls and supply the latest published snapshot at execution.

A readiness proposal cannot execute, even after the turn changes: request a
fresh proposal on the player's actual turn. The real turn still advances only
through native settlement, and a tiny smoke joint command is not a completed
physical move. Evidence reports sanitized model/usage/latency, proposal status,
dispatch status and whether the authoritative match actually committed. Attach
the separate native observer report for execution evidence. A callback failure
retains the pending move and consumes the proposal, preventing silent replay.
