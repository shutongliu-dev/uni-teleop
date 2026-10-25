# Task 4: opposing-arm tower extraction

The new scene profile `trays_v1` builds a **12-layer, 36-block tower directly on
the original table**, with a physical collection tray for each player. Blocks
measure 200 × 45 × 30 mm. Parallel neighbors have a 10 mm gap and each end
overhangs the crossing layer by 22.5 mm, compared with the earlier 1 mm gap and
6.5 mm overhang. These are geometric access improvements, not a grasp or
extraction qualification. The tower is 360 mm tall, from table z = 0.78 m to
z = 1.14 m; there is no tray, plinth or hidden support under the tower.

The current asset version is `assets/task_parts/living_room/tower_extraction/v3`.
The task release is
`<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/tower_extraction/v8`.
Its manifest SHA256 is `cc31b454e8a1421bbbc85bbfdd72f615b2a585652150bed3da1ca40141644499`;
`definition.py` binds the active path and hash. The frozen manifest retains
`prepared_unvalidated`, `physics_accepted=false` and
`task_policy_success_claimed=false`. Existing v5/v6/v7 releases and their
failed or incomplete reset evidence remain preserved.

The new tabletop image（开发存储 `demo/interface/scenes/living_trays_towers_20261019/tower_extraction/tabletop.png`）
and workcell image（开发存储 `demo/interface/scenes/living_trays_towers_20261019/tower_extraction/workcell.png`）
show the 12-layer tower on the table and the two empty collection trays. Both
are actual 1920 × 1440 RTX images at 128 spp, with provenance beside them.
The rendering source is `development/living_trays_towers_20261019/source01`;
it recorded 4000 explicit 0.5 ms task-object settling steps with static robot
meshes and no robot actions. This is geometry/render evidence, not the
canonical referee's reset acceptance or extraction success.

## Numerical profile v2 candidate

The v8 candidate selects `tower_extraction_tgs_external_forces_v2`:
`physics_dt_s=0.00025` and `enable_external_forces_every_iteration=true`.
Both settings apply to the whole PhysicsScene, including both robots and all
task rigid bodies. The public control period remains 20 ms, now containing
80 native steps rather than 40. Robot geometry, masses, inertias, gains,
controller algorithms, material settings and solver iteration counts retain
their existing definitions. The discrete integration changes; the existing
arm gravity compensation also reads and applies its torques before each
0.25 ms step, so its sampling interval changes with the integrator.

The scene's primary and fallback timestep, frozen evaluator timestep and
profile provenance must agree. Historical v6/v7 releases retain 0.5 ms and
remain loadable. The native velocity readings, endpoint-motion bounds,
contact support checks, 1.5 mm/s stability threshold and full one-second
stable window are unchanged. The external-force flag's metadata records its
USD readback; it is not an independent native PhysX flag-getter measurement.

The isolated half-step probe supports this candidate selection only. The
separate canonical result below covers reset and a bounded robot-control
smoke check. Preserve v7 and its own canonical attempt as independent history,
alongside the v5/v6 failures below.

## Current v8 bounded canonical result

The source02 canonical smoke report（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/native_extraction02/report.json`）
records `passed=true` for seed 0, `player_0/front`, episode
`eeef751dc81148f3a370fbd92fa00166`. Its frozen source manifest SHA256 is
`6ad8ababa6b216297772d18dccb239ce37585aeda8a91ba72ef58cd494fe1b7f`, bound
to the v8 release manifest above. The authenticated public poll first observed
`await_intent` at 1.2 s. The smoke then completed its initial hold, requested a
2 mm upward wrist motion and completed the final hold at 1.6 s. The measured
wrist-target error was 0.224 mm and maximum joint change was 0.005476 rad.
All five configured RGB-D streams (`overview`, `task`, `room`, `front_wrist`,
`rear_wrist`) were retrieved with matching media hashes. The episode records
`physics_valid=true`; the native worker exited 0.

The independent frozen replay（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/canonical_review02.json`）
checks 6401 frames including the baseline, 6400 native-step receipts and
81 control observations. The unmodified frozen referee first enters
`await_intent` at tick 4003, 1.00075 s. Its reset predicates hold continuously
from tick 3 (0.00075 s) through tick 4003: 4001 valid endpoints with a
recorded dwell of 0.9999999999999178 s,
accepted by the original one-second comparison with its unchanged 1e-12 s
floating-point tolerance. All native receipts are complete, consecutive and fresh;
control states match the native archive exactly. This is a read-only replay of
the recorded run, not a second simulation. The independent public-motion review（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/http_motion_review02.json`）
also confirms unchanged commanded targets during all six reset holds and the
reported wrist error and joint response. The independent metadata review（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/canonical_metadata_review02.json`）
passes all 28 source, import, configuration and result-identity checks. The
qualification record（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/qualification.json`）
indexes these source02/v8 results separately from later integration CPU checks.

The harness finished the episode as `cancelled`; no extraction, collection,
completed game turn or winner is qualified. The run uses in-process ASGI HTTP
with an isolated native worker and no model calls. It does not establish
rear-first robot motion, a loaded tray placement, handoff, full gameplay or
physical fidelity. There are no robot/block contacts or completed turns in
the recorded native frames.

## Historical v7 screening result

The v7 `tower_extraction_tgs_external_forces_v1` release selected per-iteration
external forces at 0.5 ms. Its separate `source01` canonical screening run（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/native_extraction01/report.json`）
stopped after its 1.4 s smoke readiness budget without admitting motion. The
smoke report is `passed=false`; this was not the frozen referee's 10 s reset
timeout and does not establish a v7 `invalid_reset_not_stable` terminal result.

The independent frozen replay（开发存储 `datasets/fr3_sharpa_bench/development/living_task4_reset_repair_20261019/canonical_review01.json`）
checks all 2801 native frames, including the baseline, and remains
`reset_pending` throughout. Its longest continuous reset-valid endpoint dwell
is 7.5 ms, below the required one second. No robot/block contacts, collected
blocks, completed turns or winner are established. This v7 result remains
independent of the v2 half-step candidate.

## Historical v6 native reset result

The separate `source02` canonical check（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/native_extraction01/report.json`）
finished with `passed=false`. Front-start episode
`5aef9574d0fe4a44ba167c74cd953e4a` ran the complete 10 s reset window, comprising
20000 native 0.5 ms steps. Its outcome is `task_failure`; the frozen final score
is `interrupted / invalid_reset_not_stable`. The game aborted with zero
completed turns for either side, no collected blocks, and no winner or loser.
It never admitted the subsequent wrist micro-motion or extraction.

The canonical episode result records `physics_valid=true`, and all five
required general physics-audit checks passed. The sampling receipt records
20000 acquisitions and deliveries, each consumed once, with no raw-array or
state-capture reuse. These checks do not satisfy the scene's reset stability
gate. The smoke harness reported `episode_terminated` during `reset_wait_49`
at 10 s; it did not end because of wall-time or action timeout. Its report omits
`physics_valid` and `worker_exit_code`; those missing fields are not false
values or evidence of a worker exit code. The report explicitly retains
`task_success_qualified=false` and `physical_fidelity_qualified=false`.

The reset review（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/task4_reset_review01.json`）
checks adjacent native-frame intervals at 50 ms spacing. All 200 sampled points
fail the motion gate, with 24–28 moving blocks per point. At the final point,
26 blocks fail that gate and the maximum native linear speed is approximately
3.125 mm/s, above the unchanged 1.5 mm/s threshold. Sampled support, original
slots, deep-contact, hands-clear and collapse checks have no failures. This
sampling diagnostic does not certify every-frame support or a complete stable
dwell. Empty collection trays establish contact-filter availability, not loaded
placement. The experiment qualification record（开发存储 `datasets/fr3_sharpa_bench/development/living_trays_towers_20261019/qualification.json`）
links this v6 evidence separately from the preserved historical v5 failure.
For that v6 check, room v7, shared controls, physics settings and the one-second
stability gate were unchanged; no robot extraction or gameplay success is qualified.

## Historical v5 reset evidence

Version 5 fixed the scene adapter's exact worktop contact name and normalized
angular endpoint bounds, and preserves the first invalid-evidence reason. Its
reference rules, referee, reset, robot and physics are byte-identical to v4.
The failed v4 native attempt is retained; replaying its 41 recorded frames
through v5 reaches only `reset_pending`, not reset readiness or task success.

The shared `fr3_sharpa_instances/namespaced_v1` contract composes two unchanged
FR3/right-SHARPA assemblies at `(.65, ±.76, .78)` m. These are two right hands,
not the card-room left/right pair. Authenticated `/turn-games` binds `player_0`
to `front` and `player_1` to `rear`; only the active player's targets may change.
The scene runtime reads native bodies and contacts and passes them to the
release-frozen scorer. No task intent changes physical state.

The historical v5 native front-start check completed the original 10 s reset window
and closed with `invalid_reset_not_stable` (`task_failure`, `physics_valid=true`).
The frozen referee never admitted robot micro-motion or extraction. This was
not a wall-time, action or launcher timeout. The rear-start check shares the
same physical initialization and was stopped after that common reset blocker
was confirmed; it was still `reset_pending` before cancellation at 6.62 s.
There is no full-window or motion pass claim. Authentication and acknowledged-drive holds have separate evidence.
That initial tower did not pass the scene stability requirement; the bench did not
relax thresholds, shorten the one-second stability requirement or tune physics.
See the source-bound integration report（开发存储 `datasets/fr3_sharpa_bench/development/living_bench_integration_20261018/report.json`）
for the original failure, cancellation and native sampling records.

## Authored scene candidate

`assets.author_task(stage, root_path='/World/LivingTask', *, presentation=False,
profile='trays_v1')` adds 36 independent wooden rigid bodies, two physical trays
and thirty noncolliding collection-slot markings. It does not add a room,
robot, light or PhysicsScene. Omitting the profile preserves the historical
18-block, six-layer author and flat table collection marks.

Each tray has a 6 mm floor resting on the original table, an interior floor at
z = 0.786 m, and four 6 mm thick walls rising 12 mm above the floor. All ten
parts have real static primitive-box collision. Static fixation is an explicit
task-fixture assumption; geometric construction mass is recorded separately
from simulation mass, and the fixtures have no dynamic mass, articulation or
joint. Floor and wall contact actor IDs are distinct, so a wall cannot satisfy
the floor-support criterion. Front/rear trims are blue/rust; the thirty slot
marks remain noncolliding ink on the tray floors.

Each side has 15 single-layer slots in a 3 × 5 grid, sufficient for the default
15-turn allowance. Slot-to-slot block clearances are 40 mm along X and 15 mm
along Y. The nearest outer tray rim is 64 mm from the tower footprint, and its
top lies 12 mm below the bottom of the lowest extractable layer. These authored
clearances do not establish a collision-free robot trajectory.

Bodies live at `<root>/Blocks/block_00` through `block_35`. A full-size primitive
box encloses each block's 0.4 mm rounded visual edges. Density 650 kg/m³ gives
175.5 g per block; uniform-box inertia and static/dynamic friction 0.45/0.35
are explicit uncalibrated engineering priors. Shared CC0 `kitchen_wood` PBR maps
are referenced with provenance and hashes. Small two-digit vector ink marks
travel with each block and have no collision.

The JSON receipt includes dynamic paths, initial poses/bounds, mass/inertia,
configuration identity, physical fixtures, floor/wall contact paths, geometric
clearance audit, authoring-source hashes, mirrored slots, dependencies and scope. Both values of
`presentation` preserve this same task4 tower reset; neither claims settling,
extraction or competitive success. CPU checks in `tests/test_living_tower_assets.py`
cover independent bodies, table bounds, initial overlap, materials and the
absence of marker collision. Native stability and manipulation remain separate.

## Public game rules

- Front and rear alternate. Even episode seeds start front; odd seeds start rear.
- The bottom layer is protected. The **highest complete original layer and every
  layer above it** are also protected. Complete means all three original blocks
  remain uncollected and within the published pose tolerances of their original
  slots. Legality is checked at selection and again at first contact.
- Select one legal block. Selection can change or be cancelled before first
  physical robot contact. After contact it is locked for the turn; pushing,
  pulling, releasing and regrasping the same block are allowed.
- Put that block in a free collection slot on your own side. **Do not put it back
  on the tower.** Collection is single-layer; the full oriented block must fit.
- A turn commits only when the block has direct support from its own tray floor
  (the table in the historical profile), both hands are clear,
  all remaining tower blocks have support paths to the table, all previously
  collected blocks remain in their own slots, and the entire assembly stays
  within the published motion bounds for at least one second. Handoff occurs
  after that interval, never merely on a release command.
- A nonselected, uncollected tower block dropping at least `0.8 H` from its
  original height or tilting at least 45 degrees is the reference collapse rule.
  The selected block is exempt only after actual contact, so extracting,
  dropping and recovering it alone does not automatically lose the game.
- After current-player tower contact, collapse before handoff loses that player
  the game, including delayed collapse while settling. Handoff clears causal
  attribution. If the next player has not contacted the tower and it collapses,
  the episode aborts without a winner. Executing a hold or a motion away from the
  tower does not establish contact causality.
  First contact and collapse in the same normalized interval have no proven
  event order and therefore abort; prior-interval tower contact is required to
  attribute the loss. Minimum interval height and maximum interval tilt retain
  evidence of a fall followed by an endpoint rebound.
- Noncurrent API commands are rejected without execution or changing turns.
  **Actual** noncurrent robot contact or an execution receipt aborts the episode
  without a winner. Undeclared direct contact with another block also aborts.
  Physical interference and invalid evidence take precedence over loss scoring.
- The default equal-turn cap is half the initially legal inventory, rounded
  down: 15 turns each for the new 12-layer tower, six for the historical
  six-layer topology. A smaller explicit
  limit is a diagnostic/configuration choice. Reaching the cap requires both
  sides to finish the same number of turns and results in a draw. If no legal
  targets remain, a supported, released stable interval also produces a draw.
  An oracle cannot declare a player defeated because a block seems unreachable.
- Timeout or budget exhaustion is incomplete, never a manufactured loss.
  Reset, invalid initial support, missing evidence and adapter failure abort
  without a winner. Terminal outcomes are latched.

These are provisional simulation adjudication thresholds. They do not constitute
physical calibration or proof that the chosen primitives cover every real-world
way a tower can collapse. The public rules and thresholds are bound to the
configuration hash. The published release already binds its candidate dimensions. Separate reference
configurations continue to require explicit geometry.

## Configuration and API

`task_configuration(geometry, *, seed=0, max_turns_per_player=None,
protected_bottom_layers=1, thresholds=None, collection_trays=None)` validates the shared geometry and
returns a new configuration. `rules.public_rules` takes the rule options;
`collection_trays` belongs to the task configuration's physical layout.
`geometry['block_size_m'] = [L, W, H]` is mandatory for the parameter factory.
The `trays_v1` author explicitly chooses the dimensions above and passes
`tower_trays.extraction_collection_trays()`. Each tray specification binds its
floor actor, distinct wall actors, interior XY bounds, floor height and slot
pitch. Omitted `collection_trays` preserves historical table collection.
Collection slots are mirrored by a 180-degree rotation around the tower center,
fit the parent table and have positive gaps. Their capacity equals the entire
configured per-player turn limit. A table that cannot accommodate that capacity
raises `ValueError`; objects are never stacked or silently overlapped to fit.
Each slot's XY acceptance bounds include one quarter of the layout gap per side
for placement tolerance, leaving at least half the gap between neighboring
acceptance volumes and preserving table-edge/tower clearance.

The factory includes the full pinned `parent_environment`, rules, geometry,
inventory, collection slots and `configuration_sha256`. The SHA256 is over UTF-8
JSON with sorted keys, compact separators and no hash field. The referee rebuilds
the canonical factory output and compares the entire configuration at construction,
so a substituted parent, rule, block or slot cannot be accepted just by recomputing
its hash. Parameter-only factory configurations retain `status='planned'`, `native_adapter=None`,
`source_release=None` and `native_execution_available=False`.

```python
configuration = task_configuration(approved_geometry, seed=episode_seed)
game = TowerExtractionGame(configuration, trusted_normalized_input=True)
state = game.observe(service_owned_observation)
reply = game.command('front', 'select', 'block_03')
state = game.snapshot()
# The host calls finish before any further observation once its budget ends.
state = game.finish('budget')
```

Other commands are `cancel`, `push`, `pull`, `regrasp`, `release` and `wait`.
Commands express intent and permission only; they do not execute a controller.
`finish` accepts `timeout`, `budget`, `reset`, `evidence_missing` and
`adapter_failure`. An expired host budget must be delivered before another
observation; this reference has no wall-clock scheduler or native step budget.

The constructor defaults to `trusted_normalized_input=False`. In that mode an
observation aborts with `trusted_adapter_unavailable`. Setting the flag is a
**service-side integration assertion, not authentication**. Agent-authored JSON
must never be passed to `observe`, and the flag must not be exposed to an agent.
The scene-owned `runtime.py` identifies native bodies and robot sides, reads
complete substep evidence and enforces command authorization. This reference
module remains independent of simulator APIs. During reset settling only unchanged
drive targets are admitted; missing evidence cannot manufacture a completed game.

## Strict normalized observation contract, version 1

Coordinates use world axes, origin `(tower_center_x, tower_center_y, table_top_z)`
and length unit **block height H**. Time is seconds; force is N and impulse is
N·s. Quaternions are normalized **wxyz**. Every numeric field must be finite;
booleans are not numbers. Unknown/missing fields abort rather than being ignored.

The top-level mapping has exactly these fields:

| Field | Required meaning |
| --- | --- |
| `schema` | `living_tower_extraction_normalized_observation.v1` |
| `configuration_sha256` | Exact factory configuration digest; a mismatched frame aborts. |
| `stream_id`, `episode_id`, `reset_id` | Nonempty immutable identities for one episode; any change aborts. |
| `tick`, `time_s` | Start at `0, 0`; tick increments exactly one and time strictly increases by at most 50 ms. |
| `blocks` | Mapping containing every configured block, including collected blocks. |
| `contacts` | Complete relevant contact evidence, including zero-load touches; no filtered support-only stream. |
| `executed_sides` | Unique `front`/`rear` receipts for actual robot execution in this interval; rejected API commands do not appear. These receipts enforce ownership, not collapse causality. |

Each block has exactly `center_h` (3), `quaternion_wxyz` (4),
`linear_velocity_h_s` (3), `angular_velocity_rad_s` (3),
`interval_max_linear_speed_h_s` (scalar),
`interval_max_angular_speed_rad_s` (scalar), and `hand_distance_h`.
Each block also requires `interval_min_center_z_h` and `interval_max_tilt_rad`
over the entire interval, for collapse detection; these extrema must include
the endpoint height and tilt.
The latter is a mapping of both sides to the **minimum signed surface clearance
over the entire interval**. Interval speed maxima cover the entire interval,
including excursions that return to the same endpoint; they may not be less
than the endpoint speed. Endpoint differences are independently checked too.
For tick zero these summaries describe the initial sample.

Every contact row has exactly:

| Field | Required meaning |
| --- | --- |
| `a`, `b` | Distinct actor IDs: block ID, `table`, `floor`, `hand/front`, `hand/rear`, `robot/front`, `robot/rear`, or a configured collection-tray floor/wall actor. Robot actors include the movable arm links, excluding static stands. |
| `normal_z_on_a` | World-Z component of the unit contact normal force on `a`, in `[-1,1]`; its sign reverses for `b`. |
| `normal_impulse_n_s` | Nonnegative normal impulse integrated over the observation interval. |
| `point_h` | Contact point in normalized world coordinates at the current endpoint. |
| `separation_h` | Minimum signed separation over the interval; negative is penetration. |
| `supported_interval_s` | Duration this actor pair continuously provides the reported support within the interval, zero if only touching. It may not exceed elapsed observation time. |
| `minimum_normal_force_n` | Minimum normal support force during that supported interval; zero for a zero-load touch. Force × duration cannot exceed the recorded impulse. |

A stable support edge needs positive impulse and minimum force, upward normal,
full-interval support duration, acceptable separation and a contact point on
the upper block's bottom face and the lower block's top face (or inside the
table plane). Tower support graphs must terminate at the actual table. A
collected block instead requires direct support from its own configured tray
floor, with a geometrically valid point inside that floor; opposite-side floor,
tray walls, or another collected block cannot substitute. Jumping a
layer, a support point outside either block, support from a selected/collected
block, and floor contact cannot qualify the remaining tower. Excessive reported
penetration anywhere in the relevant contact set clears the stable window.
Contact with either side's movable robot is retained even at zero impulse.

The reference validates this numerical contract; it cannot prove the values came
from a simulator. A trusted adapter must compute interval summaries from all
underlying native substeps and fail closed if any source interval is missing.
The engineering adapter records every native substep at the selected frozen
timestep (0.5 ms for v5/v6/v7; 0.25 ms for the v2 profile candidate) and computes
conservative clearance from authored collision enclosures transformed by actual
native body poses. Its startup and sampling evidence is recorded separately in
the integration report（开发存储 `datasets/fr3_sharpa_bench/development/living_bench_integration_20261018/report.json`）.
Consecutive normalized ticks and synthetic tests alone do **not** establish
physical correctness, manipulation success or release qualification.

## CPU verification

Run `python -m pytest -q tests/test_living_tower_extraction.py` from the repository.
Fixtures use explicitly artificial dimensions, numerical poses and generated
contact records; their instantaneous pose changes are test inputs, not robot
motions. Tests cover release/support dwell, mirrored capacity, protected layers,
intent locking, side permissions, corruption, loss attribution, aborts, incomplete
budgets and balanced/exhaustion draws. GPU and native physics are not involved.
