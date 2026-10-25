# Unitree H2 asset source

The H2 MJCF and the 32 referenced STL meshes are copied from Unitree
Robotics' official `unitree_mujoco` repository at commit
`4134cb5dc7ff1ba7f484deda48b5274b58694519`:

- [`unitree_robots/h2/h2_mujoco.xml`](https://github.com/unitreerobotics/unitree_mujoco/blob/4134cb5dc7ff1ba7f484deda48b5274b58694519/unitree_robots/h2/h2_mujoco.xml)
- `unitree_robots/h2/meshes/*.stl`

Upstream: <https://github.com/unitreerobotics/unitree_mujoco>

The upstream `h2_mujoco.xml` is renamed to `h2.xml`, and its `meshdir` is
adjusted so it resolves when included by packaged scene files. This copy also
adds named egocentric stereo and wrist cameras for the shared observation
interface. It does not change Unitree's bodies, inertias, joints, collision
geometry, actuators, sensors, or motor order. Camera mounts are simulation
conventions, not calibrated H2 sensor extrinsics.

This upstream MJCF ends at `left_wrist_yaw_link` and
`right_wrist_yaw_link`. It contains no independently addressable hand body or
finger joint; any fixed end geometry is part of the wrist mesh. H2 Plus
dexterous-hand assets are a separate embodiment and are not included here.

The package-authored empty scene uses a named `home` keyframe adapted from
Unitree's pinned
[H2 MjLab configuration](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/assets/robots/unitree_h2/h2_constants.py).
The XML stores qpos in MuJoCo tree order; the runtime resolves every joint by
name and validates that keyframe against the upstream MJCF actuator order.
Nominal PD gains are taken from the pinned SDK2 C++ low-level example linked
below.

This integration follows the actuator order encoded by the pinned upstream H2
MJCF (`unitree_mujoco@4134cb5`), referred to here as the upstream MJCF order.
It is an internal simulation index and is not claimed to be a firmware- or
channel-independent physical motor-slot order. Real-hardware adapters must
provide and validate an explicit name-to-slot map for the target firmware and
control channel. The order is identified in code as
`unitree_mujoco_h2_4134cb5_actuator_order`.

For provenance, the pinned Unitree SDK2 C++ low-level example uses the same
order, while other official Python and high-level examples have used different
waist, wrist, ankle, or head slot conventions:

- [SDK2 C++ low-level example](https://github.com/unitreerobotics/unitree_sdk2/blob/9754cd153af3da471b0fe5f3aa535e426fb11db3/example/h2/low_level/h2_ankle_swing_example.cpp)
- [SDK2 Python low-level example](https://github.com/unitreerobotics/unitree_sdk2_python/blob/65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5/example/h2/low_level/h2_ankle_swing_example.py)
- [SDK2 Python high-level example](https://github.com/unitreerobotics/unitree_sdk2_python/blob/65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5/example/h2/high_level/h2_arm_sdk_dds_example.py)

The files are distributed under Unitree's
[BSD 3-Clause license](https://github.com/unitreerobotics/unitree_mujoco/blob/4134cb5dc7ff1ba7f484deda48b5274b58694519/LICENSE).
A verbatim copy is included beside the model as `LICENSE`.
