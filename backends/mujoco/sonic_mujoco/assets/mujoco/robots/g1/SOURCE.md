# G1 asset source

Copied from `xiaohu-art/GR00T-WholeBodyControl` at commit `ba0f0e83862c19a0e5f86aa0b4f29786da2727cf`:

- `gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml`
- only the mesh files referenced by that XML

The XML's `meshdir` is adjusted from `meshes` to `../../robots/g1/meshes` because the new root scene lives under `assets/mujoco/scenes/g1`.

The serial neck base, yaw link, pitch link, inertial parameters, and three
`open2dof_*.STL` meshes are adapted from `co1one/open_2DOF_robot_head` at
commit `43322b1929a53c0d53fe96a92c59ac6644a076df`. The source URDF is:

- `urdf/g1_2dof_head/g1_2dof_head.urdf`

The URDF joint transforms are preserved. Joint names and the simulator's
conservative limits remain compatible with the existing PICO neck command.
The yaw axis sign is expressed in the simulator convention where positive yaw
turns left. Virtual camera poses are local simulation conventions and are not
claimed as calibrated D435, D455, or ZED extrinsics.

The virtual ego stereo pair has a 60 mm baseline and a neutral 20 degree downward
pitch. `head_camera_scan` retains its level reconstruction anchor. The wrist
cameras are mounted 100 mm above each wrist frame, with a 10 degree downward
pitch, to avoid the hand occupying the entire image. These are massless virtual
mounts, not physical camera housings or a measured real-robot configuration.

Source code and software components in the source repository are licensed under Apache License 2.0. See the source repository's `LICENSE` for details.
