# Native FR3 multi-hand support

Current scope (2026-10-16): SHARPA single-arm and bimanual development only.
Allegro, LEAP and Shadow assets and historical qualification records are retained;
new integration work on those hands is paused. Current release choices and
acceptance boundaries are in [README](../../../README.md). The evidence below
describes the pinned 2026-10-11 import/control checks, not current task acceptance.

Selection:
    from bench.simulation.isaacsim_backend.robot_registry import list_hands, scene_with_hand
    from bench.simulation.isaacsim_backend.scene_variants import apply_scene_variant
    scene = scene_with_hand(scene, "leap_v1_right")
    scene = apply_scene_variant(scene, "enriched_v2")
    env = FR3DexterousIsaacEnv(scene, device="cuda:0")
    action = env.action_spec()

Always use the selected action_spec.joint_names, joint_limits_rad and initial
positions. The internal physical joint inventory is state_joint_names. SHARPA
is 29 active/29 physical DOF including FR3; Allegro and LEAP are 23/23; Shadow
is 27 active/31 physical. Shadow has four passive 1:1 distal mimic constraints,
an explicit approximation to its 20-motor, 24-joint tendon mechanism. This is
not a validated tendon differential or hardware actuator interface. Mimic
followers cannot be supplied independently through the control API.

Mounts are versioned simulation assumptions, not measured adapters.
scene_with_hand applies the selected assembly's transform and gains; a frozen
scene must retain its robot.assembly_version. Current registered Allegro and
LEAP assemblies include a 10 mm adapter disc with modeled mass and inertia,
centred on the flange; the older 0.12 m clearance description is obsolete.
Shadow's palm orientation was also corrected. Releases from v12 pin the
flange_axis_centred_v2 assemblies (each hand turned 180 deg about the flange axis
relative to its aligned v1 mounting). Exact transforms belong to robot_registry.py
and the versioned assembly, not the historical import examples below; see
[embodiments](../../../docs/agent_bench/embodiments.md). Recalibration requires a new
assembly/version.
LEAP uses the official canonical thumb home [1.104, 1.163, .953, -.138] rad;
mapped from Gym order using the official sim_to_real_indices; the three long fingers start straight. The all-zero thumb intersects the palm.

Downloaded assets and original licenses:
    assets/robots/allegro/v5_80bd4a8 (official Wonik V5, BSD-2-Clause)
    assets/robots/leap/v1_150bc3d (official V1 simulation repository, MIT;
        separate CAD restrictions noted in LICENSE_NOTES.txt)
    assets/robots/shadow/sr_common_172c3d9 (official Shadow E3M5, BSD-3-Clause)
Each SOURCE.json pins upstream commit, archive SHA256 and every upstream file.
Derived assemblies include source URDF/mesh/license hashes, mass/inertia audit,
mount assumptions and native USD import manifest. Do not edit versioned assets.

Physics:
    Original URDF masses and collision models are preserved unless explicitly
    recorded by the selected assembly. LEAP uses convex decomposition because
    whole-link convex hulls fill motor cavities (the official Gym setup also
    enables VHACD). The LEAP v4 USD used in the 2026-10-11 checks overrides 17 collision
    meshes to native convexDecomposition (64 hulls, 64 vertices, voxel resolution
    300000). Isaac 6.1 importer collision_type alone does not change existing
    URDF collision meshes; failed v2/v3 imports are preserved as diagnostics.
    Later aligned assemblies use finer decomposition and a separately versioned
    thumb/palm filter opt-in; see the embodiment notes for their distinct scope.
    Allegro applies only four official SRDF palm/proximal
    structural exclusions, not its broad Never/cross-finger exclusions.
    Self-collision otherwise remains enabled. Shadow fixed links are merged so
    zero-inertia fixed point-mass markers do not become independent rigid bodies.
    Seven selected Shadow structural exclusions come from official sr_interface
    commit 01aca16f5a8d0ff468ff76eabf7c30af2df6c539; broad Never/cross-finger
    exclusions are not imported. Isaac 6.1 NewtonMimicAPI is replaced by native
    PhysxMimicJointAPI for the four passive distal coordinates.
    Shadow adds 0.0001 kg m^2 armature to every hand DOF for numerical
    conditioning of tiny inertias with native passive mimic and self-collision.
    This is a recorded simulation assumption, NOT measured motor inertia.
    Source link masses/inertias remain intact; cross-finger collisions remain.
    Any native driver diagnostic override is recorded in run metadata.
    Direct control always sets joint targets. Object state is reset-only.

Reusable physical assets:
    assets/task_parts/manipulation_collection/v2: hollow mug and bowl, capped
    bottle, tin can, wooden block, mallet, screwdriver, open tray, peg rack,
    draping cylinder. Density/friction/inertia are explicit simulation
    assumptions. Hollow containers use separate collision walls, never a convex
    hull sealing the opening. These are original procedural CC0 assets.
    assets/scenes/manipulation_collection/v2: household and cloth-care scenes.
    enriched_v2 adds distractor objects/fixtures while preserving the frozen
    four-task object IDs, placements and evaluator inputs. Scores must be
    stratified by hand ID, scene variant, and effective physics/scene hashes.

These assets do not imply a trained policy, task success, MANUS hardware
connection, sensor calibration, or equivalent difficulty across hands.
Native validation outputs live under datasets/fr3_sharpa_bench/isaacsim/
validation/multihand_20261011; only explicitly qualified results may be cited.

Official references:
https://github.com/Wonikrobotics-git/allegro_hand_ros2_v5
https://github.com/leap-hand/LEAP_Hand_Sim
https://github.com/shadow-robot/sr_common
https://shadowrobot.com/dexterous-hand-series/
https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_6_0/urdf_mjcf_importer_exporter_pipeline.html

Qualified evidence index:
    datasets/fr3_sharpa_bench/isaacsim/validation/multihand_20261011/qualification_index.json
    SHARPA: cloth_care_01 (29 active/29 physical); Allegro: allegro_v5_right_02
    (23/23); LEAP: leap_v1_right_05 (23/23); Shadow: shadow_hand_e_right_09 (27/31).
    Each runs 1.2 seconds = 2400 native physics steps with all active joints
    commanded, self-collision enabled, finite states, limits/base/contact audit,
    deterministic reset and rendering that does not advance physics. This is
    short-horizon import/control qualification; grasping and long episodes need
    task-specific validation. Every run pins its own immutable source snapshot.
    household_01 independently loads all ten objects/fixtures and drops a block
    onto each static fixture to prove positive load-bearing native contacts.
    cloth_care_01 has 4225 moving vertices, plane-fit RMS 34.3 mm, edge-length
    ratios 0.9869 to 1.0096, and 130 vertices on the cylinder top. This is live
    geometry support evidence; the rigid contact API does not expose cloth
    contact impulses. Failed runs are retained and excluded from qualification.

Historical qualification tools (not a byte-for-byte replay of the old runs):
    Do not rebuild frozen assets or resume paused-hand experiments from this list.
    Any newly authorized run needs a new output path, pinned source/assembly,
    and a GPU selected by UUID/PCI under AGENTS.md. Bind CUDA_VISIBLE_DEVICES to
    that single UUID and set ISAAC_BENCH_SINGLE_CUDA_DEVICE=1; cuda:0 below then
    denotes the process-local device, not a fixed host GPU index.
    python scripts/assets/prepare_multihand_sources.py
    .venv/bin/python scripts/assets/build_manipulation_objects.py
    .venv/bin/python scripts/assets/prepare_manipulation_scenes.py
    ISAAC_ALLOW_UNVALIDATED_DRIVER=1 scripts/isaac_python.sh -m bench.simulation.isaacsim_backend.multihand_smoke --hand leap_v1_right --output <开发存储>/datasets/YOUR_NEW_RUN --device cuda:0
    .venv/bin/python scripts/assets/audit_multihand.py <开发存储>/datasets/YOUR_NEW_RUN --output <开发存储>/datasets/YOUR_NEW_AUDIT.json
    ISAAC_ALLOW_UNVALIDATED_DRIVER=1 scripts/isaac_python.sh scripts/assets/probe_manipulation_scene.py --output <开发存储>/datasets/YOUR_NEW_SCENE_RUN --device cuda:0
    .venv/bin/python scripts/assets/audit_manipulation_scene.py <开发存储>/datasets/YOUR_NEW_SCENE_RUN --output <开发存储>/datasets/YOUR_NEW_SCENE_AUDIT.json

Additional official references:
https://github.com/shadow-robot/sr_interface/tree/01aca16f5a8d0ff468ff76eabf7c30af2df6c539
https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/joints/physx_joint_schema.html
https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/dev_guide/guides/articulation_stability_guide.html
