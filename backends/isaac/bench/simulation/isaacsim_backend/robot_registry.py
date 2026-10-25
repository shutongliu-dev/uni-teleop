"""Pinned, explicitly different dexterous hand embodiments for native Isaac.

Model IDs identify simulation embodiments, not interchangeable hardware. In
particular Shadow's 24 URDF joints use an explicit 1:1 distal mimic approximation, which does
not implement its real tendon differential transmission. All FR3 mounting transforms are uncalibrated simulation adapters.

Mount conventions. Every hand has a hand frame H on the flange axis in which the
fingers point along +z, the palm normal along +x and the thumb toward +y, with the
hand's wrist-end mating face centred on the flange axis. SHARPA mounts directly (its
with_flange root is the flange interface); a hand without one sits on an explicit
aluminium adapter disc (real mass and collision). Yaw about the flange axis is a
mounting choice (4-bolt pattern: 0/90/180/270 degrees); upstream SHARPA does not
specify it ("FR3 to hand adapter transform still needs calibration").
* flange_axis_centred_v1: H = fr3_link8 (identity yaw, assumed in 2026-10). The
  natural top-down right-hand posture in front of the robot (palm toward the robot,
  thumb to its left) then needs fr3_joint7 at about +-180 deg, beyond its +-174.8
  deg limit, so tasks were done back-handed (palm away, thumb right).
* flange_axis_centred_v2: H = fr3_link8 rotated 180 deg about the flange axis; that
  posture sits at the middle of joint 7's range. Releases from 2026-10-15 pin it.
Scenes without a pinned assembly keep each hand's v1-era assembly unchanged.
"""
from copy import deepcopy
from pathlib import Path
import math

DEFAULT_HAND = 'sharpa_w01_right'
# 2026-10-15 user decision: build the bench with SHARPA first ("现在先用sharpa搭建 其他手资产你保存好
# 目前暂缓接入"). The other hands keep their registry entries, versioned assemblies,
# releases and records (inventory: assets/robots/assemblies/PAUSED_HANDS_20261015.json),
# but their integration is paused: no new work, and they are not part of the active bench.
PAUSED_HANDS = ('allegro_v5_right', 'leap_v1_right', 'shadow_hand_e_right')
MOUNT_CONVENTION = 'flange_axis_centred_v1'
MOUNT_CONVENTIONS = {'flange_axis_centred_v1': {'hand_frame_yaw_rad': 0.},
                     'flange_axis_centred_v2': {'hand_frame_yaw_rad': math.pi}}
ADAPTER_DISC_10MM = {'thickness_m': .010, 'radius_m': .0315, 'density_kg_m3': 2700., 'collision_clearance_m': .0005,
                     'description': 'Flange-sized (ISO 9409-1-31.5) aluminium disc; simulation adapter, not a vendor part'}
HANDS = {
    DEFAULT_HAND: {
        'label': 'SHARPA Wave/W01 right', 'hand_dof': 22, 'prefix': '',
        'urdf': 'robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf',
        'license': 'robots/sharpa/wave_01/upstream/LICENSE.txt',
        'license_id': 'Upstream LICENSE.txt', 'package_roots': {},
        'root_link': 'right_hand_flange', 'palm_link': 'right_hand_C_MC',
        'fingertip_links': {n: f'right_{"pinky" if n == "little" else n}_fingertip'
                            for n in ('thumb', 'index', 'middle', 'ring', 'little')},
        'assembly_version': 'isaacsim_6_1_v1', 'release_assembly_version': 'isaacsim_6_1_v2',
        'assemblies': {'isaacsim_6_1_v2': {'mount_convention': 'flange_axis_centred_v2', 'mount_rpy_rad': [0., 0., math.pi]}},
        'mount_translation_m': [0., 0., 0.], 'mount_rpy_rad': [0., 0., 0.],
        'hand_kp': 3., 'hand_kd': .1, 'source_url': 'https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e',
        'pinch_links': {f: [f'right_{f}_elastomer'] for f in ('thumb', 'index', 'middle')},
        'control_model': '22 independent position-driven hand joints; motor calibration not measured',
    },
    'allegro_v5_right': {
        'label': 'Wonik Allegro V5 right, type B', 'hand_dof': 16, 'prefix': 'allegro_',
        'urdf': 'robots/allegro/v5_80bd4a8/upstream/src/allegro_hand_controllers/urdf/allegro_hand_description_right_B.urdf',
        'license': 'robots/allegro/v5_80bd4a8/upstream/LICENSE', 'license_id': 'BSD-2-Clause',
        'package_roots': {'allegro_hand_controllers': 'robots/allegro/v5_80bd4a8/upstream/src/allegro_hand_controllers'},
        'root_link': 'palm_link', 'remove_root': 'world', 'palm_link': 'palm_link',
        'fingertip_links': {'index': 'link_3_0_tip', 'middle': 'link_7_0_tip', 'ring': 'link_11_0_tip', 'thumb': 'link_15_0_tip'},
        # v3: flat wrist face (was an 18.0 mm air gap, 4.5 mm off axis) seated on
        # the adapter top and centred on the flange axis; orientation already matched.
        'assembly_version': 'isaacsim_6_1_v3', 'adapter': ADAPTER_DISC_10MM,
        'mount_translation_m': [.0045, -.000023, .112029], 'mount_rpy_rad': [0., 0., 0.],
        # v4: flange_axis_centred_v2 (v3 rotated 180 deg about the flange axis).
        'release_assembly_version': 'isaacsim_6_1_v4',
        'assemblies': {'isaacsim_6_1_v4': {'mount_convention': 'flange_axis_centred_v2',
                                           'mount_translation_m': [-.0045, .000023, .112029], 'mount_rpy_rad': [0., 0., math.pi]}},
        'controller_fingertip_semantics': 'Upstream link_*_tip frames: centres of the hemispherical fingertips.',
        # Evaluator pinch channels: each finger's distal segment (fingertip + distal link bodies).
        'pinch_links': {'thumb': ['allegro_link_15_0_tip', 'allegro_link_15_0'], 'index': ['allegro_link_3_0_tip', 'allegro_link_3_0'],
                        'middle': ['allegro_link_7_0_tip', 'allegro_link_7_0']},
        'hand_kp': 3., 'hand_kd': .1,
        'source_url': 'https://github.com/Wonikrobotics-git/allegro_hand_ros2_v5/tree/80bd4a88d2c59b8ad0242ec3730302bde61c84fb',
        'control_model': '16 independent position-driven hand joints; URDF limits preserved',
    },
    'leap_v1_right': {
        # v5: flat wrist face (was a 19.9 mm air gap, 26.2 mm off axis) seated on
        # the adapter top and centred on the flange axis; orientation already matched.
        # v6: shrink-wrapped, finer convex decomposition. v5's hulls bridged a palm
        # concavity and pressed the thumb PIP link into the palm at the home pose
        # (persistent self-contact; the source meshes are 1.9 mm apart there).
        # v6 still bridges the 1.2-1.5 mm gap at opposition postures, so releases
        # from v9_embodiments_r4 opt into the audited thumb_pip/palm exclusion in
        # their physics profile (robot_asset.LEAP_THUMB_PIP_PALM_AUDIT); runtime
        # scene_with_hand never adds it.
        'release_collision_filters': {'filter_leap_thumb_pip_palm': True},
        'assembly_version': 'isaacsim_6_1_v6', 'adapter': ADAPTER_DISC_10MM, 'collision_type': 'Convex Decomposition',
        'native_mesh_approximation': 'convexDecomposition',
        'convex_decomposition': {'max_convex_hulls': 128, 'hull_vertex_limit': 64, 'voxel_resolution': 1000000,
                                 'shrink_wrap': True, 'error_percentage': 2.},
        'label': 'LEAP Hand V1 right', 'hand_dof': 16, 'prefix': 'leap_',
        'urdf': 'robots/leap/v1_150bc3d/upstream/assets/leap_hand/robot.urdf',
        'license': 'robots/leap/v1_150bc3d/upstream/LICENSE.txt', 'license_id': 'MIT (simulation repository); CAD terms separately retained',
        'package_roots': {}, 'root_link': 'palm_lower', 'palm_link': 'palm_lower',
        'fingertip_links': {'index': 'fingertip', 'middle': 'fingertip_2', 'ring': 'fingertip_3', 'thumb': 'thumb_fingertip'},
        'mount_translation_m': [-.011722, .023393, .110095], 'mount_rpy_rad': [0., -1.5707963267948966, 0.],
        # v7: flange_axis_centred_v2 (v6 rotated 180 deg about the flange axis).
        'release_assembly_version': 'isaacsim_6_1_v7',
        'assemblies': {'isaacsim_6_1_v7': {'mount_convention': 'flange_axis_centred_v2',
                                           'mount_translation_m': [.011722, -.023393, .110095],
                                           'mount_rpy_rad': [0., -1.5707963267948966, math.pi]}},
        'hand_kp': 3., 'hand_kd': .1,
        'source_url': 'https://github.com/leap-hand/LEAP_Hand_Sim/tree/150bc3d4b61fd6619193ba5a8ef209f3609ced89',
        'control_model': '16 independent position-driven hand joints; not LEAP V2',
        'home_joint_positions_rad': {'leap_12': 1.104, 'leap_13': 1.163, 'leap_14': .953, 'leap_15': -.138},
        # The LEAP 'fingertip' links start at the DIP joint (~50 mm from the tip).
        # Executor tip point: centroid of the distal collision vertices within
        # 10 mm of the distal extreme along the link's long axis (fingertip.stl /
        # thumb_fingertip.stl), in the link frame.
        'controller_fingertips': {'index': ['fingertip', [-.00375, -.04578, .0174]],
                                  'middle': ['fingertip_2', [-.00375, -.04578, .0174]],
                                  'ring': ['fingertip_3', [-.00375, -.04578, .0174]],
                                  'thumb': ['thumb_fingertip', [-.00371, -.05838, -.0164]]},
        'controller_fingertip_semantics': 'Distal tip-cap centroid derived from the collision mesh; not a measured contact point.',
        'pinch_links': {'thumb': ['leap_thumb_fingertip'], 'index': ['leap_fingertip'], 'middle': ['leap_fingertip_2']},
        'home_source': 'Thumb entries 4:8 from official leap_hand_grasp.py canonical pose, mapped by LeapHandRot.yaml sim_to_real_indices; three long fingers straight.',
    },
    # Two FR3 arms, each with the official SHARPA with_flange hand at the identity
    # mount (left/right). Two articulations: PhysX caps one at 64 links.
    'dual_fr3_sharpa_w01': {
        'label': 'Dual FR3 + SHARPA Wave/W01 (left and right)', 'bimanual': True, 'hand_dof': 44, 'prefix': '',
        'assembly_version': 'isaacsim_6_1_v1', 'release_assembly_version': 'isaacsim_6_1_v2',
        # v2: both hands flange_axis_centred_v2 (180 deg about each flange axis; mirror-symmetric).
        'assemblies': {'isaacsim_6_1_v2': {'mount_convention': 'flange_axis_centred_v2', 'mount_rpy_rad': [0., 0., math.pi]}},
        'arms': {'left': {'base_offset_m': [0., .30, 0.]}, 'right': {'base_offset_m': [0., -.30, 0.]}},
        # Default home (mirror images: joints 1/3/5/7 negated); scenes may override it.
        'arm_home_rad_by_side': {'right': [.4, -.5, 0., -2.3, 0., 1.8, .8], 'left': [-.4, -.5, 0., -2.3, 0., 1.8, -.8]},
        'urdf': 'robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf',
        'license': 'robots/sharpa/wave_01/upstream/LICENSE.txt', 'license_id': 'Upstream LICENSE.txt', 'package_roots': {},
        'root_link': None, 'palm_link': 'right_hand_C_MC',
        'fingertip_links': {f'{side}_{n}': f'{side}_{"pinky" if n == "little" else n}_fingertip'
                            for side in ('left', 'right') for n in ('thumb', 'index', 'middle', 'ring', 'little')},
        'mount_translation_m': [0., 0., 0.], 'mount_rpy_rad': [0., 0., 0.],
        'hand_kp': 3., 'hand_kd': .1,
        'source_url': 'https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e',
        'control_model': 'Two FR3 arms (7 joints each) and two SHARPA W01 hands (22 joints each); 58 independent position drives',
        'pinch_links': {side: {n: [f'{side}_{n}_elastomer'] for n in ('thumb', 'index', 'middle')} for side in ('left', 'right')},
        'finger_names': ['thumb', 'index', 'middle', 'ring', 'little'],
    },
    'shadow_hand_e_right': {
        # v4: yaw +90 deg so the palm faces +x like the reference (was -y); the
        # forearm mating face was already flush and centred.
        'assembly_version': 'isaacsim_6_1_v4', 'merge_fixed_joints': True,
        'label': 'Shadow Hand E3M5 right, PST, 1:1 distal mimic simulation', 'hand_dof': 24, 'prefix': '',
        'urdf': 'robots/shadow/sr_common_172c3d9/derived/E3M5_right_pst/shadow_right.urdf',
        'license': 'robots/shadow/sr_common_172c3d9/upstream/LICENSE', 'license_id': 'BSD-3-Clause',
        'package_roots': {'sr_description': 'robots/shadow/sr_common_172c3d9/upstream/sr_description'},
        'root_link': 'rh_forearm', 'remove_root': 'world', 'palm_link': 'rh_palm',
        'fingertip_links': {'index': 'rh_ffdistal', 'middle': 'rh_mfdistal', 'ring': 'rh_rfdistal', 'little': 'rh_lfdistal', 'thumb': 'rh_thdistal'},
        'mount_translation_m': [0., 0., 0.], 'mount_rpy_rad': [0., 0., 1.5707963267948966],
        # v5: flange_axis_centred_v2 (v4 rotated 180 deg about the flange axis).
        'release_assembly_version': 'isaacsim_6_1_v5',
        'assemblies': {'isaacsim_6_1_v5': {'mount_convention': 'flange_axis_centred_v2', 'mount_rpy_rad': [0., 0., -1.5707963267948966]}},
        'hand_kp': 3., 'hand_kd': .1,
        'source_url': 'https://github.com/shadow-robot/sr_common/tree/172c3d9c3715ee75eb10dfe4ee57e26a342b4529',
        'control_model': '24 physical joints; 20 independent targets; 4 passive 1:1 distal mimic constraints approximate coupling. Real tendon differential is NOT modeled',
        'mimic_joints': {f'rh_{f}J1': {'joint': f'rh_{f}J2', 'multiplier': 1., 'offset': 0.} for f in ('FF','MF','RF','LF')},
        # Official sr_description tip frames (fixed, 26 / 27.5 mm beyond the distal joint).
        'controller_fingertips': {'index': ['rh_fftip', [0., 0., 0.]], 'middle': ['rh_mftip', [0., 0., 0.]],
                                  'ring': ['rh_rftip', [0., 0., 0.]], 'little': ['rh_lftip', [0., 0., 0.]],
                                  'thumb': ['rh_thtip', [0., 0., 0.]]},
        'controller_fingertip_semantics': 'Official sr_description fingertip frames; not measured contact points.',
        # Distal bodies (the tip frames are merged into them natively).
        'pinch_links': {'thumb': ['rh_thdistal'], 'index': ['rh_ffdistal'], 'middle': ['rh_mfdistal']},
        'hand_armature_kg_m2': .0001,
        'armature_status': 'Explicit numerical conditioning assumption, not measured motor rotor inertia; required for stable native PhysX mimic plus self-collision with tiny source link inertias.',
        'hardware_actuator_equivalent': False,
    },
}


# Manufacturer PhysX position-mode actuator calibration from the pinned upstream
# USD (drive gains converted from per-degree to per-radian units). A release opts
# in with physics option 'hand_actuator_model'; absent keeps the uniform gains.
SHARPA_ACTUATOR_SOURCE = {
    'path': 'robots/sharpa/wave_01/upstream/wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.usda',
    'sha256': '93b536684aa4e140718e52363c4f5fada518211cd053fbf42b499aee57ad4ef7',
    'statement': 'Upstream README: Isaac Lab dynamics in the USD are tuned to match position mode.',
}
# Joint class -> (kp N*m/rad, kd N*m*s/rad, armature kg*m^2, PhysX joint friction coefficient).
SHARPA_ACTUATOR_CLASSES = {
    'MCP_FE': (4.760018770387854, .18300271976478494, .00265, .104),
    'MCP_AA': (6.621673238326925, .20798367963248884, .00265, .104),
    'PIP': (.9075651474872239, .03999245410013146, .00061, .02476),
    'IP': (.9075651474872239, .03999245410013146, .00061, .02476),
    'DIP': (.904127400716439, .03151267873219528, .00042, .000418),
    'CMC_FE': (6.954561717297932, .24098604863202422, .0032, .132),
    'CMC_AA': (13.200947599814167, .45163971158982275, .0032, .132),
    'CMC': (1.380255328470153, .039247608966461386, .00012, .013),
}
# The position-mode model adopts the calibrated gains with their reflected
# inertia (armature only with matching gains, otherwise underdamped). Joint
# friction is a separate probe: the vendor USD and MJCF friction data disagree.
HAND_ACTUATOR_MODELS = {
    'sharpa_official_position_mode_v1': ('kp', 'kd', 'armature'),
    'sharpa_official_position_mode_joint_friction_probe_v1': ('kp', 'kd', 'armature', 'joint_friction'),
}


BIMANUAL = 'dual_fr3_sharpa_w01'
SHARPA_ROBOTS = (DEFAULT_HAND, BIMANUAL, 'fr3_sharpa_instances')


def hand_actuator_values(hand_id, model, joint_names):
    """Per-joint calibrated values for an opt-in actuator model; never for the arm."""
    if model not in HAND_ACTUATOR_MODELS or hand_id not in SHARPA_ROBOTS:
        raise ValueError(f'Unknown hand actuator model {model!r} for {hand_id!r}')
    fields = HAND_ACTUATOR_MODELS[model]
    values = {}
    for name in joint_names:
        if name.startswith('fr3_'):
            continue
        kp, kd, armature, friction = SHARPA_ACTUATOR_CLASSES[name.split('_', 2)[2]]
        row = {'kp': kp, 'kd': kd, 'armature': armature, 'joint_friction': friction}
        values[name] = {key: row[key] for key in fields}
    return values


def list_hands():
    return [get_hand_spec(name) for name in HANDS]


def get_hand_spec(hand_id=DEFAULT_HAND, assembly_version=None, *, instances=None):
    """Registry spec of one hand at one assembly version (default: its v1-era assembly)."""
    from .robot_instances import ROBOT_ID, instance_spec
    if hand_id == ROBOT_ID:
        return instance_spec({'hand_id': hand_id, 'assembly_version': assembly_version,
                              'instances': instances})
    if instances is not None:
        raise ValueError('Instance declarations require the instance robot embodiment')
    if hand_id not in HANDS:
        raise ValueError(f'Unknown hand_id {hand_id!r}; choose one of {sorted(HANDS)}')
    spec = deepcopy(HANDS[hand_id])
    versions = spec.pop('assemblies', {})
    registered = [spec['assembly_version'], *versions] if spec.get('assembly_version') else []
    if assembly_version is not None and assembly_version != spec.get('assembly_version'):
        if assembly_version not in versions:
            raise ValueError(f'Unknown {hand_id} assembly {assembly_version!r}; registered: {registered}')
        spec.update(deepcopy(versions[assembly_version]), assembly_version=assembly_version)
    convention = spec.get('mount_convention', MOUNT_CONVENTION)
    r,p,y = [x/2 for x in spec['mount_rpy_rad']]
    cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    spec['mount_quaternion_wxyz'] = [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
    spec.setdefault('mimic_joints', {})
    spec.setdefault('adapter', None)
    arm_dof = 7 * len(spec.get('arms', {'right': None}))
    spec.update(integration_status='paused' if hand_id in PAUSED_HANDS else 'active')
    spec.update(hand_id=hand_id, arm_dof=arm_dof, joint_count=arm_dof + spec['hand_dof'],
                actuated_joint_count=arm_dof + spec['hand_dof'] - len(spec['mimic_joints']),
                mount_status='uncalibrated_simulation_assumption', mount_convention=convention,
                hand_frame_yaw_rad=MOUNT_CONVENTIONS[convention]['hand_frame_yaw_rad'],
                registered_assembly_versions=registered, hardware_connected=False)
    return spec


def get_robot_spec(robot):
    """Resolve the selected scene robot, including explicit instance inventory."""
    from .robot_instances import ROBOT_ID, instance_spec
    if robot.get('hand_id') == ROBOT_ID:
        return instance_spec(robot)
    if 'instances' in robot:
        raise ValueError('Instance declarations require the instance robot embodiment')
    return get_hand_spec(robot.get('hand_id', DEFAULT_HAND), robot.get('assembly_version'))


def scene_with_hand(scene, hand_id, assembly_version=None):
    """Make a new embodiment scene; never modify a frozen release in place.

    A scene that pins an assembly keeps it (the runtime worker re-applies this to
    frozen scenes). Otherwise ``assembly_version`` pins the requested one (release
    builders); without it a non-default hand pins its v1-era assembly and the
    default SHARPA stays unpinned, so frozen SHARPA scenes are byte-identical.
    """
    result = deepcopy(scene)
    robot = result['robot']
    from .robot_instances import ROBOT_ID, ASSEMBLY_VERSION, checked_instances
    if hand_id == ROBOT_ID or robot.get('hand_id') == ROBOT_ID:
        if hand_id != robot.get('hand_id'):
            raise ValueError('An instance robot cannot change embodiment implicitly')
        if assembly_version not in (None, ASSEMBLY_VERSION):
            raise ValueError('Instance composition version differs from the scene')
        robot['instances'] = checked_instances(robot)
        spec = get_robot_spec(robot)
        robot.update(hand_kp=spec['hand_kp'], hand_kd=spec['hand_kd'])
        robot.setdefault('arm_home_rad_by_side', deepcopy(spec['arm_home_rad_by_side']))
        result.setdefault('isaacsim', {}).update(robot_self_collisions=True, filter_thumb_cmc_structure=True)
        result['embodiment_change'] = {'hand_id': hand_id, 'source_task_unchanged': True,
                                      'comparison_requires_separate_embodiment_stratum': True}
        return result
    pinned = robot.get('assembly_version') if robot.get('hand_id', DEFAULT_HAND) == hand_id else None
    if pinned is not None and assembly_version is not None and pinned != assembly_version:
        raise ValueError(f'Scene pins {hand_id} assembly {pinned!r}; cannot re-pin it to {assembly_version!r}')
    try:
        spec = get_hand_spec(hand_id, pinned or assembly_version)
    except ValueError as exc:
        if pinned is None:
            raise
        # A registry that no longer provides a pinned assembly must not silently
        # change the robot under a frozen release.
        raise ValueError(f'Scene pins {hand_id} assembly {pinned!r}, but the registry provides '
                         f'{get_hand_spec(hand_id)["registered_assembly_versions"]}; use a release built for '
                         'a registered assembly') from exc
    robot.update(hand_id=hand_id, mount_position_m=spec['mount_translation_m'],
                 mount_quaternion_wxyz=spec['mount_quaternion_wxyz'], hand_kp=spec['hand_kp'], hand_kd=spec['hand_kd'])
    if pinned or assembly_version or (hand_id != DEFAULT_HAND and spec.get('assembly_version')):
        robot['assembly_version'] = spec['assembly_version']
    result.setdefault('isaacsim', {}).update(robot_self_collisions=True, filter_thumb_cmc_structure=hand_id in SHARPA_ROBOTS)
    if 'arms' in spec:
        result['robot'].setdefault('arm_home_rad_by_side', deepcopy(spec['arm_home_rad_by_side']))
    result['embodiment_change'] = {'hand_id': hand_id, 'source_task_unchanged': True,
                                  'comparison_requires_separate_embodiment_stratum': True}
    return result
