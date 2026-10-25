"""CPU reachability of the privileged engineering baselines' plans (no physics claim).

Every planned wrist pose (and fingertip shape) must have a verified IK solution
when approached in the controller-sized steps the baselines emit.
"""
from pathlib import Path

import numpy as np
import pytest

from bench.geometry import matrix_quaternion, matrix_rotvec, quaternion_matrix, rotvec_matrix
from bench.isaacsim.release import task_inputs
from bench.paths import DATA_ROOT

EXP = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation')


def walk(solve, q, position, rotation, step):
    """Approach a wrist pose in <= step moves; return the final joint solution."""
    current = np.asarray(position, float)
    start = solve.__self__.public_geometry(q)['wrist']['position_m'] if hasattr(solve, '__self__') else None
    pos = np.asarray(start, float)
    for _ in range(80):
        delta = current - pos
        if np.linalg.norm(delta) < 1e-4:
            break
        pos = pos + delta * min(1., step / np.linalg.norm(delta))
        q, _ = solve(q, wrist={'frame_id': 'world', 'position_m': pos.tolist(), 'quaternion_wxyz': matrix_quaternion(rotation).tolist()},
                     position_tolerance_m=.0005, rotation_tolerance_rad=.005)
    q, _ = solve(q, wrist={'frame_id': 'world', 'position_m': current.tolist(), 'quaternion_wxyz': matrix_quaternion(rotation).tolist()},
                 position_tolerance_m=.0005, rotation_tolerance_rad=.005)
    return q


def home(geometry, scene):
    from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec
    q = dict(zip([f'fr3_joint{i}' for i in range(1, 8)], scene['robot']['arm_home_rad']))
    q.update(get_hand_spec(scene['robot'].get('hand_id', 'sharpa_w01_right'),
                           scene['robot'].get('assembly_version')).get('home_joint_positions_rad', {}))
    return {j.name: float(np.clip(q.get(j.name, 0.), j.lower, j.upper)) for j in geometry.joints}


@pytest.mark.parametrize('hand', ['allegro_v5_right', 'leap_v1_right', 'shadow_hand_e_right'])
@pytest.mark.parametrize('generation', ['v9_embodiments_r4', 'v12_mount_v2_r1'])
def test_embodiment_push_plan_is_reachable(hand, generation):
    release = EXP / generation / hand
    if not release.is_dir():
        pytest.skip('v9 release not present')
    from scenes.manipulation.embodiment_policy import PrivilegedEmbodimentPolicy
    _, scene, task, _ = task_inputs(release, '01_cube_push')
    policy = PrivilegedEmbodimentPolicy(scene, task)
    g = policy.geometry
    q = home(g, scene)
    cube = next(o for o in scene['objects'] if o['id'] == policy.object_id)
    state = {'sequence': 1, 'sim_time_s': 0., 'objects': {policy.object_id: {'position_m': cube['position_m'],
                                                                              'quaternion_wxyz': cube['quaternion_wxyz']}}}
    observation = {'sequence': 1, 'sim_time_s': 0., 'proprioception': {'geometry': g.public_geometry(q), 'joint_positions_rad': q},
                   'tactile': {}}
    assert policy.next_decision(observation, state)['action'] == 'move_geometry'
    assert 'thumb' not in policy.tips
    for phase in policy.phases:
        if 'hold' not in phase:
            q = walk(g.solve, q, phase['position'], phase['rotation'], phase.get('translation_step', .045))


@pytest.mark.parametrize('task_id', ['05_bimanual_box_lift', '06_bimanual_dual_lift'])
@pytest.mark.parametrize('release_path', ['v10_bimanual_r1', 'v12_mount_v2_r1/bimanual'])
def test_bimanual_plans_are_reachable_and_mirrored(task_id, release_path):
    release = EXP / release_path
    if not release.is_dir():
        pytest.skip('bimanual release not present')
    from scenes.manipulation.bimanual_policy import PrivilegedBimanualPolicy
    from bench.simulation.isaacsim_backend.robot_registry import scene_with_hand
    _, scene, task, _ = task_inputs(release, task_id)
    scene = scene_with_hand(scene, 'dual_fr3_sharpa_w01')
    policy = PrivilegedBimanualPolicy(scene, task)
    g = policy.geometry
    q = {j.name: 0. for j in g.joints}
    q.update({f'fr3_{side}_joint{i}': v for side in ('left', 'right')
              for i, v in enumerate(scene['robot']['arm_home_rad_by_side'][side], 1)})
    q = {n: float(np.clip(v, g.by_name[n].lower, g.by_name[n].upper)) for n, v in q.items()}
    state = {'sequence': 1, 'sim_time_s': 0., 'objects': {o['id']: {'position_m': o['position_m'], 'quaternion_wxyz': o['quaternion_wxyz']}
                                                          for o in scene['objects']}}
    observation = {'sequence': 1, 'sim_time_s': 0., 'proprioception': {'hand_geometries': g.public_hand_geometries(q), 'joint_positions_rad': q}}
    assert policy.next_decision(observation, state)['action'] == 'move_geometry'
    M = np.diag([1., -1., 1.])
    assert np.allclose(M @ policy.hands['right']['flange'], policy.hands['left']['flange'], atol=1e-9)
    sides = g.split(q)
    table_z = scene['table']['position_m'][2] + scene['table']['half_extents_m'][2]
    clouds = {side: arm_collision_points(g.sides[side]) for side in ('left', 'right')}
    from scipy.spatial import cKDTree

    def arm_points(side):
        poses = g.sides[side].arm.forward(sides[side], base=g.sides[side].arm_base)
        return np.vstack([points @ poses[link][:3, :3].T + poses[link][:3, 3] for link, points in clouds[side].items()])
    for name, options in policy.phases:
        if 'hold' in options:
            continue
        for side in ('left', 'right'):
            position, rotation = policy._target(side, name, options)
            if options.get('joint_path'):
                # The policy's own joint-path-guided wrist waypoints, one decision at a time.
                for _ in range(45):
                    wrist = g.sides[side].public_geometry(sides[side])['wrist']
                    if (np.linalg.norm(np.asarray(position) - wrist['position_m']) <= .0025 and
                            np.linalg.norm(matrix_rotvec(rotation @ quaternion_matrix(wrist['quaternion_wxyz']).T)) <= .035):
                        break
                    full = {**g.split(q)['left'], **g.split(q)['right'], **sides['left'], **sides['right']}
                    dp, dr = policy._arm_waypoint(side, {'proprioception': {'joint_positions_rad': full}}, options.get('step', .045), 1.)
                    goal = {'frame_id': 'world', 'position_m': (np.asarray(wrist['position_m']) + dp).tolist(),
                            'quaternion_wxyz': matrix_quaternion(rotvec_matrix(dr) @ quaternion_matrix(wrist['quaternion_wxyz'])).tolist()}
                    sides[side], _ = g.sides[side].solve(sides[side], wrist=goal, position_tolerance_m=.0005, rotation_tolerance_rad=.005)
                else:
                    raise AssertionError((task_id, name, side, 'joint path did not converge'))
            else:
                sides[side] = walk(g.sides[side].solve, sides[side], position, rotation, options.get('step', .045))
            # IK ignores collisions: the planned arm pose must also clear the table
            # (0.1.1 held the tool axis level and fr3_left_link7 hit the table).
            lowest = float(arm_points(side)[:, 2].min())
            assert lowest > table_z + .02, (task_id, name, side, lowest)
        # ... and the other arm (v12 box lift with 0.1.4: the elbows swung inward and met).
        gap = float(cKDTree(arm_points('left')).query(arm_points('right'))[0].min())
        assert gap > .05, (task_id, name, gap)


def arm_collision_points(geometry):
    import trimesh
    import xml.etree.ElementTree as ET
    root = ET.parse(geometry.arm_path).getroot()
    base = Path(geometry.arm_path).parent
    result = {}
    for link in root.findall('link'):
        name = link.get('name')
        if not any(name.endswith(f'link{k}') for k in range(2, 8)):
            continue
        meshes = [element.find('geometry/mesh') for element in link.findall('collision')]
        points = [np.asarray(trimesh.load(base / m.get('filename'), force='mesh', process=False).vertices)
                  for m in meshes if m is not None]
        if points:
            result[name] = np.vstack(points)
    return result


@pytest.mark.parametrize('release_path', ['v11_soft_r2', 'v12_mount_v2_r1/soft'])
def test_soft_press_plan_is_reachable(release_path):
    release = EXP / release_path
    if not release.is_dir():
        pytest.skip('soft release not present')
    from bench.control.fingertip import Fr3SharpaGeometry
    from scenes.manipulation.soft_policy import PrivilegedSoftPressPolicy
    _, scene, task, _ = task_inputs(release, '07_sponge_press')
    policy = PrivilegedSoftPressPolicy(scene, task)
    g = Fr3SharpaGeometry(scene['robot']['assets_root'], scene['robot']['base_position_m'],
                          assembly_version=scene['robot'].get('assembly_version'))
    q = home(g, scene)
    spec = scene['soft_bodies'][0]
    centre, half = np.asarray(spec['position_m']), np.asarray(spec['size_m']) / 2
    state = {'sequence': 1, 'sim_time_s': 0., 'objects': {}, 'soft_bodies': {'sponge': {
        'centroid_m': centre.tolist(), 'bounds_min_m': (centre - half).tolist(), 'bounds_max_m': (centre + half).tolist()}}}
    observation = {'sequence': 1, 'sim_time_s': 0., 'proprioception': {'geometry': g.public_geometry(q), 'joint_positions_rad': q}}
    assert policy.next_decision(observation, state)['action'] == 'move_geometry'
    pressed = dict(policy.phases)['press']['position']
    face_z = pressed[2] - .1446                                  # knuckle face is 144.6 mm below the flange
    assert abs(face_z - (.75 + .50 * .06)) < 1e-6
    table_z = scene['table']['position_m'][2] + scene['table']['half_extents_m'][2]
    clouds = arm_collision_points(g)
    for name, phase in policy.phases:
        if 'hold' not in phase and not phase.get('posture'):
            q = walk(g.solve, q, phase['position'], phase['rotation'], phase.get('step', .045))
            poses = g.arm.forward(q, base=g.arm_base)
            lowest = min(float((points @ poses[link][:3, :3].T + poses[link][:3, 3])[:, 2].min()) for link, points in clouds.items())
            # 0.1.0 pressed palm-down with a level tool axis and fr3_link7 hit the table.
            assert lowest > table_z + .05, (name, lowest)


def test_soft_press_recentres_on_creep_and_stops_at_measured_compression():
    release = EXP / 'v11_soft_r2'
    if not release.is_dir():
        pytest.skip('v11 r2 release not present')
    from bench.control.fingertip import Fr3SharpaGeometry
    from scenes.manipulation.soft_policy import PRESS_STOP_RATIO, PrivilegedSoftPressPolicy
    _, scene, task, _ = task_inputs(release, '07_sponge_press')
    policy = PrivilegedSoftPressPolicy(scene, task)
    g = Fr3SharpaGeometry(scene['robot']['assets_root'], scene['robot']['base_position_m'])
    q = home(g, scene)
    spec = scene['soft_bodies'][0]
    half = np.asarray(spec['size_m']) / 2

    def inputs(sequence, centre, height_ratio=1.):
        centre = np.asarray(centre, float)
        low, high = centre - half, centre + half
        high[2] = low[2] + height_ratio * 2 * half[2]
        state = {'sequence': sequence, 'sim_time_s': .02 * sequence, 'objects': {}, 'soft_bodies': {'sponge': {
            'centroid_m': centre.tolist(), 'bounds_min_m': low.tolist(), 'bounds_max_m': high.tolist()}}}
        observation = {'sequence': sequence, 'sim_time_s': .02 * sequence,
                       'proprioception': {'geometry': g.public_geometry(q), 'joint_positions_rad': q}}
        return observation, state
    start = np.asarray(spec['position_m'], float)
    assert policy.next_decision(*inputs(0, start))['action'] == 'move_geometry'
    names = [name for name, _ in policy.phases]
    planned = {name: phase['position'].copy() for name, phase in policy.phases if 'position' in phase}
    # The sponge creeps 3 mm before the hand arrives: the face targets follow it (0.2.2 did not).
    policy.phase_index, policy.phase_attempts = names.index('above'), 0
    crept = start + [.003, 0., 0.]
    assert policy.next_decision(*inputs(1, crept), previous_feedback={'status': 'completed'})['action'] == 'move_geometry'
    for name in ('above', 'approach', 'press', 'release'):
        moved = dict(policy.phases)[name]['position'] - planned[name]
        assert np.allclose(moved, [.003, 0., 0.], atol=1e-9), name
    # 0.4.0: coarse 6 mm press steps above a ratio of 0.80, 1 mm steps below it (v11 r2's 6 mm
    # step crossed 0.75 -> 0.70 inside one action), and a stop inside the 0.70-0.75 window.
    from scenes.manipulation.soft_policy import PRESS_FINE_FROM_RATIO, PRESS_FINE_STEP_M, PRESS_FINE_TOLERANCE_M
    assert .70 < PRESS_STOP_RATIO < .75 and PRESS_STOP_RATIO < PRESS_FINE_FROM_RATIO
    assert (.75 - PRESS_STOP_RATIO) * .06 > PRESS_FINE_STEP_M / 2 and PRESS_FINE_STEP_M / .06 < PRESS_STOP_RATIO - .70
    policy.phase_index, policy.phase_attempts = names.index('press'), 0
    coarse = policy.next_decision(*inputs(2, crept, height_ratio=.9), previous_feedback={'status': 'completed'})
    assert coarse['action'] == 'move_geometry' and abs(np.linalg.norm(coarse['wrist_delta_m']) - .006) < 1e-9
    fine = policy.next_decision(*inputs(3, crept, height_ratio=PRESS_FINE_FROM_RATIO - .01), previous_feedback={'status': 'completed'})
    assert fine['action'] == 'move_geometry' and abs(np.linalg.norm(fine['wrist_delta_m']) - PRESS_FINE_STEP_M) < 1e-9
    # 0.4.0 sent fine steps with the default 3 mm tolerance and they completed without moving (soft_press_04).
    assert fine['position_tolerance_m'] == PRESS_FINE_TOLERANCE_M < PRESS_FINE_STEP_M
    assert 'position_tolerance_m' not in coarse
    # Once the measured height ratio reaches PRESS_STOP_RATIO the press ends and the hold starts.
    pressed = policy.next_decision(*inputs(4, crept, height_ratio=PRESS_STOP_RATIO - .001), previous_feedback={'status': 'completed'})
    assert pressed['action'] == 'hold' and policy.trace[-1]['height_ratio_phase_end']


@pytest.mark.parametrize('task_id', ['01_cube_push', '02_cube_lift_hold', '03_cube_pick_place', '04_cube_grasp_hold'])
def test_sharpa_mount_v2_plans_are_reachable_and_natural(task_id):
    """v12 (flange_axis_centred_v2): every SHARPA scripted plan has IK solutions that
    clear the table, and top-down grasps use the natural posture (palm toward the
    robot base, thumb to its left) at a mid-range joint 7."""
    release = EXP / 'v12_mount_v2_r1/sharpa_w01_right'
    if not release.is_dir():
        pytest.skip('v12 release not present')
    from bench.control.fingertip import Fr3SharpaGeometry
    from scenes.manipulation.privileged_policy import PrivilegedManipulationPolicy
    _, scene, task, _ = task_inputs(release, task_id)
    policy = PrivilegedManipulationPolicy(scene, task)
    g = Fr3SharpaGeometry(scene['robot']['assets_root'], scene['robot']['base_position_m'],
                          assembly_version=scene['robot']['assembly_version'])
    q = home(g, scene)
    cube = next(o for o in scene['objects'] if o['id'] == policy.object_id)
    state = {'sequence': 1, 'sim_time_s': 0., 'objects': {policy.object_id: {'position_m': cube['position_m'],
                                                                              'quaternion_wxyz': cube['quaternion_wxyz']}}}
    observation = {'sequence': 1, 'sim_time_s': 0., 'proprioception': {'geometry': g.public_geometry(q), 'joint_positions_rad': q},
                   'tactile': {}}
    assert policy.next_decision(observation, state)['action'] == 'move_geometry'
    table_z = scene['table']['position_m'][2] + scene['table']['half_extents_m'][2]
    clouds = arm_collision_points(g)
    for phase in policy.phases:
        if 'hold' in phase:
            continue
        q = walk(g.solve, q, phase['position'], phase['rotation'], phase.get('translation_step', .045))
        poses = g.arm.forward(q, base=g.arm_base)
        lowest = min(float((points @ poses[link][:3, :3].T + poses[link][:3, 3])[:, 2].min()) for link, points in clouds.items())
        assert lowest > table_z + .1, (phase['name'], lowest)
        if task_id != '01_cube_push' and phase['name'] in ('descend', 'close'):
            palm, thumb = phase['rotation'][:, 0], phase['rotation'][:, 1]
            assert palm[0] < -.9 and thumb[1] > .9 and abs(q['fr3_joint7']) < 1.


def test_leap_pinch_prior_keeps_unused_fingers_off_the_palm():
    """0.1.2's middle/ring fold (1.6/1.6/1.6) rested the fingertips on the palm (0.07 mm)."""
    import trimesh
    import xml.etree.ElementTree as ET
    from scipy.spatial import cKDTree
    from bench.control.fingertip import Fr3SharpaGeometry
    from bench.geometry import rpy_matrix
    from scenes.manipulation.embodiment_policy import PINCH_PRIORS
    g = Fr3SharpaGeometry(f'{DATA_ROOT}/assets', [0., 0., .77], hand_id='leap_v1_right')
    links = {l.get('name'): l for l in ET.parse(g.hand_path).getroot().findall('link')}

    def points(name):
        rows = []
        for element in links[name].findall('collision'):
            T = np.eye(4)
            origin = element.find('origin')
            if origin is not None:
                T[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                T[:3, :3] = rpy_matrix(np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
            shape = element.find('geometry')[0]
            P = np.asarray(trimesh.load(Path(g.hand_path).parent / shape.get('filename'), force='mesh',
                                        process=False).vertices)[::2]
            rows.append(P * np.fromstring(shape.get('scale', '1 1 1'), sep=' ') @ T[:3, :3].T + T[:3, 3])
        return np.vstack(rows)
    clouds = {n: points(n) for n in ('leap_palm_lower', 'leap_fingertip_2', 'leap_fingertip_3', 'leap_dip_2', 'leap_dip_3')}
    for shape in ('open', 'closed'):
        poses = g.hand.forward(PINCH_PRIORS['leap_v1_right'][shape])
        world = {n: P @ poses[n][:3, :3].T + poses[n][:3, 3] for n, P in clouds.items()}
        palm = cKDTree(world.pop('leap_palm_lower'))
        assert min(float(palm.query(P)[0].min()) for P in world.values()) > .01, shape


def test_leap_grasp_pre_shape_is_staged_thumb_fold_then_index():
    """0.1.4: moving all LEAP fingers at once jammed the index into the folding middle finger."""
    release = EXP / 'v12_mount_v2_r1/leap_v1_right'
    if not release.is_dir():
        pytest.skip('v12 release not present')
    from scenes.manipulation.embodiment_policy import PINCH_PRIORS, PrivilegedEmbodimentPolicy
    _, scene, task, _ = task_inputs(release, '04_cube_grasp_hold')
    policy = PrivilegedEmbodimentPolicy(scene, task)
    g = policy.geometry
    q = home(g, scene)
    cube = next(o for o in scene['objects'] if o['id'] == policy.object_id)
    state = {'sequence': 1, 'sim_time_s': 0., 'objects': {policy.object_id: {'position_m': cube['position_m'],
                                                                              'quaternion_wxyz': cube['quaternion_wxyz']}}}
    observation = {'sequence': 1, 'sim_time_s': 0., 'proprioception': {'geometry': g.public_geometry(q), 'joint_positions_rad': q},
                   'tactile': {}}
    assert policy.next_decision(observation, state)['action'] == 'move_geometry'
    names = [phase['name'] for phase in policy.phases]
    assert names[:4] == ['raise', 'pre_shape_stage_1', 'pre_shape_stage_2', 'pre_shape']
    prior = PINCH_PRIORS['leap_v1_right']
    first, second = policy.phases[1]['hand_goal'], policy.phases[2]['hand_goal']
    thumb, fold = prior['preshape_stages']
    index = ['leap_0', 'leap_1', 'leap_2', 'leap_3']
    assert all(first[n] == pytest.approx(prior['open'][n]) for n in thumb)
    assert all(first[n] == pytest.approx(q[n]) for n in fold + index)              # others wait
    assert all(second[n] == pytest.approx(prior['open'][n]) for n in thumb + fold)
    assert all(second[n] == pytest.approx(q[n]) for n in index)


def test_compliant_squeeze_converges_on_pad_depth():
    """0.1.6 on a compliant-pad release: ideal-plant walk of the squeeze ends with every
    long-finger pad 0-1.4 mm past the carton face (0.1.5's fixed posture sank them 4-6 mm)."""
    release = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/v13_pad_compliance_r1/bimanual')
    if not release.is_dir():
        pytest.skip('v13 bimanual release not present')
    from scenes.manipulation.bimanual_policy import SIDES, PrivilegedBimanualPolicy
    from bench.simulation.isaacsim_backend.robot_registry import scene_with_hand
    _, scene, task, _ = task_inputs(release, '05_bimanual_box_lift')
    scene = scene_with_hand(scene, 'dual_fr3_sharpa_w01')
    policy = PrivilegedBimanualPolicy(scene, task)
    assert policy.compliant_pads
    g = policy.geometry
    q = {j.name: 0. for j in g.joints}
    q.update({f'fr3_{s}_joint{i}': v for s in SIDES for i, v in enumerate(scene['robot']['arm_home_rad_by_side'][s], 1)})
    q = {n: float(np.clip(v, g.by_name[n].lower, g.by_name[n].upper)) for n, v in q.items()}
    state = {'sequence': 1, 'sim_time_s': 0., 'objects': {o['id']: {'position_m': o['position_m'], 'quaternion_wxyz': o['quaternion_wxyz']}
                                                          for o in scene['objects']}}
    observe = lambda seq: {'sequence': seq, 'sim_time_s': .02 * seq,
                           'proprioception': {'hand_geometries': g.public_hand_geometries(q), 'joint_positions_rad': q}}
    policy.next_decision(observe(1), state)
    for side in SIDES:
        h = policy.hands[side]
        wrist = {'frame_id': 'world', 'position_m': h['flange'].tolist(), 'quaternion_wxyz': matrix_quaternion(h['R']).tolist()}
        q.update(g.sides[side].solve({n: q[n] for n in g.sides[side].names}, wrist=wrist,
                                     position_tolerance_m=.0002, rotation_tolerance_rad=.002)[0])
    names = [name for name, _ in policy.phases]
    policy.phase_index, policy.phase_attempts, policy.last_sequence = names.index('squeeze'), 0, None
    for seq in range(10, 30):
        state.update(sequence=seq, sim_time_s=.02 * seq)
        decision = policy.next_decision(observe(seq), state,
                                        previous_feedback=None if policy.last_sequence is None else {'status': 'completed'})
        if policy.phases[policy.phase_index][0] != 'squeeze':
            break
        for side, target in decision['hands'].items():
            geo = g.sides[side]
            tips = geo.public_geometry({n: q[n] for n in geo.names})['fingertip_positions_m']
            goal = {f: (np.asarray(tips[f]) + np.asarray(d)).tolist() for f, d in target.get('fingertip_delta_m', {}).items()}
            q.update(geo.solve({n: q[n] for n in geo.names}, fingertip_positions_m=goal, position_tolerance_m=.002,
                               rotation_tolerance_rad=.03, fingertip_tolerance_m=.00075, fingertip_acceptance_m=.001)[0])
    else:
        raise AssertionError('squeeze did not finish')
    for side in SIDES:
        depths = policy._pad_depths(side, observe(seq), state)
        assert all(d is not None and 0. <= d <= .0014 for d in depths.values()), (side, depths)
