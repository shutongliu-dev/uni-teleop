"""flange_axis_centred_v2: every hand turned 180 deg about the FR3 flange axis.

With the v1 identity yaw the natural top-down right-hand posture in front of the
robot (palm toward the robot, thumb to its left) needs fr3_joint7 near +-180 deg,
beyond its limit; v2 puts it at the middle of the range. The public hand frame
follows the hand, so hand-relative geometry is unchanged and only the arm's
joint-7 solution moves.
"""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from bench.control.fingertip import Fr3SharpaGeometry
from bench.geometry import matrix_quaternion, quaternion_matrix
from bench.control.kinematics import URDFKinematics
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.robot_registry import DEFAULT_HAND, HANDS, get_hand_spec, scene_with_hand

ASSETS = f'{DATA_ROOT}/assets'
SINGLE = ['sharpa_w01_right', 'allegro_v5_right', 'leap_v1_right', 'shadow_hand_e_right']
READY = [0., -math.pi / 4, 0., -3 * math.pi / 4, 0., math.pi / 2, 0.]
TURN = np.diag([-1., -1., 1.])


def v2(hand):
    return get_hand_spec(hand)['release_assembly_version']


@pytest.mark.parametrize('hand', list(HANDS))
def test_v2_mount_is_the_v1_mount_turned_about_the_flange_axis(hand):
    old, new = get_hand_spec(hand), get_hand_spec(hand, v2(hand))
    assert (old['mount_convention'], new['mount_convention']) == ('flange_axis_centred_v1', 'flange_axis_centred_v2')
    assert np.allclose(TURN @ quaternion_matrix(old['mount_quaternion_wxyz']), quaternion_matrix(new['mount_quaternion_wxyz']), atol=1e-12)
    assert np.allclose(TURN @ np.asarray(old['mount_translation_m']), new['mount_translation_m'], atol=1e-12)
    assert new['hand_frame_yaw_rad'] == pytest.approx(math.pi) and old['hand_frame_yaw_rad'] == 0.
    with pytest.raises(ValueError, match='Unknown'):
        get_hand_spec(hand, 'isaacsim_6_1_v99')


def joints(geometry, arm, rng):
    q = {j.name: float(rng.uniform(max(j.lower, -1.), min(j.upper, 1.))) for j in geometry.hand.active}
    q.update(dict(zip([f'fr3_joint{i}' for i in range(1, 8)], arm)))
    return q


@pytest.mark.parametrize('hand', SINGLE)
def test_v2_hand_frame_keeps_hand_relative_geometry(hand):
    old = Fr3SharpaGeometry(ASSETS, [0., 0., .77], hand_id=hand)
    new = Fr3SharpaGeometry(ASSETS, [0., 0., .77], hand_id=hand, assembly_version=v2(hand))
    assert old.names == new.names and old.frame_id == new.frame_id == 'right_hand_flange'
    rng = np.random.default_rng(3)
    for _ in range(4):
        q = joints(new, READY, rng)
        a, b = old.public_geometry(q), new.public_geometry(q)
        for finger in a['fingertip_positions_m']:
            assert np.allclose(a['fingertip_positions_m'][finger], b['fingertip_positions_m'][finger], atol=1e-12)
        assert np.allclose(a['wrist']['position_m'], b['wrist']['position_m'], atol=1e-12)
        assert np.allclose(quaternion_matrix(a['wrist']['quaternion_wxyz']) @ TURN,
                           quaternion_matrix(b['wrist']['quaternion_wxyz']), atol=1e-12)


@pytest.mark.parametrize('hand', SINGLE)
def test_v2_controller_matches_the_native_assembly_urdf(hand):
    """Controller FK (arm * hand-frame offset * hand tree) == the assembly Isaac imports."""
    spec = get_hand_spec(hand, v2(hand))
    if hand == DEFAULT_HAND:
        source = Path(ASSETS) / 'robots/assemblies/fr3_sharpa' / spec['assembly_version'] / 'SOURCE.json'
    else:
        source = Path(ASSETS) / 'robots/assemblies' / f'fr3_{hand}' / spec['assembly_version'] / 'SOURCE.json'
    if not source.is_file():
        pytest.skip('v2 assembly not generated')
    assembly = URDFKinematics(Path(load_json(source)['urdf_path']))
    g = Fr3SharpaGeometry(ASSETS, [0., 0., 0.], hand_id=hand, assembly_version=spec['assembly_version'])
    rng = np.random.default_rng(5)
    for _ in range(3):
        q = joints(g, [rng.uniform(-1., 1.) for _ in range(7)], rng)
        full = {**q, **{n: q[n] for n in q}}
        if hasattr(g.hand, 'expand'):
            full = g.hand.expand(full)
        poses = assembly.forward({j.name: full.get(j.name, 0.) for j in assembly.active})
        public = g.public_geometry(q)
        wrist = np.eye(4)
        wrist[:3, :3], wrist[:3, 3] = quaternion_matrix(public['wrist']['quaternion_wxyz']), public['wrist']['position_m']
        for finger, link in g.fingertip_links.items():
            tip_native = poses[link][:3, :3] @ g.tip_offsets.get(finger, np.zeros(3)) + poses[link][:3, 3]
            tip_public = wrist[:3, :3] @ np.asarray(public['fingertip_positions_m'][finger]) + wrist[:3, 3]
            assert np.allclose(tip_native, tip_public, atol=1e-9), (finger, tip_native, tip_public)


@pytest.mark.parametrize('hand', SINGLE)
def test_natural_top_down_posture_needs_joint7_mid_range_only_with_v2(hand):
    natural = np.column_stack([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])   # palm toward base, thumb left, fingers down
    for version, expected in ((None, math.pi), (v2(hand), 0.)):
        g = Fr3SharpaGeometry(ASSETS, [0., 0., .77], hand_id=hand, assembly_version=version)
        best = None
        for q7 in np.linspace(-math.pi, math.pi, 721):
            q = joints(g, READY[:6] + [q7], np.random.default_rng(0))
            R = quaternion_matrix(g.public_geometry(q)['wrist']['quaternion_wxyz'])
            error = np.linalg.norm(R - natural)
            best = min(best or (error, q7), (error, q7))
        assert best[0] < 1e-6
        assert abs(math.remainder(best[1] - expected, 2 * math.pi)) < 1e-2
        limit = g.by_name['fr3_joint7'].upper
        assert (abs(best[1]) > limit) == (version is None)          # v1: beyond the joint limit


def test_v2_ik_reaches_the_natural_grasp_in_front_of_the_robot():
    g = Fr3SharpaGeometry(ASSETS, [0., 0., .77], assembly_version=v2(DEFAULT_HAND))
    q = {j.name: 0. for j in g.joints}
    q.update(dict(zip([f'fr3_joint{i}' for i in range(1, 8)], [0., -.6, 0., -2.2, 0., 1.6, .8])))
    natural = np.column_stack([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    target = {'frame_id': 'world', 'position_m': [.5, .1, 1.], 'quaternion_wxyz': matrix_quaternion(natural).tolist()}
    solved, _ = g.solve(q, wrist=target, position_tolerance_m=.0005, rotation_tolerance_rad=.005)
    reached = g.public_geometry(solved)['wrist']
    assert np.allclose(reached['position_m'], target['position_m'], atol=.001)
    assert abs(solved['fr3_joint7']) < 1.6


def test_bimanual_v2_turns_both_hands_and_stays_mirrored():
    from bench.control.bimanual import BimanualGeometry
    version = v2('dual_fr3_sharpa_w01')
    source = Path(ASSETS) / 'robots/assemblies/dual_fr3_sharpa_w01' / version / 'SOURCE_left.json'
    if not source.is_file():
        pytest.skip('dual v2 assembly not generated')
    old = BimanualGeometry(ASSETS, [0., 0., .77])
    new = BimanualGeometry(ASSETS, [0., 0., .77], assembly_version=version)
    q = {j.name: 0. for j in new.joints}
    home = get_hand_spec('dual_fr3_sharpa_w01')['arm_home_rad_by_side']
    for side in ('left', 'right'):
        q.update({f'fr3_{side}_joint{i}': v for i, v in enumerate(home[side], 1)})
    a, b = old.public_hand_geometries(q), new.public_hand_geometries(q)
    for side in ('left', 'right'):
        assert np.allclose(quaternion_matrix(a[side]['wrist']['quaternion_wxyz']) @ TURN,
                           quaternion_matrix(b[side]['wrist']['quaternion_wxyz']), atol=1e-12)
    M = np.diag([1., -1., 1.])
    R_left, R_right = (quaternion_matrix(b[s]['wrist']['quaternion_wxyz']) for s in ('left', 'right'))
    assert np.allclose(M @ R_right @ M, R_left, atol=1e-9)


def test_scene_pins_and_runtime_reapplication():
    scene = load_json(Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/sequential_workcell/v1/tasks/01_ordered_kitting/scene.json'))
    # Default SHARPA without a pin stays unpinned at runtime (frozen v8 scenes keep their bytes).
    runtime = scene_with_hand(scene, DEFAULT_HAND)
    assert 'assembly_version' not in runtime['robot'] and runtime['robot']['mount_quaternion_wxyz'] == [1., 0., 0., 0.]
    pinned = scene_with_hand(scene, DEFAULT_HAND, assembly_version=v2(DEFAULT_HAND))
    assert pinned['robot']['assembly_version'] == 'isaacsim_6_1_v2'
    assert np.allclose(quaternion_matrix(pinned['robot']['mount_quaternion_wxyz']), TURN, atol=1e-12)
    assert scene_with_hand(pinned, DEFAULT_HAND) == pinned                    # the worker keeps the pin
    with pytest.raises(ValueError, match='cannot re-pin'):
        scene_with_hand(pinned, DEFAULT_HAND, assembly_version='isaacsim_6_1_v1')
    leap = scene_with_hand(scene, 'leap_v1_right', assembly_version=v2('leap_v1_right'))
    assert leap['robot']['assembly_version'] == 'isaacsim_6_1_v7'


def test_controller_configuration_names_only_non_default_assemblies():
    from bench.runtime.controllers import robot_configuration
    scene = load_json(Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/sequential_workcell/v1/tasks/01_ordered_kitting/scene.json'))
    assert 'assembly_version' not in robot_configuration(scene_with_hand(scene, DEFAULT_HAND))
    assert 'assembly_version' not in robot_configuration(scene_with_hand(scene, 'leap_v1_right'))
    configuration = robot_configuration(scene_with_hand(scene, DEFAULT_HAND, assembly_version=v2(DEFAULT_HAND)))
    assert configuration['assembly_version'] == 'isaacsim_6_1_v2'
    geometry = Fr3SharpaGeometry(**configuration)
    assert geometry.embodiment.expected_mount['quaternion_wxyz'] == get_hand_spec(DEFAULT_HAND, 'isaacsim_6_1_v2')['mount_quaternion_wxyz']


def test_prompt_states_the_published_mount_and_v1_prompts_are_unchanged():
    from agent.model_client.runner import model_system_prompt
    v1 = {'mount_convention': 'flange_axis_centred_v1'}
    v2_mount = {'mount_convention': 'flange_axis_centred_v2', 'hand_frame_yaw_about_flange_axis_rad': math.pi,
                'status': 'uncalibrated_simulation_assumption'}
    for control, interaction in (('fingertip', 'action'), ('fingertip', 'tools'), ('hybrid', 'free'), ('joints', 'action')):
        base = model_system_prompt(control, interaction)
        assert model_system_prompt(control, interaction, mount=None) == base == model_system_prompt(control, interaction, mount=v1)
        turned = model_system_prompt(control, interaction, mount=v2_mount)
        assert 'identically mounted to fr3_link8' not in turned and 'flange_axis_centred_v2' in turned
    leap = model_system_prompt('fingertip', 'action', robot_id='leap_v1_right', mount=v2_mount)
    assert 'turned 180 deg about its axis' in leap and 'names that FR3 tool-flange frame' not in leap
    with pytest.raises(ValueError, match='Unknown published hand mount'):
        model_system_prompt('fingertip', 'action', mount={'mount_convention': 'flange_axis_centred_v9'})


def test_tasks_route_publishes_the_mount_only_for_v2_releases(tmp_path):
    from bench.runtime.server import create_app
    from fastapi.testclient import TestClient
    v12 = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/v12_mount_v2_r1/sharpa_w01_right')
    v8 = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/v8_embodiment_fidelity_r2')
    if not v12.is_dir() or not v8.is_dir():
        pytest.skip('releases not present')
    from types import SimpleNamespace
    for release, expected in ((v8, None), (v12, 'flange_axis_centred_v2')):
        service = SimpleNamespace(release=release, shutdown=lambda: None)
        rows = TestClient(create_app(service=service)).get('/tasks').json()
        assert len(rows) == 4
        assert {row.get('robot_mount', {}).get('mount_convention') for row in rows} == {expected}


def test_only_sharpa_robots_are_active_other_hands_are_paused():
    """User, 2026-10-15: build the bench with SHARPA first; keep the other hands' assets, pause their integration."""
    from bench.simulation.isaacsim_backend.robot_registry import PAUSED_HANDS, list_hands
    status = {spec['hand_id']: spec['integration_status'] for spec in list_hands()}
    assert status == {'sharpa_w01_right': 'active', 'dual_fr3_sharpa_w01': 'active', 'allegro_v5_right': 'paused',
                      'leap_v1_right': 'paused', 'shadow_hand_e_right': 'paused'}
    assert set(PAUSED_HANDS) == {h for h, s in status.items() if s == 'paused'}
