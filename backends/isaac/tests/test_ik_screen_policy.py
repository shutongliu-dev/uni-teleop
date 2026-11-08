"""Pass 3 of the multistart8_refine_screen_v2 IK policy (parameterized CPU audits).

Passes 1-2 keep the v1 contract (test_ik_solver_policy.py). Pass 3 only runs when
they found no clear solution: the same palm-down family on a dense 7.5 deg grid,
wrist-carried shapes screened with the checker's own environment test, solved
nearest-ready-extension first within a fixed budget.
"""
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from bench.geometry import rpy_matrix
from scripts.diagnostics import check_bimanual_options as options
from scripts.diagnostics.check_bimanual_options import (
    IK_POLICY, IK_SOLVER_POLICY, IK_SOLVER_POLICY_V1, checked_policy, screen_grid, solve_target,
    wrist_rigid_shapes, wrist_shapes_clear)

POINT = np.array([.6, .2, .9])
TIP = np.array([.06, .01, .2])
SHOULDER = np.array([0., 0., .333])
INITIAL = {'a': .3, 'b': .5, 'finger': .25}


def translation(xyz):
    pose = np.eye(4); pose[:3, 3] = xyz
    return pose


class Arm:
    """Two-joint stand-in: converges only for the orientations it is given."""
    names = ['a', 'b']
    active = [SimpleNamespace(name='a', lower=-1., upper=1., parent='base', origin=np.eye(4)),
              SimpleNamespace(name='b', lower=0., upper=2., parent='base', origin=translation(SHOULDER))]

    def __init__(self, converges):
        self.calls, self.converges, self.last = [], converges, translation([.3, 0., .6])

    def inverse(self, target, tip, seed, **settings):
        self.calls.append((np.array(target), dict(seed), settings))
        ok = bool(self.converges(np.array(target), dict(seed)))
        if ok:
            self.last = np.array(target)
        return {'converged': ok, 'joint_positions_rad': dict(seed)}

    def forward(self, joints, base):
        return {'base': np.array(base), 'tip': np.array(base) @ self.last}


def model(converges):
    return SimpleNamespace(arm=Arm(converges), names=['a', 'b', 'finger'], arm_tip='tip',
        arm_base=np.eye(4), wrist_offset=np.eye(4),
        public_geometry=lambda *_: {'fingertip_positions_m': {'middle': TIP.tolist()}})


def ready_distance():
    return float(np.linalg.norm(np.array([.3, 0., .6]) - SHOULDER))


def expected_order():
    """Independent statement of the declared pass-3 grid and solve order."""
    legacy = [rpy_matrix([math.pi, 0., y]) for y in [.75*math.pi, *[k*math.pi/4 for k in range(8)]]]
    legacy += [rpy_matrix([math.pi, p, y]) for p in (-math.pi/4, math.pi/4, -math.pi/2, math.pi/2)
               for y in (0., math.pi/2, math.pi, 3*math.pi/2)]
    legacy += [rpy_matrix([math.pi, 0., math.radians(y)]) for y in range(0, 360, 15) if y % 45]
    legacy += [rpy_matrix([math.pi, math.radians(p), math.radians(y)]) for p in (15., -15., 30., -30.)
               for y in range(0, 360, 30)]
    rows = []
    for p in np.arange(-90., 90.01, 7.5):
        for y in np.arange(0., 360., 7.5):
            rot = rpy_matrix([math.pi, math.radians(p), math.radians(y)])
            if any(np.allclose(rot, r, atol=1e-9) for r in legacy):
                continue
            flange = POINT - rot @ TIP
            rows.append((abs(np.linalg.norm(flange - SHOULDER) - ready_distance()), len(rows), (p, y), rot))
    return [row[2:] for row in sorted(rows, key=lambda r: (r[0], r[1]))]


def test_policy_ids_and_detail():
    assert IK_SOLVER_POLICY == 'multistart8_refine_screen_v2' and IK_SOLVER_POLICY_V1 == 'multistart8_refine_v1'
    assert checked_policy(None) is None and checked_policy(IK_SOLVER_POLICY_V1) == IK_SOLVER_POLICY_V1
    assert checked_policy(IK_SOLVER_POLICY) == IK_SOLVER_POLICY
    with pytest.raises(ValueError, match='Unknown IK solver policy'):
        checked_policy('multistart8_refine_v3')
    assert IK_POLICY['id'] == IK_SOLVER_POLICY
    third = IK_POLICY['third_pass']
    assert third['budget_orientations'] == options.SCREEN_IK_BUDGET == 24
    assert third['grid']['step_deg'] == options.SCREEN_GRID_STEP_DEG == 7.5


def test_grid_is_the_declared_palm_down_family_in_fixed_order():
    grid = screen_grid('above')
    assert len(grid) == 25*48 and len(screen_grid('handoff')) == 1200
    assert screen_grid('shelf_front') == []
    assert grid[0][0] == (-90., 0.) and grid[1][0] == (-90., 7.5) and grid[48][0] == (-82.5, 0.)
    assert grid[-1][0] == (90., 352.5)
    for (pitch, yaw), rot in grid[::97]:
        assert np.allclose(rot, rpy_matrix([math.pi, math.radians(pitch), math.radians(yaw)]))


def test_v1_never_runs_the_third_pass():
    own = model(lambda *_: False)
    row = solve_target(own, INITIAL, POINT, 'right', 0., 'above', policy=IK_SOLVER_POLICY_V1)
    assert len(own.arm.calls) == 89*(1+options.IK_RESTARTS)
    assert 'screened_dense_pass' not in row['ik_policy'] and not row['converged']


def test_failed_third_pass_is_budgeted_ordered_and_recorded():
    own = model(lambda *_: False)
    row = solve_target(own, INITIAL, POINT, 'right', 0., 'above', policy=IK_SOLVER_POLICY)
    first = 89*(1+options.IK_RESTARTS)
    assert len(own.arm.calls) == first + options.SCREEN_IK_BUDGET*(1+options.IK_RESTARTS)
    ready = [c[0][:3, :3] for c in own.arm.calls[first:] if c[1] == {'a': .3, 'b': .5}]
    expected = expected_order()
    assert len(expected) == 1200-88 and len(ready) == options.SCREEN_IK_BUDGET
    assert all(np.allclose(a, b[1], atol=1e-12) for a, b in zip(ready, expected))
    assert row == {'converged': False, 'clearance_passed': False, 'ik_policy': {
        'policy': IK_SOLVER_POLICY, 'candidate_count': 89, 'legacy_candidate_count': 25,
        'candidates_evaluated': 89+24, 'starts_attempted_total': first + 24*9,
        'legacy_outcome': {'converged': False, 'clearance_passed': False},
        'screened_dense_pass': {'grid_orientations': 1200, 'previously_evaluated': 88, 'screened': 1112,
            'screen_clear': 1112, 'budget': 24, 'solved': 24, 'screen_applied': False,
            'wrist_carried_shapes': 0}}}


def test_third_pass_finds_a_dense_orientation_v1_misses():
    target = expected_order()[4]
    own = model(lambda pose, seed: np.allclose(pose[:3, :3], target[1], atol=1e-12))
    v1 = solve_target(model(lambda pose, seed: np.allclose(pose[:3, :3], target[1], atol=1e-12)),
                      INITIAL, POINT, 'right', 0., 'above', policy=IK_SOLVER_POLICY_V1)
    assert not v1['converged']
    row = solve_target(own, INITIAL, POINT, 'right', 0., 'above', policy=IK_SOLVER_POLICY)
    assert row['converged'] and row['clearance_passed']
    record = row['ik_policy']
    assert not record['legacy_candidate'] and record['candidate_index'] >= 89
    assert record['selected_start'] == 'ready' and record['starts_for_selected_candidate'] == 1
    dense = record['screened_dense_pass']
    assert (dense['selected_pitch_deg'], dense['selected_yaw_deg']) == target[0]
    assert dense['selected_solve_rank'] == dense['solved'] == 5
    assert np.allclose(np.asarray(row['wrist_target_matrix'])[:3, :3], target[1])
    assert np.allclose(np.asarray(row['wrist_target_matrix'])[:3, 3], POINT - target[1] @ TIP)


@pytest.mark.parametrize('converges', [
    lambda pose, seed: True,                                     # pass 1, ready start
    lambda pose, seed: seed != {'a': .3, 'b': .5},               # pass 2, deterministic restart
    lambda pose, seed: np.allclose(pose[:3, :3], rpy_matrix([math.pi, math.radians(30.), math.radians(300.)]))])
def test_pass_one_and_two_results_are_unchanged(converges):
    v1_model, v2_model = model(converges), model(converges)
    v1 = solve_target(v1_model, INITIAL, POINT, 'right', 0., 'above', policy=IK_SOLVER_POLICY_V1)
    v2 = solve_target(v2_model, INITIAL, POINT, 'right', 0., 'above', policy=IK_SOLVER_POLICY)
    assert v1['converged'] and len(v1_model.arm.calls) == len(v2_model.arm.calls)
    assert v2['ik_policy'].pop('policy') == IK_SOLVER_POLICY and v1['ik_policy'].pop('policy') == IK_SOLVER_POLICY_V1
    assert v1 == v2


class ShapeChecker:
    """Arm link moves with joint 'a'; the hand box is carried by the wrist (tip)."""
    def __init__(self, obstacle):
        corners = np.array([(*c, 1.) for c in np.array(np.meshgrid([-.02, .02], [-.02, .02], [-.02, .02],
                                                                    indexing='ij')).reshape(3, -1).T])
        self.robot_shapes = {'right': [('arm_link', corners), ('hand_link', corners)]}
        self.hand_offset = translation([0., 0., .15])
        lo, hi = np.asarray(obstacle[0]), np.asarray(obstacle[1])
        self.obstacles = np.array([[lo, hi]])
        self.obstacle_corners = np.array([[[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                           for z in (lo[2], hi[2])]])
        self.models = None

    def poses(self, side, joints):
        tip = translation([joints['a'], joints['b'], .5]) @ np.diag([1., -1., -1., 1.])
        arm = translation([.1*joints['a'], 0., .2]) @ rpy_matrix_4(joints['b'])
        return {'tip': tip, 'arm_link': arm, 'hand_link': tip @ self.hand_offset}

    def boxes(self, side, joints):
        return joints

    def check(self, boxes):
        return {'clear': True, 'checks': {}}


def rpy_matrix_4(angle):
    pose = np.eye(4); pose[:3, :3] = rpy_matrix([0., 0., angle]); return pose


def shape_model():
    return model(lambda *_: False)


def test_wrist_rigid_shapes_selects_only_wrist_carried_links():
    checker = ShapeChecker([[5., 5., 5.], [6., 6., 6.]])
    rows = wrist_rigid_shapes(checker, shape_model(), 'right', INITIAL)
    assert [index for index, _, _ in rows] == [1]
    assert np.allclose(rows[0][1], checker.hand_offset)


def test_wrist_shapes_clear_uses_the_checker_environment_test():
    hand = [(1, translation([0., 0., .15]), ShapeChecker([[0, 0, 0], [1, 1, 1]]).robot_shapes['right'][1][1])]
    wrist = translation([.5, .5, .5])                        # hand box centre (.5, .5, .65)
    assert not wrist_shapes_clear(ShapeChecker([[.5, .5, .6], [.7, .7, .7]]), hand, wrist)
    assert wrist_shapes_clear(ShapeChecker([[.55, .4, .6], [.7, .7, .7]]), hand, wrist)  # 30 mm away
    # Gap below the 1 mm threshold is blocked; above it is clear.
    assert not wrist_shapes_clear(ShapeChecker([[.5205, .4, .6], [.6, .6, .7]]), hand, wrist)
    assert wrist_shapes_clear(ShapeChecker([[.5215, .4, .6], [.6, .6, .7]]), hand, wrist)
    # An enclosing AABB overlap separated by a rotated hand box is resolved by the SAT test.
    turned = translation([.5, .5, .5]); turned[:3, :3] = rpy_matrix([0., 0., math.pi/4])
    assert wrist_shapes_clear(ShapeChecker([[.524, .524, .6], [.6, .6, .7]]), hand, turned)


def test_third_pass_skips_screened_out_orientations():
    # Obstacle below the point: orientations that put the wrist-carried box into it are skipped.
    checker = ShapeChecker([[.4, 0., .7], [.8, .4, .88]])
    own = shape_model()
    row = solve_target(own, INITIAL, POINT, 'right', 0., 'above', checker, {'right': None},
                       restart_seeds=lambda *_: [], policy=IK_SOLVER_POLICY)
    dense = row['ik_policy']['screened_dense_pass']
    assert dense['screen_applied'] and dense['wrist_carried_shapes'] == 1
    assert 0 < dense['screen_clear'] < dense['screened'] and dense['solved'] == 24
    rigid = wrist_rigid_shapes(checker, own, 'right', INITIAL)
    first = 89*(1+options.IK_RESTARTS)
    solved = [c[0] for c in own.arm.calls[first:] if c[1] == {'a': .3, 'b': .5}]
    assert len(solved) == 24 and all(wrist_shapes_clear(checker, rigid, pose) for pose in solved)
    skipped = [rot for _, rot in screen_grid('above')
               if not wrist_shapes_clear(checker, rigid, translation(POINT - rot @ TIP) @ np.block(
                   [[rot, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))]
    assert skipped and not any(np.allclose(pose[:3, :3], rot) for pose in solved for rot in skipped)


def test_open_query_posture_is_used_for_the_screen():
    """The screen uses the queried hand coordinates, not the checker ready posture."""
    checker = ShapeChecker([[5., 5., 5.], [6., 6., 6.]])
    seen = []
    original = checker.poses
    checker.poses = lambda side, joints: seen.append(deepcopy(joints)) or original(side, joints)
    query = {**INITIAL, 'finger': 0.}
    wrist_rigid_shapes(checker, shape_model(), 'right', query)
    assert seen and all(q['finger'] == 0. for q in seen)
