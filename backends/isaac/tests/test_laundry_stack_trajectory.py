"""Local low-clearance anchors are continuous, bounded, and explicitly selected."""
from copy import deepcopy

import numpy as np
import pytest

from scenes.workrooms.laundry_room.assets import candidate_metadata
from scenes.workrooms.laundry_room.composition import build_task_inputs
from scripts.validation.laundry_engineering_motion import build_motion


def stack_inputs(cells=(30, 20)):
    value = build_task_inputs('towel_stacking')
    # Pure motion geometry fixture, not a new frozen task/release.
    value['task']['asset_metadata'] = candidate_metadata('towel_stacking', cells_xy=cells)
    value['task']['evaluation']['metadata'] = deepcopy(value['task']['asset_metadata'])
    return value


def test_default_stack_lift_preserves_legacy_trajectory_and_metadata():
    value = stack_inputs()
    default = build_motion(value['task'], value['scene'])
    explicit = build_motion(value['task'], value['scene'], stack_trajectory='lift')
    assert default.metadata == explicit.metadata
    assert 'stack_trajectory' not in default.metadata
    assert default.metadata['stack_lift_m'] == .16
    for time in np.linspace(0., default.duration_s, 151):
        for index in range(2):
            assert default.pose(index, time) == explicit.pose(index, time)


@pytest.mark.parametrize('cells', [(30, 20), (60, 40)])
def test_low_clearance_uses_initial_edges_and_keeps_grasps_release_and_inputs(cells):
    from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator

    value = stack_inputs(cells)
    original = deepcopy(value)
    old = build_motion(value['task'], value['scene'])
    motion = build_motion(value['task'], value['scene'], stack_trajectory='low-clearance')
    profile = motion.metadata['stack_trajectory']
    initial = {row['id']: np.asarray(row['initial_vertices_m'])
               for row in value['task']['asset_metadata']['cloths']}
    assert motion.patches == old.patches and list(map(len, motion.patches)) == [8, 6]
    assert motion.release_time_s == old.release_time_s == 8.5
    assert motion.duration_s == old.duration_s == 14.5
    assert motion.metadata['target_shape_written'] is False
    assert profile['grip_patches_fixed'] and profile['extra_constraints'] == 0
    assert profile['moving_initial_front_x_m'] == initial['towel'][:, 0].max()
    assert profile['base_initial_near_edge_x_m'] == initial['towel_base'][:, 0].min()
    assert (profile['moving_initial_front_x_m']+profile['approach_dx_m']+.030
            == pytest.approx(profile['base_initial_near_edge_x_m']))
    assert profile['total_dx_m'] == pytest.approx(.65)
    evaluator = SequenceEvaluator(value['task'])
    ids = sum(motion.patches, [])
    evaluator._check_local_mask('towel', ids)
    evaluator._check_world_grasp_regions(initial, {'towel': ids, 'towel_base': []})
    for index in range(2):
        start = np.asarray(motion.origins[index])
        for time in [0., .6]:
            np.testing.assert_allclose(motion.pose(index, time)[0], start, atol=1.e-14)
        np.testing.assert_allclose(motion.pose(index, 1.6)[0], start+[0, 0, .025])
        np.testing.assert_allclose(motion.pose(index, 4.)[0], start+[profile['approach_dx_m'], 0, .025])
        crossing = .030 if index == profile['front_crease_anchor_index'] else .020
        np.testing.assert_allclose(motion.pose(index, 4.6)[0], start+[profile['approach_dx_m'], 0, crossing])
        np.testing.assert_allclose(motion.pose(index, 7.)[0], start+[.65, 0, crossing])
        for time in [7.8, 8.5, 8.75]:
            np.testing.assert_allclose(motion.pose(index, time)[0], start+[.65, 0, .0105])
        np.testing.assert_allclose(motion.pose(index, 14.5)[0], start+[.65, 0, .1305])
    assert value == original


def test_low_clearance_phase_boundaries_have_no_position_velocity_or_acceleration_jump():
    value = stack_inputs()
    motion = build_motion(value['task'], value['scene'], stack_trajectory='low-clearance')
    eps = 1.e-4
    for index in range(2):
        for time in [.6, 1.6, 4., 4.6, 7., 7.8, 8.5]:
            before = np.r_[*motion.pose(index, time-eps)]
            now = np.r_[*motion.pose(index, time)]
            after = np.r_[*motion.pose(index, time+eps)]
            assert np.max(np.abs(after-before)) < 1.e-9
            assert np.max(np.abs((after-before)/(2*eps))) < 1.e-6
            assert np.max(np.abs((after-2*now+before)/eps**2)) < .001
        samples = [motion.pose(index, t) for t in np.linspace(0., 8.5, 2001)]
        centers = np.asarray([p for p, _ in samples])
        assert np.diff(centers[:, 0]).min() >= -1.e-14
        assert centers[:, 2].min() >= motion.origins[index][2]-1.e-14
        assert centers[:, 2].max() <= motion.origins[index][2]+.030+1.e-14
        np.testing.assert_allclose(centers[:, 1], motion.origins[index][1])
        np.testing.assert_allclose([q for _, q in samples], np.tile([1., 0., 0., 0.], (len(samples), 1)))


def test_low_clearance_recomputes_approach_for_initial_base_geometry():
    value = stack_inputs()
    old = build_motion(value['task'], value['scene'], stack_trajectory='low-clearance')
    base = value['task']['asset_metadata']['cloths'][1]
    base['initial_vertices_m'] = (np.asarray(base['initial_vertices_m'])+[.04, 0, 0]).tolist()
    shifted = build_motion(value['task'], value['scene'], stack_trajectory='low-clearance')
    assert shifted.metadata['stack_trajectory']['approach_dx_m'] == pytest.approx(
        old.metadata['stack_trajectory']['approach_dx_m']+.04)
    for index in range(2):
        np.testing.assert_allclose(shifted.pose(index, 8.5)[0], old.pose(index, 8.5)[0])


@pytest.mark.parametrize('dx', [-.4, .4, float('nan')])
def test_low_clearance_rejects_missing_approach_or_nonfinite_base(dx):
    value = stack_inputs()
    base = value['task']['asset_metadata']['cloths'][1]
    base['initial_vertices_m'] = (np.asarray(base['initial_vertices_m'])+[dx, 0, 0]).tolist()
    with pytest.raises(ValueError, match='initial base geometry|approach point'):
        build_motion(value['task'], value['scene'], stack_trajectory='low-clearance')


@pytest.mark.parametrize('task_id', ['towel_flattening', 'towel_folding'])
def test_low_clearance_rejects_nonstack_tasks(task_id):
    value = build_task_inputs(task_id)
    with pytest.raises(ValueError, match='only a stacking'):
        build_motion(value['task'], value['scene'], stack_trajectory='low-clearance')


def test_stack_trajectory_rejects_unknown_profile():
    value = stack_inputs()
    with pytest.raises(ValueError, match='Unknown stacking'):
        build_motion(value['task'], value['scene'], stack_trajectory='unsupported')


@pytest.mark.parametrize('case', ['lift_release', 'two_layer', 'towel_flattening', 'towel_folding'])
def test_cli_rejects_low_clearance_on_other_cases_before_execution(monkeypatch, tmp_path, case):
    from scripts.validation import laundry_cloth_interactions as harness
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid trajectory must not start simulation')
    monkeypatch.setattr(harness, 'run', forbidden)
    monkeypatch.setattr(harness, 'run_task', forbidden)
    monkeypatch.setattr('sys.argv', ['probe', '--case', case, '--output', str(tmp_path/'out'),
                                   '--stack-trajectory', 'low-clearance'])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 2
    assert not (tmp_path/'out').exists()


def test_cli_passes_explicit_stack_candidate_to_formal_harness(monkeypatch, tmp_path):
    from scripts.validation import laundry_cloth_interactions as harness
    received = []
    def capture(args):
        received.append(args)
        return 0
    monkeypatch.setattr(harness, 'run_task', capture)
    monkeypatch.setattr('sys.argv', ['probe', '--case', 'towel_stacking', '--output', str(tmp_path/'out'),
                                   '--stack-trajectory', 'low-clearance'])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 0
    assert received[0].stack_trajectory == 'low-clearance'
