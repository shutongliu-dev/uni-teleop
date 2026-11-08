import numpy as np
import pytest

from scenes.workrooms.laundry_room.composition import build_task_inputs
from scripts.validation.laundry_engineering_motion import build_motion


@pytest.mark.parametrize('task_id', ['towel_flattening', 'towel_folding', 'towel_stacking'])
def test_local_anchors_start_without_teleport_and_release_is_unforced(task_id):
    value = build_task_inputs(task_id)
    motion = build_motion(value['task'], value['scene'])
    initial = np.asarray(value['task']['asset_metadata']['cloths'][0]['initial_vertices_m'])
    assert sum(map(len, motion.patches)) <= len(initial)*.1
    for index, ids in enumerate(motion.patches):
        center, rotation = motion.pose(index, 0.)
        np.testing.assert_allclose(center, initial[ids].mean(0))
        np.testing.assert_allclose(rotation, [1., 0., 0., 0.])
        assert np.linalg.norm(initial[ids, None]-initial[None, ids], axis=-1).max() < .075
        for t in np.linspace(0, motion.duration_s, 20):
            p, q = motion.pose(index, t)
            assert np.isfinite(p).all()
            assert np.linalg.norm(q) == pytest.approx(1.)
        np.testing.assert_allclose(np.asarray(motion.pose(index, motion.release_time_s)[0]) + [0,0,.12],
                                   motion.pose(index, motion.duration_s)[0])
    assert motion.duration_s-motion.release_time_s >= .5


def test_fold_ends_on_opposite_half_with_inverted_material_orientation():
    value = build_task_inputs('towel_folding')
    motion = build_motion(value['task'], value['scene'])
    p, q = motion.pose(0, motion.release_time_s)
    assert p[0] == pytest.approx(.51)
    assert p[2] == pytest.approx(.9075)
    assert abs(q[2]) == pytest.approx(1.)


def test_fold_alignment_reads_free_edge_without_state_writes_or_target_jump():
    value = build_task_inputs('towel_folding')
    motion = build_motion(value['task'], value['scene'], fold_align_to_observed_edge=True)
    original = build_motion(value['task'], value['scene'])
    q = np.asarray(value['task']['asset_metadata']['cloths'][0]['initial_vertices_m'])
    q += [.022, -.003, .0001]
    before = q.copy()
    motion.observe_placement_reference(5.9995, q)
    assert motion.placement_reference is None
    motion.observe_placement_reference(6., q)
    np.testing.assert_array_equal(q, before)
    for i, ids in enumerate(motion.metadata['fold_edge_alignment']['observed_vertex_ids']):
        np.testing.assert_allclose(motion.pose(i, 6.)[0], original.pose(i, 6.)[0])
        np.testing.assert_allclose(motion.pose(i, 7.2)[0], q[ids].mean(0)+[0, 0, .006])
    reference = motion.placement_reference.copy()
    motion.observe_placement_reference(8., q+[1., 0., 0.])
    np.testing.assert_array_equal(motion.placement_reference, reference)
    assert motion.patches == original.patches


def test_fold_alignment_rejects_a_missed_observation_and_wrong_task():
    value = build_task_inputs('towel_folding')
    motion = build_motion(value['task'], value['scene'], fold_align_to_observed_edge=True)
    with pytest.raises(ValueError, match='Missed'):
        motion.observe_placement_reference(7., np.zeros((651, 3)))
    value = build_task_inputs('towel_stacking')
    with pytest.raises(ValueError, match='only a folding'):
        build_motion(value['task'], value['scene'], fold_align_to_observed_edge=True)


def test_explicit_four_centimetre_grasp_preserves_locality_and_fold_correspondence():
    from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator

    value = build_task_inputs('towel_folding')
    motion = build_motion(value['task'], value['scene'], fold_align_to_observed_edge=True,
                          fold_grip_span_m=[.04, .04])
    q = np.asarray(value['task']['asset_metadata']['cloths'][0]['initial_vertices_m'])
    assert list(map(len, motion.patches)) == [9, 9]
    evaluator = SequenceEvaluator(value['task'])
    evaluator._check_local_mask('towel', sum(motion.patches, []))
    evaluator._check_world_grasp_regions({'towel': q}, {'towel': sum(motion.patches, [])})
    motion.observe_placement_reference(6., q)
    for i, ids in enumerate(motion.metadata['fold_edge_alignment']['observed_vertex_ids']):
        np.testing.assert_allclose(motion.pose(i, 7.2)[0], q[ids].mean(0)+[0, 0, .006])
    with pytest.raises(ValueError, match='75 mm'):
        build_motion(value['task'], value['scene'], fold_grip_span_m=[.06, .06])


def test_stack_diagonal_grasps_satisfy_locality_and_end_above_base():
    from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator

    value = build_task_inputs('towel_stacking')
    motion = build_motion(value['task'], value['scene'])
    evaluator = SequenceEvaluator(value['task'])
    initial = {row['id']: np.asarray(row['initial_vertices_m'])
               for row in value['task']['asset_metadata']['cloths']}
    ids = sum(motion.patches, [])
    assert list(map(len, motion.patches)) == [8, 6]
    evaluator._check_local_mask('towel', ids)
    # The selected grip centers span both X and Y in the folded footprint;
    # they are not two clamps on the same open edge.
    separation = np.abs(np.subtract(*motion.origins))
    assert separation[0] > .25 and separation[1] > .35
    for time in np.linspace(0., motion.release_time_s, 50):
        state = {name: points.copy() for name, points in initial.items()}
        for index, patch in enumerate(motion.patches):
            center, _ = motion.pose(index, time)
            state['towel'][patch] += np.asarray(center)-motion.origins[index]
        evaluator._check_world_grasp_regions(state, {'towel': ids, 'towel_base': []})
    for index in range(2):
        at_height, _ = motion.pose(index, 2.)
        at_release, _ = motion.pose(index, motion.release_time_s)
        assert at_height[2]-motion.origins[index][2] == pytest.approx(.16)
        np.testing.assert_allclose(np.asarray(at_release)-motion.origins[index], [.65, 0., .0105])


def test_explicit_arc_matches_historical_default_without_new_metadata():
    value = build_task_inputs('towel_folding')
    default = build_motion(value['task'], value['scene'])
    explicit = build_motion(value['task'], value['scene'], fold_trajectory='arc')
    assert explicit.metadata == default.metadata
    for time in np.linspace(0., default.duration_s, 101):
        for index in range(2):
            assert explicit.pose(index, time) == default.pose(index, time)
    assert 'fold_trajectory' not in default.metadata


def test_bezier_has_correct_endpoints_peak_and_complete_candidate_receipt():
    from scripts.validation.laundry_engineering_motion import BEZIER_PAPER_URL
    value = build_task_inputs('towel_folding')
    motion = build_motion(value['task'], value['scene'], fold_trajectory='bezier', fold_grip_span_m=[.04, .04])
    profile = motion.metadata['fold_trajectory']
    assert profile['peak_height_m'] == .16 and profile['paper_url'] == BEZIER_PAPER_URL
    assert profile['grip_patches_fixed'] and profile['extra_constraints'] == 0
    assert motion.metadata['target_shape_written'] is False and motion.metadata['success_assumed'] is False
    for index in range(2):
        start = np.asarray(motion.origins[index])
        end = [2*motion.fold_x-start[0], start[1], start[2]+.006]
        np.testing.assert_allclose(motion.pose(index, .8)[0], start, atol=1.e-14)
        np.testing.assert_allclose(motion.pose(index, 7.2)[0], end, atol=1.e-14)
        np.testing.assert_allclose(motion.pose(index, 4.)[0], [motion.fold_x, start[1], start[2]+.16+.003])
        np.testing.assert_allclose(motion.pose(index, .8)[1], [1., 0., 0., 0.])
        np.testing.assert_allclose(motion.pose(index, 7.2)[1], [0., 0., -1., 0.], atol=1.e-14)
        np.testing.assert_allclose(np.asarray(motion.pose(index, motion.release_time_s)[0])+[0, 0, .12],
                                   motion.pose(index, motion.duration_s)[0])


def test_bezier_starts_and_ends_with_zero_velocity_and_acceleration():
    value = build_task_inputs('towel_folding')
    motion = build_motion(value['task'], value['scene'], fold_trajectory='bezier')
    eps = .001
    for index in range(2):
        for time in (.8, 7.2):
            # Include quaternion components: orientation also uses the same
            # quintic parameter and reaches rest without a derivative jump.
            before = np.r_[*motion.pose(index, time-eps)]
            now = np.r_[*motion.pose(index, time)]
            after = np.r_[*motion.pose(index, time+eps)]
            assert np.max(np.abs((after-before)/(2*eps))) < 1.e-6
            assert np.max(np.abs((after-2*now+before)/(eps*eps))) < .001


def test_bezier_preserves_two_fixed_local_patches_and_read_only_alignment():
    from copy import deepcopy
    from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
    value = build_task_inputs('towel_folding')
    untouched = deepcopy(value)
    options = dict(fold_trajectory='bezier', fold_align_to_observed_edge=True, fold_grip_span_m=[.04, .04])
    motion = build_motion(value['task'], value['scene'], **options)
    original = build_motion(value['task'], value['scene'], fold_grip_span_m=[.04, .04])
    pure = build_motion(value['task'], value['scene'], fold_trajectory='bezier', fold_grip_span_m=[.04, .04])
    assert motion.patches == original.patches and len(motion.patches) == 2
    ids = sum(motion.patches, [])
    q = np.asarray(value['task']['asset_metadata']['cloths'][0]['initial_vertices_m'])
    evaluator = SequenceEvaluator(value['task'])
    evaluator._check_local_mask('towel', ids)
    evaluator._check_world_grasp_regions({'towel': q}, {'towel': ids})
    observed = q+[.022, -.003, .0001]
    copy = observed.copy()
    motion.observe_placement_reference(6., observed)
    np.testing.assert_array_equal(observed, copy)
    for index, opposite in enumerate(motion.metadata['fold_edge_alignment']['observed_vertex_ids']):
        np.testing.assert_allclose(motion.pose(index, 6.)[0], pure.pose(index, 6.)[0])
        np.testing.assert_allclose(motion.pose(index, 7.2)[0], observed[opposite].mean(0)+[0, 0, .006])
        for time in np.linspace(0., motion.duration_s, 41):
            center, rotation = motion.pose(index, time)
            assert np.isfinite(center).all() and np.linalg.norm(rotation) == pytest.approx(1.)
    assert motion.patches == original.patches and value == untouched


@pytest.mark.parametrize('options', [
    {'fold_trajectory': 'unknown'},
    {'fold_peak_height_m': .16},
    {'fold_trajectory': 'arc', 'fold_peak_height_m': .16},
    {'fold_trajectory': 'bezier', 'fold_peak_height_m': 0.},
    {'fold_trajectory': 'bezier', 'fold_peak_height_m': -.1},
    {'fold_trajectory': 'bezier', 'fold_peak_height_m': float('nan')},
    {'fold_trajectory': 'bezier', 'fold_peak_height_m': float('inf')},
    {'fold_trajectory': 'bezier', 'fold_peak_height_m': True},
])
def test_invalid_bezier_options_fail_before_any_trajectory(options):
    value = build_task_inputs('towel_folding')
    with pytest.raises(ValueError):
        build_motion(value['task'], value['scene'], **options)


@pytest.mark.parametrize('task_id', ['towel_flattening', 'towel_stacking'])
def test_bezier_is_only_a_folding_candidate(task_id):
    value = build_task_inputs(task_id)
    with pytest.raises(ValueError, match='only a folding'):
        build_motion(value['task'], value['scene'], fold_trajectory='bezier')


@pytest.mark.parametrize('task_id', ['towel_flattening', 'towel_folding', 'towel_stacking'])
def test_explicit_passive_duration_only_extends_recording_after_unchanged_release(task_id):
    from copy import deepcopy
    value = build_task_inputs(task_id)
    untouched = deepcopy(value)
    default = build_motion(value['task'], value['scene'])
    none = build_motion(value['task'], value['scene'], passive_after_release_s=None)
    extended = build_motion(value['task'], value['scene'], passive_after_release_s=12.)
    assert none.metadata == default.metadata and none.duration_s == default.duration_s
    assert default.metadata['passive_after_release_s'] == 6.
    assert 'passive_duration_override' not in default.metadata
    assert extended.release_time_s == default.release_time_s
    assert extended.patches == default.patches and extended.origins == default.origins
    assert extended.duration_s == extended.release_time_s+12.
    receipt = extended.metadata['passive_duration_override']
    assert receipt['requested_s'] == 12. and receipt['total_native_steps']*.0005 == extended.duration_s
    for time in np.linspace(0., default.duration_s, 101):
        for index in range(2):
            assert extended.pose(index, time) == default.pose(index, time)
    for index in range(2):
        assert extended.pose(index, extended.duration_s) == default.pose(index, default.duration_s)
    assert value == untouched


@pytest.mark.parametrize('duration', [0., -.1, .4995, float('nan'), float('inf'), True, 110.5, 111., 6.00025])
def test_passive_duration_rejects_short_nonfinite_overbudget_or_fractional_steps(duration):
    value = build_task_inputs('towel_folding')
    with pytest.raises(ValueError):
        build_motion(value['task'], value['scene'], passive_after_release_s=duration)


def test_passive_duration_uses_frozen_stability_budget_and_period():
    value = build_task_inputs('towel_folding')
    # Boundary equality is allowed at stable_s but forbidden at max_time_s.
    motion = build_motion(value['task'], value['scene'], passive_after_release_s=.5)
    assert motion.duration_s == 10.
    value['task']['evaluation']['thresholds']['stable_s'] = .6
    with pytest.raises(ValueError, match='frozen stability'):
        build_motion(value['task'], value['scene'], passive_after_release_s=.5)
    value['task']['evaluation']['thresholds']['max_time_s'] = 15.5
    with pytest.raises(ValueError, match='time limit'):
        build_motion(value['task'], value['scene'], passive_after_release_s=6.)
    value['task']['evaluation']['thresholds']['max_time_s'] = 120.
    value['task']['evaluation']['physics_dt_s'] = .002
    with pytest.raises(ValueError, match='integer'):
        build_motion(value['task'], value['scene'], passive_after_release_s=6.0005)
