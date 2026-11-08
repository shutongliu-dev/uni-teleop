"""Leading-edge transport keeps both controls local to the connected fold."""
import numpy as np
import pytest

from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scripts.validation.laundry_engineering_motion import build_motion
from test_laundry_stack_grip_span import inputs


@pytest.mark.parametrize('cells', [(30, 20), (60, 40)])
def test_both_leading_crease_corners_leave_open_edges_and_base_uncontrolled(cells):
    value = inputs(cells)
    motion = build_motion(value['task'], value['scene'], stack_trajectory='low-clearance',
                          stack_grip_span_m=[.04, .04], stack_grip_layout='crease-edge')
    specs = value['task']['asset_metadata']['cloths']
    rest = np.asarray(specs[0]['rest_vertices_m'])
    initial = {row['id']: np.asarray(row['initial_vertices_m']) for row in specs}
    ids = sum(motion.patches, [])
    assert len(ids) == len(set(ids)) <= .1*len(rest)
    assert np.max(np.abs(rest[ids, 0])) <= .020000001
    assert not set(ids).intersection(np.flatnonzero(np.abs(rest[:, 0]) > .25))
    assert rest[motion.patches[0], 1].max() < 0
    assert rest[motion.patches[1], 1].min() > 0
    evaluator = SequenceEvaluator(value['task'])
    evaluator._check_local_mask('towel', ids)
    for t in np.linspace(0, motion.duration_s, 101):
        points = {key: q.copy() for key, q in initial.items()}
        shifts = []
        for i, patch in enumerate(motion.patches):
            center, rotation = motion.pose(i, float(t))
            assert rotation == [1., 0., 0., 0.]
            delta = np.asarray(center)-motion.origins[i]
            shifts.append(delta)
            points['towel'][patch] += delta
        np.testing.assert_allclose(shifts[0], shifts[1], atol=1.e-12)
        evaluator._check_world_grasp_regions(points, {'towel': ids, 'towel_base': []})
        np.testing.assert_array_equal(points['towel_base'], initial['towel_base'])
    np.testing.assert_allclose(np.asarray(motion.pose(0, 7.)[0])-motion.origins[0], [.65, 0, .03])
    np.testing.assert_allclose(np.asarray(motion.pose(0, 8.5)[0])-motion.origins[0], [.65, 0, .0105])
    assert motion.release_time_s == 8.5
    assert motion.metadata['target_shape_written'] is False


def test_crease_layout_requires_explicit_spans():
    value = inputs()
    with pytest.raises(ValueError, match='explicit material spans'):
        build_motion(value['task'], value['scene'], stack_grip_layout='crease-edge')


def test_asymmetric_settled_crease_keeps_common_translation():
    value = inputs((60, 40))
    spec = value['task']['asset_metadata']['cloths'][0]
    initial = np.asarray(spec['initial_vertices_m'])
    # A smooth 4 mm corner offset models the asymmetry of a settled native fold.
    rest = np.asarray(spec['rest_vertices_m'])
    initial[:, 0] += .01*rest[:, 1]
    spec['initial_vertices_m'] = initial.tolist()
    motion = build_motion(value['task'], value['scene'], stack_trajectory='low-clearance',
                          stack_grip_span_m=[.04, .04], stack_grip_layout='crease-edge')
    assert abs(motion.origins[0][0]-motion.origins[1][0]) > .003
    for t in np.linspace(0, motion.duration_s, 101):
        shifts = [np.asarray(motion.pose(i, float(t))[0])-motion.origins[i]
                  for i in range(2)]
        np.testing.assert_allclose(shifts[0], shifts[1], atol=1.e-12)
    assert motion.metadata['stack_trajectory']['crossing_lift_by_anchor_m'] == [.03, .03]


def test_legacy_layout_is_unchanged_when_named_explicitly():
    value = inputs()
    args = dict(stack_trajectory='low-clearance', stack_grip_span_m=[.04, .04])
    a = build_motion(value['task'], value['scene'], **args)
    b = build_motion(value['task'], value['scene'], stack_grip_layout='diagonal', **args)
    assert a.patches == b.patches and a.metadata == b.metadata
    for t in np.linspace(0, 14.5, 31):
        for i in range(2):
            assert a.pose(i, float(t)) == b.pose(i, float(t))
