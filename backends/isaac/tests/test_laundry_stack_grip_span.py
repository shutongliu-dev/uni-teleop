"""Explicit stack grips cover the same material area across native grids."""
from copy import deepcopy

import numpy as np
import pytest

from scenes.workrooms.laundry_room.assets import candidate_metadata
from scenes.workrooms.laundry_room.composition import build_task_inputs
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scripts.validation.laundry_engineering_motion import build_motion


def inputs(cells=(30, 20)):
    value = build_task_inputs('towel_stacking')
    # Pure CPU geometry fixture; no frozen release is rewritten.
    value['task']['asset_metadata'] = candidate_metadata('towel_stacking', cells_xy=cells)
    value['task']['evaluation']['metadata'] = deepcopy(value['task']['asset_metadata'])
    return value


@pytest.mark.parametrize('cells,counts', [((30, 20), [18, 9]), ((60, 40), [50, 25])])
@pytest.mark.parametrize('trajectory', ['lift', 'low-clearance'])
def test_four_cm_stack_grips_cover_exact_material_blocks_and_remain_local(cells, counts, trajectory):
    value = inputs(cells)
    unchanged = deepcopy(value)
    motion = build_motion(value['task'], value['scene'], stack_trajectory=trajectory,
                          stack_grip_span_m=[.04, .04])
    spec = value['task']['asset_metadata']['cloths'][0]
    rest = np.asarray(spec['rest_vertices_m'])
    initial = {row['id']: np.asarray(row['initial_vertices_m'])
               for row in value['task']['asset_metadata']['cloths']}
    # Select expected topology by physical material coordinates, independent
    # of implementation index arithmetic or x-fast vertex ordering.
    x, y = rest[:, 0], rest[:, 1]
    open_corner = ((x <= x.min()+.04+1.e-9) | (x >= x.max()-.04-1.e-9)) & (y <= y.min()+.04+1.e-9)
    crease_corner = (abs(x-(x.min()+x.max())/2) <= .02+1.e-9) & (y >= y.max()-.04-1.e-9)
    assert set(motion.patches[0]) == set(np.flatnonzero(open_corner))
    assert set(motion.patches[1]) == set(np.flatnonzero(crease_corner))
    assert list(map(len, motion.patches)) == counts
    assert len(set(sum(motion.patches, []))) == sum(counts) <= .1*len(rest)
    assert motion.metadata['patches'] == motion.patches
    assert motion.metadata['stack_grip_span_m'] == [.04, .04]
    assert motion.metadata['stack_grip_cells_xy'] == [cells[0]//15, cells[1]//10]
    assert motion.metadata['target_shape_written'] is False
    assert motion.release_time_s == 8.5 and motion.duration_s == 14.5
    evaluator = SequenceEvaluator(value['task'])
    ids = sum(motion.patches, [])
    evaluator._check_local_mask('towel', ids)
    for time in [0., 1.6, 4., 4.6, 7., 8.5]:
        state = {name: points.copy() for name, points in initial.items()}
        for index, patch in enumerate(motion.patches):
            center, rotation = motion.pose(index, time)
            np.testing.assert_array_equal(rotation, [1., 0., 0., 0.])
            state['towel'][patch] += np.asarray(center)-motion.origins[index]
        evaluator._check_world_grasp_regions(state, {'towel': ids, 'towel_base': []})
    for index, patch in enumerate(motion.patches):
        np.testing.assert_allclose(motion.pose(index, 0.)[0], initial['towel'][patch].mean(0))
        np.testing.assert_allclose(np.asarray(motion.pose(index, 8.5)[0])-motion.origins[index], [.65, 0., .0105])
        np.testing.assert_allclose(np.asarray(motion.pose(index, 14.5)[0])-motion.origins[index], [.65, 0., .1305])
    assert value == unchanged


@pytest.mark.parametrize('cells', [(30, 20), (60, 40)])
@pytest.mark.parametrize('trajectory', ['lift', 'low-clearance'])
def test_unspecified_stack_grips_preserve_legacy_indices_and_metadata(cells, trajectory):
    value = inputs(cells)
    default = build_motion(value['task'], value['scene'], stack_trajectory=trajectory)
    explicit = build_motion(value['task'], value['scene'], stack_trajectory=trajectory, stack_grip_span_m=None)
    assert list(map(len, default.patches)) == [8, 6]
    assert explicit.patches == default.patches and explicit.metadata == default.metadata
    assert 'stack_grip_span_m' not in default.metadata
    for time in np.linspace(0., default.duration_s, 31):
        for index in range(2):
            assert explicit.pose(index, time) == default.pose(index, time)


def test_fine_grid_accepts_two_cm_without_rounding_to_four_cm():
    value = inputs((60, 40))
    motion = build_motion(value['task'], value['scene'], stack_grip_span_m=[.02, .02])
    assert list(map(len, motion.patches)) == [18, 9]
    assert motion.metadata['stack_grip_cells_xy'] == [2, 2]
    assert motion.metadata['stack_grip_span_m'] == [.02, .02]


@pytest.mark.parametrize('span', [[], [.04], [.04, .04, .04], [0., .04], [-.02, .04],
    [float('nan'), .04], [.04, float('inf')], [.06, .06], [.03, .04], [.02, .04]])
def test_stack_grips_reject_invalid_or_asymmetric_coarse_spans(span):
    value = inputs()
    with pytest.raises(ValueError, match='75 mm|whole native mesh cells|even grid and span'):
        build_motion(value['task'], value['scene'], stack_grip_span_m=span)


def test_stack_grips_require_a_true_material_midline_node():
    value = inputs()
    # Asset preparation already rejects odd X counts; the motion entrypoint
    # must also reject malformed metadata before using its vertex indices.
    value['task']['asset_metadata']['cloths'][0]['cells_xy'] = [31, 20]
    with pytest.raises(ValueError, match='even grid and span'):
        build_motion(value['task'], value['scene'], stack_grip_span_m=[2*.6/31, .04])


def test_stack_grips_reject_an_initial_pose_with_nonlocal_world_corners():
    value = inputs()
    # The declared material span alone cannot make an unfolded or displaced
    # second open layer a valid single world-local clamp.
    value['task']['asset_metadata']['cloths'][0]['initial_vertices_m'][30][0] += .10
    with pytest.raises(ValueError, match='75 mm world region'):
        build_motion(value['task'], value['scene'], stack_grip_span_m=[.04, .04])


@pytest.mark.parametrize('task_id', ['towel_flattening', 'towel_folding'])
def test_stack_grip_span_is_not_accepted_for_other_tasks(task_id):
    value = build_task_inputs(task_id)
    with pytest.raises(ValueError, match='only a stacking'):
        build_motion(value['task'], value['scene'], stack_grip_span_m=[.04, .04])


@pytest.mark.parametrize('case', ['lift_release', 'two_layer', 'towel_flattening', 'towel_folding'])
def test_cli_rejects_stack_grips_for_nonstack_case_before_execution(monkeypatch, tmp_path, case):
    from scripts.validation import laundry_cloth_interactions as harness
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid grasp option must not start simulation')
    monkeypatch.setattr(harness, 'run', forbidden)
    monkeypatch.setattr(harness, 'run_task', forbidden)
    monkeypatch.setattr('sys.argv', ['probe', '--case', case, '--output', str(tmp_path/'out'),
                                   '--stack-grip-span-m', '.04', '.04'])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 2
    assert not (tmp_path/'out').exists()


def test_cli_passes_explicit_stack_grip_span_to_formal_harness(monkeypatch, tmp_path):
    from scripts.validation import laundry_cloth_interactions as harness
    received = []
    def capture(args):
        received.append(args)
        return 0
    monkeypatch.setattr(harness, 'run_task', capture)
    monkeypatch.setattr('sys.argv', ['probe', '--case', 'towel_stacking', '--output', str(tmp_path/'out'),
                                   '--stack-grip-span-m', '.04', '.04'])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 0 and received[0].stack_grip_span_m == [.04, .04]
