"""Phase/regrasp contracts; synthetic positions are not physical success evidence."""
from copy import deepcopy
import numpy as np
import pytest

from scenes.workrooms.laundry_room.garment_folding_assets import candidate_metadata, _fold_up
from scripts.validation.laundry_garment_folding_motion import build_folding_motion


def inputs(task_id):
    metadata = candidate_metadata(task_id)
    task = {'id': task_id, 'asset_metadata': metadata,
            'evaluation': {'thresholds': {'stable_s': .5, 'max_time_s': 90.}}}
    return task, {'physics': {'dt_s': .001}}


@pytest.mark.parametrize('task_id,phase_count', [('shirt_folding', 3), ('trousers_folding', 2)])
def test_phase_timing_fully_releases_before_next_regrasp(task_id, phase_count):
    task, scene = inputs(task_id)
    motion = build_folding_motion(task, scene)
    assert len(motion.phases) == phase_count
    assert motion.phases[-1]['anchor_roles'] == (['fold', 'fold'] if task_id == 'shirt_folding' else ['hold', 'fold'])
    assert motion.duration_s-motion.release_time_s == 5.
    for i, p in enumerate(motion.phases):
        assert motion.active(i, 0, p['grasp_time_s'])
        assert motion.active(i, 1, p['grasp_time_s'])
        for anchor in (0, 1):
            assert not motion.active(i, anchor, p['fold_release_time_s'])
        if i+1 < len(motion.phases):
            assert p['fold_release_time_s'] < motion.phases[i+1]['grasp_time_s']


@pytest.mark.parametrize('task_id', ['shirt_folding', 'trousers_folding'])
def test_observed_regrasp_does_not_modify_free_vertices_or_material_shape(task_id):
    task, scene = inputs(task_id)
    motion = build_folding_motion(task, scene)
    q = np.asarray(task['asset_metadata']['cloths'][0]['unfolded_reference_vertices_m'])
    before = q.copy()
    record = motion.bind_phase(0, q)
    assert np.array_equal(q, before)
    assert record['controlled_fraction'] < .10
    assert len(record['anchors']) == 2
    for i, anchor in enumerate(record['anchors']):
        assert anchor['locality']['world_diameter_m'] <= .075
        assert max(anchor['locality']['material_fragment_diameters_m']) <= .075
        origin, rotation = motion.pose(0, i, motion.phases[0]['grasp_time_s'])
        assert np.allclose(origin, anchor['origin_m'])
        assert rotation == [1., 0., 0., 0.]
    fold = record['anchors'][1]
    destination, rotation = motion.pose(0, 1, motion.phases[0]['move_end_s'])
    assert np.allclose(destination, fold['destination_m'])
    with pytest.raises(ValueError):
        motion.bind_phase(0, q)


def test_unsettled_official_worn_shell_cannot_be_bound_as_flat_target():
    task, scene = inputs('shirt_folding')
    motion = build_folding_motion(task, scene)
    with pytest.raises(ValueError, match='table-supported'):
        motion.bind_phase(0, task['asset_metadata']['cloths'][0]['initial_vertices_m'])


@pytest.mark.parametrize('task_id', ['shirt_folding'])
def test_final_transverse_fold_controls_both_actual_edge_ends(task_id):
    task, scene = inputs(task_id)
    motion = build_folding_motion(task, scene)
    q = np.asarray(task['asset_metadata']['cloths'][0]['unfolded_reference_vertices_m'])
    for i, phase in enumerate(motion.phases):
        record = motion.bind_phase(i, q)
        if i == len(motion.phases)-1:
            assert [a['role'] for a in record['anchors']] == ['fold', 'fold']
            assert np.linalg.norm(np.array(record['anchors'][0]['origin_m'])
                                  -record['anchors'][1]['origin_m']) > .075
            for j, anchor in enumerate(record['anchors']):
                assert anchor['destination_m'][1] > record['observed_hinge_m']
                assert np.allclose(motion.pose(i, j, phase['move_end_s'])[0], anchor['destination_m'])
        q = _fold_up(q, *phase['reference_fold'], .9006)


@pytest.mark.parametrize('kwargs', [dict(final_passive_s=.1), dict(fold_duration_s=.2),
                                   dict(peak_height_m=.3), dict(settle_s=float('nan'))])
def test_invalid_or_incomplete_motion_is_rejected_before_simulation(kwargs):
    with pytest.raises(ValueError):
        build_folding_motion(*inputs('shirt_folding'), **kwargs)


def test_optional_body_arc_does_not_raise_the_sleeve_arcs():
    motion = build_folding_motion(*inputs('shirt_folding'), body_fold_peak_height_m=.12)
    assert [phase['peak_height_m'] for phase in motion.phases] == [.06, .06, .12]


@pytest.mark.parametrize('task_id', ['shirt_folding'])
def test_rotated_final_hem_uses_y_separated_patches_for_opposed_arms(task_id):
    task, scene = inputs(task_id)
    task['asset_metadata'] = candidate_metadata(task_id, source_yaw_quarter_turns=1)
    motion = build_folding_motion(task, scene)
    q = np.asarray(task['asset_metadata']['cloths'][0]['unfolded_reference_vertices_m'])
    for i, phase in enumerate(motion.phases):
        record = motion.bind_phase(i, q)
        if i == len(motion.phases)-1:
            a, b = record['anchors']
            assert abs(a['origin_m'][1]-b['origin_m'][1]) > .075
            assert phase['reference_fold'][0] == 0
            for anchor in (a, b):
                assert anchor['destination_m'][0] < record['observed_hinge_m']
        q = _fold_up(q, *phase['reference_fold'], .9006)


def test_trouser_final_fold_uses_waist_hold_and_one_local_paired_cuff_clamp():
    task, scene = inputs('trousers_folding')
    task['asset_metadata'] = candidate_metadata('trousers_folding', source_yaw_quarter_turns=1)
    motion = build_folding_motion(task, scene)
    q = np.asarray(task['asset_metadata']['cloths'][0]['unfolded_reference_vertices_m'])
    first = motion.bind_phase(0, q)
    assert [anchor['role'] for anchor in first['anchors']] == ['fold', 'fold']
    assert abs(first['anchors'][0]['origin_m'][0]-first['anchors'][1]['origin_m'][0]) > .60
    for anchor in first['anchors']:
        assert anchor['destination_m'][1] > first['observed_hinge_m']
    q = _fold_up(q, *motion.phases[0]['reference_fold'], .9006)
    bound = motion.bind_phase(1, q)
    assert [anchor['role'] for anchor in bound['anchors']] == ['hold', 'fold']
    hold, fold = bound['anchors']
    assert np.linalg.norm(np.array(hold['origin_m'])-fold['origin_m']) > .30
    assert fold['destination_m'][0] < bound['observed_hinge_m']
    assert fold['locality']['world_diameter_m'] < .075
    phase = motion.phases[1]
    assert phase['hold_release_time_s'] == phase['move_begin_s']+.5*(phase['move_end_s']-phase['move_begin_s'])
    material = np.asarray(task['asset_metadata']['cloths'][0]['rest_vertices_m'])
    assert np.all(material[hold['vertex_ids'], 0] > .020)
    assert np.all(material[hold['vertex_ids'], 0] < .065)
    withdrawn, _ = motion.pose(1, 0, phase['hold_release_time_s']+.30)
    assert withdrawn[2] == pytest.approx(hold['origin_m'][2]+.03)


@pytest.mark.parametrize('task_id,turns', [('shirt_folding', 0), ('shirt_folding', 1),
                                         ('trousers_folding', 0), ('trousers_folding', 1)])
def test_optional_engineering_patch_roll_flips_wall_order_without_moving_free_state(task_id, turns):
    from scipy.spatial.transform import Rotation
    task, scene = inputs(task_id)
    task['asset_metadata'] = candidate_metadata(task_id, source_yaw_quarter_turns=turns)
    motion = build_folding_motion(task, scene, grip_rotation='fold_hinge')
    q = np.asarray(task['asset_metadata']['cloths'][0]['unfolded_reference_vertices_m'])
    before = q.copy()
    bound = motion.bind_phase(0, q)
    phase = motion.phases[0]
    for i, anchor in enumerate(bound['anchors']):
        _, start = motion.pose(0, i, phase['move_begin_s'])
        _, finish = motion.pose(0, i, phase['move_end_s'])
        assert np.allclose(start, [1., 0., 0., 0.])
        rotated_normal = Rotation.from_quat(np.roll(finish, -1)).apply([0., 0., 1.])
        assert np.allclose(rotated_normal, [0., 0., -1. if anchor['role'] == 'fold' else 1.])
        if anchor['role'] == 'fold':
            _, midway = motion.pose(0, i, (phase['move_begin_s']+phase['move_end_s'])/2)
            normal = Rotation.from_quat(np.roll(midway, -1)).apply([0., 0., 1.])
            axis, _, sign, _ = phase['reference_fold']
            assert normal[axis] == pytest.approx(-sign)
    assert np.array_equal(q, before)
    assert motion.metadata['target_shape_written'] is False


def test_general_yaw_uses_outer_material_grasps_and_world_hinge_quaternions():
    from scipy.spatial.transform import Rotation
    from scenes.workrooms.laundry_room.garment_folding_assets import transform_fold_frame, _rotate_degrees
    task, scene = inputs('trousers_folding')
    task['asset_metadata'] = candidate_metadata('trousers_folding', source_yaw_degrees=70.,
                                               source_center_xy_m=[.75, -.55])
    motion = build_folding_motion(task, scene, grip_rotation='fold_hinge',
        first_leg_grasp_material_x_m=-.15, second_hold_material_region_xy_m=[.020, .065, .18, .24])
    frame = motion.metadata['fold_coordinate_frame']
    q = np.asarray(task['asset_metadata']['cloths'][0]['unfolded_reference_vertices_m'])
    material = np.asarray(task['asset_metadata']['cloths'][0]['rest_vertices_m'])
    for phase_index, phase in enumerate(motion.phases):
        before = q.copy()
        bound = motion.bind_phase(phase_index, q)
        assert np.array_equal(q, before)
        assert bound['controlled_fraction'] < .10
        for index, anchor in enumerate(bound['anchors']):
            assert np.allclose(np.asarray(anchor['offsets_m'])+anchor['origin_m'], q[anchor['vertex_ids']])
            assert anchor['locality']['world_diameter_m'] < .075
            if anchor['role'] == 'fold':
                end, rotation = motion.pose(phase_index, index, phase['move_end_s'])
                assert np.allclose(end, anchor['destination_m'])
                assert np.allclose(Rotation.from_quat(np.roll(rotation, -1)).apply([0, 0, 1]), [0, 0, -1])
                _, rotation = motion.pose(phase_index, index, (phase['move_begin_s']+phase['move_end_s'])/2)
                axis, _, sign, _ = phase['reference_fold']
                normal = np.zeros((1, 3)); normal[0, axis] = -sign
                assert np.allclose(Rotation.from_quat(np.roll(rotation, -1)).apply([0, 0, 1]),
                                   _rotate_degrees(normal, 70., np.zeros(2))[0])
                local = transform_fold_frame(np.array([anchor['origin_m'], end]), frame, inverse=True)
                assert local[1, axis] == pytest.approx(2*bound['observed_hinge_m']-local[0, axis]+sign*.003)
            if phase_index == 0:
                assert abs(material[anchor['vertex_ids'], 0].mean()+.15) < .015
        local = transform_fold_frame(q, frame, inverse=True)
        q = transform_fold_frame(_fold_up(local, *phase['reference_fold'], .9006), frame)
    assert motion.phases[1]['hold_material_region_xy_m'] == [.020, .065, .18, .24]
