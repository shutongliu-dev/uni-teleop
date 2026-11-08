from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation.laundry_observed_sleeve_press import (
    STAGES, observed_sleeve_peaks, plan_observed_sleeve_press, relocated_press_targets,
)


def fixture():
    q = np.asarray([[-.2, 0., .95], [-.2, .1, .95], [.2, 0., .96], [.2, .1, .93]])
    row = {'id': 'towel', 'rest_vertices_m': q.tolist(), 'unfolded_reference_vertices_m': q.tolist()}
    task = {'asset_metadata': {'garment_kind': 'shirt', 'cloths': [row], 'design': {
        'fold_sequence_axis_hinge_sign_radius_m': [[0, 0., -1, .007], [0, 0., 1, .007]]}}}
    return task, q, {'towel': np.zeros(4, dtype=bool)}, np.zeros(2, dtype=bool)


def test_peak_identity_and_tie_break_are_original_material_not_world_side():
    task, q, masks, enabled = fixture()
    original = deepcopy(task), q.copy()
    q[:, 0] *= -1  # Actual cloth can cross the world-space hinge.
    peaks, proof = observed_sleeve_peaks(task, q, masks, enabled)
    assert peaks['left']['vertex_id'] == 0
    assert peaks['right']['vertex_id'] == 2
    assert peaks['left']['world_position_m'] == [.2, 0., .95]
    assert proof['all_cloth_masks_zero'] and proof['all_assists_disabled']
    assert task == original[0]


@pytest.mark.parametrize('case', ['mask', 'assist', 'missing_mask', 'wrong_mask', 'nan', 'truncated', 'other_garment', 'two_cloths'])
def test_fresh_released_complete_inventory_required(case):
    task, q, masks, enabled = fixture()
    if case == 'mask': masks['towel'][0] = True
    if case == 'assist': enabled[1] = True
    if case == 'missing_mask': masks = {}
    if case == 'wrong_mask': masks['towel'] = np.zeros(3)
    if case == 'nan': q[0, 2] = np.nan
    if case == 'truncated': q = q[:3]
    if case == 'other_garment': task['asset_metadata']['garment_kind'] = 'trousers'
    if case == 'two_cloths': task['asset_metadata']['cloths'] *= 2
    with pytest.raises(ValueError): observed_sleeve_peaks(task, q, masks, enabled)


class Arm:
    def __init__(self, side):
        self.names = [side + '_' + name for name in ('x', 'y', 'z')]
        self.active = [SimpleNamespace(name=n, lower=-5., upper=5., velocity=5.) for n in self.names]

    def forward(self, q, base):
        value = np.eye(4)
        value[:3, 3] = [q[n] for n in self.names]
        return {'tip': value}

    def inverse(self, target, tip, seed, **kwargs):
        assert kwargs['position_tolerance'] == .0005 and kwargs['rotation_tolerance'] == .005
        return {'converged': True, 'joint_positions_rad': dict(zip(self.names, target[:3, 3]))}


def planner_fixture():
    task, q, masks, enabled = fixture()
    models, park, selected, oldpeaks, export = {}, {}, {}, {}, {'colliders': []}
    for side, x in [('left', -.15), ('right', .15)]:
        arm = Arm(side)
        h = side + '_finger'
        body = side + '_hand_C_MC'
        hand = SimpleNamespace(names=[h], active=[SimpleNamespace(name=h, lower=-2., upper=2., velocity=5.)],
                               forward=lambda q, name=body: {name: np.eye(4)})
        models[side] = SimpleNamespace(arm=arm, hand=hand, arm_tip='tip', arm_base=np.eye(4),
            wrist_offset=np.eye(4), hand_posture_prior=SimpleNamespace(violation=lambda finger, q: False))
        park.update(dict(zip(arm.names, [x, 0., 1.2])))
        park[h] = .1
        T = np.eye(4); T[:3, 3] = [x, 0., .925]
        selected[side] = {'world_flange': T.tolist(), 'upward_branch_samples': [
            {'ik': {'joint_positions_rad': dict(zip(arm.names, [x, 0., 1.025]))}}]}
        oldpeaks[side] = {'peak_world': [x, 0., .96]}
        export['colliders'].append({'body_name': body, 'mesh_to_body_column_matrix': np.eye(4).tolist(),
            'pieces': [{'conservative_outer_vertices': [[-.02, -.02, -.009], [.02, -.02, 0.],
                                                       [0., .02, 0.], [0., 0., .02]]}]})
    template = {'planned_duration_s': 17., 'attachments': False, 'cloth_state_writes': 0,
                'selected_sides': selected, 'peaks': oldpeaks,
                'phase_endpoints': [{'label': l, 'duration_s': d} for l, d in STAGES]}
    return task, q, park, {'joint_positions_rad': park}, models, export, template, masks, enabled


def preflight(samples):
    return {'all_clear': True, 'samples': [{'sample_index': i, 'passed': True} for i in range(len(samples))],
            'checked_geometry': ['environment', 'arm_self', 'dual_arm', 'whole_cooked_hand']}


def run(f=None, **kwargs):
    data = planner_fixture() if f is None else f
    task, q, actual, initial, models, export, template, masks, enabled = data
    return plan_observed_sleeve_press(task, q, actual, initial, models, export, template,
        control_masks=masks, assist_enabled=enabled, collision_preflight=kwargs.get('preflight', preflight))


def test_fresh_targets_preserve_palm_height_and_all_lateral_motion_precedes_contact():
    f = planner_fixture()
    original = deepcopy(f[0]), f[1].copy(), deepcopy(f[2]), deepcopy(f[6])
    result = run(f)
    assert result['planned_duration_s'] == 17.
    assert result['attachments'] is False and result['cloth_state_writes'] == 0
    assert result['maximum_original_joint_velocity_fraction'] < 1.
    assert len(result['samples']) == 860
    for side in ('left', 'right'):
        targets = result['targets'][side]
        assert targets['press'][2][3] == .925
        assert targets['precontact_whole_cloth_clearance_m'] >= .002 - 1.e-12
        for label in ['slow_physical_palm_compression', 'physical_palm_hold', 'vertical_release_of_palm_contact']:
            poses = [s['joint_positions_rad'] for s in result['samples'] if s['label'] == label]
            assert np.ptp([p[side+'_x'] for p in poses]) < 1.e-12
            assert np.ptp([p[side+'_y'] for p in poses]) < 1.e-12
    assert f[0] == original[0] and np.array_equal(f[1], original[1])
    assert f[2] == original[2] and f[6] == original[3]


@pytest.mark.parametrize('case', ['missing', 'partial', 'duplicate', 'failure', 'uncooked'])
def test_no_inherited_or_partial_collision_qualification(case):
    def reject(samples):
        r = preflight(samples)
        if case == 'missing': return None
        if case == 'partial': r['samples'].pop()
        if case == 'duplicate': r['samples'][-1]['sample_index'] = 0
        if case == 'failure': r['samples'][10]['passed'] = False
        if case == 'uncooked': r['checked_geometry'].remove('whole_cooked_hand')
        return r
    with pytest.raises(ValueError): run(preflight=reject)


def test_unreachable_hover_and_original_velocity_are_fail_closed():
    f = planner_fixture()
    f[4]['left'].arm.inverse = lambda *a, **k: {'converged': False, 'joint_positions_rad': {}}
    with pytest.raises(ValueError, match='not reachable'): run(f)
    f = planner_fixture()
    f[4]['left'].arm.active[0].velocity = .001
    with pytest.raises(ValueError, match='velocity'): run(f)


def test_canonical_frame_is_applied_once():
    task, q, masks, enabled = fixture()
    # This explicit world transform is not a second implicit garment yaw.
    task['asset_metadata']['design']['fold_coordinate_frame'] = {
        'schema': 'laundry_fold_coordinate_frame.v1', 'yaw_degrees': 90., 'pivot_xy_m': [0., 0.]}
    original = np.asarray(task['asset_metadata']['cloths'][0]['unfolded_reference_vertices_m'])
    original[:, :2] = np.c_[-original[:, 1], original[:, 0]]
    task['asset_metadata']['cloths'][0]['unfolded_reference_vertices_m'] = original.tolist()
    peaks, _ = observed_sleeve_peaks(task, q, masks, enabled)
    assert peaks['left']['vertex_id'] == 0 and peaks['right']['vertex_id'] == 2
