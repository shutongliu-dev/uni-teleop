"""Explicit observed-boundary and released robot transitions for laundry probes.

These helpers plan original joint targets. They neither move cloth states nor
establish contact, capture, collision clearance or manipulation success.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from scripts.validation.laundry_release_hand import release_hand_profile
from scripts.validation.laundry_trousers_sequence import reference_phase_indices
from scripts.validation.laundry_prepare_timing import phase_prepare_timing
from scripts.validation.laundry_released_joint_plan import phase_released_joint_plan


RELEASE_DEFAULTS = {'release_profile': 'fixed_v1',
    'release_hand_profile': 'pinch_open_v1', 'retreat_profile': 'base_directed_v1'}
CUFF_BOUNDARIES = {
    f'{feature}_{corner}': {'feature': feature, 'material_x_corner': corner,
        'inward_xy': [1. if corner == 'minimum' else -1., 1.]}
    for feature in ('left_cuff', 'right_cuff') for corner in ('minimum', 'maximum')}


def _number(value, lo, hi, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not lo <= value <= hi):
        raise ValueError(f'{name} must be a finite number in [{lo}, {hi}]')
    return float(value)


def phase_release_settings(options, index=None):
    selected = {key: options.get(key, value) for key, value in RELEASE_DEFAULTS.items()}
    if index is not None:
        override = options.get('phases', {}).get(str(index), {}).get('release_settings', {})
        if not isinstance(override, dict) or set(override)-set(RELEASE_DEFAULTS):
            raise ValueError('Phase release_settings must contain only explicit release profiles')
        selected.update(override)
    if selected['release_profile'] not in ('fixed_v1', 'cooked_hand_clearance_v2', 'cooked_opening_lift_v3'):
        raise ValueError('Unknown explicit cloth release placement profile')
    release_hand_profile(selected)
    if selected['retreat_profile'] not in ('base_directed_v1', 'finger_axis_exit_v2'):
        raise ValueError('Unknown explicit release retreat profile')
    if options.get('transport_mode', 'fold') != 'fold' and selected['release_profile'] != 'fixed_v1':
        raise ValueError('The lift diagnostic retains its original release trajectory')
    return selected


def phase_park_transition(options, index):
    value = options.get('phases', {}).get(str(index), {}).get('park_transition')
    if value is None:
        return None
    keys = {'profile', 'side', 'duration_s', 'peak_height_m'}
    if not isinstance(value, dict) or set(value) != keys or index <= 0:
        raise ValueError('Explicit park_transition requires its four fields in a later phase')
    if value['profile'] != 'cartesian_to_park_v1' or value['side'] not in ('left', 'right'):
        raise ValueError('Unknown actual wrist-to-park transition profile or side')
    seconds = _number(value['duration_s'], 1., 5., 'park transition duration_s')
    if abs(seconds/.02-round(seconds/.02)) > 1.e-9:
        raise ValueError('Park transition duration must use complete20ms control intervals')
    _number(value['peak_height_m'], .04, .15, 'park transition peak_height_m')
    return dict(value)


def phase_released_dwell(options, index):
    value = options.get('phases', {}).get(str(index), {}).get('additional_released_dwell_s', 0.)
    seconds = _number(value, 0., 5., 'additional_released_dwell_s')
    if abs(seconds/.02-round(seconds/.02)) > 1.e-9:
        raise ValueError('Released dwell requires complete20ms control intervals')
    return seconds


def validate_phase_motions(options, task_id=None, *, towel=False):
    """Check every configured phase, including phases not selected for this run."""
    phase_release_settings(options)
    sequence = reference_phase_indices(options, task_id)
    if towel and sequence is not None:
        raise ValueError('Sequential cuffs require the original garment runner')
    for key, phase in options.get('phases', {}).items():
        index = int(key)
        phase_release_settings(options, index)
        phase_park_transition(options, index)
        phase_released_dwell(options, index)
        phase_prepare_timing(options, index)
        phase_released_joint_plan(options, index)
        boundary = phase.get('folded_cuff_boundary')
        if boundary is not None and (not isinstance(boundary, str) or boundary not in CUFF_BOUNDARIES or towel
                or task_id not in (None, 'trousers_folding') or index not in ((1, 2) if sequence else (1,))):
            raise ValueError('A folded cuff override selects one original trousers phase1 boundary')
        sleeve = phase.get('sleeve_cuff_corner')
        if sleeve is not None and (sleeve not in ('minimum', 'maximum') or boundary is not None
                or towel or task_id not in (None, 'shirt_folding') or index not in (0, 1)
                or options.get('pickup_profile') != 'shirt_cuff_corners'):
            raise ValueError('A sleeve corner override requires an original shirt_cuff_corners sleeve phase')
        for setting in phase.get('hands', {}).values():
            if 'corner_relative_bottom_m' not in setting:
                continue
            _number(setting['corner_relative_bottom_m'], -.010, 0., 'corner_relative_bottom_m')
            if boundary is None or 'bottom_z_m' in setting:
                raise ValueError('Corner-relative hand height requires a cuff corner and excludes bottom_z_m')


def cuff_boundary_descriptions(task_id, index, descriptions, configured, *, pickup_profile=None):
    sleeve = configured.get('sleeve_cuff_corner')
    if sleeve is not None:
        if (task_id != 'shirt_folding' or index not in (0, 1) or pickup_profile != 'shirt_cuff_corners'
                or sleeve not in ('minimum', 'maximum') or 'folded_cuff_boundary' in configured):
            raise ValueError('Explicit sleeve corner requires its original named shirt boundary')
        return [{'feature': 'right_cuff' if index == 0 else 'left_cuff',
            'material_y_corner': sleeve,
            'inward_xy': [-1. if index == 0 else 1., 1. if sleeve == 'minimum' else -1.]}]
    selected = configured.get('folded_cuff_boundary')
    if selected is None:
        return descriptions
    if (task_id != 'trousers_folding' or index != 1 or not isinstance(selected, str)
            or selected not in CUFF_BOUNDARIES):
        raise ValueError('Explicit folded cuff boundary requires original trousers phase1')
    return [dict(CUFF_BOUNDARIES[selected])]


def observed_corner_bottom(pick, setting, table_z):
    if 'corner_relative_bottom_m' not in setting:
        return setting.get('bottom_z_m', table_z+.0003), None
    offset = _number(setting['corner_relative_bottom_m'], -.010, 0., 'corner_relative_bottom_m')
    if pick.get('natural_corner_vertex_id') is None or 'bottom_z_m' in setting:
        raise ValueError('Observed corner height requires a real selected corner and no absolute override')
    z = float(np.asarray(pick['edge_world_m'], float)[2])
    if not np.isfinite([z, table_z]).all() or not table_z-.002 <= z <= table_z+.08:
        raise ValueError('Fresh corner must remain close to the real table support')
    bottom = max(table_z+.0003, z+offset)
    return bottom, {'schema': 'laundry_observed_corner_hand_height.v1',
        'natural_corner_vertex_id': int(pick['natural_corner_vertex_id']),
        'observed_corner_z_m': z, 'relative_bottom_m': offset,
        'table_top_z_m': float(table_z), 'minimum_table_clearance_m': .0003,
        'commanded_hand_bottom_z_m': float(bottom),
        'table_floor_clamped': z+offset < table_z+.0003,
        'source': 'Fresh actual corner position; original full-hand pinch envelope',
        'cloth_state_written': False}


class CartesianParkTransition:
    """Real wrist arc and orientation interpolation while no cloth is held."""
    def __init__(self, start, end, start_hand, end_hand, *, peak_height_m):
        self.start, self.end = np.asarray(start, float), np.asarray(end, float)
        for matrix in (self.start, self.end):
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1.e-12, rtol=0.)
                    or not np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1.e-6, rtol=0.)
                    or abs(np.linalg.det(matrix[:3, :3])-1.) > 1.e-6):
                raise ValueError('Actual and original park wrist transforms must be rigid and finite')
        if (not start_hand or set(start_hand) != set(end_hand)
                or not np.isfinite(list(start_hand.values())+list(end_hand.values())).all()):
            raise ValueError('Original start and park hand joint inventories must match')
        peak = _number(peak_height_m, .04, .15, 'park transition peak_height_m')
        self.start_hand, self.end_hand = dict(start_hand), dict(end_hand)
        self.controls = np.array([self.start[:3, 3],
            (self.start[:3, 3]+self.end[:3, 3])/2, self.end[:3, 3]])
        self.controls[1, 2] = max(self.controls[0, 2], self.controls[2, 2])+2*peak
        self.rotations = Slerp([0., 1.], Rotation.from_matrix(
            np.array([self.start[:3, :3], self.end[:3, :3]])))

    def targets(self, amount):
        u = _number(amount, 0., 1., 'park transition fraction')
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotations(u).as_matrix()
        matrix[:3, 3] = (1-u)**2*self.controls[0]+2*(1-u)*u*self.controls[1]+u*u*self.controls[2]
        hand = {name: value+u*(self.end_hand[name]-value)
                for name, value in self.start_hand.items()}
        return matrix, hand

    def receipt(self):
        return {'profile': 'cartesian_to_park_v1', 'actual_start_wrist_world_matrix': self.start.tolist(),
            'original_park_wrist_world_matrix': self.end.tolist(), 'controls_m': self.controls.tolist(),
            'start_hand_joint_positions_rad': self.start_hand,
            'original_park_hand_joint_positions_rad': self.end_hand,
            'orientation': 'Actual start-to-original-park wrist quaternion SLERP',
            'time_law': 'Original minimum-jerk action fraction', 'cloth_state_written': False,
            'scope': 'Original joint-driven released wrist transition; preflight IK and actual collision evidence remain required.'}
