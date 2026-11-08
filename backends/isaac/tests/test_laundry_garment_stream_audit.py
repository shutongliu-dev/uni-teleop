"""Synthetic stream/audit fixtures, never native simulation evidence."""
import io

import numpy as np
import pytest

from scenes.workrooms.laundry_room.composition import build_task_inputs
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scenes.workrooms.laundry_room.rules import GARMENT_PROFILES, GARMENT_PROFILE, LEGACY_PROFILE
from scripts.validation import audit_laundry_cloth as audit
from test_laundry_garment_rules import garment_goal
from test_laundry_stream_audit import SyntheticOracle


def fixture(task_id='shirt_stacking', profile=GARMENT_PROFILE, count=255):
    value = build_task_inputs(task_id, rule_profile=profile,
                              physics_options={'physics_dt_s': .002})
    task = value['task']
    goal = garment_goal(task)
    arrays = {'time_s': np.arange(count)*.002, 'native_step': np.arange(count),
              'hidden_constraints': np.zeros(count, dtype=int)}
    for spec in task['asset_metadata']['cloths']:
        name, n = spec['id'], len(spec['rest_vertices_m'])
        q = np.tile(goal[name], (count, 1, 1))
        q[0] = spec['initial_vertices_m']
        mask = np.zeros((count, n), dtype=bool)
        if name == 'towel':
            mask[1, 0] = True  # Declared local operation while moving to goal.
        arrays.update({name+'_q': q, name+'_qd': np.zeros_like(q),
                       name+'_control_mask': mask, name+'_triangles': np.asarray(spec['triangles'])})
    return value, arrays


def run_fixture(value, arrays, **oracle_args):
    return audit.audit_frames(arrays, value['scene'], value['task'], SequenceEvaluator(value['task']),
                              SyntheticOracle(**oracle_args), io.StringIO())


@pytest.mark.parametrize('task_id', ['shirt_stacking', 'trousers_stacking'])
@pytest.mark.parametrize('profile', GARMENT_PROFILES)
def test_actual_frozen_garment_scorer_and_terminal_gate_accept_released_operation(task_id, profile):
    value, arrays = fixture(task_id, profile)
    # The declared practical compression is acceptable, but the unchanged
    # strict diagnostic (1 mm) must continue reporting its failure.
    result = run_fixture(value, arrays, penetration=.0015)
    assert result['accepted_engineering_task_success']
    assert result['frozen_evaluator_result']['success']
    assert result['terminal_window']['passed']
    assert result['terminal_window']['samples_checked'] == 251
    assert result['terminal_window']['speed_node_percentile'] == 95
    assert result['physical_diagnostic_profile'] == LEGACY_PROFILE
    assert not result['physical_geometry_stream_passed']
    assert result['first_physical_failure']['native_step'] == 0
    assert result['frozen_evaluator_result']['strict_diagnostics']['violating_frames'] == 255


@pytest.mark.parametrize('failure', ['no_operation', 'held', 'collapse', 'terminal_intersection', 'short_window'])
def test_garment_profile_name_cannot_replace_actual_operation_release_geometry_and_stability(failure):
    value, arrays = fixture(count=255 if failure != 'short_window' else 4)
    oracle_args = {}
    if failure == 'no_operation':
        arrays['towel_q'][0] = arrays['towel_q'][1]
        arrays['towel_control_mask'][:] = False
    elif failure == 'held':
        arrays['towel_control_mask'][2:, 0] = True
    elif failure == 'collapse':
        arrays['towel_q'][2:, :, 2] = arrays['towel_q'][2:, :, 2].mean(axis=1)[:, None]
    elif failure == 'terminal_intersection':
        oracle_args['bad_at'] = 254
    result = run_fixture(value, arrays, **oracle_args)
    assert not result['frozen_evaluator_result']['success']
    assert not result['accepted_engineering_task_success']
    if failure in ('no_operation', 'terminal_intersection'):
        # A visually quiet terminal pose alone cannot qualify history/contact.
        assert result['terminal_window']['passed']
    elif failure == 'short_window':
        assert result['terminal_window']['reason_codes'] == ['terminal_window_incomplete']
    else:
        assert not result['terminal_window']['passed']


def test_garment_terminal_speed_uses_per_cloth_per_frame_p95_and_retains_maximum():
    value, arrays = fixture()
    arrays['towel_qd'][-1, 0, 0] = .2
    terminal = audit.terminal_window(arrays, SequenceEvaluator(value['task']))
    assert terminal['passed'] and terminal['max_node_speed_m_s'] == .2
    assert terminal['raw_speed_p95_m_s'] == 0.
    # Motion of one whole cloth cannot be hidden by the other stationary cloth.
    arrays['towel_qd'][-1, :, 0] = .09
    terminal = audit.terminal_window(arrays, SequenceEvaluator(value['task']))
    assert not terminal['passed']
    assert terminal['raw_speed_p95_m_s'] == pytest.approx(.09)
    assert 'terminal_raw_speed_exceeded' in terminal['reason_codes']


def test_garment_authored_geometry_keeps_the_original_strict_contact_limits():
    value, arrays = fixture(count=4)
    vertices = {name: arrays[name+'_q'][0] for name in ('towel', 'towel_base')}
    evaluator = SequenceEvaluator(value['task'])
    for compression, expected in ((0., True), (.0015, False)):
        result = audit.audit_authored_geometry(vertices, value['scene'], value['task'], evaluator,
            SyntheticOracle(penetration=compression), {'towel': [], 'towel_base': []})
        assert result['passed'] is expected
        assert not result['operation_credit']
