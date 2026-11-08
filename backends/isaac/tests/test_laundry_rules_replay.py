"""Rule-only replay binds old physics and never equates strict diagnostics to success."""
from copy import deepcopy
import io
import json

import numpy as np
import pytest

from scripts.validation import replay_laundry_rules as replay


def input_pair():
    manifest = {'task_id': 'towel_flattening', 'asset_catalog': {'sha256': 'asset'},
                'asset_files_sha256': {'cloth': 'hash'}, 'capabilities': {'robot_episodes': False}}
    scene = {'physics': {'gravity': -9.81}, 'isaacsim': {'physics_dt_s': .0005}}
    task = {'id': 'towel_flattening', 'asset_metadata': {'cloths': ['towel']},
        'public_rules': {'version': 1}, 'public_rules_sha256': 'old',
        'public_referee_contract_sha256': 'old',
        'evaluation': {'physics_dt_s': .0005, 'metadata': {'cloths': ['towel']},
                       'public_rules_sha256': 'old', 'thresholds': {'speed': .035}}}
    old = (manifest, scene, task, {'cloths': 'unchanged'})
    new = deepcopy(old)
    new[2]['public_rules'] = {'version': 2}
    new[2]['public_rules_sha256'] = new[2]['public_referee_contract_sha256'] = 'new'
    new[2]['evaluation'].update(rule_profile=replay.PROFILE,
                               public_rules_sha256='new', thresholds={'speed': .08})
    return old, new


def test_compatible_inputs_accepts_only_rule_changes():
    old, new = input_pair()
    assert replay.compatible_inputs(old, new)['only_public_rules_and_evaluation_changed']


@pytest.mark.parametrize('damage', ('scene', 'reset', 'asset', 'metadata', 'period', 'extra_task', 'profile'))
def test_rule_replay_rejects_changed_physical_inputs(damage):
    old, new = input_pair()
    if damage == 'scene':
        new[1]['physics']['gravity'] = 0
    elif damage == 'reset':
        new[3]['cloths'] = 'moved'
    elif damage == 'asset':
        new[0]['asset_files_sha256']['cloth'] = 'changed'
    elif damage == 'metadata':
        new[2]['evaluation']['metadata'] = {'cloths': ['other']}
    elif damage == 'period':
        new[2]['evaluation']['physics_dt_s'] = .002
    elif damage == 'extra_task':
        new[2]['hidden_reset'] = 'goal'
    else:
        new[2]['evaluation'].pop('rule_profile')
    with pytest.raises(replay.audit.EvidenceUnknown):
        replay.compatible_inputs(old, new)


LIMITS = {'max_penetration_m': .001, 'max_edge_stretch_ratio': 1.25,
          'min_triangle_area_ratio': .05}


def geometry_row(step=0, *, penetration=0.):
    return {'native_step': step, 'time_s': step*.002,
        'geometry': {'self_intersections': 0, 'intercloth_intersections': 0,
            'degenerate_triangles': 0, 'finite_thickness_compression_checked': True,
            'max_cloth_envelope_penetration_m': penetration},
        'mesh': {'towel': {'max_edge_stretch_ratio': 1., 'min_triangle_area_ratio': 1.,
                           'projected_area_ratio': 1.}},
        'table_envelope_penetration_m': 0., 'max_penetration_m': penetration,
        'physical_sample_valid': penetration <= .001}


def test_recorded_physics_preserves_strict_failure_with_explicit_scope():
    row = geometry_row(penetration=.0015)
    result = replay.recorded_physics(row, 0, 0., ['towel'], LIMITS)
    assert result['valid'] is False
    assert result['validity_scope'] == 'strict_geometry'
    assert result['max_penetration_m'] == .0015


@pytest.mark.parametrize('damage', ('missing_count', 'nonfinite', 'wrong_depth', 'wrong_step', 'wrong_cloth', 'false_verdict'))
def test_missing_or_inconsistent_geometry_evidence_is_unknown(damage):
    row = geometry_row()
    if damage == 'missing_count':
        row['geometry'].pop('self_intersections')
    elif damage == 'nonfinite':
        row['mesh']['towel']['max_edge_stretch_ratio'] = float('nan')
    elif damage == 'wrong_depth':
        row['max_penetration_m'] = .01
    elif damage == 'wrong_step':
        row['native_step'] = 2
    elif damage == 'wrong_cloth':
        row['mesh']['other'] = row['mesh'].pop('towel')
    else:
        row['physical_sample_valid'] = False
    with pytest.raises(replay.audit.EvidenceUnknown):
        replay.recorded_physics(row, 0, 0., ['towel'], LIMITS)


class Scorer:
    specs = {'towel': {}}
    limits = {'stable_s': .5}

    def __init__(self, *, fail_final=False):
        self.frames = []
        self.fail_final = fail_final

    def observe(self, frame):
        self.frames.append(frame)
        success = frame['native_step'] > 0 and not (self.fail_final and frame['native_step'] == 2)
        return {'status': 'success' if success else 'ongoing', 'success': success,
                'reason': 'success' if success else 'task_in_progress',
                'released_stable_s': .5 if success else 0.}


def sample_arrays():
    return {'native_step': np.arange(3), 'time_s': np.arange(3)*.002,
        'hidden_constraints': np.zeros(3, dtype=int),
        'towel_control_mask': np.zeros((3, 3), dtype=bool),
        'towel_q': np.zeros((3, 3, 3)), 'towel_qd': np.zeros((3, 3, 3)),
        'towel_triangles': np.array([[0, 1, 2]])}


def trace_file(tmp_path, *, frames=3, penetration=0.):
    path = tmp_path/'frame_audit.jsonl'
    path.write_text(''.join(json.dumps(geometry_row(i, penetration=penetration))+'\n' for i in range(frames)))
    return path


def test_task_success_is_independent_of_old_strict_geometry_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, 'terminal_window', lambda *_: {'passed': True})
    scorer, output = Scorer(), io.StringIO()
    result = replay.replay_arrays(sample_arrays(), trace_file(tmp_path, penetration=.0015),
        scorer, {'evaluation': {'thresholds': LIMITS}}, score_trace=output)
    assert result['success']
    assert result['strict_invalid_frame_count'] == 3
    assert all(f['physics']['valid'] is False for f in scorer.frames)
    assert len(output.getvalue().splitlines()) == 3


@pytest.mark.parametrize('fail_final,terminal_passed', [(True, True), (False, False)])
def test_earlier_success_cannot_replace_final_score_and_terminal_window(tmp_path, monkeypatch, fail_final, terminal_passed):
    monkeypatch.setattr(replay, 'terminal_window', lambda *_: {'passed': terminal_passed})
    result = replay.replay_arrays(sample_arrays(), trace_file(tmp_path), Scorer(fail_final=fail_final),
                                  {'evaluation': {'thresholds': LIMITS}})
    assert result['first_task_success']['native_step'] == 1
    assert not result['success']
    assert result['samples_checked'] == 3


@pytest.mark.parametrize('frames', [2, 4])
def test_replay_requires_complete_exact_frame_trace(tmp_path, monkeypatch, frames):
    monkeypatch.setattr(replay, 'terminal_window', lambda *_: {'passed': True})
    with pytest.raises(replay.audit.EvidenceUnknown, match='truncated|extra'):
        replay.replay_arrays(sample_arrays(), trace_file(tmp_path, frames=frames), Scorer(),
                             {'evaluation': {'thresholds': LIMITS}})
