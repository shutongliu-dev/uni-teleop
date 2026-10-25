"""Settling cannot pass on frozen poses alone, or without support evidence."""
from copy import deepcopy
import subprocess
import sys
from pathlib import Path

import pytest

from bench.paths import load_json
from scripts.validation.validate_scenario import scene_stability


def state():
    return {'objects': {'part': {'position_m': [.5, .1, .8],
        'linear_velocity_m_s': [0., 0., 0.], 'angular_velocity_rad_s': [0., 0., 0.]}}}


def test_static_pose_without_observed_support_is_not_accepted():
    result = scene_stability(state(), state(), {})
    assert not result['all_dynamic_objects_supported_and_stable']
    assert result['objects']['part']['support_impulse_ns'] == 0.


@pytest.mark.parametrize('field,value', [
    ('position_m', [.5, .1, .7]),
    ('linear_velocity_m_s', [.1, 0., 0.]),
    ('angular_velocity_rad_s', [0., 0., 1.]),
])
def test_contact_does_not_hide_fall_or_unsettled_motion(field, value):
    after = deepcopy(state())
    after['objects']['part'][field] = value
    result = scene_stability(state(), after, {'part': 1.})
    assert not result['all_dynamic_objects_supported_and_stable']


def test_supported_small_settle_is_reported_without_task_claim():
    after = state()
    after['objects']['part']['position_m'][2] -= .001
    result = scene_stability(state(), after, {'part': .02})
    assert result['all_dynamic_objects_supported_and_stable']
    assert result['objects']['part']['drift_m'] == pytest.approx(.001)
    assert 'task_success' not in result


def test_inventory_change_and_nonfinite_state_reject():
    with pytest.raises(ValueError, match='inventory'):
        scene_stability(state(), {'objects': {}}, {'part': 1.})
    after = state()
    after['objects']['part']['linear_velocity_m_s'][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        scene_stability(state(), after, {'part': 1.})


@pytest.mark.parametrize('passed,expected_code', [(True, 0), (False, 1)])
def test_report_survives_native_style_immediate_process_exit(tmp_path, passed, expected_code):
    # A subprocess uses os._exit just like Kit. A mock that simply returns from
    # close() would miss the original bug and falsely validate unreachable code.
    source = '''
import os, sys
from pathlib import Path
from scripts.validation.validate_scenario import seal_report_before_shutdown
class ImmediateExit:
    def close(self):
        os._exit(1 if sys.exc_info()[0] is not None else 0)
seal_report_before_shutdown(Path(sys.argv[1]), {'passed': sys.argv[2] == 'true'}, ImmediateExit())
raise AssertionError('Fast shutdown returned unexpectedly')
'''
    result = subprocess.run([sys.executable, '-B', '-c', source, str(tmp_path), str(passed).lower()],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == expected_code
    report = load_json(tmp_path/'report.json')
    assert report['passed'] is passed
    assert report['native_shutdown_verified'] is False
    assert report['native_shutdown_status'] == 'not_observed_by_report'
    assert 'observed_process_exit_code' in report['external_acceptance_required']
