"""Engineering privileges stay confined to a bounded independent tool wrench."""
from copy import deepcopy
from io import StringIO
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation.gear_transmission_checks import probe_contract
from scripts.validation.validate_gear_scene import StageObserver, GoalObserver
from scripts.validation.validate_assembly_contacts import Observer


def observer(monkeypatch, *, loaded, speed=0., object_speed=0.):
    calls = []
    view = SimpleNamespace(
        get_velocities=lambda: np.array([[0., 0., 0., 0., 0., speed]]),
        apply_forces_and_torques_at_position=lambda *args: calls.append(args))
    value = StageObserver.__new__(StageObserver)
    # Only the socket view is exposed: writing a gear would fail this test.
    value.env = SimpleNamespace(_object_views={'socket_crank': view},
        _tensor=np.asarray, _one=np.array([0], dtype=np.int32))
    value.contract = probe_contract()
    value.loaded_case = True
    value.phase_definition = {'id': 'forward' if loaded else 'settle',
        'loaded': loaded, 'target_speed_rad_s': np.deg2rad(15.) if loaded else 0.}
    value.stream, value.wrench_stream = StringIO(), StringIO()
    value.rows, value.applied_wrench_calls = [], 0
    row = {'phase': value.phase_definition['id'], 'native_step': 1, 'time_s': .0005,
        'poses_xyzw': {'input_gear': [.4, -.13, .802, 0., 0., 0., 1.]},
        'velocities': {'input_gear': [object_speed, 0., 0., 0., 0., 0.]},
        'robot_interference': [], 'maximum_penetration_m': 0.,
        'raw_contacts': [{'actor0': '/World/Objects/input_gear',
            'actor1': '/World/Fixtures/gear_workplate/collision',
            'normal_impulses_ns': [1.e-3], 'normals': [[0., 0., 1.]]}]}

    def step(self, *args, **kwargs):
        sample = deepcopy(row)
        self.rows.append(sample)
        self.stream.write(json.dumps(sample)+'\n')
        return 'native-step-result'

    monkeypatch.setattr(Observer, 'step', step)
    return value, calls


def test_passive_phase_never_calls_force_api_and_retains_original_points(monkeypatch):
    value, calls = observer(monkeypatch, loaded=False, speed=.01)
    assert value.step(render=False) == 'native-step-result'
    assert calls == [] and value.applied_wrench_calls == 0
    receipt = json.loads(value.wrench_stream.getvalue())
    assert receipt['torque_nm'] == [0., 0., 0.] and not receipt['wrench_api_called']
    assert 'raw_contacts' in json.loads(value.stream.getvalue())
    assert 'raw_contacts' not in value.rows[0]
    assert value.rows[0]['contact_totals'][0]['signed_actor0_impulse_vector_ns'] == [0., 0., .001]


def test_active_phase_applies_only_clipped_world_tool_torque(monkeypatch):
    value, calls = observer(monkeypatch, loaded=True, speed=-1.)
    value.step(render=False)
    assert len(calls) == value.applied_wrench_calls == 1
    force, torque, position, indices, world = calls[0]
    np.testing.assert_array_equal(force, [[0., 0., 0.]])
    np.testing.assert_allclose(torque, [[0., 0., .8]])
    assert position is None and world is True
    assert indices.tolist() == [0]
    receipt = json.loads(value.wrench_stream.getvalue())
    assert receipt['body'] == 'socket_crank' and not receipt['explicit_wake_called']
    assert receipt == value.rows[0]['wrench']


def test_motion_abort_preserves_the_offending_raw_frame_and_wrench(monkeypatch):
    value, _ = observer(monkeypatch, loaded=True, object_speed=.6)
    with pytest.raises(RuntimeError, match='offending frame retained'):
        value.step(render=False)
    assert json.loads(value.stream.getvalue())['velocities']['input_gear'][0] == .6
    assert json.loads(value.wrench_stream.getvalue())['native_step'] == 1
    assert len(value.rows) == 1


def test_nonfinite_tool_state_aborts_before_actuation(monkeypatch):
    value, calls = observer(monkeypatch, loaded=True, speed=float('nan'))
    with pytest.raises(RuntimeError, match='finite-state'):
        value.step(render=False)
    assert calls == [] and value.rows == [] and not value.wrench_stream.getvalue()


@pytest.mark.parametrize('turn_degrees, expected_calls', [(25., 1), (29., 0)])
def test_goal_cuts_tool_loading_only_after_measured_output_reaches_target(
        monkeypatch, turn_degrees, expected_calls):
    value, calls = observer(monkeypatch, loaded=True)
    value.__class__ = GoalObserver
    value.goal_contract = {'target_rad': np.deg2rad(30.),
                           'cutoff_tolerance_rad': np.deg2rad(3.)}
    value.goal_baseline_rad, value.goal_reached, value.goal_control = 0., False, {}
    value.phase_definition['id'] = 'goal_drive'
    half = np.deg2rad(turn_degrees)/2
    value.rows.append({'time_s': 2., 'poses_xyzw': {
        'output_gear': [.4, .13, .802, 0., 0., np.sin(half), np.cos(half)]}})
    value.step(render=False)
    assert len(calls) == expected_calls
    receipt = json.loads(value.wrench_stream.getvalue())
    assert receipt['goal_control']['load_cutoff'] == (expected_calls == 0)
    assert receipt['wrench_api_called'] == bool(expected_calls)
    assert value.phase_definition['loaded'] is True  # Frozen phase stays unchanged.
