"""The wake comparison is a separate unchanged-body, bounded-load experiment."""
from copy import deepcopy
import json
import math
import pytest

from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.task import build_task_inputs
from scripts.validation.gear_transmission_checks import CONTRACT as LOAD_CONTRACT
from scripts.validation.probe_gear_tool_wake import CONTRACT, make_inputs, wake_single_body


def test_wake_uses_native_all_indices_only_for_single_selected_body():
    class View:
        count = 1
        calls = 0

        def wake_up(self):
            self.calls += 1

    view = View()
    wake_single_body(view)
    assert view.calls == 1
    for count in (0, 2):
        view.count = count
        with pytest.raises(ValueError, match='exactly one'):
            wake_single_body(view)
    assert view.calls == 1


def test_wake_cases_differ_only_by_explicit_wake():
    a, b = deepcopy(CONTRACT['cases'])
    assert a.pop('id') == 'natural'
    assert b.pop('id') == 'explicit_wake'
    assert a.pop('wake_at_loaded_phase_start') is False
    assert b.pop('wake_at_loaded_phase_start') is True
    assert a == b == {}
    assert sum(round(p['duration_s']/CONTRACT['dt_s']) for p in CONTRACT['phases']) == 4600
    assert all(round(p['duration_s']/.02)*.02 == p['duration_s'] for p in CONTRACT['phases'])
    assert CONTRACT['maximum_torque_nm'] == LOAD_CONTRACT['maximum_torque_nm']
    assert CONTRACT['coefficient_nm_s_per_rad'] == LOAD_CONTRACT['speed_load_coefficient_nm_s_per_rad']
    assert [p['target_rad_s'] for p in CONTRACT['phases'] if p['loaded']] == [math.radians(20.), -math.radians(20.)]


def test_isolated_socket_and_support_preserve_existing_physics_and_robot():
    original = build_task_inputs(solver_type='PGS')
    inputs = make_inputs()
    assert inputs['scene']['objects'] == [s for s in original['scene']['objects'] if s['id'] == 'socket_crank']
    assert inputs['scene']['fixtures'] == [s for s in original['scene']['fixtures'] if s['id'] in ('robot_stand', 'socket_crank_rest')]
    for key in ('robot', 'isaacsim', 'workroom'):
        assert inputs['scene'][key] == original['scene'][key]
    assert inputs['initial_state']['joint_positions_rad'] == original['initial_state']['joint_positions_rad']
    assert inputs['initial_state']['objects'] == {}
    assert json.loads(json.dumps(inputs, allow_nan=False)) == inputs
