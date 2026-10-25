"""Only the declared public action-status fields can reach model feedback."""
from copy import deepcopy
import json

import pytest

from agent.model_client.bench_http import BenchHTTPError
from agent.model_client.visibility import feedback_view


def receipt():
    return {'action_id': 'action_1', 'status': 'completed', 'sim_time_s': .2}


def execution_state():
    return {'protocol': 'robot_execution_state_v1', 'sim_time_s': .2,
            'completion_scope': 'requested_geometry', 'objective_status': 'reached',
            'reference_stationary': True}


def contact_history():
    return {'protocol': 'robot_rigid_contact_history_v1', 'status': 'unavailable',
        'action_id': 'action_1', 'start_sim_time_s': 0., 'sim_time_s': .2,
        'duration_s': .2, 'physics_dt_s': None, 'observed_control_windows': 0,
        'sampled_substeps': 0,
        'coverage': {'all_declared_robot_rigid_links': False, 'cloth': False,
            'contact_directions': False, 'taxels': False, 'force_calibrated': False,
            'completeness': 'assumed_from_native_step_contract'},
        'regions': None, 'pad_coactivity': None, 'self_contact': None,
        'semantics': 'positive solved normal impulses; coactivity does not imply opposition or a common object; each unique pair counted once per involved region'}


def test_minimal_response_does_not_invent_missing_fields():
    original = receipt()
    assert feedback_view(original) == original
    assert feedback_view(None) is None
    assert feedback_view({**original, 'diagnostic': None}) == {**original, 'diagnostic': None}


def test_current_status_and_cancel_fields_survive_without_extra_defaults():
    source = {**receipt(), 'episode_finished': False, 'elapsed_sim_s': .2,
              'diagnostic': {'code': 'convergence_timeout', 'elapsed_sim_s': .2},
              'termination_reason': 'controller_converged',
              'robot_execution_state': execution_state(), 'contact_history': contact_history(),
              'cancellation_requested': True}
    original = deepcopy(source)
    assert feedback_view(source) == source == original
    projected = feedback_view(source, tactile_input='off')
    assert 'contact_history' not in projected
    assert projected == {key: value for key, value in source.items() if key != 'contact_history'}
    assert source == original


@pytest.mark.parametrize('key', ['privileged_object_poses', 'task_score', 'tactile', 'error',
                                 'detail', 'wall_time_s', 'episode_id'])
@pytest.mark.parametrize('tactile', ['on', 'off'])
def test_unknown_top_level_fields_are_rejected_with_no_private_error_text(key, tactile):
    private = 'PRIVATE_CANARY /home/credentials secret object state'
    with pytest.raises(BenchHTTPError) as caught:
        feedback_view({**receipt(), key: private}, tactile_input=tactile)
    assert str(caught.value) == 'Invalid public primitive feedback'
    assert caught.value.category == 'invalid_response'
    assert private not in json.dumps(caught.value.public_metadata())
    assert caught.value.__cause__ is None


@pytest.mark.parametrize('field,value', [
    ('diagnostic', {'code': 'convergence_timeout', 'private_object': 'PRIVATE_CANARY'}),
    ('diagnostic', {'code': 'PRIVATE_CANARY'}),
    ('diagnostic', {'code': 'convergence_timeout', 'geometry_tolerances': {
        'stage': 'completion', 'wrist_position_m': .01, 'wrist_orientation_rad': .01,
        'private_object': 'PRIVATE_CANARY'}}),
    ('robot_execution_state', {**execution_state(), 'private_target': 'PRIVATE_CANARY'}),
    ('robot_execution_state', {**execution_state(), 'reference_stationary': 1}),
    ('contact_history', {**contact_history(), 'private_object': 'PRIVATE_CANARY'}),
    ('contact_history', {**contact_history(), 'coverage': {
        **contact_history()['coverage'], 'private_object': 'PRIVATE_CANARY'}}),
    ('contact_history', {**contact_history(), 'action_id': 'different_action'}),
])
@pytest.mark.parametrize('tactile', ['on', 'off'])
def test_nested_contracts_are_strict_even_when_tactile_hidden(field, value, tactile):
    with pytest.raises(BenchHTTPError) as caught:
        feedback_view({**receipt(), field: value}, tactile_input=tactile)
    assert str(caught.value) == 'Invalid public primitive feedback'
    assert 'PRIVATE_CANARY' not in str(caught.value)


@pytest.mark.parametrize('field,value', [
    ('sim_time_s', float('nan')), ('sim_time_s', True), ('sim_time_s', -1.),
    ('elapsed_sim_s', float('inf')), ('elapsed_sim_s', '0.2'),
    ('episode_finished', 1), ('episode_finished', None),
    ('cancellation_requested', 'false'), ('cancellation_requested', None),
    ('termination_reason', 'PRIVATE_CANARY'), ('status', 'PRIVATE_CANARY'),
    ('action_id', {'private': 'PRIVATE_CANARY'}),
])
def test_top_level_values_are_not_coerced_or_leaked(field, value):
    with pytest.raises(BenchHTTPError) as caught:
        feedback_view({**receipt(), field: value})
    assert caught.value.public_metadata() == {'error_category': 'invalid_response', 'http_status': None}


def test_model_loop_never_receives_unknown_receipt_fields(monkeypatch, tmp_path):
    from test_online_model_runner import setup_loop
    from agent.model_client import runner
    args, bench, model = setup_loop(monkeypatch, tmp_path, [
        {'action': 'joint_delta', 'delta_rad': {'arm': .01}, 'duration_s': .2},
        {'action': 'stop'}])
    original = bench.get_action
    def status(episode_id, action_id):
        return {**original(episode_id, action_id), 'privileged_object_poses': 'PRIVATE_CANARY'}
    bench.get_action = status
    report = runner.run(args)
    assert report['error_type'] == 'BenchHTTPError'
    assert report['run_status'] == 'infra_error'
    assert len(model.requests) == 1
    assert 'PRIVATE_CANARY' not in json.dumps(model.requests)
    # This boundary does not erase raw host audit evidence.
    assert 'PRIVATE_CANARY' in (args.output / 'decisions/0000/execution_source.json').read_text()
