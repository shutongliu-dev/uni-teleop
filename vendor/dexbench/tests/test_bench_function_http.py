"""Successful callable HTTP execution and sealed reads with a CPU worker only."""
import hashlib
import json

from fastapi.testclient import TestClient
from pydantic import TypeAdapter
import pytest

from bench.paths import load_json
from bench.protocol import PolicyRegistry
from bench.protocol.functions import ActionStatus, FunctionResponse
from bench.protocol.schemas import EpisodeResult
from bench.runtime.server import create_app
from test_agent_service import (frozen_inputs, runner, session_for, action,
    join_action, publish_targets, TimedHoldTeacher, PRIVATE_MARKER)


RESPONSE = TypeAdapter(FunctionResponse)


def post(client, endpoint, function, arguments, request_id, expected_status=200):
    response = client.post(endpoint + '/call', json={
        'function': function, 'arguments': arguments, 'request_id': request_id})
    assert response.status_code == expected_status, response.text
    value = RESPONSE.validate_json(response.content)
    assert value.ok and value.function == function and value.request_id == request_id
    assert PRIVATE_MARKER not in response.text
    return value


def commands(worker):
    return [payload['command'] for operation, payload in worker.calls if operation == 'step']


def artifact_hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


@pytest.mark.parametrize('track', ['direct_joints', 'system0'])
def test_http_motion_poll_replay_and_sealed_result_share_episode_state(runner, track):
    make, _, workers = runner
    options = {'acknowledged_targets': True, 'on_step': publish_targets}
    selection = {}
    if track == 'system0':
        registry = PolicyRegistry()
        registry.register(TimedHoldTeacher())
        options['registry'] = registry
        selection = {'track': track, 'policy_id': 'test_teacher', 'policy_version': 'v1'}
    api = make(**options)
    session, _ = session_for(api, **selection)
    worker = workers[-1]
    action_id = 'http_motion'
    duration = .4 if track == 'direct_joints' else .06
    if track == 'direct_joints':
        function = 'move_joints'
        arguments = {key: value for key, value in action(session, name=action_id, duration=duration).items()
                     if key != 'kind'}
        expected_reason = 'direct_duration_completed'
    else:
        function = 'hold'
        arguments = {'action_id': action_id, 'observation_sequence': session.sequence, 'duration_s': duration}
        expected_reason = 'time_hold_completed'
    endpoint = '/episodes/' + session.spec.episode_id
    with TestClient(create_app(service=api)) as client:
        accepted = post(client, endpoint, function, arguments, 'http/submit', expected_status=202)
        assert accepted.result.status == 'accepted'
        assert accepted.context.budget.action_attempts == 1
        assert accepted.context.request_recorded is True
        # Poll the public function route; no direct facade/status call establishes completion.
        for number in range(100):
            polled = post(client, endpoint, 'get_action', {'action_id': action_id}, f'http/poll/{number}')
            if polled.result.status == 'completed':
                break
            assert polled.result.status in ('accepted', 'running'), polled.result
        else:
            pytest.fail('CPU worker action did not finish during HTTP polling')
        join_action(session)
        assert polled.result.termination_reason == expected_reason
        assert polled.result.elapsed_sim_s == pytest.approx(duration)
        assert polled.context.budget.action_attempts == 1
        assert polled.context.budget.elapsed_sim_s == pytest.approx(duration)
        assert polled.context.episode_finished is False
        assert session.result is None
        executed = commands(worker)
        assert len(executed) == round(duration / .02)
        assert all(command['duration_s'] == .02 for command in executed)
        assert {name: worker.raw['robot']['joint_targets_rad'][name]
                for name in worker.active} == executed[-1]['joint_positions_rad']
        assert session.snapshot.proprioception.commanded_joint_positions_rad == executed[-1]['joint_positions_rad']
        if track == 'direct_joints':
            assert executed[-1]['joint_positions_rad'] == {
                **arguments['arm_joint_positions_rad'], **arguments['hand_joint_positions_rad']}
        else:
            assert all(value == 0. for command in executed for value in command['joint_positions_rad'].values())

        legacy = client.get(endpoint + '/actions/' + action_id)
        assert legacy.status_code == 200
        assert ActionStatus.model_validate(legacy.json()) == polled.result
        replay = post(client, endpoint, function, arguments, 'http/new-request-id')
        assert replay.result == polled.result
        assert replay.context.budget.action_attempts == 1
        assert replay.context.budget.elapsed_sim_s == pytest.approx(duration)
        assert commands(worker) == executed
        assert session.clock.action_attempts == 1 and session.life.active_action_id is None
        rows = [json.loads(line) for line in (session.recorder.directory / 'events.jsonl').read_text().splitlines()]
        assert sum(row['kind'] == 'agent_action' for row in rows) == 1

        finished = post(client, endpoint, 'finish_episode', {}, 'http/finish')
        assert isinstance(finished.result, EpisodeResult)
        assert finished.result.outcome == 'cancelled'  # Explicit finish is not task success.
        assert finished.result.action_attempts == 1
        assert finished.result.elapsed_sim_s == pytest.approx(duration)
        assert finished.context.episode_finished and finished.context.request_recorded
        assert session.recorder.closed and worker.closed
        assert EpisodeResult.model_validate_json((session.recorder.directory / 'result.json').read_bytes()) == finished.result
        manifest = load_json(session.recorder.directory / 'artifacts.json')
        for name, expected in manifest['files_sha256'].items():
            assert hashlib.sha256((session.recorder.directory / name).read_bytes()).hexdigest() == expected

        sealed = artifact_hashes(session.directory)
        sealed_calls = list(worker.calls)
        for function in ('get_result', 'finish_episode', 'get_result'):
            result = post(client, endpoint, function, {}, 'http/sealed/' + function)
            assert isinstance(result.result, EpisodeResult)
            assert result.result == finished.result
            assert result.context.episode_finished and result.context.request_recorded is False
            assert result.context.budget.action_attempts == 1
        assert artifact_hashes(session.directory) == sealed
        assert worker.calls == sealed_calls
    # App shutdown also leaves a previously sealed episode unchanged.
    assert artifact_hashes(session.directory) == sealed
