"""Exercise bimanual action history through ACKs, public events and HTTP status.

The worker is an explicit CPU double, not native physics qualification.
"""
from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from bench.protocol import PolicyRegistry
from bench.runtime import service
from bench.runtime.server import create_app
from bench.sensors.contact_history import BIMANUAL_PROTOCOL
from bench.sensors.contact_schema import parse_contact_history
import test_agent_service as service_tests
from test_agent_service import frozen_inputs, runner, action, join_action, session_for
from test_bimanual_contact_history import dual_metadata, dual_config, sampled_window, contact, path


@pytest.fixture
def dual_service(runner, monkeypatch, dual_config):
    class DualWorker(service_tests.FakeWorker):
        def __init__(self, *, hand_id, **kwargs):
            assert hand_id == 'dual_fr3_sharpa_w01'
            super().__init__(hand_id='sharpa_w01_right', **kwargs)
            self.arm = [f'fr3_{side}_joint{i}' for side in ('left', 'right') for i in range(1, 8)]
            self.hand = [f'{side}_test_joint{i}' for side in ('left', 'right') for i in range(22)]
            self.active = self.arm + self.hand
            self.raw['joint_positions_rad'] = dict.fromkeys(self.active, 0.)
            self.raw['joint_velocities_rad_s'] = dict.fromkeys(self.active, 0.)
            self.initial['observation'] = deepcopy(self.raw)
            self.initial['metadata']['action_spec'] = {'joint_names': self.active,
                'arm_joint_names': self.arm, 'hand_joint_names': self.hand,
                'joint_limits_rad': {name: [-1., 1.] for name in self.active}}

    monkeypatch.setattr(service_tests, 'FakeWorker', DualWorker)
    def configure(metadata, robot_id):
        assert robot_id == 'dual_fr3_sharpa_w01'
        return deepcopy(dual_config)
    monkeypatch.setattr(service, 'configure_contact_history', configure)
    return runner


def test_dual_http_action_history_includes_transient_contacts_and_preserves_prior_action(dual_service, dual_config):
    make, _, _ = dual_service
    def step(worker):
        number = round(worker.raw['sim_time_s'] / .02)
        rows = [contact(dual_config, 'left_thumb_elastomer', tick=40,
                        other=path(dual_config, 'right_index_elastomer'))] if number == 1 else []
        worker.raw.update(sampled_window(number, rows))
    api = make(registry=PolicyRegistry(), on_step=step)
    session, reply = session_for(api, robot_id='dual_fr3_sharpa_w01')
    assert reply['capabilities']['robot_id'] == 'dual_fr3_sharpa_w01'
    assert len(reply['capabilities']['arm_joints']) == 14
    assert len(reply['capabilities']['hand_joints']) == 44
    with TestClient(create_app(service=api)) as client:
        first_action = action(session, 'first', .06)
        assert client.post(f'/episodes/{session.directory.name}/actions', json=first_action).status_code == 202
        join_action(session)
        response = client.get(f'/episodes/{session.directory.name}/actions/first')
        assert response.status_code == 200
        history = response.json()['contact_history']
        assert parse_contact_history(history).protocol == BIMANUAL_PROTOCOL
        assert history['sampled_substeps'] == 120 and history['observed_control_windows'] == 3
        assert history['cross_side_contact']['contact_substeps'] == 1
        assert history['cross_side_contact']['current_contact_substeps'] == 0
        assert history['hands']['left']['regions']['thumb_pad']['contact_substeps'] == 1
        assert history['hands']['right']['regions']['index_pad']['contact_substeps'] == 1
        assert 'private_' not in response.text and '/World/' not in response.text
        session.submit(first_action)
        assert session.status('first')['contact_history'] == history
        session.submit(action(session, 'second', .02)); join_action(session)
        second = session.status('second')['contact_history']
        assert second['start_sim_time_s'] == .06 and second['sampled_substeps'] == 40
        assert second['cross_side_contact']['contact_substeps'] == 0
        assert session.status('first')['contact_history'] == history
        # Both accepted and final records use the typed public union.
        records = [event for event in session.life.events if event.action_id == 'first']
        assert records[0].contact_history.status == 'not_sampled'
        assert records[-1].contact_history.status == 'available'


@pytest.mark.parametrize('failure', ['terminal', 'missing_sampling', 'lost_ack'])
def test_dual_history_counts_only_complete_acknowledged_windows(dual_service, dual_config, failure):
    make, _, workers = dual_service
    def step(worker):
        number = round(worker.raw['sim_time_s'] / .02)
        worker.raw.update(sampled_window(number, [contact(dual_config, 'right_index_DP', window=number)]))
        if failure == 'terminal':
            worker.score['success'] = True
        elif number == 2 and failure == 'missing_sampling':
            del worker.raw['contacts']['sampling']
    session, _ = session_for(make(registry=PolicyRegistry(), on_step=step), robot_id='dual_fr3_sharpa_w01')
    worker = workers[-1]
    original = worker.request
    def request(operation, **payload):
        result = original(operation, **payload)
        if failure == 'lost_ack' and operation == 'step' and worker.raw['sim_time_s'] >= .04:
            raise RuntimeError('private_ack_failure')
        return result
    worker.request = request
    session.submit(action(session, 'move', .06)); join_action(session)
    status = session.status('move')
    history = status['contact_history']
    parse_contact_history(history)
    assert history['sampled_substeps'] == 40
    if failure == 'terminal':
        assert status['termination_reason'] == 'task_terminated'
        assert history['status'] == 'available'
        assert history['hands']['right']['regions']['index_shell']['contact_substeps'] == 1
    elif failure == 'missing_sampling':
        assert history['status'] == 'invalid' and history['sim_time_s'] == .06
        assert history['hands'] is None
    else:
        assert session.result.outcome == 'infrastructure_error'
        assert history['sim_time_s'] == .02 and history['observed_control_windows'] == 1
    assert 'private_' not in json.dumps(history)
