"""Task budgets preserve public action boundaries and acknowledged state."""
from copy import deepcopy
import json
import pytest

from bench.paths import load_json
from bench.protocol import Budgets
from test_agent_service import runner, frozen_inputs, session_for, action, join_action
from test_task_referee_contract import install_contract, score


def test_task_budget_stops_after_ack_and_preserves_ongoing_physical_referee(
        runner, frozen_inputs, monkeypatch):
    install_contract(monkeypatch, frozen_inputs)
    make, _, workers = runner
    def exhausted(worker):
        worker.score = {**score(), 'status': 'incomplete', 'terminal': True,
                        'reason': 'budget_exhausted'}
    session, _ = session_for(make(on_step=exhausted), budgets=Budgets(sim_time_s=10.))
    submitted = action(session, duration=.1)
    session.submit(submitted); join_action(session)
    steps = [payload for operation, payload in workers[-1].calls if operation == 'step']
    assert len(steps) == 1 and steps[0]['action_id'] == submitted['action_id']
    assert session.sequence == 1 and session.raw['sim_time_s'] == .02
    assert session.result.outcome == 'sim_timeout' and session.result.completion_sim_s is None
    public = session.task_requirements()
    assert public['referee']['state'] == 'ongoing' and public['referee']['terminal'] is False
    assert public['episode'] == {'terminal': True, 'outcome': 'sim_timeout'}


@pytest.mark.parametrize('ack_time', [.02, .04], ids=['exact_sim_horizon', 'late_sim_ack'])
@pytest.mark.parametrize('terminal', [False, True], ids=['ongoing', 'success'])
def test_simulation_budget_keeps_every_ack_in_command_sequence_and_contact_records(
        runner, frozen_inputs, monkeypatch, ack_time, terminal):
    install_contract(monkeypatch, frozen_inputs)
    make, _, workers = runner

    def native_reply(worker):
        worker.raw['sim_time_s'] = ack_time
        worker.score = score('success' if terminal else 'ongoing')

    session, _ = session_for(make(on_step=native_reply), budgets=Budgets(sim_time_s=.02))
    observed = []
    observe = session.contact_history.observe

    def record_contact(raw):
        observed.append(deepcopy(raw))
        return observe(raw)

    monkeypatch.setattr(session.contact_history, 'observe', record_contact)
    session.submit(action(session)); join_action(session)
    expected = 'success' if terminal and ack_time == .02 else 'sim_timeout'
    assert session.result.outcome == expected
    assert session.result.completion_sim_s == (.02 if expected == 'success' else None)
    assert session.sequence == 1 and session.snapshot.sequence == 1
    assert session.raw['sim_time_s'] == session.result.elapsed_sim_s == ack_time
    steps = [payload for operation, payload in workers[-1].calls if operation == 'step']
    records = [json.loads(line) for line in (session.recorder.directory/'events.jsonl').read_text().splitlines()]
    commands = [row['payload'] for row in records if row['kind'] == 'robot_joint_command']
    assert commands == [steps[0]['command']] and len(steps) == 1
    assert observed == [session.raw]
    evidence = load_json(session.directory/'private/evaluator_evidence.json')
    assert evidence == score('success' if terminal else 'ongoing')
    assert session.task_requirements()['episode'] == {'terminal': True, 'outcome': expected}


def test_one_public_action_identity_is_forwarded_for_every_native_command(runner):
    make, _, workers = runner
    session, _ = session_for(make())
    submitted = action(session, duration=.1)
    session.submit(submitted); join_action(session)
    steps = [payload for operation, payload in workers[-1].calls if operation == 'step']
    assert len(steps) > 1 and {payload['action_id'] for payload in steps} == {submitted['action_id']}


def test_public_direct_hold_keeps_acknowledged_targets_for_requested_duration(runner):
    make, _, workers = runner
    session, _ = session_for(make(acknowledged_targets=True))
    reply = session.functions.invoke({'request_id': 'hold_check', 'function': 'hold',
        'arguments': {'action_id': 'study_release_hold', 'observation_sequence': 0,
                      'duration_s': .3}})
    assert reply.ok
    join_action(session)
    steps = [payload for operation, payload in workers[-1].calls if operation == 'step']
    assert session.result is None and session.raw['sim_time_s'] == .3 and len(steps) == 15
    assert all(set(payload['command']['joint_positions_rad'].values()) == {0.} for payload in steps)


def test_task_action_budget_allows_last_whole_action_then_blocks_new_physics(
        runner, frozen_inputs, monkeypatch):
    install_contract(monkeypatch, frozen_inputs)
    make, _, workers = runner
    def first_admitted(worker):
        worker.score = {**score(), 'action_budget': {'maximum': 1, 'admitted': 1}}
    session, _ = session_for(make(on_step=first_admitted), budgets=Budgets(max_actions=10, sim_time_s=10.))
    session.score['action_budget'] = {'maximum': 1, 'admitted': 0}
    session.submit(action(session, duration=.1)); join_action(session)
    assert session.result is None and session.raw['sim_time_s'] == .1
    assert len([op for op, _ in workers[-1].calls if op == 'step']) == 5
    session.submit(action(session, name='extra', duration=.1)); join_action(session)
    assert session.result.outcome == 'action_budget' and session.raw['sim_time_s'] == .1
    assert len([op for op, _ in workers[-1].calls if op == 'step']) == 5
    assert session.task_requirements()['referee']['state'] == 'ongoing'


@pytest.mark.parametrize('budget', [None, {'maximum': True, 'admitted': 0},
    {'maximum': 0, 'admitted': 0}, {'maximum': 1, 'admitted': False},
    {'maximum': None, 'admitted': -1}, {'maximum': None}])
def test_malformed_task_admission_budget_never_steps(runner, budget):
    make, _, workers = runner
    session, _ = session_for(make())
    session.score['action_budget'] = budget
    session.submit(action(session)); join_action(session)
    assert session.result.outcome == 'infrastructure_error' and session.raw['sim_time_s'] == 0.
    assert not [op for op, _ in workers[-1].calls if op == 'step']


