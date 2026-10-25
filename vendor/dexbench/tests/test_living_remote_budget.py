"""Real remote budget receipts stop the generic service without task success."""
from copy import deepcopy
from bench.protocol import Budgets
from bench.protocol.task_referee import contract_sha256
from bench.runtime import service
from test_agent_service import action, frozen_inputs, join_action, runner, session_for
from test_living_remote_evaluator import Replay, television_replay
from test_task_referee_contract import install_contract


def test_remote_budget_receipt_stops_after_first_ack_with_no_completion(
        runner, frozen_inputs, monkeypatch):
    remote = television_replay()
    # The same budget branch handles explicit exhaustion and time > max_time_s.
    # Construct a valid consecutive native control observation, then exhaust it.
    for _ in range(remote.control_steps):
        remote.send()
    receipt = remote.observe(budget_exhausted=True)
    assert receipt['success'] is False
    assert receipt['status'] == 'incomplete'
    assert receipt['terminal'] is True
    assert receipt['reason'] == 'budget_exhausted'
    assert receipt['referee']['state'] == 'ongoing'
    assert receipt['referee']['phase'] == 'budget_exhausted'
    assert receipt['referee']['terminal'] is False

    make, _, workers = runner
    install_contract(monkeypatch, frozen_inputs)
    inputs = list(service.task_inputs(None))
    task = deepcopy(inputs[2])
    contract = task['public_rules']['referee']
    # Generic CPU lifecycle fixture identity stays unchanged. Declare the real
    # remote receipt's coarse phase; forward its actual score without rewriting.
    description = remote.task['public_rules']['referee']['phases']['budget_exhausted']
    contract['phases']['budget_exhausted'] = description
    contract['reason_descriptions']['budget_exhausted'] = description
    task.update(public_rules_sha256=contract_sha256(task['public_rules']),
                public_referee_contract_sha256=contract_sha256(contract))
    inputs[2] = task
    monkeypatch.setattr(service, 'task_inputs', lambda *args: deepcopy(inputs))

    def exhausted(worker):
        worker.score = deepcopy(receipt)

    session, _ = session_for(make(on_step=exhausted), budgets=Budgets(sim_time_s=None))
    submitted = action(session, duration=.1)
    session.submit(submitted)
    join_action(session)
    steps = [payload for operation, payload in workers[-1].calls if operation == 'step']
    assert len(steps) == 1
    assert steps[0]['action_id'] == submitted['action_id']
    assert session.sequence == 1
    assert session.raw['sim_time_s'] == .02
    assert session.result.outcome == 'sim_timeout'
    assert session.result.completion_sim_s is None
    public = session.task_requirements()
    assert public['referee']['state'] == 'ongoing'
    assert public['referee']['phase'] == 'budget_exhausted'
    assert public['referee']['terminal'] is False
    assert public['episode'] == {'terminal': True, 'outcome': 'sim_timeout'}


def test_legacy_remote_budget_receipt_keeps_its_frozen_shape():
    remote = Replay()
    for _ in range(remote.control_steps):
        remote.send()
    receipt = remote.observe(budget_exhausted=True)
    assert receipt['success'] is False
    assert receipt['reason'] == 'incomplete_budget'
    assert 'status' not in receipt
    assert 'terminal' not in receipt
    assert 'referee' not in receipt
