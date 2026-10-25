"""Recovery tests use a public-only CPU double; no model or native physics."""
from copy import deepcopy

import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPClient, BenchHTTPError
from agent.model_client.learning.robot_api import ActionPendingError, ActionRecoveryError, RobotAPI


def observation():
    return {'episode_id': 'ep1', 'sequence': 0, 'sim_time_s': 0.,
        'instruction': 'Move', 'profile': 'vision_proprio',
        'proprioception': {'joint_positions_rad': {'joint': 0.},
                          'joint_velocities_rad_s': {'joint': 0.}},
        'cameras': [], 'last_action_status': 'idle'}


def command(action_id='a1', sequence=0):
    return {'kind': 'direct_joints', 'action_id': action_id, 'observation_sequence': sequence,
            'duration_s': .2, 'arm_joint_positions_rad': {'joint': .1}, 'hand_joint_positions_rad': {}}


class Bench:
    def __init__(self):
        self.obs = observation()
        self.posts = []
        self.queries = []
        self.cancels = []
        self.actions = {}
        self.lose_acceptance = False
        self.lose_status = False
        self.deadline = None

    def submit(self, episode, payload):
        self.posts.append((episode, deepcopy(payload)))
        receipt = {'action_id': payload['action_id'], 'status': 'accepted', 'sim_time_s': self.obs['sim_time_s']}
        self.actions[payload['action_id']] = receipt
        if self.lose_acceptance:
            raise BenchHTTPError('transport failed', category='transport_timeout')
        return deepcopy(receipt)

    def get_action(self, episode, action_id):
        self.queries.append((episode, action_id))
        if self.lose_status:
            raise BenchHTTPError('transport failed', category='transport_error')
        if action_id not in self.actions:
            raise BenchHTTPError('unknown action', status_code=404)
        return deepcopy(self.actions[action_id])

    def get_observation(self, episode):
        return deepcopy(self.obs)

    def cancel_action(self, episode, action_id):
        self.cancels.append((episode, action_id))
        return {**self.actions[action_id], 'cancellation_requested': True}

    def finish_action(self, action_id='a1', status='completed', steps=1):
        self.obs.update(sequence=self.obs['sequence']+steps,
                        sim_time_s=self.obs['sim_time_s']+steps*.2,
                        last_action_id=action_id, last_action_status=status)
        self.actions[action_id] = {'action_id': action_id, 'status': status, 'sim_time_s': self.obs['sim_time_s']}


def api(tmp_path, bench, **kwargs):
    return RobotAPI(bench, tmp_path/'actions', episode_id='ep1', trial_id='trial1', **kwargs)


def test_acceptance_response_lost_queries_same_id_and_never_resubmits(tmp_path):
    bench = Bench()
    host = api(tmp_path, bench)
    bench.lose_acceptance = True
    with pytest.raises(BenchHTTPError):
        host.submit(command(), bench.obs)
    assert host.pending['action_id'] == 'a1'
    with pytest.raises(ActionPendingError):
        host.submit(command('a2'), bench.obs)
    bench.finish_action()
    restored = api(tmp_path, bench)
    settled = restored.reconcile()
    assert settled['receipt']['status'] == 'completed' and restored.pending is None
    assert len(bench.posts) == 1 and bench.queries == [('ep1', 'a1')]
    events = restored.journal.read()
    assert [row['kind'] for row in events][:3] == [
        'action_ledger_bound', 'action_prepared', 'action_submit_started']
    assert events[1]['payload']['command'] == command()
    assert len(events[1]['payload']['payload_sha256']) == 64


def test_terminal_read_lost_survives_restart_without_new_post(tmp_path):
    bench = Bench()
    host = api(tmp_path, bench)
    host.submit(command(), bench.obs)
    bench.finish_action()
    bench.lose_status = True
    with pytest.raises(BenchHTTPError):
        host.reconcile()
    bench.lose_status = False
    restored = api(tmp_path, bench)
    assert restored.wait(timeout_s=1)['observation']['sequence'] == 1
    assert len(bench.posts) == 1 and restored.pending is None
    with pytest.raises(ActionRecoveryError, match='reused'):
        restored.submit(command(sequence=1), bench.obs)


def test_active_action_and_cancel_ack_do_not_unlock(tmp_path):
    bench = Bench()
    host = api(tmp_path, bench)
    host.submit(command(), bench.obs)
    bench.actions['a1']['status'] = 'running'
    assert host.reconcile() is None
    assert host.cancel()['cancellation_requested']
    with pytest.raises(ActionPendingError):
        host.submit(command('a2'), bench.obs)
    restored = api(tmp_path, bench)
    with pytest.raises(ActionPendingError, match='already requested'):
        restored.cancel()
    assert len(bench.cancels) == 1
    bench.finish_action(status='cancelled')
    assert restored.reconcile()['receipt']['status'] == 'cancelled'
    restored.submit(command('a2', sequence=1), bench.obs)
    assert len(bench.posts) == 2


@pytest.mark.parametrize('change', [
    lambda value: value.update(episode_id='another'),
    lambda value: value.update(last_action_id='another'),
    lambda value: value.update(last_action_status='running'),
    lambda value: value.update(sequence=0),
    lambda value: value.update(sim_time_s=.1),
])
def test_terminal_with_stale_or_mismatched_snapshot_remains_pending(tmp_path, change):
    bench = Bench()
    host = api(tmp_path, bench)
    host.submit(command(), bench.obs)
    bench.finish_action()
    correct = deepcopy(bench.obs)
    change(bench.obs)
    with pytest.raises(BenchHTTPError):
        host.reconcile()
    assert host.pending is not None
    with pytest.raises(ActionPendingError):
        host.assert_idle()
    bench.obs = correct
    assert host.reconcile()['receipt']['status'] == 'completed'


@pytest.mark.parametrize('status', ['failed', 'cancelled', 'completed'])
def test_zero_step_terminal_publishes_new_action_identity(tmp_path, status):
    bench = Bench()
    host = api(tmp_path, bench)
    host.submit(command(), bench.obs)
    bench.finish_action(status=status, steps=0)
    result = host.reconcile()
    assert result['observation']['sequence'] == 0
    assert result['receipt']['status'] == status and host.pending is None


def test_404_is_not_permission_to_replay_or_start_another_action(tmp_path):
    bench = Bench()
    host = api(tmp_path, bench)
    host.submit(command(), bench.obs)
    bench.actions.clear()
    with pytest.raises(BenchHTTPError) as caught:
        api(tmp_path, bench).reconcile()
    assert caught.value.status_code == 404
    with pytest.raises(ActionPendingError):
        api(tmp_path, bench).submit(command('a2'), bench.obs)
    assert len(bench.posts) == 1


@pytest.mark.parametrize('changed', ['episode', 'trial', 'contract'])
def test_restart_requires_original_episode_trial_and_contract(tmp_path, changed):
    bench = Bench()
    host = api(tmp_path, bench)
    host.submit(command(), bench.obs)
    kwargs = {'episode_id':'ep1', 'trial_id':'trial1', 'binding':{}}
    kwargs[{'episode':'episode_id', 'trial':'trial_id', 'contract':'binding'}[changed]] = (
        {'bundle_hash':'different'} if changed == 'contract' else 'other')
    with pytest.raises(ActionRecoveryError, match='binding'):
        RobotAPI(bench, tmp_path/'actions', **kwargs)
    assert len(bench.posts) == 1


def test_persisted_prepared_intent_is_not_replayed_after_crash(tmp_path):
    from agent.model_client.learning.robot_api import _hash
    bench = Bench()
    host = api(tmp_path, bench)
    host.journal.append('action_prepared', {'action_id':'a1', 'command':command(),
        'payload_sha256':_hash(command()), 'expected_sequence':0,
        'initial_observation':{'sequence':0, 'sim_time_s':0., 'last_action_id':None,
                               'sha256':_hash(bench.obs)}})
    restored = api(tmp_path, bench)
    assert restored.pending['submit_started'] is False
    with pytest.raises(BenchHTTPError):
        restored.reconcile()
    with pytest.raises(ActionPendingError):
        restored.submit(command('a2'), bench.obs)
    assert bench.posts == []


def test_submit_is_durable_before_transport_runs(tmp_path):
    bench = Bench()
    host = api(tmp_path, bench)
    submit = bench.submit
    def check(episode, payload):
        records = host.journal.read()
        assert records[-1]['kind'] == 'action_submit_started'
        assert records[-2]['payload']['command'] == payload
        return submit(episode, payload)
    bench.submit = check
    host.submit(command(), bench.obs)
    assert len(bench.posts) == 1


def test_http_recovery_wrappers_use_original_identifiers_and_no_post_retry():
    seen = []
    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == 'POST':
            raise httpx.ReadTimeout('private upstream body', request=request)
        return httpx.Response(200, json={'action_id':'a1','status':'running'})
    client = BenchHTTPClient('http://bench.test', client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.get_action('ep1', 'a1')['status'] == 'running'
    with pytest.raises(BenchHTTPError) as caught:
        client.cancel_action('ep1', 'a1')
    assert caught.value.public_metadata() == {'error_category':'transport_timeout', 'http_status':None}
    assert 'private' not in str(caught.value)
    assert seen == [('GET','/episodes/ep1/actions/a1'), ('POST','/episodes/ep1/actions/a1/cancel')]


def test_wait_deadline_keeps_pending_without_post_retry(tmp_path, monkeypatch):
    from agent.model_client.learning import robot_api
    bench = Bench()
    host = api(tmp_path, bench)
    host.submit(command(), bench.obs)
    now = [0.]
    monkeypatch.setattr(robot_api.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(robot_api.time, 'sleep', lambda value: now.__setitem__(0, now[0]+value))
    with pytest.raises(BenchHTTPError) as caught:
        host.wait(timeout_s=.5, poll_interval_s=.25)
    assert caught.value.category == 'request_budget_exhausted'
    assert host.pending is not None and len(bench.posts) == 1 and bench.deadline is None


def test_second_host_cannot_submit_while_first_owns_ledger(tmp_path):
    from agent.model_client.learning.journal import JournalBusyError
    bench = Bench()
    first = api(tmp_path, bench)
    second = api(tmp_path, bench)
    with first.journal.transaction():
        with pytest.raises(JournalBusyError):
            second.submit(command(), bench.obs)
    assert bench.posts == []
    first.submit(command(), bench.obs)
    with pytest.raises(ActionPendingError):
        second.submit(command('a2'), bench.obs)
    assert len(bench.posts) == 1


def test_interaction_recovers_lost_acceptance_and_continues_from_new_observation(monkeypatch, tmp_path):
    from test_online_model_runner import setup_loop
    from agent.model_client import runner
    args, bench, model = setup_loop(monkeypatch, tmp_path, [
        {'action':'joint_delta', 'delta_rad':{'arm':.01}, 'duration_s':.2}, {'action':'stop'}])
    submit = bench.submit
    def lose_response(episode, payload):
        submit(episode, payload)
        raise BenchHTTPError('Response lost', category='transport_timeout')
    bench.submit = lose_response
    report = runner.run(args)
    assert report['error_type'] is None
    assert report['submitted_actions'] == report['completed_actions'] == len(bench.commands) == 1
    assert report['model_input_observation_sequences'] == [0, 20]
    from agent.model_client.learning.journal import Journal
    events = Journal(args.output/'host'/'action_journal').read()
    assert sum(row['kind'] == 'action_submit_started' for row in events) == 1
    assert any(row['kind'] == 'action_transport_error' for row in events)
    assert events[-1]['kind'] == 'action_reconciled'


def test_interaction_finish_does_not_erase_unresolved_action(monkeypatch, tmp_path):
    from test_online_model_runner import setup_loop
    from agent.model_client import runner
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [
        {'action':'joint_delta', 'delta_rad':{'arm':.01}, 'duration_s':.2}])
    def lose_status(*args):
        raise BenchHTTPError('Response lost', category='transport_timeout')
    bench.get_action = lose_status
    report = runner.run(args)
    assert report['error_type'] == 'BenchHTTPError' and len(bench.commands) == 1
    assert bench.finished
    from agent.model_client.learning.journal import Journal
    events = Journal(args.output/'host'/'action_journal').read()
    assert events[-1]['kind'] == 'action_transport_error'
    assert not any(row['kind'] == 'action_reconciled' for row in events)


def test_restart_cannot_switch_bench_origin(tmp_path):
    bench = Bench()
    bench.base_url = 'http://bench-one.test'
    host = api(tmp_path, bench)
    host.submit(command(), bench.obs)
    bench.base_url = 'http://bench-two.test'
    with pytest.raises(ActionRecoveryError, match='binding'):
        api(tmp_path, bench)
    assert len(bench.posts) == 1


@pytest.mark.parametrize('replacement', [{'action_id':'wrong'}, {'status':'unknown'},
    {'sim_time_s':float('nan')}, {'episode_id':'wrong'}])
def test_invalid_action_receipt_never_releases_pending(tmp_path, replacement):
    bench = Bench()
    host = api(tmp_path, bench)
    host.submit(command(), bench.obs)
    bench.finish_action()
    bench.actions['a1'].update(replacement)
    with pytest.raises(BenchHTTPError):
        host.reconcile()
    assert host.pending is not None and len(bench.posts) == 1


def test_journal_failure_before_post_is_not_counted_as_a_submission(monkeypatch, tmp_path):
    from test_online_model_runner import setup_loop
    from agent.model_client import runner
    from agent.model_client.learning.journal import Journal
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [
        {'action':'joint_delta', 'delta_rad':{'arm':.01}, 'duration_s':.2}])
    append = Journal.append
    def disk_error(self, kind, payload, **kwargs):
        if kind == 'action_prepared':
            raise OSError('simulated journal write failure')
        return append(self, kind, payload, **kwargs)
    monkeypatch.setattr(Journal, 'append', disk_error)
    report = runner.run(args)
    assert report['error_type'] == 'OSError'
    assert report['primitive_attempts'] == 1
    assert report['submitted_actions'] == 0 and bench.commands == []


@pytest.mark.parametrize('lose_response', [False, True])
def test_action_crossing_host_deadline_keeps_pending_and_unknown_result(monkeypatch, tmp_path, lose_response):
    from types import SimpleNamespace
    from test_online_model_runner import setup_loop
    from agent.model_client import runner, interaction
    from agent.model_client.learning.journal import Journal
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [
        {'action':'joint_delta', 'delta_rad':{'arm':.01}, 'duration_s':.2}])
    now = [100.]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(runner, 'time', clock)
    monkeypatch.setattr(interaction, 'time', clock)
    submit = bench.submit
    def late_submission(episode, payload):
        accepted = submit(episode, payload)
        now[0] = 161.
        if lose_response:
            raise BenchHTTPError('Response lost', category='transport_timeout')
        return accepted
    bench.submit = late_submission
    report = runner.run(args)
    assert report['error_type'] == 'BenchHTTPError'
    assert report['task_outcome'] == 'unknown' and report['task_success'] is None
    assert report['run_status'] == 'infra_error'
    assert report['submitted_actions'] == len(bench.commands) == 1
    assert bench.finished
    events = Journal(args.output/'host'/'action_journal').read()
    assert not any(row['kind'] == 'action_reconciled' for row in events)
    if lose_response:
        assert report['run_error']['error_category'] == 'transport_timeout'


@pytest.mark.parametrize('prior_state', ['empty', 'rejected_attempt', 'pending', 'settled'])
def test_new_interaction_session_cannot_reset_existing_trial_budgets(tmp_path, prior_state):
    from agent.model_client.actions import ActionValidationError
    from agent.model_client.interaction import InteractionSession
    bench = Bench()
    def create_session():
        return InteractionSession(bench, 'ep1', {}, {}, bench.obs, tmp_path/'run',
            control_mode='joints', visual_input='rgb', tactile_input='off',
            limits=None, deadline=None, max_actions=1, trial_id='trial1')
    first = create_session()
    if prior_state == 'rejected_attempt':
        with pytest.raises(ActionValidationError):
            first.invoke('hold', {'duration_s':True}, folder=tmp_path/'decision', decision_index=0)
        assert first.move_attempts == 1 and first.submitted == 0
    elif prior_state in ('pending', 'settled'):
        first.action_api.submit(command(), bench.obs)
        if prior_state == 'settled':
            bench.finish_action()
            first.action_api.reconcile()
    previous_posts = len(bench.posts)
    with pytest.raises(ActionRecoveryError, match='cannot reopen'):
        create_session()
    assert len(bench.posts) == previous_posts
    # Bookkeeping recovery remains available, without reopening a model/code
    # session or granting fresh action/model/invalid-request budgets.
    recovery = RobotAPI(bench, tmp_path/'run/host/action_journal', episode_id='ep1',
                       trial_id='trial1', binding=first.action_api.binding['contract'])
    if prior_state == 'pending':
        recovery.cancel()
        assert recovery.pending is not None
        bench.finish_action(status='cancelled')
        assert recovery.reconcile()['receipt']['status'] == 'cancelled'
    assert recovery.pending is None
    assert len(bench.posts) == previous_posts


@pytest.mark.parametrize('status', ['accepted', 'running'])
def test_unlimited_wall_still_has_finite_action_status_query_budget(tmp_path, monkeypatch, status):
    from agent.model_client.learning import robot_api
    bench = Bench()
    host = api(tmp_path, bench, max_status_queries=3)
    host.submit(command(), bench.obs)
    bench.actions['a1']['status'] = status
    monkeypatch.setattr(robot_api.time, 'sleep', lambda _: None)
    with pytest.raises(BenchHTTPError) as caught:
        host.wait(timeout_s=None)
    assert caught.value.category == 'request_budget_exhausted'
    assert bench.queries == [('ep1', 'a1')] * 3
    assert len(bench.posts) == 1 and bench.cancels == []
    assert host.pending is not None and bench.deadline is None
    assert host.binding['recovery_policy'] == {
        'max_status_queries':3, 'scope':'per_wait_status_operations'}
    exhausted = host.journal.read()[-1]
    assert exhausted['kind'] == 'action_wait_exhausted'
    assert exhausted['payload']['status_queries'] == 3
    assert exhausted['payload']['max_status_queries'] == 3
    assert exhausted['payload']['reason'] == 'status_query_limit'
    with pytest.raises(ActionPendingError):
        host.submit(command('a2'), bench.obs)


def test_wait_can_tighten_but_never_enlarge_bound_query_limit(tmp_path, monkeypatch):
    from agent.model_client.learning import robot_api
    bench = Bench()
    host = api(tmp_path, bench)
    host.submit(command(), bench.obs)
    monkeypatch.setattr(robot_api.time, 'sleep', lambda _: None)
    assert host.max_status_queries == host.binding['recovery_policy']['max_status_queries'] == 1200
    with pytest.raises(ValueError, match='bound'):
        host.wait(timeout_s=None, max_status_queries=1201)
    assert bench.queries == []
    with pytest.raises(BenchHTTPError):
        host.wait(timeout_s=None, max_status_queries=3)
    assert len(bench.queries) == 3
    events = host.journal.read()
    started = next(row for row in events if row['kind'] == 'action_wait_started')
    assert started['payload']['max_status_queries'] == 3
    assert events[-1]['payload']['wait_id'] == started['payload']['wait_id']


@pytest.mark.parametrize('value', [None, True, False, 0, -1, 1.5, '3'])
def test_bound_query_limit_requires_positive_integer(tmp_path, value):
    with pytest.raises(ValueError, match='positive integer'):
        api(tmp_path, Bench(), max_status_queries=value)
    assert not (tmp_path/'actions').exists()


def test_query_budget_is_bound_across_recovery_restarts(tmp_path):
    bench = Bench()
    host = api(tmp_path, bench, max_status_queries=3)
    host.submit(command(), bench.obs)
    with pytest.raises(ActionRecoveryError, match='binding'):
        api(tmp_path, bench, max_status_queries=4)
    restored = api(tmp_path, bench, max_status_queries=3)
    bench.finish_action()
    assert restored.wait(timeout_s=None)['receipt']['status'] == 'completed'
    assert len(bench.posts) == 1
