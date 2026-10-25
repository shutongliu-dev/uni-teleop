"""Lifecycle checks for the engineering caller, without pretending to run physics."""
import json
from types import SimpleNamespace

import pytest

from bench.control.fingertip import ExecutorLimits
from bench.paths import load_json
from scripts.validation.validate_mechanical_functions import Script, acceptance_result, grip_execution_scope


def caller(tmp_path, diagnostic, *, ended=False, status='failed', hold_diagnostic=None):
    private = tmp_path / 'episode/private'
    private.mkdir(parents=True)
    session = SimpleNamespace(directory=private.parent, result=None, sequence=0,
        raw={'sim_time_s': 0.}, thread=None,
        snapshot=SimpleNamespace(proprioception=SimpleNamespace(commanded_joint_positions_rad={'joint': .1})))
    calls = []

    class Reply:
        def __init__(self, value): self.value = value
        def model_dump(self, **kwargs): return self.value

    def invoke(payload):
        calls.append(payload)
        if payload['function'] == 'get_action':
            session.result = {'outcome': 'safety_stop'} if ended else None
            if payload['arguments']['action_id'].endswith('_hold_grasp'):
                result = ({'status': 'failed', 'diagnostic': {'code': hold_diagnostic}}
                          if hold_diagnostic else {'status': 'completed', 'diagnostic': None})
            else:
                result = {'status': status, 'diagnostic': {'code': diagnostic}}
            session.sequence += 1
            session.raw['sim_time_s'] += .5
        else:
            result = {'status': 'accepted'}
        return Reply({'ok': True, 'result': result})

    session.functions = SimpleNamespace(invoke=invoke)
    return Script(session, tmp_path), calls


@pytest.mark.parametrize('diagnostic', ['tracking_stalled', 'convergence_timeout'])
@pytest.mark.parametrize('grasp_passed', [True, False])
def test_recoverable_close_holds_then_uses_physical_grip_gate(tmp_path, diagnostic, grasp_passed):
    script, calls = caller(tmp_path, diagnostic)
    gate = {'passed': grasp_passed, 'observed_window_s': .3, 'evidence': 'native grip receipt'}
    script.progress = lambda: {'grip_last_300ms': gate}
    row = script.execute('close', 'set_hand_targets', {}, allow_contact_failure=True)
    assert row['recoverable_contact_failure'] is True
    assert script.session.result is None
    checks = {}
    script.observe_grasp({'name': 'hold_grasp', 'duration_s': .2}, checks, close_row=row)
    assert [c['function'] for c in calls] == ['set_hand_targets', 'get_action', 'hold', 'get_action']
    assert calls[2]['arguments']['duration_s'] == 1.
    assert calls[2]['arguments']['observation_sequence'] == 1
    phases = load_json(tmp_path / 'phases.json')
    assert phases[0]['status'] == {'status': 'failed', 'diagnostic': {'code': diagnostic}}
    assert phases[1]['status']['status'] == 'completed'
    persisted = load_json(tmp_path / 'checks.json')
    recovery = persisted['close_recovery']
    assert recovery['close_status'] == phases[0]['status']  # Failure is never reclassified.
    assert recovery['hold_action_id'] == phases[1]['arguments']['action_id']
    assert recovery['state'] == 'hold_observed'
    assert recovery['opposed_grasp'] == persisted['opposed_grasp'] == gate
    assert recovery['turn_allowed_by_grip_gate'] is grasp_passed


@pytest.mark.parametrize('diagnostic,ended,status', [
    ('invalid_robot_state', False, 'failed'), ('tracking_stalled', True, 'failed'),
    ('convergence_timeout', True, 'failed'), ('trajectory_limit_guard', False, 'failed'),
    ('unknown_failure', False, 'failed'), ('tracking_stalled', False, 'cancelled')])
def test_unsafe_or_terminal_failure_cannot_enter_recovery(tmp_path, diagnostic, ended, status):
    script, calls = caller(tmp_path, diagnostic, ended=ended, status=status)
    with pytest.raises(RuntimeError, match='action failed'):
        script.execute('close', 'set_hand_targets', {}, allow_contact_failure=True)
    with pytest.raises(RuntimeError, match='Unsafe close recovery'):
        script.observe_grasp({'name': 'hold_grasp', 'duration_s': 1.}, {}, close_row=script.phases[0])
    assert len(calls) == 2
    assert 'recoverable_contact_failure' not in script.phases[0]


def test_failed_recovery_hold_is_not_itself_recoverable(tmp_path):
    script, calls = caller(tmp_path, 'tracking_stalled', hold_diagnostic='tracking_stalled')
    script.progress = lambda: {'grip_last_300ms': {'passed': True}}
    row = script.execute('close', 'set_hand_targets', {}, allow_contact_failure=True)
    checks = {}
    with pytest.raises(RuntimeError, match='hold_grasp action failed'):
        script.observe_grasp({'name': 'hold_grasp', 'duration_s': 1.}, checks, close_row=row)
    assert len(calls) == 4
    recovery = load_json(tmp_path / 'checks.json')['close_recovery']
    assert recovery['state'] == 'hold_failed'
    assert recovery['hold_status']['status'] == 'failed'
    assert recovery['turn_allowed_by_grip_gate'] is False
    assert 'opposed_grasp' not in checks


@pytest.mark.parametrize('mode', ['direct', 'feedback'])
def test_scope_does_not_claim_feedback_timeout_applies_direct_slow_cap(mode):
    plan = {'phases': [{'name': 'close', 'duration_s': 1.52, 'hand_speed_cap_rad_s': .12}]}
    scope = grip_execution_scope(mode, plan, SimpleNamespace(limits=ExecutorLimits()))
    assert scope['plan_hand_speed_cap_rad_s'] == .12
    assert scope['plan_hand_speed_cap_applied_to_close'] is (mode == 'direct')
    assert scope['feedback_hand_velocity_limit_rad_s'] == 1.
    assert scope['feedback_hand_acceleration_limit_rad_s2'] == 3.
    assert scope['feedback_close_timeout_sim_s'] == (None if mode == 'direct' else 3.)
    assert scope['timeout_controls_motion_speed'] is False


def test_grasp_only_cannot_qualify_task_or_release():
    checks = {name: {'passed': True} for name in
              ('observer_seal', 'final_source_verification', 'opposed_grasp')}
    result = {'physics_valid': True, 'outcome': 'cancelled'}
    diagnostic = acceptance_result(None, result, checks, turn_degrees=0., stop_after_grasp=True)
    assert diagnostic['passed'] is True
    assert diagnostic['task_qualified'] is False
    assert diagnostic['acceptance_scope'] == 'grasp_diagnostic_only'
    assert acceptance_result(None, result, checks, turn_degrees=0.)['passed'] is False
    assert acceptance_result(None, result, checks, turn_degrees=30.)['passed'] is False
    with pytest.raises(ValueError, match='disable turning'):
        acceptance_result(None, result, checks, turn_degrees=30., stop_after_grasp=True)
    for name in checks:
        failed = {**checks, name: {'passed': False}}
        assert acceptance_result(None, result, failed,
                                 turn_degrees=0., stop_after_grasp=True)['passed'] is False
    assert acceptance_result({'type': 'KeyboardInterrupt'}, result, checks,
                             turn_degrees=0., stop_after_grasp=True)['passed'] is False


@pytest.mark.parametrize('settled_after', [0, 2, None])
def test_task_baseline_is_required_before_contact_with_bounded_extra_wait(tmp_path, settled_after):
    session = SimpleNamespace(score={'settled_baseline_recorded': settled_after == 0}, result=None)
    script = Script(session, tmp_path)
    calls = []

    def hold(name, seconds):
        calls.append((name, seconds))
        session.score['settled_baseline_recorded'] = len(calls) == settled_after

    script.hold = hold
    # A broad engineering support gate is deliberately insufficient.
    checks = {'preinserted_reset_stability': {'passed': True}}
    if settled_after is None:
        with pytest.raises(RuntimeError, match='robot approach was not attempted'):
            script.ensure_task_settled(checks)
        assert len(calls) == 4
    else:
        script.ensure_task_settled(checks)
        assert len(calls) == settled_after
    assert all(seconds == .5 for _, seconds in calls)
    persisted = load_json(tmp_path / 'checks.json')['task_settlement_before_grasp']
    assert persisted['passed'] is (settled_after is not None)
    assert persisted['extra_wait_s'] == len(calls)*.5


@pytest.mark.parametrize('first_phase', ['high_approach', 'thumb_base_prepare', 'preclose', 'close'])
@pytest.mark.parametrize('settled_at_s', [None, .5, 2.])
def test_full_runner_checks_baseline_before_every_possible_first_motion(
        tmp_path, monkeypatch, first_phase, settled_at_s):
    report, calls = run_with_fake_worker(tmp_path, monkeypatch,
        first_phase=first_phase, settled_at_s=settled_at_s, turn_degrees=30.)
    motions = [c for c in calls if c['function'] in ('move_joints', 'set_hand_targets')]
    gate = report['checks']['task_settlement_before_grasp']
    assert report['checks']['preinserted_reset_stability']['passed'] is True
    if settled_at_s is None:
        assert not motions
        assert report['error']['detail'].endswith('robot approach was not attempted')
        assert gate['passed'] is False and gate['extra_wait_s'] == 2.
        assert {c['function'] for c in calls} <= {
            'get_robot', 'get_hand_presets', 'hold', 'get_action'}
    else:
        assert report['error'] is None
        assert len(motions) == 1
        assert motions[0]['baseline_recorded'] is True
        assert motions[0]['start_sim_time_s'] == settled_at_s
        assert gate['passed'] is True
        assert gate['extra_wait_s'] == settled_at_s-.5
    assert report['task_qualified'] is False  # This harness never runs physics.


@pytest.mark.parametrize('stop_after_grasp', [False, True])
def test_runner_keeps_zero_turn_diagnostic_scope_without_task_baseline(
        tmp_path, monkeypatch, stop_after_grasp):
    report, calls = run_with_fake_worker(tmp_path, monkeypatch,
        first_phase='preclose', settled_at_s=None, turn_degrees=0.,
        stop_after_grasp=stop_after_grasp)
    assert report['error'] is None
    assert 'task_settlement_before_grasp' not in report['checks']
    assert any(c['function'] == 'move_joints' and not c['baseline_recorded'] for c in calls)
    assert report['task_qualified'] is False


def run_with_fake_worker(tmp_path, monkeypatch, *, first_phase, settled_at_s,
                         turn_degrees, stop_after_grasp=False):
    """Exercise run/Script/dispatcher order, replacing only external dependencies.

    The fake worker ends at the first actual motion submission. Holding the reset
    drive is allowed; any approach or closure is visible in the dispatcher log.
    """
    import scripts.validation.validate_mechanical_functions as runner
    import scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.public_motion_plan as motion

    plan = {'phases': [{'name': first_phase, 'duration_s': .5,
                       'joint_positions_rad': {'arm': .1, 'hand': .2}}]}
    if first_phase != 'close':
        plan['phases'].append({'name': 'close', 'duration_s': .5,
                              'joint_positions_rad': {'arm': .1, 'hand': .3}})
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    args = SimpleNamespace(output=tmp_path/'result', motion_plan=path,
        motion_plan_sha256='a'*64, release=tmp_path/'release', task_id='fastener_assembly',
        device='cpu', grip_execution='feedback', turn_degrees=turn_degrees,
        stop_after_grasp=stop_after_grasp)
    calls = []

    class Reply:
        def __init__(self, value): self.value = value
        def model_dump(self, **kwargs): return self.value

    class Session:
        def __init__(self, request, directory, **kwargs):
            self.directory = directory
            (directory/'private').mkdir(parents=True)
            self.result = self.worker = self.thread = None
            self.sequence = 0
            self.raw = {'sim_time_s': 0.}
            self.score = {'settled_baseline_recorded': False}
            self.snapshot = SimpleNamespace(proprioception=SimpleNamespace(
                commanded_joint_positions_rad={'arm': 0., 'hand': 0.}))
            self.capabilities = SimpleNamespace(arm_joints=['arm'], hand_joints=['hand'])
            self.functions = SimpleNamespace(invoke=self.invoke, catalog=lambda: Reply({}))

        def invoke(self, payload):
            calls.append({**payload, 'baseline_recorded': self.score['settled_baseline_recorded'],
                          'start_sim_time_s': self.raw['sim_time_s']})
            function = payload['function']
            if function == 'hold':
                self.raw['sim_time_s'] += payload['arguments']['duration_s']
                self.score['settled_baseline_recorded'] = (
                    settled_at_s is not None and self.raw['sim_time_s'] >= settled_at_s)
            elif function in ('move_joints', 'set_hand_targets'):
                self.finish()
            if function == 'get_action':
                self.sequence += 1
                result = {'status': 'completed'}
            else:
                result = {'status': 'accepted'}
            return Reply({'ok': True, 'result': result})

        def finish(self):
            self.result = Reply({'physics_valid': False, 'outcome': 'cancelled'})

    policy = SimpleNamespace(limits=ExecutorLimits(),
                             descriptor=SimpleNamespace(policy_id='classical_fingertip', version='1'))
    monkeypatch.setattr(runner, 'verify_if_snapshot', lambda *a, **k: {})
    monkeypatch.setattr(runner, 'digest', lambda _: 'a'*64)
    monkeypatch.setattr(runner, 'task_inputs', lambda *a: ({'id': args.task_id}, {}, {}, {}))
    monkeypatch.setattr(runner, 'robot_configuration', lambda scene: {})
    monkeypatch.setattr(runner, 'configured_policy', lambda _: policy)
    monkeypatch.setattr(runner, 'PolicyRegistry', lambda: SimpleNamespace(register=lambda _: None))
    monkeypatch.setattr(runner, 'EpisodeSession', Session)
    monkeypatch.setattr(runner.Script, 'progress', lambda _: {'preinserted_reset_stability': {'passed': True}})
    monkeypatch.setattr(motion, 'public_move_arguments', lambda phase, *a, **kw: {
        'arm_joint_positions_rad': {'arm': phase['joint_positions_rad']['arm']},
        'hand_joint_positions_rad': {'hand': phase['joint_positions_rad']['hand']},
        'duration_s': phase['duration_s']})
    return runner.run(args), calls
