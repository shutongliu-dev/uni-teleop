"""CPU native-frame fixtures test scoring logic, not robot qualification."""
from copy import deepcopy
import json
from types import MappingProxyType

import pytest

from bench.protocol.task_referee import PublicContract, project_referee_state
from scenes.workrooms.living_room.tasks.remote_operation.evaluator import NativeTaskObserver, SequenceEvaluator
from scenes.workrooms.living_room.tasks.remote_operation.rules import (
    LEGACY_RULES_VERSION, RULES_VERSION, evaluation_configuration, public_rules)


def task(seed=0, *, version=LEGACY_RULES_VERSION, physics_dt_s=.0005):
    rules = public_rules(seed, version=version)
    goal = ({'button_sequence': rules['button_sequence']} if version == LEGACY_RULES_VERSION else
            {'operation': 'cycle_television_power', 'object_id': 'remote',
             'controlled_object_id': 'television', 'required_on_s': 10.,
             'power_button': 'power', 'initial_power_on': False,
             'target_id': 'holder', 'holder_return_required': True, 'stable_duration_s': .300})
    return {'schema_version': 'living-room-manipulation-task-v1', 'id': 'remote_operation',
        'episode_seed': seed,
        'evaluation': evaluation_configuration(seed=seed, physics_dt_s=physics_dt_s, rules_version=version),
        'public_rules': rules, 'public_goal': goal}


class Replay:
    def __init__(self, seed=0, *, version=LEGACY_RULES_VERSION, physics_dt_s=.0005):
        self.task = task(seed, version=version, physics_dt_s=physics_dt_s)
        self.dt = physics_dt_s
        self.control_steps = round(.02/self.dt)
        self.observer = NativeTaskObserver(self.task)
        self.evaluator = SequenceEvaluator(self.task)
        self.step = 0
        self.frame = None
        self.result = None
        self.send(baseline=True)
        self.observe()

    def send(self, *, baseline=False, pressed=(), held=True, table=False, seated=False,
             support=True, yaw_wrong=False, velocity=0., no_finger=False, extra_support=False,
             touching=(), remote_position=None):
        if not baseline:
            self.step += 1
        time = self.step*self.dt
        remote_p = [.82, .075, .7974] if seated else [.49, -.12, .9]
        q = [0., 1., 0., 0.] if baseline else ([0., 0., 0., 1.] if yaw_wrong else [1., 0., 0., 0.])
        if baseline:
            remote_p = [.49, -.12, .79365]
        if remote_position is not None:
            remote_p = list(remote_position)
        def body(p, orient):
            return {'position_m': p, 'quaternion_wxyz': orient,
                'linear_velocity_m_s': [velocity, 0., 0.], 'angular_velocity_rad_s': [0., 0., 0.]}
        objects = {'remote': body(remote_p, q), 'holder': body([.82, .075, .78], [1., 0., 0., 0.])}
        joints = {}
        for n, y in (('power', .0215), ('confirm', .0565)):
            displacement = -.0016 if n in pressed else 0.
            objects[n] = body([remote_p[0], remote_p[1]+y, remote_p[2]+.01+displacement], q)
            joints[n] = {'position_m': displacement, 'body0': 'remote', 'body1': n}
        a = self.task['evaluation']['actors']
        contacts = []
        def contact(first, second):
            contacts.append({'time_s': time, 'actor0': first, 'actor1': second,
                'point_count': 1, 'impulse_magnitude_sum_ns': .001, 'min_separation_m': -.00005})
        if not baseline:
            if held:
                contact(a['remote'], '/World/Robot/hand/right_thumb_DP')
            for n in set(pressed) | set(touching):
                if not no_finger:
                    contact(a[n], '/World/Robot/hand/right_index_DP')
            if table:
                contact(a['remote'], '/World/Geometry/worktop/Collider')
            if seated and support:
                contact(a['remote'], a['holder'])
            if extra_support:
                contact(a['remote'], '/World/Geometry/wall/Collider')
        self.frame = {'baseline': baseline, 'time_s': time, 'native_time_s': time+.001,
            'native_step': self.step+2, 'physics_dt_s': self.dt,
            'objects': objects, 'joints': joints, 'particle_sets': {}, 'contacts': contacts}
        self.observer.on_frame(MappingProxyType(self.frame))

    def observe(self, **kwargs):
        state = self.observer.on_observe()
        self.result = self.evaluator.update({'time_s': self.step*self.dt, 'task_state': state, **kwargs})
        return self.result

    def run(self, count=40, **kwargs):
        for _ in range(count):
            self.send(**kwargs)
            if self.step % self.control_steps == 0:
                self.observe()
        return self.result

    def cycle(self, button, **kwargs):
        self.run(40)
        self.run(40, pressed=(button,), **kwargs)
        self.run(40)

    def sequence(self):
        for n in self.task['evaluation']['button_sequence']:
            self.cycle(n)

    def seat(self, count=680, **kwargs):
        return self.run(count, held=False, seated=True, **kwargs)


def test_ordered_handheld_release_then_300ms_holder_support_passes():
    r = Replay()
    assert r.result['reason'] == 'reset_baseline_not_scored'
    r.sequence()
    assert r.result['completed_button_cycles'] == 2
    assert r.result['success'] is False
    r.seat(600)
    assert r.result['success'] is False
    r.seat(40)
    assert r.result['success'] is True
    assert len(r.observer.on_observe()['events']) == 4
    json.dumps(r.observer.on_observe(), allow_nan=False)


def test_seed_reverses_published_and_evaluated_sequence():
    r = Replay(1)
    assert r.task['public_goal']['button_sequence'] == ['confirm', 'power']
    r.sequence()
    assert r.seat()['success'] is True


@pytest.mark.parametrize('kwargs', [{'table': True}, {'no_finger': True}, {'held': False}])
def test_no_credit_without_handheld_finger_contact(kwargs):
    r = Replay()
    r.cycle('power', **kwargs)
    assert r.result['completed_button_cycles'] == 0
    assert not r.seat()['success']


def test_held_button_counts_once_and_requires_release():
    r = Replay()
    r.run(40)
    r.run(800, pressed=('power',))
    assert r.result['completed_button_cycles'] == 0
    assert len(r.observer.on_observe()['events']) == 1
    r.run(40)
    assert r.result['completed_button_cycles'] == 1


def test_simultaneous_buttons_are_ambiguous_and_recoverable():
    r = Replay()
    r.run(40)
    r.run(40, pressed=('power', 'confirm'))
    r.run(40)
    assert r.result['completed_button_cycles'] == 0
    r.sequence()
    assert r.seat()['success']


def test_wrong_button_resets_progress_then_real_retry_succeeds():
    r = Replay()
    r.cycle('confirm')
    assert r.result['completed_button_cycles'] == 0
    r.sequence()
    assert r.seat()['success']


@pytest.mark.parametrize('kwargs', [{'support': False}, {'yaw_wrong': True}, {'velocity': .03}, {'extra_support': True}])
def test_holder_requires_orientation_contact_rest_and_no_external_support(kwargs):
    r = Replay()
    r.sequence()
    assert not r.seat(**kwargs)['success']


def test_prior_contact_continuity_explicitly_handles_static_contact_sleep():
    r = Replay()
    r.sequence()
    r.seat(80)
    assert r.seat(600, support=False)['success']
    assert r.observer.on_observe()['holder']['support_source'] == 'prior_native_contact_and_unchanged_unforced_pose'


def test_missing_native_step_and_stale_contact_fail_closed():
    r = Replay()
    r.step += 1
    r.send()
    assert r.observer.on_observe()['complete'] is False
    assert not r.observe()['success']
    r = Replay()
    r.send()
    frame = deepcopy(r.frame)
    frame.update(time_s=.001, native_time_s=.002, native_step=4)
    r.observer.on_frame(frame)
    assert r.observer.on_observe()['complete'] is False


def test_private_state_and_frozen_configuration_required():
    r = Replay()
    assert 'invalid_observation' in r.evaluator.update({'time_s': .02})['reason']
    bad = task()
    bad['evaluation']['stable_s'] = .01
    with pytest.raises(ValueError, match='threshold'):
        SequenceEvaluator(bad)
    bad = task()
    bad['public_rules']['button_sequence'].reverse()
    with pytest.raises(ValueError, match='Public rules'):
        NativeTaskObserver(bad)


def test_event_history_cannot_be_rewritten_and_budget_precedes_success():
    r = Replay()
    r.sequence()
    r.seat(600)
    for _ in range(40):
        r.send(held=False, seated=True)
    assert r.observe(budget_exhausted=True)['reason'] == 'incomplete_budget'
    r = Replay()
    r.cycle('power')
    for _ in range(40):
        r.send()
    s = r.observer.on_observe()
    s['events'][0]['handheld_valid'] = False
    assert 'invalid_observation' in r.evaluator.update({'time_s': r.step*.0005, 'task_state': s})['reason']


def test_supported_scene_without_any_button_cycles_never_succeeds():
    r = Replay()
    assert not r.seat(1000)['success']


def test_short_press_and_release_between_20ms_observations_is_preserved():
    r = Replay()
    r.run(40)
    r.run(5, pressed=('power',))
    r.run(35)
    assert r.result['completed_button_cycles'] == 1
    assert [e['kind'] for e in r.observer.on_observe()['events']] == ['pressed', 'released']


def test_unsupported_handheld_cycle_loss_and_broken_postsuccess_rest_revoke_credit():
    r = Replay()
    r.run(40)
    r.run(20, pressed=('power',))
    r.run(20, pressed=('power',), table=True)
    r.run(40)
    assert r.result['completed_button_cycles'] == 0
    r.sequence()
    assert r.seat()['success']
    assert not r.run(40, held=True, seated=True)['success']


def test_readonly_input_remains_unmodified_and_tuple_contacts_supported():
    r = Replay()
    r.send()
    f = deepcopy(r.frame)
    f.update(time_s=.001, native_time_s=.002, native_step=4)
    f['contacts'] = tuple({**row, 'time_s': .001} for row in f['contacts'])
    expected = deepcopy(f)
    r.observer.on_frame(MappingProxyType(f))
    assert f == expected
    assert r.observer.on_observe()['complete']


@pytest.mark.parametrize('dt', [0., -.001, .003, float('nan'), True, .0007])
def test_invalid_configuration_time_steps_rejected(dt):
    with pytest.raises(ValueError):
        evaluation_configuration(physics_dt_s=dt)


def television_replay(seed=0):
    # Both native rates are supported; 2 ms keeps the ten-second CPU fixtures small.
    return Replay(seed, version=RULES_VERSION, physics_dt_s=.002)


def test_television_rules_are_seed_independent_and_preserve_explicit_v2():
    assert public_rules(0) == public_rules(1)
    assert 'button_sequence' not in public_rules()
    assert public_rules()['television']['required_on_s'] == 10.
    assert public_rules()['success']['holder_return_required'] is True
    assert public_rules()['success']['stable_duration_s'] == .300
    assert public_rules()['referee']['thresholds']['stable_s'] == .300
    assert 'await_return' in public_rules()['referee']['phases']
    assert evaluation_configuration()['schema'] == 'living_remote_evaluation.v3'
    assert 'button_sequence' not in evaluation_configuration()
    old = task()
    assert old['public_rules']['schema'] == LEGACY_RULES_VERSION
    assert old['evaluation']['schema'] == 'living_remote_evaluation.v2'
    assert old['evaluation']['button_sequence'] == ['power', 'confirm']
    with pytest.raises(ValueError, match='Unknown'):
        public_rules(version='unknown')


def test_television_initially_off_and_confirm_never_changes_goal_progress():
    r = television_replay(1)
    assert r.result['television'] == {
        'power_on': False, 'continuous_on_s': 0., 'required_on_s': 10., 'phase': 'await_power_on'}
    r.cycle('confirm')
    assert r.result['completed_button_cycles'] == 0
    assert r.result['progress'] == 0.
    assert not r.result['success']
    r.cycle('power')
    r.cycle('confirm')
    assert r.result['television']['power_on']
    assert r.result['completed_button_cycles'] == 1
    assert r.result['television'] == r.observer.on_observe()['television']
    public = PublicContract.model_validate(r.task['public_rules']['referee'])
    projected = project_referee_state(r.result, public.model_dump(mode='json', by_alias=True))
    assert projected['phase'] == 'holding_power_on'
    assert projected['state'] == 'ongoing'
    json.dumps(r.result, allow_nan=False)


def test_television_toggle_requires_complete_physical_release_and_counts_long_hold_once():
    r = television_replay()
    r.run(40)
    r.run(100, pressed=('power',))
    assert not r.result['television']['power_on']
    assert r.result['completed_button_cycles'] == 0
    assert len(r.observer.on_observe()['events']) == 1
    r.run(20, touching=('power',))
    assert not r.result['television']['power_on']  # Joint rebound while still touched is insufficient.
    r.run(20)
    assert r.result['television']['power_on']
    assert r.result['completed_button_cycles'] == 1
    r.run(200)
    assert r.result['completed_button_cycles'] == 1


@pytest.mark.parametrize('kwargs', [{'table': True}, {'no_finger': True}, {'held': False}])
def test_television_rejects_nonhandheld_or_nonphysical_power_cycles(kwargs):
    r = television_replay()
    r.cycle('power', **kwargs)
    assert not r.result['television']['power_on']
    assert r.result['completed_button_cycles'] == 0


def test_television_support_during_a_press_invalidates_that_whole_cycle_and_allows_retry():
    r = television_replay()
    r.run(40)
    r.run(10, pressed=('power',))
    r.run(10, pressed=('power',), table=True)
    r.run(20, pressed=('power',))
    r.run(40)
    assert not r.result['television']['power_on']
    assert r.result['completed_button_cycles'] == 0
    r.cycle('power')
    assert r.result['television']['power_on']


def test_television_confirm_during_power_does_not_invalidate_independent_power_button():
    r = television_replay()
    r.run(40)
    r.run(40, pressed=('power', 'confirm'))
    r.run(40)
    assert r.result['television']['power_on']
    assert r.result['completed_button_cycles'] == 1


def test_television_requires_manual_off_then_released_stable_holder_return():
    r = television_replay()
    r.cycle('power')
    r.run(5000)
    assert r.result['television']['phase'] == 'await_power_off'
    assert r.result['television']['power_on']
    assert r.result['television']['continuous_on_s'] >= 10.
    assert not r.result['success']
    r.cycle('power')
    assert not r.result['success']
    assert r.result['television'] == {
        'power_on': False, 'continuous_on_s': 0., 'required_on_s': 10., 'phase': 'await_return'}
    assert not r.observer.on_observe()['holder']['supported']
    assert r.result['progress'] == .75
    assert project_referee_state(r.result, r.task['public_rules']['referee'])['phase'] == 'await_return'
    r.seat(150)
    assert r.observer.on_observe()['released_stable_s'] == pytest.approx(.298)
    assert not r.result['success']
    r.send(held=False, seated=True)
    assert r.observer.on_observe()['released_stable_s'] == pytest.approx(.300)
    assert r.observer.on_observe()['television']['phase'] == 'completed'
    r.seat(9)
    assert r.result['success']
    assert r.result['television']['phase'] == 'completed'
    assert r.result['stable_s'] >= .300
    assert project_referee_state(r.result, r.task['public_rules']['referee'])['state'] == 'success'
    # A renewed hand contact breaks released stability, while the completed cycle remains.
    r.run(10, held=True, seated=True)
    assert not r.result['success']
    assert r.result['television']['phase'] == 'await_return'
    assert r.seat(160)['success']
    # A new legal cycle changes the same television state used by the scorer.
    r.cycle('power')
    assert not r.result['success']
    assert r.result['television']['phase'] == 'holding_power_on'


def test_television_short_periods_never_accumulate_and_retry_starts_a_fresh_interval():
    r = television_replay()
    for _ in range(3):
        r.cycle('power')
        r.run(2000)
        r.cycle('power')
        assert not r.result['success']
        assert r.result['television']['phase'] == 'await_power_on'
        assert r.result['television']['continuous_on_s'] == 0.
    assert r.step*r.dt > 12.
    r.cycle('power')
    assert r.result['television']['continuous_on_s'] < .1
    r.run(5000)
    r.cycle('power')
    assert not r.result['success']
    assert r.result['television']['phase'] == 'await_return'
    assert r.seat(160)['success']


@pytest.mark.parametrize(('duration', 'success'), [(9.998, False), (10., True)])
def test_television_ten_second_boundary_uses_native_release_times(duration, success):
    r = television_replay()
    r.run(40)
    r.run(1, pressed=('power',))
    r.run(9)
    assert r.result['television']['power_on']  # Entire first cycle fits between control observations.
    on_event = r.observer.on_observe()['events'][-1]
    off_step = round((on_event['time_s']+duration)/r.dt)
    r.run(off_step-r.step-2)
    r.run(1, pressed=('power',))
    r.run(1)
    r.run((-r.step) % r.control_steps)
    assert not r.result['success']
    assert r.result['television']['phase'] == ('await_return' if success else 'await_power_on')
    off_event = r.observer.on_observe()['events'][-1]
    assert off_event['time_s']-on_event['time_s'] == pytest.approx(duration)
    assert r.seat(160)['success'] is success


def test_television_second_press_may_span_the_ten_second_boundary_without_retrigger():
    r = television_replay()
    r.cycle('power')
    r.run(5000, pressed=('power',))
    assert r.result['television']['phase'] == 'await_power_off'
    assert not r.result['success']
    assert r.result['completed_button_cycles'] == 1
    r.run(10)
    assert not r.result['success']
    assert r.result['television']['phase'] == 'await_return'
    assert r.result['completed_button_cycles'] == 2
    assert r.seat(160)['success']


@pytest.mark.parametrize('kwargs', [
    {'support': False}, {'yaw_wrong': True}, {'velocity': .03}, {'extra_support': True},
    {'held': True}, {'touching': ('confirm',)}, {'pressed': ('confirm',)},
    {'remote_position': [.86, .075, .7974]}])
def test_television_return_requires_whole_remote_orientation_support_release_and_rest(kwargs):
    r = television_replay()
    r.cycle('power')
    r.run(5000)
    r.cycle('power')
    r.run(160, **{'held': False, 'seated': True, **kwargs})
    assert r.result['television']['phase'] == 'await_return'
    assert not r.result['success']
    assert r.observer.on_observe()['released_stable_s'] == 0.
    assert r.seat(160)['success']


def test_television_return_before_completed_power_cycle_never_succeeds():
    r = television_replay()
    assert not r.seat(160)['success']
    assert r.result['television']['phase'] == 'await_power_on'
    r.cycle('power')
    r.seat(5200)
    assert r.result['television']['phase'] == 'await_power_off'
    assert r.observer.on_observe()['released_stable_s'] == 0.
    assert not r.result['success']
    r.cycle('power')
    assert r.result['television']['phase'] == 'await_return'
    assert r.observer.on_observe()['released_stable_s'] == 0.
    assert not r.result['success']
    assert r.seat(160)['success']


def test_television_return_keeps_contact_sleep_evidence_and_requires_full_dwell_after_regrasp():
    r = television_replay()
    r.cycle('power')
    r.run(5000)
    r.cycle('power')
    r.seat(80)
    assert r.seat(80, support=False)['success']
    assert r.observer.on_observe()['holder']['support_source'] == 'prior_native_contact_and_unchanged_unforced_pose'
    r.run(10, held=True, seated=True)
    assert not r.result['success']
    assert not r.seat(140)['success']
    assert r.seat(10)['success']


@pytest.mark.parametrize('position', [[1.2501, -.12, .9], [.49, .7001, .9], [.49, -.12, .7199]])
@pytest.mark.parametrize('version', [RULES_VERSION, LEGACY_RULES_VERSION])
def test_native_workspace_excursion_between_observations_latches_v3_failure_only(position, version):
    r = Replay(version=version)
    r.run(40)
    r.send(remote_position=position)
    assert r.observer.on_observe()['terminal_failure'] == 'remote_outside_workspace_or_below_table'
    r.run(39)
    failure = 'remote_outside_workspace_or_below_table' if version == RULES_VERSION else None
    assert r.observer.on_observe()['terminal_failure'] == failure
    assert r.result['irrecoverable_reason'] == failure
    r.run(40)
    assert r.observer.on_observe()['terminal_failure'] == failure
    assert r.result['irrecoverable_reason'] == failure
    assert not r.result['success']
    if failure:
        assert project_referee_state(r.result, r.task['public_rules']['referee'])['state'] == 'task_failure'
    r.observer.reset()
    r.evaluator.reset()
    r.step = 0
    r.send(baseline=True)
    assert r.observe()['irrecoverable_reason'] is None
    assert r.observer.on_observe()['terminal_failure'] is None


@pytest.mark.parametrize('tamper', ['television', 'event_time', 'event_position', 'event_cycle'])
def test_television_self_report_or_malformed_physical_events_fail_closed(tamper):
    r = television_replay()
    r.run(40)
    r.run(1, pressed=('power',))
    for _ in range(9):
        r.send()
    s = r.observer.on_observe()
    if tamper == 'television':
        s['television'].update(power_on=False, continuous_on_s=0., phase='completed')
    elif tamper == 'event_time':
        s['events'][0]['time_s'] += .001  # Within 20 ms, but between native steps.
    elif tamper == 'event_position':
        s['events'][0]['position_m'] = 0.
    else:
        s['events'][1].update(kind='pressed', position_m=-.0016)
    result = r.evaluator.update({'time_s': r.step*r.dt, 'task_state': s})
    assert not result['success']
    assert result['reason'].startswith('invalid_observation:')
    assert project_referee_state(result, r.task['public_rules']['referee'])['state'] == 'evaluator_invalid'


def test_television_budget_and_faults_take_priority_over_completed_power_goal():
    r = television_replay()
    r.cycle('power')
    r.run(5000)
    r.cycle('power')
    r.seat(150)
    for _ in range(10):
        r.send(held=False, seated=True)
    assert r.observer.on_observe()['television']['phase'] == 'completed'
    assert r.observe(budget_exhausted=True)['reason'] == 'budget_exhausted'
    assert r.result['status'] == 'incomplete' and r.result['terminal'] is True
    assert not r.result['success']
    assert project_referee_state(r.result, r.task['public_rules']['referee'])['phase'] == 'budget_exhausted'
    # Even a completed state cannot override an observer evidence fault.
    r.step += 1
    r.send()
    assert r.observe()['reason'].startswith('invalid_observation:')


def test_television_native_clock_fault_and_reset_are_not_wall_time_progress():
    r = television_replay()
    r.cycle('power')
    r.send()
    bad = deepcopy(r.frame)
    bad.update(time_s=bad['time_s']+r.dt, native_step=bad['native_step']+1)
    r.observer.on_frame(bad)  # Native clock did not advance with the episode clock.
    assert not r.observer.on_observe()['complete']
    assert not r.observe()['success']
    r.observer.reset()
    r.evaluator.reset()
    r.step = 0
    r.send(baseline=True)
    assert r.observe()['television']['phase'] == 'await_power_on'
    assert r.result['completed_button_cycles'] == 0


@pytest.mark.parametrize(('field', 'value'), [
    ('required_on_s', 1.), ('power_button', 'confirm'), ('initial_power_on', True),
    ('operation', 'press_button_sequence'), ('button_sequence', ['power', 'confirm']),
    ('target_id', 'table'), ('holder_return_required', False), ('stable_duration_s', .1)])
def test_television_public_goal_cannot_contradict_frozen_rules(field, value):
    bad = task(version=RULES_VERSION)
    bad['public_goal'][field] = value
    with pytest.raises(ValueError, match='goal'):
        SequenceEvaluator(bad)


def test_television_holder_time_before_qualified_off_never_counts():
    r = television_replay()
    r.seat(200)
    assert r.observer.on_observe()['released_stable_s'] == 0.
    assert r.result['television']['phase'] == 'await_power_on'
    r.cycle('power')
    r.seat(5200)
    assert r.result['television']['phase'] == 'await_power_off'
    assert r.observer.on_observe()['released_stable_s'] == 0.
    r.cycle('power')
    assert r.result['television']['phase'] == 'await_return'
    assert r.result['stable_s'] == 0.
    r.seat(150)
    assert not r.result['success']
    assert r.seat(10)['success']


def test_television_reopening_clears_partial_return_and_requires_new_full_interval():
    r = television_replay()
    r.cycle('power')
    r.run(5000)
    r.cycle('power')
    r.seat(100)
    assert 0. < r.result['stable_s'] < .3
    r.cycle('power')
    assert r.result['stable_s'] == 0.
    assert r.result['television']['phase'] == 'holding_power_on'
    assert not r.seat(200)['success']
    r.run(5000)
    r.cycle('power')
    assert r.result['television']['phase'] == 'await_return'
    assert not r.result['success']
    assert r.seat(200)['success']
    assert not r.run(10, held=True, seated=True)['success']
    assert r.result['television']['phase'] == 'await_return'


def test_television_rejects_return_duration_that_predates_qualified_off():
    r = television_replay()
    r.cycle('power')
    r.run(5000)
    r.cycle('power')
    for _ in range(10):
        r.send(held=False, seated=True)
    s = r.observer.on_observe()
    s['released_stable_s'] = .3
    s['television']['phase'] = 'completed'
    result = r.evaluator.update({'time_s': r.step*r.dt, 'task_state': s})
    assert not result['success']
    assert result['reason'].startswith('invalid_observation:')
    assert 'must follow qualified' in result['reason']
