"""CPU receipt checks only; no native execution or task-success claim."""
import pytest

from scripts.validation.living_task_release import TASK_OBJECTS, private_coverage


def recording(task_id):
    dt = .0005
    task = {'id': task_id, 'evaluation': {'particle_count': 10, 'particle_ids_sha256': 'fixed-ids'}}
    metadata = {'physics_dt_s': dt, 'object_names': sorted(TASK_OBJECTS[task_id])}
    records = []
    for tick in range(16):
        start, end = max(0, tick-1)*.02, tick*.02
        count = 0 if tick == 0 else 40
        sampling = {'protocol': 'native_rigid_contact_sampling_v1', 'physics_dt_s': dt,
            'complete': True, 'sampled_substeps': count, 'start_sim_time_s': start,
            'end_sim_time_s': end, 'native_start_time_s': start+.001, 'native_end_time_s': end+.001,
            'native_start_step_index': round(start/dt)+2, 'native_end_step_index': round(end/dt)+2}
        state = {'errors': [], 'time_s': end}
        if task_id == 'remote_operation':
            state.update(complete=True, baseline=tick == 0, native_step=sampling['native_end_step_index'],
                         native_time_s=end+.001, buttons={'power': {}, 'confirm': {}})
        else:
            state.update(sampling={'start_step': sampling['native_start_step_index'],
                'end_step': sampling['native_end_step_index'], 'start_time_s': start+.001,
                'end_time_s': end+.001, 'sampled_substeps': count, 'physics_dt_s': dt},
                water={'particle_count': 10, 'ids_sha256': 'fixed-ids', 'counts': {'kettle': 10, 'cup': 0}},
                vessels={'kettle': {}, 'cup': {}})
        records.append({'kind': 'reset' if tick == 0 else 'step', 'score': {'success': False, 'reason': 'pending'},
            'observation': {'sim_time_s': end, 'time_s': end, 'task_state': state,
                'objects': dict.fromkeys(TASK_OBJECTS[task_id], {}),
                'contacts': {'errors': [], 'rigid': [], 'sampling': sampling}}})
    return metadata, records, task, .3


@pytest.mark.parametrize('identity', ['remote_operation', 'water_pouring'])
def test_complete_native_windows_and_non_success_are_bounded_smoke(identity):
    result = private_coverage(*recording(identity))
    assert result['passed'] and result['native_substeps'] == 600
    assert result['control_observations'] == 16
    assert result['task_success_claimed'] is False


@pytest.mark.parametrize('change', ['skipped', 'false_success', 'observer_error', 'stale_clock', 'missing_ids'])
def test_broken_water_evidence_never_becomes_a_pass(change):
    args = list(recording('water_pouring'))
    records = args[1]
    if change == 'skipped':
        records.pop(4)
    elif change == 'false_success':
        records[-1]['score']['success'] = True
    elif change == 'observer_error':
        records[-1]['observation']['task_state']['errors'] = ['stale']
    elif change == 'stale_clock':
        records[-1]['observation']['task_state']['sampling']['end_step'] -= 1
    else:
        records[-1]['observation']['task_state']['water']['ids_sha256'] = 'changed'
    with pytest.raises(AssertionError):
        private_coverage(*args)


def test_remote_incomplete_hook_is_rejected_even_with_valid_contact_receipts():
    args = list(recording('remote_operation'))
    args[1][-1]['observation']['task_state']['complete'] = False
    with pytest.raises(AssertionError):
        private_coverage(*args)


@pytest.mark.parametrize('change', [None, 'power_on', 'wrong_score', 'false_progress'])
def test_television_reset_and_passive_hold_remain_off_without_credit(change):
    args = list(recording('remote_operation'))
    args[2]['evaluation']['schema'] = 'living_remote_evaluation.v3'
    for row in args[1]:
        television = {'power_on': False, 'continuous_on_s': 0., 'required_on_s': 10., 'phase': 'await_power_on'}
        row['observation']['task_state']['television'] = dict(television)
        row['score'].update(television=dict(television), referee={'phase': 'await_power_on'}, progress=0.)
    if change == 'power_on':
        args[1][-1]['observation']['task_state']['television']['power_on'] = True
    elif change == 'wrong_score':
        args[1][-1]['score']['television']['phase'] = 'holding_power_on'
    elif change == 'false_progress':
        args[1][-1]['score']['progress'] = .1
    if change:
        with pytest.raises(AssertionError, match='television state'):
            private_coverage(*args)
    else:
        assert private_coverage(*args)['passed']
