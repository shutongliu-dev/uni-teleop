"""Fail-closed CPU checks for offline native evidence; these do not run physics."""
from copy import deepcopy

import pytest

from scripts.validation.card_room_task0 import PIECES, support_check


def recording():
    dt = .0005
    declarations = {name: {'fixture_id': 'reserve_black_west', 'surface_z_m': .8} for name in PIECES}
    info = {'objects': {name: {'path': '/World/Objects/'+name} for name in PIECES},
            'fixtures': {'reserve_black_west': {'contact_filter_paths': ['/World/Fixtures/reserve_black_west']}},
            'robot_link_paths': {'right_hand_flange': '/World/Robot/right_hand_flange'}}
    metadata = {'object_names': sorted(PIECES), 'physics_dt_s': dt, 'scene_info': info}
    objects = {name: {'position_m': [.14, -.38-i*.14, .81],
                     'quaternion_wxyz': [1., 0., 0., 0.],
                     'linear_velocity_m_s': [0., 0., 0.],
                     'angular_velocity_rad_s': [0., 0., 0.]} for i, name in enumerate(sorted(PIECES))}
    records = [{'kind': 'reset', 'observation': {'objects': objects}}]
    for tick in range(25):
        contacts = [{'time_s': step*dt, 'actor0': '/World/Objects/'+name,
                     'actor1': '/World/Fixtures/reserve_black_west',
                     'impulse_magnitude_sum_ns': .0001, 'min_separation_m': -.00001}
                    for step in range(tick*40+1, (tick+1)*40+1) for name in PIECES]
        sampling = {'protocol': 'native_rigid_contact_sampling_v1', 'physics_dt_s': dt,
                    'start_sim_time_s': tick*.02, 'end_sim_time_s': (tick+1)*.02,
                    'native_start_step_index': tick*40+1, 'native_end_step_index': (tick+1)*40+1,
                    'sampled_substeps': 40, 'complete': True}
        records.append({'kind': 'step', 'observation': {'sim_time_s': (tick+1)*.02,
                        'objects': deepcopy(objects), 'contacts': {'rigid': contacts, 'errors': [],
                                                                  'sampling': sampling}}})
    return metadata, records, declarations


def test_complete_support_contacts_remain_distinct_from_every_substep_pose():
    result = support_check(*recording())
    assert result['passed']
    assert result['native_contact_samples'] == 1000
    assert result['terminal_support_substeps'] == 601
    assert result['support_observed_span_s'] == .3
    assert result['every_substep_pose_verified'] is False
    assert result['pose_sample_period_s'] == .02


@pytest.mark.parametrize('change', ['missing_contact', 'wrong_fixture', 'robot_contact', 'terminal_motion', 'lower_support'])
def test_missing_or_conflicting_physics_never_qualifies(change):
    metadata, records, declarations = recording()
    observation = records[-1]['observation']
    if change == 'missing_contact':
        observation['contacts']['rigid'].pop()
    elif change == 'wrong_fixture':
        observation['contacts']['rigid'][-1]['actor1'] = '/World/Fixtures/other_tray'
    elif change == 'robot_contact':
        row = deepcopy(observation['contacts']['rigid'][-1])
        row['actor1'] = '/World/Robot/right_hand_flange'
        observation['contacts']['rigid'].append(row)
    elif change == 'terminal_motion':
        observation['objects']['black_00']['linear_velocity_m_s'] = [.1, 0., 0.]
    else:
        observation['objects']['black_00']['position_m'][2] -= .05
    assert support_check(metadata, records, declarations)['passed'] is False


@pytest.mark.parametrize('change', ['missing_sampling', 'duplicate_interval', 'missing_separation', 'negative_impulse'])
def test_broken_native_sampling_evidence_is_rejected(change):
    metadata, records, declarations = recording()
    contacts = records[-1]['observation']['contacts']
    if change == 'missing_sampling':
        contacts['sampling']['sampled_substeps'] = 39
    elif change == 'duplicate_interval':
        records[-1] = deepcopy(records[-2])
    elif change == 'missing_separation':
        contacts['rigid'][-1]['min_separation_m'] = None
    else:
        contacts['rigid'][-1]['impulse_magnitude_sum_ns'] = -.1
    with pytest.raises(AssertionError):
        support_check(metadata, records, declarations)
