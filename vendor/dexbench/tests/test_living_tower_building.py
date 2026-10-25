"""Synthetic normalized observations, never native tower acceptance evidence."""
from copy import deepcopy
import math

import pytest

from scenes.workrooms.living_room.tasks.tower_building.evaluator import TowerBuildingEvaluator, evaluate_tower
from scenes.workrooms.living_room.tasks.tower_building.rules import OBSERVATION_SCHEMA, rules_for_geometry
from scenes.workrooms.living_room.tasks.tower_building.task import _bounds, task_configuration

# Test-only dimensions. No asset, release or user size selection is implied.
TEST_GEOMETRY = {'block_size_m': [.11, .035, .020]}


def plan(**kwargs):
    return task_configuration(TEST_GEOMETRY, **kwargs)


def tower_states(config, *, shift=0., yaw_offset=0.):
    return {name: {'position_m': [slot['position_m'][0]+shift, *slot['position_m'][1:]],
        'quaternion_wxyz': [math.cos((slot['yaw_rad']+yaw_offset)/2), 0., 0., math.sin((slot['yaw_rad']+yaw_offset)/2)]}
        for name, slot in zip(config['block_ids'], config['target_slots'])}


def contacts(config, states):
    ids, slots = config['block_ids'], config['target_slots']
    rows = []
    for i, name in enumerate(ids):
        layer = slots[i]['layer']
        lower = 'table' if layer == 0 else ids[(layer-1)*3]
        lo, hi = _bounds(states[name], config['geometry']['block_size_m'])
        if lower == 'table':
            point = [*states[name]['position_m'][:2], config['geometry']['table_top_z_m']]
        else:
            a, b = _bounds(states[lower], config['geometry']['block_size_m'])
            point = [(max(lo[j], a[j])+min(hi[j], b[j]))/2 for j in (0, 1)]+[(lo[2]+b[2])/2]
        rows.append({'a': name, 'b': lower, 'point_m': point, 'separation_m': 0.,
                     'support_for': name, 'minimum_upward_force_n': .1, 'continuous': True})
    return rows


def observation(config, tick, *, states=None, contact_rows=None, vmax=0., wmax=0., **extra):
    state = deepcopy(config['initial_objects'] if tick == 0 else tower_states(config)) if states is None else deepcopy(states)
    for row in state.values():
        row.update(linear_velocity_m_s=[0., 0., 0.], angular_velocity_rad_s=[0., 0., 0.],
                   interval_max_linear_speed_m_s=vmax, interval_max_angular_speed_rad_s=wmax)
    dt = config['rules']['tolerances']['sample_period_s']
    return {'schema': OBSERVATION_SCHEMA, 'configuration_sha256': config['configuration_sha256'],
        'episode_id': 'synthetic_fixture_only', 'sample_index': tick, 'time_s': tick*dt,
        'interval_start_s': max(0, tick-1)*dt, 'sampling_complete': True,
        'contacts_complete': True, 'physics_valid': True, 'robot_only_actions': True,
        'object_pose_writes': False, 'budget_exhausted': False,
        'robot_action_count': 0 if tick == 0 else 1, 'objects': state,
        'contacts': ([] if tick == 0 else contacts(config, state)) if contact_rows is None else contact_rows,
        **extra}


def started(config=None):
    config = config or plan()
    evaluator = TowerBuildingEvaluator(config, trusted_normalized_input=True)
    assert evaluator.update(observation(config, 0))['reason'] == 'reset_baseline_not_scored'
    assert evaluator.update(observation(config, 1, vmax=100., wmax=100.))['status'] == 'incomplete'
    return config, evaluator


def settle(config, evaluator, start=2, end=53, **kwargs):
    for tick in range(start, end):
        result = evaluator.update(observation(config, tick, **kwargs))
    return result


def test_explicit_dimensions_only_and_planned_front_arm():
    with pytest.raises(ValueError, match='required'):
        task_configuration({})
    with pytest.raises(ValueError, match='in_layer_gap_m'):
        task_configuration({**TEST_GEOMETRY, 'layer_gap_m': .001})
    config = plan()
    assert config['status'] == 'planned' and config['implementation'] is None
    assert config['evaluator']['native_adapter'] is None
    assert config['native_execution_available'] is False and config['source_release'] is None
    assert config['environment']['version'] == 'living_room_residential_20261018_v7'
    assert len(config['environment']['usd_sha256']) == 64
    assert config['robot_layout']['selected_sides'] == ['front']
    assert config['robot_layout']['base_position_m'] == [.65, -.76, .78]
    assert len(config['initial_objects']) == 18


def test_supply_seed_reproducible_and_no_solved_reset():
    first, same, different = plan(seed=3), plan(seed=3), plan(seed=4)
    assert first == same
    assert first['initial_objects'] != different['initial_objects']
    with pytest.raises(ValueError, match='outside every target'):
        plan(supply_layout={name: {'position_m': [*slot['position_m'][:2], .79],
             'quaternion_wxyz': [math.cos(slot['yaw_rad']/2), 0, 0, math.sin(slot['yaw_rad']/2)]}
             for name, slot in zip(first['block_ids'], first['target_slots'])})


def test_supply_too_large_or_overlapping_rejected():
    with pytest.raises(ValueError, match='do not fit'):
        task_configuration({'block_size_m': [.8, .25, .1]})
    config = plan()
    bad = deepcopy(config['initial_objects'])
    bad['block_01'] = deepcopy(bad['block_00'])
    with pytest.raises(ValueError, match='separated'):
        plan(supply_layout=bad)


def test_trust_cannot_be_asserted_in_observation_json():
    config = plan()
    result = TowerBuildingEvaluator(config).update(observation(config, 0, trusted=True))
    assert result['status'] == 'invalid' and not result['success']
    assert 'Caller' in result['reason']


def test_goal_or_rules_tampering_and_short_dwell_rejected():
    config = plan()
    config['target_slots'][0]['position_m'][2] += .1
    with pytest.raises(ValueError, match='differs'):
        TowerBuildingEvaluator(config, trusted_normalized_input=True)
    with pytest.raises(ValueError, match='at least one second'):
        plan(stable_duration_s=.9)
    with pytest.raises(ValueError, match='distinct slots'):
        rules_for_geometry(TEST_GEOMETRY, tolerances={'position_xy_m': .1})


def test_preassembled_initial_state_cannot_receive_credit():
    config = plan()
    result = TowerBuildingEvaluator(config, trusted_normalized_input=True).update(
        observation(config, 0, states=tower_states(config)))
    assert result['status'] == 'invalid' and 'Reset' in result['reason']


@pytest.mark.parametrize('tolerance', [{'initial_position_m': 1.}, {'initial_angle_rad': 4.}])
def test_supply_baseline_tolerance_cannot_be_widened_to_a_solved_tower(tolerance):
    with pytest.raises(ValueError, match='Tolerances'):
        plan(tolerances=tolerance)


def test_parent_environment_identity_and_native_availability_are_bound():
    for key, value in (('environment', {}), ('native_execution_available', True), ('source_release', 'fake')):
        config = plan()
        config[key] = value
        with pytest.raises(ValueError, match='differs'):
            TowerBuildingEvaluator(config, trusted_normalized_input=True)


def test_all_slots_chain_release_and_full_second_required():
    config, evaluator = started()
    result = settle(config, evaluator, end=52)
    assert result['status'] == 'incomplete' and result['stable_duration_s'] == pytest.approx(.98)
    result = evaluator.update(observation(config, 52))
    assert result['success'] and result['reference_success']
    assert result['stable_duration_s'] == pytest.approx(1.)
    assert result['native_qualified'] is False
    assert len({row['slot_id'] for row in result['blocks'].values()}) == 18


def test_identical_blocks_can_swap_ids_and_rotate_180_degrees():
    config = plan()
    canonical = tower_states(config, yaw_offset=math.pi)
    rename = {'block_00': 'block_17', 'block_17': 'block_00'}
    states = {rename.get(name, name): state for name, state in canonical.items()}
    rows = contacts(config, canonical)
    for row in rows:
        for key in ('a', 'b', 'support_for'):
            row[key] = rename.get(row[key], row[key])
    evaluator = TowerBuildingEvaluator(config, trusted_normalized_input=True)
    evaluator.update(observation(config, 0))
    evaluator.update(observation(config, 1, states=states, contact_rows=rows, vmax=100., wmax=200.))
    assert settle(config, evaluator, states=states, contact_rows=rows)['success']


@pytest.mark.parametrize('field,value', [('sampling_complete', False), ('contacts_complete', False),
    ('physics_valid', False), ('robot_only_actions', False), ('object_pose_writes', True),
    ('configuration_sha256', 'wrong'), ('episode_id', 'other_episode')])
def test_missing_or_invalid_evidence_fails_closed(field, value):
    config, evaluator = started()
    bad = observation(config, 2)
    bad[field] = value
    result = evaluator.update(bad)
    assert result['status'] == 'invalid'
    assert evaluator.update(observation(config, 3)) == result


def test_missing_block_nan_and_skipped_time_rejected():
    for kind in ('missing', 'nan', 'skip'):
        config, evaluator = started()
        bad = observation(config, 3 if kind == 'skip' else 2)
        if kind == 'missing': del bad['objects']['block_17']
        if kind == 'nan': bad['objects']['block_17']['position_m'][0] = float('nan')
        assert evaluator.update(bad)['status'] == 'invalid'


def test_reported_zero_speed_cannot_hide_object_pose_jump():
    config = plan()
    evaluator = TowerBuildingEvaluator(config, trusted_normalized_input=True)
    evaluator.update(observation(config, 0))
    result = evaluator.update(observation(config, 1))
    assert result['status'] == 'invalid' and 'speed bound' in result['reason']


@pytest.mark.parametrize('defect', ['missing_base_support', 'skipped_layer', 'floating', 'wrong_point',
                                  'zero_support', 'intermittent_support', 'penetration'])
def test_contact_backed_support_chain_is_required(defect):
    config, evaluator = started()
    states = tower_states(config)
    if defect == 'floating': states['block_17']['position_m'][2] += .002
    rows = contacts(config, states)
    if defect == 'missing_base_support': rows = [r for r in rows if r['a'] != 'block_00']
    if defect == 'skipped_layer': rows[-1]['b'] = 'block_09'
    if defect == 'wrong_point': rows[-1]['point_m'][0] += .3
    if defect == 'zero_support': rows[-1]['minimum_upward_force_n'] = 0.
    if defect == 'intermittent_support': rows[-1]['continuous'] = False
    if defect == 'penetration': rows[-1]['separation_m'] = -.01
    result = evaluator.update(observation(config, 2, states=states, contact_rows=rows, vmax=1.))
    assert not result['success'] and result['status'] == 'incomplete'
    assert any('continuous_contact_support_chain_required' in r['reasons'] for r in result['blocks'].values())


def test_count_height_or_duplicate_slot_does_not_make_a_tower():
    config, evaluator = started()
    states = tower_states(config)
    states['block_01'] = deepcopy(states['block_00'])
    result = evaluator.update(observation(config, 2, states=states, vmax=100.))
    assert result['status'] == 'incomplete'
    assert 'slot_has_multiple_blocks' in result['blocks']['block_00']['reasons']
    states = tower_states(config)
    states['block_17']['quaternion_wxyz'] = [1, 0, 0, 0]
    result = evaluator.update(observation(config, 3, states=states, vmax=100., wmax=100.))
    assert 'not_in_one_legal_slot' in result['blocks']['block_17']['reasons']


@pytest.mark.parametrize('report_contact', [True, False])
def test_deep_non_supporting_overlap_rejected_despite_valid_table_support(report_contact):
    config, evaluator = started()
    states = tower_states(config)
    states['block_00']['position_m'][1] += .004
    states['block_01']['position_m'][1] -= .004
    rows = contacts(config, states)
    if report_contact:
        rows.append({'a': 'block_00', 'b': 'block_01', 'point_m': [.65, -.018, .79],
            'separation_m': -.007, 'support_for': None, 'minimum_upward_force_n': 0., 'continuous': True})
    evaluator.update(observation(config, 2, states=states, contact_rows=rows, vmax=1.))
    result = settle(config, evaluator, start=3, end=55, states=states, contact_rows=rows)
    assert result['status'] == 'incomplete' and result['stable_duration_s'] == 0.
    for name in ('block_00', 'block_01'):
        assert 'geometric_block_penetration' in result['blocks'][name]['reasons']
        if report_contact:
            assert 'excessive_contact_penetration' in result['blocks'][name]['reasons']


def test_non_supporting_deep_contact_resets_a_previous_good_window():
    config, evaluator = started()
    settle(config, evaluator, end=30)
    rows = contacts(config, tower_states(config))
    rows.append({'a': 'block_00', 'b': 'block_01', 'point_m': [.65, -.018, .79],
        'separation_m': -.007, 'support_for': None, 'minimum_upward_force_n': 0., 'continuous': True})
    result = evaluator.update(observation(config, 30, contact_rows=rows))
    assert result['stable_duration_s'] == 0.
    assert 'excessive_contact_penetration' in result['blocks']['block_00']['reasons']


@pytest.mark.parametrize('actor', ['robot/right_index_tip', 'external/wall', 'external/tray', 'external/sofa'])
def test_robot_or_external_touch_resets_dwell_even_with_zero_force(actor):
    config, evaluator = started()
    settle(config, evaluator, end=30)
    rows = contacts(config, tower_states(config))
    rows.append({'a': 'block_17', 'b': actor, 'point_m': [.65, 0, .9],
        'separation_m': 0., 'support_for': None, 'minimum_upward_force_n': 0., 'continuous': False})
    interrupted = evaluator.update(observation(config, 30, contact_rows=rows))
    assert interrupted['stable_duration_s'] == 0.
    assert 'robot_or_external_contact' in interrupted['blocks']['block_17']['reasons']
    assert not settle(config, evaluator, start=31, end=81)['success']
    assert evaluator.update(observation(config, 81))['success']


def test_support_recovery_after_collapse_without_floor_is_allowed():
    config, evaluator = started()
    states = tower_states(config)
    states['block_17']['position_m'][0] += .1
    assert evaluator.update(observation(config, 2, states=states, vmax=100.))['status'] == 'incomplete'
    evaluator.update(observation(config, 3, vmax=100.))
    assert settle(config, evaluator, start=4, end=55)['success']


def test_floor_drop_can_be_terminal_or_recoverable_by_explicit_rule():
    for terminal in (True, False):
        config, evaluator = started(plan(floor_drop_is_terminal=terminal))
        rows = contacts(config, tower_states(config))
        rows.append({'a': 'block_17', 'b': 'floor', 'point_m': [0, 0, 0],
            'separation_m': 0., 'support_for': None, 'minimum_upward_force_n': 0., 'continuous': False})
        result = evaluator.update(observation(config, 2, contact_rows=rows))
        assert result['status'] == ('failure' if terminal else 'incomplete')
        if terminal: assert evaluator.update(observation(config, 3)) == result


def test_budget_is_terminal_incomplete_and_never_success():
    config, evaluator = started(plan(budget={'max_sim_time_s': .5}))
    result = settle(config, evaluator, end=26)
    assert result['status'] == 'incomplete' and result['terminal']
    assert result['reason'] == 'budget_exhausted' and not result['success']


@pytest.mark.parametrize('kind', ['overshoot', 'deadline', 'explicit', 'actions'])
def test_budget_precedes_success_on_the_same_final_sample(kind):
    budget = {'max_sim_time_s': 1.03} if kind == 'overshoot' else {'max_sim_time_s': 1.04} if kind == 'deadline' else {'max_robot_actions': 2} if kind == 'actions' else {}
    config, evaluator = started(plan(budget=budget))
    assert not settle(config, evaluator, end=52)['success']
    result = evaluator.update(observation(config, 52, budget_exhausted=kind == 'explicit',
                                         robot_action_count=2 if kind == 'actions' else 1))
    assert result['status'] == 'incomplete' and result['terminal']
    assert result['reason'] == 'budget_exhausted'


def test_interval_speed_bound_must_prove_stability_between_samples():
    config, evaluator = started(plan(tolerances={'sample_period_s': .5}))
    # Exact endpoints with this bound still permit a 2 mm excursion per interval.
    result = settle(config, evaluator, end=8, vmax=.008)
    assert result['status'] == 'incomplete' and result['stable_duration_s'] == 0.


def test_slow_drift_cannot_accumulate_a_stationary_second():
    config, evaluator = started()
    for tick in range(2, 53):
        shift = min((tick-2)*.00006, .003)
        states = tower_states(config, shift=shift)
        result = evaluator.update(observation(config, tick, states=states, vmax=.004))
    assert result['status'] == 'incomplete' and result['stable_duration_s'] < 1.


def test_replay_matches_streaming_and_does_not_infer_one_frame_dwell():
    config = plan()
    rows = [observation(config, 0), observation(config, 1, vmax=100., wmax=100.)]
    rows += [observation(config, i) for i in range(2, 53)]
    assert evaluate_tower(rows, config, trusted_normalized_input=True)['success']
    assert evaluate_tower(rows[:3], config, trusted_normalized_input=True)['status'] == 'incomplete'
    assert evaluate_tower([rows[-1]], config, trusted_normalized_input=True)['status'] == 'invalid'
