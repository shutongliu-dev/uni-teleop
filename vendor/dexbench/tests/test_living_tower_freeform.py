"""Synthetic referee evidence only; these tests do not qualify native physics."""
from copy import deepcopy
import math

import pytest

from scenes.workrooms.living_room.tasks.tower_building.evaluator import TowerBuildingEvaluator
from scenes.workrooms.living_room.tasks.tower_building.rules import OBSERVATION_SCHEMA
from scenes.workrooms.living_room.tasks.tower_building.task import task_configuration


def plan(**overrides):
    trays, layout = [], {}
    for side, x0, xs in (('left', .245, (.29, .37, .45)), ('right', .805, (.85, .93, 1.01))):
        name = 'supply_'+side
        trays.append({'id': name, 'interior_bounds_xy_m': [[x0, -.43], [x0+.25, .17]],
            'floor_z_m': .786, 'contact_actor_ids': ['external/'+name, 'external/'+name+'/wall_neg_x']})
        for level in range(2):
            for x in xs:
                for y in (-.33, -.13, .07):
                    layout[f'block_{len(layout):02d}'] = {'position_m': [x, y, .7985+level*.025],
                        'quaternion_wxyz': [math.sqrt(.5), 0., 0., math.sqrt(.5)]}
    options = dict(inventory_count=36, supply_trays=trays, supply_layout=layout, scoring_mode='freeform_height')
    options.update(overrides)
    return task_configuration({'block_size_m': [.15, .045, .025]}, **options)


def contact(a, b, point, **overrides):
    return {'a': a, 'b': b, 'point_m': list(point), 'support_for': a,
        'separation_m': 0., 'minimum_upward_force_n': .1, 'continuous': True, **overrides}


def built(config, count=1):
    states = deepcopy(config['initial_objects'])
    selected = config['block_ids'][-count:] if count else []
    rows = []
    for index, name in enumerate(selected):
        # Stand blocks on their ends, away from the former center target footprint.
        states[name] = {'position_m': [.65, .31, .855+index*.15],
            'quaternion_wxyz': [math.sqrt(.5), 0., math.sqrt(.5), 0.]}
        rows.append(contact(name, 'table' if index == 0 else selected[index-1], [.65, .31, .78+index*.15]))
    for index, name in enumerate(config['block_ids']):
        if name in selected:
            continue
        tray = 'external/supply_left' if index < 18 else 'external/supply_right'
        x, y, z = states[name]['position_m']
        lower = tray if index % 18 < 9 else f'block_{index-9:02d}'
        rows.append(contact(name, lower, [x, y, z-.0125]))
    return states, rows


def obs(config, tick, states=None, rows=None, vmax=0., wmax=0., **extra):
    states = deepcopy(config['initial_objects'] if states is None else states)
    for state in states.values():
        state.update(linear_velocity_m_s=[0., 0., 0.], angular_velocity_rad_s=[0., 0., 0.],
            interval_max_linear_speed_m_s=vmax, interval_max_angular_speed_rad_s=wmax)
    return {'schema': OBSERVATION_SCHEMA, 'configuration_sha256': config['configuration_sha256'],
        'episode_id': 'synthetic_freeform_only', 'sample_index': tick, 'time_s': tick*.02,
        'interval_start_s': max(0, tick-1)*.02, 'sampling_complete': True,
        'contacts_complete': True, 'physics_valid': True, 'robot_only_actions': True,
        'object_pose_writes': False, 'budget_exhausted': False,
        'robot_action_count': 0 if tick == 0 else 1, 'objects': states, 'contacts': rows or [], **extra}


def started(config=None, count=1):
    config = config or plan()
    evaluator = TowerBuildingEvaluator(config, trusted_normalized_input=True)
    baseline = evaluator.update(obs(config, 0))
    assert baseline['score'] == 0. and not baseline['success']
    states, rows = built(config, count=count)
    assert evaluator.update(obs(config, 1, states, rows, vmax=100., wmax=200.))['status'] == 'incomplete'
    return config, evaluator, states, rows


def settle(config, evaluator, states, rows, start=2, end=53, **extra):
    for tick in range(start, end):
        result = evaluator.update(obs(config, tick, states, rows, **extra))
    return result


def test_legacy_factory_outputs_remain_identical():
    # Hashed in portable form, so the values do not depend on where the data package lives.
    for size, seed, digest in (([.11, .035, .02], 0, '199b900c2fb392902101e39c8e40ec1268d71305b7070270d85accadd9da6123'),
        ([.11, .035, .02], 3, '9bd2487a9436a7f5bb739af998dd0d6f1f21f413d6cfe87c0d2cbd0f63edaf75'),
        ([.15, .045, .025], 0, '10a6120579b93a49a55c5ed8f9093530a731617d88eee5289d60cf48b41f947d')):
        assert task_configuration({'block_size_m': size}, seed=seed)['configuration_sha256'] == digest


def test_reset_has_36_blocks_in_supported_two_level_trays_without_target_geometry():
    config = plan()
    assert len(config['block_ids']) == len(config['initial_objects']) == 36
    assert config['target_slots'] == []
    assert 'layers' not in config['geometry'] and 'block_count' not in config['rules']['goal']
    assert config['rules']['goal']['score_aggregation'] == 'maximum_stable_height_reached'
    evaluator = TowerBuildingEvaluator(config, trusted_normalized_input=True)
    evaluator.update(obs(config, 0))
    _, rows = built(config, count=0)
    result = settle(config, evaluator, config['initial_objects'], rows, start=1, end=70)
    assert result['score'] == result['current_height_m'] == 0.
    assert not result['terminal'] and not result['success']


def test_legacy_target_design_parameters_do_not_constrain_freeform_geometry():
    config = plan()
    changed = task_configuration({'block_size_m': [.15, .045, .025],
        'layers': 999, 'blocks_per_layer': 1, 'center_xy_m': [50., 50.], 'in_layer_gap_m': 5.},
        inventory_count=36, supply_trays=config['supply_trays'], supply_layout=config['initial_objects'],
        scoring_mode='freeform_height')
    assert changed == config


@pytest.mark.parametrize('defect', ['missing', 'floating', 'overlapping', 'outside', 'floor', 'actor'])
def test_bad_tray_reset_declarations_fail_closed(defect):
    config = plan()
    layout, trays = deepcopy(config['initial_objects']), deepcopy(config['supply_trays'])
    if defect == 'missing': del layout['block_35']
    if defect == 'floating': layout['block_35']['position_m'][2] += .001
    if defect == 'overlapping': layout['block_35'] = deepcopy(layout['block_34'])
    if defect == 'outside': layout['block_35']['position_m'][0] = .65
    if defect == 'floor': trays[0]['floor_z_m'] = .77
    if defect == 'actor': trays[0]['contact_actor_ids'].append('external/unrelated_floor')
    with pytest.raises(ValueError):
        plan(supply_layout=layout, supply_trays=trays)


def test_single_upright_block_anywhere_scores_actual_obb_height_after_full_second():
    config, evaluator, states, rows = started()
    early = settle(config, evaluator, states, rows, end=52)
    assert early['score'] == 0. and early['stable_duration_s'] == pytest.approx(.98)
    result = evaluator.update(obs(config, 52, states, rows))
    assert result['score'] == pytest.approx(.15)
    assert result['score_unit'] == 'm' and result['score_valid']
    assert result['best_height_cm'] == pytest.approx(15.)
    assert result['current_height_m'] == pytest.approx(.15)
    assert result['best_stable_block_ids'] == ['block_35']
    assert not result['success'] and not result['terminal']


def test_best_height_survives_collapse_drop_and_budget_then_can_improve():
    config, evaluator, states, rows = started(count=2)
    assert settle(config, evaluator, states, rows)['best_height_m'] == pytest.approx(.3)
    collapsed = deepcopy(states)
    collapsed['block_35']['position_m'] = [.65, .31, .0125]
    floor_rows = [row for row in rows if row['a'] != 'block_35']
    floor_rows.append(contact('block_35', 'floor', [.65, .31, 0.]))
    result = evaluator.update(obs(config, 53, collapsed, floor_rows, vmax=100.))
    assert result['score'] == pytest.approx(.3) and not result['terminal']
    assert result['current_height_m'] == pytest.approx(.15)
    recovered, recovered_rows = built(config, count=3)
    evaluator.update(obs(config, 54, recovered, recovered_rows, vmax=100., wmax=200.))
    improved = settle(config, evaluator, recovered, recovered_rows, start=55, end=106)
    assert improved['best_height_m'] == pytest.approx(.45)
    closed = evaluator.update(obs(config, 106, recovered, recovered_rows, budget_exhausted=True))
    assert closed['terminal'] and closed['status'] == 'incomplete'
    assert closed['score'] == pytest.approx(.45) and closed['reason'] == 'budget_exhausted'


@pytest.mark.parametrize('actor', ['external/supply_left', 'external/supply_left/wall_neg_x',
                                  'external/unknown_fixture', 'robot/right_index_tip', 'floor'])
def test_tray_robot_or_external_touch_disqualifies_whole_connected_assembly(actor):
    config, evaluator, states, rows = started(count=2)
    rows.append(contact('block_34', actor, [.65, .31, .78], support_for=None, minimum_upward_force_n=0.))
    result = settle(config, evaluator, states, rows)
    assert result['score'] == 0. and result['current_height_m'] == 0.


def test_tray_support_cannot_be_laundered_through_an_extra_wood_block():
    config, evaluator, states, rows = started(count=2)
    # A trusted receipt giving the base both table and supply contact excludes
    # the entire connected assembly, even if a table support path also exists.
    rows.append(contact('block_34', 'block_00', [.65, .31, .78], support_for=None, minimum_upward_force_n=0.))
    result = settle(config, evaluator, states, rows)
    assert result['score'] == 0.


@pytest.mark.parametrize('defect', ['missing_support', 'zero_force', 'intermittent', 'wrong_point', 'cycle', 'off_table'])
def test_unrooted_or_unphysical_assembly_earns_no_height(defect):
    config, evaluator, states, rows = started(count=2)
    rows = deepcopy(rows)
    if defect == 'missing_support': rows = rows[1:]
    if defect == 'zero_force': rows[0]['minimum_upward_force_n'] = 0.
    if defect == 'intermittent': rows[0]['continuous'] = False
    if defect == 'wrong_point': rows[0]['point_m'][0] += .2
    if defect == 'cycle': rows[0]['b'] = 'block_35'
    if defect == 'off_table':
        for name in ('block_34', 'block_35'): states[name]['position_m'][1] += .3
        for row in rows[:2]: row['point_m'][1] += .3
    evaluator.update(obs(config, 2, states, rows, vmax=100.))
    assert settle(config, evaluator, states, rows, start=3, end=55)['score'] == 0.


def test_intermittent_support_and_slow_drift_restart_height_window():
    config, evaluator, states, rows = started()
    settle(config, evaluator, states, rows, end=30)
    interrupted = deepcopy(rows)
    interrupted[0]['continuous'] = False
    assert evaluator.update(obs(config, 30, states, interrupted))['stable_duration_s'] == 0.
    for tick in range(31, 90):
        drifting, drifting_rows = deepcopy(states), deepcopy(rows)
        shift = (tick-31)*.00006
        drifting['block_35']['position_m'][0] += shift
        drifting_rows[0]['point_m'][0] += shift
        result = evaluator.update(obs(config, tick, drifting, drifting_rows, vmax=.004))
    assert result['score'] == 0.


def test_unconnected_supplies_can_move_or_drop_without_erasing_a_stable_tower_score():
    config, evaluator, states, rows = started()
    settle(config, evaluator, states, rows)
    moved = deepcopy(states)
    moved['block_09']['position_m'] = [.35, -.25, .0125]
    contacts = [row for row in rows if row['a'] != 'block_09']
    contacts.append(contact('block_09', 'floor', [.35, -.25, 0.]))
    observation = obs(config, 53, moved, contacts)
    observation['objects']['block_09']['interval_max_linear_speed_m_s'] = 100.
    result = evaluator.update(observation)
    assert result['score'] == pytest.approx(.15) and result['current_height_m'] == pytest.approx(.15)
    assert result['stable_duration_s'] == pytest.approx(1.02)


def test_moving_unconnected_supply_does_not_prevent_first_height_credit():
    config, evaluator, states, rows = started()
    for tick in range(2, 53):
        sample = obs(config, tick, states, rows)
        sample['objects']['block_09']['interval_max_linear_speed_m_s'] = 100.
        result = evaluator.update(sample)
    assert result['score'] == pytest.approx(.15)


def test_final_budget_sample_can_record_a_new_stable_height_and_invalid_evidence_retains_but_disqualifies_score():
    config, evaluator, states, rows = started(plan(budget={'max_sim_time_s': 1.04}))
    result = settle(config, evaluator, states, rows)
    assert result['terminal'] and result['score'] == pytest.approx(.15)
    config, evaluator, states, rows = started()
    settle(config, evaluator, states, rows)
    bad = obs(config, 53, states, rows)
    del bad['objects']['block_00']
    invalid = evaluator.update(bad)
    assert invalid['status'] == 'invalid' and invalid['terminal']
    assert invalid['best_height_m'] == pytest.approx(.15) and not invalid['score_valid']
