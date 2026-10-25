"""Study task referee semantics using synthetic CPU evidence, never native acceptance."""
from copy import deepcopy
import math

import pytest

from scenes.workrooms.study.rules import (OBSERVATION_SCHEMA, TASK_IDS,
    build_configuration, canonical_hash, validate_configuration)
from scenes.workrooms.study.evaluation import SequenceEvaluator

Q_FLAT = [math.sqrt(.5), math.sqrt(.5), 0., 0.]
Q_UP = [1., 0., 0., 0.]


def fixture(task):
    """Each fixed object has a reachable, different reset and explicit final support."""
    shelf = [[-.8, -.7, 1.08], [-.5, .7, 1.12]]
    supports = {'table': {'bounds_m': [[0., -.7, .74], [1., .7, .78]]},
                'shelf_upper': {'bounds_m': shelf}}
    names = ['marker', 'eraser', 'notes'] if task == TASK_IDS[0] else ['robotics', 'astronomy', 'botany']
    if task == TASK_IDS[3]: names = names[:2]
    objects, targets, final = {}, {}, {}
    for i, name in enumerate(names):
        stationery = task == TASK_IDS[0]
        upright = task == TASK_IDS[3] or task == TASK_IDS[2] and name != 'astronomy'
        size = [.12, .03, .20] if not stationery else [.06, .04, .02]
        local = [[-d/2 for d in size], [d/2 for d in size]]
        q = Q_UP if stationery or upright else Q_FLAT
        if stationery:
            p, support, z = [.55, -.3 + i*.3, .79], 'table', .78
            category = ['writing', 'erasing', 'notes'][i]
        elif task == TASK_IDS[1]:
            p, support, z = [.55, 0., .795+i*.03], 'table' if i == 0 else names[i-1], .78+i*.03
            category = 'science'
        elif upright:
            p, support, z = [-.65, -.35 + i*.25, 1.22], 'shelf_upper', 1.12
            category = 'science' if i == 0 else 'arts'
        else:
            p, support, z, category = [.55, 0., .795], 'table', .78, 'science'
        height = .02 if stationery else .20 if upright else .03
        half_xy = [size[0]/2, size[1]/2 if upright or stationery else size[2]/2]
        bounds = [[p[0]-half_xy[0]-.025, p[1]-half_xy[1]-.025, z-.001],
                  [p[0]+half_xy[0]+.025, p[1]+half_xy[1]+.025, z+height+.01]]
        initial = {'position_m': [.25, -.3+i*.3, .795 if not stationery else .79],
                   'quaternion_wxyz': deepcopy(Q_UP if stationery else Q_FLAT)}
        if task == TASK_IDS[2]:
            initial = {'position_m': [-.65, -.35+i*.25, 1.22], 'quaternion_wxyz': deepcopy(Q_UP)}
        obj = {'category': category, 'local_bounds_m': local, 'initial_pose': initial}
        goal = {'slot_id': name+'_slot', 'category': category, 'bounds_m': bounds,
                'support_id': support, 'support_z_m': z}
        if not stationery:
            obj.update(up_axis_local=[0,0,1] if upright else [0,1,0], spine_axis_local=[1,0,0])
            goal.update(up_direction_world=[0,0,1], spine_direction_world=[1,0,0])
        objects[name], targets[name] = obj, goal
        final[name] = {'position_m': p, 'quaternion_wxyz': deepcopy(q),
                       'linear_velocity_m_s': [0.,0.,0.], 'angular_velocity_rad_s': [0.,0.,0.]}
    kwargs = {'stack_order': names} if task == TASK_IDS[1] else {'retrieval_target_id': 'astronomy', 'protected_ids': ['robotics','botany']} if task == TASK_IDS[2] else {}
    config = build_configuration(task, objects, targets, supports, **kwargs)
    return config, final


def contacts(config, states):
    return [{'a': name, 'b': goal['support_id'],
             'point_m': [*states[name]['position_m'][:2], goal['support_z_m']],
             'separation_m': 0., 'impulse_on_a_n_s': [0.,0.,.001]}
            for name,goal in config['targets'].items()]


def sample(config, index, states, rows=None, *, actions=1):
    return {'schema': OBSERVATION_SCHEMA, 'configuration_sha256': config['configuration_sha256'],
            'sampling_complete': True, 'contacts_complete': True, 'physics_valid': True,
            'robot_only_actions': True, 'object_pose_writes': False, 'episode_id': 'test_episode',
            'step_index': index, 'robot_action_count': actions, 'budget_exhausted': False,
            'time_s': index*.0005, 'physics_dt_s': .0005, 'objects': deepcopy(states),
            'contacts': deepcopy(contacts(config, states) if rows is None else rows)}


def start(config, *, trusted=True):
    evaluator = SequenceEvaluator(config, trusted_normalized_input=trusted)
    reset = {name: {**deepcopy(obj['initial_pose']), 'linear_velocity_m_s': [0.,0.,0.],
              'angular_velocity_rad_s': [0.,0.,0.]} for name,obj in config['objects'].items()}
    result = evaluator.update(sample(config, 0, reset, [], actions=0))
    return evaluator, result


def settle(evaluator, config, states, *, begin=1, count=603, rows=None):
    for index in range(begin, begin+count):
        result = evaluator.update(sample(config, index, states, rows))
    return result


@pytest.mark.parametrize('task', TASK_IDS)
def test_four_tasks_require_full_common_native_window_after_reset(task):
    config, final = fixture(task)
    evaluator, result = start(config)
    assert result['reason'] == 'reset_baseline_not_scored' and result['progress'] == 0.
    assert 'action_budget' not in result
    # First moved frame fails finite-difference speed; next starts the clock.
    result = settle(evaluator, config, final, count=601)
    assert not result['success'] and result['stable_duration_s'] < .3
    result = evaluator.update(sample(config, 602, final))
    assert result['success'] and result['referee']['state'] == 'success'
    assert result['stable_duration_s'] == pytest.approx(.3)
    assert evaluator.update({}) == result  # accepted terminal results latch


@pytest.mark.parametrize('change', ['downward', 'missing', 'bad_point', 'wrong_support', 'held'])
def test_stationery_needs_real_signed_support_and_all_robot_contacts_released(change):
    config, final = fixture(TASK_IDS[0])
    rows = contacts(config, final)
    if change == 'downward': rows[0]['impulse_on_a_n_s'][2] *= -1
    elif change == 'missing': rows.pop(0)
    elif change == 'bad_point': rows[0]['point_m'][0] += .3
    elif change == 'wrong_support': rows[0]['b'] = 'shelf_upper'
    else: rows.append({'a': 'marker', 'b': 'robot/fingertip', 'point_m': [.55,-.3,.8],
                       'separation_m': 0., 'impulse_on_a_n_s': [0.,0.,0.]})
    evaluator, _ = start(config)
    result = settle(evaluator, config, final, rows=rows)
    assert result['status'] == 'active' and result['stable_duration_s'] == 0.
    assert settle(evaluator, config, final, begin=604)['success']


def test_stack_cannot_substitute_table_contact_for_immediate_lower_book():
    config, final = fixture(TASK_IDS[1])
    rows = contacts(config, final)
    rows[1]['b'] = 'table'
    evaluator, _ = start(config)
    result = settle(evaluator, config, final, rows=rows)
    assert not result['objects']['astronomy']['supported']
    assert not result['objects']['botany']['supported']
    assert settle(evaluator, config, final, begin=604)['success']


@pytest.mark.parametrize('flag', ['sampling_complete','contacts_complete','physics_valid','robot_only_actions'])
def test_missing_native_evidence_is_invalid_never_physical_failure(flag):
    config, final = fixture(TASK_IDS[0])
    evaluator, _ = start(config)
    obs = sample(config, 1, final)
    obs[flag] = False
    result = evaluator.update(obs)
    assert result['status'] == 'invalid' and result['referee']['state'] == 'evaluator_invalid'
    assert evaluator.finish('budget_exhausted') == result


def test_skipped_substeps_and_untrusted_json_cannot_receive_credit():
    config, final = fixture(TASK_IDS[0])
    evaluator, _ = start(config)
    assert evaluator.update(sample(config, 2, final))['status'] == 'invalid'
    _, result = start(config, trusted=False)
    assert result['status'] == 'invalid'


def test_floor_contact_is_terminal_but_desk_drop_and_wrong_position_can_recover():
    config, final = fixture(TASK_IDS[2])
    evaluator, _ = start(config)
    dropped = deepcopy(final)
    dropped['robotics']['position_m'] = [.3,-.3,.795]
    dropped['robotics']['quaternion_wxyz'] = Q_FLAT
    rows = contacts(config, dropped)
    rows[0].update(b='table', point_m=[.3,-.3,.78])
    result = settle(evaluator, config, dropped, count=50, rows=rows)
    assert result['status'] == 'active'
    assert settle(evaluator, config, final, begin=51)['success']
    evaluator, _ = start(config)
    rows = [{'a': 'robotics', 'b': 'floor', 'point_m': [.3,-.3,0.],
             'separation_m': 0., 'impulse_on_a_n_s': [0.,0.,0.]}]
    result = evaluator.update(sample(config, 1, dropped, rows))
    assert result['status'] == 'failure' and result['reason'] == 'object_touched_floor'


def test_budget_beats_success_at_same_native_step_and_missing_finish_is_invalid():
    config, final = fixture(TASK_IDS[0])
    evaluator, _ = start(config)
    settle(evaluator, config, final, count=601)
    obs = sample(config, 602, final)
    obs['budget_exhausted'] = True
    result = evaluator.update(obs)
    assert result['status'] == 'incomplete' and not result['success']
    evaluator, _ = start(config)
    assert evaluator.finish('evidence_missing')['status'] == 'invalid'


@pytest.mark.parametrize('maximum', [1, 2, 5])
def test_action_budget_allows_every_substep_of_the_last_admitted_action(maximum):
    config, final = fixture(TASK_IDS[0])
    config['budget']['max_robot_actions'] = maximum
    config = _rebuild(config)
    assert any(f'完整的前 {maximum} 个公开机器人动作' in text
               for text in config['rules']['referee']['success_conditions'])
    evaluator, result = start(config)
    assert result['action_budget'] == {'maximum': maximum, 'admitted': 0}
    for index in range(1, 401):
        actions = min(index, maximum)
        result = evaluator.update(sample(config, index, final, actions=actions))
        assert result['status'] == 'active' and not result['terminal']
        assert result['action_budget'] == {'maximum': maximum, 'admitted': actions}
    result = evaluator.update(sample(config, 401, final, actions=maximum+1))
    assert result['status'] == 'incomplete' and result['reason'] == 'budget_exhausted'
    assert result['action_budget'] == {'maximum': maximum, 'admitted': maximum+1}


def test_last_permitted_action_can_finish_the_full_success_window():
    config, final = fixture(TASK_IDS[0])
    config['budget']['max_robot_actions'] = 1
    config = _rebuild(config)
    evaluator, _ = start(config)
    result = settle(evaluator, config, final)
    assert result['success'] and result['action_budget'] == {'maximum': 1, 'admitted': 1}


def test_frozen_public_targets_and_configuration_hash_are_bound_together():
    config, _ = fixture(TASK_IDS[1])
    changed = deepcopy(config)
    changed['targets']['robotics']['slot_id'] = 'other_slot'
    with pytest.raises(ValueError): validate_configuration(changed)
    changed['configuration_sha256'] = canonical_hash({k:v for k,v in changed.items() if k != 'configuration_sha256'})
    with pytest.raises(ValueError): validate_configuration(changed)


@pytest.mark.parametrize('task', TASK_IDS)
def test_whole_object_overhang_rejected_with_center_still_inside_its_slot(task):
    config, final = fixture(task)
    ident = next(iter(final))
    goal = config['targets'][ident]['bounds_m']
    final[ident]['position_m'][0] = goal[1][0]-.001
    assert goal[0][0] < final[ident]['position_m'][0] < goal[1][0]
    evaluator, _ = start(config)
    result = settle(evaluator, config, final)
    assert 'outside_target_slot' in result['objects'][ident]['reasons']
    assert not result['success']


@pytest.mark.parametrize('authored_margin_m', [0., .001])
@pytest.mark.parametrize('penetration_m, accepted', [
    (0.0000000258874635, True),  # native03 eraser resting envelope, 25.9 nm
    (0.00074183469521, True),    # native03 marker resting envelope, 0.742 mm
    (.001, True), (.0011, False),
])
def test_resting_envelope_uses_frozen_lower_tolerance_without_adding_to_authored_margin(
        authored_margin_m, penetration_m, accepted):
    config, final = fixture(TASK_IDS[0])
    ident = 'eraser'
    goal = config['targets'][ident]
    goal['bounds_m'][0][2] = goal['support_z_m']-authored_margin_m
    config = _rebuild(config)
    final[ident]['position_m'][2] -= penetration_m
    rows = contacts(config, final)
    next(row for row in rows if row['a'] == ident)['separation_m'] = -penetration_m
    evaluator, _ = start(config)
    result = settle(evaluator, config, final, rows=rows)
    assert result['success'] is accepted
    reasons = result['objects'][ident]['reasons']
    if accepted:
        assert reasons == []
    else:
        assert 'outside_target_slot' in reasons and 'contact_penetration' in reasons


@pytest.mark.parametrize('edge', ['x_lower', 'y_lower', 'x_upper', 'y_upper', 'z_upper'])
def test_support_plane_tolerance_never_expands_xy_or_upper_target_edges(edge):
    from scenes.workrooms.study.evaluation import world_bounds

    config, final = fixture(TASK_IDS[0])
    evaluator, _ = start(config)
    ident = 'eraser'
    axis, side = edge.split('_')
    index, bound_index = 'xyz'.index(axis), int(side == 'upper')
    state = final[ident]
    bounds = world_bounds(evaluator._state(state), config['objects'][ident]['local_bounds_m'])
    wanted = config['targets'][ident]['bounds_m'][bound_index][index]
    state['position_m'][index] += wanted-bounds[bound_index][index] + (.00001 if bound_index else -.00001)
    assert 'outside_target_slot' in evaluator._geometry(ident, evaluator._state(state))


@pytest.mark.parametrize('task', TASK_IDS[1:])
@pytest.mark.parametrize('orientation', ['backwards', 'wrong_up'])
def test_directed_book_axes_reject_backwards_spines_and_wrong_cover_or_upright(task, orientation):
    config, final = fixture(task)
    ident = next(iter(final))
    # Premultiplication by pi around world Z reverses the spine; changing to
    # a perpendicular orientation also changes the required directed up axis.
    q = final[ident]['quaternion_wxyz']
    final[ident]['quaternion_wxyz'] = [-q[3], -q[2], q[1], q[0]] if orientation == 'backwards' else deepcopy(Q_FLAT if q == Q_UP else Q_UP)
    evaluator, _ = start(config)
    result = settle(evaluator, config, final)
    assert 'incorrect_orientation' in result['objects'][ident]['reasons']
    assert result['status'] == 'active'


@pytest.mark.parametrize('task', TASK_IDS)
def test_identity_swaps_do_not_substitute_for_the_designated_objects(task):
    config, final = fixture(task)
    a, b = list(final)[:2]
    final[a], final[b] = final[b], final[a]
    evaluator, _ = start(config)
    result = settle(evaluator, config, final)
    assert not result['success'] and any(not row['geometry'] for row in result['objects'].values())


def test_support_contact_actor_order_is_signed_and_physical_point_must_be_at_bottom():
    config, final = fixture(TASK_IDS[0])
    rows = contacts(config, final)
    for row in rows:
        row['a'], row['b'] = row['b'], row['a']
        row['impulse_on_a_n_s'][2] *= -1
    evaluator, _ = start(config)
    assert settle(evaluator, config, final, rows=rows)['success']
    rows = contacts(config, final)
    rows[0]['point_m'][2] += .015
    evaluator, _ = start(config)
    assert not settle(evaluator, config, final, rows=rows)['objects']['marker']['supported']


@pytest.mark.parametrize('disturbance', ['reported_velocity', 'finite_difference', 'robot_contact', 'support_gap', 'penetration', 'wall_contact'])
def test_disturbance_restarts_the_complete_dwell_then_can_recover(disturbance):
    config, final = fixture(TASK_IDS[0])
    evaluator, _ = start(config)
    settle(evaluator, config, final, count=500)
    obs = sample(config, 501, final)
    if disturbance == 'reported_velocity': obs['objects']['marker']['linear_velocity_m_s'][0] = .011
    elif disturbance == 'finite_difference': obs['objects']['marker']['position_m'][0] += .00002
    elif disturbance == 'support_gap': obs['contacts'] = obs['contacts'][1:]
    elif disturbance == 'penetration': obs['contacts'][0]['separation_m'] = -.002
    else:
        obs['contacts'].append({'a': 'marker', 'b': 'robot/thumb' if disturbance == 'robot_contact' else 'external/wall',
            'point_m': [.55, -.3, .8], 'separation_m': 0., 'impulse_on_a_n_s': [0.,0.,0.]})
    assert evaluator.update(obs)['stable_duration_s'] == 0.
    result = settle(evaluator, config, final, begin=502, count=599)
    assert not result['success']
    assert settle(evaluator, config, final, begin=1101, count=3)['success']


def test_slow_creep_exceeding_anchor_drift_cannot_accumulate_three_hundred_ms():
    config, final = fixture(TASK_IDS[0])
    evaluator, _ = start(config)
    for i in range(1, 1001):
        shifted = deepcopy(final)
        # 5 mm/s stays beneath the speed gate, but exceeds 1 mm drift each .2s.
        shifted['marker']['position_m'][0] += i*.0000025
        result = evaluator.update(sample(config, i, shifted))
        assert not result['success']
    assert result['status'] == 'active' and result['stable_duration_s'] < .2


def test_robot_action_required_even_with_complete_solved_evidence_after_reset():
    config, final = fixture(TASK_IDS[0])
    evaluator, _ = start(config)
    for i in range(1, 650):
        result = evaluator.update(sample(config, i, final, actions=0))
    assert result['reason'] == 'robot_action_required' and not result['success']
    for i in range(650, 1251):
        result = evaluator.update(sample(config, i, final))
    assert result['success']


@pytest.mark.parametrize('fault', ['object_missing', 'extra_object', 'nan', 'missing_velocity', 'wrong_episode',
                                  'wrong_time', 'pose_write', 'wrong_hash', 'unknown_contact', 'decreasing_actions'])
def test_incomplete_or_inconsistent_native_samples_latch_invalid(fault):
    config, final = fixture(TASK_IDS[0])
    evaluator, _ = start(config)
    evaluator.update(sample(config, 1, final))
    obs = sample(config, 2, final)
    if fault == 'object_missing': del obs['objects']['marker']
    elif fault == 'extra_object': obs['objects']['fake'] = deepcopy(obs['objects']['marker'])
    elif fault == 'nan': obs['objects']['marker']['position_m'][0] = float('nan')
    elif fault == 'missing_velocity': del obs['objects']['marker']['linear_velocity_m_s']
    elif fault == 'wrong_episode': obs['episode_id'] = 'other'
    elif fault == 'wrong_time': obs['time_s'] += .001
    elif fault == 'pose_write': obs['object_pose_writes'] = True
    elif fault == 'wrong_hash': obs['configuration_sha256'] = '0'*64
    elif fault == 'unknown_contact': obs['contacts'][0]['b'] = 'unmapped_actor'
    else: obs['robot_action_count'] = 0
    result = evaluator.update(obs)
    assert result['status'] == 'invalid' and result['referee']['state'] == 'evaluator_invalid'
    assert evaluator.update(sample(config, 3, final)) == result


def _rebuild(config):
    return build_configuration(config['task_id'], config['objects'], config['targets'], config['supports'],
        budget=config['budget'], stack_order=config.get('stack_order'),
        retrieval_target_id=config.get('retrieval_target_id'), protected_ids=config.get('protected_ids'))


@pytest.mark.parametrize('fault', ['category', 'overlap', 'wrong_support_height'])
def test_authored_configuration_cannot_contradict_distinct_category_slots(fault):
    config, _ = fixture(TASK_IDS[0])
    if fault == 'category': config['targets']['marker']['category'] = 'erasing'
    elif fault == 'overlap': config['targets']['marker']['bounds_m'] = deepcopy(config['targets']['eraser']['bounds_m'])
    else: config['targets']['marker']['support_z_m'] += .001
    with pytest.raises(ValueError): _rebuild(config)


@pytest.mark.parametrize('fault', ['wrong_order', 'wrong_front', 'wrong_stack_height'])
def test_authored_stack_order_cover_and_thickness_are_consistent(fault):
    config, _ = fixture(TASK_IDS[1])
    if fault == 'wrong_order': config['stack_order'].reverse()
    elif fault == 'wrong_front': config['objects']['robotics']['up_axis_local'] = [0,0,1]
    else: config['targets']['botany']['support_z_m'] += .001
    with pytest.raises(ValueError): _rebuild(config)


def test_protected_books_must_target_their_own_original_shelf_slots():
    config, _ = fixture(TASK_IDS[2])
    config['targets']['robotics']['bounds_m'][0][1] += .1
    with pytest.raises(ValueError): _rebuild(config)


def test_shelving_accepts_three_books_and_two_categories_but_rejects_flat_authored_goals():
    config, _ = fixture(TASK_IDS[3])
    config['objects']['botany'] = deepcopy(config['objects']['robotics'])
    config['objects']['botany']['initial_pose']['position_m'][1] = .3
    goal = deepcopy(config['targets']['robotics'])
    goal.update(slot_id='botany_slot')
    for row in goal['bounds_m']: row[1] += .6
    config['targets']['botany'] = goal
    assert len(_rebuild(config)['objects']) == 3
    config['objects']['botany']['up_axis_local'] = [0,1,0]
    with pytest.raises(ValueError): _rebuild(config)


def test_reset_pose_is_bound_and_fully_solved_reset_is_rejected():
    config, final = fixture(TASK_IDS[0])
    solved = deepcopy(config)
    for name, row in final.items():
        solved['objects'][name]['initial_pose'] = {k: deepcopy(row[k]) for k in ('position_m','quaternion_wxyz')}
    with pytest.raises(ValueError): SequenceEvaluator(_rebuild(solved), trusted_normalized_input=True)
    evaluator = SequenceEvaluator(config, trusted_normalized_input=True)
    assert evaluator.update(sample(config, 0, final, [], actions=0))['status'] == 'invalid'


@pytest.mark.parametrize('reason', ['budget_exhausted', 'cancelled'])
def test_nonphysical_finish_stays_incomplete(reason):
    from bench.protocol.task_referee import PublicEpisodeState, project_referee_state

    config, _ = fixture(TASK_IDS[0])
    evaluator, _ = start(config)
    result = evaluator.finish(reason)
    assert result['status'] == 'incomplete' and result['terminal'] and not result['success']
    public = project_referee_state(result, config['rules']['referee'])
    assert public['state'] == 'ongoing' and not public['terminal'] and public['recoverable']
    outcome = 'sim_timeout' if reason == 'budget_exhausted' else 'cancelled'
    assert PublicEpisodeState.model_validate({'terminal': True, 'outcome': outcome}).terminal
    assert evaluator.finish('evidence_missing') == result


def test_frozen_chinese_titles_are_public_without_changing_object_identity():
    config, _ = fixture(TASK_IDS[1])
    config['objects']['robotics']['title_zh'] = '机器人学基础'
    config = _rebuild(config)
    conditions = config['rules']['referee']['success_conditions']
    assert sum('机器人学基础（robotics）' in text for text in conditions) == 2
    assert config['objects']['robotics']['category'] == 'science'
    assert validate_configuration(config) == config


@pytest.mark.parametrize('title', ['', '  ', None, 3, '书名\n伪造规则', '书'*121])
def test_optional_frozen_chinese_titles_require_plain_text(title):
    config, _ = fixture(TASK_IDS[0])
    config['objects']['marker']['title_zh'] = title
    with pytest.raises(ValueError):
        _rebuild(config)


def test_full_task_identity_and_public_contract_must_match_the_frozen_configuration():
    config, _ = fixture(TASK_IDS[0])
    task = {'id': config['task_id'], 'public_rules': deepcopy(config['rules']),
            'evaluation': {'configuration': config}}
    assert SequenceEvaluator(task).ident == config['task_id']
    task['public_rules']['floor_drop_is_terminal'] = False
    with pytest.raises(ValueError): SequenceEvaluator(task)


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_task_owned_rules_and_evaluator_wrappers_reject_another_task(task_id):
    import importlib

    rule_entrypoint = importlib.import_module(f'scenes.workrooms.study.tasks.{task_id}.rules')
    evaluator_entrypoint = importlib.import_module(f'scenes.workrooms.study.tasks.{task_id}.evaluator')
    config, _ = fixture(task_id)
    extra = {key: config[key] for key in ('stack_order', 'retrieval_target_id', 'protected_ids') if key in config}
    assert rule_entrypoint.build_configuration(config['objects'], config['targets'], config['supports'], **extra) == config
    assert rule_entrypoint.public_rules(config) == config['rules']
    assert evaluator_entrypoint.SequenceEvaluator(config).ident == task_id
    other, _ = fixture(TASK_IDS[(TASK_IDS.index(task_id)+1) % len(TASK_IDS)])
    with pytest.raises(ValueError, match='different study task'):
        rule_entrypoint.public_rules(other)
    with pytest.raises(ValueError, match='different study task'):
        evaluator_entrypoint.SequenceEvaluator(other)
