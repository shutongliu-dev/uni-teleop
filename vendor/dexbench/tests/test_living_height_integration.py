"""Synthetic native-format evidence across scene scoring and public contracts.

No simulator, shared assets, model call or manipulation qualification is used.
"""
from copy import deepcopy
import importlib.util
import json
import math
import sys

import pytest

from bench.paths import DATA_ROOT, load_json
from bench.protocol.task_referee import (
    PublicTaskRequirements, project_referee_state, project_task_rules,
)
from scenes.workrooms.living_room.evaluation import SequenceEvaluator
from scenes.workrooms.living_room.height_public import bind_height_rules
from scenes.workrooms.living_room.layer_release import _freeze_evaluator
from scenes.workrooms.living_room.native_evidence import normalized_contacts
from scenes.workrooms.living_room.tasks.tower_building.task import task_configuration
from scenes.workrooms.living_room.tower_trays import (
    BUILDING_GEOMETRY, building_supply_layout, building_supply_trays,
)

DT = .0005
TABLE_COLLIDER = '/World/Room/Geometry/worktop/Collider'


def task_spec():
    config = task_configuration(BUILDING_GEOMETRY, scoring_mode='freeform_height',
        inventory_count=36, supply_trays=building_supply_trays(), supply_layout=building_supply_layout())
    actor_paths = {}
    for tray in config['supply_trays']:
        root = '/World/LivingTask/Trays/'+tray['id']
        for actor in tray['contact_actor_ids']:
            actor_paths[actor] = root+('/Walls/'+actor.rsplit('wall_', 1)[1] if '/wall_' in actor else '/Floor')
    return bind_height_rules({'id': 'tower_building', 'schema_version': 'living-room-task-v1',
        'qualification': {'scope': 'synthetic_native_format_only'}, 'public_rules': deepcopy(config['rules']),
        'evaluation': {'reference_configuration': config, 'physics_dt_s': DT, 'sample_period_s': .02,
            'body_paths': {name: '/World/LivingTask/Blocks/'+name for name in config['block_ids']},
            'contact_actor_paths': actor_paths}})


def raw_contact(task, supported, other, point, *, reverse=False):
    bodies, fixtures = task['evaluation']['body_paths'], task['evaluation']['contact_actor_paths']
    a = bodies[supported]+'/Collision'
    b = TABLE_COLLIDER if other == 'table' else bodies.get(other, fixtures.get(other))
    impulse = [0., 0., .1*DT]
    if reverse:
        a, b, impulse = b, a, [0., 0., -.1*DT]
    return {'a': a, 'b': b, 'point_m': list(point), 'impulse_on_a_n_s': impulse, 'separation_m': 0.}


def scene_state(task, *, standing=True):
    config = task['evaluation']['reference_configuration']
    states = deepcopy(config['initial_objects'])
    if standing:
        # One upright block off the old target center proves there is no hidden
        # required layer count, yaw, target slot, or all-inventory placement.
        states['block_35'] = {'position_m': [.67, .28, .855],
            'quaternion_wxyz': [math.sqrt(.5), 0., math.sqrt(.5), 0.]}
    contacts = []
    for index, (name, state) in enumerate(states.items()):
        if standing and name == 'block_35':
            contacts.append(raw_contact(task, name, 'table', [.67, .28, .78], reverse=True))
            continue
        other = ('external/supply_left' if index < 18 else 'external/supply_right')
        if index % 18 >= 9:
            other = f'block_{index-9:02d}'
        x, y, z = state['position_m']
        contacts.append(raw_contact(task, name, other, [x, y, z-.0125], reverse=bool(index % 2)))
    return states, contacts


def frame(task, step, states=None, contacts=None):
    if states is None:
        states, contacts = scene_state(task, standing=False)
    objects = {name: {**deepcopy(state), 'linear_velocity_m_s': [0., 0., 0.],
        'angular_velocity_rad_s': [0., 0., 0.]} for name, state in states.items()}
    return {'schema': 'living_native_evidence_v1', 'complete': True,
        'episode_id': 'synthetic_cross_boundary', 'step_index': step, 'time_s': step*DT,
        'objects': objects, 'joints': {}, 'particle_sets': {},
        'raw_contacts': deepcopy(contacts or []), 'robot_action_count': int(step > 0)}


def establish_height(task, evaluator):
    assert evaluator.update(frame(task, 0))['score'] == 0.
    states, contacts = scene_state(task)
    for step in range(1, 2081):
        result = evaluator.update(frame(task, step, states, contacts))
        assert result['status'] != 'invalid', result
        if step < 2080:
            assert result['score'] == 0.
    assert result['score'] == pytest.approx(.15)
    return states, contacts


@pytest.fixture(scope='module')
def recorded_height():
    task = task_spec()
    evaluator = SequenceEvaluator(task)
    states, contacts = establish_height(task, evaluator)
    return task, evaluator, states, contacts


def public_snapshot(task, score):
    public = project_task_rules(task)
    return PublicTaskRequirements.model_validate({**public,
        'referee': project_referee_state(score, public['public_rules']),
        'episode': {'terminal': False, 'outcome': None}}).model_dump(mode='json', by_alias=True)


def test_public_height_contract_is_bound_and_exposes_no_private_layout():
    task = task_spec()
    snapshot = public_snapshot(task, SequenceEvaluator(task).inspect_conditions())
    assert snapshot['referee']['phase'] == 'building'
    assert snapshot['public_rules']['thresholds']['available_block_count'] == 36
    assert '不限层数、每层根数、朝向或塔形' in snapshot['instruction']
    encoded = json.dumps(snapshot)
    for private in ('/World/', DATA_ROOT, 'block_00', 'initial_objects', 'contact_actor_paths'):
        assert private not in encoded
    changed = deepcopy(task)
    changed['public_rules']['referee']['thresholds']['available_block_count'] = 18
    with pytest.raises(ValueError, match='frozen task identity'):
        project_task_rules(changed)


def test_native_substeps_produce_persistable_height_score_and_coarse_public_record(recorded_height, tmp_path):
    task, evaluator, _, _ = recorded_height
    result = evaluator.inspect_conditions()
    assert result['score_unit'] == 'm' and result['score_valid']
    assert result['current_height_m'] == result['best_height_m'] == pytest.approx(.15)
    assert result['best_height_cm'] == pytest.approx(15.)
    assert result['stable_duration_s'] == pytest.approx(1.)
    assert not result['success'] and not result['physics_accepted']
    assert result['reference_result']['best_stable_block_ids'] == ['block_35']
    path = tmp_path/'score.json'
    path.write_text(json.dumps(result))
    saved = load_json(path)
    assert saved['score'] == pytest.approx(.15)
    snapshot = public_snapshot(task, saved)
    assert snapshot['referee']['phase'] == 'height_recorded'
    assert snapshot['referee']['state'] == 'ongoing'
    assert 'best_height_m' not in json.dumps(snapshot)


def test_native_collapse_retains_best_score_without_binary_loss(recorded_height):
    task, evaluator, states, contacts = deepcopy(recorded_height)
    states['block_35']['position_m'] = [.67, .28, .075]
    floor = raw_contact(task, 'block_35', 'table', [.67, .28, 0.])
    floor['b'] = '/World/Room/Geometry/floor/Collider'
    contacts[-1] = floor
    for step in range(2081, 2121):
        result = evaluator.update(frame(task, step, states, contacts))
    assert result['status'] == 'incomplete' and not result['success']
    assert result['score'] == pytest.approx(.15) and result['score_valid']
    assert result['current_height_m'] == result['stable_duration_s'] == 0.
    assert public_snapshot(task, result)['referee']['phase'] == 'height_recorded'


@pytest.mark.parametrize('defect', ['missing_substep', 'missing_supply', 'nonfinite_supply'])
def test_native_fault_retains_record_but_invalidates_episode_evidence(recorded_height, defect):
    task, evaluator, states, contacts = deepcopy(recorded_height)
    bad = frame(task, 2082 if defect == 'missing_substep' else 2081, states, contacts)
    if defect == 'missing_supply': del bad['objects']['block_00']
    if defect == 'nonfinite_supply': bad['objects']['block_00']['position_m'][0] = float('nan')
    result = evaluator.update(bad)
    assert result['status'] == 'invalid' and not result['score_valid']
    assert result['score'] == result['best_height_m'] == pytest.approx(.15)
    snapshot = public_snapshot(task, result)
    assert snapshot['referee']['state'] == 'evaluator_invalid'
    assert snapshot['referee']['terminal'] and not snapshot['referee']['recoverable']
    assert evaluator.update(frame(task, 2082, states, contacts)) == result


def test_one_missing_native_support_sample_restarts_dwell_without_erasing_record(recorded_height):
    task, evaluator, states, contacts = deepcopy(recorded_height)
    for step in range(2081, 2121):
        result = evaluator.update(frame(task, step, states, contacts[:-1] if step == 2090 else contacts))
    assert result['score'] == pytest.approx(.15)
    assert result['current_height_m'] == result['stable_duration_s'] == 0.
    assert not result['reference_result']['blocks']['block_35']['supported_to_table']


@pytest.mark.parametrize('spoof', ['/World/LivingTask/table_fake/Collider',
    '/World/Props/tabletop_fixture/Collider', '/World/Room/Geometry/worktop_extra/Collider'])
def test_only_the_actual_room_worktop_can_anchor_height_credit(recorded_height, spoof):
    task, evaluator, states, contacts = deepcopy(recorded_height)
    # The raw pair is reversed; changing actor a must preserve impulse signs.
    contacts[-1]['a'] = spoof
    mapped = normalized_contacts([contacts[-1]], task['evaluation']['body_paths'],
        contact_actor_paths=task['evaluation']['contact_actor_paths'])
    assert mapped[0]['a'].startswith('external/')
    for step in range(2081, 2121):
        result = evaluator.update(frame(task, step, states, contacts))
    assert result['current_height_m'] == 0. and result['score'] == pytest.approx(.15)
    assert not result['reference_result']['blocks']['block_35']['supported_to_table']


def test_zero_force_native_tray_wall_touch_excludes_connected_height(recorded_height):
    task, evaluator, states, contacts = deepcopy(recorded_height)
    wall = task['evaluation']['contact_actor_paths']['external/supply_right/wall_neg_x']
    contacts.append({'a': task['evaluation']['body_paths']['block_35']+'/Collision',
        'b': wall, 'point_m': [.67, .28, .855], 'impulse_on_a_n_s': [0., 0., 0.], 'separation_m': 0.})
    for step in range(2081, 2121):
        result = evaluator.update(frame(task, step, states, contacts))
    assert result['current_height_m'] == 0. and result['score'] == pytest.approx(.15)
    assert 'tray_robot_floor_or_external_contact' in result['reference_result']['blocks']['block_35']['reasons']


def test_frozen_height_evaluator_does_not_call_live_task_or_scoring_helpers(tmp_path, monkeypatch):
    from scenes.workrooms.living_room import height_public
    from scenes.workrooms.living_room.tasks.tower_building import freeform, task as live_task
    task = task_spec()
    directory = tmp_path/'evaluator'
    _freeze_evaluator(directory)
    assert (directory/'tasks/tower_building/freeform.py').is_file()
    assert (directory/'height_public.py').is_file()
    assert 'from scenes.workrooms.living_room.' not in (directory/'tasks/tower_building/rules.py').read_text()

    def live_call_forbidden(*args, **kwargs):
        raise AssertionError('Frozen scoring called a live task or height helper')

    monkeypatch.setattr(live_task, 'task_configuration', live_call_forbidden)
    monkeypatch.setattr(freeform, 'update_height', live_call_forbidden)
    monkeypatch.setattr(height_public, 'height_verdict', live_call_forbidden)
    name = '_test_frozen_living_height'
    spec = importlib.util.spec_from_file_location(name, directory/'sequence_evaluator.py',
        submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    evaluator = module.SequenceEvaluator(task)
    establish_height(task, evaluator)
    assert evaluator.inspect_conditions()['score'] == pytest.approx(.15)
    assert evaluator.tower.__class__.__module__.startswith(name+'.tasks.tower_building.')
