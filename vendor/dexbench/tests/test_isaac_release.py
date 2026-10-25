"""Release/evaluator contract checks; these are not Isaac physics acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from bench.isaacsim.release import DEFAULT, SOURCE, load_evaluator, task_inputs, validate
from bench.isaacsim.run import load_actions


def test_prepared_release_preserves_frozen_task_and_scene_bytes():
    manifest = validate(DEFAULT)
    assert manifest['status'] == 'prepared_unvalidated'
    assert not manifest['task_policy_success_claimed']
    assert not manifest['physics_accepted']
    for entry in manifest['tasks']:
        for kind in ('scene', 'task'):
            assert (DEFAULT / entry[kind]).read_bytes() == (SOURCE / 'tasks' / entry['id'] / f'{kind}.json').read_bytes()
    for name in ('sequence_evaluator.py', 'geometry.py'):
        assert (DEFAULT / 'evaluator' / name).read_bytes() == (SOURCE / 'source_snapshot/bench' / name).read_bytes()


def test_changed_task_is_rejected_before_runtime(tmp_path):
    target = tmp_path / 'copy'
    shutil.copytree(DEFAULT, target)
    path = target / 'tasks/01_ordered_kitting/task.json'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Frozen input changed'):
        validate(target, assets=False)


def observation(scene, initial):
    objects = {}
    for spec in scene['objects']:
        pose = {**spec, **initial.get('objects', {}).get(spec['id'], {})}
        objects[spec['id']] = {'position_m': pose['position_m'], 'quaternion_wxyz': pose['quaternion_wxyz'],
                              'linear_velocity_m_s': [0., 0., 0.], 'angular_velocity_rad_s': [0., 0., 0.]}
    cloth = scene['cloth']
    if cloth['enabled']:
        nx, ny = cloth['cells_xy']; sx, sy = cloth['size_m']; origin = np.asarray(cloth['origin_m'])
        vertices = np.array([[i * sx / nx, j * sy / ny, 0.] for j in range(ny + 1) for i in range(nx + 1)]) + origin
    else:
        vertices = np.empty((0, 3))
    return {'time_s': 0., 'metrics': {'state_finite': True}, 'objects': objects,
            'cloth': {'vertices_m': vertices.tolist(), 'velocities_m_s': np.zeros_like(vertices).tolist()}}


@pytest.mark.parametrize('task_id', ['01', '02', '03', '04'])
def test_static_qualification_initialization_does_not_count_as_task_success(task_id):
    _, scene, task, initial = task_inputs(DEFAULT, task_id, 'qualification_probe')
    evaluator = load_evaluator(DEFAULT)(task)
    obs = observation(scene, initial)
    for index in range(61):
        obs['time_s'] = index * .02
        score = evaluator.update(obs)
        assert not score['success']
        assert score['progress'] == 0.
    obs['time_s'] += .02
    obs['metrics']['state_finite'] = False
    score = evaluator.update(obs)
    assert score['reason'].startswith('invalid_observation:') and not score['success']


def test_object_write_in_submitted_actions_is_rejected(tmp_path):
    path = tmp_path / 'actions.json'
    path.write_text(json.dumps([{'joint_positions_rad': {}, 'duration_s': .02, 'objects': {'gear_large': [0, 0, 0]}}]))
    with pytest.raises(ValueError, match='Only joint_positions_rad'):
        load_actions(path)


def test_table_support_contact_cannot_be_counted_as_hand_push():
    from bench.isaacsim.controllers import contact_sample, contact_result
    samples = []
    for index in range(101):
        obs = {'time_s': index * .02, 'objects': {'spool': {'position_m': [.6 + .0006 * index, -.1, .775],
                   'linear_velocity_m_s': [0., 0., 0.]}},
               'contacts': {'rigid': [{'actor0': '/World/Objects/spool', 'actor1': '/World/Table',
                   'collider0': '/World/Objects/spool/collision', 'collider1': '/World/Table/collision',
                   'point_count': 3, 'impulse_magnitude_sum_ns': 1., 'min_separation_m': -.0001}]}}
        samples.append(contact_sample(obs))
    report = contact_result(samples, 'push')
    assert report['final_xy_displacement_m'] >= .05
    assert not report['checks']['hand_spool_solver_contact']
    assert not report['provisional_contact_push_response']


def test_hand_spool_pair_must_have_nonzero_solved_impulse():
    from bench.isaacsim.controllers import contact_sample
    report = {'actor0': '/World/Robot/right_index_DP', 'actor1': '/World/Objects/spool',
              'collider0': '/World/Robot/right_index_DP/collision', 'collider1': '/World/Objects/spool/collision',
              'point_count': 1, 'impulse_magnitude_sum_ns': .02, 'min_separation_m': -.0001}
    obs = {'time_s': .02, 'objects': {'spool': {'position_m': [.62, -.1, .775], 'linear_velocity_m_s': [0., 0., 0.]}},
           'contacts': {'rigid': [report]}}
    sample = contact_sample(obs)
    assert sample['hand_object_contact_points'] == 1
    assert sample['hand_object_impulse_ns'] == .02
    report['actor0'] = '/World/Robot/fr3_link7'
    report['collider0'] = '/World/Robot/fr3_link7/collision'
    assert contact_sample(obs)['hand_object_contact_points'] == 0
