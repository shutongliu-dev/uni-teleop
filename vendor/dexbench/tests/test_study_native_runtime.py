"""CPU native-adapter contracts; these doubles do not qualify task physics."""
from copy import deepcopy
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scenes.runtime import build_task_runtime
from scenes.workrooms.study.native_observer import NativeObserver, configuration_sha256, raw_contacts
from scenes.workrooms.study.runtime import StudyTaskRuntime


BODY = '/World/StudyTask/Objects/book'
ROBOT = '/World/Robot/fr3/link1'
SUPPORTS = {
    'table': '/World/Room/Geometry/worktop/Collider',
    'floor': '/World/Room/Geometry/room_floor/Collider',
    'shelf_lower': '/World/Room/Geometry/rear_shelf_level_1/Collider',
    'shelf_upper': '/World/Room/Geometry/rear_shelf_level_2/Collider',
}


def spec():
    configuration = {'schema': 'test_configuration', 'rules': {}, 'objects': {'book': {}}}
    configuration['configuration_sha256'] = configuration_sha256(configuration)
    return {'schema_version': 'study-task-v1', 'id': 'book_shelving',
        'evaluation': {'physics_dt_s': .0005, 'configuration': configuration,
            'body_paths': {'book': BODY}, 'support_paths': deepcopy(SUPPORTS)}}


class Evaluator:
    def __init__(self, task, *, trusted_normalized_input=False):
        assert trusted_normalized_input is True
        self.frames = []
        self.reason = None

    def update(self, frame):
        self.frames.append(deepcopy(frame))

    def inspect_conditions(self):
        return {'success': False, 'referee': {'state': 'ongoing'}, 'reason': self.reason}

    def finish(self, reason='budget_exhausted'):
        self.reason = reason


class Environment:
    def __init__(self):
        self.dt = .0005
        self.time_s = self._current_step_time = 0.
        self._reset_count = 1
        self._contact_exchange = None
        self.scene_info = {'objects': {'book': {'path': BODY}}, 'robot_link_paths': {'link1': ROBOT}}
        self._contact_sensor_paths = [ROBOT, BODY]
        self._contact_filter_paths = [[ROBOT, BODY, *SUPPORTS.values()]] * 2
        self.arrays = [np.zeros((8, 1)), np.zeros((8, 3)), np.zeros((8, 3)), np.zeros((8, 1)),
            np.zeros((2, 6), dtype=int), np.zeros((2, 6), dtype=int)]
        self.contact_reads = 0
        self._rigid_contacts = SimpleNamespace(max_contact_data_count=8, get_contact_data=self.contacts)
        self.world = SimpleNamespace(current_time_step_index=10, current_time=.25, step=self.advance)
        self.object_q = np.array([[.65, 0., .80, 0., 0., 0., 1.]])

    def contacts(self, *, dt):
        assert dt == 1.
        self.contact_reads += 1
        return self.arrays

    def advance(self, *, render):
        assert render is False
        self.world.current_time_step_index += 1
        self.world.current_time += self.dt

    def capture_state(self):
        return {'object_q': self.object_q.copy(), 'object_qd': np.zeros((1, 6)),
            'native_physics_step_index': np.asarray(self.world.current_time_step_index),
            'native_world_time_s': np.asarray(self.world.current_time)}

    def task_runtime_state(self, state):
        return {'schema': 'bench_task_runtime_v1', 'sampling': {'complete': True, 'physics_steps_added': 0},
            'native_physics_step_index': int(state['native_physics_step_index']),
            'native_world_time_s': float(state['native_world_time_s']),
            'objects': {'book': {'path': BODY, 'position_m': state['object_q'][0, :3].tolist(),
                'quaternion_wxyz': [1., 0., 0., 0.], 'linear_velocity_m_s': [0., 0., 0.],
                'angular_velocity_rad_s': [0., 0., 0.]}}, 'joints': {}, 'particle_sets': {}}

    def _project_rigid_contacts(self, sample):
        sample.for_env(self)
        return []

    def tick(self):
        self._current_step_time = self.time_s + self.dt
        self.world.step(render=False)
        self.time_s = self._current_step_time
        self._contact_exchange.consume(self)


def test_runtime_uses_each_existing_substep_once_and_preserves_state(tmp_path):
    env = Environment()
    before = env.capture_state()
    original = env.world.step
    runtime = build_task_runtime(env, {}, spec(), tmp_path, evaluator_factory=Evaluator)
    assert isinstance(runtime, StudyTaskRuntime)
    assert runtime.public_state() is None
    assert len(runtime.evaluator.frames) == 1
    baseline = runtime.evaluator.frames[0]
    assert baseline['step_index'] == baseline['time_s'] == baseline['robot_action_count'] == 0
    assert baseline['contacts'] == [] and baseline['objects']['book']['position_m'] == [.65, 0., .8]
    runtime.begin_action('first')
    runtime.authorize_step({'joint_positions_rad': {}}, None)
    for _ in range(3):
        env.tick()
    assert env.contact_reads == 3
    assert [f['step_index'] for f in runtime.evaluator.frames] == [0, 1, 2, 3]
    assert np.array_equal(before['object_q'], env.capture_state()['object_q'])
    receipt = runtime.close()
    assert receipt['acquisitions'] == receipt['deliveries'] == 3
    assert receipt['all_samples_consumed_once'] is True
    assert env.world.step == original and runtime.evaluator.reason == 'budget_exhausted'
    assert runtime.close() is None
    with zipfile.ZipFile(tmp_path / 'study_native_evidence.zip') as archive:
        assert len(archive.namelist()) == 4
        assert json.loads(archive.read('000000003.json'))['robot_action_count'] == 1


def test_public_actions_count_once_across_native_commands_and_reject_reuse(tmp_path):
    env = Environment()
    runtime = StudyTaskRuntime(env, {}, spec(), tmp_path, evaluator_factory=Evaluator)
    with pytest.raises(ValueError, match='action identity'):
        runtime.authorize_step({}, None)
    for invalid in (None, '', ' ', 1):
        with pytest.raises(ValueError, match='action identity'):
            runtime.begin_action(invalid)
    for _ in range(15):
        runtime.begin_action('hold')
        runtime.authorize_step({}, None)
        env.tick()
    assert runtime.actions == 1
    assert {frame['robot_action_count'] for frame in runtime.evaluator.frames[1:]} == {1}
    runtime.begin_action('move')
    runtime.authorize_step({}, None)
    env.tick()
    assert runtime.actions == runtime.evaluator.frames[-1]['robot_action_count'] == 2
    with pytest.raises(ValueError, match='cannot resume'):
        runtime.begin_action('hold')
    runtime.evaluator.inspect_conditions = lambda: {'terminal': True, 'status': 'incomplete'}
    with pytest.raises(RuntimeError, match='terminal'):
        runtime.authorize_step({}, None)
    with pytest.raises(RuntimeError, match='terminal'):
        runtime.begin_action('extra')
    assert runtime.actions == 2
    runtime.close()


def test_native_admission_rejects_action_above_frozen_limit(tmp_path):
    task = spec()
    task['evaluation']['configuration']['budget'] = {'max_robot_actions': 1}
    configuration = task['evaluation']['configuration']
    configuration['configuration_sha256'] = configuration_sha256(configuration)
    env = Environment()
    runtime = StudyTaskRuntime(env, {}, task, tmp_path, evaluator_factory=Evaluator)
    for _ in range(3):
        runtime.begin_action('last_allowed')
        runtime.authorize_step({}, None)
        env.tick()
    with pytest.raises(RuntimeError, match='action budget'):
        runtime.begin_action('over_budget')
    assert runtime.actions == 1 and runtime.steps == 3
    runtime.close()


@pytest.mark.parametrize('fault', ['mixed_tick', 'incomplete', 'nonfinite', 'robot_nonfinite', 'missing_body', 'pose_write'])
def test_bad_native_evidence_stops_without_task_credit(tmp_path, fault):
    env = Environment()
    runtime = StudyTaskRuntime(env, {}, spec(), tmp_path, evaluator_factory=Evaluator)
    original_payload = env.task_runtime_state

    def corrupted(state):
        result = original_payload(state)
        if fault == 'mixed_tick':
            result['native_physics_step_index'] -= 1
        elif fault == 'incomplete':
            result['sampling']['complete'] = False
        elif fault == 'missing_body':
            result['objects'].clear()
        elif fault == 'nonfinite':
            result['objects']['book']['position_m'][0] = float('nan')
        elif fault == 'pose_write':
            env.object_q[0, 0] += .1
        return result

    env.task_runtime_state = corrupted
    if fault == 'robot_nonfinite':
        capture = env.capture_state
        env.capture_state = lambda: {**capture(), 'joint_q': np.array([float('nan')])}
    with pytest.raises((ValueError, RuntimeError)):
        env.tick()
    assert runtime.evaluator.reason == 'evidence_missing'
    assert len(runtime.evaluator.frames) == 1 and runtime.score()['success'] is False
    receipt = runtime.close('runtime_error')
    assert receipt['all_samples_consumed_once'] is False


def test_native_mapping_rejects_missing_support_and_changed_inventory():
    env = Environment()
    env._contact_filter_paths[1] = env._contact_filter_paths[1][:-1]
    with pytest.raises(ValueError, match='omit required support'):
        NativeObserver(env, spec())
    env = Environment()
    env.scene_info['objects']['hidden'] = {'path': '/World/Hidden'}
    with pytest.raises(ValueError, match='inventory'):
        NativeObserver(env, spec())
    changed = spec()
    changed['evaluation']['configuration']['objects']['changed'] = {}
    with pytest.raises(ValueError, match='configuration hash'):
        NativeObserver(Environment(), changed)


def test_explicit_room_support_roles_and_unknown_props_are_never_guessed():
    observer = NativeObserver(Environment(), spec())
    for role, path in SUPPORTS.items():
        assert observer.actor(path) == role
    assert observer.actor(ROBOT) == 'robot/link1'
    assert observer.actor(BODY + '/Collider') == 'book'
    for path in ('/World/Workroom/Geometry/worktop/Collider', SUPPORTS['table'] + '_extra',
                 '/World/Props/table', '/World/Room/Geometry/rear_shelf_level_3/Collider'):
        assert observer.actor(path).startswith('external/')


def test_native_signed_contacts_deduplicate_and_reject_disagreement_or_full_buffer():
    arrays = [np.array([[.2], [-.2], [0.]]), np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 0.]]),
        np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 0.]]), np.zeros((3, 1)),
        np.ones((2, 1), dtype=int), np.array([[0], [1]])]
    rows = raw_contacts(arrays, ['a', 'b'], [['b'], ['a']])
    assert rows == [{'a': 'a', 'b': 'b', 'point_m': [0., 0., 1.],
        'impulse_on_a_n_s': [0., 0., .2], 'separation_m': 0.}]
    arrays[0][1] = -.3
    with pytest.raises(ValueError, match='disagree'):
        raw_contacts(arrays, ['a', 'b'], [['b'], ['a']])
    arrays[5][1] = 2
    with pytest.raises(ValueError, match='full or overflowed'):
        raw_contacts(arrays, ['a', 'b'], [['b'], ['a']])


def test_foreign_schema_does_not_select_study_runtime(tmp_path):
    assert build_task_runtime(None, {}, {'schema_version': 'unknown'}, tmp_path,
        evaluator_factory=Evaluator) is None


@pytest.mark.parametrize('impulse_shape,separation_shape', [((3, 2), (3, 1)), ((3, 1), (3, 2)), ((0, 1), (0, 1))])
def test_native_contact_scalar_layout_must_be_complete(impulse_shape, separation_shape):
    arrays = [np.zeros(impulse_shape), np.zeros((impulse_shape[0], 3)),
        np.zeros((impulse_shape[0], 3)), np.zeros(separation_shape),
        np.zeros((1, 1), dtype=int), np.zeros((1, 1), dtype=int)]
    with pytest.raises(ValueError, match='scalar contact arrays'):
        raw_contacts(arrays, ['a'], [['b']])
