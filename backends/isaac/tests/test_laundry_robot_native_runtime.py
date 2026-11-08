"""CPU evidence-boundary counterexamples; these do not certify native cloth."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bench.evaluation.cloth.cloth_robot_collision import SCHEMA, SUPPORT_SCHEMA
from bench.simulation.isaacsim_backend.env import named_cloth_initialization
from scenes.workrooms.laundry_room.native_observer import CookedRobotGeometry
from tests.test_surface_cloth_collection import make_env, spec


@pytest.mark.parametrize('path', ['/physicsScene', '/World/CustomPhysics'])
@pytest.mark.parametrize('valid', [True, False])
def test_native_observer_uses_world_physics_context_path(monkeypatch, path, valid):
    import sys
    from scenes.workrooms.laundry_room import native_observer

    class BoundNativeScene(Exception):
        pass

    class Scene:
        def __init__(self, prim):
            assert prim is native_prim
            raise BoundNativeScene

    native_prim = SimpleNamespace(IsA=lambda cls: cls is Scene)
    stage = SimpleNamespace(GetPrimAtPath=lambda requested: native_prim if valid and requested == path else None)
    env = SimpleNamespace(dt=.0005, scene_info={'cloths': {}}, cloth_names=[], cloth_triangles={},
        device='cuda:0', body_names=[], stage=stage,
        world=SimpleNamespace(get_physics_context=lambda: SimpleNamespace(prim_path=path)))
    monkeypatch.setitem(sys.modules, 'pxr', SimpleNamespace(
        UsdPhysics=SimpleNamespace(Scene=Scene), PhysxSchema=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, 'bench.evaluation.cloth.gpu_intersections',
        SimpleNamespace(ClothIntersectionAudit=lambda *args, **kwargs: None))
    monkeypatch.setattr(native_observer, 'collect_colliders', lambda env: {})
    monkeypatch.setattr(native_observer, 'CookedRobotGeometry', lambda *args, **kwargs: None)
    monkeypatch.setattr(native_observer, 'wrist_frame_bindings', lambda *args: {})
    task = {'evaluation': {'control_mode': 'robot_joint_targets', 'physics_dt_s': .0005,
        'thresholds': {'max_penetration_m': .002, 'hard_penetration_m': .01}},
        'asset_metadata': {'cloths': []}}
    with pytest.raises(BoundNativeScene if valid else ValueError):
        native_observer.NativeObserver(env, {'named_cloth_initialization': 'author-only'}, task)


def clock_observer():
    from scenes.workrooms.laundry_room.native_observer import NativeObserver
    observer = NativeObserver.__new__(NativeObserver)
    observer.env = SimpleNamespace(dt=.0005, body_names=[])
    observer.native_baseline_step = 2
    observer.native_baseline_time = float(np.float32(.0005))*2
    observer.previous_step_index = -1
    observer.previous_native_time = observer.native_baseline_time
    observer.specs = observer.topology = observer.static_evidence = {}
    observer.measure_crossings = False
    observer.read_physics = lambda: {}
    observer.oracle = SimpleNamespace(check=lambda vertices: {
        'finite_thickness_compression_checked': True, 'degenerate_triangles': 0,
        'self_intersections': 0, 'intercloth_intersections': 0,
        'max_cloth_envelope_penetration_m': 0.})
    observer.robot = SimpleNamespace(check=lambda *args: {
        'robot_cloth_penetration_m': 0., 'robot_cloth_geometry_uncertain': False})
    observer.scene = {'table': {'position_m': [0., 0., 0.], 'half_extents_m': [1., 1., 1.]}}
    return observer


def clock_state(index):
    return {'native_physics_step_index': index+2,
        'native_world_time_s': float(np.float32(.0005))*(index+2),
        'body_q': np.empty((0, 7))}


def test_native_clock_accepts_1601_float32_step_callbacks_without_accumulated_false_alarm():
    observer = clock_observer()
    for index in range(1601):
        frame = observer.frame(clock_state(index), step_index=index, time_s=index*.0005)
        assert frame['native_step'] == index and frame['physics']['valid']
    assert abs(observer.previous_native_time-observer.native_baseline_time-.8) > 1.e-8


@pytest.mark.parametrize('fault', ['repeated', 'skipped', 'native_step', 'native_time', 'public_time'])
def test_native_clock_still_rejects_missing_repeated_and_mixed_tick_samples(fault):
    observer = clock_observer()
    observer.frame(clock_state(0), step_index=0, time_s=0.)
    index = {'repeated': 0, 'skipped': 2}.get(fault, 1)
    state, time_s = clock_state(index), index*.0005
    if fault == 'native_step':
        state['native_physics_step_index'] += 1
    if fault == 'native_time':
        state['native_world_time_s'] += .0005
    if fault == 'public_time':
        time_s += .0005
    with pytest.raises(ValueError, match='mixed-tick'):
        observer.frame(state, step_index=index, time_s=time_s)


def test_author_only_preserves_native_warmup_q_and_qd_without_setters():
    env = make_env([spec(), spec('towel_base')])
    env.scene_spec['named_cloth_initialization'] = 'author-only'
    env.world = SimpleNamespace(current_time=.001, current_time_step_index=2)
    for view in env._cloth_views.values():
        view.points[:, :, 2] -= .00003
        view.set_simulation_nodal_positions = lambda *_: pytest.fail('unexpected q reset')
        view.set_simulation_nodal_velocities = lambda *_: pytest.fail('unexpected qd reset')
    before = {name: (view.points.copy(), view.velocities.copy()) for name, view in env._cloth_views.items()}
    validated = env._validated_named_cloth_reset({})
    env._reset_named_cloths(validated)
    for name, view in env._cloth_views.items():
        assert np.array_equal(view.points, before[name][0])
        assert np.array_equal(view.velocities, before[name][1])
    assert env.named_cloth_initialization_receipt['native_warmup_preserved'] is True
    assert env.named_cloth_initialization_receipt['position_write_calls'] == 0


def test_author_only_rejects_changed_initials_and_reused_environments_before_writing():
    env = make_env([spec()])
    env.scene_spec['named_cloth_initialization'] = 'author-only'
    with pytest.raises(ValueError, match='authored scene'):
        env._validated_named_cloth_reset({'towel': (env._cloth_initials['towel']+[0, 0, .1]).tolist()})
    env._reset_count = 1
    with pytest.raises(ValueError, match='fresh environment'):
        env.reset()
    with pytest.raises(ValueError, match='named cloths'):
        named_cloth_initialization({'named_cloth_initialization': 'author-only'})
    with pytest.raises(ValueError, match='must be'):
        named_cloth_initialization({'named_cloth_initialization': 'guess'})
    assert named_cloth_initialization({}) == 'tensor-reset'


def geometry():
    names = ['left_hand_flange', 'right_hand_flange', 'left_hand_C_MC', 'right_hand_C_MC', 'fr3_left_link4']
    planes = np.array([[1, 0, 0, -.01], [-1, 0, 0, -.01], [0, 1, 0, -.01],
                       [0, -1, 0, -.01], [0, 0, 1, -.01], [0, 0, -1, -.01]])
    vertices = np.array([[x, y, z] for x in (-.01, .01) for y in (-.01, .01) for z in (-.01, .01)])
    piece = {'geometry_representation': SUPPORT_SCHEMA,
             'conservative_outer_vertices': vertices.tolist(),
             'conservative_outer_planes': planes.tolist(), 'conservative_inner_planes': planes.tolist()}
    export = {'schema': SCHEMA, 'status': 'complete', 'issues': [], 'visual_mesh_fallback': False,
              'state_hash_before': 'a'*64, 'state_hash_after': 'a'*64,
              'layer_hashes_before': {'root': 'b'*64}, 'layer_hashes_after': {'root': 'b'*64},
              'colliders': [{'status': 'verified_cooking_representation', 'path': '/World/Robot/'+name,
                 'body_name': name, 'mesh_to_body_column_matrix': np.eye(4).tolist(),
                 'pieces': [deepcopy(piece)]} for name in names[2:]]}
    topology = {'towel': np.array([[0, 1, 2]], dtype=np.int32)}
    poses = np.tile([1., 1., 1., 0., 0., 0., 1.], (len(names), 1))
    return export, names, topology, poses


def towel(z):
    return {'towel': np.array([[-.005, -.005, z], [.005, -.005, z], [0, .005, z]])}


def test_cooked_native_hand_pose_controls_near_release_and_preserves_input():
    export, names, topology, poses = geometry()
    auditor = CookedRobotGeometry(export, names, topology, {'towel': .003})
    vertices = towel(.014)
    before = deepcopy(vertices)
    assert auditor.check(vertices, poses)['fully_separated'] is True
    poses[2, :3] = 0
    result = auditor.check(vertices, poses)
    assert result['hands']['left']['near_towel'] is True
    assert result['hands']['left']['wrist_body_name'] == 'left_hand_flange'
    assert result['hands']['left']['wrist_position_m'] == poses[0, :3].tolist()
    assert result['hands']['right']['near_towel'] is False
    assert result['fully_separated'] is False
    assert result['robot_cloth_penetration_m'] == 0
    assert result['contact_force_measured'] is False
    assert np.array_equal(vertices['towel'], before['towel'])
    assert auditor.check(towel(.018), poses)['fully_separated'] is True


def test_arm_support_blocks_release_without_inventing_hand_interaction():
    export, names, topology, poses = geometry()
    auditor = CookedRobotGeometry(export, names, topology, {'towel': .003})
    poses[4, :3] = 0
    result = auditor.check(towel(.014), poses)
    assert result['fully_separated'] is False
    assert result['hand_near_cloth'] is False
    poses[2, :3] = 0
    result = auditor.check(towel(0), poses)
    assert result['robot_cloth_penetration_m'] >= .008
    assert result['robot_cloth_penetration_checked'] is True


def test_cooked_geometry_missing_or_unverified_shape_never_passes():
    export, names, topology, poses = geometry()
    export['colliders'] = export['colliders'][:1]
    with pytest.raises(ValueError, match='Both original hands'):
        CookedRobotGeometry(export, names, topology, {'towel': .003})
    export, names, topology, poses = geometry()
    export['status'] = 'unchecked'
    with pytest.raises(ValueError, match='Complete unchanged'):
        CookedRobotGeometry(export, names, topology, {'towel': .003})


def runtime_fixture(monkeypatch, tmp_path):
    from tests.test_study_native_runtime import Environment
    from scenes.workrooms.laundry_room import native_observer
    from scenes.workrooms.laundry_room.runtime import LaundryTaskRuntime

    class Observer:
        topology = {'towel': [[0, 1, 2]]}
        export = {'test_double': True}
        def __init__(self, env, scene, task):
            self.env = env
        def frame(self, state, *, step_index, time_s):
            return {'complete': True, 'native_step': step_index, 'time_s': time_s,
                    'cloths': {}, 'controlled_vertices': {'towel': []}}
        def close(self):
            return {'test_double': True}

    class Evaluator:
        def __init__(self, task):
            self.frames = []
            self.invalid = False
        def observe(self, frame):
            self.frames.append(deepcopy(frame))
            self.invalid |= frame.get('complete') is not True
        def report(self):
            return {'success': False, 'referee': {'state': 'evaluator_invalid' if self.invalid else 'ongoing',
                                                'terminal': self.invalid}}

    monkeypatch.setattr(native_observer, 'NativeObserver', Observer)
    env = Environment()
    task = {'schema_version': 'laundry-cloth-task-v1', 'capabilities': {'robot_episodes': True},
            'evaluation': {'control_mode': 'robot_joint_targets'}}
    return env, LaundryTaskRuntime(env, {}, task, tmp_path, evaluator_factory=Evaluator)


def test_laundry_runtime_records_every_existing_native_step_and_joint_action(monkeypatch, tmp_path):
    import io
    import json
    import zipfile
    env, runtime = runtime_fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='active ordinary'):
        runtime.authorize_step({})
    runtime.begin_action('one')
    with pytest.raises(ValueError, match='no players'):
        runtime.authorize_step({}, 'left')
    runtime.authorize_step({})
    for _ in range(3):
        runtime.begin_action('one')
        env.tick()
    assert runtime.actions == 1 and runtime.steps == 3 and env.contact_reads == 3
    assert [row['native_step'] for row in runtime.evaluator.frames] == [0, 1, 2, 3]
    runtime.begin_action('two')
    with pytest.raises(ValueError, match='cannot resume'):
        runtime.begin_action('one')
    receipt = runtime.close()
    assert receipt['complete'] is True
    assert receipt['contact_exchange']['all_samples_consumed_once'] is True
    assert receipt['native_steps'] == 3
    assert runtime.close() is None
    rows = [json.loads(line) for line in (tmp_path/'laundry_native_frames.jsonl').read_text().splitlines()]
    assert [row['native_step'] for row in rows] == [0, 1, 2, 3]
    with zipfile.ZipFile(tmp_path/'laundry_native_evidence.zip') as archive:
        states = np.load(io.BytesIO(archive.read('states/000000.npz')), allow_pickle=False)
        assert states['native_physics_step_index'].tolist() == [10, 11, 12, 13]


def test_laundry_readonly_violation_invalidates_score_and_native_receipt(monkeypatch, tmp_path):
    env, runtime = runtime_fixture(monkeypatch, tmp_path)
    original = runtime.observer.frame
    def corrupt(state, **kwargs):
        env.object_q[0, 0] += .01
        return original(state, **kwargs)
    runtime.observer.frame = corrupt
    runtime.begin_action('bad')
    with pytest.raises(RuntimeError, match='changed native physical state'):
        env.tick()
    assert runtime.score()['success'] is False
    assert runtime.score()['referee']['state'] == 'evaluator_invalid'
    receipt = runtime.close(reason='runtime_error')
    assert receipt['complete'] is False
