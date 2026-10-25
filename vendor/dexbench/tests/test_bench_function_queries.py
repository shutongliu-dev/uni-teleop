"""Callable public sensor/robot queries: CPU doubles, never native acceptance."""
from copy import deepcopy
import json
import pickle
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest
from pydantic import ValidationError

from bench.control.factory import public_robot_geometry
from bench.geometry import quaternion_matrix
from bench.paths import DATA_ROOT
from bench.protocol.functions import (
    FUNCTION_INPUTS, FUNCTION_OUTPUTS, RESPONSE_ADAPTER, function_schemas,
)
from bench.protocol.schemas import AgentObservation, RobotCapabilities
from bench.runtime.server import create_app
from test_agent_service import PRIVATE_MARKER, frozen_inputs, runner, session_for


def call(client, session, function, arguments=None, *, status=200):
    response = client.post(f'/episodes/{session.spec.episode_id}/call', json={
        'request_id': 'query_' + function, 'function': function,
        'arguments': {} if arguments is None else arguments})
    assert response.status_code == status, response.text
    value = response.json()
    RESPONSE_ADAPTER.validate_python(value)
    assert value['function'] == function
    assert value['ok'] is (status == 200)
    assert value['context']['episode_id'] == session.spec.episode_id
    assert value['context']['budget']['action_attempts'] == 0
    return value


def assert_public(value, session):
    serialized = json.dumps(value)
    for private in (PRIVATE_MARKER, DATA_ROOT, str(session.directory),
                    'passive_mimic', 'depth_path', 'camera_calibration',
                    'evaluator_debug', '"vertices_m"', 'render_synchronization'):
        assert private not in serialized


def assert_local_references_resolve(document):
    def visit(node):
        if isinstance(node, dict):
            refs = [node['$ref']] if '$ref' in node else []
            # Pydantic's standalone nested discriminator can contain an inline
            # schema; recurse through that too, while resolving string refs.
            refs += [value for value in node.get('discriminator', {}).get('mapping', {}).values()
                     if isinstance(value, str)]
            for ref in refs:
                assert ref.startswith('#/'), ref
                target = document
                for part in ref[2:].split('/'):
                    target = target[part.replace('~1', '/').replace('~0', '~')]
                assert isinstance(target, dict)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(document)


def test_discovery_standalone_schemas_and_openapi_resolve_all_references(runner):
    make, _, workers = runner
    api = make(depth=True)
    session, _ = session_for(api)
    before = deepcopy(workers[-1].calls)
    with TestClient(create_app(service=api)) as client:
        response = client.get('/functions/schema')
        assert response.status_code == 200
        published = response.json()
        assert published['reference'] is None
        assert published['protocol'] == 'bench_functions_v1'
        assert published['capability_version'] == '1.0.0'
        assert published['schemas'] == function_schemas()
        for schema in published['schemas'].values():
            assert_local_references_resolve(schema)
        openapi = client.get('/openapi.json').json()
        assert_local_references_resolve(openapi)
        endpoint = openapi['paths']['/episodes/{episode_id}/call']['post']
        assert endpoint['requestBody']['content']['application/json']['schema']['discriminator']['propertyName'] == 'function'
        assert endpoint['responses']['200']['content']['application/json']['schema']['discriminator']['propertyName'] == 'ok'
        catalog = client.get(f'/episodes/{session.spec.episode_id}/functions').json()
        assert catalog['reference'] is None
        entries = {entry['name']: entry for entry in catalog['functions']}
        assert set(entries) == set(FUNCTION_INPUTS) == set(FUNCTION_OUTPUTS)
        for name, entry in entries.items():
            assert entry['input_schema'] in published['schemas']
            assert entry['output_schema'] in published['schemas']
            if name in ('get_robot', 'observe', 'read_camera', 'project_pixel',
                        'transform_points', 'query_ik', 'get_hand_presets', 'hand_preset'):
                assert entry['advances_physics'] is False
                assert entry['consumes_action_attempt'] is False
                assert entry['asynchronous_action'] is False
        assert_public(catalog, session)
        assert workers[-1].calls == before


def test_read_camera_and_pixel_projection_use_published_frame_without_recapture(runner):
    make, _, workers = runner
    # Nonzero skew and 90-degree world rotation catch x/y, sign, pixel-center,
    # camera-z versus radial-range, and extrinsic convention mistakes.
    api = make(depth=True, depth_frame_updates={
        'intrinsics_3x3': [5., 1., 4., 0., 5., 3., 0., 0., 1.],
        'world_from_camera': {'frame_id': 'world', 'position_m': [1., 2., 3.],
                              'quaternion_wxyz': [2**-.5, 0., 0., 2**-.5]}})
    session, _ = session_for(api)
    before = deepcopy(workers[-1].calls)
    snapshot = session.snapshot.model_dump(mode='json')
    with TestClient(create_app(service=api)) as client:
        observed = call(client, session, 'observe')['result']
        assert observed == snapshot
        assert observed.get('privileged_object_poses') is None
        assert observed.get('privileged_cloth_vertices_m') is None
        args = {'observation_sequence': session.sequence, 'camera_id': 'overview'}
        camera = call(client, session, 'read_camera', args)['result']['camera']
        assert camera == snapshot['cameras'][0]
        projected = call(client, session, 'project_pixel', {**args, 'u': 6, 'v': 1})
        result = projected['result']
        assert result['pixel_uv'] == [6, 1]
        assert result['depth_m'] == .75
        assert result['source'] == 'public_rgbd_camera_z'
        np.testing.assert_allclose(result['point_m'], [1.225, 2.42, 3.75], atol=1e-12)
        assert result['observation_sequence'] == session.sequence
        assert result['sim_time_s'] == camera['sim_time_s'] == camera['depth_sim_time_s']
        assert_public(projected, session)
        assert_public(observed, session)
        assert workers[-1].calls == before
        assert session.snapshot.model_dump(mode='json') == snapshot
        assert session.clock.sim_time_s == 0.


@pytest.mark.parametrize('updates,status,code', [
    ({'u': 0, 'v': 0}, 422, 'invalid_depth'),
    ({'u': 8}, 422, 'invalid_arguments'),
    ({'v': 6}, 422, 'invalid_arguments'),
    ({'u': -1}, 422, 'invalid_arguments'),
    ({'u': 1.5}, 422, 'invalid_arguments'),
    ({'u': True}, 422, 'invalid_arguments'),
    ({'camera_id': 'missing'}, 404, 'not_found'),
    ({'object_id': PRIVATE_MARKER}, 422, 'invalid_arguments'),
])
def test_pixel_invalid_inputs_do_not_fabricate_measurements(runner, updates, status, code):
    make, _, workers = runner
    api = make(depth=True)
    session, _ = session_for(api)
    before = deepcopy(workers[-1].calls)
    with TestClient(create_app(service=api)) as client:
        value = call(client, session, 'project_pixel', {
            'observation_sequence': session.sequence, 'camera_id': 'overview',
            'u': 3, 'v': 2, **updates}, status=status)
        assert value['error']['code'] == code
        assert 'result' not in value
        assert_public(value, session)
        assert workers[-1].calls == before


def test_depth_unavailable_is_explicit_in_discovery_and_call(runner):
    make, _, _ = runner
    api = make()
    session, _ = session_for(api)
    with TestClient(create_app(service=api)) as client:
        catalog = client.get(f'/episodes/{session.spec.episode_id}/functions').json()
        entry = next(item for item in catalog['functions'] if item['name'] == 'project_pixel')
        assert entry['available'] is False and entry['unavailable_reason']
        value = call(client, session, 'project_pixel', {
            'observation_sequence': session.sequence, 'camera_id': 'overview', 'u': 2, 'v': 2}, status=422)
        assert value['error']['code'] == 'unsupported_capability'


@pytest.mark.parametrize('function,args', [
    ('read_camera', {'camera_id': 'overview'}),
    ('project_pixel', {'camera_id': 'overview', 'u': 2, 'v': 2}),
    ('transform_points', {'source_frame': 'camera:overview', 'target_frame': 'world',
                          'points_m': [[0., 0., 1.]]}),
])
@pytest.mark.parametrize('state', ['stale', 'busy'])
def test_snapshot_bound_queries_reject_stale_or_in_motion_observations(runner, monkeypatch, function, args, state):
    make, _, workers = runner
    api = make(depth=True)
    session, _ = session_for(api)
    before = deepcopy(workers[-1].calls)
    with TestClient(create_app(service=api)) as client:
        with monkeypatch.context() as patch:
            if state == 'busy':
                patch.setattr(session.life, 'active_action_id', 'in_progress')
            value = call(client, session, function, {
                **args, 'observation_sequence': session.sequence + (state == 'stale')}, status=409)
            assert value['error']['code'] == ('stale_observation' if state == 'stale' else 'episode_busy')
            assert workers[-1].calls == before


@pytest.mark.parametrize('corruption', ['wrong_episode_uri', 'changed_depth_bytes', 'malformed_depth'])
def test_pixel_projection_rejects_untrusted_depth_without_leaking_private_paths(runner, monkeypatch, corruption):
    make, _, workers = runner
    api = make(depth=True)
    session, _ = session_for(api)
    camera = session.snapshot.cameras[0]
    if corruption == 'wrong_episode_uri':
        camera.depth_uri = '/episodes/another_episode/depth/0/overview'
    elif corruption == 'changed_depth_bytes':
        path, _ = next(iter(session.public_depths.values()))
        path.write_bytes(b'PRIVATE_FILE_CANARY /private/depth.npy')
    else:
        monkeypatch.setattr(session, 'read_public_depth', lambda *args: b'PRIVATE_FILE_CANARY /private/depth.npy')
    before = deepcopy(workers[-1].calls)
    with TestClient(create_app(service=api)) as client:
        value = call(client, session, 'project_pixel', {
            'observation_sequence': session.sequence, 'camera_id': 'overview', 'u': 2, 'v': 2}, status=503)
        assert value['error']['code'] == 'infrastructure_error'
        assert 'PRIVATE_FILE_CANARY' not in json.dumps(value)
        assert_public(value, session)
        assert workers[-1].calls == before


@pytest.fixture(params=['single', 'dual'])
def geometry_session(runner, request):
    """Install real public FK in a CPU session; private worker state stays unrelated.

    These tests exercise the facade boundary, not native creation/robot binding.
    That binding has independent tests in test_agent_geometry_visibility.
    """
    make, _, workers = runner
    api = make(depth=True)
    session, _ = session_for(api)
    original = (session.capabilities, session.robot_geometry, session.snapshot)
    geometry = public_robot_geometry({'assets_root': f'{DATA_ROOT}/assets',
        'base_position_m': [.2, -.1, .77], 'assembly_version': 'isaacsim_6_1_v2',
        **({'hand_id': 'dual_fr3_sharpa_w01'} if request.param == 'dual' else {})})
    models = geometry.sides if hasattr(geometry, 'sides') else {'right': geometry}
    q = {j.name: float(np.clip(0., j.lower, j.upper)) for j in geometry.joints}
    for model in models.values():
        q.update(dict(zip(model.arm.names, [0., -.785, 0., -2.356, 0., 1.57, .785])))
    caps = session.capabilities.model_dump()
    caps.update(robot_id=geometry.robot_id,
        arm_joints=[name for model in models.values() for name in model.arm.names],
        hand_joints=[name for model in models.values() for name in model.hand.names],
        joint_limits={j.name: {'lower_rad': j.lower, 'upper_rad': j.upper} for j in geometry.joints},
        frame_ids=['world', 'robot_base', *[model.frame_id for model in models.values()]])
    session.capabilities = RobotCapabilities.model_validate(caps)
    session.robot_geometry = geometry
    observed = session.snapshot.model_dump()
    observed['proprioception'] = {'joint_positions_rad': q,
        'joint_velocities_rad_s': dict.fromkeys(q, 0.),
        **({'hand_geometries': geometry.public_hand_geometries(q)} if hasattr(geometry, 'sides')
           else {'geometry': geometry.public_finger_geometry(q)})}
    session.snapshot = AgentObservation.model_validate(observed)
    # The query-only HTTP host does not finalize the episode. Restore the fake's
    # actual inventory before the shared runner performs normal finalization.
    host = SimpleNamespace(get=api.get, shutdown=lambda: None)
    yield host, session, workers[-1], models
    session.capabilities, session.robot_geometry, session.snapshot = original


def test_frame_transforms_use_robot_and_camera_public_poses_and_are_invertible(geometry_session):
    api, session, worker, models = geometry_session
    before = deepcopy(worker.calls)
    raw = deepcopy(session.raw)
    geometry_before = pickle.dumps(session.robot_geometry)
    snapshot = session.snapshot.model_dump(mode='json')
    with TestClient(create_app(service=api)) as client:
        points = [[.03, -.02, .04], [0., 0., 0.]]
        frames = {'robot_base': session.robot_geometry.base}
        for side, model in models.items():
            public = (snapshot['proprioception'].get('hand_geometries') or {'right': snapshot['proprioception'].get('geometry')})[side]
            pose = public['wrist']
            matrix = np.eye(4)
            matrix[:3, :3] = quaternion_matrix(pose['quaternion_wxyz'])
            matrix[:3, 3] = pose['position_m']
            frames[model.frame_id] = matrix
        camera_matrix = np.eye(4)
        camera_matrix[:3, 3] = [1., 2., 3.]
        frames['camera:overview'] = camera_matrix
        for frame, matrix in frames.items():
            converted = call(client, session, 'transform_points', {
                'observation_sequence': session.sequence, 'source_frame': frame,
                'target_frame': 'world', 'points_m': points})
            expected = (matrix @ np.column_stack([points, np.ones(len(points))]).T).T[:, :3]
            np.testing.assert_allclose(converted['result']['points_m'], expected, atol=1e-12)
            restored = call(client, session, 'transform_points', {
                'observation_sequence': session.sequence, 'source_frame': 'world',
                'target_frame': frame, 'points_m': converted['result']['points_m']})
            np.testing.assert_allclose(restored['result']['points_m'], points, atol=1e-12)
            assert_public(converted, session)
        assert worker.calls == before
        assert session.raw == raw
        assert pickle.dumps(session.robot_geometry) == geometry_before
        assert session.snapshot.model_dump(mode='json') == snapshot


@pytest.mark.parametrize('updates', [
    {'source_frame': 'object:' + PRIVATE_MARKER},
    {'target_frame': 'missing_hand'}, {'source_frame': 'camera:missing'},
])
def test_frame_transform_cannot_lookup_private_object_or_unknown_frame(runner, updates):
    make, _, _ = runner
    api = make(depth=True)
    session, _ = session_for(api)
    with TestClient(create_app(service=api)) as client:
        value = call(client, session, 'transform_points', {
            'observation_sequence': session.sequence, 'source_frame': 'world',
            'target_frame': 'world', 'points_m': [[0., 0., 0.]], **updates}, status=422)
        assert value['error']['code'] == 'unsupported_capability'
        assert_public(value, session)


def test_hand_preset_queries_are_side_specific_parameters_without_execution(geometry_session):
    api, session, worker, models = geometry_session
    before = deepcopy(worker.calls)
    geometry_before = pickle.dumps(session.robot_geometry)
    snapshot = session.snapshot.model_dump(mode='json')
    with TestClient(create_app(service=api)) as client:
        functions = client.get(f'/episodes/{session.spec.episode_id}/functions').json()
        available = {entry['name']: entry['available'] for entry in functions['functions']}
        assert available['get_hand_presets'] is available['hand_preset'] is True
        catalog = call(client, session, 'get_hand_presets')['result']
        assert catalog['available'] is True
        assert set(catalog['sides']) == set(models)
        assert {item['preset'] for item in catalog['presets']} == {'open', 'pinch_prepose'}
        for side, model in models.items():
            opened = call(client, session, 'hand_preset', {'side': side, 'preset': 'open'})['result']
            anchor = call(client, session, 'hand_preset', {'side': side, 'preset': 'pinch_prepose'})['result']
            halfway = call(client, session, 'hand_preset', {'side': side, 'preset': 'pinch_prepose', 'amount': .5})['result']
            assert set(opened['hand_posture_rad']) == set(model.hand.names)
            assert all(q == 0. for q in opened['hand_posture_rad'].values())
            assert halfway['hand_posture_rad'] == {name: .5 * q for name, q in anchor['hand_posture_rad'].items()}
            assert halfway['frame_id'] == model.frame_id
            for name, q in anchor['hand_posture_rad'].items():
                limit = session.capabilities.joint_limits[name]
                assert limit.lower_rad <= q <= limit.upper_rad
            assert set(anchor['fingertip_poses']) == {'thumb', 'index', 'middle', 'ring', 'little'}
            for flag in ('advances_physics', 'mutates_controller', 'collision_checked',
                         'trajectory_checked', 'native_grasp_qualified'):
                assert anchor['validation'][flag] is False
            assert_public(anchor, session)
        if set(models) == {'right'}:
            value = call(client, session, 'hand_preset', {'side': 'left', 'preset': 'open'}, status=422)
            assert value['error']['code'] == 'invalid_arguments'
        assert worker.calls == before
        assert pickle.dumps(session.robot_geometry) == geometry_before
        assert session.snapshot.model_dump(mode='json') == snapshot
        assert session.life.active_action_id is None


def test_robot_and_ik_queries_use_published_joint_state_and_preserve_controller(geometry_session):
    api, session, worker, models = geometry_session
    before = deepcopy(worker.calls)
    geometry_before = pickle.dumps(session.robot_geometry)
    snapshot = session.snapshot.model_dump(mode='json')
    with TestClient(create_app(service=api)) as client:
        guide = call(client, session, 'get_robot')['result']
        assert set(guide['hands']) == set(models)
        assert guide['ik_query']['available'] is True
        assert_public(guide, session)
        # A dual query may select just one side; it cannot invent an action for
        # the unrequested hand. The target is the published measured wrist.
        side = next(iter(models))
        proprio = snapshot['proprioception']
        public = (proprio.get('hand_geometries') or {'right': proprio.get('geometry')})[side]
        target = {'wrist': public['wrist'], 'fingertip_frame_id': models[side].frame_id}
        args = {'observation_sequence': session.sequence,
            **({'hand_geometries': {side: target}} if len(models) == 2 else {'geometry': target})}
        value = call(client, session, 'query_ik', args)
        result = value['result']
        assert result['solution_found'] is True
        assert set(result['hands']) == {side}
        assert result['sim_time_s'] == snapshot['sim_time_s']
        for flag in ('advances_physics', 'mutates_controller', 'collision_checked', 'trajectory_checked'):
            assert result[flag] is False
        assert_public(value, session)
        stale = call(client, session, 'query_ik', {
            **args, 'observation_sequence': session.sequence + 1}, status=409)
        assert stale['error']['code'] == 'stale_observation'
        assert worker.calls == before
        assert pickle.dumps(session.robot_geometry) == geometry_before
        assert session.snapshot.model_dump(mode='json') == snapshot
        assert session.life.active_action_id is None


def test_missing_geometry_has_explicit_preset_catalog_and_unsupported_queries(runner):
    make, _, _ = runner
    api = make()
    session, _ = session_for(api)
    assert session.robot_geometry is None
    with TestClient(create_app(service=api)) as client:
        catalog = call(client, session, 'get_hand_presets')['result']
        assert catalog['available'] is False and catalog['reason']
        assert catalog['sides'] == catalog['presets'] == []
        preset = call(client, session, 'hand_preset', {'preset': 'open', 'side': 'right'}, status=422)
        assert preset['error']['code'] == 'unsupported_capability'
        ik = call(client, session, 'query_ik', {'observation_sequence': session.sequence,
            'geometry': {'hand_posture_rad': {'right_index_PIP': .2}}}, status=422)
        assert ik['error']['code'] == 'unsupported_capability'


@pytest.mark.parametrize('update', [
    {'amount': 1.01}, {'amount': '0.5'}, {'amount': True},
    {'object_id': PRIVATE_MARKER}, {'preset': 'grasp'}, {'side': 'both'},
])
def test_hand_preset_rejects_out_of_contract_arguments(runner, update):
    make, _, _ = runner
    api = make()
    session, _ = session_for(api)
    with TestClient(create_app(service=api)) as client:
        value = call(client, session, 'hand_preset', {
            'side': 'right', 'preset': 'open', **update}, status=422)
        assert value['error']['code'] == 'invalid_arguments'
        assert_public(value, session)


def test_function_response_discriminator_rejects_wrong_result_and_extra_oracle(runner, monkeypatch):
    make, _, _ = runner
    api = make(depth=True)
    session, _ = session_for(api)
    with TestClient(create_app(service=api)) as client:
        valid = call(client, session, 'observe')
        for update in ({'function': 'get_robot'},
                       {'result': {**valid['result'], 'raw_objects': {PRIVATE_MARKER: [1., 2., 3.]}}}):
            with pytest.raises(ValidationError):
                RESPONSE_ADAPTER.validate_python({**valid, **update})
        monkeypatch.setattr(session.functions, '_observe', lambda args: {'private_path': '/private/robot.usd'})
        value = call(client, session, 'observe', status=503)
        assert value['error']['code'] == 'infrastructure_error'
        assert '/private/' not in json.dumps(value)
        assert_public(value, session)
