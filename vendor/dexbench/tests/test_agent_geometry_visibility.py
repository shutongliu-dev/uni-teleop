"""CPU protocol checks for equal robot-only FK visibility across control tracks.

The worker is a test double. Its robot contract comes from a recorded native
episode and the shared URDFs; these tests make no native physics claim.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT, load_json
from bench.protocol import EpisodeClock, PolicyRegistry
from bench.control.fingertip import ClassicalFingertipPolicy, Fr3SharpaGeometry
from bench.geometry import quaternion_matrix
from bench.runtime import service
from bench.runtime.controllers import robot_configuration
from bench.runtime.service import AgentService, ServiceError
from bench.sensors.tactile import SOURCE
from test_agent_service import (
    FakeWorker, ManualTime, PRIVATE_MARKER, frozen_inputs, join_action, publish_targets, request,
)


NATIVE_METADATA = Path(
    f'{DATA_ROOT}/datasets/fr3_sharpa_bench/agent_bench/'
    'classical_fingertip_20261011/server_episodes_r2/'
    'ecfc4ff675194225becba5f85ae5d238/private/metadata.json')


def publish_native_window(worker):
    """Publish drive targets and the complete contact receipt of a native window."""
    publish_targets(worker)
    end = worker.raw['sim_time_s']
    window = round(end / .02)
    worker.raw['contacts'] = {'source': SOURCE, 'errors': [], 'rigid': [], 'sampling': {
        'protocol': 'native_rigid_contact_sampling_v1', 'physics_dt_s': .0005,
        'start_sim_time_s': round(end - .02, 10), 'end_sim_time_s': end,
        'native_start_time_s': 1. + end - .02, 'native_end_time_s': 1. + end,
        'native_start_step_index': 2000 + (window - 1) * 40, 'native_end_step_index': 2000 + window * 40,
        'sampled_substeps': 40, 'complete': True}}


@pytest.fixture(scope='module')
def native_robot_metadata():
    if not NATIVE_METADATA.is_file():
        pytest.skip('Recorded native FR3/SHARPA robot metadata is unavailable')
    recorded = load_json(NATIVE_METADATA)
    # Keep the recorded robot identity, installation and joint contract. Camera
    # dimensions and runtime source hashes belong to this CPU double, not the
    # earlier native run, so they must not be copied from that run.
    return {key: recorded[key] for key in (
        'native_isaacsim_app', 'robot_usd_sha256', 'robot_asset',
        'scene_spec', 'action_spec')}


@pytest.fixture(scope='module')
def geometry(frozen_inputs, native_robot_metadata):
    result = Fr3SharpaGeometry(**robot_configuration(frozen_inputs[1][1]))
    assert native_robot_metadata['native_isaacsim_app'] is True
    assert native_robot_metadata['robot_asset']['sources'] == result.asset_sha256
    assert robot_configuration(native_robot_metadata['scene_spec']) == robot_configuration(frozen_inputs[1][1])
    return result


@pytest.fixture
def geometry_service(tmp_path, monkeypatch, frozen_inputs, native_robot_metadata, geometry):
    manifest, inputs = frozen_inputs
    monkeypatch.setattr(service, 'validate', lambda release: deepcopy(manifest))
    monkeypatch.setattr(service, 'task_inputs', lambda release, task_id: deepcopy(inputs))
    clock = ManualTime()
    monkeypatch.setattr(service, 'EpisodeClock', lambda budgets: EpisodeClock(budgets, monotonic=clock))
    created = []

    def make(*, track='direct_joints', robot_id='sharpa_w01_right', native=True, mutate=None, metadata=None):
        workers = []

        def factory(**kwargs):
            worker = FakeWorker(**kwargs, on_step=publish_native_window if native else publish_targets)
            if robot_id == 'sharpa_w01_right' or metadata is not None:
                worker.initial['metadata'].update(deepcopy(metadata or native_robot_metadata))
                action_spec = worker.initial['metadata']['action_spec']
                worker.arm = list(action_spec['arm_joint_names'])
                worker.hand = list(action_spec['hand_joint_names'])
                worker.active = worker.arm + worker.hand
                worker.raw['joint_positions_rad'] = {
                    **action_spec['initial_joint_positions_rad'], 'passive_mimic': .123}
                worker.raw['joint_velocities_rad_s'] = dict.fromkeys(worker.raw['joint_positions_rad'], 0.)
                # This native-shaped double replaces the joint inventory. Bind
                # the reset drive acknowledgement to that same active inventory;
                # FakeWorker's measured feedback follows its targets exactly.
                worker.raw['robot'] = {'joint_targets_rad': {
                    name: action_spec['initial_joint_positions_rad'][name] for name in worker.active}}
                worker.initial['observation'] = deepcopy(worker.raw)
            worker.initial['metadata']['native_isaacsim_app'] = native
            if mutate:
                mutate(worker.initial['metadata'])
            workers.append(worker)
            return worker

        registry = PolicyRegistry()
        changes = dict(track=track, robot_id=robot_id)
        if track == 'system0':
            policy = ClassicalFingertipPolicy(**robot_configuration(inputs[1]))
            policy.robot_configuration = robot_configuration(inputs[1])
            registry.register(policy)
            changes.update(policy_id=policy.descriptor.policy_id, policy_version=policy.descriptor.version)
        api = AgentService(tmp_path / f'run_{len(created)}', registry=registry, worker_factory=factory)
        created.append(api)
        reply = api.create(request(**changes))
        return api.get(reply['episode_id']), reply, workers[0]

    yield make
    for api in created:
        for session in api.sessions.values():
            if session.result is None:
                session.finish()
            if session.thread:
                session.thread.join(timeout=3)


def assert_geometry_matches_measured_joints(observation, geometry):
    proprio = observation['proprioception']
    q = proprio['joint_positions_rad']
    result = proprio['geometry']
    assert set(q) == set(geometry.names)
    assert result == geometry.public_finger_geometry(q)
    assert {k: result[k] for k in geometry.public_geometry(q)} == geometry.public_geometry(q)
    wrist = geometry.base @ geometry.arm.forward(q)['fr3_link8']
    np.testing.assert_allclose(result['wrist']['position_m'], wrist[:3, 3], atol=1e-12)
    np.testing.assert_allclose(quaternion_matrix(result['wrist']['quaternion_wxyz']), wrist[:3, :3], atol=1e-12)
    assert result['wrist']['frame_id'] == 'world'
    assert result['fingertip_frame_id'] == 'right_hand_flange'
    assert set(result['fingertip_positions_m']) == {'thumb', 'index', 'middle', 'ring', 'little'}
    return result


def test_direct_and_classical_have_identical_public_robot_fk(geometry_service, geometry):
    direct, direct_reply, _ = geometry_service()
    classical, classical_reply, _ = geometry_service(track='system0')
    assert direct.policy is None
    assert isinstance(classical.policy, ClassicalFingertipPolicy)
    assert direct_reply['spec']['track'] == 'direct_joints'
    assert classical_reply['spec']['track'] == 'system0'
    direct_fk = assert_geometry_matches_measured_joints(direct_reply['observation'], geometry)
    classical_fk = assert_geometry_matches_measured_joints(classical_reply['observation'], geometry)
    assert direct_fk == classical_fk
    assert direct.sensor_config['robot_geometry'] == classical.sensor_config['robot_geometry']
    assert direct.sensor_config['robot_geometry']['assets_sha256'] == geometry.asset_sha256
    assert direct.sensor_config['robot_geometry']['available_in_both_control_tracks'] is True


@pytest.mark.parametrize('track', ['direct_joints', 'system0'])
def test_geometry_never_uses_or_exposes_private_object_data(geometry_service, geometry, track):
    session, reply, _ = geometry_service(track=track)
    initial = assert_geometry_matches_measured_joints(reply['observation'], geometry)
    session.raw['objects'] = {PRIVATE_MARKER: {'position_m': [100., 200., 300.],
                                              'quaternion_wxyz': [1., 0., 0., 0.]}}
    session.raw['cloth']['vertices_m'] = [[123., 456., 789.]]
    session.raw['geometry'] = {'wrist': {'position_m': [999., 999., 999.]}}
    projected = session._projection().model_dump(mode='json', exclude_none=True)
    assert assert_geometry_matches_measured_joints(projected, geometry) == initial
    public = json.dumps({'ready': reply, 'observation': projected})
    for forbidden in (PRIVATE_MARKER, 'passive_mimic', 'privileged_object_poses',
                      'privileged_cloth_vertices_m', 'evaluator_debug', 'private_detail',
                      str(session.directory), str(NATIVE_METADATA), f'{DATA_ROOT}/assets'):
        assert forbidden not in public


@pytest.mark.parametrize('track', ['direct_joints', 'system0'])
def test_next_decision_fk_tracks_new_public_joint_feedback(geometry_service, geometry, track):
    session, reply, worker = geometry_service(track=track)
    before = reply['observation']['proprioception']['geometry']
    target_q = dict(reply['observation']['proprioception']['joint_positions_rad'])
    target_q['fr3_joint1'] += .02
    target_q['right_index_PIP'] += .15
    payload = {'kind': track, 'action_id': 'move_geometry_visibility', 'observation_sequence': 0}
    if track == 'direct_joints':
        # Enough time for the largest requested move (.15 rad at the published
        # 1 rad/s hand cap), including the native minimum-jerk trajectory.
        payload.update(duration_s=.4,
            arm_joint_positions_rad={n: target_q[n] for n in session.capabilities.arm_joints},
            hand_joint_positions_rad={n: target_q[n] for n in session.capabilities.hand_joints})
    else:
        target = geometry.public_geometry(target_q)
        payload.update(skill='move_geometry', timeout_sim_s=3., geometry={
            'wrist': target['wrist'], 'fingertip_positions_m': {'index': target['fingertip_positions_m']['index']}})
    session.submit(payload)
    join_action(session)
    assert session.status(payload['action_id'])['status'] == 'completed'
    after = session.observe()
    assert after['sequence'] > reply['observation']['sequence']
    actual = assert_geometry_matches_measured_joints(after, geometry)
    if track == 'direct_joints':
        assert after['proprioception']['joint_positions_rad'] == pytest.approx(target_q)
        assert after['proprioception']['commanded_joint_positions_rad'] == pytest.approx(target_q)
        steps = [data['command'] for operation, data in worker.calls if operation == 'step']
        assert steps[-1]['joint_positions_rad'] == pytest.approx(target_q)
    assert not np.allclose(actual['wrist']['position_m'], before['wrist']['position_m'])
    assert not np.allclose(actual['fingertip_positions_m']['index'], before['fingertip_positions_m']['index'])
    assert PRIVATE_MARKER not in json.dumps(after)


def assembly_metadata(robot_id, scene):
    """Native-shaped robot contract of a registered hand, from its versioned assembly."""
    from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec, scene_with_hand
    spec = get_hand_spec(robot_id)
    directory = Path(f'{DATA_ROOT}/assets/robots/assemblies') / ('fr3_' + robot_id) / spec['assembly_version']
    source = load_json(directory / 'SOURCE.json')
    names = [n for n in source['joint_limits'] if n not in spec['mimic_joints']]
    arm = [n for n in names if n.startswith('fr3_')]
    limits = {n: [source['joint_limits'][n]['lower'], source['joint_limits'][n]['upper']] for n in names}
    home = {n: float(np.clip(0., *limits[n])) for n in names}
    return {'native_isaacsim_app': True, 'robot_usd_sha256': '1' * 64, 'robot_asset': source,
            'scene_spec': scene_with_hand(scene, robot_id),
            'action_spec': {'joint_names': names, 'arm_joint_names': arm, 'hand_joint_names': [n for n in names if n not in arm],
                            'joint_limits_rad': limits, 'initial_joint_positions_rad': home}}


@pytest.mark.parametrize('robot_id', ['allegro_v5_right', 'leap_v1_right', 'shadow_hand_e_right'])
def test_registered_hand_publishes_its_own_verified_fk(geometry_service, frozen_inputs, robot_id):
    metadata = assembly_metadata(robot_id, frozen_inputs[1][1])
    session, reply, _ = geometry_service(robot_id=robot_id, metadata=metadata)
    geometry = session.robot_geometry
    assert geometry is not None and geometry.robot_id == robot_id
    assert session.sensor_config['robot_geometry']['assets_sha256'] == geometry.asset_sha256
    assert 'right_hand_flange' in reply['capabilities']['frame_ids']
    published = reply['observation']['proprioception']['geometry']
    q = reply['observation']['proprioception']['joint_positions_rad']
    assert published == geometry.public_geometry(q)
    assert set(published['fingertip_positions_m']) == set(geometry.fingertip_links)


@pytest.mark.parametrize('robot_id', ['allegro_v5_right', 'shadow_hand_e_right'])
def test_registered_hand_rejects_inconsistent_native_contract(geometry_service, robot_id):
    # The recorded SHARPA native contract cannot back another hand's public FK.
    with pytest.raises(ServiceError, match='differs'):
        geometry_service(robot_id=robot_id, metadata=None, mutate=lambda m: m.update(native_isaacsim_app=True))


def test_unverified_direct_worker_does_not_advertise_native_fk(geometry_service):
    session, reply, _ = geometry_service(native=False)
    assert session.robot_geometry is None
    assert 'geometry' not in reply['observation']['proprioception']
    assert 'robot_geometry' not in session.sensor_config


@pytest.mark.parametrize('track', ['direct_joints', 'system0'])
@pytest.mark.parametrize('mismatch,message', [
    ('source', 'URDF differs from native robot sources'),
    ('base', 'Native robot base differs'),
    ('mount', 'Native hand mount differs'),
])
def test_mismatched_native_geometry_is_rejected_at_setup(geometry_service, track, mismatch, message):
    def mutate(metadata):
        if mismatch == 'source':
            sources = metadata['robot_asset']['sources']
            sources[next(iter(sources))] = '0' * 64
        elif mismatch == 'base':
            metadata['scene_spec']['robot']['base_position_m'][2] += .1
        else:
            metadata['robot_asset']['mount']['translation_m'][0] += .01

    with pytest.raises(ServiceError, match=message) as error:
        geometry_service(track=track, mutate=mutate)
    assert error.value.status == 422
