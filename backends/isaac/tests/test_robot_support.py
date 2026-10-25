"""Public robot facts and read-only IK on real URDF models; no native physics."""
from copy import deepcopy
import pickle
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from bench.control.bimanual import BimanualFingertipPolicy
from bench.control.fingertip import ClassicalFingertipPolicy
from bench.paths import DATA_ROOT
from bench.protocol.schemas import AgentObservation, JointLimit, RobotCapabilities
from bench.runtime.robot_support import IKQuery
from bench.runtime.server import create_app
from bench.runtime.service import EpisodeSession


@pytest.fixture(params=['single', 'dual'])
def session(request):
    cls = ClassicalFingertipPolicy if request.param == 'single' else BimanualFingertipPolicy
    policy = cls(f'{DATA_ROOT}/assets', [0., 0., .77])
    geometry = policy.geometry
    models = geometry.sides if hasattr(geometry, 'sides') else {'right': geometry}
    arm = [name for model in models.values() for name in model.arm.names]
    hand = [name for model in models.values() for name in model.hand.names]
    q = {joint.name: float(np.clip(0., joint.lower, joint.upper)) for joint in geometry.joints}
    for model in models.values():
        q.update(dict(zip(model.arm.names, [0., -.785, 0., -2.356, 0., 1.57, .785])))
    value = EpisodeSession.__new__(EpisodeSession)
    value.lock = threading.RLock()
    value.result = None
    value.thread = None
    value.life = SimpleNamespace(active_action_id=None)
    value.policy = policy
    value.robot_geometry = geometry
    value.snapshot = AgentObservation(episode_id='test', sequence=7, sim_time_s=.4,
        instruction='Public task', profile='vision_proprio',
        proprioception={'joint_positions_rad': q, 'joint_velocities_rad_s': dict.fromkeys(q, 0.)})
    value.capabilities = RobotCapabilities(robot_id=geometry.robot_id,
        robot_asset_sha256='a'*64, arm_joints=arm, hand_joints=hand,
        joint_limits={j.name: JointLimit(lower_rad=j.lower, upper_rad=j.upper) for j in geometry.joints},
        tracks=['direct_joints', 'system0'], system0=policy.descriptor,
        frame_ids=['world', *[model.frame_id for model in models.values()]])
    # Neither support operation may inspect these private/native fields.
    value.worker = object()
    value.raw = {'objects': {'PRIVATE_ORACLE_CANARY': [9., 8., 7.]}}
    return value


def body_for(session):
    q = session.snapshot.proprioception.joint_positions_rad
    geometry = session.robot_geometry
    if hasattr(geometry, 'sides'):
        targets = {side: {'wrist': model.public_geometry({n: q[n] for n in model.names})['wrist'],
                          'fingertip_frame_id': model.frame_id} for side, model in geometry.sides.items()}
        return {'observation_sequence': 7, 'hand_geometries': targets}
    return {'observation_sequence': 7, 'geometry': {'wrist': geometry.public_geometry(q)['wrist']}}


def client_for(session):
    def get(identifier):
        assert identifier == 'test'
        return session
    return TestClient(create_app(service=SimpleNamespace(get=get, shutdown=lambda: None)))


def test_robot_guide_and_ik_share_real_geometry_without_private_state_or_execution(session):
    before = pickle.dumps(vars(session.policy))
    snapshot = session.snapshot.model_dump()
    with client_for(session) as client:
        guide = client.get('/episodes/test/robot')
        assert guide.status_code == 200
        value = guide.json()
        assert value['ik_query']['available'] is True
        presets = value['hand_presets']
        assert presets['available'] is True
        assert set(presets['sides']) == set(value['hands'])
        assert {entry['preset'] for entry in presets['presets']} == {'open', 'pinch_prepose'}
        assert presets['validation']['native_grasp_qualified'] is False
        assert presets['validation']['collision_checked'] is False
        assert value['contact_feedback']['grasp_detected_available'] is False
        assert set(value['joint_limits']) == set(session.robot_geometry.names)
        assert all(hand['fingers'] for hand in value['hands'].values())
        result = client.post('/episodes/test/robot/ik', json=body_for(session))
        assert result.status_code == 200, result.text
        answer = result.json()
        assert answer['solution_found'] is True
        assert answer['observation_sequence'] == 7 and answer['sim_time_s'] == .4
        assert answer['advances_physics'] is answer['mutates_controller'] is False
        assert answer['collision_checked'] is answer['trajectory_checked'] is False
        assert 'PRIVATE_ORACLE_CANARY' not in result.text + guide.text
        assert DATA_ROOT not in result.text + guide.text
        for solved in answer['hands'].values():
            assert solved['solution_found'] is True
            assert solved['residuals']['wrist_position_error_m'] < .0015
            for name, position in solved['joint_positions_rad'].items():
                limit = session.capabilities.joint_limits[name]
                assert limit.lower_rad <= position <= limit.upper_rad
    assert pickle.dumps(vars(session.policy)) == before
    assert session.snapshot.model_dump() == snapshot
    assert session.life.active_action_id is None


@pytest.mark.parametrize('change', ['stale', 'busy', 'finished', 'no_geometry'])
def test_queries_refuse_unusable_episode_without_mutating_it(session, change):
    body = body_for(session)
    if change == 'stale':
        body['observation_sequence'] = 6
    elif change == 'busy':
        session.life.active_action_id = 'running_action'
    elif change == 'finished':
        session.result = object()
    else:
        session.robot_geometry = None
    before = pickle.dumps(vars(session.policy))
    with client_for(session) as client:
        result = client.post('/episodes/test/robot/ik', json=body)
    assert result.status_code == (422 if change == 'no_geometry' else 409)
    assert pickle.dumps(vars(session.policy)) == before


def test_unverified_ik_is_reported_without_implying_an_impossibility_proof(session):
    body = body_for(session)
    target = body.get('geometry') or body['hand_geometries']['left']
    target['wrist']['position_m'] = [100., 100., 100.]
    with client_for(session) as client:
        result = client.post('/episodes/test/robot/ik', json=body)
    assert result.status_code == 200
    value = result.json()
    assert value['solution_found'] is False
    failed = next(v for v in value['hands'].values() if not v['solution_found'])
    assert failed['diagnostic']['code'] == 'ik_solution_unverified'
    assert 'joint_positions_rad' not in failed


@pytest.mark.parametrize('change', ['unknown', 'wrong_frame', 'objects', 'wrong_form'])
def test_strict_robot_query_contract_rejects_oracle_or_unsupported_input(session, change):
    body = body_for(session)
    target = body.get('geometry') or body['hand_geometries']['left']
    if change == 'unknown':
        target['hand_posture_rad'] = {'not_a_joint': .2}
    elif change == 'wrong_frame':
        target['wrist']['frame_id'] = 'object'
    elif change == 'objects':
        body['objects'] = {'target': 'find_true_pose'}
    else:
        body = {'observation_sequence': 7, **({'geometry': target} if 'hand_geometries' in body
                                             else {'hand_geometries': {'right': target}})}
    with client_for(session) as client:
        result = client.post('/episodes/test/robot/ik', json=body)
    assert result.status_code == 422


def test_snapshot_change_during_query_is_rejected(session, monkeypatch):
    from bench.runtime import robot_support
    original = robot_support.query_ik
    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        session.snapshot = session.snapshot.model_copy(update={'sequence': 8})
        return value
    monkeypatch.setattr(robot_support, 'query_ik', changed)
    with client_for(session) as client:
        result = client.post('/episodes/test/robot/ik', json=body_for(session))
    assert result.status_code == 409


def test_query_never_claims_support_for_contact_or_trajectory_inputs():
    with pytest.raises(ValueError, match='only'):
        IKQuery.model_validate({'observation_sequence': 0, 'geometry': {
            'hand_posture_rad': {'right_index_PIP': .2},
            'wrist_trajectory': [{'time_s': 1., 'wrist': {
                'frame_id': 'world', 'position_m': [0., 0., 1.], 'quaternion_wxyz': [1., 0., 0., 0.]}}]}})


def test_robot_query_contract_is_public_and_runtime_keeps_compatible_import():
    from bench.protocol.robot import IKQuery as PublicIKQuery
    assert IKQuery is PublicIKQuery


def test_robot_support_responses_roundtrip_without_adding_or_dropping_fields(session):
    from bench.protocol.robot import IKQueryResult, RobotGuide
    with client_for(session) as client:
        guide = client.get('/episodes/test/robot').json()
        result = client.post('/episodes/test/robot/ik', json=body_for(session)).json()
    assert RobotGuide.model_validate(guide).model_dump(mode='json', exclude_unset=True) == guide
    assert IKQueryResult.model_validate(result).model_dump(mode='json', exclude_unset=True) == result


@pytest.mark.parametrize('configuration', ['matched', 'absent', 'unconstrained'])
def test_robot_guide_and_callable_catalog_publish_the_same_preset_support(session, configuration):
    from bench.control.hand_posture import HandPosturePrior
    from bench.protocol.robot import RobotGuide
    from bench.protocol.runtime import EpisodeClock
    from bench.protocol.schemas import Budgets
    from bench.runtime.robot_support import geometry_sides
    if configuration == 'absent':
        session.robot_geometry = None
    elif configuration == 'unconstrained':
        for model in geometry_sides(session.robot_geometry).values():
            model.hand_posture_prior = HandPosturePrior.unconstrained()
    session.spec = SimpleNamespace(episode_id='test', track='system0', budgets=Budgets())
    session.clock = EpisodeClock(session.spec.budgets)
    records = []
    session.recorder = SimpleNamespace(closed=False, function_call=records.append)
    before = pickle.dumps(vars(session.policy))
    snapshot = session.snapshot.model_dump()
    with client_for(session) as client:
        guide = client.get('/episodes/test/robot').json()
        outputs = {}
        for function in ('get_robot', 'get_hand_presets'):
            response = client.post('/episodes/test/call', json={
                'function': function, 'arguments': {}, 'request_id': function})
            assert response.status_code == 200, response.text
            assert response.json()['ok'] is True
            outputs[function] = response.json()['result']
        # Callable envelopes serialize optional defaults; the legacy route omits them.
        assert RobotGuide.model_validate(guide) == RobotGuide.model_validate(outputs['get_robot'])
        assert guide['hand_presets'] == outputs['get_robot']['hand_presets']
        assert guide['hand_presets'] == outputs['get_hand_presets']
        assert guide['hand_presets']['available'] is (configuration == 'matched')
        if configuration != 'matched':
            assert guide['hand_presets']['reason']
            assert guide['hand_presets']['sides'] == []
            assert guide['hand_presets']['presets'] == []
    assert pickle.dumps(vars(session.policy)) == before
    assert session.snapshot.model_dump() == snapshot
    assert session.clock.action_attempts == 0 and session.clock.sim_time_s == 0.
    assert len(records) == 4  # The two read-only calls retain request/response provenance.


def test_robot_guide_preserves_legacy_unavailable_preset_shape(session):
    from bench.protocol.robot import HandPresetSupport, RobotGuide
    value = session.robot_guide()
    value['hand_presets'] = {'available': False, 'reason': 'No preset library in this source version.'}
    parsed = RobotGuide.model_validate(value)
    assert isinstance(parsed.hand_presets, HandPresetSupport)
    assert parsed.model_dump(mode='json', exclude_unset=True) == value
    for invalid in (True, 0, 'false'):
        value['hand_presets']['available'] = invalid
        with pytest.raises(ValueError):
            RobotGuide.model_validate(value)


def test_robot_guide_rejects_preset_sides_without_public_geometry(session):
    from bench.protocol.robot import RobotGuide
    value = session.robot_guide()
    value['hands'] = {}
    value['ik_query']['available'] = False
    with pytest.raises(ValueError, match='Preset sides'):
        RobotGuide.model_validate(value)


@pytest.mark.parametrize('change', ['false_as_zero', 'unknown_field', 'wrong_finger_joint', 'wrong_side_frame', 'tactile_mismatch'])
def test_public_robot_guide_rejects_malformed_or_inconsistent_data(session, change):
    from bench.protocol.robot import RobotGuide
    value = session.robot_guide()
    if change == 'false_as_zero':
        value['ik_query']['collision_planning'] = 0
    elif change == 'unknown_field':
        value['objects'] = {'hidden': [1., 2., 3.]}
    elif change == 'wrong_finger_joint':
        next(iter(next(iter(value['hands'].values()))['fingers'].values()))['joint_names'] = ['unknown_joint']
    elif change == 'wrong_side_frame':
        next(iter(value['hands'].values()))['frame_id'] = 'world'
    else:
        value['contact_feedback']['tactile_available'] = True
        value['contact_feedback']['mode'] = 'unavailable'
    with pytest.raises(ValueError):
        RobotGuide.model_validate(value)


@pytest.mark.parametrize('change', ['false_as_zero', 'overall_result', 'nan_residual', 'wrong_frame', 'failed_with_solution', 'missing_sample_time'])
def test_public_ik_result_rejects_malformed_or_inconsistent_data(session, change):
    from bench.protocol.robot import IKQueryResult
    value = session.query_ik(IKQuery.model_validate(body_for(session)))
    hand = next(iter(value['hands'].values()))
    if change == 'false_as_zero':
        value['advances_physics'] = 0
    elif change == 'overall_result':
        value['solution_found'] = False
    elif change == 'nan_residual':
        hand['residuals']['wrist_position_error_m'] = float('nan')
    elif change == 'wrong_frame':
        hand['verified_geometry']['wrist']['frame_id'] = 'object'
    elif change == 'failed_with_solution':
        hand['solution_found'] = False
        hand['diagnostic'] = {'code': 'ik_solution_unverified'}
    else:
        del value['sim_time_s']
    with pytest.raises(ValueError):
        IKQueryResult.model_validate(value)


def test_runtime_ik_revalidates_modified_input_instances(session):
    from bench.runtime.robot_support import query_ik
    request = IKQuery.model_validate(body_for(session)).model_copy(update={'observation_sequence': -1})
    with pytest.raises(ValueError):
        query_ik(session.robot_geometry, session.snapshot.proprioception.joint_positions_rad, request)


def test_runtime_ik_validates_solver_output_before_publishing(session, monkeypatch):
    from bench.runtime.robot_support import geometry_sides, query_ik
    model = next(iter(geometry_sides(session.robot_geometry).values()))
    def invalid_result(q, **kwargs):
        return q, {'fingertip_errors_m': {}, 'wrist_position_error_m': float('inf'),
                   'wrist_rotation_error_rad': 0.}
    monkeypatch.setattr(model, 'solve', invalid_result)
    with pytest.raises(ValueError):
        query_ik(session.robot_geometry, session.snapshot.proprioception.joint_positions_rad,
                 IKQuery.model_validate(body_for(session)))


def test_robot_guide_without_geometry_and_unconstrained_posture_keep_existing_shapes(session):
    from bench.control.hand_posture import HandPosturePrior
    from bench.protocol.robot import RobotGuide
    from bench.runtime.robot_support import geometry_sides, robot_guide
    absent = robot_guide(session.capabilities, None)
    assert absent['hands'] == {} and absent['ik_query']['available'] is False
    for model in geometry_sides(session.robot_geometry).values():
        model.hand_posture_prior = HandPosturePrior.unconstrained()
    value = robot_guide(session.capabilities, session.robot_geometry)
    assert RobotGuide.model_validate(value)
    assert all(hand['posture_constraint'] == HandPosturePrior.unconstrained().describe()
               for hand in value['hands'].values())
