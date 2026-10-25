"""Original-source identity, FK and isolated CPU control; no native qualification."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from bench.control.factory import configured_policy, public_robot_geometry, robot_configuration
from bench.control.fingertip import Fr3SharpaGeometry
from bench.paths import DATA_ROOT, load_json
from bench.protocol.schemas import AgentObservation, JointLimit, System0Action
from bench.runtime.board_game_control import BoardGameFingertipPolicy
from bench.simulation.isaacsim_backend.instance_asset import assemble_instance
from bench.simulation.isaacsim_backend.robot_asset import collision_configuration
from bench.simulation.isaacsim_backend.robot_instances import (ASSEMBLY_VERSION, ROBOT_ID,
    checked_instances, instance_name)
from bench.simulation.isaacsim_backend.robot_layout import world_base_poses
from bench.simulation.isaacsim_backend.robot_registry import get_robot_spec, scene_with_hand

ASSETS = Path(f'{DATA_ROOT}/assets')


def robot(names=('front', 'rear')):
    return {'hand_id': ROBOT_ID, 'assembly_version': ASSEMBLY_VERSION,
        'assets_root': str(ASSETS), 'base_position_m': [0., 0., 0.],
        'base_quaternion_wxyz': [1., 0., 0., 0.],
        'instances': {name: {'hand_id': 'sharpa_w01_right', 'assembly_version': 'isaacsim_6_1_v2',
            'base_position_m': [.65, -.76 + index*1.52, .78],
            'base_quaternion_wxyz': [2**-.5, 0., 0., (1 if index % 2 == 0 else -1)*2**-.5]}
            for index, name in enumerate(names)}}


@pytest.mark.parametrize('damage', ['missing', 'one', 'nine', 'unsafe', 'reserved', 'extra_gain',
                                   'source_hand', 'version', 'pose', 'rotation', 'mixed_layout'])
def test_invalid_instance_declarations_fail_before_geometry_or_asset_writes(damage):
    value = robot()
    if damage == 'missing': del value['instances']
    elif damage == 'one': del value['instances']['rear']
    elif damage == 'nine': value = robot(tuple('member' + str(i) for i in range(9)))
    elif damage == 'unsafe': value['instances']['../outside'] = value['instances'].pop('rear')
    elif damage == 'reserved': value['instances']['fr3'] = value['instances'].pop('rear')
    elif damage == 'extra_gain': value['instances']['front']['hand_kp'] = 100.
    elif damage == 'source_hand': value['instances']['front']['hand_id'] = 'dual_fr3_sharpa_w01'
    elif damage == 'version': value['instances']['front']['assembly_version'] = 'unknown'
    elif damage == 'pose': value['instances']['front']['base_position_m'][0] = float('nan')
    elif damage == 'rotation': value['instances']['front']['base_quaternion_wxyz'] = [2., 0., 0., 0.]
    elif damage == 'mixed_layout': value['arm_base_poses'] = {}
    with pytest.raises(ValueError):
        robot_configuration({'robot': value})


def test_instance_inventory_is_deterministic_and_retains_original_source_chirality():
    value = robot()
    before = deepcopy(value)
    reversed_order = {**value, 'instances': dict(reversed(list(value['instances'].items())))}
    assert robot_configuration({'robot': reversed_order}) == robot_configuration({'robot': value})
    spec = get_robot_spec(value)
    assert list(spec['arms']) == ['front', 'rear']
    assert spec['joint_count'] == 58 and spec['arm_dof'] == 14 and spec['hand_dof'] == 44
    assert all(item['hand_id'] == 'sharpa_w01_right' for item in spec['source_hand_specs'].values())
    assert spec['pinch_links']['rear']['thumb'] == ['rear_thumb_elastomer']
    assert scene_with_hand({'robot': value}, ROBOT_ID)['robot']['instances'] == checked_instances(value)
    assert value == before
    with pytest.raises(ValueError, match='implicitly'):
        scene_with_hand({'robot': value}, 'dual_fr3_sharpa_w01')


def test_each_instance_fk_is_original_right_robot_under_its_own_rigid_transform():
    value = robot()
    geometry = public_robot_geometry(robot_configuration({'robot': value}))
    assert len(geometry.names) == len(set(geometry.names)) == 58
    assert set(geometry.frame_ids) == {'front_hand_flange', 'rear_hand_flange'}
    poses = world_base_poses(value)
    for name, item in value['instances'].items():
        original = Fr3SharpaGeometry(ASSETS, item['base_position_m'], item['base_quaternion_wxyz'],
                                    hand_id=item['hand_id'], assembly_version=item['assembly_version'])
        q = {joint.name: float(np.clip(0., joint.lower, joint.upper)) for joint in original.joints}
        q.update(zip(original.arm.names, [.2, -.5, 0., -2.3, 0., 1.8, .8]))
        remapped = {instance_name(name, joint): value for joint, value in q.items()}
        actual = geometry.sides[name]
        np.testing.assert_allclose(actual.arm_base, original.arm_base, atol=1e-14)
        np.testing.assert_allclose(actual.arm_base[:3, 3], poses[name]['position_m'], atol=1e-14)
        for field in ('position_m', 'quaternion_wxyz'):
            np.testing.assert_array_equal(actual.public_geometry(remapped)['wrist'][field], original.public_geometry(q)['wrist'][field])
        for finger, point in original.public_geometry(q)['fingertip_positions_m'].items():
            np.testing.assert_array_equal(actual.public_geometry(remapped)['fingertip_positions_m'][finger], point)
        assert [(j.lower, j.upper, j.velocity) for j in actual.joints] == [(j.lower, j.upper, j.velocity) for j in original.joints]
        assert actual.hand_posture_prior.maximum_dip_minus_pip_rad == original.hand_posture_prior.maximum_dip_minus_pip_rad
        assert actual.hand_posture_prior.pairs == {finger: tuple(instance_name(name, j) for j in pair)
            for finger, pair in original.hand_posture_prior.pairs.items()}


def test_namespaced_assembly_preserves_every_physical_xml_value_and_original_files(tmp_path):
    source_path = ASSETS/'robots/assemblies/fr3_sharpa/isaacsim_6_1_v2/SOURCE.json'
    source = load_json(source_path)
    paths = [source_path, Path(source['urdf_path']), *(Path(p) for p in source['sources'])]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    source_tree = ET.parse(source['urdf_path']).getroot()
    for namespace in ('front', 'rear'):
        path, manifest = assemble_instance(namespace, ASSETS, source_version='isaacsim_6_1_v2', output=tmp_path/namespace)
        generated = ET.parse(path).getroot()
        assert len(source_tree) == len(generated)
        for old, new in zip(source_tree, generated):
            assert old.tag == new.tag
            if old.tag in ('link', 'joint'):
                assert new.get('name') == instance_name(namespace, old.get('name'))
                for tag in ('inertial', 'visual', 'collision', 'origin', 'axis', 'limit', 'dynamics'):
                    assert [ET.tostring(node) for node in new.findall(tag)] == [ET.tostring(node) for node in old.findall(tag)]
                if old.tag == 'joint':
                    assert old.get('type') == new.get('type')
                    for edge in ('parent', 'child'):
                        assert new.find(edge).get('link') == instance_name(namespace, old.find(edge).get('link'))
        assert manifest['mesh_inputs'] == source['mesh_inputs']
        assert manifest['license_sha256'] == source['license_sha256']
        assert manifest['joint_limits'] == {instance_name(namespace, name): value for name, value in source['joint_limits'].items()}
        assert assemble_instance(namespace, ASSETS, source_version='isaacsim_6_1_v2', output=tmp_path/namespace) == (path, manifest)
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_original_structural_collision_pairs_are_mapped_per_instance_only():
    scene = {'robot': robot(), 'isaacsim': {'robot_self_collisions': True, 'filter_thumb_cmc_structure': True}}
    result = collision_configuration(scene)
    assert result['excluded_link_pairs'] == [['front_hand_C_MC', 'front_thumb_MC'], ['rear_hand_C_MC', 'rear_thumb_MC']]
    assert result['cross_finger_exclusions'] == []


def prepared_policy(names=('front', 'rear'), board=False):
    config = robot_configuration({'robot': robot(names)})
    policy = BoardGameFingertipPolicy(**config) if board else configured_policy(config)
    policy.reset(SimpleNamespace(episode_id='instances', observation_profile='vision_proprio'),
        SimpleNamespace(control_period_s=.02, joint_limits={j.name: JointLimit(lower_rad=j.lower, upper_rad=j.upper)
                                                          for j in policy.geometry.joints}))
    q = {joint.name: float(np.clip(0., joint.lower, joint.upper)) for joint in policy.geometry.joints}
    for model in policy.geometry.sides.values():
        q.update(zip(model.arm.names, [.2, -.5, 0., -2.3, 0., 1.8, .8]))
    return policy, dict(q=q.copy(), targets=q.copy(), sequence=0)


def observation(state):
    return AgentObservation(episode_id='instances', sequence=state['sequence'], sim_time_s=.02*state['sequence'],
        instruction='CPU test', profile='vision_proprio', cameras=[], proprioception={
            'joint_positions_rad': state['q'], 'joint_velocities_rad_s': dict.fromkeys(state['q'], 0.),
            'commanded_joint_positions_rad': state['targets']})


@pytest.mark.parametrize('owner', ['front', 'rear'])
def test_named_geometry_action_runs_original_controller_and_holds_other_instance(owner):
    policy, state = prepared_policy()
    own = policy.geometry.sides[owner]
    q_goal = {name: state['q'][name] for name in own.names}
    q_goal[own.arm.names[0]] += .04
    target = own.public_geometry(q_goal)['wrist']
    original = dict(state['targets'])
    action = System0Action(action_id='move', observation_sequence=0, skill='move_geometry', timeout_sim_s=3.,
        hand_geometries={owner: {'wrist': target, 'fingertip_frame_id': own.frame_id}})
    policy.begin(action, observation(state))
    for _ in range(151):
        result = policy.step(observation(state))
        if result.status != 'running': break
        for name, model in policy.geometry.sides.items():
            if name != owner:
                assert {j: result.command.joint_positions_rad[j] for j in model.names} == {j: original[j] for j in model.names}
        state['q'] = state['targets'] = dict(result.command.joint_positions_rad)
        state['sequence'] += 1
        policy.commit_pending_step()
    assert result.status == 'completed', result
    assert result.robot_execution_state.objective_status == 'reached'
    actual = own.public_geometry({name: state['q'][name] for name in own.names})
    np.testing.assert_allclose(actual['wrist']['position_m'], target['position_m'], atol=.003)


def test_three_player_instances_keep_inactive_drive_and_goals_across_turns(monkeypatch):
    policy, state = prepared_policy(('front', 'rear', 'observer'), board=True)
    for owner in ('front', 'rear', 'observer', 'front'):
        peer_names = [j for name, model in policy.geometry.sides.items() if name != owner for j in model.names]
        before = {name: state['targets'][name] for name in peer_names}
        # Measured peer drift must never become a new unowned drive command.
        for name, model in policy.geometry.sides.items():
            if name != owner: state['q'][model.arm.names[0]] = state['targets'][model.arm.names[0]] + .01
        policy.set_player_hand(owner, state['targets'])
        action = System0Action(action_id='hold'+str(state['sequence']), observation_sequence=state['sequence'],
                               skill='hold', timeout_sim_s=.04, hold_duration_s=.04)
        policy.begin(action, observation(state))
        for _ in range(4):
            result = policy.step(observation(state))
            if result.status != 'running': break
            assert {name: result.command.joint_positions_rad[name] for name in peer_names} == before
            state['targets'] = dict(result.command.joint_positions_rad)
            state['q'].update({name: state['targets'][name] for name in policy.geometry.sides[owner].names})
            state['sequence'] += 1
            policy.commit_pending_step()
        assert result.status == 'completed', result
    fresh, _ = prepared_policy(('front', 'rear', 'observer'), board=True)
    assert fresh is not policy and fresh.sides['front'] is not policy.sides['front']
    assert fresh.sides['front'].previous is None
    assert policy.sides['front'].previous is not policy.sides['rear'].previous


def test_composite_native_target_routing_and_bad_width_are_atomic():
    from bench.simulation.isaacsim_backend.composite_view import CompositeArticulationView

    class Tensor(np.ndarray):
        def contiguous(self): return self

    class View:
        count = 1
        def __init__(self, namespace):
            self.shared_metatype = SimpleNamespace(dof_names=[f'fr3_{namespace}_joint1', f'{namespace}_thumb_IP'],
                                                  link_names=[f'fr3_{namespace}_link0'])
            self.calls = []
        def set_dof_position_targets(self, value, indices):
            self.calls.append((np.asarray(value).copy(), indices))

    views = [View('front'), View('rear')]
    combined = CompositeArticulationView(views)
    indices = object()
    values = np.array([[.1, .2, .3, .4]]).view(Tensor)
    combined.set_dof_position_targets(values, indices)
    np.testing.assert_array_equal(views[0].calls[0][0], [[.1, .2]])
    np.testing.assert_array_equal(views[1].calls[0][0], [[.3, .4]])
    assert all(view.calls[0][1] is indices for view in views)
    for bad in (np.zeros((1, 3)), np.zeros((1, 5)), np.zeros((4,))):
        with pytest.raises(ValueError):
            combined.set_dof_position_targets(bad.view(Tensor), indices)
    assert [len(view.calls) for view in views] == [1, 1]


@pytest.mark.parametrize('invalid_peer', [False, True])
def test_namespaced_stage_fixed_roots_preflight_all_instances(tmp_path, invalid_peer):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics
    from bench.simulation.isaacsim_backend.robot_base import apply_base_constraint, MODE
    from bench.simulation.isaacsim_backend.robot_layout import reference_robot
    value = robot()
    references = {}
    for name in value['instances']:
        path = tmp_path / f'{name}.usda'
        source = Usd.Stage.CreateNew(str(path))
        root = UsdGeom.Xform.Define(source, '/Model').GetPrim()
        source.SetDefaultPrim(root)
        body = UsdGeom.Xform.Define(source, f'/Model/fr3_{name}_link0').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.ArticulationRootAPI.Apply(body)
        if not (invalid_peer and name == 'rear'):
            anchor = UsdPhysics.FixedJoint.Define(source, f'/Model/fr3_{name}_base_joint')
            anchor.CreateBody0Rel().SetTargets([root.GetPath()])
            anchor.CreateBody1Rel().SetTargets([body.GetPath()])
        source.GetRootLayer().Save()
        references[name] = path
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
    reference_robot(stage, root, dict(reversed(list(references.items()))), value)
    before = stage.GetRootLayer().ExportToString()
    if invalid_peer:
        with pytest.raises(ValueError, match='fixed joint'):
            apply_base_constraint(stage, root, {'robot': value, 'isaacsim': {'robot_base_constraint': MODE}})
        assert stage.GetRootLayer().ExportToString() == before
    else:
        record = apply_base_constraint(stage, root, {'robot': value, 'isaacsim': {'robot_base_constraint': MODE}})
        assert [item['root_body_name'] for item in record['articulations']] == ['fr3_front_link0', 'fr3_rear_link0']
        for name, item in zip(('front', 'rear'), record['articulations']):
            assert item['articulation_path'].startswith(f'/World/Robot/{name}/')
            np.testing.assert_allclose(item['initial_root_position_m'], value['instances'][name]['base_position_m'])
