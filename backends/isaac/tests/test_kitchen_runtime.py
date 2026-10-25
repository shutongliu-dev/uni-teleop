"""Kitchen composition preserves robot settings and binds task asset identities."""
from copy import deepcopy
import json

import numpy as np
import pytest

from bench.isaacsim.release import digest, dump
from bench.paths import load_json
from scenes.workrooms.kitchen import composition, task_robot_layout
from scenes.workrooms.mechanical_assembly import robot_layout as original_layout


def test_task_facing_changes_only_base_and_preserves_ready_joints():
    original = original_layout.robot_configuration()
    initial, _ = original_layout.ready_configuration(original)
    for task_id in task_robot_layout.TASK_IDS:
        robot = task_robot_layout.robot_configuration(task_id)
        altered = {key for key in set(original) | set(robot) if original.get(key) != robot.get(key)}
        assert altered == {'base_position_m', 'base_quaternion_wxyz'}
        reset, ready = task_robot_layout.ready_configuration(task_id, robot)
        assert reset == initial
        sign = 1 if task_id == 'pot_lid_placement' else -1
        np.testing.assert_allclose(ready['wrist_position_m'], [.65, sign*.33, 1.40], atol=1e-4)
        corrupt = deepcopy(robot)
        corrupt['arm_home_rad'][0] += .01
        with pytest.raises(ValueError, match='preserve'):
            task_robot_layout.ready_configuration(task_id, corrupt)


def test_frozen_ready_is_exact_across_source_ik_roundoff(monkeypatch):
    baseline = {task_id: (task_robot_layout.robot_configuration(task_id),
                         task_robot_layout.ready_configuration(task_id)[0])
                for task_id in task_robot_layout.TASK_IDS}
    original_solve = original_layout.solve_wrist
    def perturbed(*args, **kwargs):
        result = deepcopy(original_solve(*args, **kwargs))
        result['joint_positions_rad'] = {
            name: value+4.e-16 for name, value in result['joint_positions_rad'].items()}
        return result
    monkeypatch.setattr(original_layout, 'solve_wrist', perturbed)
    for task_id, (robot, initial) in baseline.items():
        assert task_robot_layout.robot_configuration(task_id) == robot
        assert task_robot_layout.ready_configuration(task_id, robot)[0] == initial
        corrupt = deepcopy(robot)
        corrupt['arm_kp'] += 1.e-8
        with pytest.raises(ValueError, match='preserve'):
            task_robot_layout.ready_configuration(task_id, corrupt)


def test_frozen_ready_rejects_material_source_ik_drift(monkeypatch):
    original_solve = original_layout.solve_wrist
    def perturbed(*args, **kwargs):
        result = deepcopy(original_solve(*args, **kwargs))
        result['joint_positions_rad']['fr3_joint1'] += 2.e-12
        return result
    monkeypatch.setattr(original_layout, 'solve_wrist', perturbed)
    with pytest.raises(ValueError, match='frozen kitchen v1 reset'):
        task_robot_layout.robot_configuration('breakfast_serving')


@pytest.fixture
def task_asset(tmp_path):
    metadata = {'task_id': 'breakfast_serving', 'root_path': composition.ROOT_PRIM,
        'actors': {'tray': composition.ROOT_PRIM+'/Tray'}}
    (tmp_path/'task.usda').write_text('#usda 1.0\n')
    dump(tmp_path/'metadata.json', metadata)
    dump(tmp_path/'structural_audit.json', {'status': 'passed'})
    (tmp_path/'ATTRIBUTION.txt').write_text('Original CC0-1.0 task props\n')
    manifest = {'schema': composition.ASSET_SCHEMA, 'task_id': 'breakfast_serving',
        'root_prim_path': composition.ROOT_PRIM, 'presentation': False, 'dependencies': {},
        'usd_sha256': digest(tmp_path/'task.usda'), 'metadata_sha256': digest(tmp_path/'metadata.json'),
        'structural_audit_sha256': digest(tmp_path/'structural_audit.json'),
        'attribution_sha256': digest(tmp_path/'ATTRIBUTION.txt')}
    dump(tmp_path/'manifest.json', manifest)
    return tmp_path/'manifest.json'


def test_asset_descriptor_binds_metadata_and_provenance(task_asset):
    descriptor, metadata = composition.load_asset_manifest('breakfast_serving', task_asset)
    assert len(descriptor['auxiliary_files_sha256']) == 2
    assert metadata['actors']['tray'].startswith(composition.ROOT_PRIM+'/')
    composition.load_asset_manifest('breakfast_serving', descriptor)
    (task_asset.parent/'metadata.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen input changed'):
        composition.load_asset_manifest('breakfast_serving', descriptor)


@pytest.mark.parametrize('mutation', ['presentation', 'task_id', 'root_prim_path'])
def test_mismatched_asset_cannot_become_canonical_task(task_asset, mutation):
    manifest = load_json(task_asset)
    manifest[mutation] = {'presentation': True, 'task_id': 'utensil_sorting',
        'root_prim_path': '/World/OtherTask'}[mutation]
    dump(task_asset, manifest)
    with pytest.raises(ValueError, match='dynamic kitchen reset asset'):
        composition.load_asset_manifest('breakfast_serving', task_asset)


def test_asset_actor_cannot_escape_task_subtree(task_asset):
    metadata = load_json(task_asset.parent/'metadata.json')
    metadata['actors']['tray'] = '/World/Robot'
    dump(task_asset.parent/'metadata.json', metadata)
    manifest = load_json(task_asset)
    manifest['metadata_sha256'] = digest(task_asset.parent/'metadata.json')
    dump(task_asset, manifest)
    with pytest.raises(ValueError, match='escapes'):
        composition.load_asset_manifest('breakfast_serving', task_asset)


def test_public_goal_projects_only_authored_static_descriptions():
    class Rules:
        @staticmethod
        def public_rules(task_id, seed):
            return {'instruction_zh': '按分类放入对应容器。'}

        @staticmethod
        def evaluation_configuration(task_id, metadata, **kwargs):
            return {'private_metadata': deepcopy(metadata)}

        @staticmethod
        def bind_task_rules(task):
            return task

    task = {'id': 'pantry_stocking', 'asset_metadata': {'actors': {'oil': '/World/Private'},
        'public_goal': {'objects': [{'object_id': 'oil', 'description': '绿盖橄榄油瓶',
            'category': 'bottles', 'destination': 'bottles_bin',
            'initial_pose_xyzw': [1, 2, 3, 0, 0, 0, 1], 'body_path': '/World/Private'}],
            'destinations': [{'destination_id': 'bottles_bin', 'description': '瓶类收纳盒',
                'category': 'bottles', 'support_path': '/World/PrivateBin'}],
            'context_objects': [{'object_id': 'towel', 'description': '固定折叠厨房巾'}],
            'goal_witness': {'oil': [4, 5, 6]}}}}
    bound = composition.bind_episode_task(task, 7, Rules)
    goal = bound['public_goal']
    assert goal['objects'] == [{'object_id': 'oil', 'description': '绿盖橄榄油瓶',
        'category': 'bottles', 'destination': 'bottles_bin'}]
    assert goal['destinations'] == [{'destination_id': 'bottles_bin',
        'description': '瓶类收纳盒', 'category': 'bottles'}]
    assert 'Private' not in json.dumps(goal) and 'goal_witness' not in goal
    goal['objects'][0]['description'] = 'changed'
    assert task['asset_metadata']['public_goal']['objects'][0]['description'] == '绿盖橄榄油瓶'
    del task['asset_metadata']['public_goal']
    assert composition.bind_episode_task(task, 7, Rules)['public_goal'] == {
        'operation': 'pantry_stocking', 'frame_id': 'world_m_z_up',
        'description': '按分类放入对应容器。'}


def test_generic_loader_dispatches_kitchen_and_rejects_reset_at_goal(tmp_path, monkeypatch):
    from bench.isaacsim import release as generic
    from scenes.workrooms.kitchen import release
    marker = {'schema': release.SCHEMA}
    dump(tmp_path/'release_manifest.json', marker)
    seen = []
    monkeypatch.setattr(release, 'validate_release', lambda root, manifest, *, assets: seen.append((root, assets)) or manifest)
    assert generic.validate(tmp_path, assets=False) == marker
    assert seen == [(tmp_path.resolve(), False)]
    manifest = {'schema': release.SCHEMA, 'task_id': 'breakfast_serving',
        'tasks': [{'id': 'breakfast_serving', 'scene': 'scene.json', 'task': 'task.json'}],
        'physics_profile': 'physics_profile.json'}
    monkeypatch.setattr(generic, 'validate', lambda root: manifest)
    for name, value in [('scene.json', {}), ('task.json', {}), ('physics_profile.json', {'isaacsim_options': {}})]:
        dump(tmp_path/name, value)
    with pytest.raises(ValueError, match='exact task and original reset'):
        generic.task_inputs(tmp_path, 'breakfast_serving', phase='at_goal')


@pytest.mark.parametrize('changes', [{'robot_id': 'dual_fr3_sharpa_w01'}, {'scene_variant': 'enriched_v2'}])
@pytest.mark.parametrize('task_id', task_robot_layout.TASK_IDS)
def test_service_rejects_kitchen_robot_or_scene_changes_before_loading(tmp_path, monkeypatch, changes, task_id):
    from bench.protocol import PolicyRegistry
    from bench.runtime import service
    from test_agent_service import request
    monkeypatch.setattr(service, 'validate', lambda root: {
        'schema_version': 'isaacsim-kitchen-manipulation-release-v1',
        'tasks': [{'id': task_id}]})
    calls = []
    monkeypatch.setattr(service, 'task_inputs', lambda *args, **kwargs: calls.append(kwargs))
    with pytest.raises(service.ServiceError, match='frozen single-arm'):
        service.EpisodeSession(request(task_id=task_id, **changes), tmp_path/'episode',
            release=tmp_path, device='cpu', registry=PolicyRegistry(), worker_factory=lambda **kwargs: None)
    assert not calls


@pytest.mark.parametrize('task_id', task_robot_layout.TASK_IDS)
def test_service_passes_kitchen_episode_seed_to_canonical_loader(tmp_path, monkeypatch, task_id):
    from bench.protocol import PolicyRegistry
    from bench.runtime import service
    from test_agent_service import request
    monkeypatch.setattr(service, 'validate', lambda root: {
        'schema_version': 'isaacsim-kitchen-manipulation-release-v1',
        'tasks': [{'id': task_id}]})
    calls = []
    class ReachedLoader(Exception):
        pass
    def load(root, task_id, **kwargs):
        calls.append((task_id, kwargs))
        raise ReachedLoader()
    monkeypatch.setattr(service, 'task_inputs', load)
    with pytest.raises(ReachedLoader):
        service.EpisodeSession(request(task_id=task_id, seed=71), tmp_path/'episode',
            release=tmp_path, device='cpu', registry=PolicyRegistry(), worker_factory=lambda **kwargs: None)
    assert calls == [(task_id, {'seed': 71})]
