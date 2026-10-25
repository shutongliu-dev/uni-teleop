"""CPU freeze and canonical loader checks; no simulator or native acceptance."""
from copy import deepcopy
import shutil

import pytest

from bench.isaacsim.release import digest, dump, load_evaluator, task_inputs, validate
from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly import release
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.rules import thresholds
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.task import build_task_inputs


@pytest.fixture(scope='module')
def frozen_template(tmp_path_factory):
    parent = tmp_path_factory.mktemp('fastener_release')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(release, 'DATASET_ROOT', parent)
        root = parent/'candidate'
        manifest = release.prepare_task_release(root)
    return root, manifest


@pytest.fixture
def frozen(tmp_path, monkeypatch, frozen_template):
    monkeypatch.setattr(release, 'DATASET_ROOT', tmp_path)
    source, manifest = frozen_template
    root = tmp_path/'candidate'
    shutil.copytree(source, root)
    return root, deepcopy(manifest)


def repin(root, relative, value):
    dump(root/relative, value)
    manifest = load_json(root/'release_manifest.json')
    manifest['files'][relative]['sha256'] = digest(root/relative)
    if relative == 'physics_profile.json':
        manifest['physics_profile_sha256'] = digest(root/relative)
    dump(root/'release_manifest.json', manifest)


def test_new_release_loads_through_canonical_worker_with_original_assets(frozen):
    root, manifest = frozen
    assert validate(root) == manifest
    entry, scene, task, initial = task_inputs(root, 'fastener_assembly')
    assert entry['title_zh'] and all(row['instruction_zh'] for row in task['steps'])
    assert {row['id'] for row in scene['objects']} == {'bolt', 'screwdriver'}
    assert {row['id'] for row in scene['fixtures']} == {'robot_stand', 'threaded_fixture'}
    assert len(initial['joint_positions_rad']) == 29
    assert initial == task['initial_state']
    assert load_evaluator(root)(task).task == task
    assert task['public_rules']['success']['nominal_turn_degrees'] == 30.
    assert task['public_rules']['success']['nominal_additional_advance_m'] == .00025
    assert not manifest['physics_accepted'] and not manifest['task_policy_success_claimed']
    inventory = load_json(root/'asset_file_hashes.json')
    assert any('/threaded_sharpa/v3/' in path for path in inventory)
    assert not list(root.rglob('*.usd')) and not list(root.rglob('*.usda'))
    # The prepared task changes inventory only; it preserves the proven source profile/reset.
    from scenes.workrooms.mechanical_assembly.dynamics import preinserted_scene_v3
    original, original_initial, _, _ = preinserted_scene_v3(release.STAND_ROOT, release.THREADED_ROOT)
    assert initial == original_initial
    for key in ('isaacsim', 'physics', 'robot', 'fixtures', 'workroom'):
        assert scene[key] == original[key]
    assert scene['objects'] == [row for row in original['objects'] if row['id'] in ('bolt', 'screwdriver')]


def test_release_cannot_overwrite_or_reset_at_goal(frozen):
    root, _ = frozen
    with pytest.raises(FileExistsError):
        release.prepare_task_release(root)
    with pytest.raises(ValueError, match='reset-at-goal'):
        task_inputs(root, 'fastener_assembly', 'qualification_probe')


def test_source_or_frozen_tamper_is_detected(frozen):
    root, _ = frozen
    path = root/'evaluator'/'rules.py'
    path.write_text(path.read_text()+'\n# unpinned edit\n')
    with pytest.raises(ValueError, match='Frozen input changed'):
        load_evaluator(root)


@pytest.mark.parametrize('damage', ['extra_reset_object', 'missing_joint', 'unknown_joint', 'nonfinite_joint',
                                    'joint_out_of_range', 'wrong_schema', 'different_initial_file', 'physics_change',
                                    'acceptance_claim', 'missing_evaluator'])
def test_even_repinned_invalid_mechanical_contract_is_rejected(frozen, damage):
    root, manifest = frozen
    task = load_json(root/'task.json')
    joints = task['initial_state']['joint_positions_rad']
    if damage == 'extra_reset_object': task['initial_state']['objects'] = {'bolt': {'position_m': [0,0,0]}}
    elif damage == 'missing_joint': joints.pop(next(iter(joints)))
    elif damage == 'unknown_joint': joints['unsupported_joint'] = 0.
    elif damage == 'nonfinite_joint': joints[next(iter(joints))] = 'nan'
    elif damage == 'joint_out_of_range': joints[next(iter(joints))] = 100.
    elif damage == 'wrong_schema': task['schema_version'] = 'cloth-manipulation-task-v1'
    elif damage == 'different_initial_file': repin(root, 'initial_state.json', {'joint_positions_rad': {}})
    elif damage == 'physics_change':
        profile = load_json(root/'physics_profile.json')
        profile['isaacsim_options']['physics_dt_s'] = .001
        repin(root, 'physics_profile.json', profile)
    elif damage == 'acceptance_claim':
        manifest['physics_accepted'] = True
        dump(root/'release_manifest.json', manifest)
    else:
        del manifest['files']['evaluator/sequence_evaluator.py']
        dump(root/'release_manifest.json', manifest)
    if damage in {'extra_reset_object','missing_joint','unknown_joint','nonfinite_joint','joint_out_of_range','wrong_schema'}:
        repin(root, 'task.json', task)
    with pytest.raises((ValueError, KeyError)):
        validate(root, assets=False)


def test_goal_is_explicitly_parameterized_and_invalid_goals_are_rejected():
    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.evaluator import FastenerEvaluator
    values = build_task_inputs(target_turn_degrees=20.)
    task = values['task']
    assert task['public_rules']['success']['nominal_turn_degrees'] == 20.
    assert task['public_rules']['success']['nominal_additional_advance_m'] == pytest.approx(.003*20/360)
    assert FastenerEvaluator(task).c['nominal_turn_rad'] == thresholds(20.)['nominal_turn_rad']
    tampered = deepcopy(task)
    tampered['evaluation']['minimum_turn_rad'] = 0.
    with pytest.raises(ValueError, match='threshold differs'):
        FastenerEvaluator(tampered)
    for value in [0., -30., 46., float('nan'), True, '30']:
        with pytest.raises(ValueError): thresholds(value)


def test_tool_clocking_is_reset_only_and_preserves_bolt_axis_physics_and_goal(tmp_path, monkeypatch):
    import numpy as np
    from bench.geometry import quaternion_matrix, rotvec_matrix
    original = build_task_inputs()
    clocked = build_task_inputs(tool_clocking_degrees=-30.)
    assert clocked['initial_state'] == original['initial_state']
    for key in ('robot', 'fixtures', 'isaacsim', 'physics', 'workroom'):
        assert clocked['scene'][key] == original['scene'][key]
    old_objects = {row['id']: row for row in original['scene']['objects']}
    new_objects = {row['id']: row for row in clocked['scene']['objects']}
    assert new_objects['bolt'] == old_objects['bolt']
    old_tool, new_tool = old_objects['screwdriver'], new_objects['screwdriver']
    assert {k:v for k,v in old_tool.items() if k != 'quaternion_wxyz'} == {
        k:v for k,v in new_tool.items() if k != 'quaternion_wxyz'}
    before, after = (quaternion_matrix(t['quaternion_wxyz']) for t in (old_tool, new_tool))
    np.testing.assert_allclose(after[:, 2], before[:, 2], atol=1e-14)
    np.testing.assert_allclose(before.T @ after, rotvec_matrix([0., 0., -np.pi/6]), atol=1e-14)
    assert clocked['task']['public_rules'] == original['task']['public_rules']
    assert clocked['task']['reset_conditions']['task_credit'] == 0
    monkeypatch.setattr(release, 'DATASET_ROOT', tmp_path)
    root = tmp_path/'clocked'
    release.prepare_task_release(root, tool_clocking_degrees=-30.)
    _, scene, task, initial = task_inputs(root, 'fastener_assembly')
    assert initial == original['initial_state']
    assert scene['objects'] == clocked['scene']['objects']
    evaluator = load_evaluator(root)(task)
    assert not evaluator.success and evaluator.contact_backed_turn == 0
    np.testing.assert_allclose(evaluator.expected['screwdriver']['r'], after, atol=1e-14)
    for value in [-30.1, 1., float('nan'), True, '-30']:
        with pytest.raises(ValueError, match='clocking'):
            build_task_inputs(tool_clocking_degrees=value)
