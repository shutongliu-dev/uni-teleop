"""Deliberate dexbench differences from interface, each kept to one place.

- The default robot profile is the original robot (interface defaults to the L-mount robot).
- Copied USD layers are compared by a layer identity that ignores where the package put them.
- Bytecode caches that frozen inventories pin are provenance, not package files.
"""
import hashlib
from pathlib import Path

import pytest

import bench.paths as paths
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.parameterized.catalog import DEFAULT_PROFILE, checked_profile
from scenes.workrooms.parameterized.pairs import PROFILES


def test_default_robot_profile_is_the_original_robot():
    assert DEFAULT_PROFILE == 'original' and checked_profile(None) == 'original'
    assert checked_profile('l_mount') == 'l_mount'
    with pytest.raises(ValueError):
        checked_profile('other')
    for family, task_id, configuration in (('card_room', 'task0', 'single_arm'),
                                           ('mechanical_assembly', 'peg_insertion', 'dual_opposed')):
        default = task_definition(family, task_id, configuration)
        assert default['robot_requirements']['assembly_version'] == PROFILES['original'] == 'isaacsim_6_1_v2'
        assert default == task_definition(family, task_id, configuration, robot_profile='original')
        revised = task_definition(family, task_id, configuration, robot_profile='l_mount')
        assert revised['robot_requirements']['assembly_version'] == PROFILES['l_mount']


def test_agent_selection_and_default_service_follow_the_default_profile():
    from agent.model_client.scene_tasks import select_task
    from scenes.workrooms.robot_revision.catalog import default_service_release
    selected = select_task('peg_insertion', family='mechanical_assembly')
    assert selected.robot_requirements['assembly_version'] == 'isaacsim_6_1_v2'
    assert selected == select_task('peg_insertion', family='mechanical_assembly', robot_profile='original')
    revised = select_task('peg_insertion', family='mechanical_assembly', robot_profile='l_mount')
    assert revised.robot_requirements['assembly_version'] == 'isaacsim_6_1_l_mount_v3'
    original = task_definition('card_room', 'task0')['robot_configurations']['single_arm']['source_release']
    assert default_service_release(Path('/unused/legacy')) == Path(original['path'])


def _layer(directory, texture):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'initial.usda'
    path.write_text('#usda 1.0\ndef Shader "s"\n{\n    asset inputs:file = @' + texture + '@\n}\n')
    return path


def test_copied_layers_keep_their_identity_where_the_package_puts_them(tmp_path, monkeypatch):
    monkeypatch.setenv('DEXBENCH_DATA', str(tmp_path))
    source = _layer(tmp_path / 'assets/task/v5', './textures/a.png')
    copy = _layer(tmp_path / 'assets/seeds/v1/task/seed_00/canonical', '../../../../../task/v5/textures/a.png')
    other = _layer(tmp_path / 'assets/seeds/v1/task/seed_01/canonical', '../../../../../task/v5/textures/b.png')
    assert hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(copy.read_bytes()).digest()
    assert paths.usd_layer_identity(source) == paths.usd_layer_identity(copy)
    assert paths.usd_layer_identity(other) != paths.usd_layer_identity(source)
    binary = tmp_path / 'assets/task/v5/initial.usdc'
    binary.write_bytes(b'PXR-USDC\0')
    assert paths.usd_layer_identity(binary) == hashlib.sha256(b'PXR-USDC\0').hexdigest()


def test_seed_zero_accepts_a_relocated_copy_of_its_canonical_layer(tmp_path, monkeypatch):
    from scenes.workrooms.initial_randomization import physical_inputs
    monkeypatch.setenv('DEXBENCH_DATA', str(tmp_path))
    source = _layer(tmp_path / 'assets/task/v5', './textures/a.png')
    copy = _layer(tmp_path / 'assets/seeds/v1/task/seed_00/canonical', '../../../../../task/v5/textures/a.png')
    other = _layer(tmp_path / 'assets/seeds/v1/task/seed_01/canonical', '../../../../../task/v5/textures/b.png')

    def scene(path, sha=None):
        sha = sha or hashlib.sha256(path.read_bytes()).hexdigest()
        return {'objects': [], 'fixtures': [], 'task_layers': [{'id': 'task', 'usd_path': str(path), 'sha256': sha}]}

    physical_inputs.check_canonical_physics(scene(source), scene(copy))
    with pytest.raises(ValueError, match='task layer'):
        physical_inputs.check_canonical_physics(scene(source), scene(other))
    with pytest.raises(ValueError, match='task layer'):  # a wrong pin is never replaced by the layer identity
        physical_inputs.check_canonical_physics(scene(source), scene(copy, sha='0' * 64))


def test_pinned_bytecode_caches_are_undistributed_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA_ROOT', str(tmp_path))
    cache = tmp_path / 'datasets/releases/x/evaluator/__pycache__/rules.cpython-312.pyc'
    assert paths.undistributed_bytecode(cache)
    assert not paths.undistributed_bytecode(cache.with_name('rules.py'))
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'')
    assert not paths.undistributed_bytecode(cache)  # shipped files are checked as usual
