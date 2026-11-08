"""Read-only checks of prepared physical inputs; mutations affect copies only.

These tests load actual frozen releases/assets without creating native workers,
rendering, model requests or report-based admission dependencies.
"""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import stat

import pytest

from bench.isaacsim.release import task_inputs, validate
from bench.paths import DATA_ROOT, usd_layer_identity
from bench.runtime.worker import public_task_projection
from scenes.workrooms.initial_randomization.catalog import bind_definition
from scenes.workrooms.initial_randomization.common import file_hash, read_json, write_json
from scenes.workrooms.initial_randomization.release import _parent_configuration
from scenes.workrooms.initial_randomization.physical_inputs import (
    check_actor_inventory, check_canonical_physics,
)


RELEASES = Path(os.environ.get('WORKROOM_SEED_TEST_RELEASES',
    f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/workroom_initial_randomization_v1/publication01/releases'))
TASKS = ('stationery_sorting', 'book_stacking', 'book_shelving')
CONFIGURATIONS = ('single_arm', 'dual_same_side', 'dual_opposed')
CASES = [(task, configuration) for task in TASKS for configuration in CONFIGURATIONS]
CASES += [(task, 'single_arm') for task in (
    'produce_transfer', 'utensil_sorting', 'pantry_stocking', 'table_clearing')]


def prepared(task_id='book_stacking', configuration='single_arm'):
    root = RELEASES/task_id/configuration
    if not (root/'release_manifest.json').is_file():
        pytest.skip('Independent prepared physical input has not been published')
    return root


@pytest.mark.parametrize('task_id,configuration', CASES)
def test_actual_seed_roundtrip_preserves_installation_and_public_geometry(task_id, configuration):
    root = prepared(task_id, configuration)
    manifest = validate(root)
    parent = Path(manifest['base_release']['path'])
    original_scene = read_json(parent/'scene.json')
    original_initial = read_json(parent/'initial_state.json')
    identities = set()
    for seed in (0, 1, 9):
        entry, scene, task, initial = task_inputs(root, task_id, seed=seed)
        assert entry['task_kind'] == 'single_agent'
        assert _parent_configuration(scene) == configuration
        assert initial == original_initial
        for field in ('robot', 'workroom', 'isaacsim', 'physics', 'controller', 'policy', 'robot_configuration', 'cameras'):
            assert scene.get(field) == original_scene.get(field), field
        assert task['reset_case']['seed'] == seed
        assert task['reset_case']['configuration_id'] == configuration
        identities.add(task['reset_case']['physical_geometry_fingerprint'])
        public = public_task_projection(task)['reset_case']
        assert public['case_sha256'] == task['reset_case']['case_sha256']
        assert public['static_geometry'] == task['reset_case']['static_geometry']
        assert 'actual_body_poses_m' not in public and 'interaction_targets' not in public
        if seed == 0:
            assert scene.get('objects') == original_scene.get('objects')
            assert scene.get('fixtures') == original_scene.get('fixtures')
            # dexbench: a relocated copy keeps its parent layer's identity (bench.paths.usd_layer_identity);
            # its bytes differ by the layer-relative asset paths the package writes.
            for layer, original in zip(scene.get('task_layers', []), original_scene.get('task_layers', [])):
                assert file_hash(layer['usd_path']) == layer['sha256']
                assert file_hash(original['usd_path']) == original['sha256']
                assert usd_layer_identity(layer['usd_path']) == usd_layer_identity(original['usd_path'])
            if 'native_task_bundle' in scene:
                bundle, original = scene['native_task_bundle'], original_scene['native_task_bundle']
                assert file_hash(bundle['usd_path']) == bundle['usd_sha256']
                assert file_hash(original['usd_path']) == original['usd_sha256']
                assert usd_layer_identity(bundle['usd_path']) == usd_layer_identity(original['usd_path'])
    assert len(identities) == 3


def _copy_release(tmp_path):
    root = tmp_path/'candidate'
    shutil.copytree(prepared(), root, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Published inputs are sealed read-only. Only the independent temporary
    # copy needs owner-write permissions for deliberate corruption below.
    for path in (root, *root.rglob('*')):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    return root, read_json(root/'release_manifest.json')


def test_canonical_guard_compares_actual_layer_bytes_not_just_its_case_label():
    root = prepared()
    manifest = read_json(root/'release_manifest.json')
    parent = read_json(Path(manifest['base_release']['path'])/'scene.json')
    case = read_json(root/manifest['cases']['0']['scene'])
    check_canonical_physics(parent, case)
    changed = deepcopy(case)
    changed['task_layers'][0]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='canonical task layer'):
        check_canonical_physics(parent, changed)


def test_actor_guard_rejects_matching_public_private_omission_from_actual_scene():
    root = prepared()
    manifest = read_json(root/'release_manifest.json')
    parent = read_json(Path(manifest['base_release']['path'])/'scene.json')
    scene = read_json(root/manifest['cases']['1']['scene'])
    case = read_json(root/manifest['cases']['1']['case'])
    check_actor_inventory(parent, scene, case['actual_body_poses_m'])
    poses = deepcopy(case['actual_body_poses_m'])
    poses.pop(next(iter(poses)))
    with pytest.raises(ValueError, match='actual scene declarations'):
        check_actor_inventory(parent, scene, poses)


def _reseal(root, manifest, relative, value):
    write_json(root/relative, value)
    sha = file_hash(root/relative)
    manifest['files'][relative]['sha256'] = sha
    manifest['files_sha256'][relative] = sha
    write_json(root/'release_manifest.json', manifest)


@pytest.mark.parametrize('fault', ('public_mass', 'private_pose', 'usd_binding', 'robot', 'missing_actor'))
def test_resealed_case_cannot_drift_from_real_asset_and_original_parent(tmp_path, fault):
    root, manifest = _copy_release(tmp_path)
    record = manifest['cases']['1']
    kind = 'case' if fault in ('public_mass', 'private_pose', 'missing_actor') else 'scene'
    value = read_json(root/record[kind])
    if fault == 'public_mass':
        value['static_geometry']['bodies'][0]['nominal_mass_kg'] *= 1.1
    elif fault == 'private_pose':
        next(iter(value['actual_body_poses_m'].values()))['position_m'][0] += .02
    elif fault == 'missing_actor':
        name = next(iter(value['actual_body_poses_m']))
        value['actual_body_poses_m'].pop(name)
        value['static_geometry']['bodies'] = [r for r in value['static_geometry']['bodies'] if r['object_id'] != name]
    elif fault == 'usd_binding':
        value['task_layers'][0]['sha256'] = '0'*64
    elif fault == 'robot':
        value['robot']['seed_review_unapproved_field'] = True
    _reseal(root, manifest, record[kind], value)
    with pytest.raises(ValueError):
        validate(root)


def test_release_rejects_wrong_parent_configuration_and_asset_hash(tmp_path):
    root, manifest = _copy_release(tmp_path)
    manifest['configuration_id'] = 'dual_opposed'
    write_json(root/'release_manifest.json', manifest)
    with pytest.raises(ValueError, match='installation'):
        validate(root)
    manifest['configuration_id'] = 'single_arm'
    manifest['cases']['1']['asset_manifest_sha256'] = '0'*64
    write_json(root/'release_manifest.json', manifest)
    with pytest.raises(ValueError, match='input differs'):
        validate(root)


@pytest.mark.parametrize('fault', ('missing_snapshot', 'missing_pin', 'incorrect_pin'))
def test_source_snapshot_is_required_even_after_outer_hashes_are_resealed(tmp_path, fault):
    root, manifest = _copy_release(tmp_path)
    value = read_json(root/'source_provenance.json')
    assert value['source_snapshot']['verified'] is True
    assert value['source_manifest_sha256']
    if fault == 'missing_snapshot':
        value.pop('source_snapshot')
    elif fault == 'missing_pin':
        value.pop('source_manifest_sha256')
    else:
        value['source_manifest_sha256'] = '0'*64
    _reseal(root, manifest, 'source_provenance.json', value)
    with pytest.raises(ValueError):
        validate(root)


# Kitchen breakfast/pot-lid/wiping and the mechanical placement tasks joined
# the finite reset scope (PLACEMENT_TASK_FAMILIES); these remain outside it.
@pytest.mark.parametrize('task_id,family', (
    ('tower_building', 'living_room'), ('tower_extraction', 'living_room'),
    ('polyomino_territory', 'card_room'), ('polyomino_territory_practice', 'card_room'),
    ('remote_operation', 'living_room'),
))
def test_excluded_tasks_never_read_the_new_pin_catalog(monkeypatch, task_id, family):
    import scenes.workrooms.initial_randomization.catalog as catalog
    assert task_id not in catalog.TASK_FAMILIES
    def forbidden(*args, **kwargs):
        raise AssertionError('Excluded task consulted new publication pins')
    monkeypatch.setattr(catalog, 'read_json', forbidden)
    original = {'task_id': task_id, 'family': family, 'source_release': {'path': 'unchanged'}}
    assert bind_definition(original) is original
