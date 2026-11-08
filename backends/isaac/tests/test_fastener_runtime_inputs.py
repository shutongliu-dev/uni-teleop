"""Current startup verifies physical inputs without historical experiment files."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.isaacsim.release import digest, validate
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scenes.workrooms.catalog import load_task, task_definition
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.source_inputs import ASSET_IDS, INPUT_ROOT, SOURCE_ROOT

HISTORY = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/threaded_asset_native_20261016')


@pytest.mark.parametrize('configuration_id', ['single_arm', 'dual_same_side', 'dual_opposed'])
def test_current_fastener_loads_without_historical_experiment_files(monkeypatch, configuration_id):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve().is_relative_to(HISTORY):
            raise AssertionError('Current startup read historical experiment output: '+str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)
    loaded = load_task('mechanical_assembly', 'fastener_assembly', configuration_id)
    root = Path(loaded['definition']['source_release']['path'])
    manifest = validate(root)
    assert not any('/development/threaded_asset_native_' in path for path in manifest['asset_files_sha256'])
    assert loaded['scene']['mechanical_input_inventory_version'] == 2
    for asset_id in ASSET_IDS:
        load_task_asset(asset_id, INPUT_ROOT, verify_hashes=True)


def physical_input_release(robot_profile):
    """The rebound fastener release under the seed-case and robot-revision wrappers."""
    root = Path(task_definition('mechanical_assembly', 'fastener_assembly',
                                robot_profile=robot_profile)['source_release']['path'])
    manifest = load_json(root/'release_manifest.json')
    while manifest['schema'] in ('workroom_robot_revision_release.v1', 'workroom_randomized_release.v1'):
        root = Path(manifest['base_release']['path'])
        manifest = load_json(root/'release_manifest.json')
    return root, manifest


def test_rebinding_preserves_frozen_rules_reset_and_physics():
    (root, manifest), = {str(path): (path, value) for path, value in
                         map(physical_input_release, ('original', 'l_mount'))}.values()
    old = Path(manifest['physical_input_derivation']['source_release'])
    old_manifest = load_json(old/'release_manifest.json')
    changed = {'scene.json', 'source_provenance.json', 'asset_file_hashes.json'}
    for relative, row in old_manifest['files'].items():
        if relative not in changed:
            assert digest(root/relative) == row['sha256']
    original = load_json(old/'scene.json')
    current = load_json(root/'scene.json')
    current.pop('mechanical_input_inventory_version')
    for scene in (original, current):
        for spec in scene['objects'] + scene['fixtures']:
            if spec.get('asset_id') in ASSET_IDS:
                spec.pop('asset_root')
                spec.pop('usd_physics_fallbacks_provenance', None)
    assert current == original


@pytest.mark.parametrize('asset_id', ASSET_IDS)
def test_physical_asset_descriptors_are_identical(asset_id):
    original = load_json(SOURCE_ROOT/'manifest.json')['assets'][asset_id]
    current = load_json(INPUT_ROOT/'manifest.json')['assets'][asset_id]
    original = deepcopy(original)
    for spec, root in ((original, SOURCE_ROOT), (current, INPUT_ROOT)):
        spec.pop('qualification', None)
        spec.pop('source_dependencies_sha256')
        spec['recommended_spec_options'].pop('usd_physics_fallbacks_provenance', None)
        for part in spec['collision_parts'] + spec['visual_parts']:
            if 'mesh_path' in part:
                part['mesh_path'] = str((root/part['mesh_path']).resolve())
    assert current == original


def test_actual_physical_inputs_are_still_required(monkeypatch):
    manifest = load_json(INPUT_ROOT/'manifest.json')
    denied = Path(manifest['assets'][ASSET_IDS[0]]['usd_path']).resolve()
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == denied:
            raise PermissionError('physical input denied')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)
    root = Path(task_definition('mechanical_assembly', 'fastener_assembly')['source_release']['path'])
    with pytest.raises(PermissionError, match='physical input denied'):
        validate(root)
