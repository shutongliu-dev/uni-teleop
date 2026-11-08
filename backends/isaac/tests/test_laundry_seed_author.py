"""Actual-parent CPU derivation checks; no native launches or model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scenes.workrooms.laundry_room.robot_configuration_catalog import RELEASES
from scenes.workrooms.laundry_room.randomization import CONFIGURATION_IDS, TASK_IDS
from scenes.workrooms.laundry_room.randomization import appearance, author, cases, common, release
from scenes.workrooms.laundry_room.randomization.public import project_reset_case


def parent(task='towel_folding', configuration='single_arm'):
    return Path(RELEASES[task]['configurations'][configuration]['source_release']['path'])


@pytest.fixture(scope='module')
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp('laundry_seed_author')
    patch = pytest.MonkeyPatch()
    asset_root, release_root = root/'assets', root/'releases'
    patch.setattr(common, 'ASSET_ROOT', asset_root)
    patch.setattr(common, 'RELEASE_ROOT', release_root)
    patch.setattr(appearance, 'ASSET_ROOT', asset_root)
    from scenes.workrooms.laundry_room import fabric_visual_release
    patch.setattr(fabric_visual_release, 'ASSETS', asset_root)
    snapshot = root/'test_source'; snapshot.mkdir()
    hashes = {}
    for name in common.SOURCE_FILES:
        destination = snapshot/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common.SOURCE_ROOT/name, destination)
        hashes[name] = common.file_hash(destination)
    common.write_json(snapshot/'source_snapshot_manifest.json', {
        'schema': 'complete_benchmark_source_snapshot_v1', 'files_sha256': hashes,
        'test_fixture_only': True})
    provenance = {'source_snapshot': {'snapshot_root': str(snapshot), 'runtime_imports_verified': True,
                                      'test_fixture_only': True},
                  'source_manifest_sha256': common.file_hash(snapshot/'source_snapshot_manifest.json'),
                  'author_sources_sha256': hashes}
    patch.setattr(common, 'source_provenance', lambda: deepcopy(provenance))
    patch.setattr(appearance, 'source_provenance', lambda: deepcopy(provenance))
    original = parent()
    scene = common.read_json(original/'scene.json')
    pin = scene['cloths'][0]['visual_profile']
    assets = {}
    for seed in range(10):
        profiles = None if seed == 0 else {'towel': appearance.prepare_profile(
            pin, seed, asset_root/'appearance'/f'seed_{seed:02d}')}
        assets[seed] = author.prepare_asset(original, seed, asset_root/'cases'/f'seed_{seed:02d}',
                                           visual_profiles=profiles)
    destination = release_root/'towel_folding/single_arm'
    manifest = release.prepare_release(original, destination, assets_by_seed=assets, configuration_id='single_arm')
    yield {'root': destination, 'manifest': manifest, 'parent': original, 'assets': assets,
           'asset_root': asset_root, 'release_root': release_root, 'source': provenance}
    patch.undo()


@pytest.mark.parametrize('task_id', TASK_IDS)
@pytest.mark.parametrize('configuration', CONFIGURATION_IDS)
def test_all_original_installations_only_translate_initial_movable_cloth(task_id, configuration):
    metadata = common.read_json(parent(task_id, configuration)/'task.json')['asset_metadata']
    fingerprints = set()
    for seed in range(10):
        plan = cases.case_plan(task_id, configuration, seed, metadata)
        new = cases.transformed_metadata(metadata, plan)
        fingerprints.add(common.canonical_hash([r['initial_vertices_m'] for r in new['cloths']]))
        if seed == 0:
            assert new == metadata
        for original, current in zip(metadata['cloths'], new['cloths'], strict=True):
            delta = np.asarray(current['initial_vertices_m'])-np.asarray(original['initial_vertices_m'])
            assert np.max(np.abs(delta[:, :2]-plan['translations_xy_m'][original['id']])) < 2e-16
            assert np.array_equal(delta[:, 2], np.zeros(len(delta)))
            assert {k: v for k, v in original.items() if k != 'initial_vertices_m'} == {
                k: v for k, v in current.items() if k != 'initial_vertices_m'}
            if original['id'] == 'towel_base':
                assert original == current
        assert new['target_center_xy_m'] == metadata['target_center_xy_m']
    assert len(fingerprints) == 10


def test_declared_edge_bound_clips_only_tight_shirt_direction():
    metadata = common.read_json(parent('shirt_folding', 'dual_same_side')/'task.json')['asset_metadata']
    plan = cases.case_plan('shirt_folding', 'dual_same_side', 1, metadata)
    assert plan['translations_xy_m']['towel'][0] == -.04
    assert -.009 < plan['translations_xy_m']['towel'][1] < -.008
    other = cases.case_plan('shirt_folding', 'dual_same_side', 4, metadata)
    assert other['translations_xy_m']['towel'] == [.04, .04]


@pytest.mark.parametrize('seed', [True, -1, 10, 1.0, '1', None])
def test_finite_seed_selection_rejects_noninteger_and_out_of_range(seed):
    metadata = common.read_json(parent()/'task.json')['asset_metadata']
    with pytest.raises(ValueError, match='seed'):
        cases.case_plan('towel_folding', 'single_arm', seed, metadata)


def test_full_frozen_roundtrip_and_public_privacy(candidate):
    from bench.isaacsim.release import load_evaluator, task_inputs, validate
    manifest = validate(candidate['root'])
    original_scene, original_task = [common.read_json(candidate['parent']/(n+'.json')) for n in ('scene', 'task')]
    for seed in range(10):
        _, scene, task, initial = task_inputs(candidate['root'], 'towel_folding', seed=seed)
        assert initial == task['initial_state']
        assert task['reset_case']['seed'] == seed
        assert task['public_goal'] == original_task['public_goal']
        assert task['evaluation']['thresholds'] == original_task['evaluation']['thresholds']
        assert task['public_rules']['referee'] == original_task['public_rules']['referee']
        for key in ('robot', 'physics', 'isaacsim', 'workroom', 'cameras', 'robot_configuration'):
            assert scene[key] == original_scene[key]
        public = project_reset_case(task, {'task_id': 'towel_folding'})
        assert set(public['reset_case']) == {'schema', 'family', 'task_id', 'configuration_id',
            'seed', 'case_sha256', 'case_fingerprint', 'parent_release_sha256', 'variation', 'scope'}
        assert all(word not in json.dumps(public) for word in ('translations_xy_m', 'initial_vertices', 'plan', 'asset_reference'))
        load_evaluator(candidate['root'])(task)
    assert len(manifest['cases']) == 10


def test_seed_zero_exact_appearance_geometry_and_single_literal_evaluator_derivation(candidate):
    root, old = candidate['root'], candidate['parent']
    scene, original_scene = [common.read_json(p/'scene.json') for p in (root, old)]
    assert scene['cloths'] == original_scene['cloths']
    assert (root/'initial_state.json').read_bytes() == (old/'initial_state.json').read_bytes()
    for name in ('sequence_evaluator.py', 'native_observer.py'):
        assert (root/'evaluator'/name).read_bytes() == (old/'evaluator'/name).read_bytes()
    assert (root/'evaluator/rules.py').read_text() == (old/'evaluator/rules.py').read_text().replace(
        release.OLD_DESCRIPTION, release.NEW_DESCRIPTION)
    assert (root/'physics_profile.json').read_bytes() == (old/'physics_profile.json').read_bytes()


@pytest.mark.parametrize('change', ['cloth_mass', 'rest_vertex', 'robot', 'threshold', 'initial_xy', 'visual_normal'])
def test_rehashed_case_tampering_is_rejected(candidate, tmp_path, change):
    copied = tmp_path/'tampered'; shutil.copytree(candidate['root'], copied)
    manifest = common.read_json(copied/'release_manifest.json')
    record = manifest['cases']['1']
    kind = 'scene' if change in ('robot', 'initial_xy', 'visual_normal') else 'task'
    path = copied/record[kind]; value = common.read_json(path)
    if change == 'cloth_mass':
        value['asset_metadata']['cloths'][0]['mass_kg'] *= 1.1
    elif change == 'rest_vertex':
        value['asset_metadata']['cloths'][0]['rest_vertices_m'][0][0] += .001
    elif change == 'robot':
        value['robot']['arm_home_rad'][0] += .01
    elif change == 'threshold':
        value['evaluation']['thresholds']['stable_s'] = .1
    elif change == 'initial_xy':
        value['cloths'][0]['initial_vertices_m'][0][0] += .001
    else:
        value['cloths'][0]['isaacsim']['cloth_visual_color'][0] += .01
    common.write_json(path, value)
    manifest['files'][record[kind]]['sha256'] = manifest['files_sha256'][record[kind]] = common.file_hash(path)
    common.write_json(copied/'release_manifest.json', manifest)
    with pytest.raises(ValueError, match='differs'):
        release.validate_release(copied, manifest)


def test_unknown_source_and_provenance_missing_are_rejected(candidate, tmp_path):
    copied = tmp_path/'tampered'; shutil.copytree(candidate['root'], copied)
    manifest = common.read_json(copied/'release_manifest.json')
    path = copied/'source_provenance.json'; value = common.read_json(path)
    value.pop('source_manifest_sha256'); common.write_json(path, value)
    manifest['files'][path.name]['sha256'] = manifest['files_sha256'][path.name] = common.file_hash(path)
    common.write_json(copied/'release_manifest.json', manifest)
    with pytest.raises(ValueError, match='Frozen input differs'):
        release.validate_release(copied, manifest)


def test_public_projection_rejects_other_task_and_broken_private_hash(candidate):
    task = common.read_json(candidate['root']/'task.json')
    with pytest.raises(ValueError, match='identities'):
        project_reset_case(task, {'task_id': 'towel_stacking'})
    task['reset_case']['plan']['translations_xy_m']['towel'][0] = .2
    with pytest.raises(ValueError, match='Unbound'):
        project_reset_case(task, {'task_id': task['id']})


def test_catalog_resources_remain_packaged_and_frozen():
    import tomllib
    from scripts.maintenance.freeze_project_sources import DEFAULT_EXTRAS
    pyproject = tomllib.loads((common.SOURCE_ROOT/'pyproject.toml').read_text())
    assert pyproject['tool']['setuptools']['package-data']['scenes.workrooms.laundry_room.randomization'] == ['catalog_pins.json']
    assert 'scenes/workrooms/laundry_room/randomization/catalog_pins.json' in DEFAULT_EXTRAS


def test_empty_pins_preserve_original_all_five_catalog_installations(tmp_path, monkeypatch):
    from scenes.workrooms.catalog import task_definition
    from scenes.workrooms.laundry_room.randomization import catalog
    from scenes.workrooms.parameterized import catalog as parameterized
    from scenes.workrooms.robot_revision import catalog as revised_catalog
    path = tmp_path/'empty_pins.json'
    common.write_json(path, {'schema': 'laundry_seed_catalog.v1', 'tasks': {}})
    monkeypatch.setattr(catalog, 'PINS', path)
    # The published robot overlays pin the seed releases as their parents and
    # must refuse the changed inventory rather than fall back to it.
    for profile in ('original', 'l_mount'):
        with pytest.raises(ValueError, match='inventory'):
            task_definition('laundry_room', 'towel_folding', robot_profile=profile)
    common.write_json(tmp_path/'revision_pins.json', {'schema': 'robot_revision_catalog.v1', 'tasks': {}})
    monkeypatch.setattr(revised_catalog, 'PINS', tmp_path/'revision_pins.json')
    common.write_json(tmp_path/'parameterized_pins.json', {'schema': parameterized.SCHEMA, 'tasks': {}})
    monkeypatch.setattr(parameterized, 'PINS', tmp_path/'parameterized_pins.json')
    for task in TASK_IDS:
        for configuration in CONFIGURATION_IDS:
            value = task_definition('laundry_room', task, configuration)
            assert value['source_release'] == RELEASES[task]['configurations'][configuration]['source_release']
