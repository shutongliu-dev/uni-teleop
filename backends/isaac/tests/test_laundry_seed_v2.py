"""Explicit near-robot shirt cases preserve all v1 inputs and all physical properties."""
from copy import deepcopy
import inspect
from pathlib import Path
import shutil

import numpy as np
import pytest

from bench.paths import DATA_ROOT
from scenes.workrooms.laundry_room.randomization import (CONFIGURATION_IDS, GENERATOR_VERSION,
    SHIRT_NEAR_GENERATOR_VERSION, TASK_IDS)
from scenes.workrooms.laundry_room.randomization import author, cases, common, release, validation_cache
from scenes.workrooms.laundry_room.robot_configuration_catalog import RELEASES


def parent(task, cfg):
    return Path(RELEASES[task]['configurations'][cfg]['source_release']['path'])


@pytest.mark.parametrize('cfg', ('single_arm', 'dual_same_side'))
def test_twenty_near_robot_positions_are_unique_and_only_rigid_xy(cfg):
    metadata = common.read_json(parent('shirt_folding', cfg)/'task.json')['asset_metadata']
    positions = []
    directions = [(0, 0), (-1, -1), (1, -1), (-1, -.5), (1, -.5),
                  (-1, 0), (1, 0), (0, -1), (0, -.5), (.5, -.5)]
    for seed, direction in enumerate(directions):
        plan = cases.case_plan('shirt_folding', cfg, seed, metadata, generator_version=SHIRT_NEAR_GENERATOR_VERSION)
        xy = plan['translations_xy_m']['towel']; positions.append(tuple(xy))
        y_amplitude = .02 if cfg == 'single_arm' else .008008070468902508
        assert np.allclose(xy, [direction[0]*.02, direction[1]*y_amplitude], atol=1e-15, rtol=0)
        assert xy[1] <= 0 and plan['source_bounds_guard']['maximum_axis_translation_m'] == .02
        current = cases.transformed_metadata(metadata, plan)
        assert current['target_center_xy_m'] == metadata['target_center_xy_m']
        for old, new in zip(metadata['cloths'], current['cloths'], strict=True):
            delta = np.asarray(new['initial_vertices_m'])-old['initial_vertices_m']
            assert np.allclose(delta[:, :2], xy, atol=2e-16, rtol=0)
            assert np.array_equal(delta[:, 2], np.zeros(len(delta)))
            assert {k: v for k, v in old.items() if k != 'initial_vertices_m'} == {
                k: v for k, v in new.items() if k != 'initial_vertices_m'}
        if seed == 0:
            assert current == metadata
    assert len(set(positions)) == 10


@pytest.mark.parametrize('task,cfg', [(t, c) for t in TASK_IDS for c in CONFIGURATION_IDS
                                    if (t, c) not in (('shirt_folding', 'single_arm'), ('shirt_folding', 'dual_same_side'))])
def test_v2_is_rejected_for_every_other_task_and_installation(task, cfg):
    metadata = common.read_json(parent(task, cfg)/'task.json')['asset_metadata']
    with pytest.raises(ValueError, match='generator'):
        cases.case_plan(task, cfg, 0, metadata, generator_version=SHIRT_NEAR_GENERATOR_VERSION)


@pytest.mark.parametrize('version', [None, '', 'v2', 'laundry_xy_fabric_seed_v3', True])
def test_unknown_version_is_not_silently_mapped_to_default(version):
    metadata = common.read_json(parent('shirt_folding', 'single_arm')/'task.json')['asset_metadata']
    with pytest.raises(ValueError, match='generator'):
        cases.case_plan('shirt_folding', 'single_arm', 1, metadata, generator_version=version)


def test_all_150_actual_v1_plans_remain_exact_and_default_stays_v1():
    assert inspect.signature(cases.case_plan).parameters['generator_version'].default == GENERATOR_VERSION
    assert inspect.signature(author.prepare_asset).parameters['generator_version'].default == GENERATOR_VERSION
    for task in TASK_IDS:
        for cfg in CONFIGURATION_IDS:
            metadata = common.read_json(parent(task, cfg)/'task.json')['asset_metadata']
            root = common.RELEASE_ROOT/'publication01/releases'/task/cfg
            manifest = common.read_json(root/'release_manifest.json')
            for seed in range(10):
                asset = Path(manifest['cases'][str(seed)]['asset_manifest'])
                old = common.read_json(asset.parent/'plan.json')
                assert cases.case_plan(task, cfg, seed, metadata) == old
                assert cases.validate_plan(old, metadata) == old


@pytest.fixture(scope='module')
def candidates(tmp_path_factory):
    root = tmp_path_factory.mktemp('laundry_shirt_v2'); patch = pytest.MonkeyPatch()
    snapshot = root/'test_source'; snapshot.mkdir(); hashes = {}
    for name in common.SOURCE_FILES:
        target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common.SOURCE_ROOT/name, target); hashes[name] = common.file_hash(target)
    common.write_json(snapshot/'source_snapshot_manifest.json', {
        'schema': 'complete_benchmark_source_snapshot_v1', 'files_sha256': hashes, 'test_fixture_only': True})
    provenance = {'source_snapshot': {'snapshot_root': str(snapshot), 'runtime_imports_verified': True,
                                     'test_fixture_only': True},
        'source_manifest_sha256': common.file_hash(snapshot/'source_snapshot_manifest.json'),
        'author_sources_sha256': hashes}
    # Unit authoring is confined to tmp_path; actual read-only appearance pins
    # remain under the original shared assets, without copying or baking maps.
    patch.setattr(common, 'ASSET_ROOT', Path('/'))
    patch.setattr(common, 'RELEASE_ROOT', root/'releases')
    patch.setattr(common, 'source_provenance', lambda: deepcopy(provenance))
    from scenes.workrooms.laundry_room.randomization import appearance
    patch.setattr(appearance, 'prepare_profile', lambda *_a, **_k: pytest.fail('v2 must reuse original appearance'))
    results = {}
    shared_releases = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/laundry_initial_randomization_v1/publication01/releases')
    for cfg in ('single_arm', 'dual_same_side'):
        old = parent('shirt_folding', cfg); existing = shared_releases/'shirt_folding'/cfg
        old_manifest = common.read_json(existing/'release_manifest.json'); seed_assets = {}
        for seed in range(10):
            scene = common.read_json(existing/old_manifest['cases'][str(seed)]['scene'])
            profiles = {r['id']: r['visual_profile'] for r in scene['cloths']}
            seed_assets[seed] = author.prepare_asset(old, seed, root/'assets'/cfg/str(seed),
                visual_profiles=profiles, generator_version=SHIRT_NEAR_GENERATOR_VERSION)
        output = root/'releases'/cfg
        release.prepare_release(old, output, assets_by_seed=seed_assets, configuration_id=cfg)
        results[cfg] = (output, old, existing, seed_assets)
    yield root, results
    validation_cache.clear(); patch.undo()


def test_twenty_temporary_v2_assets_and_wrappers_reuse_exact_original_profiles(candidates):
    root, results = candidates
    assert not list(root.rglob('*.png')) and not list(root.rglob('*.jpg'))
    for cfg, (output, old, existing, assets) in results.items():
        manifest = common.read_json(output/'release_manifest.json')
        assert release.validate_release(output, manifest) == manifest
        old_manifest = common.read_json(existing/'release_manifest.json')
        for seed in range(10):
            _, scene, task, initial = release.task_inputs(output, 'shirt_folding', manifest, seed=seed)
            before = common.read_json(existing/old_manifest['cases'][str(seed)]['scene'])
            assert [r['visual_profile'] for r in scene['cloths']] == [r['visual_profile'] for r in before['cloths']]
            assert task['reset_case']['generator_version'] == SHIRT_NEAR_GENERATOR_VERSION
            assert common.read_json(assets[seed])['generator_version'] == SHIRT_NEAR_GENERATOR_VERSION
            assert task['reset_case']['plan']['generator_version'] == SHIRT_NEAR_GENERATOR_VERSION
            if seed == 0:
                assert initial == common.read_json(old/'initial_state.json')
                assert task['asset_metadata'] == common.read_json(old/'task.json')['asset_metadata']


def test_asset_and_plan_versions_cannot_be_mixed(candidates, tmp_path):
    _, results = candidates
    _, parent_path, _, assets = results['single_arm']
    folder = tmp_path/'asset'; shutil.copytree(assets[1].parent, folder)
    path = folder/'asset_manifest.json'; value = common.read_json(path)
    value['generator_version'] = GENERATOR_VERSION; common.write_json(path, value)
    with pytest.raises(ValueError, match='versions differ'):
        author.load_asset(parent_path, path, 1)
