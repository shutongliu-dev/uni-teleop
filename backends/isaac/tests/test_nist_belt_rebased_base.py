"""Task7 is re-based on the usability_v5 belt world for every installation and robot profile."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT, load_json

FAMILY, TASK_ID = 'mechanical_assembly', 'nist_belt_drive_assembly'
RELEASES = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases')
ASSETS = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_belt_drive_assembly')
BASE = {'path': str(RELEASES/'nist_usability_20261107b/batch01/nist_belt_drive_assembly/desktop_v1'),
        'manifest_sha256': 'ec43adbeec31682e74923e0c5220e00c33bfaae8a7411abadea8359ab34b0fde'}
DUAL = {cid: {'path': str(RELEASES/'nist_usability_20261107b/dual_batch01/mechanical_assembly'
                          /TASK_ID/cid/'v1'), 'manifest_sha256': sha}
        for cid, sha in (('dual_same_side', 'eb4a9f2ce2f920b6be8aff622f54884b55f27cea17bfde0a160b7f7e8cb3a3e2'),
                         ('dual_opposed', '264988b622e9c480250be511c53bcf88f1fb55f8e47c251f31b17b5ba5b4ca5a'))}
INSTALLATIONS = {'single_arm': BASE, **DUAL}
SUPERSEDED = {'path': str(RELEASES/'nist_usability_20261023/batch02/nist_belt_drive_assembly/desktop_v1'),
              'manifest_sha256': '4e1b0371aee0483a9a11a11f3bdb78d040aa45e7293233a5a52d6cc465b53f62'}
ASSET_MANIFEST = {'path': str(ASSETS/'usability_v5/manifest.json'),
                  'sha256': 'ee6fee0805bbc3a6b5e2d4b75ff2d86ecf56592f35d3adb1662858cb45359aec'}
pytestmark = pytest.mark.skipif(not Path(BASE['path']).is_dir(), reason='Shared belt base releases unavailable')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return load_json(Path(path))


def original_definition():
    from scenes.workrooms.robot_configurations import attach_robot_configurations
    module = importlib.import_module(f'scenes.workrooms.{FAMILY}.tasks.{TASK_ID}.definition')
    return attach_robot_configurations(deepcopy(module.TASK))


def test_definition_and_installations_pin_the_validated_new_base():
    from bench.isaacsim.release import validate
    definition = original_definition()
    assert definition['source_release'] == BASE and definition['previous_source_release'] == SUPERSEDED
    assert definition['objects']['asset_root'] == str(ASSETS/'usability_v5')
    options = definition['robot_configurations']
    assert {key: row['source_release'] for key, row in options.items()} == INSTALLATIONS
    for cid, pin in INSTALLATIONS.items():
        root = Path(pin['path'])
        assert digest(root/'release_manifest.json') == pin['manifest_sha256']
        manifest = validate(root)
        if cid == 'single_arm':
            assert manifest['schema'] == definition['release_schema'] and 'configuration_id' not in manifest
            assert manifest['asset_catalog'] == ASSET_MANIFEST
        else:
            assert manifest['configuration_id'] == cid
            assert manifest['base_release'] == {**BASE, 'schema': definition['release_schema']}
        assert read(root/'scene.json')['native_task_bundle']['usd_path'] == str(ASSETS/'usability_v5/task.usda')
    assert digest(ASSET_MANIFEST['path']) == ASSET_MANIFEST['sha256']
    # Discrete CPU workspace evidence was re-run on, and bound to, these exact releases.
    qualification = definition['qualification']
    binding = qualification['workspace_release_binding']
    assert digest(binding['path']) == binding['sha256']
    rows = {row['configuration_id']: row for row in read(binding['path'])['rows']}
    assert {key: row['source_release'] for key, row in rows.items()} == INSTALLATIONS
    for cid, row in rows.items():
        evidence = (qualification if cid == 'single_arm' else options[cid]['qualification'])['scene_geometry_evidence']
        assert {key: evidence[key] for key in ('path', 'sha256')} == row['workspace_report'] and evidence['passed']
        assert read(evidence['path'])['release'] == BASE['path'] and read(evidence['path'])['passed']
    # Physical and native evidence of the superseded base is never inherited.
    assert not qualification['object_interactions'] and not qualification['scene_contact_evidence']
    assert qualification['historical_usability_v3_evidence']['source_release'] == SUPERSEDED
    for cid in DUAL:
        assert not any(options[cid]['qualification'][key] for key in (
            'native_preflight', 'agent_bench_interface', 'robot_manipulation', 'physics_accepted'))


def test_usability_v5_only_moves_the_belt_supply_and_clears_the_idler_shoulder():
    from pxr import Usd
    from scenes.workrooms.mechanical_assembly.nist_assets import ROOT_PRIM, BELT_SUPPLY_CENTER_XY_M
    from scenes.workrooms.mechanical_assembly.placement_authoring import _belt_initial_clearance
    old, new = (read(ASSETS/version/'metadata.json') for version in ('usability_v3', 'usability_v5'))
    assert BELT_SUPPLY_CENTER_XY_M == (.280, -.056)
    assert {key: value for key, value in old.items() if key != 'objects'} == {
        key: value for key, value in new.items() if key != 'objects'}
    assert [row['id'] for row in old['objects']] == [row['id'] for row in new['objects']]
    for before, after in zip(old['objects'], new['objects']):
        if not before['id'].startswith('belt_link_'):
            assert before == after
            continue
        assert np.allclose(np.subtract(after['initial_position_m'], before['initial_position_m']), [0., -.001, 0.], atol=1e-12)
        assert np.allclose(after['initial_pose_xyzw'][3:], before['initial_pose_xyzw'][3:], atol=1e-12)
        assert after['link_length_m'] == pytest.approx(before['link_length_m'], abs=1e-15)
    for version, clear in (('usability_v3', False), ('usability_v5', True)):
        meta = read(ASSETS/version/'metadata.json')
        stage = Usd.Stage.Open(str(ASSETS/version/'task.usda'))
        moving = {meta['actors'][name] for name in meta['actors'] if name != 'drive_pulley'}
        if not clear:
            # The superseded world: links crossed the 20 mm idler support shoulder.
            with pytest.raises(ValueError, match='intersects a fixed collider'):
                _belt_initial_clearance(stage, ROOT_PRIM, moving)
            continue
        result = _belt_initial_clearance(stage, ROOT_PRIM, moving)
        assert max(row['common_inscribed_sphere_radius_m'] for row in result['narrow_phase']) < -1.5e-4
        assert min(result['workplate_xy_margin_m']) == pytest.approx(.011278758, abs=1e-8)   # x as authored
        assert result['minimum_original_support_gap_m'] == pytest.approx(.0003, abs=1e-6)


def test_installation_identity_of_an_original_parent_release():
    from scenes.workrooms.parameterized.catalog import _installation
    for cid, pin in INSTALLATIONS.items():
        assert _installation(read(Path(pin['path'])/'release_manifest.json'), pin) == cid


@pytest.mark.parametrize('robot_profile', ['original', 'l_mount'])
def test_catalog_binds_the_new_base_for_both_robot_profiles(robot_profile, monkeypatch):
    from scenes.workrooms import catalog
    from scenes.workrooms.initial_randomization import catalog as seeds
    from scenes.workrooms.parameterized import catalog as parameterized
    # The seed row is republished from the new base, and the paired row names those seed releases.
    seed_row = read(seeds.PINS)[TASK_ID]
    assert seed_row['parents'] == INSTALLATIONS and seed_row['parent_default_release'] == BASE
    assert set(seed_row['releases']) == set(INSTALLATIONS)
    assert all(pin != INSTALLATIONS[cid] and pin != SUPERSEDED for cid, pin in seed_row['releases'].items())
    assert read(parameterized.PINS)['tasks'][FAMILY+'/'+TASK_ID]['parent_releases'] == seed_row['releases']
    task = catalog.task_definition(FAMILY, TASK_ID, robot_profile=robot_profile)
    assert task['robot_profile'] == robot_profile
    # The public single-arm requirements the seed layer used to merge are unchanged.
    assert task['robot_requirements'] == {
        'arm': 'fr3', 'hand_id': 'sharpa_w01_right', 'articulations': 1, 'layout_id': 'mechanical_single_arm_v1',
        'base_position_m': [-.23, 0, .78], 'active_sides': ['right'],
        'assembly_version': 'isaacsim_6_1_l_mount_v3' if robot_profile == 'l_mount' else 'isaacsim_6_1_v2'}
    assert set(task['robot_configurations']) == set(INSTALLATIONS)
    for cid in INSTALLATIONS:
        selected = catalog.task_definition(FAMILY, TASK_ID, cid, robot_profile=robot_profile)
        assert selected['configuration_id'] == cid
        assert selected['robot_requirements']['assembly_version'] == (
            'isaacsim_6_1_l_mount_v3' if robot_profile == 'l_mount' else 'isaacsim_6_1_v2')
    # A paired row still naming a superseded parent no longer binds.
    stale = read(parameterized.PINS)
    stale['tasks'][FAMILY+'/'+TASK_ID]['parent_releases']['single_arm'] = SUPERSEDED
    monkeypatch.setattr(parameterized, 'read', lambda path: stale if Path(path) == parameterized.PINS else read(path))
    with pytest.raises(ValueError, match='inventory'):
        catalog.task_definition(FAMILY, TASK_ID, robot_profile=robot_profile)
