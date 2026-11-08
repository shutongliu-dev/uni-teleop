"""The gear task's new base world: the open-end wrench 50 mm closer to the robot on every installation."""
import hashlib
import json
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly import task as gear

FAMILY, TASK_ID = 'mechanical_assembly', 'gear_train_assembly'
RELEASES = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases')
BASE = {'path': str(RELEASES/'mechanical_assembly/gear_train_assembly/wrench_reach_v1'),
        'manifest_sha256': '1768eea9144936eae13f408bead4fb881939f6299589b87e50d4f5870e2c35a7'}
DUAL = {cid: {'path': str(RELEASES/'robot_configurations_20261107/gear_wrench01/mechanical_assembly'/TASK_ID/cid/'v1'),
              'manifest_sha256': sha}
        for cid, sha in (('dual_same_side', '853ec4e3b878eaa33e990a52401b417997ce5b2a64752f561d20dd6a0897cf1a'),
                         ('dual_opposed', '7cc132b608abd452981c0e13caff973b1fbc143580d5f5e33bf4447b346a7c9d'))}
INSTALLATIONS = {'single_arm': BASE, **DUAL}
SUPERSEDED = {
    'single_arm': {'path': str(RELEASES/'mechanical_assembly/gear_train_assembly/idler_grip_v2'),
                   'manifest_sha256': '1445f61c09816c139cf700ff87dae82c6b1ba3745e6a33c6eeb2733770caace8'},
    **{cid: {'path': str(RELEASES/'robot_configurations_20261026/gear_grip02/mechanical_assembly'/TASK_ID/cid/'v1'),
             'manifest_sha256': sha}
       for cid, sha in (('dual_same_side', '731c66bd2137225b7d358e86820105a924e5e0a47c3b73c22669ea66d123f3b3'),
                        ('dual_opposed', 'a2f60d954716b0e18a2e43d77144d7f65ca56ad14fe2d85672b4db3e3ea3cd8f'))}}
STANDS = Path(f'{DATA_ROOT}/assets/task_parts/robot_configuration_stands/mechanical_gear_wrench_inputs_20261107/mechanical_assembly')
OLD_STANDS = Path(f'{DATA_ROOT}/assets/task_parts/robot_configuration_stands/mechanical_gear_grip_inputs_20261026/mechanical_assembly')
needs_releases = pytest.mark.skipif(not Path(BASE['path']).is_dir(), reason='Shared gear base releases unavailable')
# The three catalog rows of the republished gear pairs (pins are written by the catalog selection).
ROW_DATA = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105/admission_gearbase01/catalog_row_data.json',
            'sha256': '9332a3257d7777d7a31782cfd05f744ef896d713212864a934ce432e4010a709'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return load_json(Path(path))


def wrench(scene):
    return next(row for row in scene['objects'] if row['id'] == 'open_end_wrench')


def audit(root):
    scene, task = read(Path(root)/'scene.json'), read(Path(root)/'task.json')
    layout = task['supply_layout']
    return gear.reset_audit(scene, minimum_table_edge_clearance_m=layout['minimum_table_edge_clearance_m'],
                            minimum_fixture_clearance_m=layout['minimum_fixture_clearance_m'])


def test_task_layout_places_the_wrench_at_its_reach_position():
    assert gear.OPEN_END_WRENCH_XY_M == (.75, .29) and gear.ORIGINAL_OPEN_END_WRENCH_XY_M == (.80, .29)
    inputs = gear.build_task_inputs()
    tool = wrench(inputs['scene'])
    assert tool['position_m'] == [.75, .29, .7802] and tool['quaternion_wxyz'] == [1., 0., 0., 0.]
    assert inputs['task']['evaluation']['initial_objects']['open_end_wrench'] == {
        'position_m': [.75, .29, .7802], 'quaternion_wxyz': [1., 0., 0., 0.]}
    reset = inputs['scene']['mechanical_task_provenance']['reset_audit']
    assert reset['passed'] and reset['objects']['open_end_wrench']['fixture_clearance']
    # Every other canonical supply position is unchanged.
    positions = {row['id']: row['position_m'][:2] for row in inputs['scene']['objects']}
    assert positions == {'input_gear': [.225, -.20], 'idler_gear': [.28, .34], 'output_gear': [.225, .10],
                         'socket_crank': [.65, -.28], 'open_end_wrench': [.75, .29], 'straight_hex_screwdriver': [.68, .10]}


@needs_releases
def test_new_base_world_only_moves_the_wrench():
    from bench.isaacsim.release import validate
    from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.wrench_release import moved_scene
    old, new = Path(SUPERSEDED['single_arm']['path']), Path(BASE['path'])
    for pin in (SUPERSEDED['single_arm'], BASE):
        assert digest(Path(pin['path'])/'release_manifest.json') == pin['manifest_sha256']
    before, after = validate(old), validate(new)
    assert 'configuration_id' not in after and after['schema'] == before['schema']
    changed = {name for name in after['files'] if after['files'][name] != before['files'][name]}
    assert set(after['files']) == set(before['files']) and changed == {'scene.json', 'task.json', 'source_provenance.json'}
    assert after['asset_files_sha256'] == before['asset_files_sha256'] and after['asset_catalog'] == before['asset_catalog']
    scene, task = moved_scene(read(old/'scene.json'), read(old/'task.json'))
    assert scene == read(new/'scene.json') and task == read(new/'task.json')
    assert wrench(scene) == wrench(gear.build_task_inputs()['scene'])
    assert scene['mechanical_task_provenance']['reset_audit'] == audit(new) and audit(new)['passed']
    derivation = after['wrench_reach_derivation']
    assert derivation['source_release'] == str(old) and derivation['source_release_sha256'] == before_sha(old)
    assert derivation['previous_position_xy_m'] == [.8, .29] and derivation['position_xy_m'] == [.75, .29]
    assert read(new/'source_provenance.json')['wrench_reach_derivation'] == derivation


def before_sha(root):
    return digest(Path(root)/'release_manifest.json')


@needs_releases
@pytest.mark.parametrize('cid', ['dual_same_side', 'dual_opposed'])
def test_dual_installations_are_configuration_releases_of_the_new_base(cid):
    from bench.isaacsim.release import validate
    old, new = Path(SUPERSEDED[cid]['path']), Path(DUAL[cid]['path'])
    assert digest(new/'release_manifest.json') == DUAL[cid]['manifest_sha256']
    manifest, previous = validate(new), validate(old)
    assert manifest['configuration_id'] == cid and manifest['base_release'] == {**BASE, 'schema': 'mechanical_assembly_task_release.v1'}
    scene, base = read(new/'scene.json'), read(Path(BASE['path'])/'scene.json')
    assert scene['objects'] == base['objects'] and scene['mechanical_task_provenance'] == base['mechanical_task_provenance']
    stand = next(row for row in scene['fixtures'] if row['id'] == 'bimanual_robot_stand')
    assert Path(stand['asset_root']) == STANDS/cid
    # Apart from the moved wrench and the fresh stand directory the installation is the published one.
    configuration, prior = read(new/'robot_configuration.json'), read(old/'robot_configuration.json')
    prior['stand_fixture']['asset_root'] = str(STANDS/cid)
    assert configuration == prior
    old_scene = read(old/'scene.json')
    for row in old_scene['fixtures']:
        if row['id'] == 'bimanual_robot_stand':
            row['asset_root'] = str(STANDS/cid)
    wrench(old_scene)['position_m'] = wrench(scene)['position_m']
    old_scene['mechanical_task_provenance'] = base['mechanical_task_provenance']
    assert old_scene == scene
    stands = {str(p.relative_to(STANDS/cid)): digest(p) for p in (STANDS/cid).rglob('*') if p.is_file()}
    assert stands and stands == {str(p.relative_to(OLD_STANDS/cid)): digest(p) for p in (OLD_STANDS/cid).rglob('*') if p.is_file()}
    assert {k: v for k, v in manifest['asset_files_sha256'].items() if '/robot_configuration_stands/' not in k} == {
        k: v for k, v in previous['asset_files_sha256'].items() if '/robot_configuration_stands/' not in k}


@needs_releases
def test_reset_audit_keeps_the_dual_opposed_canonical_exemption():
    for cid in INSTALLATIONS:
        new, old = audit(INSTALLATIONS[cid]['path']), audit(SUPERSEDED[cid]['path'])
        assert new['passed'] == old['passed'] == (cid != 'dual_opposed')
        assert new['failures'] == old['failures'] and not new['object_aabb_overlaps']
    failed = audit(DUAL['dual_opposed']['path'])
    # The opposed bimanual mount's AABB spans the table: a canonical installation property, not the wrench.
    assert failed['failures'] == ['input_gear', 'idler_gear', 'output_gear', 'socket_crank', 'open_end_wrench']
    assert all(failed['objects'][name]['fixture_xy_clearances_m']['bimanual_robot_stand'] == 0. for name in failed['failures'])
    assert failed['objects']['open_end_wrench']['table_edge_clearance'] and failed['objects']['open_end_wrench']['exact_initial_gap']


@needs_releases
def test_installation_identity_of_an_original_parent_release():
    from scenes.workrooms.parameterized.catalog import _installation
    for cid, pin in INSTALLATIONS.items():
        assert _installation(read(Path(pin['path'])/'release_manifest.json'), pin) == cid


def original_definition():
    from copy import deepcopy
    import importlib
    from scenes.workrooms.robot_configurations import attach_robot_configurations
    module = importlib.import_module(f'scenes.workrooms.{FAMILY}.tasks.{TASK_ID}.definition')
    return attach_robot_configurations(deepcopy(module.TASK))


@needs_releases
def test_definition_and_installations_pin_the_new_base():
    definition = original_definition()
    assert definition['source_release'] == BASE and definition['previous_source_release'] == SUPERSEDED['single_arm']
    assert definition['candidate_referee']['release'] == BASE
    assert definition['design']['supply_layout_candidate']['release'] == BASE
    bindings = definition['objects']['asset_bindings']
    assert bindings['frozen_scene'] == BASE['path']+'/scene.json'
    assert bindings['frozen_inventory'] == BASE['path']+'/asset_file_hashes.json'
    options = definition['robot_configurations']
    assert {key: row['source_release'] for key, row in options.items()} == INSTALLATIONS
    revision = definition['design']['tool_layout_revision']
    assert revision['open_end_wrench_xy_m'] == [.75, .29] and revision['previous_open_end_wrench_xy_m'] == [.8, .29]
    assert digest(revision['release_validation']['path']) == revision['release_validation']['sha256']
    # No physical, native or manipulation evidence is inherited from the superseded base.
    qualification = definition['qualification']
    assert not any(qualification[key] for key in ('native_preflight', 'agent_bench_interface', 'manipulation',
                                                   'physics_accepted', 'task_policy_success_claimed'))
    assert qualification['scene_contact_evidence'] == []
    for cid in DUAL:
        assert not any(options[cid]['qualification'][key] for key in (
            'native_preflight', 'agent_bench_interface', 'robot_manipulation', 'physics_accepted'))


def published_rows():
    """The three catalog rows of the republished gear pairs, as the catalog selection pins them."""
    assert digest(ROW_DATA['path']) == ROW_DATA['sha256']
    rows = read(ROW_DATA['path'])['rows']
    seed = {'family': FAMILY, 'parent_default_release': rows['initial_randomization']['parent_default_release'],
            'parents': rows['initial_randomization']['parents'], 'releases': rows['initial_randomization']['releases']}
    paired = {'family': FAMILY, 'task_id': TASK_ID, 'parent_releases': rows['parameterized']['parent_releases'],
              'pairs': rows['parameterized']['pairs']}
    revision = {'family': FAMILY, 'task_id': TASK_ID, **rows['robot_revision']}
    return seed, paired, revision


@needs_releases
@pytest.mark.parametrize('robot_profile', ['original', 'l_mount'])
def test_catalog_binds_the_published_rows_for_both_robot_profiles(robot_profile, monkeypatch):
    from scenes.workrooms import catalog
    from scenes.workrooms.initial_randomization import catalog as seeds
    from scenes.workrooms.parameterized import catalog as parameterized
    from scenes.workrooms.robot_revision import catalog as revisions
    seed, paired, revision = published_rows()
    assert seed['parents'] == INSTALLATIONS and seed['parent_default_release'] == BASE
    assert all(pin != INSTALLATIONS[cid] for cid, pin in seed['releases'].items())
    assert paired['parent_releases'] == seed['releases']
    # The catalog pins carry exactly the three republished rows.
    assert read(seeds.PINS)[TASK_ID] == seed
    assert read(parameterized.PINS)['tasks'][FAMILY+'/'+TASK_ID] == paired
    assert read(revisions.PINS)['tasks'][FAMILY+'/'+TASK_ID] == revision
    task = catalog.task_definition(FAMILY, TASK_ID, robot_profile=robot_profile)
    assert task['robot_profile'] == robot_profile
    assert set(task['robot_configurations']) == set(INSTALLATIONS)
    for cid in INSTALLATIONS:
        selected = catalog.task_definition(FAMILY, TASK_ID, cid, robot_profile=robot_profile)
        pair = read(paired['pairs'][cid]['path'])
        assert selected['configuration_id'] == cid and selected['source_release'] == pair['robot_profiles'][robot_profile]
        assert selected['robot_requirements']['assembly_version'] == (
            'isaacsim_6_1_l_mount_v3' if robot_profile == 'l_mount' else 'isaacsim_6_1_v2')
        manifest = read(Path(seed['releases'][cid]['path'])/'release_manifest.json')
        assert manifest['base_release']['path'] == INSTALLATIONS[cid]['path']
        assert manifest['base_release']['manifest_sha256'] == INSTALLATIONS[cid]['manifest_sha256']
    # A paired row still naming a superseded parent no longer binds.
    stale = read(parameterized.PINS)
    stale['tasks'][FAMILY+'/'+TASK_ID]['parent_releases']['single_arm'] = SUPERSEDED['single_arm']
    monkeypatch.setattr(parameterized, 'read', lambda path: stale if Path(path) == parameterized.PINS else read(path))
    with pytest.raises(ValueError, match='inventory'):
        catalog.task_definition(FAMILY, TASK_ID, robot_profile=robot_profile)


@needs_releases
def test_published_robot_revision_row_pairs_every_author_release():
    _, paired, revision = published_rows()
    assert revision['parents'] == paired['parent_releases'] and revision['parent_default_release'] == paired['parent_releases']['single_arm']
    for cid, pin in revision['releases'].items():
        pair = read(paired['pairs'][cid]['path'])
        assert pair['robot_profiles'] == {'original': paired['parent_releases'][cid], 'l_mount': pin}
        manifest = read(Path(pin['path'])/'release_manifest.json')
        assert manifest['schema'] == 'workroom_robot_revision_release.v1'
        assert {key: manifest['base_release'][key] for key in ('path', 'manifest_sha256')} == paired['parent_releases'][cid]
