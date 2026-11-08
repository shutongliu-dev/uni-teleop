"""Scene/task separation and fail-closed loading; no native simulation."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from bench.paths import load_json
from scenes.workrooms import catalog
from scenes.workrooms.registry import FAMILIES, main

KITCHEN_TASKS = ('breakfast_serving', 'produce_transfer', 'utensil_sorting',
                 'pot_lid_placement', 'pantry_stocking', 'table_clearing')
ALL_KITCHEN_TASKS = (*KITCHEN_TASKS, 'surface_wiping')
KITCHEN_RELEASE_VERSIONS = {
    'breakfast_serving': 'household_v2',
    'produce_transfer': 'household_access_v2',
    'utensil_sorting': 'household_grip_v1',
    'pot_lid_placement': 'household_access_v3',
    'pantry_stocking': 'household_v2',
    'table_clearing': 'household_grip_v1',
    'surface_wiping': 'elastic_cell_lattice_v2',
}

NIST_TASKS = ('nist_connector_insertion', 'nist_belt_drive_assembly',
              'nist_cable_routing', 'nist_wire_harness')
CARD_ROOM_EXPANSION_TASKS = ('polyomino_territory', 'barrier_race', 'memory_pairs', 'disk_curling')
CARD_ROOM_PRACTICE_TASKS = tuple(task+'_practice' for task in CARD_ROOM_EXPANSION_TASKS)
CARD_ROOM_PLANNED_TASKS = frozenset()
SEEDED_TASKS = {
    'stationery_sorting': 'study', 'book_stacking': 'study', 'book_shelving': 'study',
    'produce_transfer': 'kitchen', 'utensil_sorting': 'kitchen',
    'pantry_stocking': 'kitchen', 'table_clearing': 'kitchen',
    'sample_sorting': 'laboratory', 'vial_racking': 'laboratory',
    'parts_sorting': 'mechanical_assembly',
    'peg_insertion': 'mechanical_assembly', 'shaft_spacer_assembly': 'mechanical_assembly',
    'fastener_assembly': 'mechanical_assembly', 'gear_train_assembly': 'mechanical_assembly',
    'nist_connector_insertion': 'mechanical_assembly', 'nist_belt_drive_assembly': 'mechanical_assembly',
    'nist_cable_routing': 'mechanical_assembly', 'nist_wire_harness': 'mechanical_assembly',
    'cooperative_insertion': 'mechanical_assembly',
    'book_retrieval': 'study', 'drawer_retrieval': 'study', 'reading_stand': 'study',
    'breakfast_serving': 'kitchen', 'pot_lid_placement': 'kitchen', 'surface_wiping': 'kitchen',
    'cap_placement': 'laboratory', 'solid_weighing': 'laboratory',
    'pipette_transfer': 'laboratory', 'tube_clamp_transfer': 'laboratory',
}
SEEDED_SCHEMA = 'workroom_randomized_release.v1'
REVISION_SCHEMA = 'workroom_robot_revision_release.v1'
PROFILES = ('original', 'l_mount')


def original_definition(family, task_id):
    return importlib.import_module(
        f'scenes.workrooms.{family}.tasks.{task_id}.definition').TASK


def canonical_release(task):
    """The new wrapper retains the exact original single-arm input identity."""
    if task['task_id'] not in SEEDED_TASKS:
        assert task['release_schema'] != SEEDED_SCHEMA
        return task['source_release']
    root = Path(task['source_release']['path'])
    manifest = load_json(root/'release_manifest.json')
    assert task['release_schema'] == manifest['schema'] == SEEDED_SCHEMA
    assert digest(root/'release_manifest.json') == task['source_release']['manifest_sha256']
    original = original_definition(task['family'], task['task_id'])
    parent = {key: manifest['base_release'][key] for key in ('path', 'manifest_sha256')}
    assert parent == original['source_release']
    assert manifest['base_release']['schema'] == original['release_schema']
    assert manifest['seed_values'] == list(range(10))
    return parent


def l_mount_revision(original):
    """The L-mount profile is a robot-only revision of each exact original-profile release."""
    from scenes.workrooms.robot_revision.inputs import canonical_hash
    revised = catalog.task_definition(original['family'], original['task_id'], robot_profile='l_mount')
    originals = original.get('robot_configurations') or {'fixed': original}
    revisions = revised.get('robot_configurations') or {'fixed': revised}
    assert set(revisions) == set(originals)
    for key, row in revisions.items():
        root, parent = Path(row['source_release']['path']), originals[key]
        manifest = load_json(root/'release_manifest.json')
        assert digest(root/'release_manifest.json') == row['source_release']['manifest_sha256']
        assert row['release_schema'] == manifest['schema'] == REVISION_SCHEMA
        assert manifest['base_release'] == {**parent['source_release'], 'schema': parent['release_schema']}
        parent_root = Path(parent['source_release']['path'])
        assert manifest['seed_values'] == load_json(parent_root/'release_manifest.json').get('seed_values')
        revision = load_json(root/'robot_revision.json')
        assert revision['parent_robot_sha256'] == canonical_hash(
            load_json(parent_root/'scene.json')['robot'])
        assert row['robot_requirements']['assembly_version'] == 'isaacsim_6_1_l_mount_v3'
    return revised


def frozen_candidate(definition):
    """The exact original release under seed-case and L-mount revision wrappers."""
    reference = {key: definition['source_release'][key] for key in ('path', 'manifest_sha256')}
    while True:
        manifest = load_json(Path(reference['path'])/'release_manifest.json')
        if manifest['schema'] not in (SEEDED_SCHEMA, REVISION_SCHEMA):
            return reference
        reference = {key: manifest['base_release'][key] for key in ('path', 'manifest_sha256')}


def published_nist_tasks():
    return [catalog.task_definition('mechanical_assembly', name) for name in NIST_TASKS
            if catalog.task_definition('mechanical_assembly', name)['status'] == 'implemented']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('robot_profile', PROFILES)
def test_original_six_fixed_scenes_include_single_arm_card_room_task0(robot_profile):
    scenes = [catalog.scene_definition(family) for family in FAMILIES]
    assert [row['family'] for row in scenes] == list(FAMILIES)
    for row in scenes:
        assert row['environment']['family'] == row['family']
        assert row['coordinates']['table_top_z_m'] == (.90 if row['family'] in ('kitchen', 'laboratory') else .78)
        tasks = catalog.list_tasks(row['family'], include_planned=True, robot_profile=robot_profile)
        assert len(tasks) == {'mechanical_assembly': 10, 'card_room': 12, 'living_room': 5, 'study': 6, 'kitchen': 7, 'laboratory': 6}[row['family']]
        assert all(task['family'] == row['family'] for task in tasks)
    # All four dedicated expansion sessions are now published in the original six families.
    # Laundry installations have their own five-task capability tests.
    runnable = [task for task in catalog.list_tasks(robot_profile=robot_profile) if task['family'] in FAMILIES]
    new = published_nist_tasks()
    assert {(task['family'], task['task_id']) for task in runnable} == {('card_room', game) for game in ('task0', 'tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7', *CARD_ROOM_EXPANSION_TASKS, *CARD_ROOM_PRACTICE_TASKS)} | {('mechanical_assembly', task) for task in ('parts_sorting', 'peg_insertion', 'shaft_spacer_assembly', 'fastener_assembly', 'gear_train_assembly', 'cooperative_insertion')} | {('living_room', task) for task in ('remote_operation', 'water_pouring', 'tower_building', 'tower_extraction', 'storage_box')} | {('study', task) for task in ('stationery_sorting', 'book_stacking', 'book_retrieval', 'book_shelving', 'drawer_retrieval', 'reading_stand')} | {('kitchen', task) for task in ALL_KITCHEN_TASKS} | {('laboratory', task) for task in ('sample_sorting', 'vial_racking', 'cap_placement', 'solid_weighing', 'pipette_transfer', 'tube_clamp_transfer')} | {('mechanical_assembly', task['task_id']) for task in new}
    assert sorted(task['difficulty'] for task in runnable) == sorted(
        [0]+[1]*6+[2]*6+[3]*7+[4]*6+[5]*5+[6,6,6,6,7,7,10]+[task['task_number'] for task in new]+[20,21,22,23])
    assert len(runnable) == 42+len(new)
    assert sum(task['status'] == 'planned' for task in catalog.list_tasks(include_planned=True, robot_profile=robot_profile)
               if task['family'] in FAMILIES) == 4-len(new)+len(CARD_ROOM_PLANNED_TASKS)
    assert {task['task_id'] for task in catalog.list_tasks('card_room', include_planned=True, robot_profile=robot_profile)
            if task['status'] == 'planned'} == CARD_ROOM_PLANNED_TASKS
    # Paired parameterized instances never inherit manipulation qualification.
    assert runnable[0]['qualification']['robot_manipulation'] is False


def test_planned_task_refuses_loading_before_reading_any_release(monkeypatch):
    from scenes.workrooms.laboratory.catalog_definition import task_definition
    planned = task_definition('sample_sorting')
    planned['runnable'] = False
    monkeypatch.setattr(catalog, 'task_definition', lambda *args, **kwargs: deepcopy(planned))
    monkeypatch.setattr(catalog, 'scene_definition',
                        lambda family: pytest.fail('A planned task must stop before environment loading'))
    with pytest.raises(ValueError, match='cannot run'):
        catalog.load_task('laboratory', 'sample_sorting')


@pytest.mark.parametrize('task_id,family', SEEDED_TASKS.items())
def test_published_seed_catalog_keeps_exact_original_installations_and_seed_zero(task_id, family):
    from scenes.workrooms.initial_randomization.common import canonical_hash
    from scenes.workrooms.initial_randomization.physical_inputs import check_canonical_physics
    from scenes.workrooms.robot_configurations import attach_robot_configurations

    original = attach_robot_configurations(deepcopy(original_definition(family, task_id)))
    selected = catalog.task_definition(family, task_id, robot_profile='original')
    default = 'dual_same_side' if task_id == 'cooperative_insertion' else 'single_arm'
    parents = original.get('robot_configurations') or {default: original}
    releases = selected.get('robot_configurations') or {default: selected}
    assert set(releases) == set(parents) == (
        {'dual_same_side'} if task_id == 'cooperative_insertion' else
        {'single_arm'} if family == 'kitchen' else {'single_arm', 'dual_same_side', 'dual_opposed'})
    for configuration, row in releases.items():
        root = Path(row['source_release']['path'])
        manifest = load_json(root/'release_manifest.json')
        assert digest(root/'release_manifest.json') == row['source_release']['manifest_sha256']
        assert row['release_schema'] == manifest['schema'] == SEEDED_SCHEMA
        assert manifest['configuration_id'] == configuration and manifest['task_id'] == task_id
        assert manifest['seed_values'] == list(range(10))
        parent = parents[configuration]
        assert manifest['base_release'] == {
            **parent['source_release'], 'schema': parent['release_schema']}
        parent_root = Path(parent['source_release']['path'])
        assert digest(parent_root/'release_manifest.json') == parent['source_release']['manifest_sha256']
        scene = load_json(root/'scene.json')
        parent_scene = load_json(parent_root/'scene.json')
        check_canonical_physics(parent_scene, scene)
        assert scene['robot'] == parent_scene['robot']
        assert scene['workroom'] == parent_scene['workroom']
        assert load_json(root/'initial_state.json') == load_json(parent_root/'initial_state.json')
        task = load_json(root/'task.json')
        case = load_json(root/manifest['cases']['0']['case'])
        assert task['reset_case'] == case and case['seed'] == 0
        assert case['case_sha256'] == canonical_hash({k: v for k, v in case.items() if k != 'case_sha256'})
        assert case['configuration_id'] == configuration
        for layer, baseline in zip(scene.get('task_layers', []), parent_scene.get('task_layers', [])):
            assert digest(Path(layer['usd_path'])) == digest(Path(baseline['usd_path']))
        if 'native_task_bundle' in scene:
            assert digest(Path(scene['native_task_bundle']['usd_path'])) == digest(
                Path(parent_scene['native_task_bundle']['usd_path']))
    l_mount_revision(selected)


def test_kitchen_catalog_exposes_seven_frozen_candidates_without_loading_assets(monkeypatch):
    expected = list(ALL_KITCHEN_TASKS)
    monkeypatch.setattr(catalog, 'verify_reference',
                        lambda *_: pytest.fail('Listing metadata must not load shared assets'))
    tasks = catalog.list_tasks('kitchen', robot_profile='original')
    assert [task['task_id'] for task in tasks] == expected
    assert [task['task_id'] for task in catalog.list_tasks('kitchen', robot_profile='l_mount')] == expected
    assert [task['difficulty'] for task in tasks] == list(range(1, 8))
    for task in tasks:
        assert task['status'] == 'implemented' and task['runnable']
        version = KITCHEN_RELEASE_VERSIONS[task['task_id']]
        assert canonical_release(task)['path'].endswith('/kitchen/'+task['task_id']+'/'+version)
        if task['task_id'] == 'surface_wiping':
            assert task['objects']['inventory_counts'] == {'sponge': 1, 'return_tray': 1}
            assert task['objects']['dynamic_count'] == task['objects']['fixed_fixture_count'] == 1
            assert task['objects']['dynamic_count_scope'] == 'logical_task_objects'
            assert task['objects']['native_rigid_body_count'] == 28
        assert len(task['source_release']['manifest_sha256']) == 64
        assert task['objects']['asset_bindings'] and task['layout']['initialization']
        assert task['evaluator']['implementation'] == 'scenes.workrooms.kitchen.evaluation'
        assert task['rules']['contract']['task_id'] == task['task_id']
        assert task['rules']['contract']['schema'] == ('kitchen_rules.v6' if task['task_id'] == 'surface_wiping' else
            'kitchen_rules.v3' if task['task_number'] >= 5 else 'kitchen_rules.v2')
        assert task['objects']['contextual_prop_count'] == 12
        assert task['objects']['contextual_props_scored'] is False
        assert 'candidate_success' not in task['rules']
        assert 'candidate_design' not in task
        assert task['layout']['same_environment_for_all_tasks']
        assert task['robot_requirements']['articulations'] == 1
        assert task['robot_requirements']['base_position_m'] == [.65, 0., .78]
        assert not task['difficulty_calibrated']
        assert task['qualification']['robot_manipulation'] is False
        assert not task['qualification']['physics_accepted']
        l_mount_revision(task)
        json.dumps(task)


@pytest.mark.parametrize('task_id, counts, dynamic_count, fixture_count, yaw', [
    ('breakfast_serving', {'tray': 1, 'plate': 1, 'cup': 1, 'bread': 2}, 5, 0, -90.),
    ('produce_transfer', {'tomato': 3, 'short_carrot': 3, 'source_board': 1,
                          'sorting_tray': 2}, 6, 3, -90.),
    ('utensil_sorting', {'spoon': 2, 'fork': 2, 'blunt_table_knife': 2,
                        'source_pad': 1, 'cutlery_tray': 1}, 6, 2, -90.),
    ('pot_lid_placement', {'pot': 1, 'lid': 1, 'source_mat': 1}, 2, 1, 90.),
    ('pantry_stocking', {'olive_oil_bottle': 1, 'vinegar_bottle': 1, 'salt_shaker': 1,
                         'pepper_shaker': 1, 'paprika_jar': 1, 'rice_carton': 1,
                         'pasta_carton': 1, 'tomato_can': 1, 'source_mat': 1,
                         'storage_bin': 3}, 8, 4, -90.),
    ('table_clearing', {'dinner_plate': 1, 'side_plate': 1, 'cereal_bowl': 1,
                        'small_bowl': 1, 'mug': 1, 'tea_cup': 1, 'fork': 1, 'spoon': 1,
                        'source_mat': 1, 'collection_tray': 3}, 8, 4, -90.),
])
def test_kitchen_inventory_and_practical_frozen_rules(
        task_id, counts, dynamic_count, fixture_count, yaw):
    task = catalog.task_definition('kitchen', task_id)
    assert task['objects']['inventory_counts'] == counts
    assert task['objects']['dynamic_count'] == dynamic_count
    assert task['objects']['fixed_fixture_count'] == fixture_count
    assert sum(counts.values()) == dynamic_count+fixture_count
    assert task['robot_requirements']['base_yaw_deg'] == yaw
    rules = task['rules']['contract']
    limits = rules['referee']['thresholds']
    assert limits['stable_s'] == .4
    assert limits['stability_pose_window_s'] == .020
    assert limits['maximum_linear_speed_m_s'] == .025
    assert limits['maximum_angular_speed_rad_s'] == .25
    assert rules['referee']['recovery'] and not rules['reset_credit']
    assert task['evaluator']['thresholds'] == limits


def test_kitchen_current_catalog_returns_independent_objects_and_practical_rules():
    task = catalog.task_definition('kitchen', 'breakfast_serving')
    task['objects']['inventory_counts']['cup'] = 99
    task['rules']['contract']['referee']['thresholds']['stable_s'] = 99.
    current = catalog.task_definition('kitchen', 'breakfast_serving')
    assert current['objects']['inventory_counts']['cup'] == 1
    assert current['rules']['contract']['referee']['thresholds']['stable_s'] == .4
    assert current['rules']['contract']['referee']['thresholds']['carry_clearance_m'] == .015
    utensils = catalog.task_definition('kitchen', 'utensil_sorting')['rules']['contract']
    assert any('柄朝两端均可' in text for text in utensils['referee']['success_conditions'])
    cookware = catalog.task_definition('kitchen', 'pot_lid_placement')['rules']['contract']
    assert cookware['referee']['thresholds']['lid_seat_height_tolerance_m'] == .008


@pytest.mark.parametrize('robot_profile', PROFILES)
@pytest.mark.parametrize('task_id', ALL_KITCHEN_TASKS)
def test_kitchen_catalog_and_canonical_loader_bind_same_release(task_id, robot_profile):
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.kitchen.task_robot_layout import robot_configuration
    from scenes.workrooms.robot_revision.inputs import canonical_hash
    loaded = catalog.load_task('kitchen', task_id, robot_profile=robot_profile)
    definition = loaded['definition']
    _, scene, task, initial = task_inputs(definition['source_release']['path'], task_id)
    # Catalog returns the portable frozen path; runtime resolves it against
    # the verified release that owns the hook. An L-mount revision replaces
    # only the robot and keeps its original seed-case release's observer.
    owner = Path((loaded['release_manifest']['base_release'] if robot_profile == 'l_mount'
                  else definition['source_release'])['path'])
    frozen_scene = deepcopy(loaded['scene'])
    observer = frozen_scene['native_task_bundle']['observer']
    observer['path'] = str(owner/observer['path'])
    assert (scene, task, initial) == (frozen_scene, loaded['task'], loaded['initial_state'])
    if robot_profile == 'original':
        assert scene['robot'] == robot_configuration(task_id)
    else:
        revision = load_json(Path(definition['source_release']['path'])/'robot_revision.json')
        assert scene['robot'] == revision['robot']
        assert revision['parent_robot_sha256'] == canonical_hash(robot_configuration(task_id))
    assert task['public_rules'] == definition['rules']['contract']
    assert task['public_rules']['referee']['thresholds'] == definition['evaluator']['thresholds']
    assert len(scene['native_task_bundle']['rigid_bodies']) == definition['objects'].get(
        'native_rigid_body_count', definition['objects']['dynamic_count'])
    assert len(initial['joint_positions_rad']) == 29
    assert loaded['release_manifest']['status'] == 'prepared_unvalidated'
    assert not loaded['release_manifest']['physics_accepted']
    assert definition['qualification']['robot_manipulation'] is False
    # Deployment contracts depend on frozen inputs, not historical validation
    # reports. Those reports retain their own independent development audits.
    root = Path(definition['source_release']['path'])
    manifest = loaded['release_manifest']
    assert digest(root/'release_manifest.json') == definition['source_release']['manifest_sha256']
    for name in ('scene.json', 'task.json', 'rules.json', 'initial_state.json', 'physics_profile.json'):
        assert digest(root/name) == manifest['files_sha256'][name]
    assert load_json(root/'rules.json') == task['public_rules']
    assert scene['isaacsim']['physics_dt_s'] == task['evaluation']['physics_dt_s']
    assert task['public_rules']['referee']['thresholds']['stable_s'] >= .3


@pytest.mark.parametrize('task_id', KITCHEN_TASKS)
def test_kitchen_public_contract_is_actionable_and_excludes_native_state(task_id):
    from bench.protocol.task_referee import project_task_rules
    loaded = catalog.load_task('kitchen', task_id)
    task = loaded['task']
    public = project_task_rules(task)
    expected_duration = 240. if task_id in ('pantry_stocking', 'table_clearing') else 120.
    assert task['recommended_budgets']['sim_time_s'] == expected_duration
    assert public['public_rules']['thresholds']['max_time_s'] == expected_duration
    assert task['recommended_budgets']['max_actions'] == public['public_rules']['thresholds']['max_actions']
    assert public['task_id'] == task_id
    assert public['create_endpoint'] == 'POST /episodes'
    assert public['instruction'] == task['public_rules']['instruction_zh']
    assert public['public_rules']['success_conditions']
    forbidden = ('/World/', 'initial_pose_xyzw', 'body_path', 'fixture_metadata',
                 'collision_support_points_m', 'goal_witness', 'support_memory')
    # Public thresholds intentionally describe the timing budget; the actual
    # native support-memory state and contact observations never get projected.
    text = json.dumps({key: value for key, value in public.items() if key != 'public_rules'})
    assert all(key not in text for key in forbidden)
    assert not {'asset_metadata', 'evaluation', 'initial_state', 'goal'} & public.keys()
    metadata = task['asset_metadata']
    assert len(metadata['actors']) == loaded['definition']['objects']['dynamic_count']
    assert len(metadata.get('contextual_props', {})) == 12
    goal = task['public_goal']
    assert {row['object_id'] for row in goal['objects']} == set(metadata['actors'])
    destinations = {row['destination_id'] for row in goal['destinations']}
    assert len(destinations) == (2 if task_id in ('produce_transfer', 'pot_lid_placement') else 3)
    assert all(row['description'] and row['category'] and row['destination'] in destinations
               for row in goal['objects'])
    assert len(goal['context_objects']) == 12
    assert all(key not in json.dumps(goal) for key in forbidden)


@pytest.mark.parametrize('family, task_id', [
    ('unknown', 'hex7x7'), ('card_room', '../hex7x7'), ('card_room', 'missing'),
])
def test_unknown_or_unsafe_selection_fails(family, task_id):
    with pytest.raises(ValueError):
        catalog.task_definition(family, task_id)


def test_task_definitions_return_independent_data():
    task = catalog.task_definition('card_room', 'hex7x7')
    task['objects']['inventory_counts']['black'] = 0
    assert catalog.task_definition('card_room', 'hex7x7')['objects']['inventory_counts']['black'] == 25


def fake_release(tmp_path, monkeypatch, *, parent_mismatch=False, validate_contract=False):
    environment = {'schema': 'workroom-reference-v1', 'family': 'card_room',
                   'version': 'test_room_v1', 'usd_path': '/unused/scene.usda',
                   'usd_sha256': 'a' * 64, 'source_prim': '/World',
                   'dependency_manifest_path': '/unused/scene_description.json',
                   'dependency_manifest_sha256': 'b' * 64}
    actual = deepcopy(environment)
    if parent_mismatch:
        actual['version'] = 'other_room_v1'
    contents = {'scene.json': {'workroom': actual}, 'task.json': {'task_id': 'hex7x7'},
                'initial_state.json': {'joint_positions_rad': {}}}
    for name, value in contents.items():
        (tmp_path / name).write_text(json.dumps(value))
    manifest = {'schema': 'robot_board_game_task_release.v1', 'task_id': 'hex7x7', 'files_sha256': {
        name: digest(tmp_path / name) for name in contents}}
    path = tmp_path / 'release_manifest.json'
    path.write_text(json.dumps(manifest))
    definition = {'family': 'card_room', 'task_id': 'hex7x7', 'runnable': True,
                  'release_schema': manifest['schema'],
                  'source_release': {'path': str(tmp_path), 'manifest_sha256': digest(path)}}
    if not validate_contract:
        from scenes.board_games import runtime_release
        # These minimal fixtures isolate catalog identity and parent checks.
        # Complete canonical contracts are exercised with frozen release copies.
        monkeypatch.setattr(runtime_release, 'validate_board_game_release', lambda *args, **kwargs: None)
    monkeypatch.setattr(catalog, 'task_definition', lambda *_, **__: deepcopy(definition))
    monkeypatch.setattr(catalog, 'scene_definition', lambda _: {'environment': environment})
    verified = []
    monkeypatch.setattr(catalog, 'verify_reference', lambda ref: verified.append(ref))
    return definition, manifest, verified


def test_loading_verifies_release_and_exact_parent_environment(tmp_path, monkeypatch):
    _, _, verified = fake_release(tmp_path, monkeypatch)
    result = catalog.load_task('card_room', 'hex7x7')
    assert result['initial_state'] == {'joint_positions_rad': {}}
    assert verified == [result['scene']['workroom']]
    assert not (tmp_path / 'native_run').exists()


def test_same_family_different_environment_version_is_rejected(tmp_path, monkeypatch):
    _, _, verified = fake_release(tmp_path, monkeypatch, parent_mismatch=True)
    with pytest.raises(ValueError, match='fixed parent scene'):
        catalog.load_task('card_room', 'hex7x7')
    assert not verified


def test_parent_reference_cannot_add_unpinned_fields(tmp_path, monkeypatch):
    definition, manifest, verified = fake_release(tmp_path, monkeypatch)
    path = tmp_path / 'scene.json'
    scene = load_json(path)
    scene['workroom']['unrecorded_override'] = True
    path.write_text(json.dumps(scene))
    manifest['files_sha256']['scene.json'] = digest(path)
    manifest_path = tmp_path / 'release_manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    definition['source_release']['manifest_sha256'] = digest(manifest_path)
    with pytest.raises(ValueError, match='fixed parent scene'):
        catalog.load_task('card_room', 'hex7x7')
    assert not verified


@pytest.mark.parametrize('schema', [None, 'unsupported_task_release.v1'])
def test_implemented_task_requires_explicit_supported_release_type(monkeypatch, schema):
    from scenes.workrooms.card_room.tasks.tic_tac_toe_3x3 import definition
    value = deepcopy(definition.TASK)
    value['release_schema'] = schema
    monkeypatch.setattr(definition, 'TASK', value)
    with pytest.raises(ValueError, match='supported release schema'):
        catalog.task_definition('card_room', 'tic_tac_toe_3x3')


def test_modified_frozen_payload_fails_before_parent_verification(tmp_path, monkeypatch):
    _, _, verified = fake_release(tmp_path, monkeypatch)
    (tmp_path / 'initial_state.json').write_text('{}')
    with pytest.raises(ValueError):
        catalog.load_task('card_room', 'hex7x7')
    assert not verified


def test_unhashed_initial_state_is_rejected(tmp_path, monkeypatch):
    definition, manifest, _ = fake_release(tmp_path, monkeypatch)
    del manifest['files_sha256']['initial_state.json']
    path = tmp_path / 'release_manifest.json'
    path.write_text(json.dumps(manifest))
    definition['source_release']['manifest_sha256'] = digest(path)
    with pytest.raises(ValueError, match='hash scene, task and initial state'):
        catalog.load_task('card_room', 'hex7x7')


def test_current_board_game_catalog_checks_public_rules(tmp_path, monkeypatch):
    definition, manifest, verified = fake_release(tmp_path, monkeypatch, validate_contract=True)
    manifest.update(schema='robot_board_game_task_release.v1', public_rules='rules.json')
    path = tmp_path/'release_manifest.json'
    path.write_text(json.dumps(manifest))
    definition['source_release']['manifest_sha256'] = digest(path)
    # A current tabletop manifest cannot omit its public contract, even when
    # the remaining payloads and manifest are mutually hash-consistent.
    with pytest.raises(ValueError):
        catalog.load_task('card_room', 'hex7x7')
    assert not verified


def test_release_cannot_reference_files_outside_its_directory(tmp_path, monkeypatch):
    definition, manifest, _ = fake_release(tmp_path, monkeypatch)
    manifest['files_sha256']['../outside.py'] = 'a' * 64
    path = tmp_path / 'release_manifest.json'
    path.write_text(json.dumps(manifest))
    definition['source_release']['manifest_sha256'] = digest(path)
    with pytest.raises(ValueError, match='escapes'):
        catalog.load_task('card_room', 'hex7x7')


def test_python_catalog_definitions_are_in_the_complete_source_inventory():
    from scripts.maintenance.freeze_project_sources import source_tree_files
    root = Path(__file__).resolve().parents[1]
    files = source_tree_files(root, 'scenes')
    definitions = {str(path.relative_to(root)) for path in catalog.ROOT.glob('*/tasks/*/definition.py')}
    scenes = {str(path.relative_to(root)) for path in catalog.ROOT.glob('*/scene.py')}
    current = catalog.list_tasks(include_planned=True)
    current_definitions = {
        f"scenes/workrooms/{task['family']}/tasks/{task['task_id']}/definition.py"
        for task in current
    }
    historical_definitions = {
        'scenes/workrooms/laundry_room/tasks/shirt_stacking/definition.py',
        'scenes/workrooms/laundry_room/tasks/trousers_stacking/definition.py',
    }
    assert len(current_definitions) == 51
    assert len(catalog.list_tasks()) == 51
    assert definitions == current_definitions | historical_definitions
    assert not current_definitions & historical_definitions
    assert len(scenes) == len(catalog.CURRENT_FAMILIES)
    assert definitions | scenes <= files


def test_namespace_family_helpers_are_in_wheel_package_data():
    import fnmatch
    import tomllib
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / 'pyproject.toml').read_text())
    patterns = config['tool']['setuptools']['package-data']['scenes.workrooms']
    sources = [path for family in catalog.ROOT.iterdir() if family.is_dir()
               for path in family.glob('*.py')]
    assert sources
    for path in sources:
        relative = str(path.relative_to(catalog.ROOT))
        assert any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns), relative


def test_seed_publication_catalog_json_is_in_wheel_package_data():
    import tomllib
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root/'pyproject.toml').read_text())
    assert config['tool']['setuptools']['package-data'][
        'scenes.workrooms.initial_randomization'] == ['catalog_pins.json']


def test_seed_publication_catalog_json_is_a_default_source_snapshot_resource():
    from scripts.maintenance.freeze_project_sources import DEFAULT_EXTRAS
    relative = 'scenes/workrooms/initial_randomization/catalog_pins.json'
    assert DEFAULT_EXTRAS.count(relative) == 1
    assert (Path(__file__).resolve().parents[1]/relative).is_file()


def test_public_cli_lists_implemented_families(capsys):
    main(['tasks', '--family', 'card_room'])
    result = json.loads(capsys.readouterr().out)
    assert [row['task_id'] for row in result] == ['task0', 'tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7', *CARD_ROOM_EXPANSION_TASKS, *CARD_ROOM_PRACTICE_TASKS]
    main(['tasks', '--family', 'laboratory'])
    result = json.loads(capsys.readouterr().out)
    assert [row['task_id'] for row in result] == [
        'sample_sorting', 'vial_racking', 'cap_placement', 'solid_weighing', 'pipette_transfer', 'tube_clamp_transfer']
    assert all(row['status'] == 'implemented' for row in result)


def test_mechanical_tasks_share_one_scene_with_nine_single_arm_and_one_cooperative_candidate():
    tasks = catalog.list_tasks('mechanical_assembly', include_planned=True, robot_profile='original')
    assert [task['task_id'] for task in tasks] == [
        'parts_sorting', 'peg_insertion', 'shaft_spacer_assembly',
        'fastener_assembly', 'gear_train_assembly',
    ]+list(NIST_TASKS)+['cooperative_insertion']
    assert [task['difficulty'] for task in tasks] == list(range(1, 11))
    assert [task['task_id'] for task in catalog.list_tasks('mechanical_assembly')] == [
        'parts_sorting', 'peg_insertion', 'shaft_spacer_assembly', 'fastener_assembly', 'gear_train_assembly'
    ]+[task['task_id'] for task in published_nist_tasks()]+['cooperative_insertion']
    for task in tasks:
        assert task['layout']['same_environment_for_all_tasks']
        if task['runnable']:
            l_mount_revision(task)
        if task['task_id'] == 'cooperative_insertion':
            assert task['robot_requirements']['articulations'] == 2
            assert task['robot_requirements']['hand_id'] == 'dual_fr3_sharpa_w01'
            assert task['robot_requirements']['active_sides'] == ['left', 'right']
            assert task['robot_requirements']['layout_id'] == 'mechanical_same_side_left_right_sharpa_v1'
            # Frozen seed cases wrap the original cooperative release.
            assert original_definition('mechanical_assembly', task['task_id'])['release_schema'] == (
                'mechanical_cooperative_task_release.v1')
            canonical_release(task)
            assert task['status'] == 'implemented' and task['runnable']
            assert task['qualification']['robot_manipulation'] is False
            assert not task['qualification']['physics_accepted']
            continue
        assert task['robot_requirements']['articulations'] == 1
        assert task['robot_requirements']['layout_id'] == 'mechanical_single_arm_v1'
        assert task['qualification']['robot_manipulation'] is False
        assert not task['qualification']['physics_accepted']
        if task['task_id'] in NIST_TASKS:
            assert task['robot_requirements']['base_position_m'] == [-.23, 0., .78]
            assert task['layout']['parent_scene'] == 'mechanical_assembly/scene.py'
            assert task['runnable'] == (task['status'] == 'implemented')
            assert bool(task['source_release']) == task['runnable']
            continue
        assert {k:v for k,v in task['layout'].items() if k != 'initialization'} == {k:v for k,v in tasks[0]['layout'].items() if k != 'initialization'}
        assert task['status'] == 'implemented' and task['runnable']
        assert task['source_release'] and task['evaluator']['implementation']
        assert task['design']['status'] == 'frozen_candidate_with_separate_scene_evidence'


def test_mechanical_common_design_data_does_not_leak_between_tasks_or_catalog_calls():
    first = catalog.task_definition('mechanical_assembly', 'parts_sorting')
    first['layout']['candidate_zones']['assembly_center_xy_m'][0] = 99.
    first['design']['physical_contract']['forbidden_shortcuts'].clear()
    for task in catalog.list_tasks('mechanical_assembly', include_planned=True):
        if task['task_id'] in (*NIST_TASKS, 'cooperative_insertion'):
            continue  # These independent tasks do not share the legacy design dict.
        assert task['layout']['candidate_zones']['assembly_center_xy_m'][0] != 99.
        assert 'object_pose_playback' in task['design']['physical_contract']['forbidden_shortcuts']


def test_study_has_six_ordered_tasks_with_one_original_robot_and_no_document_slot():
    from scenes.workrooms.parameterized.catalog import QUALIFICATION
    tasks = catalog.list_tasks('study', include_planned=True, robot_profile='original')
    assert [task['task_id'] for task in tasks] == [
        'stationery_sorting', 'book_stacking', 'book_retrieval', 'book_shelving', 'drawer_retrieval', 'reading_stand']
    assert [task['difficulty'] for task in tasks] == [1, 2, 3, 4, 5, 6]
    assert [task['task_number'] for task in tasks] == [1, 2, 3, 4, 5, 6]
    assert catalog.list_tasks('study', robot_profile='original') == tasks
    for task in tasks:
        original = original_definition('study', task['task_id'])
        l_mount_revision(task)
        if task['task_id'] in SEEDED_TASKS:
            assert {k: v for k, v in task['layout'].items() if k != 'initialization'} == {
                k: v for k, v in original['layout'].items() if k != 'initialization'}
            assert task['layout']['initialization'] == 'Ten frozen physical initial cases selected before creation; seed 0 is canonical.'
        else:
            assert task['layout'] == original['layout']
        assert task['robot_requirements'] == tasks[0]['robot_requirements']
        assert task['layout']['same_environment_for_all_tasks']
        assert task['robot_requirements']['articulations'] == 1
        assert task['robot_requirements']['hand_id'] == 'sharpa_w01_right'
        assert task['robot_requirements']['base_position_m'] == [-.23, 0., .78]
        parent = canonical_release(task)
        assert original['release_schema'] == 'study_task_release.v1'
        assert task['rules']['frozen_contract'] == parent['path'] + '/rules.json'
        assert task['evaluator']['thresholds']['minimum_stable_s'] == (.4 if task['task_id'] == 'reading_stand' else .3)
        if task['task_id'] in SEEDED_TASKS:
            # A paired parameterized instance inherits no historical evidence.
            assert task['qualification'] == QUALIFICATION
            assert task['qualification']['status'] == 'prepared_unvalidated'
            manifest = load_json(Path(task['source_release']['path'])/'release_manifest.json')
            assert manifest['task_policy_success_claimed'] is False
        else:
            assert task['release_schema'] == 'study_task_release.v1'
            assert task['qualification']['release_status'] == 'prepared_unvalidated'
            assert not any(task['qualification'][key] for key in (
                'manipulation', 'physics_accepted', 'task_policy_success_claimed'))
        if task['task_id'] == 'drawer_retrieval':
            # The historical evidence remains bound to the original release.
            history = original['qualification']
            assert history['native_preflight']
            assert history['agent_bench_interface']
            assert not history['object_interactions']
            assert not history['full_rule_physical_revalidation_complete']
            assert history['historical_v2_evidence']['object_interactions']
            assert (history['native_interface_evidence']['release_manifest_sha256']
                    == original['source_release']['manifest_sha256'] == parent['manifest_sha256'])
            assert task['evaluator']['thresholds']['maximum_closed_gap_m'] == .012
            assert task['evaluator']['thresholds']['minimum_opening_m'] == .100
            assert task['recommended_budgets']['sim_time_s'] == 180.
            assert task['objects']['inventory_counts'] == {'drawer': 1, 'notebook': 1}
        elif task['task_id'] != 'reading_stand':
            assert not original['qualification']['native_preflight']
    assert tasks[1]['objects']['stack_order'] == ['robotics', 'astronomy', 'botany']
    assert tasks[2]['objects']['retrieval_target_id'] == 'robotics'
    assert tasks[2]['objects']['protected_ids'] == ['astronomy', 'botany']
    assert tasks[3]['objects']['categories'] == {'robotics': 'science', 'botany': 'science', 'architecture': 'arts'}
    with pytest.raises(ValueError, match='Unknown task'):
        catalog.task_definition('study', 'document_placement')
    tasks[0]['layout']['table_top_z_m'] = 99
    tasks[0]['evaluator']['thresholds']['minimum_stable_s'] = 0
    tasks[1]['objects']['stack_order'].clear()
    refreshed = catalog.list_tasks('study', robot_profile='original')
    assert refreshed[0]['layout']['table_top_z_m'] == .78
    assert refreshed[0]['evaluator']['thresholds']['minimum_stable_s'] == .3
    assert len(refreshed[1]['objects']['stack_order']) == 3


@pytest.mark.parametrize('task_id', ['stationery_sorting', 'book_stacking', 'book_retrieval', 'book_shelving', 'drawer_retrieval', 'reading_stand'])
def test_study_catalog_and_canonical_loader_select_the_same_frozen_contract(task_id):
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.study.rules import validate_configuration

    loaded = catalog.load_task('study', task_id)
    definition = loaded['definition']
    _, scene, task, initial = task_inputs(definition['source_release']['path'], task_id)
    frozen_task = deepcopy(task)
    if task_id in SEEDED_TASKS:
        assert 'evaluation_phase' not in frozen_task
        assert task['reset_case']['seed'] == 0
    else:
        assert frozen_task.pop('evaluation_phase') == 'nominal'
    assert (scene, frozen_task, initial) == (loaded['scene'], loaded['task'], loaded['initial_state'])
    assert scene['workroom'] == catalog.scene_definition('study')['environment']
    assert task['public_rules'] == load_json(Path(definition['rules']['frozen_contract']))
    assert task['public_rules']['thresholds'] == definition['evaluator']['thresholds']
    configuration = validate_configuration(task['evaluation']['configuration'])
    assert set(configuration['objects']) == set(definition['objects']['inventory_counts'])
    assert all(row.get('title_zh') for row in configuration['objects'].values())
    assert len(initial['joint_positions_rad']) == 29
    assert loaded['release_manifest']['status'] == 'prepared_unvalidated'
    assert not loaded['release_manifest']['physics_accepted']
    assert not loaded['release_manifest']['task_policy_success_claimed']


def test_gear_plan_requires_assembly_and_actual_tool_transmission_without_tool_id_score():
    task = catalog.task_definition('mechanical_assembly', 'gear_train_assembly')
    gears = task['design']['candidate_gear_geometry']
    assert task['objects']['inventory_counts']['gear'] >= 3
    if 'teeth_per_gear' in gears:
        assert gears['pitch_diameter_m'] == pytest.approx(gears['module_m'] * gears['teeth_per_gear'])
        assert gears['outside_diameter_m'] == pytest.approx(gears['module_m'] * (gears['teeth_per_gear'] + 2))
        assert gears['adjacent_axis_distance_m'] == pytest.approx(gears['pitch_diameter_m'])
    else:
        assert gears['teeth_in_order'] == [20, 32, 20]
        assert gears['pitch_diameters_m'] == pytest.approx(
            [gears['module_m'] * teeth for teeth in gears['teeth_in_order']])
        assert gears['outside_diameters_m'] == pytest.approx(
            [gears['module_m'] * (teeth + 2) for teeth in gears['teeth_in_order']])
        assert gears['adjacent_axis_distance_m'] == pytest.approx(.13)
    # Entry guidance must be distinct from the final rotating fit.
    assert gears['bore_entry_diameter_m'] > gears['bore_diameter_m'] > gears['shaft_diameter_m']
    assert gears['bore_diameter_m'] - gears['shaft_diameter_m'] < .001
    tools = task['design']['tools']
    assert tools['all_tools_free_rigid_bodies']
    assert len(tools['options']) >= 3
    assert not tools['agent_goal_names_correct_tool']
    assert not task['design']['success_design']['tool_id_affects_score']
    assert 'socket_crank' not in task['rules']['intent']
    assert '套筒' not in task['rules']['intent']
    assert 'gear_constraint' in task['design']['physical_contract']['forbidden_shortcuts']


def test_fastener_plan_pins_selected_geometry_and_loose_bolt_default():
    # The selected catalog rows are seed-case wrappers (and L-mount revisions)
    # of this exact frozen plan; its geometry and reset design stay pinned here.
    task = original_definition('mechanical_assembly', 'fastener_assembly')
    selected = catalog.task_definition('mechanical_assembly', 'fastener_assembly', robot_profile='original')
    assert canonical_release(selected) == task['source_release']
    l_mount_revision(selected)
    source = task['objects']['source_asset_bindings']['screwdriver']
    assert task['objects']['asset_bindings']['frozen_scene'] == task['source_release']['path']+'/scene.json'
    assert '/threaded_sharpa/v3/' in source['path']
    assert source['sha256'] == 'ef8b809c9cbe6410c7b8a9eaaf87633a7225f1081bd05bba671460eabaf84f54'
    assert source['source_scale'] == [1., 1., 1.]
    design = task['design']
    assert design['existing_asset_geometry_m']['per_side_geometric_clearance'] == .001125
    assert design['reset_design']['variant'] == 'loose_bolt'
    assert 'not engaged' in design['reset_design']['bolt']
    assert 'not inserted' in design['reset_design']['tool']
    assert task['previous_source_release'] == design['loose_bolt_candidate']['release']
    assert task['source_release'] != design['historical_preinserted']['release']
    assert task['previous_source_release']['path'].endswith('/loose_bolt_candidate_v10_pgs')
    root = Path(task['source_release']['path'])
    assert digest(root/'release_manifest.json') == task['source_release']['manifest_sha256']
    assert 'screw_joint' in design['physical_contract']['forbidden_shortcuts']
    assert not task['qualification']['manipulation']
    assert selected['qualification']['robot_manipulation'] is False
    assert not design['historical_contact_qualification']['robot_grasp_or_task_accepted']
    assert design['existing_asset_geometry_m']['handle_diameter'] == .044
    gear_task = original_definition('mechanical_assembly', 'gear_train_assembly')
    # The task-specific input catalog may omit historical acceptance outputs
    # while preserving the original driver and its actual physical declaration.
    binding = gear_task['objects']['asset_bindings']['screwdriver']
    manifest_path = Path(binding['asset_root'])/'manifest.json'
    assert digest(manifest_path) == binding['manifest_sha256']
    gear_asset = load_json(manifest_path)['assets'][binding['asset_id']]
    assert gear_asset['source_dependencies_sha256'][source['path']] == source['sha256']
    assert gear_asset['source_scale'] == source['source_scale']
    scene = load_json(root/'scene.json')
    driver = next(obj for obj in scene['objects'] if obj['id'] == 'screwdriver')
    asset_manifest = load_json(Path(driver['asset_root'])/'manifest.json')
    fastener_asset = asset_manifest['assets'][driver['asset_id']]
    for field in ('usd_path', 'source_scale', 'mass_kg', 'center_of_mass_m',
                  'inertia_diag_kg_m2', 'collision_parts', 'material_prim_paths'):
        assert gear_asset[field] == fastener_asset[field]


def test_threaded_asset_validator_rejects_changed_source_before_using_its_paths(tmp_path):
    from scenes.workrooms.mechanical_assembly.threaded_assets import validated_threaded_assets
    (tmp_path / 'SOURCE.json').write_text('{"files": []}')
    with pytest.raises(ValueError, match='source hash differs'):
        validated_threaded_assets(tool_root=tmp_path, assembly_root=tmp_path)


def test_threaded_asset_source_hashes_and_license_scopes_are_preserved_when_available():
    from scenes.workrooms.mechanical_assembly.threaded_assets import TOOL_ROOT, validated_threaded_assets
    if not TOOL_ROOT.exists():
        pytest.skip('Shared asset mirror not installed in this checkout')
    contract = validated_threaded_assets()
    assert set(contract['assets']) == {'screwdriver', 'bolt', 'fixture'}
    assert contract['licenses']['tool']['terms'] == 'CC0-1.0'
    assert 'NVIDIA' in contract['licenses']['assembly']['terms']['factory_threads_and_factory_assets']
    assert not contract['native_physics_validated']


@pytest.mark.parametrize('robot_profile', PROFILES)
def test_fastener_catalog_and_canonical_loader_use_identical_frozen_candidate(robot_profile):
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.mechanical_assembly.robot_layout import robot_configuration
    from scenes.workrooms.parameterized.catalog import QUALIFICATION
    from scenes.workrooms.robot_revision.inputs import canonical_hash
    loaded = catalog.load_task('mechanical_assembly', 'fastener_assembly', robot_profile=robot_profile)
    definition = loaded['definition']
    entry, scene, task, initial = task_inputs(definition['source_release']['path'], 'fastener_assembly')
    assert (loaded['scene'], loaded['task'], loaded['initial_state']) == (scene, task, initial)
    if robot_profile == 'original':
        assert loaded['scene']['robot'] == robot_configuration()
    else:
        revision = load_json(Path(definition['source_release']['path'])/'robot_revision.json')
        assert loaded['scene']['robot'] == revision['robot']
        assert revision['parent_robot_sha256'] == canonical_hash(robot_configuration())
    assert loaded['task']['public_rules'] == definition['rules']['contract']
    # Seed cases (and the L-mount revision) wrap the frozen loose-bolt candidate.
    candidate = original_definition('mechanical_assembly', 'fastener_assembly')['source_release']
    candidate_manifest = load_json(Path(candidate['path'])/'release_manifest.json')
    assert digest(Path(candidate['path'])/'release_manifest.json') == candidate['manifest_sha256']
    assert candidate_manifest['reset_variant'] == 'loose_bolt'
    assert task['evaluation']['reset_variant'] == 'loose_bolt'
    assert [step['id'] for step in task['steps']] == [
        'pickup_bolt', 'start_thread', 'insert_tool', 'settle', 'advance', 'release']
    objects = {row['id']: row for row in scene['objects']}
    fixtures = {row['id']: row for row in scene['fixtures']}
    assert {'bolt_supply_rest', 'tool_supply_rest', 'thread_entry_guide'} <= fixtures.keys()
    assert objects['bolt']['position_m'][:2] != fixtures['threaded_fixture']['position_m'][:2]
    assert objects['screwdriver']['position_m'][:2] != objects['bolt']['position_m'][:2]
    assert task['reset_conditions']['pose_snap'] is False
    assert task['evaluation']['engagement_requires_post_reset_contacts'] is True
    assert loaded['task']['public_rules']['success']['nominal_turn_degrees'] == 30.
    assert loaded['task']['public_rules']['success']['nominal_additional_advance_m'] == .00025
    assert loaded['release_manifest']['status'] == 'prepared_unvalidated'
    # A paired parameterized instance inherits no native or physics evidence.
    assert definition['qualification'] == QUALIFICATION
    tasks = [task for task in catalog.list_tasks(include_planned=True, robot_profile=robot_profile)
             if task['family'] in FAMILIES]
    assert sum(task['status'] == 'implemented' for task in tasks) == 42+len(published_nist_tasks())
    assert sum(task['status'] == 'planned' for task in tasks) == 4-len(published_nist_tasks())+len(CARD_ROOM_PLANNED_TASKS)
    assert {(task['family'], task['task_id']) for task in tasks if task['status'] == 'planned'} == {
        ('card_room', task_id) for task_id in CARD_ROOM_PLANNED_TASKS}


@pytest.mark.parametrize('robot_profile', PROFILES)
@pytest.mark.parametrize('task_id', ['parts_sorting', 'peg_insertion', 'shaft_spacer_assembly', 'gear_train_assembly'])
def test_mechanical_scene_handoffs_load_canonical_supply_and_frozen_rules(task_id, robot_profile):
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.parameterized.catalog import QUALIFICATION
    loaded = catalog.load_task('mechanical_assembly', task_id, robot_profile=robot_profile)
    definition = loaded['definition']
    _, scene, task, initial = task_inputs(definition['source_release']['path'], task_id)
    assert (scene, task, initial) == (loaded['scene'], loaded['task'], loaded['initial_state'])
    assert task['public_rules'] == definition['rules']['contract']
    assert not task.get('engineering_probe')
    assert not scene['mechanical_task_provenance'].get('engineering_probe')
    assert scene['isaacsim']['physics_dt_s'] == definition['evaluator']['thresholds']['physics_dt_s']
    assert len(initial['joint_positions_rad']) == 29
    assert loaded['release_manifest']['status'] == 'prepared_unvalidated'
    assert definition['qualification'] == QUALIFICATION
    # Both profiles wrap the same frozen original supply release.
    original = original_definition('mechanical_assembly', task_id)
    assert frozen_candidate(definition) == original['source_release']
    if task_id == 'parts_sorting':
        assert [row['position_m'][1] for row in scene['objects']] == [-.17, 0., .17]
        assert scene['objects'][2]['asset_id'] == 'sorting_ring_sdf1024'
    elif task_id == 'gear_train_assembly':
        candidate = frozen_candidate(definition)
        assert candidate == definition['design']['supply_layout_candidate']['release']
        assert definition['candidate_referee']['release'] == candidate
        assert definition['previous_source_release'] != candidate
        assert digest(Path(candidate['path'])/'release_manifest.json') == candidate['manifest_sha256']
        assert task['supply_layout']['revision'] == 'table_inset_v1'
        assert task['evaluation']['gear_teeth'] == {'input_gear': 20, 'idler_gear': 32, 'output_gear': 20}
        assert len(scene['objects']) == 6 and task['supply_layout']['display_pads_or_supply_rests'] is False
    else:
        assert scene['isaacsim']['solver_type'] == 'PGS'
        assert all(row['position_m'][0] == .25 for row in scene['objects'])
        if task_id == 'shaft_spacer_assembly':
            assert all(row['asset_root'].endswith('/dense1024_v1') for row in scene['objects'])


def test_catalog_rejects_definition_release_schema_mismatch(tmp_path, monkeypatch):
    definition, _, verified = fake_release(tmp_path, monkeypatch)
    definition['release_schema_version'] = 'isaacsim-mechanical-fastener-release-v1'
    with pytest.raises(ValueError, match='schema differs'):
        catalog.load_task('card_room', 'hex7x7')
    assert not verified


def test_mechanical_catalog_invokes_canonical_frozen_contract(tmp_path, monkeypatch):
    from bench.isaacsim import release
    definition, manifest, verified = fake_release(tmp_path, monkeypatch)
    manifest.update(schema='mechanical_fastener_task_release.v1',
                    schema_version='isaacsim-mechanical-fastener-release-v1')
    definition['release_schema'] = manifest['schema']
    path = tmp_path/'release_manifest.json'
    path.write_text(json.dumps(manifest))
    definition['source_release']['manifest_sha256'] = digest(path)
    validated = []
    monkeypatch.setattr(release, 'validate', lambda root: validated.append(root))
    def reject(root):
        assert root == tmp_path
        raise ValueError('Canonical mechanical contract rejection')
    monkeypatch.setattr(release, 'load_evaluator', reject)
    with pytest.raises(ValueError, match='Canonical mechanical contract rejection'):
        catalog.load_task('card_room', 'hex7x7')
    assert validated == [tmp_path]
    assert not verified
