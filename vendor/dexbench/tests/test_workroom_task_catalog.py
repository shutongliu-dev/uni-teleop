"""Scene/task separation and fail-closed loading; no native simulation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from bench.paths import load_json
from scenes.workrooms import catalog
from scenes.workrooms.registry import FAMILIES, main

KITCHEN_TASKS = ('breakfast_serving', 'produce_transfer', 'utensil_sorting',
                 'pot_lid_placement', 'pantry_stocking', 'table_clearing')

NIST_TASKS = ('nist_connector_insertion', 'nist_belt_drive_assembly',
              'nist_cable_routing', 'nist_wire_harness')


def published_nist_tasks():
    return [catalog.task_definition('mechanical_assembly', name) for name in NIST_TASKS
            if catalog.task_definition('mechanical_assembly', name)['status'] == 'implemented']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_six_fixed_scenes_include_single_arm_card_room_task0():
    scenes = catalog.list_scenes()
    assert [row['family'] for row in scenes] == list(FAMILIES)
    for row in scenes:
        assert row['environment']['family'] == row['family']
        assert row['coordinates']['table_top_z_m'] == (.90 if row['family'] in ('kitchen', 'laboratory') else .78)
        tasks = catalog.list_tasks(row['family'], include_planned=True)
        assert len(tasks) == {'mechanical_assembly': 9, 'card_room': 4, 'living_room': 4, 'study': 4, 'kitchen': 6, 'laboratory': 5}[row['family']]
        assert all(task['family'] == row['family'] for task in tasks)
    runnable = catalog.list_tasks()
    new = published_nist_tasks()
    assert {(task['family'], task['task_id']) for task in runnable} == {('card_room', game) for game in ('task0', 'tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7')} | {('mechanical_assembly', task) for task in ('parts_sorting', 'peg_insertion', 'shaft_spacer_assembly', 'fastener_assembly', 'gear_train_assembly')} | {('living_room', task) for task in ('remote_operation', 'water_pouring', 'tower_building', 'tower_extraction')} | {('study', task) for task in ('stationery_sorting', 'book_stacking', 'book_retrieval', 'book_shelving')} | {('kitchen', task) for task in KITCHEN_TASKS} | {('laboratory', task) for task in ('sample_sorting', 'vial_racking', 'cap_placement', 'solid_weighing', 'pipette_transfer')} | {('mechanical_assembly', task['task_id']) for task in new}
    assert sorted(task['difficulty'] for task in runnable) == [0]+[1]*6+[2]*6+[3]*6+[4]*5+[5]*3+[6]+[task['task_number'] for task in new]
    assert len(runnable) == 28+len(new)
    assert sum(task['status'] == 'planned' for task in catalog.list_tasks(include_planned=True)) == 4-len(new)
    assert not runnable[0]['qualification']['manipulation']


def test_planned_task_refuses_loading_before_reading_any_release(monkeypatch):
    from scenes.workrooms.laboratory.catalog_definition import task_definition
    planned = task_definition('sample_sorting')
    planned['runnable'] = False
    monkeypatch.setattr(catalog, 'task_definition', lambda *args: deepcopy(planned))
    monkeypatch.setattr(catalog, 'scene_definition',
                        lambda family: pytest.fail('A planned task must stop before environment loading'))
    with pytest.raises(ValueError, match='cannot run'):
        catalog.load_task('laboratory', 'sample_sorting')


def test_kitchen_catalog_exposes_six_frozen_candidates_without_loading_assets(monkeypatch):
    expected = list(KITCHEN_TASKS)
    monkeypatch.setattr(catalog, 'verify_reference',
                        lambda *_: pytest.fail('Listing metadata must not load shared assets'))
    tasks = catalog.list_tasks('kitchen')
    assert [task['task_id'] for task in tasks] == expected
    assert [task['difficulty'] for task in tasks] == list(range(1, 7))
    for task in tasks:
        assert task['status'] == 'implemented' and task['runnable']
        version = ('v4' if task['task_id'] == 'pot_lid_placement' else
                   'v2' if task['task_number'] >= 5 else 'v3')
        assert task['source_release']['path'].endswith('/kitchen/'+task['task_id']+'/'+version)
        assert len(task['source_release']['manifest_sha256']) == 64
        assert task['objects']['asset_bindings'] and task['layout']['initialization']
        assert task['evaluator']['implementation'] == 'scenes.workrooms.kitchen.evaluation'
        assert task['rules']['contract']['task_id'] == task['task_id']
        assert task['rules']['contract']['schema'] == ('kitchen_rules.v3' if task['task_number'] >= 5 else 'kitchen_rules.v2')
        assert task['objects']['contextual_prop_count'] == 12
        assert task['objects']['contextual_props_scored'] is False
        assert 'candidate_success' not in task['rules']
        assert 'candidate_design' not in task
        assert task['layout']['same_environment_for_all_tasks']
        assert task['robot_requirements']['articulations'] == 1
        assert task['robot_requirements']['base_position_m'] == [.65, 0., .78]
        assert not task['difficulty_calibrated']
        assert not task['qualification']['manipulation']
        assert not task['qualification']['physics_accepted']
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


def test_kitchen_current_rules_are_independent_of_historical_design_candidates():
    from scenes.workrooms.kitchen.task_design import task_plan, task_plans
    task = catalog.task_definition('kitchen', 'breakfast_serving')
    task['objects']['inventory_counts']['cup'] = 99
    task['rules']['contract']['referee']['thresholds']['stable_s'] = 99.
    current = catalog.task_definition('kitchen', 'breakfast_serving')
    assert current['objects']['inventory_counts']['cup'] == 1
    assert current['rules']['contract']['referee']['thresholds']['stable_s'] == .4
    assert current['rules']['contract']['referee']['thresholds']['carry_clearance_m'] == .015
    plans = task_plans()
    assert 'Historical' in plans['catalog_note'] and plans['candidate_parameters']
    plans['robot']['base_position_m'][0] = 99.
    assert task_plans()['robot']['base_position_m'][0] == .65
    assert task_plan('breakfast_serving')['success_candidate']['stability_window_s'] == 1.
    utensils = catalog.task_definition('kitchen', 'utensil_sorting')['rules']['contract']
    assert any('柄朝两端均可' in text for text in utensils['referee']['success_conditions'])
    cookware = catalog.task_definition('kitchen', 'pot_lid_placement')['rules']['contract']
    assert cookware['referee']['thresholds']['lid_seat_height_tolerance_m'] == .008


@pytest.mark.parametrize('task_id', KITCHEN_TASKS)
def test_kitchen_catalog_and_canonical_loader_bind_same_release(task_id):
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.kitchen.task_robot_layout import robot_configuration
    loaded = catalog.load_task('kitchen', task_id)
    definition = loaded['definition']
    _, scene, task, initial = task_inputs(definition['source_release']['path'], task_id)
    # Catalog returns the portable frozen path; runtime resolves it against
    # the same verified release before handing it to the native worker.
    frozen_scene = deepcopy(loaded['scene'])
    observer = frozen_scene['native_task_bundle']['observer']
    observer['path'] = str(Path(definition['source_release']['path'])/observer['path'])
    assert (scene, task, initial) == (frozen_scene, loaded['task'], loaded['initial_state'])
    assert scene['robot'] == robot_configuration(task_id)
    assert task['public_rules'] == definition['rules']['contract']
    assert task['public_rules']['referee']['thresholds'] == definition['evaluator']['thresholds']
    assert len(scene['native_task_bundle']['rigid_bodies']) == definition['objects']['dynamic_count']
    assert len(initial['joint_positions_rad']) == 29
    assert loaded['release_manifest']['status'] == 'prepared_unvalidated'
    assert not loaded['release_manifest']['physics_accepted']
    assert not definition['qualification']['manipulation']
    for evidence in definition['qualification']['scene_contact_evidence']:
        path = Path(evidence['path'])
        assert digest(path) == evidence['sha256']
        report = load_json(path)
        assert report['passed'] is True
        assert report['task_id'] == task_id
        assert report['release_manifest_sha256'] == definition['source_release']['manifest_sha256']
        assert report['task_success_claimed'] is False
        support = report['checks']['native_support']
        assert support['passed'] is True
        assert support['complete_native_contact_coverage'] is True
        assert support['stability_method'] == 'native_rolling_pose_and_independent_saved_pose'
        assert support['duration_s']+1.e-8 >= support['terminal_dwell_s']
        assert support['terminal_dwell_s'] >= task['public_rules']['referee']['thresholds']['stable_s']
        assert all(row['passed'] and not row['tensor_velocity_gate_applied']
                   and row['saved_pose_stability']['passed'] for row in support['objects'].values())


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
    monkeypatch.setattr(catalog, 'task_definition', lambda *_: deepcopy(definition))
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
    assert len(definitions) == 32 and len(scenes) == 6
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


def test_public_cli_lists_implemented_families(capsys):
    main(['tasks', '--family', 'card_room'])
    result = json.loads(capsys.readouterr().out)
    assert [row['task_id'] for row in result] == ['task0', 'tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7']
    main(['tasks', '--family', 'laboratory'])
    result = json.loads(capsys.readouterr().out)
    assert [row['task_id'] for row in result] == [
        'sample_sorting', 'vial_racking', 'cap_placement', 'solid_weighing', 'pipette_transfer']
    assert all(row['status'] == 'implemented' for row in result)


def test_mechanical_tasks_share_one_scene_and_single_arm_layout_as_frozen_candidates():
    tasks = catalog.list_tasks('mechanical_assembly', include_planned=True)
    assert [task['task_id'] for task in tasks] == [
        'parts_sorting', 'peg_insertion', 'shaft_spacer_assembly',
        'fastener_assembly', 'gear_train_assembly',
    ]+list(NIST_TASKS)
    assert [task['difficulty'] for task in tasks] == list(range(1, 10))
    assert [task['task_id'] for task in catalog.list_tasks('mechanical_assembly')] == [
        'parts_sorting', 'peg_insertion', 'shaft_spacer_assembly', 'fastener_assembly', 'gear_train_assembly'
    ]+[task['task_id'] for task in published_nist_tasks()]
    for task in tasks:
        assert task['layout']['same_environment_for_all_tasks']
        assert task['robot_requirements']['articulations'] == 1
        assert task['robot_requirements']['layout_id'] == 'mechanical_single_arm_v1'
        assert not task['qualification']['manipulation']
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
        if task['task_id'] in NIST_TASKS:
            continue  # These independent tasks do not share the legacy design dict.
        assert task['layout']['candidate_zones']['assembly_center_xy_m'][0] != 99.
        assert 'object_pose_playback' in task['design']['physical_contract']['forbidden_shortcuts']


def test_study_has_four_ordered_tasks_with_one_original_robot_and_no_document_slot():
    tasks = catalog.list_tasks('study', include_planned=True)
    assert [task['task_id'] for task in tasks] == [
        'stationery_sorting', 'book_stacking', 'book_retrieval', 'book_shelving']
    assert [task['difficulty'] for task in tasks] == [1, 2, 3, 4]
    assert [task['task_number'] for task in tasks] == [1, 2, 3, 4]
    assert catalog.list_tasks('study') == tasks
    for task in tasks:
        assert task['layout'] == tasks[0]['layout']
        assert task['robot_requirements'] == tasks[0]['robot_requirements']
        assert task['layout']['same_environment_for_all_tasks']
        assert task['robot_requirements']['articulations'] == 1
        assert task['robot_requirements']['hand_id'] == 'sharpa_w01_right'
        assert task['robot_requirements']['base_position_m'] == [-.23, 0., .78]
        assert task['release_schema'] == 'study_task_release.v1'
        assert task['rules']['frozen_contract'] == task['source_release']['path'] + '/rules.json'
        assert task['evaluator']['thresholds']['minimum_stable_s'] == .3
        assert task['qualification']['release_status'] == 'prepared_unvalidated'
        assert not any(task['qualification'][key] for key in (
            'native_preflight', 'manipulation', 'physics_accepted', 'task_policy_success_claimed'))
    assert tasks[1]['objects']['stack_order'] == ['robotics', 'astronomy', 'botany']
    assert tasks[2]['objects']['retrieval_target_id'] == 'robotics'
    assert tasks[2]['objects']['protected_ids'] == ['astronomy', 'botany']
    assert tasks[3]['objects']['categories'] == {'robotics': 'science', 'botany': 'science', 'architecture': 'arts'}
    with pytest.raises(ValueError, match='Unknown task'):
        catalog.task_definition('study', 'document_placement')
    tasks[0]['layout']['table_top_z_m'] = 99
    tasks[0]['evaluator']['thresholds']['minimum_stable_s'] = 0
    tasks[1]['objects']['stack_order'].clear()
    refreshed = catalog.list_tasks('study')
    assert refreshed[0]['layout']['table_top_z_m'] == .78
    assert refreshed[0]['evaluator']['thresholds']['minimum_stable_s'] == .3
    assert len(refreshed[1]['objects']['stack_order']) == 3


@pytest.mark.parametrize('task_id', ['stationery_sorting', 'book_stacking', 'book_retrieval', 'book_shelving'])
def test_study_catalog_and_canonical_loader_select_the_same_frozen_contract(task_id):
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.study.rules import validate_configuration

    loaded = catalog.load_task('study', task_id)
    definition = loaded['definition']
    _, scene, task, initial = task_inputs(definition['source_release']['path'], task_id)
    frozen_task = deepcopy(task)
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
    task = catalog.task_definition('mechanical_assembly', 'fastener_assembly')
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
    assert task['source_release'] == design['loose_bolt_candidate']['release']
    assert task['source_release'] != design['historical_preinserted']['release']
    assert task['source_release']['path'].endswith('/loose_bolt_candidate_v10_pgs')
    assert 'screw_joint' in design['physical_contract']['forbidden_shortcuts']
    assert not task['qualification']['manipulation']
    assert not design['historical_contact_qualification']['robot_grasp_or_task_accepted']
    assert design['existing_asset_geometry_m']['handle_diameter'] == .044
    gear_task = catalog.task_definition('mechanical_assembly', 'gear_train_assembly')
    assert gear_task['objects']['asset_bindings']['screwdriver'] == source


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


def test_fastener_catalog_and_canonical_loader_use_identical_frozen_candidate():
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.mechanical_assembly.robot_layout import robot_configuration
    loaded = catalog.load_task('mechanical_assembly', 'fastener_assembly')
    definition = loaded['definition']
    entry, scene, task, initial = task_inputs(definition['source_release']['path'], 'fastener_assembly')
    assert (loaded['scene'], loaded['task'], loaded['initial_state']) == (scene, task, initial)
    assert loaded['scene']['robot'] == robot_configuration()
    assert loaded['task']['public_rules'] == definition['rules']['contract']
    assert loaded['release_manifest']['reset_variant'] == 'loose_bolt'
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
    assert not definition['qualification']['native_preflight']
    assert not definition['qualification']['physics_accepted']
    tasks = catalog.list_tasks(include_planned=True)
    assert sum(task['status'] == 'implemented' for task in tasks) == 28+len(published_nist_tasks())
    assert sum(task['status'] == 'planned' for task in tasks) == 4-len(published_nist_tasks())


@pytest.mark.parametrize('task_id', ['parts_sorting', 'peg_insertion', 'shaft_spacer_assembly', 'gear_train_assembly'])
def test_mechanical_scene_handoffs_load_canonical_supply_and_frozen_rules(task_id):
    from bench.isaacsim.release import task_inputs
    loaded = catalog.load_task('mechanical_assembly', task_id)
    definition = loaded['definition']
    _, scene, task, initial = task_inputs(definition['source_release']['path'], task_id)
    assert (scene, task, initial) == (loaded['scene'], loaded['task'], loaded['initial_state'])
    assert task['public_rules'] == definition['rules']['contract']
    assert not task.get('engineering_probe')
    assert not scene['mechanical_task_provenance'].get('engineering_probe')
    assert scene['isaacsim']['physics_dt_s'] == definition['evaluator']['thresholds']['physics_dt_s']
    assert len(initial['joint_positions_rad']) == 29
    assert loaded['release_manifest']['status'] == 'prepared_unvalidated'
    assert not definition['qualification']['physics_accepted']
    assert not definition['qualification']['manipulation']
    for evidence in definition['qualification']['scene_contact_evidence']:
        assert digest(Path(evidence['path'])) == evidence['sha256']
    if task_id == 'parts_sorting':
        assert [row['position_m'][1] for row in scene['objects']] == [-.17, 0., .17]
        assert scene['objects'][2]['asset_id'] == 'sorting_ring_sdf1024'
    elif task_id == 'gear_train_assembly':
        assert definition['source_release'] == definition['design']['supply_layout_candidate']['release']
        assert definition['candidate_referee']['release'] == definition['source_release']
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
