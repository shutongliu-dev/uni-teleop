"""Installation selection and single-hand evidence, never native qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.paths import DATA_ROOT
from bench.protocol.robot_configurations import project_robot_configurations
from scenes.workrooms.laundry_room import composition, release, robot_configurations as layouts
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scenes.workrooms.laundry_room.rules import CURRENT_TASK_IDS, configuration_rules, bind_task_rules, digest
from test_laundry_robot_rules import robot_task, robot_frame
from test_laundry_practical_rules import goal_vertices

STANDS = Path(f'{DATA_ROOT}/assets/task_assets/laundry_room/robot_installation/contact_options_v1')


@pytest.mark.parametrize('configuration_id', layouts.CONFIGURATION_IDS)
def test_nearer_installation_is_explicitly_versioned_and_old_candidate_is_reproducible(configuration_id):
    old = layouts.installation(configuration_id, STANDS/configuration_id/'manifest.json')
    path = STANDS.parent/'contact_options_v2'/configuration_id/'manifest.json'
    new = layouts.installation(configuration_id, path)
    assert 'layout_version' not in old and new['layout_version'] == 2
    assert old['layout_id'].endswith('_v1') and new['layout_id'].endswith('_v2')
    inputs = composition.build_task_inputs('towel_folding', configuration_id=configuration_id, stand_manifest=path)
    assert inputs['scene']['robot'] == layouts.robot_inputs(configuration_id, version=2)[0]
    poses = layouts.placements(configuration_id, version=2)
    if configuration_id != 'dual_opposed':
        assert all(p['base_position_m'][1] == -1.15 for p in poses.values())
        assert all(p['base_position_m'][1] == -1.25 for p in layouts.placements(configuration_id, version=1).values())
    else:
        assert poses == layouts.placements(configuration_id, version=1)


@pytest.mark.parametrize('configuration_id', layouts.CONFIGURATION_IDS)
@pytest.mark.parametrize('task_id', CURRENT_TASK_IDS)
def test_configurations_keep_cloth_physics_and_complete_reset(task_id, configuration_id):
    original = composition.build_task_inputs(task_id)
    configured = composition.build_task_inputs(task_id, configuration_id=configuration_id,
        stand_manifest=STANDS/configuration_id/'manifest.json')
    scene, task, initial = (configured[k] for k in ('scene', 'task', 'initial_state'))
    assert scene['cloths'] == original['scene']['cloths']
    assert initial['cloths_vertices_m'] == original['initial_state']['cloths_vertices_m']
    assert task['evaluation']['thresholds'] == original['task']['evaluation']['thresholds']
    assert scene['workroom'] == original['scene']['workroom']
    assert scene['laundry_task_provenance']['cloth_physics_options'] == original['scene']['isaacsim']
    model = public_robot_geometry(robot_configuration(scene))
    assert set(initial['joint_positions_rad']) == set(model.names)
    assert len(model.names) == (29 if configuration_id == 'single_arm' else 58)
    assert task['active_sides'] == layouts.active_sides(configuration_id)
    assert len(scene['cameras']) == (4 if configuration_id == 'single_arm' else 5)
    assert scene['named_cloth_initialization'] == 'author-only'
    assert task['robot_configuration_id'] == configuration_id
    assert not task['physics_accepted'] and not task['task_policy_success_claimed']


def single_task():
    task = robot_task()
    task.update(robot_configuration_id='single_arm', active_sides=['right'])
    task['public_rules'] = configuration_rules(task['id'], 0, task['evaluation']['rule_profile'], 'single_arm')
    task['evaluation']['public_rules_sha256'] = digest(task['public_rules'])
    return bind_task_rules(task)


def single_frame(task, step, points=None, **kwargs):
    frame = robot_frame(task, step, points, **kwargs)
    hand = frame['robot_interaction']['hands'].pop('left')
    frame['robot_interaction']['hands']['right'] = hand
    return frame


def test_single_hand_can_earn_operation_and_release_without_a_second_hand():
    task = single_task(); scorer = SequenceEvaluator(task)
    scorer.observe(single_frame(task, 0))
    scorer.observe(single_frame(task, 1, near=True))
    goal = goal_vertices(task)
    scorer.observe(single_frame(task, 2, goal, near=True, x=.015))
    for step in range(3, 254):
        score = scorer.observe(single_frame(task, step, goal))
    assert score['success'], score
    assert not score['robot_manipulation_qualified']


def test_single_cooked_observer_keeps_arm_clearance_separate_from_hand_operation():
    import numpy as np
    from test_laundry_robot_native_runtime import geometry, towel
    from scenes.workrooms.laundry_room.native_observer import CookedRobotGeometry
    export, names, topology, poses = geometry()
    keep = [i for i, name in enumerate(names) if not name.startswith('left_')]
    names, poses = [names[i] for i in keep], poses[keep]
    export['colliders'] = [row for row in export['colliders'] if row['body_name'] in names]
    auditor = CookedRobotGeometry(export, names, topology, {'towel': .003}, active_sides=['right'])
    assert set(auditor.check(towel(.014), poses)['hands']) == {'right'}
    arm = next(i for i, name in enumerate(names) if not name.startswith('right_'))
    poses[arm, :3] = 0.
    result = auditor.check(towel(.014), poses)
    assert not result['fully_separated'] and not result['hand_near_cloth']
    with pytest.raises(ValueError, match='hide a second robot'):
        CookedRobotGeometry(export, names+['left_hand_flange'], topology, {'towel': .003}, active_sides=['right'])


@pytest.mark.parametrize('task_id', CURRENT_TASK_IDS[:3])
def test_new_towel_material_is_bound_without_changing_legacy_thickness(monkeypatch, task_id):
    from scenes.workrooms.laundry_room import assets
    metadata = assets.candidate_metadata(task_id, material_profile='towel_4mm_density_v1')
    monkeypatch.setattr(composition, 'candidate_metadata', lambda task: deepcopy(metadata))
    inputs = composition.build_task_inputs(task_id, configuration_id='single_arm',
        stand_manifest=STANDS/'single_arm/manifest.json')
    SequenceEvaluator(inputs['task'])
    assert inputs['scene']['isaacsim']['cloth_thickness_m'] == .004
    bad = deepcopy(inputs['task']); bad['asset_metadata']['schema'] = 'laundry_towel_asset.v1'
    bad['evaluation']['metadata'] = deepcopy(bad['asset_metadata'])
    with pytest.raises(ValueError, match='finite-thickness'):
        SequenceEvaluator(bad)


@pytest.mark.parametrize('fault', ['extra_hand', 'missing_hand', 'passive', 'attachment'])
def test_single_hand_still_requires_complete_unassisted_operation_evidence(fault):
    task = single_task(); scorer = SequenceEvaluator(task); goal = goal_vertices(task)
    for step in range(254):
        frame = single_frame(task, step, goal)
        if fault == 'extra_hand':
            frame['robot_interaction']['hands']['left'] = deepcopy(frame['robot_interaction']['hands']['right'])
        elif fault == 'missing_hand':
            frame['robot_interaction']['hands'] = {}
        elif fault == 'attachment':
            frame['controlled_vertices']['towel'] = [0]
        score = scorer.observe(frame)
    assert not score['success']
    assert score['status'] == ('ongoing' if fault == 'passive' else 'unknown')


@pytest.mark.parametrize('configuration_id', layouts.CONFIGURATION_IDS)
def test_selected_configuration_release_keeps_cloth_reset_and_rejects_relabel(tmp_path, monkeypatch, configuration_id):
    from scenes.workrooms.laundry_room import assets
    from scripts.maintenance import freeze_project_sources
    from bench.runtime.robot_configurations import checked_configuration_release
    from bench.isaacsim.release import digest as file_digest, task_inputs
    monkeypatch.setattr(assets, 'ASSETS', tmp_path)
    monkeypatch.setattr(release, 'DATASETS', tmp_path)
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot', lambda *a, **k: {'fixture_only': True})
    monkeypatch.setattr(release, 'asset_inventory', lambda scene: {
        path: {'sha256': sha} for path, sha in scene['laundry_task_provenance']['asset_candidate']['files_sha256'].items()})
    asset = tmp_path/'asset'; assets.prepare_assets('towel_folding', asset)
    root = tmp_path/'release'
    manifest = release.prepare_task_release('towel_folding', root, asset_manifest=asset/'manifest.json',
        configuration_id=configuration_id, stand_manifest=STANDS/configuration_id/'manifest.json')
    assert manifest['configuration_id'] == configuration_id
    _, scene, task, initial = task_inputs(root, 'towel_folding', seed=7)
    assert task['episode_seed'] == 7 and 'cloths_vertices_m' in initial
    requirement = {'hand_id': scene['robot']['hand_id'], 'assembly_version': scene['robot']['assembly_version']}
    reference = {'path': str(root), 'manifest_sha256': file_digest(root/'release_manifest.json')}
    def binding(selected):
        return {'task_id': 'towel_folding', 'releases': {selected: reference},
                'robot_configurations': {selected: {'robot_requirements': requirement}}}
    assert checked_configuration_release(binding(configuration_id), configuration_id) == root
    other = 'dual_opposed' if configuration_id == 'single_arm' else 'single_arm'
    with pytest.raises(ValueError, match='another robot installation'):
        checked_configuration_release(binding(other), other)


def test_configured_catalog_supports_opposed_default_and_public_agent_selection(monkeypatch, tmp_path):
    from scenes.workrooms.laundry_room import robot_configuration_catalog
    from scenes.workrooms.robot_configurations import attach_robot_configurations, configured_definition
    from scenes.workrooms.catalog import task_definition
    from agent.model_client.scene_tasks import select_task, select_configuration
    from scenes.workrooms.laundry_room.randomization import catalog as seeded_catalog
    from scenes.workrooms.robot_revision import catalog as revised_catalog
    empty = tmp_path/'seed_pins.json'
    empty.write_text(json.dumps({'schema': 'laundry_seed_catalog.v1', 'tasks': {}}))
    monkeypatch.setattr(seeded_catalog, 'PINS', empty)
    revision = tmp_path/'revision_pins.json'
    revision.write_text(json.dumps({'schema': 'robot_revision_catalog.v1', 'tasks': {}}))
    monkeypatch.setattr(revised_catalog, 'PINS', revision)
    from scenes.workrooms.parameterized import catalog as parameterized
    neutral = tmp_path/'parameterized_pins.json'
    neutral.write_text(json.dumps({'schema': parameterized.SCHEMA, 'tasks': {}}))
    monkeypatch.setattr(parameterized, 'PINS', neutral)
    definition = task_definition('laundry_room', 'towel_folding')
    options = {}
    for i, identifier in enumerate(layouts.CONFIGURATION_IDS):
        source = deepcopy(definition['source_release']) if identifier == 'dual_opposed' else {
            'path': '/private/fixture/'+identifier, 'manifest_sha256': str(i+1)*64}
        options[identifier] = {'source_release': source, 'release_schema': release.SCHEMA,
            'release_schema_version': release.VERSION, 'qualification': {'native_preflight': False},
            'layout': layouts.public_layout(identifier),
            'robot_requirements': {'hand_id': 'sharpa_w01_right' if identifier == 'single_arm' else 'dual_fr3_sharpa_w01',
                'assembly_version': 'isaacsim_6_1_v2', 'articulations': 1 if identifier == 'single_arm' else 2,
                'active_sides': layouts.active_sides(identifier)},
            'rules': {'contract': configuration_rules('towel_folding', 0, 'laundry_practical_task.v2', identifier)}}
    monkeypatch.setattr(robot_configuration_catalog, 'RELEASES', {'towel_folding': {
        'base_release': definition['source_release'], 'default_configuration_id': 'dual_opposed',
        'configurations': options}})
    published = attach_robot_configurations(definition)
    assert published['default_configuration_id'] == 'dual_opposed'
    public = project_robot_configurations(published['robot_configurations'])
    assert '/private/' not in json.dumps(public)
    selection = select_task('task2', family='laundry_room')
    selected = select_configuration(selection, {'robot_configurations': public}, 'single_arm')
    assert selected.robot_requirements['articulations'] == 1
    configured = configured_definition(published, 'single_arm')
    assert configured['rules'] == options['single_arm']['rules']
    assert configured['qualification']['native_preflight'] is False
