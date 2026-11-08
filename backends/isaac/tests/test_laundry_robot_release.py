"""Explicit robot promotion preserves historical engineering releases."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.laundry_room import assets, composition, release
from scenes.workrooms.laundry_room.robot_layout import LAYOUT_ID, STAND_ROOT, STAND_MANIFEST_SHA256
from scenes.workrooms.laundry_room.rules import ROBOT_CONTROL


def test_robot_inputs_expose_one_agent_two_hands_and_separate_physics():
    ordinary = composition.build_task_inputs('towel_folding')
    robot = composition.build_task_inputs('towel_folding', robot_layout=LAYOUT_ID)
    scene, task, initial = (robot[key] for key in ('scene', 'task', 'initial_state'))
    assert task['task_kind'] == 'single_agent'
    assert task['active_sides'] == ['left', 'right']
    assert task['capabilities'] == composition.ROBOT_CAPABILITIES
    assert task['evaluation']['control_mode'] == ROBOT_CONTROL
    assert len(initial['joint_positions_rad']) == 58
    assert initial['cloths_vertices_m'] == ordinary['initial_state']['cloths_vertices_m']
    assert scene['named_cloth_initialization'] == 'author-only'
    assert set(scene['cameras']) == {'overview', 'task', 'room', 'left_wrist', 'right_wrist'}
    assert {scene['cameras'][side+'_wrist']['mount_link'] for side in ('left', 'right')} == {
        'left_hand_flange', 'right_hand_flange'}
    assert scene['workroom'] == ordinary['scene']['workroom']
    assert scene['cloths'] == ordinary['scene']['cloths']
    assert {key: value for key, value in scene['isaacsim'].items()
            if key not in composition.ROBOT_PHYSICS_OPTIONS} == ordinary['scene']['isaacsim']
    assert scene['laundry_task_provenance']['stand_manifest_sha256'] == STAND_MANIFEST_SHA256
    assert scene['fixtures'][0]['asset_root'] == str(STAND_ROOT)
    assert ordinary['task']['capabilities'] == composition.CAPABILITIES
    assert composition.build_task_inputs('towel_flattening', robot_layout=LAYOUT_ID)['task']['capabilities']['robot_episodes']
    with pytest.raises(ValueError, match='separately pinned robot profile'):
        composition.build_task_inputs('towel_folding', robot_layout=LAYOUT_ID,
                                      physics_options={'robot_self_collisions': False})


def test_derived_stack_accepts_only_its_original_cloth_physics(monkeypatch):
    metadata = assets.candidate_metadata('towel_stacking')
    metadata['settled_fold_reset'] = {'physics_options': deepcopy(composition.PHYSICS_PRIORS)}
    monkeypatch.setattr(composition, 'load_asset_manifest', lambda *_: ({'fixture': True}, deepcopy(metadata)))
    value = composition.build_task_inputs('towel_stacking', asset_manifest='fixture', robot_layout=LAYOUT_ID)
    assert value['scene']['laundry_task_provenance']['cloth_physics_options'] == metadata['settled_fold_reset']['physics_options']
    with pytest.raises(ValueError, match='successful fold physics'):
        composition.build_task_inputs('towel_stacking', asset_manifest='fixture', robot_layout=LAYOUT_ID,
                                      physics_options={'cloth_iterations': 32})


def test_robot_release_freezes_observer_and_rejects_mode_or_profile_tampering(tmp_path, monkeypatch):
    from scripts.maintenance import freeze_project_sources
    from bench.isaacsim.release import task_inputs, validate

    monkeypatch.setattr(assets, 'ASSETS', tmp_path)
    monkeypatch.setattr(release, 'DATASETS', tmp_path)
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot', lambda *a, **k: {'fixture_only': True})
    monkeypatch.setattr(release, 'asset_inventory', lambda scene: {
        path: {'sha256': sha} for path, sha in scene['laundry_task_provenance']['asset_candidate']['files_sha256'].items()})
    asset_root = tmp_path/'assets'
    assets.prepare_assets('towel_folding', asset_root)
    output = tmp_path/'release'
    manifest = release.prepare_task_release('towel_folding', output,
        asset_manifest=asset_root/'manifest.json', robot_layout=LAYOUT_ID)
    assert manifest['task_kind'] == manifest['tasks'][0]['task_kind'] == 'single_agent'
    assert 'evaluator/native_observer.py' in manifest['files']
    assert validate(output) == manifest
    _, scene, task, initial = task_inputs(output, 'towel_folding', seed=19)
    assert task['episode_seed'] == 19
    assert task['evaluation']['control_mode'] == ROBOT_CONTROL
    assert len(initial['joint_positions_rad']) == 58
    for change in ('reset', 'robot_profile', 'cloth_profile', 'layout', 'capability'):
        bad_scene, bad_task = deepcopy(scene), deepcopy(task)
        if change == 'reset':
            bad_task['initial_state']['joint_positions_rad']['fr3_left_joint1'] += .001
        elif change == 'robot_profile':
            bad_scene['isaacsim']['robot_self_collisions'] = False
        elif change == 'cloth_profile':
            bad_scene['isaacsim']['cloth_thickness_m'] = .001
        elif change == 'layout':
            bad_scene['laundry_task_provenance']['robot_layout_id'] = 'unreviewed'
        else:
            bad_task['capabilities'] = deepcopy(composition.CAPABILITIES)
        with pytest.raises(ValueError):
            release.initial_state(bad_scene, bad_task)
    # Hash-bound files alone cannot authorize dropping the native observer.
    bad_manifest = deepcopy(manifest)
    bad_manifest['files'].pop('evaluator/native_observer.py')
    bad_manifest['files_sha256'].pop('evaluator/native_observer.py')
    with pytest.raises(ValueError, match='Incomplete frozen laundry'):
        release.validate_release(output, bad_manifest)


@pytest.mark.parametrize('version', ['v1', 'v3', 'practical_v2'])
def test_existing_fold_release_reconstructs_exactly(version):
    root = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/releases/towel_folding')/version
    scene, task, initial = [load_json(root/name) for name in
                            ('scene.json', 'task.json', 'initial_state.json')]
    assert release.initial_state(scene, task) == initial
    assert task['capabilities'] == composition.CAPABILITIES
