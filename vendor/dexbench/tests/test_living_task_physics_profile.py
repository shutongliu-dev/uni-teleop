"""Scene selection and frozen-input regression checks; no native qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.isaacsim.release import digest, task_inputs
from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.living_room import layer_release
from scenes.workrooms.living_room.composition import AUTHORING_TASK_IDS, build_task_inputs, stand_usda


PREVIOUS_EXTRACTION = Path(
    f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/tower_extraction/v6')
OPTION = 'enable_external_forces_every_iteration'
PROFILE_ID = 'tower_extraction_tgs_external_forces_v2'
TASK4_OPTIONS = {'physics_dt_s': .00025, OPTION: True}


def read_json(path):
    return load_json(path)


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp('living_physics_profiles')
    result = {}
    for task_id in AUTHORING_TASK_IDS:
        if task_id == 'tower_extraction':
            # Reuse the immutable stand identity so the complete scene can be
            # compared with v6 without discarding unrelated fields.
            stand = PREVIOUS_EXTRACTION / 'robot_stands.usda'
        else:
            stand = root / (task_id + '.usda')
            stand.write_text(stand_usda(task_id))
        result[task_id] = build_task_inputs(task_id, stand_path=stand)
    return result


@pytest.mark.parametrize('task_id', AUTHORING_TASK_IDS)
def test_only_task4_selects_half_step_with_external_forces_each_iteration(inputs, task_id):
    scene = inputs[task_id]['scene']
    expected = read_json(PREVIOUS_EXTRACTION / 'physics_profile.json')['isaacsim_options']
    if task_id == 'tower_extraction':
        expected.update(TASK4_OPTIONS)
    assert scene['isaacsim'] == expected
    assert scene['physics']['dt_s'] == inputs[task_id]['task']['evaluation']['physics_dt_s'] == expected['physics_dt_s']
    assert inputs[task_id]['task']['evaluation']['sample_period_s'] == .02
    if task_id != 'tower_extraction':
        assert scene['living_task_provenance']['physics_profile'] == {
            'physics_dt_s': .0005,
            'prior_water_probe_dt_s': 1/960 if task_id == 'water_pouring' else None,
            'qualification_transferred': False,
            'selection': 'Explicit common bench integration profile',
        }


def test_task4_keeps_complete_v6_assets_rules_reset_and_controls(inputs):
    values = inputs['tower_extraction']
    previous_task = read_json(PREVIOUS_EXTRACTION / 'task.json')
    normalized_task = deepcopy(values['task'])
    assert normalized_task['evaluation']['physics_dt_s'] == .00025
    normalized_task['evaluation']['physics_dt_s'] = .0005
    assert normalized_task == previous_task
    assert values['initial_state'] == read_json(PREVIOUS_EXTRACTION / 'initial_state.json')
    previous_scene = read_json(PREVIOUS_EXTRACTION / 'scene.json')
    normalized = deepcopy(values['scene'])
    assert normalized['isaacsim'].pop(OPTION) is True
    assert normalized['isaacsim']['physics_dt_s'] == normalized['physics']['dt_s'] == .00025
    normalized['isaacsim']['physics_dt_s'] = normalized['physics']['dt_s'] = .0005
    profile = normalized['living_task_provenance']['physics_profile']
    assert profile['profile_id'] == PROFILE_ID
    assert profile['physics_dt_s'] == .00025
    assert profile['isaacsim_options'] == TASK4_OPTIONS
    assert profile['integrator']['solver_type'] == 'TGS'
    assert profile['integrator']['control_dt_s'] == .02
    assert profile['integrator']['usd_attribute'] == 'physxScene:enableExternalForcesEveryIteration'
    assert profile['qualification_transferred'] is False
    normalized['living_task_provenance']['physics_profile'] = (
        previous_scene['living_task_provenance']['physics_profile'])
    assert normalized == previous_scene


@pytest.mark.parametrize('version', ['v6', 'v7'])
def test_historical_task4_releases_keep_their_original_integration_profile(version):
    root = PREVIOUS_EXTRACTION.parent / version
    manifest = read_json(root / 'release_manifest.json')
    before = {name: digest(root / name) for name in [*manifest['files'], 'release_manifest.json']}
    _, scene, task, initial = task_inputs(root, 'tower_extraction')
    assert scene == read_json(root / 'scene.json')
    assert task.pop('evaluation_phase') == 'nominal'
    assert task == read_json(root / 'task.json')
    assert initial == read_json(root / 'initial_state.json')
    assert scene['isaacsim']['physics_dt_s'] == scene['physics']['dt_s'] == task['evaluation']['physics_dt_s'] == .0005
    assert {name: digest(root / name) for name in before} == before


@pytest.mark.parametrize('change', [
    'missing_profile', 'malformed_profile', 'missing_id', 'old_id', 'wrong_id',
    'missing_options_dt', 'wrong_options_dt', 'missing_force_flag', 'scene_force_false',
    'provenance_dt', 'evaluation_dt', 'fallback_dt', 'control_period', 'profile_control_period',
    'v2_with_old_dt', 'unsupported_dt',
])
def test_half_step_requires_explicit_consistent_task4_profile(inputs, change):
    scene, task = deepcopy(inputs['tower_extraction']['scene']), deepcopy(inputs['tower_extraction']['task'])
    profile = scene['living_task_provenance']['physics_profile']
    if change == 'missing_profile':
        del scene['living_task_provenance']['physics_profile']
    elif change == 'malformed_profile':
        scene['living_task_provenance']['physics_profile'] = None
    elif change == 'missing_id':
        del profile['profile_id']
    elif change in ('old_id', 'wrong_id'):
        profile['profile_id'] = 'tower_extraction_tgs_external_forces_v1' if change == 'old_id' else 'undeclared'
    elif change == 'missing_options_dt':
        del profile['isaacsim_options']['physics_dt_s']
    elif change == 'wrong_options_dt':
        profile['isaacsim_options']['physics_dt_s'] = .0005
    elif change == 'missing_force_flag':
        del profile['isaacsim_options'][OPTION]
    elif change == 'scene_force_false':
        scene['isaacsim'][OPTION] = False
    elif change == 'provenance_dt':
        profile['physics_dt_s'] = .0005
    elif change == 'evaluation_dt':
        task['evaluation']['physics_dt_s'] = .0005
    elif change == 'fallback_dt':
        scene['physics']['dt_s'] = .0005
    elif change == 'control_period':
        task['evaluation']['sample_period_s'] = .01
    elif change == 'profile_control_period':
        profile['integrator']['control_dt_s'] = .01
    else:
        dt = .0005 if change == 'v2_with_old_dt' else .0002
        scene['isaacsim']['physics_dt_s'] = scene['physics']['dt_s'] = task['evaluation']['physics_dt_s'] = dt
        profile['physics_dt_s'] = profile['isaacsim_options']['physics_dt_s'] = dt
    with pytest.raises(ValueError, match='integration'):
        layer_release.initial_state(scene, task)


@pytest.mark.parametrize('task_id', [task_id for task_id in AUTHORING_TASK_IDS if task_id != 'tower_extraction'])
def test_other_tasks_cannot_select_task4_half_step(inputs, task_id):
    scene, task = deepcopy(inputs[task_id]['scene']), deepcopy(inputs[task_id]['task'])
    scene['living_task_provenance']['physics_profile'] = deepcopy(
        inputs['tower_extraction']['scene']['living_task_provenance']['physics_profile'])
    scene['isaacsim'].update(TASK4_OPTIONS)
    scene['physics']['dt_s'] = task['evaluation']['physics_dt_s'] = .00025
    with pytest.raises(ValueError, match='integration profile'):
        layer_release.initial_state(scene, task)


def test_release_freezes_the_selected_profile_and_its_source(tmp_path, monkeypatch):
    monkeypatch.setattr(layer_release, 'DATASETS', tmp_path)
    root = tmp_path / 'release'
    manifest = layer_release.prepare_task_release('tower_extraction', root)
    scene = read_json(root / 'scene.json')
    frozen = read_json(root / 'physics_profile.json')
    assert frozen['isaacsim_options'][OPTION] is True
    assert frozen['profile_selection'] == scene['living_task_provenance']['physics_profile']
    assert frozen['isaacsim_options']['physics_dt_s'] == .00025
    assert frozen['profile_selection']['isaacsim_options'] == TASK4_OPTIONS
    assert frozen['profile_selection']['provenance']['source_module'] == (
        'scenes.workrooms.living_room.tasks.tower_extraction.physics_profile')
    sources = read_json(root / 'source_provenance.json')['sources_sha256']
    assert any(path.endswith('/tasks/tower_extraction/physics_profile.py') for path in sources)
    assert manifest['asset_catalog'] == read_json(PREVIOUS_EXTRACTION / 'release_manifest.json')['asset_catalog']
    assert manifest['status'] == 'prepared_unvalidated'
    assert manifest['physics_accepted'] is False
    layer_release.validate_release(root, manifest)
    # Even a correctly hashed profile file must agree with the scene's selection.
    frozen['profile_selection']['physics_dt_s'] = .0005
    (root / 'physics_profile.json').write_text(json.dumps(frozen))
    sha = digest(root / 'physics_profile.json')
    manifest['files']['physics_profile.json']['sha256'] = sha
    manifest['files_sha256']['physics_profile.json'] = manifest['physics_profile_sha256'] = sha
    with pytest.raises(ValueError, match='profile mismatch'):
        layer_release.validate_release(root, manifest)
