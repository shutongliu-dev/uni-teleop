from copy import deepcopy

import pytest

from bench.isaacsim.release import digest, dump, load_evaluator, task_inputs, validate
from bench.paths import load_json
from scenes.workrooms.mechanical_assembly import release


@pytest.fixture(params=('parts_sorting', 'peg_insertion', 'shaft_spacer_assembly', 'gear_train_assembly'))
def frozen(tmp_path, monkeypatch, request):
    monkeypatch.setattr(release, 'DATASETS', tmp_path)
    root = tmp_path/'task_release'
    release.prepare_task_release(request.param, root)
    return root, request.param


def test_canonical_worker_loads_original_reset_and_frozen_scorer(frozen):
    root, ident = frozen
    manifest = validate(root)
    entry, scene, task, initial = task_inputs(root, ident)
    assert manifest['physics_accepted'] is False and manifest['status'] == 'prepared_unvalidated'
    assert set(initial) == {'joint_positions_rad'} and len(initial['joint_positions_rad']) == 29
    assert entry['id'] == ident == task['id']
    assert load_evaluator(root)(task).task['id'] == ident
    with pytest.raises(ValueError, match='original task reset'):
        task_inputs(root, ident, phase='qualification_probe')
    with pytest.raises(ValueError, match='exact task identity'):
        task_inputs(root, ident[:2])


def test_missing_asset_cannot_be_hidden_by_rehashing_inventory(frozen):
    root, _ = frozen
    inventory = load_json(root/'asset_file_hashes.json')
    removed = next(p for p in inventory if p.endswith('.obj'))
    del inventory[removed]
    dump(root/'asset_file_hashes.json', inventory)
    manifest = load_json(root/'release_manifest.json')
    h = digest(root/'asset_file_hashes.json')
    manifest['files']['asset_file_hashes.json']['sha256'] = h
    manifest['files_sha256']['asset_file_hashes.json'] = h
    del manifest['asset_files_sha256'][removed]
    dump(root/'release_manifest.json', manifest)
    with pytest.raises(ValueError, match='inventory is incomplete'):
        validate(root)


def test_diagnostic_reset_is_not_a_task_release(frozen):
    root, ident = frozen
    _, scene, task, _ = task_inputs(root, ident)
    scene = deepcopy(scene)
    scene['mechanical_task_provenance']['engineering_probe'] = {'reset_only': True}
    with pytest.raises(ValueError, match='diagnostic'):
        release.initial_state(scene, task)


def test_evaluator_reset_cannot_diverge_from_actual_objects(frozen):
    root, ident = frozen
    _, scene, task, _ = task_inputs(root, ident)
    scene['objects'][0]['position_m'][0] += .001
    with pytest.raises(ValueError, match='evaluator reset differs'):
        release.initial_state(scene, task)


def test_frozen_scoring_source_tampering_is_rejected(frozen):
    root, _ = frozen
    with (root/'evaluator/rules.py').open('a') as stream:
        stream.write('\n# altered\n')
    with pytest.raises(ValueError, match='Frozen input changed'):
        load_evaluator(root)
