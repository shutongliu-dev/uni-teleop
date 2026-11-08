"""A cover successor must not change the selected physical/task instance."""
from copy import deepcopy
import json

import pytest

from bench.paths import load_json
from scenes.workrooms.parameterized import study_appearance_successor as successor
from scenes.workrooms.seed_geometry import file_hash


@pytest.fixture
def frozen_successor(tmp_path):
    def write(name, value):
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); return str(path)
    files = {}
    for name in successor.SOURCE_FILES:
        path = write('source/'+name, {'code': name}); files[name] = file_hash(path)
    write('source/source_snapshot_manifest.json', {'files_sha256': files})
    source = successor._source_identity(tmp_path/'source')
    image = write('new.png', {'pixels': 'new'}); old = write('old.usda', {'old': True})
    new = write('new.usda', {'new': True})
    layer = {'id': 'task', 'usd_path': old, 'sha256': file_hash(old),
             'dependencies_sha256': {}, 'rigid_bodies': {'book': '/Book'}}
    data = {'scene_changes': [{'path': ['task_layers'], 'before': ['canonical'], 'after': [layer]}],
            'task_changes': [{'path': ['mass'], 'before': 1., 'after': 1.2}],
            'actual_body_poses_m': {'book': [1, 2, 3]}}
    metadata = write('old_metadata.json', data)
    prior = {'family': 'study', 'task_id': 'book_stacking', 'seed': 9,
        'parameter_instance': {'canonical': False, 'instance_sha256': 'original-instance',
            'appearance': {'pattern_id': 'stripes', 'pattern_scale_m': .06}},
        'applied_parameters': {'layout': [1, 2, 3], 'body_physics': {'mass': 1.2},
            'appearance': {'meshes': ['paper'], 'artifacts': []}},
        'metadata': metadata, 'metadata_sha256': file_hash(metadata),
        'dependencies_sha256': {old: file_hash(old), metadata: file_hash(metadata)},
        'author_source': {'original': True}, 'source_snapshot': {'original': True},
        'canonical_asset': {'original': True}, 'native_qualified': False,
        'physical_instance_sha256': 'old-identity'}
    prior_path = write('prior.json', prior)
    recipe = {'schema': 'study_book_cover_background_v1', 'canonical': False,
        'instance_sha256': 'original-instance', 'pattern_id': 'stripes',
        'sampled_pattern_scale_m': .06, 'pattern_scale_m_applied': False,
        'artifacts': [{'path': image, 'sha256': file_hash(image)}]}
    new_layer = {**layer, 'usd_path': new, 'sha256': file_hash(new),
                 'dependencies_sha256': {image: file_hash(image)}}
    new_data = successor._metadata_successor(data, layer, new_layer)
    new_metadata = write('new_metadata.json', new_data)
    result = deepcopy(prior)
    result['applied_parameters']['appearance'][successor.RECIPE_KEY] = recipe
    result.update(metadata=new_metadata, metadata_sha256=file_hash(new_metadata),
        appearance_author_source=source, physical_instance_sha256='new-identity',
        appearance_successor={'schema': successor.VERSION,
            'previous_asset': {'path': prior_path, 'sha256': file_hash(prior_path)},
            'previous_task_layer': layer, 'task_layer': new_layer})
    result['dependencies_sha256'].update({prior_path: file_hash(prior_path),
        **source['source_files_sha256'], source['manifest_path']: source['manifest_sha256'],
        image: file_hash(image), new: file_hash(new), new_metadata: file_hash(new_metadata)})
    return result, new_data


def test_valid_successor_retains_old_author_and_has_no_usd_or_pixel_work(frozen_successor, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('PIL', 'pxr'):
            raise AssertionError('Runtime cover binding must remain JSON-only')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    successor.validate_successor(*frozen_successor)


@pytest.mark.parametrize('field', ['author_source', 'source_snapshot', 'canonical_asset', 'parameter_instance'])
def test_original_authorship_and_sampling_cannot_be_relabelled(frozen_successor, field):
    manifest, data = frozen_successor
    manifest[field]['changed'] = True
    with pytest.raises(ValueError, match='immutable predecessor'):
        successor.validate_successor(manifest, data)


@pytest.mark.parametrize('field', ['layout', 'body_physics'])
def test_appearance_recipe_cannot_hide_physical_change(frozen_successor, field):
    manifest, data = frozen_successor; manifest['applied_parameters'][field] = 'changed'
    with pytest.raises(ValueError, match='prior physical'):
        successor.validate_successor(manifest, data)


@pytest.mark.parametrize('field', ['task_changes', 'actual_body_poses_m'])
def test_task_and_projection_drift_rejected(frozen_successor, field):
    manifest, data = frozen_successor; data[field] = ['changed']
    with pytest.raises(ValueError, match='physical metadata'):
        successor.validate_successor(manifest, data)


def test_complete_dependency_inventory_required(frozen_successor):
    manifest, data = frozen_successor
    recipe = manifest['applied_parameters']['appearance'][successor.RECIPE_KEY]
    del manifest['dependencies_sha256'][recipe['artifacts'][0]['path']]
    with pytest.raises(ValueError, match='dependency inventory'):
        successor.validate_successor(manifest, data)


def test_recipe_cannot_replace_a_prior_global_pin(frozen_successor):
    manifest, data = frozen_successor
    marker = manifest['appearance_successor']
    old_path = marker['previous_task_layer']['usd_path']
    recipe = manifest['applied_parameters']['appearance'][successor.RECIPE_KEY]
    recipe['artifacts'][0].update(source_path=old_path, source_sha256='0'*64)
    manifest['dependencies_sha256'][old_path] = '0'*64
    with pytest.raises(ValueError, match='replace a frozen dependency'):
        successor.validate_successor(manifest, data)


def test_changed_predecessor_metadata_rejected(frozen_successor):
    from pathlib import Path
    manifest, data = frozen_successor
    prior = load_json(Path(manifest['appearance_successor']['previous_asset']['path']))
    Path(prior['metadata']).write_text('{}')
    with pytest.raises(ValueError, match='predecessor metadata changed'):
        successor.validate_successor(manifest, data)


def test_wrong_author_source_and_metric_claim_rejected(frozen_successor):
    manifest, data = frozen_successor; original = deepcopy(manifest)
    manifest['appearance_author_source']['manifest_sha256'] = '0'*64
    with pytest.raises(ValueError, match='source identity'):
        successor.validate_successor(manifest, data)
    original['applied_parameters']['appearance'][successor.RECIPE_KEY]['pattern_scale_m_applied'] = True
    with pytest.raises(ValueError, match='retained parameter'):
        successor.validate_successor(original, data)


def test_legacy_noop_and_orphan_recipe_rejected():
    successor.validate_successor({'applied_parameters': {'appearance': {}}}, {})
    with pytest.raises(ValueError, match='Orphan'):
        successor.validate_successor({'appearance_author_source': {}}, {})


@pytest.mark.parametrize('mutation', ['mass', 'units'])
def test_stage_gate_rejects_hidden_physics_change(tmp_path, mutation):
    from pxr import Usd, UsdGeom, UsdPhysics
    from scenes.workrooms.study.book_assets import author_book
    from scenes.workrooms.study.book_cover_appearance import apply_book_cover_background
    from scenes.workrooms.parameterized.spec import sample_parameters
    stage = Usd.Stage.CreateInMemory(); author_book(stage, '/Book', 'robotics')
    stage = Usd.Stage.Open(stage.Flatten())
    before = successor.stage_properties(stage)
    recipe = apply_book_cover_background(stage, {'book': '/Book'},
        sample_parameters('study', 'book_stacking', 'single_arm', 9), tmp_path/'cover')
    successor.check_only_cover_changes(before, successor.stage_properties(stage), recipe)
    if mutation == 'mass':
        UsdPhysics.MassAPI(stage.GetPrimAtPath('/Book')).GetMassAttr().Set(123.)
    else:
        UsdGeom.SetStageMetersPerUnit(stage, .01)
    with pytest.raises(ValueError, match='nonappearance'):
        successor.check_only_cover_changes(before, successor.stage_properties(stage), recipe)
