"""Synthetic evidence fixtures only; these tests do not qualify native cloth."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from bench.paths import load_json, portable
from scenes.workrooms.laundry_room import assets, composition, evaluation, settled_reset
from bench.isaacsim import release as loader
from scripts.maintenance import freeze_project_sources


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, 'ASSETS', tmp_path)
    monkeypatch.setattr(assets, 'TEXTURE_HASHES', {})
    # Small provenance fixtures keep these CPU tests independent of downloaded
    # production textures; the renderer tests separately decode actual pixels.
    dependencies = {}
    for name in ('albedo.jpg', 'API_INFO.json', 'SOURCE.json'):
        path = tmp_path/name
        path.write_text('synthetic appearance provenance: '+name)
        dependencies[str(path)] = assets.digest(path)
    monkeypatch.setattr(assets, 'DERIVED_TEXTURE_HASHES', dependencies)
    monkeypatch.setattr(assets, 'DERIVED_TEXTURES', {**assets.TEXTURES, 'albedo': str(tmp_path/'albedo.jpg')})
    monkeypatch.setattr(loader, 'validate', lambda root: {'fixture_only': True})
    monkeypatch.setattr(loader, 'load_evaluator', lambda root: evaluation.SequenceEvaluator)
    monkeypatch.setattr(freeze_project_sources, 'verify', lambda root: {'verified': True, 'fixture_only': True})
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot', lambda *a, **k: {'verified': True, 'fixture_only': True})
    inputs = composition.build_task_inputs('towel_folding', physics_options={'physics_dt_s': .002}, rule_profile='laundry_strict_task.v1')
    task, scene = inputs['task'], inputs['scene']
    root = tmp_path/'source_release'; root.mkdir()
    write = lambda path, value: path.write_text(json.dumps(value))
    for name, value in (('task.json', task), ('scene.json', scene),
                        ('physics_profile.json', {'isaacsim_options': scene['isaacsim']})):
        write(root/name, value)
    write(root/'release_manifest.json', {'files_sha256': {'physics_profile.json': assets.digest(root/'physics_profile.json')}})
    source = tmp_path/'source'; source.mkdir()
    write(source/'source_snapshot_manifest.json', {'fixture_only': True})
    q = assets.folded_vertices(np.asarray(task['asset_metadata']['cloths'][0]['rest_vertices_m']), [.65, -.55])
    count = 260
    archive = tmp_path/'native_states.npz'
    np.savez_compressed(archive, towel_q=np.tile(q, (count, 1, 1)),
        towel_qd=np.zeros((count, len(q), 3)), towel_control_mask=np.zeros((count, len(q)), dtype=bool),
        towel_triangles=np.asarray(task['asset_metadata']['cloths'][0]['triangles'], dtype=np.int32),
        native_step=np.arange(count), time_s=np.arange(count)*.002)
    bind = lambda path: {'path': str(path), 'sha256': assets.digest(path)}
    native_path = tmp_path/'native_report.json'
    native = {'fixture_only': True, 'audit_contract': 'laundry_native_stream.v1', 'task_id': 'towel_folding',
        'complete': True, 'native_steps': count-1, 'state_archive': bind(archive),
        'release': bind(root/'release_manifest.json'), 'scene_input': bind(root/'scene.json'),
        'task_input': bind(root/'task.json'), 'source_snapshot_manifest': bind(source/'source_snapshot_manifest.json')}
    write(native_path, native)
    trace = tmp_path/'frame_audit.jsonl'; trace.write_text('CPU synthetic fixture only\n')
    audit_path = tmp_path/'audit_report.json'
    audit = {'fixture_only': True, 'schema': 'laundry_complete_stream_audit.v1', 'task_id': 'towel_folding',
        'status': 'passed', 'passed': True, 'accepted_engineering_task_success': True,
        'all_native_samples_checked': True, 'physical_geometry_stream_passed': True,
        'first_physical_failure': None, 'samples_checked': count,
        'terminal_window_required': True,
        'terminal_window': dict(passed=True, complete=True, complete_release=True, stable=True, geometry_passed=True),
        'audit_device': 'cuda:0',
        'audit_source_manifest': bind(source/'source_snapshot_manifest.json'), 'frame_audit': bind(trace),
        'inputs_sha256': {str(path): assets.digest(path) for path in (native_path, archive,
            root/'release_manifest.json', root/'scene.json', root/'task.json', root/'physics_profile.json')}}
    write(audit_path, audit)
    return native_path, audit_path, scene['isaacsim'], q


def test_default_v1_metadata_remains_byte_identical():
    expected = {'towel_flattening': '515e0f42a6e29d72b8e503a83d12417a7cf0e61b8b2891997e626d22be37b775',
        'towel_folding': '2abac1ac61723337aaa82cc6800f855094476ad4dd0e9fb34f4e919de51dc680',
        'towel_stacking': 'd11d8da4ef42864f87574ea31c3277ca910c528f11dec1815d5106b6c1e65967'}
    for task_id, sha in expected.items():
        # Hashed in portable form, so the values do not depend on where the data package lives.
        data = json.dumps(portable(assets.candidate_metadata(task_id)), sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(data).hexdigest() == sha
        for row in composition.build_task_inputs(task_id)['scene']['cloths']:
            assert 'visual_texture_tile_size_m' not in row
            assert 'albedo' not in row['visual_texture_paths']


def test_derived_asset_copies_only_small_terminal_geometry_and_binds_profile(tmp_path, monkeypatch):
    native, audit, profile, q = fixture(tmp_path, monkeypatch)
    output = tmp_path/'derived'
    assets.prepare_assets('towel_stacking', output, source_fold_native_report=native,
        source_fold_audit=audit, physics_options=profile)
    manifest, metadata = assets.load_asset_manifest('towel_stacking', output/'manifest.json')
    assert metadata['schema'] == 'laundry_towel_asset.v2'
    assert {p.name for p in output.iterdir()} == {'metadata.json', 'manifest.json', 'ATTRIBUTION.txt'}
    for row, target in zip(metadata['cloths'], ([.30, -.55], [.95, -.55])):
        points = np.asarray(row['initial_vertices_m'])
        assert np.allclose((points.min(axis=0)[:2]+points.max(axis=0)[:2])/2, target)
        assert np.array_equal(points[:, 2], q[:, 2])
        assert np.allclose(points-points[0], q-q[0])
    assert str(native) in manifest['files_sha256'] and str(audit) in manifest['files_sha256']
    assert not metadata['settled_fold_reset']['reset_task_credit']
    value = composition.build_task_inputs('towel_stacking', asset_manifest=output/'manifest.json', physics_options=profile)
    assert len(value['scene']['cloths']) == 2
    for row, spec in zip(metadata['cloths'], value['scene']['cloths']):
        assert row['visual_texture_paths'] == assets.DERIVED_TEXTURES
        assert spec['visual_texture_paths'] == row['visual_texture_paths']
        assert spec['visual_texture_tile_size_m'] == [.367053595, .363200009]
    for path, sha in assets.DERIVED_TEXTURE_HASHES.items():
        assert manifest['files_sha256'][path] == sha
    with pytest.raises(ValueError, match='successful fold physics profile'):
        composition.build_task_inputs('towel_stacking', asset_manifest=output/'manifest.json',
            physics_options={**profile, 'cloth_youngs_modulus_pa': 123.})
    changed = deepcopy(metadata); changed['cloths'][0]['initial_vertices_m'][0][2] += .01
    with pytest.raises(ValueError, match='differs from successful native'):
        assets.validate_metadata(changed)
    changed = deepcopy(metadata); changed['cloths'][0]['visual_texture_tile_size_m'] = [.20, .20]
    with pytest.raises(ValueError, match='differs from successful native'):
        assets.validate_metadata(changed)
    # The same declared appearance must also remain bound at the asset boundary.
    manifest_path = output/'manifest.json'
    changed = load_json(manifest_path)
    changed['dependencies_sha256'].pop(str(tmp_path/'API_INFO.json'))
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='texture provenance or dependencies differ'):
        assets.load_asset_manifest('towel_stacking', manifest_path)


@pytest.mark.parametrize('field,value', [('passed', False), ('all_native_samples_checked', False),
    ('physical_geometry_stream_passed', False), ('samples_checked', 2), ('task_id', 'towel_flattening')])
def test_failed_partial_or_wrong_task_audit_is_rejected(tmp_path, monkeypatch, field, value):
    native, audit, profile, _ = fixture(tmp_path, monkeypatch)
    data = load_json(audit); data[field] = value; audit.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='passed complete-stream'):
        settled_reset.metadata_from_native_fold(native, audit, physics_options=profile)


def test_profile_mesh_and_source_evidence_changes_are_rejected(tmp_path, monkeypatch):
    native, audit, profile, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='physics profile differs'):
        settled_reset.metadata_from_native_fold(native, audit, physics_options={**profile, 'cloth_thickness_m': .001})
    with pytest.raises(ValueError, match='physical/rest property differs'):
        settled_reset.metadata_from_native_fold(native, audit, cells_xy=(40, 20))
    data = load_json(native); data['extra'] = 'changed'; native.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Audit does not bind'):
        settled_reset.metadata_from_native_fold(native, audit)


def test_derivation_requires_explicit_both_evidence_paths_and_stack_only(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, 'ASSETS', tmp_path)
    for name, report, audit in (('towel_stacking', 'missing', None), ('towel_folding', 'missing', 'missing')):
        with pytest.raises(ValueError, match='Only stacking'):
            assets.prepare_assets(name, tmp_path/'unused', source_fold_native_report=report, source_fold_audit=audit)
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('field', ('passed', 'complete', 'complete_release', 'stable', 'geometry_passed'))
def test_derived_reset_requires_independent_terminal_audit(tmp_path, monkeypatch, field):
    native, audit, _, _ = fixture(tmp_path, monkeypatch)
    data = load_json(audit); data['terminal_window'][field] = False
    audit.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='actual terminal window'):
        settled_reset.metadata_from_native_fold(native, audit)


@pytest.mark.parametrize('mutation, message', [('velocity', 'not released and stable'),
    ('control', 'not complete and released'), ('geometry', 'frozen geometry rules'),
    ('topology', 'Native topology differs')])
def test_terminal_state_is_rechecked_even_if_prior_task_score_passed(tmp_path, monkeypatch, mutation, message):
    native_path, audit_path, _, _ = fixture(tmp_path, monkeypatch)
    native = load_json(native_path)
    archive = native['state_archive']['path']
    with np.load(archive, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if mutation == 'velocity':
        arrays['towel_qd'][-1, 0, 0] = .04
    elif mutation == 'control':
        arrays['towel_control_mask'][-1, 0] = True
    elif mutation == 'geometry':
        arrays['towel_q'][:, :, 0] += .10
    else:
        arrays['towel_triangles'][0, 0] += 1
    np.savez_compressed(archive, **arrays)
    native['state_archive']['sha256'] = assets.digest(archive)
    native_path.write_text(json.dumps(native))
    audit = load_json(audit_path)
    audit['inputs_sha256'][archive] = assets.digest(archive)
    audit['inputs_sha256'][str(native_path)] = assets.digest(native_path)
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match=message):
        settled_reset.metadata_from_native_fold(native_path, audit_path)
