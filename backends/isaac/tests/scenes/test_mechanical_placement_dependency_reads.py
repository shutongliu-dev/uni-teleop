"""CPU fixtures: cheap actor lookup follows complete, uncached file checks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.paths import load_json
from scenes.workrooms.initial_randomization import physical_inputs
from scenes.workrooms.mechanical_assembly import placement_variants as variants
from scenes.workrooms.seed_geometry import file_hash, write_json


@pytest.fixture
def composed_inputs(tmp_path):
    # Small synthetic assets exercise real file reads, not a native qualification.
    assets = tmp_path / 'assets'
    assets.mkdir()
    files = {}
    for name in ('combined.obj', 'visual.obj', 'collision.obj'):
        path = assets / name
        path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
        files[name] = path
    asset = {'mesh_path': 'combined.obj', 'sha256': file_hash(files['combined.obj']),
             'visual_parts': [{'mesh_path': 'visual.obj', 'sha256': file_hash(files['visual.obj'])}],
             'collision_parts': [{'mesh_path': 'collision.obj', 'sha256': file_hash(files['collision.obj'])}]}
    catalog = assets / 'manifest.json'
    write_json(catalog, {'assets': {'gear': asset}})
    files['catalog'] = catalog
    external = {str(path): file_hash(path) for path in files.values()}

    parent = tmp_path / 'parent'
    (parent / 'evaluator').mkdir(parents=True)
    scene = {'objects': [{'id': 'gear', 'asset_id': 'gear', 'asset_root': str(assets),
                          'position_m': [.5, .1, .8]}], 'fixtures': []}
    task = {'id': 'gear_train_assembly', 'evaluation': {}}
    initial = {'joints': [0.]}
    for name, value in (('scene', scene), ('task', task), ('initial_state', initial)):
        write_json(parent / (name + '.json'), value)
    (parent / 'evaluator' / 'rules.py').write_text('# Synthetic frozen evaluator.\n')
    release_files = {str(p.relative_to(parent)): file_hash(p) for p in parent.rglob('*') if p.is_file()}
    write_json(parent / 'release_manifest.json', {'files_sha256': release_files,
                                                 'asset_files_sha256': external})
    references = {str(parent / name): sha for name, sha in release_files.items()}
    references.update(external)
    references[str(parent / 'release_manifest.json')] = file_hash(parent / 'release_manifest.json')

    source = tmp_path / 'fixture_source'
    names = ('scenes/workrooms/mechanical_assembly/placement_variants.py',
             'scenes/workrooms/mechanical_assembly/placement_authoring.py',
             'scenes/workrooms/seed_geometry.py',
             'bench/simulation/isaacsim_backend/asset_catalog.py',
             'bench/simulation/isaacsim_backend/presentation.py')
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Explicit synthetic source provenance fixture.\n')
    snapshot = source / 'source_snapshot_manifest.json'
    source_hashes = {name: file_hash(source / name) for name in names}
    write_json(snapshot, {'schema': 'complete_benchmark_source_snapshot_v1',
                          'snapshot_root': str(source), 'files_sha256': source_hashes})
    proof = {'manifest_path': str(snapshot), 'manifest_sha256': file_hash(snapshot),
             'snapshot_root': str(source),
             'source_files_sha256': {str(source / name): sha for name, sha in source_hashes.items()}}
    plan = variants.plan_case(scene, task, 5)
    data = {'plan': plan, 'presentations': {}, 'appearance': [],
            'actual_body_poses_m': {'gear': {'prim_path': '/World/Objects/gear',
                'position_m': variants._translate(scene['objects'][0]['position_m'], plan['translation_xy_m']),
                'quaternion_wxyz': [1., 0., 0., 0.]}},
            'interaction_targets': [], 'fixed_support_paths': [],
            'static_geometry': {'bodies': [{'object_id': 'gear'}]},
            'protected_physics_before_sha256': 'same', 'protected_physics_after_sha256': 'same'}
    metadata = tmp_path / 'metadata.json'
    write_json(metadata, data)
    manifest = {'schema': 'workroom_seed_asset.v1', 'generator_version': variants.VERSION,
                'family': 'mechanical_assembly', 'task_id': task['id'], 'seed': 5,
                'variant_scope': variants.VARIANT_SCOPE, 'parent_references': references,
                'dependencies_sha256': {str(metadata): file_hash(metadata)}, 'author_source': proof,
                'metadata': str(metadata), 'metadata_sha256': file_hash(metadata),
                **{key: deepcopy(data[key]) for key in ('actual_body_poses_m', 'interaction_targets',
                                                       'fixed_support_paths', 'static_geometry')}}
    path = tmp_path / 'seed_manifest.json'
    write_json(path, manifest)
    return parent, path, files


def test_compose_checks_asset_bytes_once_before_metadata_actor_lookup(composed_inputs, monkeypatch):
    parent, manifest, files = composed_inputs
    original = variants.file_hash
    checked = []

    def record(path):
        checked.append(str(path))
        return original(path)

    monkeypatch.setattr(variants, 'file_hash', record)
    def forbidden(*args, **kwargs):
        pytest.fail('Compose actor lookup must not reopen all asset geometry')
    monkeypatch.setattr(variants, 'catalog_body_path', forbidden)
    expected_lookup = physical_inputs.object_actor_path

    def lookup(spec):
        assert all(checked.count(str(path)) == 1 for path in files.values())
        return expected_lookup(spec)

    monkeypatch.setattr(physical_inputs, 'object_actor_path', lookup)
    result = variants.compose_case(parent, manifest, 'single_arm')
    assert result['actual_body_poses_m']['gear']['prim_path'] == '/World/Objects/gear'
    assert result['scene']['objects'][0]['position_m'] == [.49, .1, .8]
    # A second compose still rechecks actual bytes: this is not a cache.
    checked.clear()
    assert variants.compose_case(parent, manifest, 'single_arm') == result


@pytest.mark.parametrize('changed', ['catalog', 'combined.obj', 'visual.obj', 'collision.obj'])
def test_modified_actual_catalog_or_geometry_rejected_before_lookup(composed_inputs, monkeypatch, changed):
    parent, manifest, files = composed_inputs
    variants.compose_case(parent, manifest, 'single_arm')
    files[changed].write_bytes(files[changed].read_bytes() + b'\nchanged')
    def forbidden(*args, **kwargs):
        pytest.fail('A changed dependency must fail before actor-path lookup')
    monkeypatch.setattr(physical_inputs, 'object_actor_path', forbidden)
    with pytest.raises(ValueError, match='Changed frozen input'):
        variants.compose_case(parent, manifest, 'single_arm')


def test_independent_catalog_helper_still_hashes_actual_geometry(composed_inputs):
    parent, _, files = composed_inputs
    spec = load_json(parent / 'scene.json')['objects'][0]
    assert variants.catalog_body_path(spec) == '/World/Objects/gear'
    files['collision.obj'].write_text('changed real collision bytes')
    with pytest.raises(ValueError, match='hash mismatch'):
        variants.catalog_body_path(spec)
