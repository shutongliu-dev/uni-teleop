"""Content changes must never inherit a previous semantic validation."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from scenes.workrooms.laundry_room.randomization import common, release, validation_cache as cache


@pytest.fixture
def graph(tmp_path, monkeypatch):
    """Small pinned graph tests cache boundaries; semantic callback is explicit."""
    from scenes.workrooms.laundry_room import robot_configuration_catalog
    cache.clear()
    snapshot = tmp_path/'source'; hashes = {}
    for name in common.SOURCE_FILES:
        path = snapshot/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture author\n'); hashes[name] = common.file_hash(path)
    common.write_json(snapshot/'source_snapshot_manifest.json', {
        'schema': 'complete_benchmark_source_snapshot_v1', 'files_sha256': hashes})
    provenance = {'source_snapshot': {'snapshot_root': str(snapshot), 'runtime_imports_verified': True},
        'source_manifest_sha256': common.file_hash(snapshot/'source_snapshot_manifest.json'),
        'author_sources_sha256': hashes}
    parent = tmp_path/'completion_inputs_v1'; parent.mkdir()
    common.write_json(parent/'task.json', {'original': True})
    common.write_json(parent/'release_manifest.json', {'files_sha256': {'task.json': common.file_hash(parent/'task.json')}})
    parent_pin = {'path': str(parent), 'manifest_sha256': common.file_hash(parent/'release_manifest.json')}
    monkeypatch.setattr(robot_configuration_catalog, 'RELEASES', {
        'towel_folding': {'configurations': {'single_arm': {'source_release': parent_pin}}}})
    root = tmp_path/'release'; root.mkdir()
    shared = tmp_path/'shared.usda'; shared.write_text('immutable physical input\n')
    dependencies = {str(shared): common.file_hash(shared)}
    cases = {}
    for seed in range(10):
        folder = tmp_path/'assets'/str(seed); folder.mkdir(parents=True)
        common.write_json(folder/'plan.json', {'seed': seed})
        common.write_json(folder/'initial_vertices.json', {'values': [seed]})
        profile_dir = tmp_path/'profiles'/str(seed); profile_dir.mkdir(parents=True)
        common.write_json(profile_dir/'profile.json', {'seed': seed})
        common.write_json(profile_dir/'SOURCE.json', provenance)
        pin = {'path': str(profile_dir/'profile.json'), 'sha256': common.file_hash(profile_dir/'profile.json')}
        common.write_json(folder/'asset_manifest.json', {'source_provenance': provenance,
            'files_sha256': {n: common.file_hash(folder/n) for n in ('plan.json', 'initial_vertices.json')},
            'visual_profiles': {'towel': pin}})
        cases[str(seed)] = {'asset_manifest': str(folder/'asset_manifest.json'),
                           'asset_manifest_sha256': common.file_hash(folder/'asset_manifest.json')}
        dependencies.update({str(p): common.file_hash(p) for p in (*folder.iterdir(), *profile_dir.iterdir())})
    common.write_json(root/'task.json', {'safe': True})
    common.write_json(root/'source_provenance.json', provenance)
    manifest = {'schema': 'laundry_randomized_release.v1', 'task_id': 'towel_folding',
        'configuration_id': 'single_arm', 'physics_accepted': False, 'task_policy_success_claimed': False,
        'base_release': parent_pin, 'cases': cases,
        'files_sha256': {n: common.file_hash(root/n) for n in ('task.json', 'source_provenance.json')},
        'asset_files_sha256': dependencies}
    common.write_json(root/'release_manifest.json', manifest)
    calls = []
    def full(path, value, *, assets=True):
        calls.append(assets)
        if common.read_json(path/'task.json') != {'safe': True}:
            raise ValueError('Fixture semantic mismatch')
    monkeypatch.setattr(release, '_validate_uncached', full)
    yield {'root': root, 'manifest': manifest, 'parent': parent, 'source': snapshot,
           'shared': shared, 'calls': calls, 'full': full}
    cache.clear()


def validate(graph, **kwargs):
    return release.validate_release(graph['root'], common.read_json(graph['root']/'release_manifest.json'), **kwargs)


def test_warm_reuses_only_semantics_and_returns_independent_values(graph):
    result = validate(graph)
    result['cases']['0']['asset_manifest_sha256'] = 'f'*64
    assert validate(graph) == graph['manifest'] and graph['calls'] == [True]


def test_false_does_not_authorize_true_or_skip_shared_file_hash(graph):
    validate(graph, assets=False); validate(graph, assets=True)
    validate(graph, assets=False); validate(graph, assets=True)
    assert graph['calls'] == [False, True]
    graph['shared'].write_text('changed')
    with pytest.raises(ValueError, match='Frozen input differs'):
        validate(graph, assets=False)


@pytest.mark.parametrize('field', ['physics_accepted', 'task_policy_success_claimed'])
def test_in_memory_bool_to_number_change_cannot_hit_cache(graph, field):
    validate(graph)
    value = deepcopy(graph['manifest']); value[field] = 0
    with pytest.raises(ValueError, match='manifest argument'):
        release.validate_release(graph['root'], value)
    assert graph['calls'] == [True]


@pytest.mark.parametrize('target', ['root', 'parent', 'asset', 'shared', 'snapshot_manifest', 'author_source', 'profile_source'])
def test_warm_detects_every_nested_byte_change(graph, target):
    validate(graph)
    row = graph['manifest']['cases']['1']; asset = Path(row['asset_manifest'])
    paths = {'root': graph['root']/'task.json', 'parent': graph['parent']/'task.json',
        'asset': asset.parent/'plan.json', 'shared': graph['shared'],
        'snapshot_manifest': graph['source']/'source_snapshot_manifest.json',
        'author_source': graph['source']/common.SOURCE_FILES[2],
        'profile_source': graph['root'].parent/'profiles/1/SOURCE.json'}
    paths[target].write_bytes(paths[target].read_bytes()+b'\n')
    with pytest.raises(ValueError, match='Frozen input differs'):
        validate(graph)
    assert graph['calls'] == [True]


@pytest.mark.parametrize('target', ['root', 'parent', 'asset'])
def test_warm_detects_added_directory_entries(graph, target):
    validate(graph)
    directories = {'root': graph['root'], 'parent': graph['parent'],
        'asset': Path(graph['manifest']['cases']['0']['asset_manifest']).parent}
    (directories[target]/'undeclared.json').write_text('{}')
    with pytest.raises(ValueError, match='directory inventory'):
        validate(graph)


def test_same_bytes_symlink_cannot_replace_a_protected_local_file(graph, tmp_path):
    validate(graph)
    path = graph['root']/'task.json'; copy = tmp_path/'outside.json'
    copy.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(copy)
    with pytest.raises(ValueError, match='logical path|escapes'):
        validate(graph)


def test_retargeted_same_byte_shared_symlink_cannot_inherit_path_identity(graph, tmp_path):
    target = tmp_path/'target.usda'; target.write_bytes(graph['shared'].read_bytes())
    graph['shared'].unlink(); graph['shared'].symlink_to(target)
    validate(graph)
    alternate = tmp_path/'alternate.usda'; alternate.write_bytes(target.read_bytes())
    graph['shared'].unlink(); graph['shared'].symlink_to(alternate)
    with pytest.raises(ValueError, match='logical path'):
        validate(graph)


def test_overflow_json_number_is_rejected_even_when_python_parses_infinity(graph):
    validate(graph)
    path = graph['root']/'release_manifest.json'
    value = path.read_text().rstrip()[:-1]+', "overflow": 1e999}\n'
    path.write_text(value)
    with pytest.raises(ValueError, match='Nonfinite JSON'):
        release.validate_release(graph['root'], json.loads(value))


@pytest.mark.parametrize('target', ['parent', 'asset', 'source'])
def test_rehashing_nested_local_pins_cannot_inherit_old_receipt(graph, target):
    validate(graph)
    manifest = deepcopy(graph['manifest'])
    if target == 'parent':
        path = graph['parent']/'task.json'; common.write_json(path, {'original': False})
        parent_manifest = common.read_json(graph['parent']/'release_manifest.json')
        parent_manifest['files_sha256']['task.json'] = common.file_hash(path)
        common.write_json(graph['parent']/'release_manifest.json', parent_manifest)
        manifest['base_release']['manifest_sha256'] = common.file_hash(graph['parent']/'release_manifest.json')
    elif target == 'asset':
        path = Path(manifest['cases']['1']['asset_manifest'])
        plan = path.parent/'plan.json'; common.write_json(plan, {'seed': 999})
        asset = common.read_json(path); asset['files_sha256']['plan.json'] = common.file_hash(plan)
        common.write_json(path, asset)
        manifest['cases']['1']['asset_manifest_sha256'] = common.file_hash(path)
        manifest['asset_files_sha256'][str(path)] = common.file_hash(path)
        # Other unchanged declaration still binds the original plan bytes.
    else:
        path = graph['source']/common.SOURCE_FILES[0]; path.write_text('# changed author\n')
        snapshot = graph['source']/'source_snapshot_manifest.json'; source = common.read_json(snapshot)
        source['files_sha256'][common.SOURCE_FILES[0]] = common.file_hash(path)
        common.write_json(snapshot, source)
        root_source = graph['root']/'source_provenance.json'; provenance = common.read_json(root_source)
        provenance['source_manifest_sha256'] = common.file_hash(snapshot)
        provenance['author_sources_sha256'][common.SOURCE_FILES[0]] = common.file_hash(path)
        common.write_json(root_source, provenance)
        manifest['files_sha256']['source_provenance.json'] = common.file_hash(root_source)
    common.write_json(graph['root']/'release_manifest.json', manifest)
    with pytest.raises(ValueError):
        validate(graph)
    assert graph['calls'] == [True]


def test_self_consistent_rehash_misses_cache_and_runs_semantics_again(graph):
    validate(graph)
    path = graph['root']/'task.json'; common.write_json(path, {'safe': False})
    manifest = deepcopy(graph['manifest']); manifest['files_sha256']['task.json'] = common.file_hash(path)
    common.write_json(graph['root']/'release_manifest.json', manifest)
    with pytest.raises(ValueError, match='semantic mismatch'):
        validate(graph)
    assert graph['calls'] == [True, True]


def test_cold_mid_check_change_cannot_create_a_successful_receipt(graph, monkeypatch):
    before = graph['shared'].read_bytes()
    def change(path, value, *, assets=True):
        graph['full'](path, value, assets=assets)
        graph['shared'].write_text('changed during semantic verification')
    monkeypatch.setattr(release, '_validate_uncached', change)
    with pytest.raises(ValueError, match='Frozen input differs'):
        validate(graph)
    graph['shared'].write_bytes(before)
    monkeypatch.setattr(release, '_validate_uncached', graph['full'])
    validate(graph)
    assert graph['calls'] == [True, True]


def test_real_release_cold_warm_and_rehashed_case_rejection(tmp_path, monkeypatch):
    source = common.RELEASE_ROOT/'publication01/releases/towel_folding/single_arm'
    if not source.is_dir():
        pytest.skip('Optional formal publication is unavailable')
    root = tmp_path/'actual_release'; shutil.copytree(source, root)
    # Formal publications are read-only; tampering is confined to this copy.
    for path in root.rglob('*'):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    cache.clear(); original = release._validate_uncached; calls = []
    def full(*args, **kwargs):
        calls.append(kwargs['assets']); return original(*args, **kwargs)
    monkeypatch.setattr(release, '_validate_uncached', full)
    manifest = common.read_json(root/'release_manifest.json')
    release.validate_release(root, manifest); release.validate_release(root, manifest)
    assert calls == [True]
    record = manifest['cases']['1']; path = root/record['scene']
    scene = common.read_json(path); scene['cloths'][0]['initial_vertices_m'][0][0] += .001
    common.write_json(path, scene)
    manifest['files_sha256'][record['scene']] = common.file_hash(path)
    manifest['files'][record['scene']]['sha256'] = common.file_hash(path)
    common.write_json(root/'release_manifest.json', manifest)
    with pytest.raises(ValueError, match='differs from exact parent derivation'):
        release.validate_release(root, manifest)
    assert calls == [True, True]
    cache.clear()


@pytest.mark.parametrize('target', ['root', 'parent', 'asset'])
def test_explicit_epoch_retains_full_directory_inventory_checks(graph, target):
    from bench.isaacsim.validation_epoch import ValidationEpoch
    with pytest.raises(ValueError, match='directory inventory'):
        with ValidationEpoch('directory mutation'):
            validate(graph); validate(graph)
            directories = {'root': graph['root'], 'parent': graph['parent'],
                'asset': Path(graph['manifest']['cases']['0']['asset_manifest']).parent}
            (directories[target]/'undeclared.json').write_text('{}')
            validate(graph)


def test_explicit_epoch_clears_semantic_reuse_at_both_boundaries(graph):
    from bench.isaacsim.validation_epoch import ValidationEpoch
    validate(graph)
    assert len(graph['calls']) == 1
    with ValidationEpoch('fresh batch'):
        validate(graph); validate(graph)
        assert len(graph['calls']) == 2
    validate(graph)
    assert len(graph['calls']) == 3
