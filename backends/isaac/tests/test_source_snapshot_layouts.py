"""Historical manifests describe data, never instructions to import old code."""
import json
import shutil
from pathlib import Path

import pytest

from scripts.maintenance import derive_project_sources as derive


freeze = derive.freeze
LAYOUTS = {
    4: ('interface', 'bench', 'simulation'),
    5: ('interface', 'bench', 'simulation', 'agent', 'scenes'),
    6: ('bench', 'agent', 'scenes'),
}


def snapshot_fixture(tmp_path, monkeypatch, version, *, with_asset=True):
    """Construct explicit old/new format bytes without executing a past tool."""
    roots = LAYOUTS[max(4, version)]
    snapshot = tmp_path/'parent'
    for tree in roots:
        (snapshot/tree).mkdir(parents=True)
    native_root = 'bench/simulation' if version == 6 else 'simulation'
    kit = ('bench/apps' if version == 6 else 'apps')+'/native.kit'
    names = [f'{root}/__init__.py' for root in roots]
    names += [native_root+'/native.py', 'scripts/isaac_python.sh', freeze.RUNTIME_GUARD_SOURCE, kit]
    for name in names:
        path = snapshot/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('raise AssertionError("Historical sources must not execute")\n')
        path.chmod(0o755 if name.endswith('.sh') else 0o644)
    shared = tmp_path/'shared'
    asset = shared/'hand'
    asset.mkdir(parents=True)
    (asset/'mesh.stl').write_text('shared asset, not source content\n')
    monkeypatch.setattr(freeze, 'SHARED_ASSET_ROOT', shared)
    references = {}
    if with_asset:
        relative = native_root+'/assets/hand'
        link = snapshot/relative
        link.parent.mkdir(parents=True)
        link.symlink_to(asset, target_is_directory=True)
        references[relative] = {'original_link': relative, 'original_link_target': str(asset),
                               'resolved_target': str(asset), 'snapshot_link_target': str(asset)}
    files = {name: freeze.digest(snapshot/name) for name in names}
    manifest = {'schema': 'complete_benchmark_source_snapshot_v1',
        'source_utility_version': f'complete_benchmark_source_verifier_v{version}',
        'source_root': str(tmp_path/'original_source'), 'snapshot_root': str(snapshot),
        'files_sha256': files, 'explicit_extra_resources': ['scripts/isaac_python.sh', kit],
        'suite_source_pins': {name: value for name, value in files.items() if name != kit
                             and (version >= 4 or name != freeze.RUNTIME_GUARD_SOURCE)},
        'shared_asset_references': references}
    if version >= 3:
        manifest.update(files_mode_octal={name: '0755' if name.endswith('.sh') else '0644' for name in names},
                        file_mode_policy='exact_or_write_bits_removed_v1')
    if version >= 5:
        manifest['source_trees'] = list(roots)
    (snapshot/'source_snapshot_manifest.json').write_text(json.dumps(manifest))
    return snapshot, manifest


@pytest.mark.parametrize('version', [1, 2, 3, 4, 5, 6])
def test_each_manifest_version_keeps_its_real_layout(tmp_path, monkeypatch, version):
    snapshot, manifest = snapshot_fixture(tmp_path, monkeypatch, version)
    before = (snapshot/'source_snapshot_manifest.json').read_bytes()
    result = freeze.verify(snapshot)
    assert result['verified'] and result['shared_asset_references'] == 1
    assert result['file_modes_verified'] is (version >= 3)
    assert freeze.snapshot_source_trees(manifest) == LAYOUTS[max(4, version)]
    assert (snapshot/'source_snapshot_manifest.json').read_bytes() == before


@pytest.mark.parametrize('version', [4, 5, 6])
def test_asset_whitelist_does_not_fall_back_between_layouts(tmp_path, monkeypatch, version):
    snapshot, manifest = snapshot_fixture(tmp_path, monkeypatch, version)
    old, reference = next(iter(manifest['shared_asset_references'].items()))
    wrong = ('simulation' if version == 6 else 'bench/simulation')+'/assets/hand'
    (snapshot/wrong).parent.mkdir(parents=True, exist_ok=True)
    (snapshot/old).rename(snapshot/wrong)
    manifest['shared_asset_references'] = {wrong: {**reference, 'original_link': wrong}}
    (snapshot/'source_snapshot_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Asset link must be below'):
        freeze.verify(snapshot)


@pytest.mark.parametrize('version', [4, 5, 6])
def test_derivation_preserves_parent_layout_and_asset_prefix(tmp_path, monkeypatch, version):
    parent, manifest = snapshot_fixture(tmp_path, monkeypatch, version)
    before = (parent/'source_snapshot_manifest.json').read_bytes()
    candidate = tmp_path/'candidate.py'
    candidate.write_text('raise AssertionError("Overlay source must not execute")\n')
    native = ('bench/simulation' if version == 6 else 'simulation')+'/native.py'
    overlay = tmp_path/'overlay.json'
    overlay.write_text(json.dumps({'schema': 'complete_benchmark_source_overlay_v1',
        'parent_manifest_sha256': freeze.digest(parent/'source_snapshot_manifest.json'),
        'entries': [{'target': native, 'expected_parent_sha256': manifest['files_sha256'][native],
                     'source_path': str(candidate), 'source_sha256': freeze.digest(candidate)}]}))
    child = tmp_path/'derived'
    result = derive.derive(parent, child, overlay)
    assert result['source_utility_version'] == manifest['source_utility_version']
    assert result.get('source_trees') == manifest.get('source_trees')
    assert result['shared_asset_references'] == manifest['shared_asset_references']
    assert (parent/'source_snapshot_manifest.json').read_bytes() == before
    assert (child/native).read_bytes() == candidate.read_bytes()
    # Offline verification retains original documents and requires no parent or
    # candidate code, even after those synthetic fixtures are removed.
    shutil.rmtree(parent)
    candidate.unlink()
    overlay.unlink()
    assert derive.verify_derivation(child)['offline_parent_transition_verified']


@pytest.mark.parametrize('version', [5, 6])
def test_declared_roots_cannot_relabel_another_layout(tmp_path, monkeypatch, version):
    snapshot, manifest = snapshot_fixture(tmp_path, monkeypatch, version)
    manifest['source_trees'] = list(LAYOUTS[6 if version == 5 else 5])
    (snapshot/'source_snapshot_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='versioned layout'):
        freeze.verify(snapshot)


@pytest.mark.parametrize('tree', ['interface', 'simulation'])
def test_v6_rejects_obsolete_source_roots(tmp_path, monkeypatch, tree):
    snapshot, _ = snapshot_fixture(tmp_path, monkeypatch, 6)
    (snapshot/tree).mkdir()
    (snapshot/tree/'hidden.py').write_text('# must not be ignored\n')
    with pytest.raises(ValueError, match='outside versioned layout'):
        freeze.verify(snapshot)


def test_new_freeze_matches_suite_inventory_without_treating_kit_or_json_as_python(tmp_path, monkeypatch):
    from agent.model_client import runner, suite
    source = tmp_path/'source'
    for tree in ('bench', 'agent', 'scenes'):
        (source/tree).mkdir(parents=True)
    names = (*runner.CLIENT_SOURCE_DEPENDENCIES, 'scripts/isaac_python.sh',
             'agent/model_client/runner.py', 'bench/simulation/native.py',
             'bench/adapters/manus.py', 'bench/data_tools/prepare.py',
             'bench/apps/native.kit', 'scenes/catalog.json')
    for name in names:
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture resource\n')
    (source/'scripts/isaac_python.sh').chmod(0o755)
    result = freeze.freeze(source, tmp_path/'snapshot',
                           ('scripts/isaac_python.sh', 'bench/apps/native.kit', 'scenes/catalog.json'))
    monkeypatch.setattr(suite, 'PROJECT', source)
    assert result['source_utility_version'] == 'complete_benchmark_source_verifier_v6'
    assert result['source_trees'] == ['bench', 'agent', 'scenes']
    assert result['suite_source_pins'] == suite.source_pins()
    for name in ('bench/apps/native.kit', 'scenes/catalog.json'):
        assert name in result['files_sha256'] and name not in result['suite_source_pins']
