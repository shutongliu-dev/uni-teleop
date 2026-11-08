"""A native worker reuses its server's release validation only after re-proving it."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from bench.isaacsim import release
from bench.paths import load_json
from bench.runtime import release_cache

PAST = 1_700_000_000*10**9
SOURCE = 'bench/runtime/release_cache.py'


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A fixture release validated by this process acting as the server."""
    root = tmp_path/'release'; root.mkdir()
    (root/'pinned.bin').write_bytes(b'pinned bytes')
    (root/'task.json').write_text(json.dumps({'id': 'fixture', 'title_zh': '夹具',
                                              'steps': [{'instruction_zh': '保持'}]}))
    (root/'scene.json').write_text('{}')
    (root/'physics_profile.json').write_text('{"isaacsim_options": {}}')
    manifest = {'tasks': [{'id': 'fixture', 'scene': 'scene.json', 'task': 'task.json'}],
                'physics_profile': 'physics_profile.json',
                'pins': {p.name: release.digest(p) for p in root.iterdir()}}
    (root/'release_manifest.json').write_text(json.dumps(manifest))
    for path in (*root.iterdir(), root):
        os.utime(path, ns=(PAST, PAST))
    calls = []

    def validate_fixture(path, *, assets=True):
        calls.append(assets)
        path = Path(path).resolve()
        value = load_json(path/'release_manifest.json')
        if {p.name for p in path.iterdir()} != {*value['pins'], 'release_manifest.json'}:
            raise ValueError('Undeclared fixture file')
        for name, expected in value['pins'].items():
            release.checked(path/name, expected)
        return value
    monkeypatch.setattr(release, '_validate', validate_fixture)
    release_cache.clear()
    release_cache.validate_cached(root, assets=True)
    release_cache.validate_cached(root, assets=False)
    sources = {SOURCE: hashlib.sha256((release_cache._PROJECT/SOURCE).read_bytes()).hexdigest()}
    yield root, calls, release_cache.worker_handoff(root, sources=sources)
    release_cache.clear()


def pinned(directory, payload, *, tamper=None):
    """Write a handoff the way NativeWorker does and return its configuration pin."""
    path = directory/'release_validation_handoff.json'
    data = json.dumps(payload, separators=(',', ':')).encode()
    path.write_bytes(data if tamper is None else tamper(data))
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def test_reproved_handoff_returns_the_server_results_without_validation(served, tmp_path):
    root, calls, payload = served
    assert calls == [True, False] and [row['assets'] for row in payload['validations']] == [False, True]
    assert all(row['result_is_manifest'] and row['result'] is None for row in payload['validations'])
    results, record = release_cache.adopt_worker_handoff(pinned(tmp_path, payload), root)
    assert results == {False: (release.validate(root, assets=False), {}), True: (release.validate(root), {})}
    assert calls == [True, False, False, True]  # Only the two explicit comparisons above.
    assert record['reused'] == [False, True] and not record['revalidate'] and 'rejected' not in record
    assert record['sources_checked'] == 1 and record['inputs_checked'] >= 6


@pytest.mark.parametrize('fault', ['tampered', 'manifest', 'source', 'release', 'project', 'missing'])
def test_unproved_handoff_is_rejected_whole(served, tmp_path, fault):
    root, calls, payload = served
    tamper = None
    if fault == 'tampered':
        tamper = lambda data: data.replace(b'"result_is_manifest":true', b'"result_is_manifest":false', 1)
    elif fault == 'manifest':
        (root/'release_manifest.json').write_text((root/'release_manifest.json').read_text()+' ')
    elif fault == 'source':
        payload['sources_sha256'][SOURCE] = '0'*64
    elif fault == 'project':
        payload['project_root'] = str(tmp_path)
    reference = None if fault == 'missing' else pinned(tmp_path, payload, tamper=tamper)
    target = tmp_path/'other' if fault == 'release' else root
    results, record = release_cache.adopt_worker_handoff(reference, target)
    assert results == {} and record['reused'] == [] and record['rejected']
    assert calls == [True, False]


@pytest.mark.parametrize('change', ['mtime', 'content', 'listed_directory'])
def test_changed_recorded_input_revalidates(served, tmp_path, change):
    root, calls, payload = served
    if change == 'mtime':
        os.utime(root/'pinned.bin', ns=(PAST, PAST+10**9))
    elif change == 'content':
        (root/'pinned.bin').write_bytes(b'PINNED BYTES')
    else:
        (root/'undeclared.json').write_text('{}')
    results, record = release_cache.adopt_worker_handoff(pinned(tmp_path, payload), root)
    assert results == {} and record['reused'] == [] and set(record['revalidate']) == {'False', 'True'}
    assert 'rejected' not in record and calls == [True, False]


def test_export_needs_a_current_cached_validation_and_sources(served):
    root, calls, payload = served
    assert release_cache.worker_handoff(root, sources={}) is None
    (root/'release_manifest.json').write_text((root/'release_manifest.json').read_text()+' ')
    assert release_cache.worker_handoff(root, sources={SOURCE: '0'*64}) is None
    release_cache.clear()
    assert release_cache.worker_handoff(root, sources={SOURCE: '0'*64}) is None


@pytest.mark.parametrize('handoff', [True, False])
def test_worker_binds_reproved_inputs_and_otherwise_validates(served, tmp_path, monkeypatch, handoff):
    from bench.runtime import worker
    from bench.simulation.isaacsim_backend import robot_registry
    root, calls, payload = served
    bound = []

    class Bound(Exception):
        pass

    def stop(scene, hand_id):
        bound.append(scene)
        raise Bound('release inputs selected')

    class Socket:
        def __init__(self, *args):
            pass
        bind = listen = close = lambda self, *args: None
        def accept(self):
            return self, None

    monkeypatch.setattr(robot_registry, 'scene_with_hand', stop)
    monkeypatch.setattr(worker, 'verify_if_snapshot', lambda *args, **kwargs: None)
    monkeypatch.setattr(worker.socket, 'socket', Socket)
    monkeypatch.setattr(worker, 'send', lambda connection, reply: None)
    config = {'output': str(tmp_path/'native'), 'release': str(root), 'task_id': 'fixture',
              'seed': 0, 'hand_id': 'unused', 'device': 'cuda:0'}
    if handoff:
        (tmp_path/'server').mkdir()
        config['release_validation_handoff'] = pinned(tmp_path/'server', payload)
    (tmp_path/'config.json').write_text(json.dumps(config))
    monkeypatch.setattr(sys, 'argv', ['worker', '--config', str(tmp_path/'config.json'), '--socket', 'unused'])
    with pytest.raises(Bound):
        worker.main()
    assert bound == [{'isaacsim': {}}]
    reuse = tmp_path/'native/release_validation_reuse.json'
    if handoff:
        assert calls == [True, False] and load_json(reuse)['reused'] == [False, True]
    else:
        assert calls == [True, False, True] and not reuse.exists()


def test_native_worker_pins_only_an_explicit_handoff(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from bench.runtime import native
    process = MagicMock(); process.poll.return_value = None

    def socket_directory(**kwargs):  # close() removes it, so each worker gets its own.
        temporary = tmp_path/f'socket_{len(list(tmp_path.glob("socket_*")))}'
        temporary.mkdir(); (temporary/'worker.sock').touch()
        return str(temporary)
    monkeypatch.setattr(native.tempfile, 'mkdtemp', socket_directory)
    monkeypatch.setattr(native.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(native.socket, 'socket', lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(native, 'receive', lambda connection, *, wait_for_data=None: {'ok': True})
    monkeypatch.setattr(native, 'execution_environment', lambda device: {'capture_status': 'recorded'})
    for name, handoff in (('plain', None), ('handed', {'schema': release_cache.HANDOFF_SCHEMA})):
        worker = native.NativeWorker(output=tmp_path/name, task_id='test', hand_id='test', seed=0,
                                     release=tmp_path, device='cuda:0', release_validation=handoff)
        worker.close(force=True)
        config = load_json(tmp_path/name/'worker_config.json')
        path = tmp_path/name/'release_validation_handoff.json'
        if handoff is None:
            assert 'release_validation_handoff' not in config and not path.exists()
        else:
            assert config['release_validation_handoff'] == {
                'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            assert load_json(path) == handoff


@pytest.fixture
def revised(served, tmp_path, monkeypatch):
    """A child release whose own validation validates the served fixture as its parent."""
    parent, calls, _ = served
    child = tmp_path/'child'; child.mkdir()
    (child/'release_manifest.json').write_text(json.dumps({'tasks': [], 'parent': str(parent)}))
    for path in (child/'release_manifest.json', child):
        os.utime(path, ns=(PAST, PAST))

    def with_parents(root, *, assets=True):
        if Path(root).resolve() != child.resolve():
            return release._validate(root, assets=assets), {}
        calls.append(('child', assets))
        value = load_json(child/'release_manifest.json')
        return value, {str(parent.resolve()): release.validate(parent, assets=assets)}
    monkeypatch.setattr(release, 'validate_with_parents', with_parents)
    release_cache.clear()
    del calls[:]
    return child, parent, calls


def test_child_proof_carries_its_parent_until_a_parent_file_changes(revised, monkeypatch):
    child, parent, calls = revised
    _, parents = release_cache._validated(child, True)
    assert parents == {str(parent.resolve()): load_json(parent/'release_manifest.json')}
    assert calls == [('child', True), True]  # The parent is validated once, inside its child.
    seen = []
    monkeypatch.setattr(release, 'validated_task_inputs', lambda root, manifest, task_id, *args, parents=None,
                        **kwargs: seen.append(parents) or (None, {}, {}, {}))
    release_cache.task_inputs_cached(child, 'fixture', seed=3)
    assert seen == [parents] and calls == [('child', True), True]
    os.utime(parent/'pinned.bin', ns=(PAST, PAST+10**9))
    release_cache.task_inputs_cached(child, 'fixture', seed=3)
    assert seen == [parents, parents] and calls == [('child', True), True, ('child', True), True]


@pytest.mark.parametrize('fault', [None, 'parent_input', 'parent_manifest_pin'])
def test_handoff_reuses_a_parent_only_inside_its_reproved_child(revised, tmp_path, fault):
    child, parent, calls = revised
    release_cache._validated(child, True)
    payload = release_cache.worker_handoff(child, sources={SOURCE: hashlib.sha256(
        (release_cache._PROJECT/SOURCE).read_bytes()).hexdigest()})
    (row,) = payload['validations']
    assert [record['root'] for record in row['parents']] == [str(parent.resolve())]
    if fault == 'parent_input':
        os.utime(parent/'pinned.bin', ns=(PAST, PAST+10**9))
    elif fault == 'parent_manifest_pin':
        row['parents'][0]['manifest_sha256'] = '0'*64
    results, record = release_cache.adopt_worker_handoff(pinned(tmp_path, payload), child)
    expected = {str(parent.resolve()): load_json(parent/'release_manifest.json')}
    if fault == 'parent_input':  # Inside the child's proof, so the child revalidates too.
        assert results == {} and record['revalidate'] == {'True': 'Recorded identity changed: '+str(parent/'pinned.bin')}
    elif fault == 'parent_manifest_pin':
        assert results[True][1] == {} and list(record['revalidate_parents']) == [str(parent.resolve())]
    else:
        assert results[True][1] == expected and record['reused_parents'] == {'True': list(expected)}
    assert calls == [('child', True), True]


@pytest.mark.parametrize('handed', ['matching', 'other_parent', None])
def test_revision_dispatch_validates_its_parent_unless_handed_that_result(tmp_path, monkeypatch, handed):
    from scenes.workrooms.robot_revision import release as revision
    parent, root = tmp_path/'parent', tmp_path/'revision'
    parent.mkdir(); root.mkdir(); (root/'robot_revision.json').write_text('{}')
    calls = []
    monkeypatch.setattr(release, 'task_inputs', lambda path, task_id, phase, **kwargs:
                        calls.append(('validate', Path(path))) or ({}, {}, {}, {}))
    monkeypatch.setattr(release, 'validated_task_inputs', lambda path, manifest, task_id, phase, **kwargs:
                        calls.append(('reuse', Path(path), manifest)) or ({}, {}, {}, {}))
    monkeypatch.setattr(revision, 'derive_inputs', lambda scene, task, initial, value: (scene, task, initial))
    manifest = {'task_id': 'fixture', 'base_release': {'path': str(parent)}, 'selection_kind': 'finite_seed',
                'tasks': [{'id': 'fixture'}]}
    parents = {'matching': {str(parent.resolve()): {'validated': True}},
               'other_parent': {str(tmp_path/'elsewhere'): {'validated': True}}, None: None}[handed]
    revision.task_inputs(root, 'fixture', manifest, seed=2, parents=parents)
    assert calls == ([('reuse', parent, {'validated': True})] if handed == 'matching' else [('validate', parent)])
