"""Server request paths reuse a release validation only while its inputs are unchanged."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from bench.isaacsim import release
from bench.isaacsim.validation_epoch import VALIDATION_MODES, validation_context
from bench.paths import load_json
from bench.protocol import PolicyRegistry
from bench.runtime import release_cache
from bench.runtime.server import create_app

# Fixture inputs carry an old explicit mtime, so a later write differs from the
# recorded identity even within one coarse filesystem timestamp tick.
PAST = 1_700_000_000*10**9


@pytest.fixture
def frozen(tmp_path, monkeypatch):
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
        # Stands in for a schema validator: exact inventory plus pinned bytes.
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
    yield root, calls
    release_cache.clear()


def test_repeated_http_discovery_validates_the_release_once(frozen):
    root, calls = frozen
    manager = SimpleNamespace(release=root, registry=PolicyRegistry(), shutdown=lambda: None)
    with TestClient(create_app(service=manager)) as client:
        rows = [client.get('/tasks').json() for _ in range(3)]
        assert client.get('/interfaces', params={'task_id': 'fixture'}).status_code == 200
    assert rows[0] == rows[1] == rows[2] and rows[0][0]['task_id'] == 'fixture'
    assert calls == [False]


@pytest.mark.parametrize('change', ['content', 'mtime', 'replaced', 'listed_directory'])
def test_changed_input_identity_revalidates(frozen, change):
    root, calls = frozen
    expected = release_cache.validate_cached(root)
    assert release_cache.validate_cached(root) == expected and calls == [True]
    pinned = root/'pinned.bin'
    if change == 'content':
        pinned.write_bytes(b'PINNED BYTES')  # Same size; the write sets a new mtime.
        with pytest.raises(ValueError, match='Frozen input changed'):
            release_cache.validate_cached(root)
        pinned.write_bytes(b'pinned bytes')
    elif change == 'mtime':
        os.utime(pinned, ns=(PAST, PAST+10**9))
    elif change == 'replaced':
        (root/'copy.tmp').write_bytes(pinned.read_bytes()); (root/'copy.tmp').replace(pinned)
    else:
        (root/'undeclared.json').write_text('{}')
        with pytest.raises(ValueError, match='Undeclared'):
            release_cache.validate_cached(root)
        (root/'undeclared.json').unlink()
    assert release_cache.validate_cached(root) == expected
    revalidated = calls[1:]
    assert revalidated == [True]*(2 if change in ('content', 'listed_directory') else 1)
    assert release_cache.validate_cached(root) == expected and calls[1:] == revalidated


def test_failures_and_inputs_changed_during_validation_are_not_cached(frozen, monkeypatch):
    root, calls = frozen
    (root/'pinned.bin').write_bytes(b'corrupted')
    for _ in range(2):
        with pytest.raises(ValueError, match='Frozen input changed'):
            release_cache.validate_cached(root, assets=False)
    assert calls == [False, False]
    (root/'pinned.bin').write_bytes(b'pinned bytes'); os.utime(root/'pinned.bin', ns=(PAST, PAST))
    validate_fixture = release._validate

    def concurrent_writer(path, *, assets=True):
        value = validate_fixture(path, assets=assets)
        if len(calls) == 3:  # Touch an input after the validator has read it.
            os.utime(root/'pinned.bin', ns=(PAST, PAST+10**9))
        return value
    monkeypatch.setattr(release, '_validate', concurrent_writer)
    for _ in range(3):
        release_cache.validate_cached(root, assets=False)
    assert calls == [False]*4


@pytest.mark.parametrize('mode', VALIDATION_MODES)
def test_explicit_validation_scope_calls_plain_validate(frozen, mode):
    root, calls = frozen
    release_cache.validate_cached(root)
    with validation_context('process cache bypass', mode):
        release_cache.validate_cached(root); release_cache.validate_cached(root)
    # Only validate()'s own once-hashed scope memo may reuse, and only inside its scope.
    within = 1 if mode == 'full_bytes_once_v1' else 2
    assert calls == [True]*(1+within)
    with validation_context('no cross-scope reuse', mode):
        release_cache.validate_cached(root)
    assert len(calls) == 2+within
    release_cache.clear()
    with validation_context('scope result stays private', mode):
        release_cache.validate_cached(root)
    release_cache.validate_cached(root)
    assert len(calls) == 4+within


def test_copies_concurrency_and_uncached_worker_validation(frozen, monkeypatch):
    root, calls = frozen
    first = release_cache.validate_cached(root)
    first['tasks'].clear()
    assert release_cache.validate_cached(root)['tasks'][0]['id'] == 'fixture'
    release_cache.clear()
    validate_fixture = release._validate

    def slow(path, *, assets=True):
        time.sleep(.2)
        return validate_fixture(path, assets=assets)
    monkeypatch.setattr(release, '_validate', slow)
    with ThreadPoolExecutor(4) as pool:
        values = list(pool.map(lambda _: release_cache.validate_cached(root), range(4)))
    assert calls == [True, True] and all(value == values[0] for value in values)
    _, _, task, _ = release_cache.task_inputs_cached(root, 'fixture')
    assert task['id'] == 'fixture' and len(calls) == 2
    release.validate(root); release.task_inputs(root, 'fixture')
    assert len(calls) == 4


def test_import_machinery_is_process_state_not_a_release_input(frozen, monkeypatch, tmp_path):
    root, calls = frozen
    modules = tmp_path/'modules'; modules.mkdir()
    (modules/'release_cache_lazy_fixture.py').write_text('VALUE = 1\n')
    for path in (*modules.iterdir(), modules):
        os.utime(path, ns=(PAST, PAST))
    monkeypatch.syspath_prepend(str(modules))
    monkeypatch.delitem(sys.modules, 'release_cache_lazy_fixture', raising=False)
    validate_fixture = release._validate

    def first_use_import(path, *, assets=True):
        importlib.import_module('release_cache_lazy_fixture')  # Finder lists sys.path.
        list(importlib.metadata.distributions())  # So does package metadata discovery.
        return validate_fixture(path, assets=assets)
    monkeypatch.setattr(release, '_validate', first_use_import)
    release_cache.validate_cached(root)
    (modules/'unrelated_log.txt').write_text('churn')
    (modules/'release_cache_lazy_fixture.py').write_text('VALUE = 2\n')
    release_cache.validate_cached(root)
    assert calls == [True]


def test_substituted_release_loaders_run_on_every_request(frozen, monkeypatch):
    root, calls = frozen
    seen, loaded = [], []
    monkeypatch.setattr(release, 'validate', lambda path, assets=True: seen.append(assets) or {'tasks': []})
    release_cache.validate_cached(root); release_cache.validate_cached(root)
    assert seen == [True, True] and calls == []
    monkeypatch.setattr(release, 'task_inputs', lambda path, task_id, **kw: loaded.append(kw) or (None, {}, {}, {}))
    release_cache.task_inputs_cached(root, 'fixture', seed=3)
    assert loaded == [{'seed': 3}] and seen == [True, True]
