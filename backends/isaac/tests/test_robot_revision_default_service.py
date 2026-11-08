"""Default HTTP selection follows complete pins, without changing legacy loads."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenes.workrooms.robot_revision import catalog
from bench.runtime import server


def pins(tmp_path, monkeypatch, tasks):
    path=tmp_path/'pins.json';path.write_text(json.dumps({'schema':'robot_revision_catalog.v1','tasks':tasks}))
    monkeypatch.setattr(catalog,'PINS',path)
    return path


def test_empty_pins_retain_historical_default_and_partial_pins_fail(tmp_path,monkeypatch):
    path=pins(tmp_path,monkeypatch,{})
    legacy=tmp_path/'old_release'
    assert catalog.default_service_release(legacy)==legacy
    path.write_text(json.dumps({'schema':'robot_revision_catalog.v1','tasks':{'card_room/task0':{}}}))
    with pytest.raises(ValueError,match='complete current'):
        catalog.default_service_release(legacy)


def test_complete_pins_resolve_actual_current_task0_release_and_verify_sha(tmp_path,monkeypatch):
    from scenes.workrooms import catalog as tasks
    pins(tmp_path,monkeypatch,{key:{} for key in catalog.CURRENT_TASKS})
    release=tmp_path/'revised';release.mkdir()
    manifest=release/'release_manifest.json';manifest.write_text('{"test_only":true}')
    pin={'path':str(release),'manifest_sha256':catalog.file_hash(manifest)}
    value={'release_schema':catalog.SCHEMA,'source_release':pin,'default_configuration_id':'single_arm',
           'robot_configurations':{'single_arm':{'source_release':pin}}}
    calls=[]
    def definition(family,task):
        calls.append((family,task));return value
    monkeypatch.setattr(tasks,'task_definition',definition)
    assert catalog.default_service_release(tmp_path/'legacy')==release
    assert calls==[('card_room','task0')]
    manifest.write_text('{"test_only":false}')
    with pytest.raises(ValueError):catalog.default_service_release(tmp_path/'legacy')


@pytest.mark.parametrize('mode',['implicit','environment','explicit','service'])
def test_server_default_resolution_respects_explicit_history_and_injection(tmp_path,monkeypatch,mode):
    monkeypatch.delenv('BENCH_TASK_RELEASE',raising=False)
    calls=[]
    def resolve(legacy):
        calls.append(legacy);return tmp_path/'current'
    monkeypatch.setattr(catalog,'default_service_release',resolve)
    monkeypatch.setattr(server,'verify_if_snapshot',lambda *a,**k:None)
    def service(directory,**options):
        return SimpleNamespace(release=options['release'])
    monkeypatch.setattr(server,'AgentService',service)
    kwargs={}
    if mode=='environment':monkeypatch.setenv('BENCH_TASK_RELEASE',str(tmp_path/'historical_env'))
    elif mode=='explicit':
        monkeypatch.setenv('BENCH_TASK_RELEASE',str(tmp_path/'historical_env'))
        kwargs['release']=tmp_path/'historical_explicit'
    elif mode=='service':kwargs['service']=SimpleNamespace(release=tmp_path/'injected')
    app=server.create_app(tmp_path/'episodes',**kwargs)
    expected={'implicit':'current','environment':'historical_env','explicit':'historical_explicit','service':'injected'}[mode]
    assert app.state.manager.release==tmp_path/expected
    assert bool(calls)==(mode=='implicit')
