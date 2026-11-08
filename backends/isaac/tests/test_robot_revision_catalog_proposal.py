"""Exact catalog proposal bookkeeping; publication validators are separately tested."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT, load_json
from scripts.assets import propose_robot_revision_catalog as tool
from scripts.assets.prepare_robot_revision_releases import installation_id
from scenes.workrooms.robot_revision.inputs import canonical_hash


@pytest.fixture(params=('isaacsim_6_1_l_mount_v2','isaacsim_6_1_l_mount_v3'))
def inputs(tmp_path,monkeypatch,request):
    matrix=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/robot_l_mount_update_20261104/bench/parent_matrix02.json')
    old=load_json(matrix)['rows']
    expected=[];prepared=[];version=request.param
    for row in old:
        robot={'hand_id':'sharpa_w01_right','assembly_version':version}
        if row['task_id']=='tower_extraction':
            robot={'hand_id':'fr3_sharpa_instances','assembly_version':'namespaced_v1',
                   'instances':{side:{'assembly_version':version} for side in ('front','rear')}}
        item={k:row[k] for k in ('family','task_id','configuration_id','internal_configuration_id')}
        item.update(installation_id=installation_id(row),parent_release=row['source_release'],
                    ready_reference={'path':'/test/only/ready','sha256':'a'*64},revision={'robot':robot})
        expected.append(item)
        root=tmp_path/'releases'/item['installation_id'];root.mkdir(parents=True)
        manifest={'robot_revision_sha256':canonical_hash(item['revision']),
                  'configuration_id':row['internal_configuration_id'],'selection_kind':row['selection_kind']}
        path=root/'release_manifest.json';path.write_text(json.dumps(manifest))
        prepared.append({k:deepcopy(v) for k,v in item.items() if k!='revision'}|{
            'revision_sha256':manifest['robot_revision_sha256'],'selection_kind':row['selection_kind'],
            'status':'prepared_unvalidated','release':{'path':str(root),'manifest_sha256':tool.file_hash(path)}})
    request_path=tmp_path/'request.json';request_path.write_text(json.dumps({'output':str(tmp_path)}))
    report={'schema':'robot_revision_preparation.v1','complete':True,'requested_installations':113,
        'rows':prepared,'request':{'path':str(request_path),'sha256':tool.file_hash(request_path)},
        'physics_accepted':False,'task_policy_success_claimed':False,'inherited_qualification':False,
        'catalog_modified':False,'old_inputs_modified':False,'native_steps':0,'model_calls':0}
    path=tmp_path/'preparation.json';path.write_text(json.dumps(report))
    pins=tmp_path/'pins.json';pins.write_text(json.dumps({'schema':'robot_revision_catalog.v1','tasks':{}}))
    monkeypatch.setattr(tool.catalog,'PINS',pins)
    monkeypatch.setattr(tool,'read_request',lambda path:load_json(Path(path)))
    monkeypatch.setattr(tool,'descriptors',lambda request:deepcopy(expected))
    calls=[]
    def publication(definition,releases):
        # Observe the real original definitions and exact public-menu keys;
        # do not create 113 synthetic robot USDs to retest the existing loader.
        original=definition.get('robot_configurations') or {'fixed':definition}
        assert set(releases)==set(original)
        calls.append((definition['family'],definition['task_id'],deepcopy(releases)))
        return {'test_only_publication_validator_spy':True}
    monkeypatch.setattr(tool.catalog,'publication_row',publication)
    return path,report,expected,calls,pins


def test_all_51_tasks_113_installations_reuse_publication_validator(inputs):
    path,report,expected,calls,pins=inputs
    before=pins.read_bytes()
    version=expected[0]['revision']['robot']['assembly_version']
    proposal,refs=tool.proposal(path,tool.file_hash(path),version)
    assert set(proposal['tasks'])==tool.catalog.CURRENT_TASKS and len(refs)==113 and len(calls)==51
    assert pins.read_bytes()==before
    task0=next(r for r in calls if r[:2]==('card_room','task0'))
    assert set(task0[2])=={'single_arm','dual_same_side','dual_opposed'}
    assert task0[2]['single_arm'].name=='fixed'  # private legacy identity is None
    tower=next(r for r in calls if r[:2]==('living_room','tower_extraction'))
    assert set(tower[2])=={'fixed'}


@pytest.mark.parametrize('fault',['partial','duplicate','parent','ready','revision','private_config',
                                  'qualification','source_overlay','wrong_version','wrong_sha','changed_manifest'])
def test_proposal_cannot_promote_incomplete_mismatched_or_qualified_inputs(inputs,fault):
    path,report,expected,calls,pins=inputs
    version=expected[0]['revision']['robot']['assembly_version']
    if fault=='partial':report['rows'].pop()
    elif fault=='duplicate':report['rows'][-1]=deepcopy(report['rows'][0])
    elif fault=='parent':report['rows'][0]['parent_release']['manifest_sha256']='0'*64
    elif fault=='ready':report['rows'][0]['ready_reference']['sha256']='0'*64
    elif fault=='revision':report['rows'][0]['revision_sha256']='0'*64
    elif fault=='private_config':report['rows'][0]['internal_configuration_id']='changed'
    elif fault=='qualification':report['physics_accepted']=True
    elif fault=='source_overlay':pins.write_text(json.dumps({'schema':'robot_revision_catalog.v1','tasks':{'x':{}}}))
    elif fault=='wrong_version':version='isaacsim_6_1_l_mount_v1'
    elif fault=='changed_manifest':
        (Path(report['rows'][0]['release']['path'])/'release_manifest.json').write_text('{}')
    path.write_text(json.dumps(report))
    digest='0'*64 if fault=='wrong_sha' else tool.file_hash(path)
    with pytest.raises(ValueError):tool.proposal(path,digest,version)
    assert not calls
