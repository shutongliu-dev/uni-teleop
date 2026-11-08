from copy import deepcopy
import json
from pathlib import Path
import pytest

from bench.paths import DATA_ROOT, load_json
from scripts.assets.prepare_robot_revision_releases import read_request,installation_id
from test_robot_revision_release import original_robot_catalog


def test_internal_selection_id_does_not_create_public_menu():
    row={'family':'kitchen','task_id':'produce_transfer','configuration_id':None,'internal_configuration_id':'single_arm'}
    assert installation_id(row)=='kitchen/produce_transfer/single_arm'
    assert row['configuration_id'] is None
    row.update(family='living_room',task_id='tower_extraction',internal_configuration_id=None)
    assert installation_id(row)=='living_room/tower_extraction/fixed'


@pytest.mark.parametrize('fault',['duplicate','empty','unknown_field','relative_output','reference_hash'])
def test_request_fail_closed_before_output_creation(tmp_path,fault):
    from scenes.workrooms.robot_revision.assets import file_hash
    p=tmp_path/'input.json';p.write_text('{}')
    ref={'path':str(p),'sha256':file_hash(p)}
    value={'schema':'robot_revision_preparation_request.v1','parent_matrix':deepcopy(ref),
        'ready_report':deepcopy(ref),'robot_asset_manifest':deepcopy(ref),
        'output':f'{DATA_ROOT}/datasets/test_only_not_created'}
    if fault=='duplicate':value['installations']=['same','same']
    elif fault=='empty':value['installations']=[]
    elif fault=='unknown_field':value['qualification']=True
    elif fault=='relative_output':value['output']='relative'
    elif fault=='reference_hash':value['ready_report']['sha256']='0'*64
    request=tmp_path/'request.json';request.write_text(json.dumps(value))
    with pytest.raises(ValueError):read_request(request)


def test_actual_runtime_inputs_and_113_ready_descriptors_are_read_only(original_robot_catalog):
    from scripts.assets.prepare_robot_revision_releases import descriptors
    from scenes.workrooms.robot_revision.assets import file_hash,read_asset_manifest,check_robot_binding
    base=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/robot_l_mount_update_20261104')
    paths={'parent_matrix':base/'bench/parent_matrix02.json','ready_report':base/'ready01/report.json',
           'robot_asset_manifest':Path(f'{DATA_ROOT}/assets/robots/assemblies/fr3_sharpa/isaacsim_6_1_l_mount_v1/runtime_inputs_manifest.json')}
    if any(not p.is_file() for p in paths.values()):pytest.skip('Complete immutable robot/ready inputs are unavailable')
    value={'schema':'robot_revision_preparation_request.v1',**{k:{'path':str(p),'sha256':file_hash(p)} for k,p in paths.items()},
           'output':str(base/'test_only_not_created')}
    before={str(p):file_hash(p) for p in paths.values()}
    parents={installation_id(row):row['source_release']
             for row in load_json(paths['parent_matrix'])['rows']}
    # The user-approved base-world fixes of 2026-11-07 rebased exactly the three belt installations
    # (nist_usability_20261107b) and the three gear installations (wrench_reach_v1); this historical
    # preparation must refuse them and reproduce the other 107.
    rebased={f'mechanical_assembly/{task}/{c}' for task in ('nist_belt_drive_assembly','gear_train_assembly')
             for c in ('single_arm','dual_same_side','dual_opposed')}
    assert rebased<=set(parents)
    with pytest.raises(ValueError,match='no longer selects the exact original parent'):
        descriptors({**value,'installations':sorted(rebased)})
    result=descriptors({**value,'installations':sorted(set(parents)-rebased)})
    asset=read_asset_manifest(paths['robot_asset_manifest'])
    assert len(result)==107 and sum(r['selection_kind']=='practice' for r in result)==4
    for row in result:
        check_robot_binding(row['revision'],asset)
        assert row['parent_release']==parents[row['installation_id']]
    assert {str(p):file_hash(p) for p in paths.values()}==before
    assert not Path(value['output']).exists()
