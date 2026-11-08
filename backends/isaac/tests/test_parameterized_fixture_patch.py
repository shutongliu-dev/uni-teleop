"""A shared fixture layout cannot replace another installation's robot stand."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.parameterized.workroom import _patch


def test_preserves_current_installation_stand_and_exact_task_changes():
    original=[{'id':'robot_stand','position_m':[0,0,0]}, {'id':'rest','position_m':[1,0,0]}]
    altered=deepcopy(original);altered[1]['position_m']=[1,.1,0]
    current={'fixtures':[deepcopy(original[1]),{'id':'bimanual_robot_stand','position_m':[0,1,0]}]}
    before=deepcopy(current)
    result=_patch(current,[{'path':['fixtures'],'before':original,'after':altered}])
    assert current==before
    assert result=={'fixtures':[altered[1],current['fixtures'][1]]}


@pytest.mark.parametrize('kind',('stand','current_task','duplicate','new_task','reordered_task'))
def test_rejects_stand_edits_and_task_inventory_or_parent_mismatch(kind):
    before=[{'id':'robot_stand'}, {'id':'rest','x':1}, {'id':'fixed_target','x':2}]
    after=deepcopy(before);after[1]['x']=3
    current=deepcopy(before)
    if kind=='stand':after[0]['extra']='not allowed'
    elif kind=='current_task':current[2]['x']=4
    elif kind=='duplicate':after.append(deepcopy(after[1]))
    elif kind=='new_task':after.append({'id':'extra'})
    else:after[1:]=after[:0:-1]
    with pytest.raises(ValueError):
        _patch({'fixtures':current},[{'path':['fixtures'],'before':before,'after':after}])


@pytest.mark.parametrize('configuration',('single_arm','dual_same_side','dual_opposed'))
@pytest.mark.parametrize('seed',(0,1))
def test_actual_frozen_peg_asset_composes_for_all_three_installations(configuration,seed):
    base=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105')
    request=base/'requests01/workroom.json'
    asset=Path(f'{DATA_ROOT}/assets/task_parts/seed_randomization/v1/mechanical_assembly/parameterized_physics_v1/author_mechanical_first04/peg_insertion')/f'seed_{seed:02d}/seed_manifest.json'
    if not request.is_file() or not asset.is_file():pytest.skip('Frozen publication regression fixture unavailable')
    from scenes.workrooms.initial_randomization.release import adapter
    parent=load_json(request)['parents']['peg_insertion']['configurations'][configuration]['path']
    original=load_json(Path(parent)/'scene.json')
    values=adapter('peg_insertion').compose_case(parent,asset,configuration)
    assert [row for row in values['scene']['fixtures'] if 'robot_stand' in row['id']]==[
        row for row in original['fixtures'] if 'robot_stand' in row['id']]
    assert next(row for row in values['scene']['fixtures'] if row['id']=='socket_workplate')==next(
        row for row in original['fixtures'] if row['id']=='socket_workplate')
    if seed:
        assert next(row for row in values['scene']['fixtures'] if row['id']=='supply_cradle_0')!=next(
            row for row in original['fixtures'] if row['id']=='supply_cradle_0')
