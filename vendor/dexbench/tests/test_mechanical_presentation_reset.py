"""Pad removal on all four pinned task inputs, with independent source bounds."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.mechanical_assembly.presentation_reset import remove_supply_pads


RULES_PARENT_RELEASES = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_referee_20261018/rules_v1/releases')


@pytest.fixture(params=['parts_sorting','peg_insertion','shaft_spacer_assembly','fastener_assembly'])
def inputs(request):
    task_id=request.param
    root=RULES_PARENT_RELEASES/task_id
    return task_id,load_json(root/'scene.json'),load_json(root/'task.json')


def test_actual_releases_keep_goals_assets_and_required_fixtures(inputs):
    task_id,scene,task=inputs;before=deepcopy((scene,task))
    changed,new_task,report=remove_supply_pads(scene,task)
    assert (scene,task)==before
    expected={'parts_sorting':0,'peg_insertion':2,'shaft_spacer_assembly':3,'fastener_assembly':0}[task_id]
    assert len(report['removed_fixture_ids'])==expected
    kept=[f for f in scene['fixtures'] if f['id'] not in report['removed_fixture_ids']]
    assert changed['fixtures']==kept
    assert changed['robot']==scene['robot'] and new_task['initial_state']==task['initial_state']
    assert changed['isaacsim']==scene['isaacsim'] and changed['physics']==scene['physics']
    assert new_task['public_rules']==task['public_rules']
    if task_id!='fastener_assembly':
        from scenes.workrooms.mechanical_assembly.evaluation import SequenceEvaluator
        assert SequenceEvaluator(new_task).task_id==task_id
    if not expected:
        assert (changed,new_task)==(scene,task)
        return
    evaluation=deepcopy(task['evaluation']);evaluation['initial_objects']=new_task['evaluation']['initial_objects']
    assert new_task['evaluation']==evaluation
    assert {f['id'] for f in changed['fixtures']}==({'robot_stand','socket_workplate'} if task_id=='peg_insertion' else {'robot_stand','shaft_fixture'})
    for old,new in zip(scene['objects'],changed['objects']):
        comparison=deepcopy(new);comparison['position_m']=old['position_m'];assert comparison==old
        assert new['position_m'][:2]==old['position_m'][:2]
        # These pinned pegs and ring meshes use the lowest source face as Z=0.
        # Check USD ring source directly, independently of the helper's loader.
        if task_id=='shaft_spacer_assembly':
            from pxr import Usd,UsdGeom
            stage=Usd.Stage.Open(str(Path(new['asset_root'])/(new['id']+'.usda')))
            points=np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath('/Ring/collision')).GetPointsAttr().Get())
            assert points[:,2].min()==pytest.approx(0.,abs=1.e-10)
        assert new['position_m'][2]==pytest.approx(.7802,abs=1.e-10)
        assert old['position_m'][2]-new['position_m'][2]==pytest.approx(.0058,abs=1.e-10)
        assert new_task['evaluation']['initial_objects'][new['id']]['position_m']==new['position_m']
        assert new_task['reset_conditions']['supply_positions_m'][new['id']]==new['position_m']
        evidence=report['object_updates'][new['id']]
        assert evidence['old_support_surface_z_m']==pytest.approx(.786,abs=1.e-8)
        if task_id=='peg_insertion':assert evidence['pad_maximum_z_m']==pytest.approx(.792,abs=1.e-8)
    assert not report['native_validation_claimed']
    # Calling on the already changed inputs must not lower them again.
    again,again_task,again_report=remove_supply_pads(changed,new_task)
    assert (again,again_task)==(changed,new_task) and not again_report['object_updates']


def test_foreign_pad_goal_or_seated_task_state_is_rejected():
    root=RULES_PARENT_RELEASES/'peg_insertion';scene=load_json(root/'scene.json');task=load_json(root/'task.json')
    task['evaluation']['expected_rest_targets']={'peg_0':'/World/Fixtures/supply_cradle_0'}
    with pytest.raises(ValueError,match='still has task/reset references'):
        remove_supply_pads(scene,task)
    task['evaluation'].pop('expected_rest_targets')
    scene['objects'][0]['position_m'][2]+=.01
    task['evaluation']['initial_objects']['peg_0']['position_m']=deepcopy(scene['objects'][0]['position_m'])
    with pytest.raises(ValueError,match='not seated'):
        remove_supply_pads(scene,task)
