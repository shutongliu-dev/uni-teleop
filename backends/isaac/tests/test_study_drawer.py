"""Meaningful sequence and native-evidence negative cases, not physics qualification."""
from copy import deepcopy

import pytest

from scenes.workrooms.study.drawer_rules import configuration, JOINT_PATH
from scenes.workrooms.study.drawer_evaluation import SequenceEvaluator


class Trace:
    def __init__(self):
        self.config=configuration(); self.index=-1
        self.eval=SequenceEvaluator(self.config,trusted_normalized_input=True)
        self.states={n:{**deepcopy(o['initial_pose']),'linear_velocity_m_s':[0.,0.,0.],
            'angular_velocity_rad_s':[0.,0.,0.]} for n,o in self.config['objects'].items()}
        self.contacts=[]; self.q=0.

    def step(self, **changes):
        self.index+=1
        obs={'schema':'study_native_observation_v1','configuration_sha256':self.config['configuration_sha256'],
            'episode_id':'test','step_index':self.index,'time_s':self.index*.0005,
            'physics_dt_s':.0005,'sampling_complete':True,'contacts_complete':True,
            'physics_valid':True,'robot_only_actions':True,'object_pose_writes':False,
            'budget_exhausted':False,'robot_action_count':int(self.index>0),'objects':deepcopy(self.states),
            'contacts':deepcopy(self.contacts),'joints':{JOINT_PATH:{'kind':'prismatic','position_m':self.q,
                'velocity_m_s':0.,'source':'derived_from_native_rigid_transforms_and_COM_velocities'}}}
        obs.update(changes)
        return self.eval.update(obs)

    def open_and_extract(self):
        self.step(); self.q=.14; self.states['drawer']['position_m'][0]=.43
        self.states['notebook']['position_m']=[.37,.23,.8172]; self.step()
        self.states['notebook']['position_m'][2]=.89; self.step()

    def target(self, gap=.010):
        self.q=gap; self.states['drawer']['position_m'][0]=.57-gap
        self.states['notebook']['position_m']=[.49,-.24,.804]
        self.contacts=[{'a':'notebook','b':'tray','point_m':[.49,-.24,.792],
            'impulse_on_a_n_s':[0.,0.,.00054],'separation_m':0.}]

    def settle(self, count=603):
        for _ in range(count): result=self.step()
        return result


def test_real_sequence_with_generous_close_gap_and_free_heading_passes():
    t=Trace();t.open_and_extract();t.target(.011)
    t.states['notebook']['quaternion_wxyz']=[.9238795325,0.,0.,.3826834324]
    result=t.settle()
    assert result['success'] and result['sequence']=={'opened':True,'extracted':True}


def test_unopened_target_placement_never_scores():
    t=Trace();t.step();t.target()
    result=t.settle()
    assert not result['success'] and not result['sequence']['opened']


@pytest.mark.parametrize('late_opening', [.100, .14])
def test_early_extraction_cannot_be_credited_by_later_opening(late_opening):
    t = Trace(); t.step()
    t.q = .09; t.states['drawer']['position_m'][0] = .48
    t.states['notebook']['position_m'] = [.37, .23, .89]
    result = t.step()
    assert result['sequence'] == {'opened': False, 'extracted': False}
    t.q = late_opening; t.states['drawer']['position_m'][0] = .57-late_opening
    result = t.step()
    assert result['sequence'] == {'opened': True, 'extracted': False}
    t.target()
    assert not t.settle()['success']


def test_premature_extraction_is_recoverable_by_returning_and_reextracting():
    t = Trace(); t.step()
    t.q = .09; t.states['drawer']['position_m'][0] = .48
    t.states['notebook']['position_m'] = [.37, .23, .89]; t.step()
    t.q = .14; t.states['drawer']['position_m'][0] = .43
    assert not t.step()['sequence']['extracted']
    t.states['notebook']['position_m'] = [.37, .23, .8172]
    assert not t.step()['sequence']['extracted']
    t.states['notebook']['position_m'][2] = .89
    assert t.step()['sequence']['extracted']
    t.target()
    assert t.settle()['success']


def test_same_sample_opening_and_extraction_has_no_verified_order():
    t = Trace(); t.step()
    t.q = .14; t.states['drawer']['position_m'][0] = .43
    t.states['notebook']['position_m'] = [.37, .23, .89]
    assert not t.step()['sequence']['extracted']
    t.target()
    assert not t.settle()['success']


@pytest.mark.parametrize('outside_position', [
    [.49, -.24, .804],  # Correct target tray, not the open drawer.
    [.37, -.24, .8172],  # Low but outside the drawer laterally.
    [.20, .23, .8172],  # Low but in front of the drawer.
    [.29, .23, .8172],  # Center enters its envelope, whole notebook does not.
])
def test_lowering_outside_drawer_cannot_rearm_a_premature_extraction(outside_position):
    t = Trace(); t.step()
    t.q = .09; t.states['drawer']['position_m'][0] = .48
    t.states['notebook']['position_m'] = [.37, .23, .89]; t.step()
    t.q = .14; t.states['drawer']['position_m'][0] = .43
    assert not t.step()['sequence']['extracted']
    t.states['notebook']['position_m'] = outside_position
    assert not t.step()['sequence']['extracted']
    t.states['notebook']['position_m'] = [.37, .23, .89]
    assert not t.step()['sequence']['extracted']
    t.target()
    assert not t.settle()['success']


@pytest.mark.parametrize('kind',['no_support','robot_support','open_drawer','no_extraction'])
def test_incomplete_or_supported_by_robot_does_not_score(kind):
    t=Trace()
    if kind=='no_extraction':
        t.step();t.q=.14;t.states['drawer']['position_m'][0]=.43;t.step()
    else:t.open_and_extract()
    t.target(.06 if kind=='open_drawer' else .01)
    if kind=='no_support':t.contacts=[]
    if kind=='robot_support':
        t.contacts.append({'a':'notebook','b':'robot/finger','point_m':[.49,-.24,.81],
            'impulse_on_a_n_s':[.001,0.,0.],'separation_m':0.})
    assert not t.settle()['success']


def test_recovery_from_wrong_placement_is_allowed():
    t=Trace();t.open_and_extract();t.target();t.states['notebook']['position_m'][1]=-.6
    assert not t.settle(100)['success']
    t.target();assert t.settle()['success']


@pytest.mark.parametrize('changes',[{'joints':{}},{'step_index':4},{'object_pose_writes':True},{'contacts_complete':False}])
def test_missing_joint_skipped_frame_or_untrusted_evidence_is_invalid(changes):
    t=Trace();t.step();result=t.step(**changes)
    assert result['status']=='invalid' and result['terminal']


def test_fabricated_joint_position_cannot_stand_in_for_native_slide():
    t=Trace();t.step();t.q=.14
    assert t.step()['status']=='invalid'


def test_floor_is_failure_but_budget_is_incomplete():
    t=Trace();t.step();t.contacts=[{'a':'notebook','b':'floor','point_m':[.3,0.,0.],
        'impulse_on_a_n_s':[0.,0.,.001],'separation_m':0.}]
    assert t.step()['status']=='failure'
    t=Trace();t.step();assert t.step(budget_exhausted=True)['status']=='incomplete'


def test_prepositioned_reset_and_relaxed_configuration_rejected():
    t=Trace();t.states['notebook']['position_m']=[.49,-.24,.804]
    assert t.step()['status']=='invalid'
    c=configuration();c['mechanism']['travel_m']=.5
    with pytest.raises(ValueError):SequenceEvaluator(c)


def test_published_drawer_roundtrip_has_real_joint_and_independent_frozen_referee():
    from pathlib import Path
    from bench.isaacsim.release import task_inputs, load_evaluator, digest
    from scenes.workrooms.study.tasks.drawer_retrieval.definition import TASK
    from scenes.workrooms.study.release import validate_authored_reset
    root=Path(TASK['source_release']['path'])
    assert digest(root/'release_manifest.json')==TASK['source_release']['manifest_sha256']
    _,scene,task,initial=task_inputs(root,'drawer_retrieval',seed=17)
    assert task['episode_seed']==17 and len(initial['joint_positions_rad'])==29
    assert task['evaluation']['joint_paths']==[JOINT_PATH]
    assert scene['workroom']['family']=='study'
    validate_authored_reset(scene,task)
    evaluator=load_evaluator(root)(task,trusted_normalized_input=True)
    assert evaluator.inspect_conditions()['sequence']=={'opened':False,'extracted':False}
    wrong=deepcopy(task);wrong['evaluation']['joint_paths']=[]
    with pytest.raises(ValueError,match='authored joints'):
        validate_authored_reset(scene,wrong)


def test_actual_collider_boxes_leave_a_clear_open_drawer_to_tray_corridor():
    """Object-only fit witness; this deliberately does not model a robot grasp."""
    from itertools import product
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scenes.workrooms.study.task_scene import candidate_inputs
    candidate,_,_,_=candidate_inputs('drawer_retrieval')
    stage=Usd.Stage.CreateInMemory()
    stage.DefinePrim('/World/StudyTask','Xform').GetReferences().AddReference(candidate['initial_usd'],'/World/StudyTask')
    drawer='/World/StudyTask/Objects/drawer';note='/World/StudyTask/Objects/notebook'
    UsdGeom.Xformable(stage.GetPrimAtPath(drawer)).GetOrderedXformOps()[0].Set(Gf.Vec3d(.41,.23,.825))
    colliders=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    fixed=[p for p in colliders if not str(p.GetPath()).startswith(note+'/')]
    payload=[p for p in colliders if str(p.GetPath()).startswith(note+'/')]
    def bounds(prim):
        assert prim.IsA(UsdGeom.Cube)
        h=UsdGeom.Cube(prim).GetSizeAttr().Get()/2
        matrix=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        points=np.asarray([matrix.Transform(Gf.Vec3d(*p)) for p in product((-h,h),repeat=3)])
        return points.min(axis=0),points.max(axis=0)
    fixed_bounds=[bounds(p) for p in fixed]
    route=[(.35,.23,float(z)) for z in np.linspace(.8172,.92,21)]
    route += [(.35+.14*t,.23-.47*t,.92) for t in np.linspace(0.,1.,31)]
    route += [(.49,-.24,float(z)) for z in np.linspace(.92,.8042,21)]
    for position in route:
        UsdGeom.Xformable(stage.GetPrimAtPath(note)).GetOrderedXformOps()[0].Set(Gf.Vec3d(*position))
        payload_bounds=[bounds(p) for p in payload]
        for lo,hi in payload_bounds:
            for flo,fhi in fixed_bounds:
                assert np.any(hi<=flo+1.e-6) or np.any(fhi<=lo+1.e-6), position
