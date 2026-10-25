"""CPU streaming/observer contracts, never a successful native robot pour."""
from copy import deepcopy
import math
from types import MappingProxyType
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
from pxr import Usd, UsdGeom

from scenes.workrooms.living_room.tasks.water_pouring import assets, rules
from scenes.workrooms.living_room.tasks.water_pouring.evaluator import SequenceEvaluator
from scenes.workrooms.living_room.tasks.water_pouring.native_observer import NativeTaskObserver


def task():
    return {'id':'water_pouring','episode_seed':0,'evaluation':rules.evaluation_configuration(),
            'public_rules':rules.public_rules()}


def body(configuration,name):
    c=configuration['evaluation']['containers'][name]
    return {'position_m':list(c['initial_position_m']),
        'quaternion_wxyz':[math.cos(c['initial_yaw_rad']/2),0.,0.,math.sin(c['initial_yaw_rad']/2)],
        'linear_velocity_m_s':[0.,0.,0.],'angular_velocity_rad_s':[0.,0.,0.],
        'support_force_min_n':1.,'robot_contact_force_max_n':0.}


class Stream:
    def __init__(self, configuration=None):
        self.task=configuration or task();self.e=SequenceEvaluator(self.task);self.step_index=0;self.time=0.;self.left=0;self.spilled=0
        self.cup=0;self.outside=0;self.unresolved=0;self.lift=0.
        self.bodies={name:body(self.task,name) for name in ('kettle','cup')}
        self.obs=self.make(first=True)
        assert self.e.update(self.obs)['reason']=='reset_baseline_not_scored'
    def make(self,first=False):
        c=self.task['evaluation'];count=0 if first else 40
        self.time+=count*c['physics_dt_s'];start=self.step_index;self.step_index+=count
        self.left=max(self.left,self.cup+self.outside+self.unresolved);self.spilled=max(self.spilled,self.outside)
        return {'time_s':self.time,'sim_time_s':self.time,'metrics':{'state_finite':True},'task_state':{
            'schema':'living_water_observation_v1','contract_sha256':c['contract_sha256'],'reset_id':'test-episode',
            'time_s':self.time,'errors':[],
            'sampling':{'start_step':start,'end_step':self.step_index,'start_time_s':start*c['physics_dt_s'],
                        'end_time_s':self.step_index*c['physics_dt_s'],'sampled_substeps':count,'physics_dt_s':c['physics_dt_s']},
            'water':{'particle_count':c['particle_count'],'ids_sha256':c['particle_ids_sha256'],
                'counts':{'kettle':c['particle_count']-self.cup-self.outside-self.unresolved,'cup':self.cup,
                    'spilled_outside':self.outside,'in_flight_or_unresolved':self.unresolved},
                'cumulative_unique_spilled_particles':self.spilled,'ever_left_kettle_particles':self.left,
                'first_kettle_exit_regions':{'front_rim':self.left,'other_rim':0,'below_rim':0}},
            'vessels':deepcopy(self.bodies),'activity':{'max_kettle_lift_m':self.lift}}}
    def update(self):
        self.obs=self.make();return self.e.update(self.obs)
    def pour(self):
        self.cup=3704;self.lift=.03
    def settle(self,n=18):
        result=None
        for _ in range(n):result=self.update()
        return result


def test_target_return_release_and_300ms_success_then_motion_revokes():
    s=Stream();s.pour()
    for _ in range(15):assert not s.update()['success']
    result=s.update();assert result['success'] and result['stable_duration_s']==pytest.approx(.3)
    assert result['cup_water_ml']==pytest.approx(3704*.25/9259*1000)
    s.bodies['kettle']['linear_velocity_m_s'][0]=.1
    assert not s.update()['success']


def test_reset_and_empty_cup_never_earn_credit():
    s=Stream();assert not s.settle()['success']
    s.cup=3704;obs=s.make(first=True)
    e=SequenceEvaluator(s.task)
    assert e.update(obs)['evidence_status']=='invalid'


@pytest.mark.parametrize('change,reason',[
    (lambda s:s.bodies['kettle'].__setitem__('robot_contact_force_max_n',.2),'release_kettle'),
    (lambda s:s.bodies['cup'].__setitem__('support_force_min_n',0.),'cup_not_supported'),
    (lambda s:s.bodies['kettle']['position_m'].__setitem__(0,.70),'return_kettle_to_zone'),
    (lambda s:setattr(s,'unresolved',50),'water_still_in_flight'),
])
def test_meaningful_completion_gates(change,reason):
    s=Stream();s.pour();change(s);r=s.settle();assert not r['success'] and r['reason']==reason


def test_small_unresolved_water_is_tolerated_but_spill_is_cumulative():
    s=Stream();s.pour();s.unresolved=20;assert s.settle()['success']
    s.unresolved=0;s.outside=186;r=s.update();assert r['reason']=='spill_limit_exceeded' and not r['success']
    s.outside=0;assert s.update()['reason']=='spill_limit_exceeded'


def coarse_stream():
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1.)
    meta=assets.author_task(stage,particle_spacing_m=.008)
    configuration=task()
    configuration['evaluation']=rules.evaluation_configuration(asset_metadata=meta)
    return Stream(configuration)


def test_water_difference_excludes_flight_and_preserves_recaptured_spill_history():
    s=coarse_stream();s.cup=183;s.lift=.03;s.unresolved=8
    r=s.update();ml=250/488
    assert r['water_difference_ml']==pytest.approx(8*ml)
    assert r['current_spill_ml']==0 and r['cumulative_spill_ml']==0
    assert r['remaining_spill_budget_ml']==5
    s.unresolved=0;s.outside=8;r=s.update()
    assert r['current_spill_ml']==pytest.approx(8*ml)
    assert r['initial_water_ml']-r['kettle_water_ml']-r['cup_water_ml']==pytest.approx(r['current_spill_ml'])
    assert r['cumulative_spill_ml']==pytest.approx(8*ml)
    s.outside=0;s.cup+=8;r=s.update()
    assert r['water_difference_ml']==r['current_spill_ml']==0
    assert r['cumulative_spill_ml']==pytest.approx(8*ml)
    assert r['remaining_spill_budget_ml']==pytest.approx(5-8*ml)


def test_coarse_water_spill_budget_rejects_ten_particles_but_allows_nine():
    s=coarse_stream();s.cup=183;s.lift=.03;s.outside=9
    assert s.settle()['success']  # 4.61 mL, below the unchanged 5 mL budget.
    s.outside=10;r=s.update()
    assert not r['success'] and r['reason']=='spill_limit_exceeded'
    assert r['current_spill_ml']==pytest.approx(2500/488)
    assert r['remaining_spill_budget_ml']==0


@pytest.mark.parametrize('mutate',[
    lambda o:o['task_state'].__setitem__('contract_sha256','0'*64),
    lambda o:o['task_state'].__setitem__('reset_id','other'),
    lambda o:o['task_state']['water'].__setitem__('ids_sha256','f'*64),
    lambda o:o['task_state']['water']['counts'].__setitem__('kettle',0),
    lambda o:o['task_state']['sampling'].__setitem__('sampled_substeps',39),
    lambda o:o['task_state'].__setitem__('errors',['missing particle readback']),
])
def test_faults_are_invalid_not_success_or_competitive_loss(mutate):
    s=Stream();s.pour();obs=s.make();mutate(obs);r=s.e.update(obs)
    assert r['evidence_status']=='invalid' and not r['success'] and r['irrecoverable_reason'] is None


def test_duplicate_or_gap_and_rollback_are_rejected():
    s=Stream();s.pour();s.update();assert s.e.update(s.obs)['evidence_status']=='invalid'
    s=Stream();s.pour();s.update();obs=s.make();obs['task_state']['activity']['max_kettle_lift_m']=0
    assert s.e.update(obs)['evidence_status']=='invalid'


def test_volume_drift_resets_short_settle_and_timeout_is_incomplete():
    s=Stream();s.pour();s.settle(10);s.cup+=80
    assert s.update()['reason']=='cup_water_still_changing'
    s=Stream();obs=s.make();obs['terminated']=True
    assert s.e.update(obs)['reason']=='incomplete_terminated'


def native_fixture():
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1.);meta=assets.author_task(stage)
    t=task();t['evaluation']=rules.evaluation_configuration(asset_metadata=meta)
    points=np.array(UsdGeom.Points(stage.GetPrimAtPath(meta['water']['particles_path'])).GetPointsAttr().Get())
    frame={'baseline':True,'time_s':0.,'native_time_s':.01,'native_step':20,'physics_dt_s':.0005,
        'objects':{name:body(t,name) for name in ('kettle','cup')},'joints':{},'contacts':[],
        'particle_sets':{'water':{'positions_m':points,'velocities_m_s':np.zeros_like(points),
                                'ids':np.arange(len(points)),'total_mass_kg':.25}}}
    return t,frame


def test_native_observer_classifies_frozen_cavities_and_counts_spill_once():
    t,f=native_fixture();observer=NativeTaskObserver(t)
    for value in f['particle_sets']['water'].values():
        if isinstance(value,np.ndarray):value.setflags(write=False)
    observer.on_frame(MappingProxyType(f));baseline=observer.on_observe()
    assert baseline['water']['counts']['kettle']==9259 and baseline['sampling']['sampled_substeps']==0
    points=f['particle_sets']['water']['positions_m'].copy();points[0]=[.65,0,.787]
    f['particle_sets']['water']['positions_m']=points
    for tick in range(1,3):
        f.update(baseline=False,time_s=tick*.0005,native_time_s=.01+tick*.0005,native_step=20+tick)
        observer.on_frame(f)
    r=observer.on_observe()
    assert r['water']['cumulative_unique_spilled_particles']==1
    assert r['water']['counts']['spilled_outside']==1 and r['sampling']['sampled_substeps']==2
    f['particle_sets']['water']['ids']=np.arange(9259)[::-1]
    f.update(time_s=.0015,native_time_s=.0115,native_step=23)
    observer.on_frame(f);assert observer.on_observe()['errors']


def test_observer_holds_sleeping_support_only_at_unchanged_pose():
    t,f=native_fixture();o=NativeTaskObserver(t);o.on_frame(f);o.on_observe()
    for tick in range(1,4):
        f.update(baseline=False,time_s=tick*.0005,native_time_s=.01+tick*.0005,native_step=20+tick)
        f['contacts']=[{'actor0':t['evaluation']['vessel_actor_paths']['kettle'],
            'actor1':'/World/LivingTask/CatchTray/Floor','impulse_magnitude_sum_ns':.001}] if tick==1 else []
        if tick==3:f['objects']['kettle']['position_m'][2]+=.03
        o.on_frame(f);r=o.on_observe()
        assert r['vessels']['kettle']['support_force_min_n']==(2. if tick<3 else 0.)


def test_frozen_score_imports_only_sibling_rules(tmp_path):
    source=Path(sys.modules[SequenceEvaluator.__module__].__file__).parent
    (tmp_path/'__init__.py').write_text('')
    for name in ['rules.py','evaluator.py']:(tmp_path/name).write_text((source/name).read_text())
    spec=importlib.util.spec_from_file_location('frozen_water_score',tmp_path/'__init__.py',submodule_search_locations=[str(tmp_path)])
    package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
    from importlib import import_module
    module=import_module('frozen_water_score.evaluator')
    assert module.SequenceEvaluator(task()).inspect_conditions()['reason']=='not_sampled'


def test_frozen_cavity_helper_matches_authored_spout_and_wall_regions():
    from scenes.workrooms.living_room.tasks.water_pouring.native_observer import cavity_contains
    points=np.random.default_rng(141).uniform([-.08,-.08,0.],[.08,.08,.12],(12000,3))
    for profile in (assets.KETTLE_INNER,assets.CUP_INNER):
        assert np.array_equal(cavity_contains(points,profile),assets.cavity_contains(points,profile))


def test_native_observer_rejects_nonfinite_mass_before_any_credit():
    t,f=native_fixture();f['particle_sets']['water']['total_mass_kg']=float('nan')
    observer=NativeTaskObserver(t);observer.on_frame(f)
    assert 'mass' in observer.on_observe()['errors'][0]
