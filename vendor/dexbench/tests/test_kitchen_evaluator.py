"""Native-value logic fixtures; no simulator or robot qualification is inferred."""
from copy import deepcopy
import json
import math
from types import MappingProxyType

import pytest

from scenes.workrooms.kitchen.evaluation import NativeTaskObserver, SequenceEvaluator, native_room_path
from scenes.workrooms.kitchen.native_evidence import fixed_state
from scenes.workrooms.kitchen.rules import (TASK_IDS, GROUPED_INVENTORIES, bind_task_rules, evaluation_configuration,
                                          public_rules)

SURFACE = '/World/Geometry/worktop/Collider'
HAND = '/World/Robot/hand/right_index_DP'


def metadata(task):
    root = '/World/KitchenTask'
    m = {'task_id': task, 'actors': {}, 'objects': [], 'fixtures': {}, 'fixture_metadata': {},
         'goals': {}, 'surface_z_m': .9, 'workspace_bounds_m': [[-.7,-1.5,.6],[1.4,1.4,1.8]],
         'work_surface_paths': [SURFACE]}
    def obj(name, size, p):
        path = root+'/Actors/'+name
        row = {'id': name, 'body_path': path, 'local_bounds_m': [[-size[0]/2,-size[1]/2,0], [size[0]/2,size[1]/2,size[2]]],
               'initial_pose_xyzw': [*p,0,0,0,1]}
        m['actors'][name] = path
        m['objects'].append(row)
        return row
    def fixture(name, pos, bounds, top=.005):
        path = root+'/Fixtures/'+name
        row = {'path':path,'initial_pose_xyzw':[*pos,0,0,0,1], 'interior_bounds_xy_m':bounds,'floor_top_local_z_m':top}
        m['fixtures'][name]=path
        m['fixture_metadata'][name]=row
        return row
    if task == 'breakfast_serving':
        tray=obj('tray',(.34,.24,.02),(.16,-.84,.901))
        tray.update(interior_bounds_xy_m=[[-.164,-.114],[.164,.114]],floor_top_local_z_m=.005)
        plate=obj('plate',(.14,.14,.015),(.51,-.88,.901))
        plate.update(interior_radius_m=.059,interior_bounds_xy_m=[[-.059,-.059],[.059,.059]],floor_top_local_z_m=.005)
        obj('cup',(.064,.064,.08),(.49,-.68,.901))
        obj('bread_0',(.05,.03,.03),(.64,-.72,.901))
        obj('bread_1',(.05,.03,.03),(.64,-.86,.901))
        m['goals'].update(source_region_xy_m=[[-.06,-1.025],[.38,-.61]],
            destination_region_xy_m=[[.66,-1.025],[1.1,-.61]])
    elif task == 'produce_transfer':
        fixture('source_board',(.15,-.9,.9),[[-.21,-.27],[.21,.27]],.012)
        fixture('tomato_tray',(.6,-.9,.9),[[-.12,-.16],[.12,.16]])
        fixture('carrot_tray',(.94,-.9,.9),[[-.12,-.16],[.12,.16]])
        for i in range(3):
            obj('tomato_'+str(i),(.056,.056,.05),(.04,-1.08+.18*i,.913))
            obj('carrot_'+str(i),(.03,.101,.03),(.25,-1.08+.18*i,.913))
    elif task == 'utensil_sorting':
        fixture('source_pad',(.15,-.9,.9),[[-.2,-.27],[.2,.27]],.004)
        tray=fixture('cutlery_tray',(.8,-.9,.9),[[-.17,-.16],[.17,.16]])
        tray['channels']={}
        for index, kind in enumerate(('spoon','fork','knife')):
            center=(index-1)*.1
            tray['channels'][kind]={'interior_bounds_xy_m':[[center-.045,-.13],[center+.045,.13]],'floor_top_local_z_m':.005}
            for i in range(2):
                obj(kind+'_'+str(i),(.03,.16,.008),(.02+.13*index,-.75-.3*i,.905))
    elif task in GROUPED_INVENTORIES:
        fixture('source_mat',(-.15,.65,.9),[[-.3,-.35],[.3,.35]],.004)
        m['goals']['class_to_fixture'] = {}
        m['source_support_by_actor'] = {}
        index = 0
        for group_index, (kind, (target, names)) in enumerate(GROUPED_INVENTORIES[task].items()):
            fixture(target,(.35+.4*group_index,-.8,.9),[[-.17,-.27],[.17,.27]])
            m['goals']['class_to_fixture'][kind] = target
            orientation = 'upright' if kind in ('bottles','seasonings','dishware','drinkware') else 'any'
            for name in names:
                height = .14 if kind == 'bottles' else .025 if kind in ('dishware','cutlery') else .07
                row = obj(name,(.08,.08,height),(-.25+.17*(index%2),.4+.14*(index//2),.9043))
                row.update(category=kind,assigned_fixture=target,placement_orientation=orientation)
                m['source_support_by_actor'][name] = 'source_mat'
                index += 1
    else:
        m['work_surface_paths']=[SURFACE,'/World/Geometry/kitchen_cooktop/Collider']
        fixture('source_mat',(-.115,.9,.9),[[-.3,-.22],[.3,.22]],.003)
        obj('pot',(.25,.18,.1),(-.22,.8,.904))
        obj('lid',(.188,.188,.05),(.11,1.01,.904))
        m['goals'].update(burner_center_xy_m=[.485,.815],burner_support_paths=['/World/Geometry/kitchen_cooktop/Collider'],
                         pot_rim_local_z_m=.1,lid_seat_local_z_m=.002)
    return m


class Replay:
    def __init__(self, ident, *, meta=None):
        self.meta=metadata(ident) if meta is None else meta
        self.task=bind_task_rules({'schema_version':'kitchen-manipulation-task-v1','id':ident,'episode_seed':0,
            'evaluation':evaluation_configuration(ident,self.meta,physics_dt_s=.002),'public_rules':public_rules(ident)})
        self.observer=NativeTaskObserver(self.task)
        self.evaluator=SequenceEvaluator(self.task)
        self.objects={r['id']:fixed_state(r['initial_pose_xyzw']) for r in self.meta['objects']}
        self.contacts=[]
        self.step=0
        self.frame=None
        self.send(baseline=True)
        self.observe()

    def put(self,name,p,q=(1.,0,0,0)):
        self.objects[name]={'position_m':list(p),'quaternion_wxyz':q,'linear_velocity_m_s':[0.,0,0],'angular_velocity_rad_s':[0.,0,0]}

    def pair(self,a,b,impulse=(0,0,.001),separation=-.0001):
        paths={**self.meta['actors'],**self.meta['fixtures']}
        return {'actor0':native_room_path(paths.get(a,a)),'actor1':native_room_path(paths.get(b,b)),'point_count':1,
                'impulse_magnitude_sum_ns':math.sqrt(sum(x*x for x in impulse)),
                'impulse_vector_ns':list(impulse),'min_separation_m':separation}

    def send(self,baseline=False):
        if not baseline:self.step+=1
        t=self.step*.002
        self.frame={'baseline':baseline,'time_s':t,'native_time_s':t+.004,'native_step':self.step+2,
            'physics_dt_s':.002,'objects':deepcopy(self.objects),'contacts':[] if baseline else [{**x,'time_s':t} for x in self.contacts]}
        self.observer.on_frame(MappingProxyType(self.frame))

    def observe(self,**kw):
        self.result=self.evaluator.update({'time_s':self.step*.002,'task_state':self.observer.on_observe(),**kw})
        return self.result

    def run(self,count=220):
        for _ in range(count):
            self.send()
            if self.step%10==0:self.observe()
        return self.result

    def targets(self):
        ident=self.task['id']
        self.contacts=[]
        if ident=='produce_transfer':
            for i in range(3):
                for kind,x in (('tomato',.6),('carrot',.94)):
                    name=kind+'_'+str(i)
                    self.put(name,(x,-1.0+.1*i,.9053))
                    self.contacts.append(self.pair(name,kind+'_tray'))
        elif ident=='utensil_sorting':
            for index,kind in enumerate(('spoon','fork','knife')):
                for i in range(2):
                    name=kind+'_'+str(i)
                    self.put(name,(.7+.1*index+(-.02 if i==0 else .02),-.9,.9053),q=(1,0,0,0) if i==0 else (0,0,0,1))
                    self.contacts.append(self.pair(name,'cutlery_tray'))
        elif ident=='breakfast_serving':self.breakfast(.88,.9003)
        elif ident in GROUPED_INVENTORIES:
            for group_index, (_, (target, names)) in enumerate(GROUPED_INVENTORIES[ident].items()):
                for index,name in enumerate(names):
                    self.put(name,(.35+.4*group_index,-.98+.12*index,.9053))
                    self.contacts.append(self.pair(name,target))
        else:
            self.put('pot',(.485,.815,.9003))
            self.contacts=[self.pair('pot',self.meta['goals']['burner_support_paths'][0])]

    def breakfast(self,x,z,hand=False,gap=False):
        for name,p in {'tray':(x,-.84,z),'plate':(x-.07,-.845,z+.0053),
                       'cup':(x+.09,-.845,z+.0053),'bread_0':(x-.096,-.845,z+.0106),
                       'bread_1':(x-.044,-.845,z+.0106)}.items():self.put(name,p)
        self.contacts=[] if gap else [self.pair(a,b) for a,b in (('plate','tray'),('cup','tray'),('bread_0','plate'),('bread_1','plate'))]
        self.contacts.append(self.pair('tray',HAND if hand else SURFACE))

    def source_ready(self):
        self.breakfast(.16,.9003)
        self.run(180)
        assert self.result['detail']['source_ready'], self.result

    def carry(self):
        self.breakfast(.16,.94,hand=True)
        self.run(10)
        for i in range(1,51):
            self.breakfast(.16+.72*i/50,.94,hand=True)
            self.run(10)
        self.breakfast(.88,.9003)
        return self.run(220)


@pytest.mark.parametrize('ident',TASK_IDS)
def test_unscored_native_reset_and_public_projection(ident):
    from bench.protocol.task_referee import project_task_rules,project_referee_state
    r=Replay(ident)
    assert r.result['reason']=='reset_baseline_not_scored'
    r.run()
    assert not r.result['success'] and not r.result['physical_failure']
    assert r.result['status']=='ongoing'
    public=project_task_rules(r.task)
    assert project_referee_state(r.result,public['public_rules'])['state']=='ongoing'
    json.dumps(r.observer.on_observe(),allow_nan=False)


@pytest.mark.parametrize('ident',['produce_transfer','utensil_sorting'])
def test_all_six_supported_released_stable_then_success(ident):
    r=Replay(ident);r.targets();r.run(200)
    assert not r.result['success']
    r.run(20)
    assert r.result['success'],r.result


def test_utensils_reverse_handle_and_light_same_class_overlap_are_accepted():
    r=Replay('utensil_sorting');r.targets()
    r.put('spoon_1',(.69,-.9,.914),(0,0,0,1))
    r.contacts=[x for x in r.contacts if x['actor0']!=r.meta['actors']['spoon_1']]
    r.contacts.append(r.pair('spoon_1','spoon_0'))
    assert r.run()['success']


@pytest.mark.parametrize('case',['no_contact','side_contact','downward_contact','wrong_class','riding_divider','robot_touch','hover'])
def test_wrong_or_unsupported_sorting_cannot_pass_and_can_retry(case):
    r=Replay('utensil_sorting');r.targets()
    if case in ('no_contact','side_contact','downward_contact'):
        r.contacts=[x for x in r.contacts if x['actor0']!=r.meta['actors']['fork_0']]
        if case!='no_contact':r.contacts.append(r.pair('fork_0','cutlery_tray',impulse=(.001,0,0) if case=='side_contact' else (0,0,-.001)))
    elif case=='wrong_class':r.put('fork_0',(.7,-.9,.9053))
    elif case=='riding_divider':r.put('fork_0',(.85,-.9,.925))
    elif case=='robot_touch':r.contacts.append(r.pair('fork_0',HAND))
    elif case=='hover':r.put('fork_0',(.78,-.9,.935))
    assert not r.run()['success']
    assert r.result['status']=='ongoing'
    r.targets();assert r.run()['success']


def test_sleeping_support_retains_prior_upward_witness_but_pose_change_invalidates_it():
    r=Replay('produce_transfer');r.targets();r.run(20);r.contacts=[]
    assert r.run()['success']
    other=Replay('produce_transfer');other.targets();other.run(20);other.contacts=[]
    other.put('tomato_0',(.61,-1.,.9053))
    assert not other.run()['success']


def test_breakfast_requires_source_loaded_real_carry_and_released_delivery():
    r=Replay('breakfast_serving');r.source_ready()
    assert r.carry()['success'],r.result
    assert [e['kind'] for e in r.observer.events]==['source_loaded','loaded_lift_started','loaded_delivery']


@pytest.mark.parametrize('method',['destination_assembly','slide','empty_tray'])
def test_breakfast_shortcuts_do_not_pass(method):
    r=Replay('breakfast_serving')
    if method=='slide':r.source_ready()
    if method=='empty_tray':
        r.put('tray',(.88,-.84,.94));r.contacts=[r.pair('tray',HAND)];r.run(20)
    r.breakfast(.88,.9003)
    assert not r.run(260)['success']
    assert r.result['status']=='ongoing'


def test_breakfast_brief_contact_gap_and_small_passive_slip_allowed():
    r=Replay('breakfast_serving');r.source_ready()
    r.breakfast(.16,.94,hand=True);r.run(10)
    r.breakfast(.18,.94,hand=True,gap=True)
    r.objects['bread_0']['position_m'][0]+=.002
    r.run(10)
    assert r.result['detail']['carry_active']
    for i in range(1,41):
        r.breakfast(.18+.70*i/40,.94,hand=True);r.run(10)
    r.breakfast(.88,.9003)
    assert r.run()['success'],r.result


def test_breakfast_midway_landing_clears_attempt_but_source_retry_is_allowed():
    r=Replay('breakfast_serving');r.source_ready()
    r.breakfast(.16,.94,hand=True);r.run(10)
    r.breakfast(.5,.9003);r.run(10)
    assert not r.result['detail']['source_ready']
    r.breakfast(.88,.9003);assert not r.run()['success']
    r.source_ready();assert r.carry()['success']


def test_pot_then_lid_with_practical_offset_height_and_yaw_tolerance():
    r=Replay('pot_lid_placement');r.targets();r.run(180)
    assert r.result['detail']['pot_ready'],r.result
    r.put('lid',(.495,.815,1.004),(math.cos(.4),0,0,math.sin(.4)))
    r.contacts.append(r.pair('lid','pot'))
    assert r.run()['success'],r.result


@pytest.mark.parametrize('case',['no_contact','preassembled','large_offset','upside_down'])
def test_pot_lid_invalid_assembly_not_success(case):
    r=Replay('pot_lid_placement');r.targets()
    if case!='preassembled':r.run(180)
    r.put('lid',(.485+(.02 if case=='large_offset' else 0),.815,.9986),q=(0,1,0,0) if case=='upside_down' else (1,0,0,0))
    if case!='no_contact':r.contacts.append(r.pair('lid','pot'))
    assert not r.run(300)['success']
    assert r.result['status']=='ongoing'


def test_preassembled_sleeping_pot_cannot_skip_lid_sequence_and_can_retry():
    r=Replay('pot_lid_placement');r.targets()
    r.put('lid',(.485,.815,.9983))
    r.contacts.append(r.pair('lid','pot'))
    r.run(20)
    r.contacts=[]  # Sleeping pairs disappear; prior support remains valid.
    r.run(400)
    assert not r.result['success'] and not r.result['detail']['pot_ready']
    assert not r.observer.events
    assert r.result['status']=='ongoing'
    # A small clear lift separates the lid; the pot need not be moved again.
    r.put('lid',(.485,.815,1.0203))
    r.contacts=[r.pair('lid',HAND)]
    r.run(20)
    assert r.result['detail']['pot_ready']
    r.put('lid',(.485,.815,.9983))
    r.contacts=[r.pair('lid','pot')]
    assert r.run()['success'],r.result


@pytest.mark.parametrize('fault',['native_gap','wrong_time','missing_body','no_impulse_vector','missing_control','reset_changed'])
def test_bad_evidence_is_unknown_without_invented_physical_failure(fault):
    r=Replay('produce_transfer');r.targets()
    if fault=='native_gap':r.step+=1;r.send()
    elif fault=='missing_control':r.run(10);r.run(10);r.step+=10;r.observe()
    elif fault=='reset_changed':
        r.frame['baseline']=True;r.observer.on_frame(r.frame)
    else:
        r.send();f=deepcopy(r.frame)
        if fault=='wrong_time':f['contacts'][0]['time_s']-=.01
        if fault=='missing_body':f['objects'].pop('tomato_0')
        if fault=='no_impulse_vector':f['contacts'][0].pop('impulse_vector_ns')
        # Submit this malformed frame as the next consecutive frame.
        f['native_step']+=1;f['time_s']+=.002;f['native_time_s']+=.002
        for row in f['contacts']:row['time_s']+=.002
        r.observer.on_frame(f)
    result=r.observe()
    assert result['status']=='unknown' and not result['physical_failure'] and not result['success']


def test_floor_fall_is_physical_failure_and_penetration_is_separate_safety_stop():
    r=Replay('produce_transfer');r.put('tomato_0',(.04,-1.08,.3));r.run(10)
    assert r.result['status']=='task_failure' and r.result['physical_failure']
    r=Replay('produce_transfer');r.targets();r.contacts[0]['min_separation_m']=-.01;r.run(10)
    assert r.result['status']=='safety_stop' and not r.result['physical_failure']


def test_frozen_threshold_or_rule_mutation_rejected():
    r=Replay('produce_transfer');task=deepcopy(r.task)
    task['evaluation']['thresholds']['stable_s']=0
    with pytest.raises(ValueError):NativeTaskObserver(task)
    task=deepcopy(r.task);task['public_rules']['referee']['instruction_zh']='different'
    with pytest.raises(ValueError):SequenceEvaluator(task)


def test_malformed_terminal_field_does_not_turn_unknown_evidence_into_physical_failure():
    r=Replay('produce_transfer')
    for _ in range(10):r.send()
    private=r.observer.on_observe()
    private['terminal_failure']='undeclared_failure'
    result=r.evaluator.update({'time_s':.02,'task_state':private})
    assert result['status']=='unknown'
    assert result['physical_failure'] is False
    assert result['irrecoverable_reason'] is None


def test_actual_hull_extrema_prevent_rotated_round_produce_bbox_false_rejection():
    # Octahedral contact sample of a rounded produce body: after a 45 degree
    # roll its actual lowest vertex is above the imaginary box corner.
    meta=metadata('produce_transfer')
    tomato=next(x for x in meta['objects'] if x['id']=='tomato_0')
    radius=.025
    tomato['local_bounds_m']=[[-radius,-radius,0],[radius,radius,2*radius]]
    tomato['collision_support_points_m']=[[radius,0,radius],[-radius,0,radius],
        [0,radius,radius],[0,-radius,radius],[0,0,0],[0,0,2*radius]]
    r=Replay('produce_transfer',meta=meta);r.targets()
    q=(math.cos(math.pi/8),math.sin(math.pi/8),0,0)
    # Rotation about local origin leaves the true hull's min local-world Z at
    # zero; the rotated bounding box's fictional lowest corner lies -17.7 mm.
    r.put('tomato_0',(.6,-1.,.9053),q)
    assert r.run()['success'],r.result


@pytest.mark.parametrize('ident',TASK_IDS)
def test_goal_only_reset_can_validate_support_but_never_earn_manipulation_credit(ident):
    authored=Replay(ident);authored.targets()
    if ident=='pot_lid_placement':
        authored.put('lid',(.485,.815,.9986))
        authored.contacts.append(authored.pair('lid','pot'))
    meta=deepcopy(authored.meta)
    for row in meta['objects']:
        obj=authored.objects[row['id']]
        q=obj['quaternion_wxyz']
        row['initial_pose_xyzw']=[*obj['position_m'],*q[1:],q[0]]
    diagnostic=Replay(ident,meta=meta)
    diagnostic.contacts=deepcopy(authored.contacts)
    assert diagnostic.run(300)['status']=='ongoing'
    assert diagnostic.result['success'] is False
    assert diagnostic.result['released_stable_s']==0
    native=diagnostic.observer.on_observe()['native_support_by_actor']
    assert all(row['supported'] and row['released_quiet_native_s']>=.4 for row in native.values())


def test_single_native_pose_excursion_resets_quiet_even_when_ack_pose_returns():
    r=Replay('produce_transfer');r.run(220)
    before=r.observer.on_observe()['native_support_by_actor']['tomato_0']['released_quiet_native_s']
    assert before>=.4
    r.objects['tomato_0']['position_m']=list(r.objects['tomato_0']['position_m'])
    r.objects['tomato_0']['position_m'][0]+=.01
    r.send()
    r.objects['tomato_0']['position_m'][0]-=.01
    for _ in range(9):r.send()
    r.observe()
    after=r.observer.on_observe()['native_support_by_actor']['tomato_0']['released_quiet_native_s']
    assert after==pytest.approx(.018)


def _yaw(q, radians):
    c,s=math.cos(radians/2),math.sin(radians/2)
    w,x,y,z=q
    return (c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w)


def _microvibration(r, count=230):
    anchors=deepcopy(r.objects)
    for i in range(count):
        sign=1 if i%2 else -1
        for name,obj in r.objects.items():
            obj['position_m']=list(anchors[name]['position_m'])
            obj['position_m'][0]+=sign*.000005
            obj['quaternion_wxyz']=_yaw(anchors[name]['quaternion_wxyz'], sign*.0001)
            # Raw native tensor pulses remain finite diagnostics; actual
            # pose variation stays tiny and is checked at every native step.
            obj['linear_velocity_m_s']=[sign*.04,0,0]
            obj['angular_velocity_rad_s']=[0,0,sign*3.]
        r.run(1)
    return r.result


@pytest.mark.parametrize('ident',['produce_transfer','utensil_sorting'])
def test_small_native_microvibration_with_tensor_pulses_is_stable(ident):
    r=Replay(ident);r.targets()
    assert _microvibration(r)['success'],r.result
    assert all(x['released_quiet_native_s']>=.4
        for x in r.observer.on_observe()['native_support_by_actor'].values())


def test_breakfast_and_pot_stages_share_rolling_pose_stability():
    breakfast=Replay('breakfast_serving');breakfast.breakfast(.16,.9003)
    _microvibration(breakfast,180)
    assert breakfast.result['detail']['source_ready']
    assert breakfast.carry()['success']
    pot=Replay('pot_lid_placement');pot.targets()
    _microvibration(pot,180)
    assert pot.result['detail']['pot_ready']
    pot.put('lid',(.485,.815,.9983));pot.contacts.append(pot.pair('lid','pot'))
    assert _microvibration(pot)['success'],pot.result


@pytest.mark.parametrize('motion',['continuous_yaw','ack_alias_translation','ack_alias_rotation'])
def test_native_motion_cannot_hide_behind_quiet_ack_endpoints_or_tensor_values(motion):
    r=Replay('produce_transfer');r.targets()
    anchor=deepcopy(r.objects['tomato_0'])
    for _ in range(330):
        t=(r.step+1)*.002
        x=deepcopy(anchor)
        if motion=='continuous_yaw':
            x['quaternion_wxyz']=_yaw(anchor['quaternion_wxyz'],.6*t)
        else:
            excursion=math.sin(2*math.pi*t/.02)
            if motion=='ack_alias_translation':x['position_m'][0]+=.006*excursion
            else:x['quaternion_wxyz']=_yaw(anchor['quaternion_wxyz'],.12*excursion)
        # Tensor velocities are zero deliberately: the pose evidence itself
        # must reject real motion, including 20 ms endpoint aliasing.
        r.objects['tomato_0']=x
        r.run(1)
    assert not r.result['success'] and r.result['status']=='ongoing'
    assert r.observer.quiet_s['tomato_0']<.02
    if motion!='continuous_yaw':
        assert r.objects['tomato_0']['position_m']==pytest.approx(anchor['position_m'])
        assert abs(r.objects['tomato_0']['quaternion_wxyz'][3])<1e-10


@pytest.mark.parametrize('fault',['held_between_acks','missing_native_sample'])
def test_microvibration_does_not_relax_release_or_complete_native_sampling(fault):
    r=Replay('utensil_sorting');r.targets()
    base=deepcopy(r.contacts)
    for _ in range(230):
        r.contacts=deepcopy(base)
        if fault=='held_between_acks' and (r.step+1)%10==5:
            r.contacts.append(r.pair('spoon_0',HAND))
        if fault=='missing_native_sample' and r.step==100:
            r.step+=1
        r.run(1)
    assert not r.result['success']
    assert r.result['status']==('unknown' if fault=='missing_native_sample' else 'ongoing')


def test_rolling_pose_velocity_window_is_public_and_requires_complete_history():
    r=Replay('produce_transfer')
    rules=r.task['public_rules']['referee']
    assert rules['thresholds']['stability_pose_window_s']==.02
    assert any('20 毫秒' in text for text in rules['success_conditions'])
    r.run(9)
    assert all(v==0 for v in r.observer.quiet_s.values())
    r.run(1)
    assert all(v==pytest.approx(.002) for v in r.observer.quiet_s.values())


@pytest.mark.parametrize('ident', GROUPED_INVENTORIES)
def test_eight_grouped_objects_use_full_released_common_window_and_free_yaw(ident):
    r=Replay(ident);r.targets()
    # Broad target regions and any bottle-label/cup-handle yaw are acceptable.
    for index,name in enumerate(r.objects):
        p=list(r.objects[name]['position_m']);p[0]+=.035
        yaw=.24+.07*index
        r.put(name,p,(math.cos(yaw/2),0,0,math.sin(yaw/2)))
    r.run(200)
    assert not r.result['success']
    assert r.run(20)['success'],r.result
    assert r.result['detail']['accepted_count']==8
    public=r.task['public_rules']['referee']
    inventory=' '.join(public['success_conditions'])
    assert all(name in inventory for name in r.objects)
    assert all(target in inventory for target in r.meta['goals']['class_to_fixture'].values())
    assert all(cue in public['instruction_zh'] for cue in ('蓝色','奶油色','木色','源垫','固定','不计分'))


@pytest.mark.parametrize('ident', GROUPED_INVENTORIES)
@pytest.mark.parametrize('fault', ['wrong_class','held','floating','spanning_edge','side_force','inverted_force','upside_down'])
def test_grouped_wrong_or_unsupported_placement_is_recoverable(ident,fault):
    r=Replay(ident);r.targets()
    name=next(iter(r.objects));p=list(r.objects[name]['position_m'])
    target=r.meta['objects'][0]['assigned_fixture']
    if fault=='wrong_class':
        p[0]+=.4
        r.contacts=[x for x in r.contacts if x['actor0']!=r.meta['actors'][name]]
        other=list(r.meta['goals']['class_to_fixture'].values())[1]
        r.contacts.append(r.pair(name,other))
    elif fault=='held':r.contacts.append(r.pair(name,HAND))
    elif fault=='floating':p[2]+=.08
    elif fault=='spanning_edge':p[0]+=.15
    elif fault in ('side_force','inverted_force'):
        r.contacts=[x for x in r.contacts if x['actor0']!=r.meta['actors'][name]]
        r.contacts.append(r.pair(name,target,impulse=(.001,0,0) if fault=='side_force' else (0,0,-.001)))
    r.put(name,p,q=(0,1,0,0) if fault=='upside_down' else (1,0,0,0))
    assert not r.run()['success']
    assert r.result['status']=='ongoing'
    r.targets();assert r.run()['success'],r.result


def test_clearing_accepts_real_nested_support_without_prescribed_stack_order():
    r=Replay('table_clearing');r.targets()
    # The plate's base lies below the parent's rim: exact top-height stacking
    # rules would incorrectly reject this ordinary nesting arrangement.
    r.put('side_plate',(.35,-.98,.917))
    r.put('small_bowl',(.35,-.98,.936))
    stacked={'side_plate','small_bowl'}
    r.contacts=[x for x in r.contacts if x['actor0'] not in {r.meta['actors'][n] for n in stacked}]
    r.contacts += [r.pair('side_plate','dinner_plate'),r.pair('small_bowl','side_plate')]
    assert r.run()['success'],r.result
    parents=r.result['detail']['support_parent_by_actor']
    assert parents['small_bowl']=='side_plate'
    assert parents['side_plate']=='dinner_plate'
    assert parents['dinner_plate']=='dishes_tray'


def test_sealed_dry_goods_can_lie_flat_and_signed_reversed_contact_is_valid():
    r=Replay('pantry_stocking');r.targets()
    r.put('rice_carton',(1.15,-.98,.9453),(math.sqrt(.5),math.sqrt(.5),0,0))
    # Native contact ordering is immaterial; the signed body load must still
    # point upward when the body is actor1 rather than actor0.
    contact=next(x for x in r.contacts if x['actor0']==r.meta['actors']['rice_carton'])
    contact['actor0'],contact['actor1']=contact['actor1'],contact['actor0']
    contact['impulse_vector_ns']=[0,0,-.001]
    assert r.run()['success'],r.result


@pytest.mark.parametrize('fault',['cycle','wrong_group_root','floating_chain','external_dressing'])
def test_grouped_support_graph_requires_a_real_correct_tray_root(fault):
    r=Replay('table_clearing');r.targets()
    r.put('side_plate',(.35,-.98,.917))
    r.contacts=[x for x in r.contacts if x['actor0']!=r.meta['actors']['side_plate']]
    if fault=='cycle':
        r.contacts=[x for x in r.contacts if x['actor0']!=r.meta['actors']['dinner_plate']]
        r.contacts += [r.pair('side_plate','dinner_plate')]
        # Two opposite directed collider records cannot root a floating cycle.
        record=r.pair('dinner_plate','side_plate')
        record['actor0']+='/ColliderA';record['actor1']+='/ColliderB'
        r.contacts.append(record)
    elif fault=='wrong_group_root':
        r.put('mug',(.35,-.98,.9053))
        r.contacts.append(r.pair('side_plate','mug'))
    elif fault=='floating_chain':
        r.put('side_plate',(.35,-.98,1.10))
        r.contacts.append(r.pair('side_plate','dinner_plate'))
    else:r.contacts.append(r.pair('side_plate','/World/KitchenTask/Context/vase'))
    assert not r.run()['success']
    assert r.result['status']=='ongoing'


@pytest.mark.parametrize('ident', GROUPED_INVENTORIES)
def test_grouped_dropped_native_sample_is_unknown_and_held_substep_resets_dwell(ident):
    r=Replay(ident);r.targets()
    base=deepcopy(r.contacts)
    for _ in range(220):
        r.contacts=deepcopy(base)
        if (r.step+1)%10==5:r.contacts.append(r.pair(next(iter(r.objects)),HAND))
        r.run(1)
    assert not r.result['success'] and r.result['status']=='ongoing'
    r.step+=1;r.send()
    assert r.observe()['status']=='unknown'
    assert not r.result['physical_failure']


@pytest.mark.parametrize('fault',['category','target','orientation','unknown_inventory'])
def test_public_grouped_inventory_cannot_diverge_from_frozen_asset(fault):
    m=metadata('pantry_stocking')
    if fault=='category':m['objects'][0]['category']='dry_goods'
    elif fault=='target':m['objects'][0]['assigned_fixture']='dry_goods_bin'
    elif fault=='orientation':m['objects'][0]['placement_orientation']='any'
    else:
        m['objects'][0]['id']='unannounced_object'
        m['actors']['unannounced_object']=m['actors'].pop('olive_oil')
    with pytest.raises(ValueError):evaluation_configuration('pantry_stocking',m)


@pytest.mark.parametrize('ident',TASK_IDS)
def test_budget_exhaustion_is_terminal_incomplete_and_latched(ident):
    from bench.protocol.task_referee import project_referee_state
    r=Replay(ident)
    for _ in range(10):r.send()
    result=r.observe(budget_exhausted=True)
    assert result['status']=='incomplete' and result['terminal'] is True
    assert result['reason']=='budget_exhausted'
    assert not result['success'] and not result['physical_failure']
    assert project_referee_state(result,r.task['public_rules']['referee'])['state']=='ongoing'
    # Further observations cannot turn a closed episode into later success.
    assert r.evaluator.update({})==result


def test_budget_stop_precedes_same_sample_completion():
    r=Replay('pantry_stocking');r.targets();r.run(200)
    assert not r.result['success']
    for _ in range(10):r.send()
    assert r.observer.stable_s>=.4
    result=r.observe(action_count=6001)
    assert result['status']=='incomplete' and result['terminal'] is True
    assert not result['success']


@pytest.mark.parametrize('ident',GROUPED_INVENTORIES)
def test_authored_eight_object_metadata_matches_referee_targets_and_wrong_class_recovery(ident):
    from pxr import Usd
    from scenes.workrooms.kitchen.assets import author_task
    meta=author_task(Usd.Stage.CreateInMemory(),ident,dressing_profile='enriched_v1')
    assert len(meta['actors'])==8 and len(meta['contextual_props'])==12
    assert not set(meta['actors']) & set(meta['contextual_props'])
    r=Replay(ident,meta=meta)
    assert r.run()['status']=='ongoing'
    def authored_targets():
        for name,pose in meta['goals']['body_poses_xyzw'].items():
            r.put(name,pose[:3],q=(pose[6],*pose[3:6]))
        r.contacts=[r.pair(name,support) for name,support in meta['goals']['support_by_actor'].items()]
    authored_targets()
    name=next(iter(r.objects))
    wrong_target=list(meta['goals']['class_to_fixture'].values())[1]
    fixture=meta['fixture_metadata'][wrong_target]
    p=fixture['initial_pose_xyzw'][:3]
    r.put(name,[p[0],p[1],p[2]+fixture['floor_top_local_z_m']+.0003])
    r.contacts=[x for x in r.contacts if x['actor0']!=meta['actors'][name]]+[r.pair(name,wrong_target)]
    assert r.run()['status']=='ongoing'
    authored_targets()
    assert r.run()['success'],r.result
    if ident=='table_clearing':
        assert r.result['detail']['support_parent_by_actor']['side_plate']=='dinner_plate'
    # The same assembled fit witness as a diagnostic reset has support but
    # cannot manufacture robot manipulation credit.
    goal_meta=deepcopy(meta)
    for obj in goal_meta['objects']:
        obj['initial_pose_xyzw']=meta['goals']['body_poses_xyzw'][obj['id']]
    goal=Replay(ident,meta=goal_meta)
    goal.contacts=deepcopy(r.contacts)
    assert goal.run()['status']=='ongoing'
    assert goal.result['released_stable_s']==0.
    assert all(row['supported'] for row in goal.observer.on_observe()['native_support_by_actor'].values())
