"""Referee regression checks; fixtures are synthetic, never native qualification."""
from copy import deepcopy
import math
import unittest
from scenes.workrooms.living_room.tasks.storage_box import rules, evaluator

ACTORS={n:'/World/LivingTask/'+n for n in ('box','lid','left_catch','right_catch','item')}
INITIAL={'box':[.65,-.12,.78],'lid':[.65,0,.894],'left_catch':[.55,-.267,.903],
         'right_catch':[.75,-.267,.903],'item':[.65,-.12,.809]}
META={'actors':ACTORS,'initial_positions_m':INITIAL,'item_size_m':[.07,.05,.036],
      'tray_floor_path':'/World/LivingTask/Tray/Floor','tray_floor_z_m':.786,
      'tray_inside_xy_m':[[.856,-.224],[1.024,-.016]]}

def task():
    return {'id':'storage_box','public_rules':rules.public_rules(),
            'evaluation':rules.evaluation_configuration(asset_metadata=META)}

def frame(step, *, opened=False, collected=False, hand=None, unlocked=False):
    objects={n:{'position_m':p.copy(),'quaternion_wxyz':[1.,0.,0.,0.],
                'linear_velocity_m_s':[0.,0.,0.],'angular_velocity_rad_s':[0.,0.,0.]}
             for n,p in INITIAL.items()}
    if opened:
        a=math.radians(-60)/2;objects['lid']['quaternion_wxyz']=[math.cos(a),math.sin(a),0.,0.]
    if opened or unlocked:
        a=math.radians(95)/2
        for n in ('left_catch','right_catch'): objects[n]['quaternion_wxyz']=[math.cos(a),0.,0.,math.sin(a)*(1 if n=='left_catch' else -1)]
    if collected: objects['item']['position_m']=[.94,-.12,.804]
    contacts=[]
    def contact(a,b):
        return {'actor0':a,'actor1':b,'time_s':step*.0005,'point_count':1,
                'impulse_vector_ns':[0.,0.,.0004],'impulse_magnitude_sum_ns':.0004,'min_separation_m':0.}
    if collected: contacts.append(contact(ACTORS['item'],META['tray_floor_path']))
    if hand: contacts.append(contact(ACTORS['lid'],'/World/Robot/'+hand+'/finger'))
    return {'time_s':step*.0005,'native_time_s':step*.0005,'native_step':step,'physics_dt_s':.0005,
            'baseline':step==0,'objects':objects,'contacts':contacts,'joints':{},'particle_sets':{}}

class StorageBoxRulesTest(unittest.TestCase):
    def run_cycle(self, *, hand='left_hand',unlocked=False):
        observer=evaluator.NativeTaskObserver(task());score=evaluator.SequenceEvaluator(task());out=None
        for step in range(681):
            f=frame(step,opened=0<step<40,collected=step>=40,hand=hand if step==1 else None,unlocked=unlocked and step>=40)
            observer.on_frame(f)
            if step%40==0:out=score.update({'time_s':f['time_s'],'task_state':observer.on_observe()})
        return observer.on_observe(),out
    def test_initial_state_never_scores(self):
        o=evaluator.NativeTaskObserver(task());s=evaluator.SequenceEvaluator(task());o.on_frame(frame(0))
        self.assertFalse(s.update({'time_s':0.,'task_state':o.on_observe()})['success'])
    def test_either_hand_can_finish(self):
        for hand in ('left_hand','right_hand'):
            state,out=self.run_cycle(hand=hand);self.assertTrue(state['complete']);self.assertTrue(out['success'])
    def test_engineering_motion_without_robot_no_credit(self):
        state,out=self.run_cycle(hand=None);self.assertGreater(state['physical_released_stable_s'],.3);self.assertFalse(out['success'])
    def test_unlatched_lid_not_complete(self):
        state,out=self.run_cycle(unlocked=True);self.assertTrue(state['collected']);self.assertFalse(state['closed']);self.assertFalse(out['success'])
    def test_missing_substep_invalidates(self):
        o=evaluator.NativeTaskObserver(task());o.on_frame(frame(0));o.on_frame(frame(2));self.assertFalse(o.on_observe()['complete'])
    def test_no_support_contact_not_collected(self):
        o=evaluator.NativeTaskObserver(task());o.on_frame(frame(0));f=frame(1,collected=True);f['contacts']=[];o.on_frame(f)
        self.assertFalse(o.on_observe()['collected'])
    def test_deep_contact_is_safety_stop(self):
        o=evaluator.NativeTaskObserver(task());s=evaluator.SequenceEvaluator(task());o.on_frame(frame(0));s.update({'time_s':0.,'task_state':o.on_observe()})
        for step in range(1,41):
            f=frame(step,hand='left_hand');f['contacts'][0]['min_separation_m']=-.004;o.on_frame(f)
        self.assertEqual(s.update({'time_s':.02,'task_state':o.on_observe()})['referee']['state'],'safety_stop')
    def test_public_contract_mutation_rejected(self):
        t=task();t['public_rules']['instruction_zh']='changed'
        with self.assertRaises(ValueError): evaluator.NativeTaskObserver(t)
    def test_budget_ends_episode_and_latches_without_task_failure(self):
        from bench.protocol.task_referee import PublicRefereeState
        o=evaluator.NativeTaskObserver(task());s=evaluator.SequenceEvaluator(task());o.on_frame(frame(0))
        s.update({'time_s':0.,'task_state':o.on_observe()})
        for step in range(1,41):o.on_frame(frame(step))
        result=s.update({'time_s':.02,'task_state':o.on_observe(),'budget_exhausted':True})
        self.assertTrue(result['terminal']);self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['reason'],'budget_exhausted');self.assertFalse(result['success'])
        self.assertEqual(result['referee']['state'],'ongoing');PublicRefereeState.model_validate(result['referee'])
        self.assertEqual(result,s.update({}))

if __name__=='__main__':unittest.main()
