import copy
import hashlib
import unittest
import numpy as np

from scripts.validation.laundry_contact_closure_stop import ContactClosureStop
from scripts.validation.laundry_capture_gate import CaptureGate


class ContactClosureStopTests(unittest.TestCase):
    def setUp(self):
        self.tri=np.array([[0,1,2],[1,3,2]])
        self.rest=np.array([[0,0,0],[.01,0,0],[0,.01,0],[.01,.01,0]])
        self.q=self.rest+[0,0,.9]
        self.bodies=['right_hand_wrist','right_thumb_fingertip','right_index_fingertip']
        self.body=np.array([[0,0,.9,0,0,0,1],[0,0,.9,0,0,0,1],[.01,0,.9,0,0,0,1]])
        self.joints=['right_thumb_IP','right_index_DIP','right_middle_PIP','right_joint1']
        self.actual=np.full(4,.5);self.target=np.full(4,.6)
        self.diag={'side':'right','distal_group_severe_penetration':False,'contact_force_measured':False,
            'surface_pair':{'thumb':'DP','index':'DP'},'multilayer_diagnostic':{'plausible':True,
            'paired_witnesses':[{'thumb':{'triangle_id':0},'index':{'triangle_id':1},'opposing_normal_dot':-.9}]}}
        self.diag['state_identity']={'body_q_float64_sha256':hashlib.sha256(self.body.astype('<f8').tobytes()).hexdigest(),
            'cloth_q_float64_sha256':hashlib.sha256(self.q.astype('<f8').tobytes()).hexdigest()}

    def planner(self,**kw):
        return ContactClosureStop('right',self.tri,self.rest,self.bodies,self.joints,
            {n:(-1.,1.) for n in self.joints},**kw)

    def observe(self,planner,**kw):
        return planner.observe(1.,kw.get('diagnostic',self.diag),self.body,self.actual,
            kw.get('target',self.target),kw.get('open_gap',.06),cloth_q=self.q,
            allowed_material_vertex_ids=kw.get('allowed',[0,1,2,3]))

    def test_stop_targets_are_bounded_partial_preload_not_original_overclosure(self):
        before=copy.deepcopy((self.diag,self.q,self.actual,self.target));r=self.observe(self.planner())
        self.assertTrue(r['stop_requested']);self.assertTrue(r['latched'])
        self.assertEqual(set(r['finger_joint_targets_rad']),set(self.joints[:2]))
        np.testing.assert_allclose(list(r['finger_joint_targets_rad'].values()),[.52,.52])
        self.assertFalse(r['capture_candidate']);self.assertFalse(r['attachment_enabled'])
        self.assertEqual(self.diag,before[0])
        for a,b in zip((self.q,self.actual,self.target),before[1:]):np.testing.assert_array_equal(a,b)

    def test_latched_target_does_not_follow_later_aggressive_command(self):
        p=self.planner();first=self.observe(p);later=self.observe(p,target=np.full(4,.9))
        self.assertEqual(first,later)
        later['finger_joint_targets_rad']['right_thumb_IP']=99
        self.assertAlmostEqual(p.receipt['finger_joint_targets_rad']['right_thumb_IP'],.52)

    def test_hold_command_retains_existing_preload_without_advancing_or_attaching(self):
        p=self.planner(target_policy='hold_command_v2');r=self.observe(p)
        self.assertEqual(r['finger_joint_targets_rad'],dict(zip(self.joints[:2],self.target[:2])))
        self.assertEqual(r,self.observe(p,target=np.full(4,.9)))
        self.assertFalse(r['capture_candidate']);self.assertFalse(r['attachment_enabled'])
        self.assertEqual(r['evidence']['preload_fraction'],1.)
        self.assertTrue(r['evidence']['original_position_commands_held_exactly'])
        g=CaptureGate('right',self.tri,self.rest,self.bodies,self.joints,profile='local_assist_v2')
        for t in np.arange(0,.121,.02):
            result=g.observe(float(t),self.diag,self.body,self.actual,np.zeros(4),
                self.target,.06,cloth_q=self.q)
        self.assertFalse(result['capture_candidate'])
        self.assertIn('finger_target_error',result['reasons'])
        with self.assertRaises(ValueError):self.planner(target_policy='stronger')

    def test_false_geometry_and_incomplete_closure_never_stop(self):
        for kind in ('stale','wrong_side','deep','missing','nonopposing','other_region','not_closing'):
            d=copy.deepcopy(self.diag);kw={'diagnostic':d}
            if kind=='stale':d['state_identity']={}
            if kind=='wrong_side':d['side']='left'
            if kind=='deep':d['distal_group_severe_penetration']=True
            if kind=='missing':d['multilayer_diagnostic']['plausible']=False
            if kind=='nonopposing':d['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot']=.3
            if kind=='other_region':kw['allowed']=[0,1,2]
            if kind=='not_closing':kw['open_gap']=.012
            p=self.planner();self.assertFalse(self.observe(p,**kw)['stop_requested'],kind);self.assertFalse(p.latched)

    def test_only_explicit_original_actuated_joints_and_original_limits(self):
        p=self.planner(actuated_joint_names=['right_thumb_IP','right_middle_PIP'])
        self.assertEqual(set(self.observe(p)['finger_joint_targets_rad']),{'right_thumb_IP'})
        p=self.planner()
        with self.assertRaises(ValueError):self.observe(p,target=np.full(4,2.))
        self.assertFalse(p.latched)

    def test_stop_does_not_bypass_actual_speed_or_hundred_ms_gate(self):
        r=self.observe(self.planner());target=self.target.copy()
        for n,v in r['finger_joint_targets_rad'].items():target[self.joints.index(n)]=v
        g=CaptureGate('right',self.tri,self.rest,self.bodies,self.joints,profile='local_assist_v2')
        a=g.observe(0.,self.diag,self.body,self.actual,np.full(4,.2),target,.06,cloth_q=self.q)
        self.assertFalse(a['capture_candidate']);self.assertIn('fingers_not_stopped',a['reasons'])
        for t in np.arange(.02,.101,.02):
            a=g.observe(float(t),self.diag,self.body,self.actual,np.zeros(4),target,.06,cloth_q=self.q)
            self.assertFalse(a['capture_candidate'])
        a=g.observe(.12,self.diag,self.body,self.actual,np.zeros(4),target,.06,cloth_q=self.q)
        self.assertTrue(a['capture_candidate'])

    def test_explicit_bridge_stops_commands_but_still_needs_real_stable_capture(self):
        self.tri=np.array([[0,1,2],[1,3,2],[2,3,4]])
        self.rest=np.vstack([self.rest,[0,.02,0]])
        self.q=self.rest+[0,0,.9]
        self.diag['state_identity']['cloth_q_float64_sha256']=hashlib.sha256(self.q.astype('<f8').tobytes()).hexdigest()
        self.diag['multilayer_diagnostic']['paired_witnesses'][0]['index']['triangle_id']=2
        self.assertFalse(self.observe(self.planner(),allowed=range(5))['stop_requested'])
        p=self.planner(target_policy='hold_command_v2',geometry_profile='adjacent_patch_v3')
        r=self.observe(p,allowed=range(5))
        self.assertTrue(r['stop_requested']);self.assertEqual(r['schema'],'laundry_contact_closure_stop.v3')
        self.assertEqual(r['evidence']['vertex_ids'],list(range(5)))
        self.assertEqual(r['evidence']['material_topology']['bridge_triangle_id'],1)
        self.assertFalse(r['attachment_enabled']);self.assertFalse(r['capture_candidate'])
        self.assertEqual(r['finger_joint_targets_rad'],dict(zip(self.joints[:2],self.target[:2])))
        g=CaptureGate('right',self.tri,self.rest,self.bodies,self.joints,profile='adjacent_patch_v3')
        for t in np.arange(0,.121,.02):
            bad=g.observe(float(t),self.diag,self.body,self.actual,np.zeros(4),self.target,.06,cloth_q=self.q)
        self.assertFalse(bad['capture_candidate']);self.assertIn('finger_target_error',bad['reasons'])
        for t in np.arange(.2,.301,.02):
            good=g.observe(float(t),self.diag,self.body,self.target,np.zeros(4),self.target,.06,cloth_q=self.q)
            if t<.3-1e-9:self.assertFalse(good['capture_candidate'])
        self.assertTrue(good['capture_candidate'])

    def test_bridge_does_not_accept_ambiguous_remote_unseen_or_deep_material(self):
        original=copy.deepcopy((self.tri,self.rest,self.q,self.diag))
        for kind in ('missing','ambiguous','remote_rest','remote_world','unseen','deep','stale'):
            self.tri=np.array([[0,1,2],[1,3,2],[2,3,4]])
            self.rest=np.vstack([original[1],[0,.02,0]]);self.q=self.rest+[0,0,.9]
            self.diag=copy.deepcopy(original[3])
            self.diag['multilayer_diagnostic']['paired_witnesses'][0]['index']['triangle_id']=2
            allowed=list(range(5))
            if kind=='missing':self.tri[1]=[0,1,4]
            if kind=='ambiguous':self.tri=np.vstack([self.tri,[1,2,3]])
            if kind=='remote_rest':self.rest[3,0]=.1
            if kind=='remote_world':self.q[3,0]=.1
            if kind=='unseen':allowed.remove(3)
            if kind=='deep':self.diag['distal_group_severe_penetration']=True
            self.diag['state_identity']['cloth_q_float64_sha256']=hashlib.sha256(self.q.astype('<f8').tobytes()).hexdigest()
            if kind=='stale':self.diag['state_identity']={}
            p=self.planner(target_policy='hold_command_v2',geometry_profile='adjacent_patch_v3')
            self.assertFalse(self.observe(p,allowed=allowed)['stop_requested'],kind)
            self.assertFalse(p.latched)


if __name__=='__main__':unittest.main()
