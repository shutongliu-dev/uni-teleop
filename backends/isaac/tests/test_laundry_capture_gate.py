import copy
import hashlib
import unittest

import numpy as np

from scripts.validation.laundry_capture_gate import CaptureGate
from scripts.validation.laundry_contact_assist import capture_constraint_vertices


class CaptureGateTests(unittest.TestCase):
    def setUp(self):
        self.tri = np.array([[0, 1, 2], [1, 3, 2]])
        self.rest = np.array([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [.01, .01, 0]])
        self.names = ['right_hand_wrist', 'right_thumb_fingertip', 'right_index_fingertip']
        self.joints = ['right_thumb_IP', 'right_index_DIP']
        self.body = np.array([[0, 0, .9, 0, 0, 0, 1],
                              [0, 0, .9, 0, 0, 0, 1], [.01, 0, .9, 0, 0, 0, 1]])
        self.q = self.rest + [0, 0, .9]
        self.diag = {'side': 'right', 'distal_group_severe_penetration': False,
            'contact_force_measured': False, 'surface_pair': {'thumb': 'elastomer', 'index': 'DP'},
            'multilayer_diagnostic': {'plausible': True, 'paired_witnesses': [{
                'thumb': {'triangle_id': 0}, 'index': {'triangle_id': 1},
                'opposing_normal_dot': -.9}]}, 'distal_contact_diagnostics': []}

    def gate(self, tri=None, rest=None):
        return CaptureGate('right', self.tri if tri is None else tri,
                           self.rest if rest is None else rest, self.names, self.joints)

    def observe(self, gate, time, **kw):
        body, cloth = kw.get('body', self.body), kw.get('cloth', self.q)
        diag = copy.deepcopy(kw.get('diag', self.diag))
        diag['state_identity'] = {
            'body_q_float64_sha256': hashlib.sha256(np.asarray(body,dtype='<f8').tobytes()).hexdigest(),
            'cloth_q_float64_sha256': hashlib.sha256(np.asarray(cloth,dtype='<f8').tobytes()).hexdigest()}
        if kw.get('stale'): diag['state_identity']['body_q_float64_sha256'] = 'old'
        return gate.observe(time, diag, body,
            kw.get('joint', [.50, .50]), kw.get('speed', [0., 0.]),
            kw.get('target', [.51, .51]), kw.get('open_gap', .045),
            cloth_q=cloth)

    def test_stable_shared_edge_capture_only(self):
        g = self.gate()
        for t in np.arange(0, .101, .02): r = self.observe(g, float(t))
        self.assertTrue(r['capture_candidate'])
        self.assertEqual(r['vertex_ids'], [0, 1, 2, 3])
        self.assertFalse(r['automatic_attachment_eligible'])
        self.assertFalse(r['friction_grasp_qualified'])
        self.assertEqual(r['evidence']['sample_count'], 6)

    def test_shared_edge_aid_uses_only_actual_witnessed_common_nodes(self):
        g=self.gate()
        for t in np.arange(0,.101,.02):r=self.observe(g,float(t))
        self.assertEqual(capture_constraint_vertices(r,self.tri,'captured_patch'),[0,1,2,3])
        self.assertEqual(capture_constraint_vertices(r,self.tri,'shared_edge'),[1,2])
        for changed in ({**r,'capture_candidate':False},{**r,'vertex_ids':[0,1,3]}):
            with self.assertRaises(ValueError):capture_constraint_vertices(changed,self.tri,'shared_edge')
        with self.assertRaises(ValueError):
            capture_constraint_vertices(r,np.array([[0,1,2],[3,4,5]]),'shared_edge')

    def test_exactly_one_hundred_ms_native_sequence(self):
        g = self.gate()
        for t in np.arange(6.9, 7.001, .002): r = self.observe(g, float(t))
        self.assertTrue(r['capture_candidate'])

    def test_explicit_local_aid_profile_allows_small_loaded_settling_only(self):
        outcomes=[]
        for profile in ('strict_v1','local_assist_v2'):
            g=CaptureGate('right',self.tri,self.rest,self.names,self.joints,profile=profile)
            for t in np.arange(0,.101,.02):
                r=self.observe(g,float(t),cloth=self.q+[0.,0.,-.015*t],target=[.535,.535])
            outcomes.append(r['capture_candidate'])
            self.assertFalse(r['friction_grasp_qualified'])
        self.assertEqual(outcomes,[False,True])
        for failure in ('no_witness','excess_slip','excess_error','moving_fingers'):
            g=CaptureGate('right',self.tri,self.rest,self.names,self.joints,profile='local_assist_v2')
            for t in np.arange(0,.101,.02):
                kw={}
                if failure=='no_witness':
                    d=copy.deepcopy(self.diag);d['multilayer_diagnostic']['plausible']=False;kw['diag']=d
                if failure=='excess_slip':kw['cloth']=self.q+[0.,0.,-.021*t]
                if failure=='excess_error':kw['target']=[.541,.541]
                if failure=='moving_fingers':kw['speed']=[.11,0.]
                r=self.observe(g,float(t),**kw)
            self.assertFalse(r['capture_candidate'],failure)

    def test_loaded_profile_changes_only_target_error_and_keeps_all_other_gates(self):
        base=CaptureGate('right',self.tri,self.rest,self.names,self.joints,profile='adjacent_patch_v3')
        loaded=CaptureGate('right',self.tri,self.rest,self.names,self.joints,profile='loaded_adjacent_patch_v5')
        self.assertEqual({**base.thresholds,'maximum_joint_error_rad':.08},loaded.thresholds)
        for t in np.arange(0,.101,.02):
            old=self.observe(base,float(t),target=[.566,.51])
            new=self.observe(loaded,float(t),target=[.566,.51])
            if t<.099:self.assertFalse(new['capture_candidate'])
        self.assertEqual(old['reasons'],['finger_target_error'])
        self.assertTrue(new['capture_candidate'])
        self.assertEqual(new['schema'],'laundry_capture_candidate.v5')
        self.assertFalse(new['contact_force_measured'])
        self.assertFalse(new['automatic_attachment_eligible'])
        self.assertFalse(new['friction_grasp_qualified'])
        for failure in ('no_witness','normal','deep','target','speed','drift','joint_range','no_closure'):
            gate=CaptureGate('right',self.tri,self.rest,self.names,self.joints,profile='loaded_adjacent_patch_v5')
            for t in np.arange(0,.101,.02):
                kw={'target':[.566,.51]};diag=copy.deepcopy(self.diag)
                if failure=='no_witness':diag['multilayer_diagnostic']['plausible']=False
                if failure=='normal':diag['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot']=-.4
                if failure=='deep':diag['distal_group_severe_penetration']=True
                if failure=='target':kw['target']=[.581,.51]
                if failure=='speed':kw['speed']=[.101,0.]
                if failure=='drift':kw['cloth']=self.q+[0.,0.,-.021*t]
                if failure=='joint_range':kw['joint']=[.5+.06*t,.5]
                if failure=='no_closure':kw['open_gap']=.012
                r=self.observe(gate,float(t),diag=diag,**kw)
            self.assertFalse(r['capture_candidate'],failure)

    def test_missing_sample_resets(self):
        g = self.gate()
        for t in [0, .02, .04, .081, .101, .121]: r = self.observe(g, t)
        self.assertFalse(r['capture_candidate'])
        self.assertAlmostEqual(r['continuous_duration_s'], .04)

    def test_stale_native_diagnostic_rejected(self):
        r = self.observe(self.gate(),0,stale=True)
        self.assertIn('diagnostic_does_not_match_actual_native_state',r['reasons'])

    def test_same_world_points_but_moving_hand_are_slip(self):
        g = self.gate()
        for t in np.arange(0, .101, .02):
            body = self.body.copy(); body[:, 2] += t*.02
            r = self.observe(g, float(t), body=body)
        self.assertIn('material_sliding_in_actual_hand_frame', r['reasons'])

    def test_retained_material_follows_actual_hand(self):
        g = self.gate()
        for t in np.arange(0, .101, .02):
            delta = np.array([0, 0, t*.02]);body = self.body.copy();body[:, :3] += delta
            r = self.observe(g, float(t), body=body, cloth=self.q+delta)
        self.assertTrue(r['capture_candidate'])

    def test_no_closure_or_stopped_fingers_rejected(self):
        for kw, reason in [({'open_gap': .012}, 'insufficient_measured_tip_closure_from_episode_open_baseline'),
                           ({'speed': [.11, 0]}, 'fingers_not_stopped'),
                           ({'target': [.6, .5]}, 'finger_target_error')]:
            r = self.observe(self.gate(), 0, **kw)
            self.assertIn(reason, r['reasons'])

    def test_moving_commands_cannot_capture(self):
        g = self.gate()
        for t in np.arange(0, .101, .02):
            r = self.observe(g, float(t), target=[.51+t*.08, .51])
        self.assertIn('fingers_or_commands_not_stable', r['reasons'])

    def test_disconnected_material_and_large_rest_patch_rejected(self):
        tri = np.array([[0, 1, 2], [3, 4, 5]])
        rest = np.vstack([self.rest[:3],self.rest[:3]])
        r = self.observe(self.gate(tri, rest), 0, cloth=rest+[0, 0, .9])
        self.assertIn('no_shared_edge_local_two_finger_material_witness', r['reasons'])
        r = self.observe(self.gate(rest=self.rest*4), 0)
        self.assertFalse(r['capture_candidate'])

    def test_changed_material_or_surface_pair_resets(self):
        g = self.gate()
        for t in [0, .02, .04, .06, .08]: self.observe(g, t)
        diag = copy.deepcopy(self.diag);diag['surface_pair']['thumb'] = 'DP'
        r = self.observe(g, .10, diag=diag)
        self.assertEqual(r['reasons'], ['material_or_surface_pair_changed'])

    def test_wrong_side_deep_penetration_or_nonopposition_rejected(self):
        for field,value in [('side','left'),('distal_group_severe_penetration',True)]:
            d=copy.deepcopy(self.diag);d[field]=value
            self.assertFalse(self.observe(self.gate(),0,diag=d)['capture_candidate'])
        d=copy.deepcopy(self.diag);d['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot']=.1
        self.assertFalse(self.observe(self.gate(),0,diag=d)['capture_candidate'])

    def test_observation_does_not_mutate_sources(self):
        body=self.body.copy();q=self.q.copy();diag=copy.deepcopy(self.diag)
        self.observe(self.gate(),0)
        np.testing.assert_array_equal(body,self.body);np.testing.assert_array_equal(q,self.q)
        self.assertEqual(diag,self.diag)

    def test_explicit_adjacent_profile_requires_one_real_bridge_and_full_dwell(self):
        tri=np.array([[0,1,2],[1,3,2],[2,3,4]])
        rest=np.vstack([self.rest,[0,.02,0]])
        diag=copy.deepcopy(self.diag);diag['multilayer_diagnostic']['paired_witnesses'][0]['index']['triangle_id']=2
        outcomes=[]
        for profile in ('strict_v1','local_assist_v2','adjacent_patch_v3'):
            g=CaptureGate('right',tri,rest,self.names,self.joints,profile=profile)
            for t in np.arange(0,.101,.02):
                r=self.observe(g,float(t),diag=diag,cloth=rest+[0,0,.9])
                if t<.1-1.e-9:self.assertFalse(r['capture_candidate'])
            outcomes.append(r['capture_candidate'])
        self.assertEqual(outcomes,[False,False,True])
        self.assertEqual(r['vertex_ids'],[0,1,2,3,4])
        self.assertEqual(r['evidence']['material_topology'],{'relation':'single_bridge_triangle',
            'bridge_triangle_id':1,'support_triangle_ids':[0,1,2]})
        self.assertEqual(capture_constraint_vertices(r,tri,'captured_patch'),[0,1,2,3,4])
        with self.assertRaises(ValueError):capture_constraint_vertices(r,tri,'shared_edge')
        missing=np.array([[0,1,2],[2,3,4]])
        diag['multilayer_diagnostic']['paired_witnesses'][0]['index']['triangle_id']=1
        g=CaptureGate('right',missing,rest,self.names,self.joints,profile='adjacent_patch_v3')
        self.assertFalse(self.observe(g,0.,diag=diag,cloth=rest+[0,0,.9])['capture_candidate'])

    def test_adjacent_profile_cannot_bridge_large_or_deeply_penetrating_material(self):
        tri=np.array([[0,1,2],[1,3,2],[2,3,4]]);rest=np.vstack([self.rest,[0,.02,0]])
        d=copy.deepcopy(self.diag);d['multilayer_diagnostic']['paired_witnesses'][0]['index']['triangle_id']=2
        for material,cloth,deep in ((rest*4,rest+[0,0,.9],False),(rest,rest*4+[0,0,.9],False),(rest,rest+[0,0,.9],True)):
            d['distal_group_severe_penetration']=deep
            g=CaptureGate('right',tri,material,self.names,self.joints,profile='adjacent_patch_v3')
            self.assertFalse(self.observe(g,0.,diag=d,cloth=cloth)['capture_candidate'])


if __name__ == '__main__':
    unittest.main()
