import unittest
from unittest.mock import patch

import numpy as np

from scripts.validation.laundry_captured_towel_tasks import (
    validate_scene,towel_picks,captured_arc,phase_edges,observed_pinch_offset,prepare_phase)
from scripts.validation.laundry_captured_garment_fold import independent_cloth_streams


class CapturedTowelTests(unittest.TestCase):
    def fixture(self,name):
        x,y=np.meshgrid(np.linspace(-.3,.3,61),np.linspace(-.2,.2,41))
        rest=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        task={'id':name,'asset_metadata':{'physical_thickness_m':.004,'table_top_z_m':.9,
            'reset_profile':'towel_partial_edge_fold_v1','cloths':[{'id':'towel',
                'rest_vertices_m':rest.tolist()}]}}
        q=rest+[.8,-.55,.902]
        return task,rest,q

    def test_flatten_only_top_flap_free_edge_and_original_heading(self):
        task,r,q=self.fixture('towel_flattening');theta=.18
        R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        q=r.copy();q[:,0]=-.18+abs(r[:,0]+.18);q[:,2]=.902+np.where(r[:,0]<-.18,.016,0.)
        q[:,:2]=q[:,:2]@R.T+[.56,-.53];before=q.copy()
        picks=towel_picks(task,q)
        self.assertEqual([p['seed_vertex_ids'] for p in picks],[[0],[2440]])
        self.assertTrue(all(np.all(r[p['allowed_material_vertex_ids'],0]<-.18) for p in picks))
        ids=[0,1,61,62];arc=captured_arc(task,{},q,ids)
        np.testing.assert_allclose(np.array(arc['controls_m'])[-1,:2],r[ids,:2].mean(0)@R.T+[.56,-.53])
        self.assertAlmostEqual(arc['controls_m'][-1][2],.927)
        np.testing.assert_array_equal(q,before)

    def test_flatten_alignment_is_separate_recapture_and_only_moves_local_goal(self):
        task,r,_=self.fixture('towel_flattening');task['asset_metadata']['target_center_xy_m']=[.65,-.55]
        theta=.18;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        q=r.copy();q[:,:2]=q[:,:2]@R.T+[.56,-.53];q[:,2]=.902;before=q.copy()
        self.assertEqual(len(phase_edges(task)),2)
        pick=towel_picks(task,q,index=1)[0]
        self.assertEqual(pick['seed_vertex_ids'],[60])
        np.testing.assert_allclose(pick['inward_world_xy'],R@np.array([-1.,1.])/np.sqrt(2))
        ids=[59,60,120,121];arc=captured_arc(task,{'phase_index':1},q,ids)
        np.testing.assert_allclose(arc['controls_m'][-1][:2],r[ids,:2].mean(0)+[.65,-.55])
        with self.assertRaises(ValueError):captured_arc(task,{'phase_index':1},q,[0,1,61,62])
        np.testing.assert_array_equal(q,before)

    def test_fold_only_positive_x_corners_about_observed_point_eight(self):
        task,r,q=self.fixture('towel_folding');picks=towel_picks(task,q)
        self.assertEqual([p['seed_vertex_ids'] for p in picks],[[60],[2500]])
        ids=[59,60,120,121];arc=captured_arc(task,{},q,ids)
        self.assertAlmostEqual(arc['observed_hinge_x_m'],.8)
        self.assertAlmostEqual(arc['controls_m'][-1][0],.515)
        with self.assertRaises(ValueError):captured_arc(task,{},q,[0,1,61,62])

    def test_explicit_edge_offsets_use_fresh_rotated_material_and_leave_capture_region(self):
        task,r,q=self.fixture('towel_folding')
        theta=.37;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        q[:,:2]=r[:,:2]@R.T+[.8,-.55];before=q.copy()
        pick=towel_picks(task,q)[0];normal=R@np.array([-1.,1.])/np.sqrt(2)
        for offset in (0.,-.003,.010,-.010):
            changed=observed_pinch_offset(task,q,pick,offset)
            np.testing.assert_allclose(changed['inward_world_xy'],normal,atol=1.e-12)
            np.testing.assert_allclose(changed['pinch_tcp_xy_m'],q[60,:2]+offset*normal)
            self.assertEqual(changed['seed_vertex_ids'],pick['seed_vertex_ids'])
            self.assertEqual(changed['allowed_material_vertex_ids'],pick['allowed_material_vertex_ids'])
        np.testing.assert_array_equal(q,before)

    def test_offset_is_per_hand_opt_in_and_old_default_plan_stays_exact(self):
        task,r,q=self.fixture('towel_folding');scene={'cloths':[{'id':'towel'}]}
        with patch('scripts.validation.laundry_captured_towel_tasks.pinch_plan',return_value={}):
            legacy=prepare_phase(None,task,scene,q,0,{})
            changed=prepare_phase(None,task,scene,q,0,{'phases':{'0':{'hands':{
                'right':{'pinch_inward_offset_m':-.003}}}}})
        left=lambda p:next(e for e in p['edges'] if e['side']=='left')
        right=lambda p:next(e for e in p['edges'] if e['side']=='right')
        self.assertEqual(left(legacy),left(changed))
        self.assertNotIn('pinch_inward_offset_m',right(legacy))
        np.testing.assert_allclose(right(legacy)['pinch_tcp_xy_m'],q[60,:2]+.006*np.array([-1,1])/np.sqrt(2))
        np.testing.assert_allclose(right(changed)['pinch_tcp_xy_m'],q[60,:2]-.003*np.array([-1,1])/np.sqrt(2))

    def test_invalid_offset_or_collapsed_observation_cannot_author_a_plan(self):
        task,r,q=self.fixture('towel_folding');pick=towel_picks(task,q)[0]
        for value in (True,None,'0',float('nan'),float('inf'),-.01001,.01001):
            with self.assertRaises(ValueError):observed_pinch_offset(task,q,pick,value)
        collapsed=q.copy();collapsed[58]=collapsed[60]
        with self.assertRaises(ValueError):observed_pinch_offset(task,collapsed,pick,0.)

    def test_relative_yaw_uses_new_observed_edge_after_unfolding_not_first_phase_heading(self):
        task,r,q=self.fixture('towel_flattening');scene={'cloths':[{'id':'towel'}]}
        opts={'phases':{'1':{'hands':{'right':{'yaw_offset_deg':-15.,'pinch_inward_offset_m':-.003}}}}}
        headings=[]
        for theta in (.18,.40):
            R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            current=q.copy();current[:,:2]=r[:,:2]@R.T+[.56,-.53]
            with patch('scripts.validation.laundry_captured_towel_tasks.pinch_plan',return_value={}):
                phase=prepare_phase(None,task,scene,current,1,opts)
            row=next(e for e in phase['edges'] if e['side']=='right')
            self.assertEqual(row['contact_plan']['inside_finger'],'thumb')
            yaw=row['contact_plan']['yaw_definition']['commanded_yaw_deg']
            self.assertAlmostEqual(yaw,(225.+np.rad2deg(theta)-15.)%360.)
            headings.append(yaw)
        self.assertNotEqual(*headings)
        for invalid in ({'yaw_offset_deg':31.},{'yaw_offset_deg':float('nan')},
                        {'yaw_offset_deg':True},{'yaw_offset_deg':0.,'yaw_deg':220.}):
            with self.assertRaises(ValueError):
                prepare_phase(None,task,scene,q,1,{'phases':{'1':{'hands':{'right':invalid}}}})

    def test_stack_uses_connected_crease_and_actual_independent_base(self):
        from scenes.workrooms.laundry_room.assets import folded_vertices
        task,r,_=self.fixture('towel_stacking');q=folded_vertices(r,(.30,-.55),lower_z=.902,bend_radius_m=.004)
        base=folded_vertices(r,(.95,-.55),lower_z=.902,bend_radius_m=.004);before=base.copy()
        picks=towel_picks(task,q)
        self.assertEqual([p['seed_vertex_ids'] for p in picks],[[30],[2470]])
        self.assertTrue(all(np.max(abs(r[p['allowed_material_vertex_ids'],0]))<=.025 for p in picks))
        ids=[29,30,90,91];arc=captured_arc(task,{},q,ids,all_cloth_q={'towel':q,'towel_base':base})
        np.testing.assert_allclose(arc['observed_translation_xy_m'],[.65,0.],atol=1.e-12)
        self.assertTrue(arc['base_state_used']);self.assertFalse(arc['base_controlled'])
        np.testing.assert_array_equal(base,before)
        with self.assertRaises(ValueError):captured_arc(task,{},q,ids)
        with self.assertRaises(ValueError):captured_arc(task,{},q,[0,1],all_cloth_q={'towel_base':base})

    def test_stack_scene_requires_two_original_independent_cloths(self):
        task,_,_=self.fixture('towel_stacking')
        with self.assertRaises(ValueError):validate_scene(task,{'cloths':[{'id':'towel'}]})
        validate_scene(task,{'cloths':[{'id':'towel'},{'id':'towel_base'}]})

    def test_stream_records_uncontrolled_base_q_qd_and_zero_mask(self):
        moving=np.ones((4,3));base=np.full((7,3),2.);velocity=np.full((7,3),.2)
        result=independent_cloth_streams('towel',{'towel':moving,'towel_base':base},
            {'towel':moving*0,'towel_base':velocity})
        np.testing.assert_array_equal(result['cloth_q_towel_base'],base)
        np.testing.assert_array_equal(result['cloth_qd_towel_base'],velocity)
        np.testing.assert_array_equal(result['control_mask_towel_base'],np.zeros(7,np.uint8))
        self.assertNotIn('cloth_q_towel',result)
        with self.assertRaises(ValueError):independent_cloth_streams('towel',{'towel':moving,'towel_base':base},{'towel':moving*0})


if __name__=='__main__':unittest.main()
