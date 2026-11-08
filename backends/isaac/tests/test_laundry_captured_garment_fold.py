import unittest
import sys
from types import SimpleNamespace

import numpy as np

from scripts.validation.laundry_captured_garment_fold import (
    phase_edges,edge_pick,captured_arc,bezier_position,planned_duration,actual_ik_frames,close_native_with_status)
from scripts.validation.laundry_captured_garment_fold import capture_batches,retreat_translation,capture_lift_arc,capture_approach
from scripts.validation.laundry_captured_garment_fold import finger_axis_exit
from scripts.validation.laundry_captured_garment_fold import selected_phase_edges,prepare_phase
from unittest.mock import patch


class CapturedFoldTests(unittest.TestCase):
    def test_exit_uses_actual_distal_axis_for_both_hands_and_rejects_downward_motion(self):
        for side in ('left','right'):
            names=[side+'_index_DP',side+'_index_fingertip']
            q=np.array([[.6,-.3,.94,0,0,0,1],[.575,-.302,.937,0,0,0,1]])
            row=finger_axis_exit(q,names,side);delta=np.asarray(row['translation_m'])
            self.assertAlmostEqual(np.linalg.norm(delta),.05)
            self.assertGreater(delta[0],0);self.assertGreater(delta[2],0)
            q[1,2]=.95
            with self.assertRaises(ValueError):finger_axis_exit(q,names,side)
        self.assertAlmostEqual(planned_duration(3,{'retreat_profile':'finger_axis_exit_v2'},2.),
                               planned_duration(3,{},2.)+3.)

    def test_native_close_preserves_failure_context(self):
        observed=[];env=SimpleNamespace(close=lambda:observed.append(sys.exc_info()[0]))
        with self.assertRaises(RuntimeError):close_native_with_status(env,1)
        self.assertEqual(observed,[RuntimeError])
        close_native_with_status(env,0)
        self.assertEqual(observed,[RuntimeError,None])

    def test_actual_ik_frame_is_public_flange_not_attachment_wrist(self):
        body=np.array([[1.,2.,3.,0.,0.,0.,1.],[1.,2.,3.0005,0.,0.,0.,1.]])
        result=actual_ik_frames(body,['right_hand_flange','right_hand_wrist'],
            {'right':SimpleNamespace(frame_id='right_hand_flange')},['right'])
        np.testing.assert_array_equal(result['right'][:3,3],[1.,2.,3.])

    def test_retreat_uses_real_installation_side(self):
        flange=np.eye(4);flange[:3,3]=[.75,-.55,1.05]
        for y,sign in [(.15,1),(-1.15,-1)]:
            base=np.eye(4);base[:3,3]=[.75,y,.9]
            delta=retreat_translation(SimpleNamespace(arm_base=base),flange,1.)
            np.testing.assert_allclose(delta,[0,sign*.05,.06])

    def test_explicit_hover_preserves_default_and_original_arm_inventory(self):
        plan={'arm_ik':[{'joint_positions_rad':{'right_joint1':.3,'right_joint2':-.1}}]}
        self.assertEqual(capture_approach(plan,{})['offset_m'],[0.,0.,.1])
        setting={'hover_offset_m':[0.,.12,.12],
                 'hover_arm_joint_positions_rad':{'right_joint1':.4,'right_joint2':-.2}}
        self.assertEqual(capture_approach(plan,setting)['arm_joint_positions_rad'],setting['hover_arm_joint_positions_rad'])
        for changed in ({'hover_offset_m':[0,0,-.01]}, {'hover_offset_m':[0,.3,.1]},
                        {'hover_arm_joint_positions_rad':{'left_joint1':.4}}, {'hover_offset_m':[0,0,float('nan')]}):
            with self.assertRaises(ValueError):capture_approach(plan,changed)

    def task(self):
        points=np.array([[-.1,-.01,.901],[-.1,.01,.901],[-.09,-.01,.901],[-.09,.01,.901],
                         [.1,-.01,.902],[.1,.01,.902],[.11,-.01,.902],[.11,.01,.902]])
        return {'id':'trousers_folding','asset_metadata':{'design':{},'cloths':[{
            'rest_vertices_m':points.tolist(),'unfolded_reference_vertices_m':points.tolist(),
            'physical_thickness_m':.0012,'boundary_features':{'left_cuff':[0,1,2,3],
                'right_cuff':[4,5,6,7],'waistband':[0,1,4,5]}}]}},points

    def test_no_interior_hold_request(self):
        task,_=self.task();rows=phase_edges(task)
        self.assertEqual([v['feature'] for v in rows[0]],['waistband','left_cuff'])
        self.assertEqual([v['feature'] for v in rows[1]],['paired_cuffs'])

    def test_explicit_one_edge_capture_keeps_material_identity_and_partial_scope(self):
        task,q=self.task();q[:,1]-=.8
        task['asset_metadata']['design']['fold_sequence_axis_hinge_sign_radius_m']=[[0,0.,-1,.007]]
        scene={'table':{'position_m':[0,0,.8],'half_extents_m':[1,1,.1]}}
        opts={'phases':{'0':{'edge_indices':[1],'sides':['right']}}}
        with patch('scripts.validation.laundry_captured_garment_fold.pinch_plan',return_value={}):
            result=prepare_phase(None,task,scene,q,0,opts)
        self.assertEqual(len(result['edges']),1)
        self.assertEqual(result['edges'][0]['feature'],'left_cuff')
        self.assertEqual(result['edges'][0]['edge_index'],1)
        self.assertEqual(result['edges'][0]['side'],'right')
        self.assertFalse(result['all_phase_edges_selected'])
        self.assertEqual(selected_phase_edges(phase_edges(task)[0],{})[1],[0,1])
        for invalid in ([],[True],[2],[-1],[1,1],'right'):
            with self.assertRaises(ValueError):selected_phase_edges(phase_edges(task)[0],{'edge_indices':invalid})

    def test_real_boundary_seed_and_local_material_only(self):
        task,q=self.task();before=q.copy()
        result=edge_pick(task,q,{'feature':'left_cuff','inward_xy':[0.,1.]})
        np.testing.assert_allclose(np.array(result['pinch_tcp_xy_m'])-np.array(result['edge_world_m'][:2]),[0,.006])
        self.assertTrue(set(result['seed_vertex_ids']).issubset({0,1,2,3}))
        self.assertTrue(set(result['allowed_material_vertex_ids']).issubset({0,1,2,3}))
        np.testing.assert_array_equal(q,before)

    def test_material_target_selects_one_level_not_four_nearby_nodes(self):
        task,q=self.task();r=np.array(task['asset_metadata']['cloths'][0]['rest_vertices_m'])
        r[:4,0]=[-.154,-.154,-.143,-.143];q[:4]=r[:4]+[.8,-.5,0.]
        task['asset_metadata']['cloths'][0]['rest_vertices_m']=r.tolist()
        result=edge_pick(task,q,{'feature':'left_cuff','material_x_m':-.143,'inward_xy':[0.,1.]})
        self.assertEqual(result['seed_vertex_ids'],[2,3])
        self.assertAlmostEqual(result['edge_material_m'][0],-.143)

    def test_quarter_yaw_rotates_edge_direction_without_rotating_fold_axes(self):
        task,q=self.task();task['asset_metadata']['source_yaw_quarter_turns']=1
        result=edge_pick(task,q,{'feature':'left_cuff','inward_xy':[0.,1.]})
        np.testing.assert_allclose(result['inward_world_xy'],[-1.,0.],atol=1.e-12)
        arc=captured_arc(task,{'reference_fold':[0,0.,-1,.007]},q,[0,1])
        self.assertAlmostEqual(arc['controls_m'][-1][0],.097)

    def test_corner_uses_existing_native_height_without_modification(self):
        task,q=self.task();q[1,2]=.907;before=q.copy()
        result=edge_pick(task,q,{'feature':'left_cuff','inward_xy':[1.,1.],
                                 'material_x_corner':'minimum'})
        self.assertEqual(result['natural_corner_vertex_id'],1)
        self.assertAlmostEqual(result['edge_world_m'][2],.907)
        np.testing.assert_array_equal(q,before)

    def test_paired_cuff_regions_preserve_each_material_fragment(self):
        task,q=self.task();q[4:,:2]=q[:4,:2]
        result=edge_pick(task,q,{'feature':'paired_cuffs','inward_xy':[0.,1.]})
        self.assertTrue(set(result['seed_vertex_ids']).issubset(result['allowed_material_vertex_ids']))

    def test_sleeve_corner_selects_material_y_and_actual_upper_layer(self):
        task,q=self.task();q[2,2]=.906;before=q.copy()
        result=edge_pick(task,q,{'feature':'left_cuff','inward_xy':[1.,1.],
                                 'material_y_corner':'minimum'})
        self.assertEqual(result['seed_vertex_ids'],[0,2])
        self.assertEqual(result['natural_corner_vertex_id'],2)
        np.testing.assert_array_equal(result['edge_world_m'],q[2])
        np.testing.assert_array_equal(q,before)
        with self.assertRaises(ValueError):
            edge_pick(task,q,{'feature':'left_cuff','inward_xy':[1.,1.],
                'material_x_corner':'minimum','material_y_corner':'minimum'})

    def test_native_capture_controls_release_above_actual_support(self):
        task,q=self.task();before=q.copy()
        arc=captured_arc(task,{'reference_fold':[0,0.,-1,.007]},q,[0,1])
        c=np.array(arc['controls_m'])
        self.assertAlmostEqual(c[-1,0],.097)
        self.assertAlmostEqual(c[-1,2],.927)
        self.assertAlmostEqual(arc['release_height_m'],.025)
        np.testing.assert_allclose(bezier_position(c,0),c[0]);np.testing.assert_allclose(bezier_position(c,1),c[2])
        self.assertGreater(bezier_position(c,.5)[2],c[:,2][[0,2]].max())
        np.testing.assert_array_equal(q,before)

    def test_release_cannot_drive_gripper_into_folded_layers(self):
        task,q=self.task()
        for gap in [0.,.01,.05]:
            with self.assertRaises(ValueError):captured_arc(task,{'reference_fold':[0,0.,-1,.007]},q,[0,1],release_height=gap)

    def test_lift_probe_moves_actual_capture_only_five_cm_vertically(self):
        _,q=self.task();before=q.copy();arc=capture_lift_arc(q,[0,1,2,3])
        np.testing.assert_allclose(np.asarray(arc['controls_m'])[-1]-arc['controls_m'][0],[0,0,.05])
        self.assertEqual(arc['kind'],'capture_lift_probe')
        np.testing.assert_array_equal(q,before)

    def test_all_allocated_steps_match_explicit_phase_schedule(self):
        self.assertEqual(planned_duration(1,{},2.),19.)
        options={'phases':{'0':{'hands':{'left':{'capture_rise_m':.002},'right':{'capture_rise_m':.002}}}}}
        self.assertEqual(planned_duration(1,options,2.),20.)
        self.assertEqual(planned_duration(3,{},.5),50.5)

    def test_sequential_capture_requires_every_side_once_and_allocates_real_time(self):
        phase={'phase_index':0,'edges':[{'side':'left'},{'side':'right'}]}
        self.assertEqual(capture_batches(phase,{}),[['left','right']])
        opts={'phases':{'0':{'capture_order':['right','left']}}}
        self.assertEqual(capture_batches(phase,opts),[['right'],['left']])
        self.assertEqual(planned_duration(1,opts,2.),25.5)
        opts['phases']['0']['hands']={'right':{'capture_rise_m':.002},'left':{'capture_rise_m':.002}}
        self.assertEqual(planned_duration(1,opts,2.),27.5)
        for order in ([],['right'],['right','right'],['left','extra']):
            opts['phases']['0']['capture_order']=order
            with self.assertRaises(ValueError):capture_batches(phase,opts)

    def test_unsupported_region_or_task_fails(self):
        task,q=self.task();task['id']='towel_stacking'
        with self.assertRaises(ValueError):phase_edges(task)
        task,_=self.task();q[4:,2]=1.2
        with self.assertRaises(ValueError):captured_arc(task,{'reference_fold':[0,0.,-1,.007]},q,[0,1])


if __name__=='__main__':unittest.main()
