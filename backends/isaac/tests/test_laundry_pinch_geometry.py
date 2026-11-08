"""Adversarial geometric checks; synthetic boxes do not qualify native grasping."""
from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial import ConvexHull

from bench.evaluation.cloth.cloth_robot_collision import SCHEMA, SUPPORT_SCHEMA
from scripts.validation.laundry_pinch_geometry import PinchGeometry, _surface


def fixture():
    names, rows = [], []
    for finger, x in [('thumb', (-.010, -.001)), ('index', (.001, .010))]:
        name = 'right_'+finger+'_elastomer'; names.append(name)
        vertices = np.array([[a,b,c] for a in x for b in [-.008,.008] for c in [-.008,.008]])
        planes = ConvexHull(vertices).equations.tolist()
        piece = {'vertices': vertices.tolist(), 'geometry_representation': SUPPORT_SCHEMA,
                 'conservative_outer_vertices': vertices.tolist(), 'conservative_outer_planes': planes,
                 'conservative_inner_planes': planes}
        rows.append({'body_name': name, 'path': '/SyntheticTest/'+name+'/collider',
                     'status': 'verified_cooking_representation', 'approximation':'convexHull',
                     'mesh_to_body_column_matrix': np.eye(4).tolist(), 'pieces':[piece]})
    export = {'schema':SCHEMA,'status':'complete','issues':[], 'visual_mesh_fallback':False,
              'state_hash_before':'synthetic','state_hash_after':'synthetic',
              'layer_hashes_before':{'synthetic':'test'},'layer_hashes_after':{'synthetic':'test'},
              'colliders':rows}
    poses = np.array([[0,0,0,0,0,0,1.]]*2)
    q = np.array([[0,-.004,-.004],[0,.004,-.004],[0,.004,.004],[0,-.004,.004]])
    tri = np.array([[0,1,2],[0,2,3]])
    return export,names,poses,q,tri


class PinchGeometryTest(unittest.TestCase):
    def test_zero_area_cooked_inner_facets_do_not_produce_nan(self):
        vertices=np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0]],float)
        faces=np.array([[0,1,2],[0,1,3]])
        planes=np.array([[0,0,1,0],[0,0,1,0]],float)
        with np.errstate(all='raise'):
            distance,point,normal=_surface(np.array([.2,.2,.05]),vertices,faces,planes)
        self.assertAlmostEqual(distance,.05)
        self.assertTrue(np.isfinite(point).all() and np.isfinite(normal).all())

    def test_distal_shell_can_contact_when_elastomers_do_not(self):
        export,names,poses,q,tri = fixture()
        shells = deepcopy(export['colliders'])
        for row in shells:
            row['body_name'] = row['body_name'].replace('elastomer','DP')
            row['path'] = row['path'].replace('elastomer','DP')
            names.append(row['body_name'])
        export['colliders'].extend(shells)
        poses = np.vstack([poses,poses.copy()]);poses[:2,2] += .04
        result=PinchGeometry(export,names).diagnose('right',poses,q,tri,q,.0012)
        self.assertFalse(result['eligible'])
        self.assertTrue(result['distal_contact_eligible'])
        passing=[r for r in result['distal_contact_diagnostics'] if r['eligible']]
        self.assertEqual(passing[0]['contact_body_names'],{'thumb':'right_thumb_DP','index':'right_index_DP'})

    def test_two_unconnected_sheets_are_not_multilayer_material(self):
        export,names,poses,q,tri = fixture()
        poses[0,0] -= .002;poses[1,0] += .002
        q=np.vstack([q+[-.002,0,0],q+[.002,0,0]]);tri=np.vstack([tri,tri+4])
        result=PinchGeometry(export,names).diagnose('right',poses,q,tri,q,.0012)
        self.assertFalse(result['eligible'])
        self.assertFalse(result['multilayer_diagnostic']['plausible'])
        self.assertTrue(all(result['multilayer_diagnostic']['per_finger_witnesses'].values()))

    def test_sewn_front_back_reported_separately_without_auto_weld(self):
        export,names,poses,q,tri = fixture()
        poses[0,0] -= .002;poses[1,0] += .002
        q=np.vstack([q+[-.002,0,0],q+[.002,0,0],
                     [[-.002,.05,-.004],[-.002,.05,.004],[.002,.05,-.004],[.002,.05,.004]]])
        tri=np.vstack([tri,tri+4,[[1,8,9],[1,9,2],[5,10,11],[5,11,6],[8,10,11],[8,11,9]]])
        result=PinchGeometry(export,names).diagnose('right',poses,q,tri,q,.0012)
        self.assertFalse(result['eligible'])
        self.assertTrue(result['multilayer_diagnostic']['plausible'])
        self.assertFalse(result['multilayer_diagnostic']['automatic_attachment_eligible'])

    def test_opposing_surfaces_bound_same_connected_patch(self):
        export,names,poses,q,tri = fixture()
        result = PinchGeometry(export,names).diagnose('right',poses,q,tri,q,.0012)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['local_vertex_ids'],[0,1,2,3])
        self.assertAlmostEqual(result['pad_support_gap_along_axis_lower_m'],.002)
        self.assertFalse(result['contact_force_measured'])

    def test_open_hand_and_far_cloth_cannot_enable(self):
        export,names,poses,q,tri = fixture(); geometry = PinchGeometry(export,names)
        poses[1,0] = .040
        self.assertFalse(geometry.diagnose('right',poses,q,tri,q,.0012)['eligible'])
        poses[1,0] = 0; q[:,2] += .050
        self.assertFalse(geometry.diagnose('right',poses,q,tri,q,.0012)['eligible'])

    def test_live_body_translation_is_used(self):
        export,names,poses,q,tri = fixture(); geometry = PinchGeometry(export,names)
        poses[:,:3] += [.4,-.2,.9]
        self.assertFalse(geometry.diagnose('right',poses,q,tri,q,.0012)['eligible'])
        self.assertTrue(geometry.diagnose('right',poses,q+[.4,-.2,.9],tri,q,.0012)['eligible'])

    def test_deep_triangle_interior_blocks_even_when_vertices_outside(self):
        export,names,poses,q,tri = fixture(); rest = q.copy()
        q = np.vstack([q,[[-.005,-.03,-.03],[-.005,.03,-.03],[-.005,0,.03]]])
        tri = np.vstack([tri,[4,5,6]]);rest = np.vstack([rest,q[4:]])
        result = PinchGeometry(export,names).diagnose('right',poses,q,tri,rest,.0012)
        self.assertFalse(result['eligible'])
        self.assertIn('thumb',result['severe_penetration_pads'])

    def test_pad_overlap_cannot_be_grasp_credit(self):
        export,names,poses,q,tri = fixture();poses[1,0] = -.004
        result = PinchGeometry(export,names).diagnose('right',poses,q,tri,q,.0012)
        self.assertFalse(result['eligible'])
        self.assertIn('pad_support_projections_overlap',result['reasons'])

    def test_folded_world_neighborhood_with_far_material_rejected(self):
        export,names,poses,q,tri = fixture();rest = q*20
        result = PinchGeometry(export,names).diagnose('right',poses,q,tri,rest,.0012)
        self.assertFalse(result['eligible'])
        self.assertIn('no_connected_material_patch_within_locality_limits',result['reasons'])

    def test_large_triangle_crossing_gap_is_not_local_patch(self):
        export,names,poses,q,tri = fixture();q *= 20
        result = PinchGeometry(export,names).diagnose('right',poses,q,tri,q,.0012)
        self.assertFalse(result['eligible'])

    def test_same_side_of_both_pads_not_between(self):
        export,names,poses,q,tri = fixture();q[:,0] = -.012
        self.assertFalse(PinchGeometry(export,names).diagnose('right',poses,q,tri,q,.0012)['eligible'])

    def test_pose_and_mesh_inputs_unchanged(self):
        export,names,poses,q,tri = fixture(); before = [x.copy() for x in (poses,q,tri)]
        PinchGeometry(export,names).diagnose('right',poses,q,tri,q,.0012)
        for original,current in zip(before,(poses,q,tri)):np.testing.assert_array_equal(original,current)

    def test_missing_uncertain_or_wrong_identity_fails_closed(self):
        export,names,poses,q,tri = fixture()
        for mutate in [lambda d:d.update(status='incomplete'),lambda d:d.update(visual_mesh_fallback=True),
                       lambda d:d['colliders'].pop(),lambda d:d['colliders'][0]['pieces'][0].pop('geometry_representation')]:
            data=deepcopy(export);mutate(data)
            with self.assertRaises(ValueError):PinchGeometry(data,names)
        geometry=PinchGeometry(export,names);poses[0,6]=.9
        with self.assertRaises(ValueError):geometry.diagnose('right',poses,q,tri,q,.0012)


if __name__ == '__main__':
    unittest.main()
