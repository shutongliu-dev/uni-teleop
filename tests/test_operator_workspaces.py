from copy import deepcopy
import unittest
import numpy as np
from uni_teleop.video.isaac_camera import operator_rear_pose, operator_workspace_pose


class OperatorWorkspaceTests(unittest.TestCase):
    def scene(self, family, task):
        return {'teleop_scene_variant': {'version':'teleop_workbench_v1','family':family,'task':task},
                'robot': {'base_position_m':[.65,0.,.78]},
                'table': {'position_m':[.45,-.95,.8775], 'half_extents_m':[1.4,.4,.0225]},
                'cameras': {'overview':{'eye':[1,2,3],'target':[0,0,0]}, 'wrist':{'offset_xyz_m':[.1,.2,.3]}}}

    def assert_in_both_eyes(self, scene, points):
        before=deepcopy(scene)
        eye,target=operator_rear_pose(scene)
        forward=target-eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        up=np.cross(right,forward)
        for sign in (-1,1):
            delta=np.asarray(points)-(eye+sign*.032*right)
            depth=delta@forward
            self.assertTrue((depth>0).all())
            # 24 mm lens, 36x27 mm aperture, plus framing margin.
            self.assertTrue((np.abs(delta@right/depth)<.75*.9).all())
            self.assertTrue((np.abs(delta@up/depth)<.5625*.9).all())
        self.assertEqual(scene,before)

    def test_cooking_targets_counter_instead_of_unrelated_bar(self):
        scene=self.scene('kitchen','pot_lid_placement')
        self.assertGreater(operator_rear_pose(scene)[1][1],0)
        self.assert_in_both_eyes(scene,[[-.03,.705,.95],[.21,.705,.95],[.6,.85,.92]])

    def test_chess_source_rack_and_board_fit_together(self):
        self.assert_in_both_eyes(self.scene('card_room','task0'),
            [[.14,y,.81] for y in (-.38,-.52,-.66)]+[[.35,-.3,.80],[.95,.3,.80]])

    def test_book_shelf_and_desktop_fit_together(self):
        for task in ('book_retrieval','book_shelving'):
            self.assert_in_both_eyes(self.scene('study',task),
                [[-.985,-.46,.88],[-.985,-.14,1.12],[.43,0,.80],[.34,.32,.80]])

    def test_presets_return_copies_and_unknown_scenes_keep_fallback(self):
        scene=self.scene('laboratory','sample_sorting')
        a=operator_workspace_pose(scene);a[0][:]=99
        self.assertLess(operator_workspace_pose(scene)[0].max(),3)
        self.assertIsNone(operator_workspace_pose(self.scene('kitchen','breakfast_serving')))
        scene.pop('teleop_scene_variant')
        self.assertIsNone(operator_workspace_pose(scene))
