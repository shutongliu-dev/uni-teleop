import unittest
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
from uni_teleop.devices.manus.native_process import normalize
from uni_teleop.mapping.pico_manus import PicoManusRetargeter


class NativeManusTests(unittest.TestCase):
    def row(self):
        return dict(glove_id=1, sequence=2, capture_time_ns=1234, source_timestamp='42', nodes=[
            dict(node_id='0', parent_id=None, semantic='Hand/Invalid', valid=True,
                 pose=dict(position=[0,0,0], quaternion_wxyz=[1.001,0,0,0]))])

    def test_normalizes_only_small_quaternion_drift_preserves_time_and_input(self):
        row=self.row();before=deepcopy(row)
        packet=normalize(row,glove_id=1,calibration_id='official/test',stream='test',clock='clock')
        self.assertEqual(row,before)
        self.assertEqual(packet['capture_time_ns'],1234)
        self.assertEqual(packet['nodes'][0]['pose']['quaternion_wxyz'],[1,0,0,0])
        self.assertEqual(packet['device']['hand'],'right')

    def test_rejects_identity_and_invalid_orientation(self):
        with self.assertRaises(ValueError):
            normalize(self.row(),glove_id=2,calibration_id='test',stream='test',clock='clock')
        row=self.row();row['nodes'][0]['pose']['quaternion_wxyz']=[0,0,0,0]
        with self.assertRaises(ValueError):
            normalize(row,glove_id=1,calibration_id='test',stream='test',clock='clock')

    def test_composite_propagates_view_and_timing_to_actual_controllers(self):
        mapper=PicoManusRetargeter.__new__(PicoManusRetargeter)
        mapper.arm=SimpleNamespace(rotation=np.eye(3),control_dt=.02)
        mapper.hand=SimpleNamespace(control_dt=.02)
        rotation=np.diag([-1,-1,1])
        mapper.rotation=rotation;mapper.control_dt=.04
        np.testing.assert_array_equal(mapper.arm.rotation,rotation)
        self.assertEqual(mapper.arm.control_dt,.04)
        self.assertEqual(mapper.hand.control_dt,.04)
