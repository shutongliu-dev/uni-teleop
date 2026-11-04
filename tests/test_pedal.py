from types import SimpleNamespace
import unittest
import numpy as np
from uni_teleop.devices.pedal import PressEdges
from uni_teleop.mapping.franka.pedal import toggle_pedal
from uni_teleop.mapping.franka.retarget import Retargeter, relative_pose_target

class PedalTests(unittest.TestCase):
    def test_repeat_bounce_and_already_held_do_not_toggle(self):
        edge = PressEdges(61, pressed=True)
        self.assertFalse(edge.feed(1,61,1,0.))
        self.assertFalse(edge.feed(1,61,2,.1))
        self.assertFalse(edge.feed(1,61,0,.2))
        self.assertTrue(edge.feed(1,61,1,.3))
        self.assertFalse(edge.feed(1,61,1,.4))
        edge.feed(1,61,0,.41)
        self.assertFalse(edge.feed(1,61,1,.42))
        edge.feed(1,61,0,.5)
        self.assertFalse(edge.feed(1,30,1,.7))
        self.assertTrue(edge.feed(1,61,1,.8))

    def mapper(self):
        m=Retargeter.__new__(Retargeter)
        m.hand_recovery=None
        m.bindings={'right':'right'}
        m.rotation=np.eye(3)
        m.acknowledged={'joint':.3}
        m.anchors={}
        m.set_active(True)
        m.anchors={'right':object()}
        return m

    def frame(self):
        return SimpleNamespace(host_ns=1_000_000_000, tracking={'focused':True,'controllers':{'right':{'grip':1.}}})

    def test_clutch_reanchors_with_grip_held_and_preserves_view_and_targets(self):
        m=self.mapper();basis=m.rotation.copy();f=self.frame()
        self.assertEqual(toggle_pedal(m,f,now_ns=f.host_ns,view_ready=True),'paused')
        self.assertFalse(m.armed);self.assertEqual(m.anchors,{})
        self.assertEqual(toggle_pedal(m,f,now_ns=f.host_ns,view_ready=True),'resumed_reanchor')
        self.assertTrue(m.armed);self.assertEqual(m.anchors,{})
        self.assertEqual(m.acknowledged,{'joint':.3})
        np.testing.assert_array_equal(m.rotation,basis)
        hand=np.eye(4);hand[:3,3]=[1.,2.,3.]
        robot=np.eye(4);robot[:3,3]=[.2,.3,.9]
        np.testing.assert_allclose(relative_pose_target(hand,hand,robot,m.rotation),robot)

    def test_invalid_tracking_cannot_resume(self):
        for missing,age,ready in ((True,0,True),(False,200_000_000,True),(False,0,False)):
            m=self.mapper();m.set_active(False);f=self.frame()
            result=toggle_pedal(m,None if missing else f,now_ns=f.host_ns+age,view_ready=ready)
            self.assertEqual(result,'resume_rejected_tracking');self.assertFalse(m.armed)
