"""Only the explicitly selected simulated arm loses its native limits."""
import math

from pxr import Usd, UsdPhysics

from bench.simulation.isaacsim_backend.arm_motion import configure_arm_motion


def test_native_stage_override_preserves_finger_limits_and_default_scene():
    stage=Usd.Stage.CreateInMemory()
    paths={n:'/World/'+n for n in ['fr3_joint2','right_index_MCP_FE']}
    for path in paths.values():
        joint=UsdPhysics.RevoluteJoint.Define(stage,path)
        joint.CreateLowerLimitAttr(-90.)
        joint.CreateUpperLimitAttr(90.)
    assert configure_arm_motion(stage,paths,None)==[]
    assert UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(paths['fr3_joint2'])).GetUpperLimitAttr().Get()==90.
    assert configure_arm_motion(stage,paths,'unrestricted_practice_v1')==['fr3_joint2']
    arm=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(paths['fr3_joint2']))
    assert arm.GetLowerLimitAttr().Get()==-math.inf
    assert arm.GetUpperLimitAttr().Get()==math.inf
    hand=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(paths['right_index_MCP_FE']))
    assert hand.GetLowerLimitAttr().Get()==-90. and hand.GetUpperLimitAttr().Get()==90.
