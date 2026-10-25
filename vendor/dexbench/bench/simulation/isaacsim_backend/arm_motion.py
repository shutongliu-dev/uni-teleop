"""Explicit simulation-practice override; never alters cached robot assets."""
import numpy as np


FLOAT32_MAX = float(np.finfo(np.float32).max)


def configure_arm_motion(stage, joint_paths, policy):
    if policy is None:
        return []
    if policy != 'unrestricted_practice_v1':
        raise ValueError(f'Unknown arm motion policy: {policy}')
    from pxr import UsdPhysics
    names = [name for name in joint_paths if name.startswith('fr3_')]
    for name in names:
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_paths[name]))
        joint.CreateLowerLimitAttr(-float('inf'))
        joint.CreateUpperLimitAttr(float('inf'))
    return names
