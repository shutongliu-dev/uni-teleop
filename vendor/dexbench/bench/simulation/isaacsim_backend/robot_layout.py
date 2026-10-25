"""Static robot installation poses shared by scene authoring, reset and FK.

``robot.arm_base_poses`` is an explicit dual-FR3 layout override. Each rigid
pose is relative to the common robot base frame, in metres and wxyz. Omission
retains the assembly registry's original side-by-side offsets. These poses
never modify the referenced robot assets or add/remove any robot geometry.
"""
from __future__ import annotations

import numpy as np

from .robot_registry import get_hand_spec

BIMANUAL_ID = 'dual_fr3_sharpa_w01'
SIDES = ('left', 'right')


def checked_pose(position, quaternion, label):
    position = np.asarray(position, dtype=float)
    quaternion = np.asarray(quaternion, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f'{label} requires a finite position_m of length 3')
    if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
            or abs(float(quaternion @ quaternion) - 1.) > 1.e-5):
        raise ValueError(f'{label} requires a finite unit quaternion_wxyz')
    quaternion = quaternion / np.linalg.norm(quaternion)
    return {'position_m': position.tolist(), 'quaternion_wxyz': quaternion.tolist()}


def robot_base_pose(robot):
    return checked_pose(robot.get('base_position_m', [0., 0., .77]),
                        robot.get('base_quaternion_wxyz', [1., 0., 0., 0.]), 'Robot base')


def arm_base_poses(robot):
    """Validated local poses in stable native articulation order (left, right)."""
    hand_id = robot.get('hand_id', 'sharpa_w01_right')
    from .robot_instances import ROBOT_ID, checked_instances
    if hand_id == ROBOT_ID:
        return {name: {'position_m': item['base_position_m'],
                       'quaternion_wxyz': item['base_quaternion_wxyz']}
                for name, item in checked_instances(robot).items()}
    override = robot.get('arm_base_poses')
    if hand_id != BIMANUAL_ID:
        if override is not None:
            raise ValueError('arm_base_poses requires the dual FR3 + SHARPA robot')
        return None
    arms = get_hand_spec(hand_id, robot.get('assembly_version'))['arms']
    if override is None:
        return {side: checked_pose(arms[side]['base_offset_m'], [1., 0., 0., 0.], side)
                for side in SIDES}
    if not isinstance(override, dict) or set(override) != set(SIDES):
        raise ValueError('arm_base_poses must contain exactly left and right')
    result = {}
    for side in SIDES:
        pose = override[side]
        if not isinstance(pose, dict) or set(pose) != {'position_m', 'quaternion_wxyz'}:
            raise ValueError(f'arm_base_poses.{side} requires position_m and quaternion_wxyz only')
        result[side] = checked_pose(pose['position_m'], pose['quaternion_wxyz'], side)
    return result


def _compose(parent, child):
    w, x, y, z = parent['quaternion_wxyz']
    rotation = np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                         [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                         [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])
    a, b, c, d = child['quaternion_wxyz']
    return {'position_m': (np.asarray(parent['position_m']) + rotation @ child['position_m']).tolist(),
            'quaternion_wxyz': [w*a - x*b - y*c - z*d, w*b + x*a + y*d - z*c,
                                w*c - x*d + y*a + z*b, w*d + x*c - y*b + z*a]}


def world_base_poses(robot):
    """One root pose per native articulation, using the same scene/FK layout."""
    base = robot_base_pose(robot)
    arms = arm_base_poses(robot)
    return {side: _compose(base, pose) for side, pose in arms.items()} if arms else {'single': base}


def reference_robot(stage, robot_prim, robot_usd, robot):
    """Reference original robot USDs and place their root Xforms, reset only."""
    from pxr import Gf, UsdGeom

    base, arms = robot_base_pose(robot), arm_base_poses(robot)
    if isinstance(robot_usd, dict) != bool(arms):
        raise ValueError('Robot USD articulation count does not match robot layout')
    if arms and set(robot_usd) != set(arms):
        raise ValueError('Robot USD references must match the declared articulation inventory')

    def place(prim, pose):
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*pose['position_m']))
        q = np.asarray(pose['quaternion_wxyz'], dtype=float)
        q /= np.linalg.norm(q)
        xform.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:].tolist())))

    place(robot_prim, base)
    if arms:
        for side, pose in arms.items():
            member = UsdGeom.Xform.Define(stage, str(robot_prim.GetPath()) + '/' + articulation_scope(robot, side)).GetPrim()
            member.GetReferences().AddReference(str(robot_usd[side]))
            place(member, pose)
    else:
        robot_prim.GetReferences().AddReference(str(robot_usd))


def articulation_scope(robot, side):
    """Stable namespace in the stage; historical Left/Right paths stay unchanged."""
    from .robot_instances import ROBOT_ID, checked_namespace
    return checked_namespace(side) if robot.get('hand_id') == ROBOT_ID else side.capitalize()
