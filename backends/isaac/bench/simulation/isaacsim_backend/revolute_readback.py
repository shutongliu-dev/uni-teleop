"""Passive hinge metadata and coordinates derived from actual rigid states.

Angles are principal values in radians, not accumulated turns, actuator targets,
constraint forces or task lock state. Import never authors a joint drive.
"""
import math

import numpy as np


def passive_revolute_metadata(prim):
    from pxr import UsdPhysics

    joint = UsdPhysics.RevoluteJoint(prim)
    axis = str(joint.GetAxisAttr().Get())
    limits = [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]
    if (axis not in ('X', 'Y', 'Z') or joint.GetJointEnabledAttr().Get() is False
            or any(v is None or not math.isfinite(v) for v in limits)
            or limits[0] >= limits[1]):
        raise ValueError('Passive revolute requires an enabled joint, XYZ axis and finite ordered limits')
    drives = [schema for schema in prim.GetAppliedSchemas() if schema.startswith('PhysicsDriveAPI:')]
    result = {'axis': axis, 'limits_deg': [float(v) for v in limits]}
    if drives:
        if drives != ['PhysicsDriveAPI:angular']:
            raise ValueError('Passive revolute permits only explicit angular damping')
        drive = UsdPhysics.DriveAPI(prim, 'angular')
        stiffness, damping = drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get()
        position, velocity = drive.GetTargetPositionAttr().Get(), drive.GetTargetVelocityAttr().Get()
        maximum = drive.GetMaxForceAttr().Get()
        if (drive.GetTypeAttr().Get() != 'force' or stiffness != 0. or position != 0. or velocity != 0.
                or damping is None or not math.isfinite(damping) or damping < 0.
                or maximum is None or math.isnan(maximum) or maximum <= 0.):
            raise ValueError('Passive revolute damping forbids stiffness, moving targets and invalid coefficients')
        result['angular_damping'] = {'type': 'force', 'damping_per_degree': float(damping),
            'stiffness': 0., 'target_position_deg': 0., 'target_velocity_deg_s': 0.,
            'maximum_force': None if math.isinf(maximum) else float(maximum),
            'maximum_force_unlimited': math.isinf(maximum)}
    return result


def revolute_coordinate(relative_rotation, axis, relative_angular_velocity=None):
    """Angle and exact derivative of one joint-frame transverse direction.

    Relative angular velocity, when present, is expressed in joint0. Retain
    off-axis constraint errors; do not project body states onto an ideal hinge.
    """
    rotation = np.asarray(relative_rotation, dtype=float)
    if axis not in ('X', 'Y', 'Z') or rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError('Invalid native revolute rotation or axis')
    i = 'XYZ'.index(axis)
    j, k = (i+1) % 3, (i+2) % 3
    cosine, sine = float(rotation[j, j]), float(rotation[k, j])
    denominator = cosine*cosine+sine*sine
    if denominator < 1.e-12:
        raise ValueError('Revolute transverse direction is singular; joint constraint has no readable angle')
    angle = math.atan2(sine, cosine)
    if relative_angular_velocity is None:
        return angle
    angular = np.asarray(relative_angular_velocity, dtype=float)
    if angular.shape != (3,) or not np.isfinite(angular).all():
        raise ValueError('Invalid native revolute angular velocity')
    derivative = np.cross(angular, rotation[:, j])
    velocity = (cosine*float(derivative[k])-sine*float(derivative[j]))/denominator
    return angle, velocity
