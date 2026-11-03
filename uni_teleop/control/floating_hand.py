"""World-frame six-axis palm impedance, used only in simulation."""
import numpy as np
from scipy.spatial.transform import Rotation


def wrist_wrench(position, quaternion, velocity, target_position, target_quaternion):
    error = (Rotation.from_quat(target_quaternion) * Rotation.from_quat(quaternion).inv()).as_rotvec()
    force = 600. * (np.asarray(target_position) - position) - 40. * velocity[:3]
    torque = 8. * error - .6 * velocity[3:]
    return np.clip(force, -60., 60.), np.clip(torque, -8., 8.)
