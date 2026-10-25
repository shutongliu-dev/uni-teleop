"""Simulator-independent neck joint conventions."""

NECK_JOINT_NAMES = ("neck_yaw_joint", "neck_pitch_joint")
NECK_JOINT_LIMITS = ((-1.2, 1.2), (-0.45, 0.55))
NECK_DOF = len(NECK_JOINT_NAMES)
