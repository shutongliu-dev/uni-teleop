"""Retarget PICO/SMPL head orientation to the simulated G1 neck."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..neck import NECK_DOF, NECK_JOINT_LIMITS


def neck_joint_targets(smpl_pose: np.ndarray) -> np.ndarray:
    """Return yaw/pitch targets from one or more 21-joint SMPL poses.

    The PICO pose uses SMPL's Y-up local coordinates.  Neck yaw is therefore
    rotation around local Y, while nodding is rotation around local X.  Roll
    is intentionally ignored because the simulated neck has two axes.
    """

    value = np.asarray(smpl_pose, dtype=np.float64)
    if value.shape[-2:] != (21, 3):
        raise ValueError("smpl_pose must end with shape (21, 3)")
    if not np.isfinite(value).all():
        raise ValueError("smpl_pose must contain finite values")

    shape = value.shape[:-2]
    neck = Rotation.from_rotvec(value[..., 11, :].reshape(-1, 3))
    head = Rotation.from_rotvec(value[..., 14, :].reshape(-1, 3))
    yaw, pitch, _ = (neck * head).as_euler("YXZ").T
    target = np.stack((yaw, pitch), axis=-1).reshape(*shape, NECK_DOF)
    limits = np.asarray(NECK_JOINT_LIMITS, dtype=np.float64)
    return np.clip(target, limits[:, 0], limits[:, 1])
