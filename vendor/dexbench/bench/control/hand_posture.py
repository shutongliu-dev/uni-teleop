"""Explicit robot-only SHARPA hand IK posture prior.

This bounds IK goals, not measured motion, contact or human-grasp quality. The
four constraints are disjoint, so projecting each PIP/DIP pair plus the other
joint boxes gives the Euclidean projection onto their intersection exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def sharpa_finger_pairs(side='right'):
    return {finger: (f'{side}_{urdf}_PIP', f'{side}_{urdf}_DIP')
            for finger, urdf in [('index', 'index'), ('middle', 'middle'), ('ring', 'ring'), ('little', 'pinky')]}


FINGER_PAIRS = sharpa_finger_pairs('right')


@dataclass(frozen=True)
class HandPosturePrior:
    maximum_dip_minus_pip_rad: float = .5
    # SHARPA side prefix of the coupled joints; None is an explicit no-coupling
    # prior for hands without a measured SHARPA-style DIP/PIP relation.
    side: str | None = 'right'

    def __post_init__(self):
        value = self.maximum_dip_minus_pip_rad
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Hand posture coupling must be finite and nonnegative')
        if self.side not in ('right', 'left', None):
            raise ValueError('Hand posture side must be right, left or None')

    @classmethod
    def unconstrained(cls):
        return cls(side=None)

    @property
    def pairs(self):
        return {} if self.side is None else FINGER_PAIRS if self.side == 'right' else sharpa_finger_pairs(self.side)

    def describe(self):
        if self.side is None:
            return {'profile_id': 'unconstrained_v1', 'joint_pairs': [], 'scope': 'requested_finger_goals_only',
                    'notes': 'No posture coupling is applied; IK goals are bounded by URDF limits only.'}
        return {'profile_id':'sharpa_dip_pip_v1' if self.side == 'right' else f'sharpa_dip_pip_{self.side}_v1',
            'max_distal_minus_proximal_rad':self.maximum_dip_minus_pip_rad,
            'joint_pairs':[{'proximal_joint':pip, 'distal_joint':dip} for pip, dip in self.pairs.values()],
            'scope':'requested_finger_goals_only',
            'measured_state_projected':False, 'unselected_fingers':'preserve',
            'grasp_success_guarantee':False,
            'notes':'Robot-only IK prior: DIP-PIP bound applies to requested fingertip/finger posture goals. '
                'Untargeted fingers and timed hold retain their reference. Raw measured joints are not projected. '
                'An initially noncompliant selected finger recovers through continuous joint references; '
                'neither an IK solution nor this profile certifies measured posture, contact or task success.'}

    def violation(self, finger, positions):
        pair = self.pairs.get(finger)
        if pair is None:
            return None
        pip, dip = pair
        excess = float(positions[dip] - positions[pip] - self.maximum_dip_minus_pip_rad)
        return (dip, excess) if excess > 0 else None

    def project(self, finger, positions, lower, upper, names):
        """Project a seed/trial before FK evaluation, never clip a solved IK goal."""
        original = np.asarray(positions, dtype=float)
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if (original.shape != lower.shape or original.shape != upper.shape
                or len(names) != len(original) or not np.isfinite(original).all()
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or (lower > upper).any()):
            raise ValueError('Invalid hand-posture projection input')
        result = np.clip(original, lower, upper)
        pair = self.pairs.get(finger)
        if pair is None:
            return result
        pip, dip = (names.index(name) for name in pair)
        c = self.maximum_dip_minus_pip_rad
        if lower[dip] - upper[pip] > c:
            raise ValueError('URDF bounds and finger coupling have empty intersection')
        if result[dip] - result[pip] > c:
            # At the active plane d=p+c, minimize the distance to the ORIGINAL
            # input (using its box-clipped value here gives the wrong projection).
            lo = max(lower[pip], lower[dip] - c)
            hi = min(upper[pip], upper[dip] - c)
            proximal = float(np.clip((original[pip] + original[dip] - c) / 2., lo, hi))
            distal = proximal + c
            if distal - proximal > c:
                distal = np.nextafter(distal, -np.inf)
            result[pip], result[dip] = proximal, distal
        if ((result < lower).any() or (result > upper).any()
                or result[dip] - result[pip] > c):
            raise ValueError('Hand-posture projection lost feasibility')
        return result
