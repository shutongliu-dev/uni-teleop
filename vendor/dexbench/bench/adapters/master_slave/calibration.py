"""Saved leader/robot joint correspondence; creating it never reads or writes motors."""
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class JointLimitError(ValueError):
    def __init__(self, joint, target, lower, upper):
        self.details = {'code':'joint_limit', 'joint':joint, 'target_rad':float(target),
                        'lower_rad':float(lower), 'upper_rad':float(upper)}
        super().__init__(f'{joint} target {np.degrees(target):+.1f} deg outside '
                         f'[{np.degrees(lower):+.1f}, {np.degrees(upper):+.1f}] deg')


SevenAngles = Annotated[tuple[FiniteFloat, ...], Field(min_length=7, max_length=7)]


class ArmCalibration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    version: Literal['factr_franka_fixed_v1'] = 'factr_franka_fixed_v1'
    status: Literal['measured', 'provisional']
    source_kind: Literal['hardware', 'synthetic']
    leader_id: str = Field(min_length=1)
    motor_ids: tuple[int, ...]
    robot_joint_names: tuple[str, ...]
    robot_urdf_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    signs: tuple[Literal[-1, 1], ...]
    directions_verified: tuple[bool, ...]
    leader_reference_rad: SevenAngles
    robot_reference_rad: SevenAngles

    @model_validator(mode='after')
    def complete(self):
        if self.motor_ids != tuple(range(1, 9)):
            raise ValueError('Require ordered motor IDs 1..8; no missing-axis substitution')
        if (len(self.robot_joint_names) != 7 or len(set(self.robot_joint_names)) != 7 or
                len(self.signs) != 7 or len(self.directions_verified) != 7 or
                (self.status == 'measured' and not all(self.directions_verified))):
            raise ValueError('Require seven named joints; measured calibration needs individually verified directions')
        return self

    @classmethod
    def from_reference(cls, model, *, leader_id, leader_reference_rad, robot_reference_rad,
                       signs, directions_verified, source_kind, status='measured'):
        calibration = cls(status=status, source_kind=source_kind, leader_id=leader_id,
            motor_ids=tuple(range(1, 9)), robot_joint_names=tuple(model.arm.names),
            robot_urdf_sha256=hashlib.sha256(model.arm_path.read_bytes()).hexdigest(),
            signs=signs, directions_verified=directions_verified,
            leader_reference_rad=leader_reference_rad, robot_reference_rad=robot_reference_rad)
        calibration.validate_robot(model)
        return calibration

    @classmethod
    def load(cls, path):
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path):
        # A new calibration must not silently replace the operator's existing one.
        with Path(path).open('x') as stream:
            stream.write(self.model_dump_json(indent=2)+'\n')

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.model_dump(mode='json'), sort_keys=True,
                                        separators=(',', ':')).encode()).hexdigest()

    def validate_robot(self, model):
        if (tuple(model.arm.names) != self.robot_joint_names or
                hashlib.sha256(model.arm_path.read_bytes()).hexdigest() != self.robot_urdf_sha256):
            raise ValueError('Arm calibration robot names/URDF do not match')
        if any(not j.lower <= q <= j.upper or j.upper-j.lower >= 2*np.pi
               for j, q in zip(model.arm.active, self.robot_reference_rad)):
            raise ValueError('Calibration reference outside limits or ambiguous full-turn joint')

    def targets(self, raw, model):
        """Select the unique encoder-equivalent angle within each original joint range.

        X-series power cycles may change the whole-turn count. This does not change
        the saved zero: ranges narrower than 2*pi admit at most one equivalent pose.
        """
        angle = np.array(self.robot_reference_rad) + np.array(self.signs)*(
            np.asarray(raw[:7])-self.leader_reference_rad)
        requested = angle.copy()
        lower = np.array([j.lower for j in model.arm.active])
        upper = np.array([j.upper for j in model.arm.active])
        angle = np.where((angle >= lower) & (angle <= upper), angle,
                         lower + (angle-lower) % (2*np.pi))
        for i, (q, joint) in enumerate(zip(angle, model.arm.active)):
            if q > joint.upper:
                raise JointLimitError(joint.name, requested[i], joint.lower, joint.upper)
        return angle

    def continuous_targets(self, raw, reference):
        """Map calibrated angles to the nearest turn without imposing robot limits."""
        angle=np.array(self.robot_reference_rad)+np.array(self.signs)*(
            np.asarray(raw[:7])-self.leader_reference_rad)
        angle+=2*np.pi*np.round((np.asarray(reference)-angle)/(2*np.pi))
        return angle

    def practice_targets(self, raw, model, reference):
        """Keep the nearest continuous turn, saturating only affected simulated axes."""
        angle=self.continuous_targets(raw,reference)
        lower=np.array([j.lower for j in model.arm.active])
        upper=np.array([j.upper for j in model.arm.active])
        bounded=np.clip(angle,lower,upper)
        limits=[JointLimitError(j.name,q,j.lower,j.upper).details
                for j,q,b in zip(model.arm.active,angle,bounded) if q!=b]
        return bounded,limits
