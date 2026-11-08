"""Finite, pure geometry calculations on explicit model-supplied numbers.

No Python evaluation, robot/client objects, files, state lookup or actions.
Coordinates are user declarations; these results are not trusted sensor poses,
IK solutions, collision checks or evidence of contact/grasp/task completion.
"""
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from bench.geometry import matrix_quaternion, quaternion_matrix, rotvec_matrix
from bench.protocol.schemas import Contract, Name, Pose, Vec3


class RotationArguments(Contract):
    quaternion_wxyz: tuple[float, float, float, float] | None = Field(
        default=None, description='Normalize a nonzero quaternion; need not already have unit length.')
    rotation_vector_rad: Vec3 | None = Field(default=None, description='Axis times angle, in radians.')
    axis_xyz: Vec3 | None = None
    angle_rad: float | None = None

    @model_validator(mode='after')
    def one_source(self):
        if ((self.axis_xyz is None) != (self.angle_rad is None)
                or sum((self.quaternion_wxyz is not None, self.rotation_vector_rad is not None,
                        self.axis_xyz is not None)) != 1):
            raise ValueError('Provide quaternion, rotation vector, or axis and angle; exactly one form')
        return self


class VectorArguments(Contract):
    operation: Literal['add', 'subtract', 'midpoint', 'scale', 'normalize', 'dot', 'cross', 'distance']
    a: Vec3
    b: Vec3 | None = None
    scale: float | None = None

    @model_validator(mode='after')
    def operands(self):
        binary = self.operation in ('add', 'subtract', 'midpoint', 'dot', 'cross', 'distance')
        if (binary != (self.b is not None)) or ((self.operation == 'scale') != (self.scale is not None)):
            raise ValueError('Binary operations require b; only scale takes a scalar')
        return self


class PoseDeltaArguments(Contract):
    pose: Pose
    translation_m: Vec3 = Field(default_factory=lambda: [0., 0., 0.])
    rotation_delta_rad: Vec3 = Field(default_factory=lambda: [0., 0., 0.])
    coordinates: Literal['parent', 'local'] = Field(default='parent', description=
        'parent translates/left-rotates in pose.frame_id; local translates/right-rotates in the pose local axes.')


class TransformArguments(Contract):
    pose: Pose = Field(description='The explicitly supplied parent-from-local transform, not a pose lookup.')
    points_m: list[Vec3] = Field(min_length=1, max_length=128)
    inverse: bool = False


class AlignmentArguments(Contract):
    local_pair_m: tuple[Vec3, Vec3]
    target_pair_m: tuple[Vec3, Vec3]
    local_reference_xyz: Vec3 = Field(default_factory=lambda: [0., 0., 1.])
    target_reference_xyz: Vec3 = Field(default_factory=lambda: [0., 0., 1.])
    target_frame: Name = 'world'


FUNCTION_INPUTS = {
    'compute_rotation': RotationArguments,
    'compute_vector': VectorArguments,
    'compute_pose_delta': PoseDeltaArguments,
    'compute_transform_points': TransformArguments,
    'compute_frame_alignment': AlignmentArguments,
}


def _unit(value):
    vector = np.asarray(value, dtype=float)
    maximum = float(np.abs(vector).max())
    if maximum == 0.:
        raise ValueError('undefined_zero_vector')
    scaled = vector / maximum
    return scaled / np.linalg.norm(scaled)


def _basis(pair, reference):
    x = _unit(pair[1] - pair[0])
    reference = _unit(reference)
    z = reference - np.dot(reference, x) * x
    if np.linalg.norm(z) < 1.e-10:
        raise ValueError('reference_parallel_to_pair')
    z = _unit(z)
    return np.column_stack((x, np.cross(z, x), z))


def _pose(position, rotation, frame):
    return {'frame_id': frame, 'position_m': np.asarray(position).tolist(),
            'quaternion_wxyz': _unit(matrix_quaternion(rotation)).tolist()}


def _calculate(name, value):
    if name == 'compute_rotation':
        if value.get('quaternion_wxyz') is not None:
            q = _unit(value['quaternion_wxyz'])
            rotation = quaternion_matrix(q)
        else:
            vector = (np.asarray(value['rotation_vector_rad']) if value.get('rotation_vector_rad') is not None
                      else _unit(value['axis_xyz']) * value['angle_rad'])
            rotation = rotvec_matrix(vector)
            q = _unit(matrix_quaternion(rotation))
        return {'quaternion_wxyz': q.tolist(), 'rotation_matrix': rotation.tolist()}
    if name == 'compute_vector':
        a, op = np.asarray(value['a']), value['operation']
        b = np.asarray(value['b']) if value.get('b') is not None else None
        if op == 'add': result = a + b
        elif op == 'subtract': result = a - b
        elif op == 'midpoint': result = a / 2. + b / 2.
        elif op == 'scale': result = a * value['scale']
        elif op == 'normalize': result = _unit(a)
        elif op == 'dot': result = np.dot(a, b)
        elif op == 'cross': result = np.cross(a, b)
        else: result = np.linalg.norm(a - b)
        return {'value': np.asarray(result).tolist()}
    if name in ('compute_pose_delta', 'compute_transform_points'):
        pose = value['pose']
        rotation = quaternion_matrix(_unit(pose['quaternion_wxyz']))
        position = np.asarray(pose['position_m'])
        if name == 'compute_transform_points':
            points = np.asarray(value['points_m'])
            transformed = ((points - position) @ rotation if value['inverse']
                           else points @ rotation.T + position)
            return {'points_m': transformed.tolist(),
                    'output_coordinates': 'pose_local' if value['inverse'] else pose['frame_id']}
        delta = np.asarray(value['translation_m'])
        delta_rotation = rotvec_matrix(value['rotation_delta_rad'])
        if value['coordinates'] == 'local':
            position, rotation = position + rotation @ delta, rotation @ delta_rotation
        else:
            position, rotation = position + delta, delta_rotation @ rotation
        return {'pose': _pose(position, rotation, pose['frame_id'])}
    local, target = np.asarray(value['local_pair_m']), np.asarray(value['target_pair_m'])
    rotation = _basis(target, value['target_reference_xyz']) @ _basis(local, value['local_reference_xyz']).T
    position = target.mean(axis=0) - rotation @ local.mean(axis=0)
    transformed = local @ rotation.T + position
    errors = np.linalg.norm(transformed - target, axis=1)
    return {'pose': _pose(position, rotation, value['target_frame']),
            'transformed_pair_m': transformed.tolist(), 'point_residuals_m': errors.tolist(),
            'pair_length_difference_m': float(np.linalg.norm(local[1] - local[0]) - np.linalg.norm(target[1] - target[0])),
            'rigid_alignment_note': 'A rigid transform cannot change pair spacing; nonzero residuals are not a successful fit.'}


def execute(name, arguments):
    """Revalidate even outside the runner; return only finite JSON calculations."""
    if name not in FUNCTION_INPUTS:
        raise ValueError('Unknown geometry calculation')
    value = FUNCTION_INPUTS[name].model_validate(arguments).model_dump(mode='json')
    # Bounding explicit inputs avoids extreme-angle range reduction and overflow
    # work; it imposes no robot workspace or physical feasibility claim.
    def finite_numbers(node):
        if isinstance(node, dict):
            return all(finite_numbers(v) for v in node.values())
        if isinstance(node, (list, tuple)):
            return all(finite_numbers(v) for v in node)
        return not isinstance(node, (float, int)) or (np.isfinite(node) and abs(node) <= 1.e6)
    if not finite_numbers(value):
        raise ValueError('Geometry numbers must be finite and within the calculation bound')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            result = _calculate(name, value)
        except (FloatingPointError, ValueError) as error:
            reason = str(error)
            return {'defined': False, 'reason': reason if reason in (
                'undefined_zero_vector', 'reference_parallel_to_pair') else 'numerical_domain_error',
                'source': 'explicit_user_geometry_only', 'advances_physics': False}
    return {'defined': True, 'result': result, 'source': 'explicit_user_geometry_only',
            'advances_physics': False, 'ik_checked': False, 'collision_checked': False, 'grasp_checked': False}
