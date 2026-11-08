"""Deterministic candidate parameters, independent of robot and global RNG state.

These are authoring candidates, not evidence of valid geometry or dynamics.
Family authors record their actual applied values separately and reject illegal
layouts. ``attempt`` changes only layout; appearance and physics stay paired.
The ranges below are this project's choices, not ranges from EmbodiedLab.
"""
from copy import deepcopy
import hashlib
import json
import math
import re

VERSION = 'parameterized_physics_v1'
FAMILIES = ('mechanical_assembly', 'study', 'kitchen', 'laboratory',
            'living_room', 'laundry_room')
CONFIGURATIONS = ('single_arm', 'dual_same_side', 'dual_opposed')
_IDENTIFIER = re.compile(r'^[a-z][a-z0-9_]{0,95}$')


def canonical_hash(value):
    """Hash JSON values with a fixed encoding; NaN/Infinity are forbidden."""
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')
    return hashlib.sha256(data).hexdigest()


def _inputs(family, task_id, configuration_id, seed, attempt):
    if family not in FAMILIES or configuration_id not in CONFIGURATIONS:
        raise ValueError('Unsupported family or installation')
    if not isinstance(task_id, str) or not _IDENTIFIER.fullmatch(task_id):
        raise ValueError('A canonical task identifier is required')
    if type(seed) is not int or seed not in range(10):
        raise ValueError('Seed must be an integer from 0 through 9')
    if type(attempt) is not int or not 0 <= attempt <= 9999:
        raise ValueError('Attempt must be an integer from 0 through 9999')
    if seed == 0 and attempt:
        raise ValueError('Canonical seed 0 cannot be resampled')


def _unit(key, label):
    # The first 53 bits have an exact binary floating-point representation.
    return (int(canonical_hash([key, label])[:14], 16) >> 3) / (1 << 53)


def _order(key, label):
    return sorted(range(9), key=lambda i: canonical_hash([key, label, i]))


def _stratified(key, label, seed, lower, upper):
    """Nine strata, restricted jitter; pairwise gap >= 0.6*(upper-lower)/9."""
    rank = _order(key, label)[seed - 1]
    fraction = (rank + .3 + .4 * _unit(key, [label, seed])) / 9
    return round(lower + (upper - lower) * fraction, 12)


def _spec(unit, bounds, visibility='public', distribution='stratified_uniform'):
    return {'unit': unit, 'bounds': list(bounds), 'distribution': distribution,
            'visibility': visibility, 'strata': 9, 'within_stratum_fraction': [.3, .7]}


def parameter_specs():
    """Declared sampling domains; seed 0 uses explicit canonical sentinels."""
    return {
        'layout.radius_m': _spec('m', [.04, .15]),
        'layout.angle_rad': _spec('rad', [0., 2 * math.pi]),
        'layout.yaw_abs_rad': _spec('rad', [math.radians(15), math.radians(40)],
                                     distribution='signed_stratified_uniform'),
        'appearance.pattern_id': _spec('1', [1, 9], distribution='permutation'),
        'appearance.hue_turns': _spec('turn', [0., 1.]),
        'appearance.contrast': _spec('1', [.35, .65]),
        'appearance.roughness': _spec('1', [.22, .82]),
        'appearance.pattern_scale_m': _spec('m', [.035, .10]),
        'physics.density_factor': _spec('1', [.65, 1.45], 'public_nominal'),
        'physics.mass_factor': _spec('1', [.65, 1.45], 'public_nominal',
                                     'alias_of_density_factor_at_fixed_geometry'),
        'physics.friction_factor': _spec('1', [.6, 1.4], 'public_nominal'),
    }


def sample_parameters(family, task_id, configuration_id, seed, attempt=0):
    """Return a fresh JSON dictionary. Robot identity is never an input.

    ``mass_factor`` aliases ``density_factor`` for fixed geometry: multiply
    mass and inertia once. With changed geometry, recompute both from density
    and the authored volume. Friction applies to actual contact materials.
    Optional elasticity or joint parameters must be declared by an implementing
    family recipe; this sampler does not fabricate unconnected physical values.
    """
    _inputs(family, task_id, configuration_id, seed, attempt)
    identity = [VERSION, family, task_id]
    streams = {name: canonical_hash([identity, name, configuration_id, attempt]
                                    if name == 'layout' else [identity, name])
               for name in ('appearance', 'layout', 'physics')}
    layout = {'translation_xy_m': [0., 0.], 'yaw_rad': 0.}
    appearance = {'pattern_id': 0, 'hue_turns': 0., 'contrast': 0.,
                  'roughness': None, 'pattern_scale_m': None}
    physics = {'density_factor': 1., 'mass_factor': 1., 'friction_factor': 1.}
    if seed:
        key = streams['layout']
        radius = _stratified(key, 'radius', seed, .04, .15)
        angle = _stratified(key, 'angle', seed, 0., 2 * math.pi)
        yaw = _stratified(key, 'yaw', seed, math.radians(15), math.radians(40))
        sign = -1 if _unit(key, ['yaw_sign', seed]) < .5 else 1
        layout = {'translation_xy_m': [round(radius * math.cos(angle), 12),
                                      round(radius * math.sin(angle), 12)],
                  'yaw_rad': round(sign * yaw, 12)}
        key = streams['appearance']
        appearance = {
            'pattern_id': _order(key, 'pattern')[seed - 1] + 1,
            'hue_turns': _stratified(key, 'hue', seed, 0., 1.),
            'contrast': _stratified(key, 'contrast', seed, .35, .65),
            'roughness': _stratified(key, 'roughness', seed, .22, .82),
            'pattern_scale_m': _stratified(key, 'scale', seed, .035, .10),
        }
        key = streams['physics']
        density = _stratified(key, 'density', seed, .65, 1.45)
        physics = {'density_factor': density, 'mass_factor': density,
                   'friction_factor': _stratified(key, 'friction', seed, .6, 1.4)}
    result = {
        'schema': VERSION, 'family': family, 'task_id': task_id,
        'configuration_id': configuration_id, 'seed': seed, 'attempt': attempt,
        'canonical': seed == 0, 'status': 'candidate', 'layout': layout,
        'appearance': appearance, 'physics': physics,
        'parameter_specs': parameter_specs(), 'stream_sha256': streams,
        'optional_physics': {'elasticity': 'not_sampled', 'joint_parameters': 'not_sampled'},
    }
    result['instance_sha256'] = canonical_hash(result)
    return result


def validate_parameters(parameters):
    """Reject edits, unknown fields (including robots), and invalid inputs.

    Applied geometry/physics records belong beside this immutable candidate,
    never in a silently modified candidate with its old identity.
    """
    if not isinstance(parameters, dict):
        raise ValueError('Parameter record must be a dictionary')
    try:
        args = [parameters[k] for k in ('family', 'task_id', 'configuration_id', 'seed', 'attempt')]
        expected = sample_parameters(*args)
        if canonical_hash(parameters) != canonical_hash(expected):
            raise ValueError('Parameters differ from the deterministic candidate')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid parameter record') from exc
    return True


def rejection_record(parameters, reasons, *, measurements=None):
    """Separate, content-bound authoring rejection; does not alter a candidate."""
    validate_parameters(parameters)
    if not isinstance(reasons, (list, tuple)) or not reasons or any(
            not isinstance(reason, str) or not reason.strip() for reason in reasons):
        raise ValueError('At least one explicit rejection reason is required')
    result = {'schema': 'parameterized_rejection.v1',
              'instance_sha256': parameters['instance_sha256'],
              'seed': parameters['seed'], 'attempt': parameters['attempt'],
              'status': 'rejected', 'reasons': list(reasons),
              'measurements': deepcopy(measurements or {})}
    result['record_sha256'] = canonical_hash(result)
    return result
