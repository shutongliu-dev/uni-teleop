"""Deterministic, bounded reset layouts; no simulator or robot strategy.

Geometry is authored before these planar group transforms. Joined bodies and
their contents move together. Trays retain axis alignment required by the
unchanged physical referees. Discrete IK and native reset checks are separate.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math

from .common import canonical_hash, selection

VERSION = 'living_room_seed_layouts_v1'
PARAMETERIZED_VERSION = 'parameterized_physics_v1'
SCHEMA = 'living_room_seed_case_plan.v1'
OFFSETS = ((-1., -1.), (-1., 0.), (-1., 1.), (0., -1.), (0., 1.),
           (1., -1.), (1., 0.), (1., 1.), (.5, -.5))


def _delta(task_id, name, seed, bounds, yaw_limit):
    if seed == 0:
        return [0., 0.], 0.
    # Explicit SHA ordering is independent of Python's random/hash versions.
    order = sorted(range(9), key=lambda i: hashlib.sha256(
        f'{VERSION}:{task_id}:{name}:{i}'.encode()).digest())
    x, y = OFFSETS[order[seed-1]]
    sign = -1. if hashlib.sha256(f'{task_id}:{name}'.encode()).digest()[0] % 2 else 1.
    return [x*bounds[0], y*bounds[1]], sign*yaw_limit*((seed-5)/4 if seed != 5 else .25)


def plan_case(task_id, metadata, seed, *, generator_version=VERSION,
              configuration_id='single_arm', attempt=0):
    """Return a reset-only plan from the pinned original task asset metadata.

    The same task seed has the same physical scene for all three installations;
    robot configuration is intentionally absent from the layout generator.
    """
    selection(task_id, seed)
    if generator_version not in (VERSION, PARAMETERIZED_VERSION):
        raise ValueError('Unknown living-room reset generator')
    if generator_version == VERSION and attempt != 0:
        raise ValueError('Historical living layouts have no candidate attempts')
    m, groups = deepcopy(metadata), {}
    root = m.get('root_path', m.get('root_prim_path', '/World/LivingTask'))

    def group(name, pivot, actors, paths, xy, yaw=0.):
        translation, angle = _delta(task_id, name, seed, xy, yaw)
        groups[name] = {'pivot_m': list(pivot), 'translation_xy_m': translation,
            'yaw_rad': angle, 'actor_ids': list(actors), 'prim_paths': list(paths)}

    if task_id == 'remote_operation':
        group('remote', m['remote']['position_m'], ['remote', 'power', 'confirm'],
              [m['remote']['assembly_path']], [.035, .035], math.radians(10.))
        group('holder', m['holder']['position_m'], ['holder'],
              [m['holder']['body_path']], [.030, .025], math.radians(7.))
    elif task_id == 'water_pouring':
        kettle, cup = m['containers']['kettle'], m['containers']['cup']
        group('kettle', kettle['initial_position_m'], ['kettle'],
              [kettle['path'], m['water']['particles_path']], [.012, .012], math.radians(7.))
        group('cup', cup['initial_position_m'], ['cup'], [cup['path']], [.012, .012], math.radians(7.))
        tray = m['tray']; lo, hi = tray['bounds_xy_m']
        group('tray', [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, tray['interior_floor_z_m']],
              [], [tray['path']], [0., 0.])
    elif task_id == 'storage_box':
        actors = m['actors']
        group('box_assembly', m['initial_positions_m']['box'], list(actors), list(actors.values()),
              [.012, .012], math.radians(3.))
        lo, hi = m['tray_inside_xy_m']
        group('tray', [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, m['tray_floor_z_m']],
              [], [root+'/CollectionTray'], [.008, .010])
    plan = {'schema': SCHEMA, 'generator_version': VERSION, 'task_id': task_id, 'seed': seed,
        'geometry_variant': 'canonical' if seed == 0 else ('narrow' if seed % 2 else 'wide'),
        'appearance_variant': 'canonical' if seed == 0 else ('warm', 'cool', 'neutral')[(seed-1) % 3],
        'group_transforms': groups}
    if generator_version == PARAMETERIZED_VERSION:
        from scenes.workrooms.parameterized.spec import sample_parameters
        selection(task_id, seed, configuration_id)
        parameters = sample_parameters('living_room', task_id, configuration_id, seed, attempt=attempt)
        source_group = {'remote_operation': 'remote', 'water_pouring': 'kettle',
                        'storage_box': 'box_assembly'}[task_id]
        for name, value in groups.items():
            # Fixed destinations stay fixed; connected mechanisms and their
            # contents receive one coherent transform before native creation.
            value['translation_xy_m'] = (deepcopy(parameters['layout']['translation_xy_m'])
                                          if name == source_group else [0., 0.])
            value['yaw_rad'] = parameters['layout']['yaw_rad'] if name == source_group else 0.
        plan.update(generator_version=generator_version, configuration_id=configuration_id,
                    attempt=attempt, parameters=parameters,
                    appearance_variant='canonical' if seed == 0 else generator_version)
    plan['plan_sha256'] = canonical_hash(plan)
    return plan


def validate_plan(plan, original_metadata):
    expected = plan_case(plan['task_id'], original_metadata, plan['seed'],
        generator_version=plan.get('generator_version', VERSION),
        configuration_id=plan.get('configuration_id', 'single_arm'), attempt=plan.get('attempt', 0))
    if plan != expected:
        raise ValueError('Reset plan differs from its versioned original-metadata recipe')
    return deepcopy(plan)
