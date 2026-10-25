"""Generate immutable tabletop candidates with explicit initial-state splits.

Sampling happens before release creation, never inside a running simulation.
The SHA-256 sampler has no platform RNG state. Bounds checks are conservative
geometric checks, not a native stability, reachability or success certificate.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

from bench.geometry import quaternion_matrix
from bench.isaacsim.release import digest, task_inputs
from bench.paths import load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset, load_obj_mesh

DEFAULT_REGISTRY = Path(__file__).with_name('catalog.json')
SCHEMA = 'bench_scenarios.v1'
SAMPLER = 'sha256_coordinate_uniform_v1'
SIMPLE_TASKS = ('01_cube_push', '02_cube_lift_hold', '03_cube_pick_place', '04_cube_grasp_hold')


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return load_json(Path(path), object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def vector(value, length, label):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length:
        raise ValueError(label + ' has the wrong shape')
    if any(isinstance(x, (bool, np.bool_)) for x in value):
        raise ValueError(label + ' contains a boolean')
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(label + ' must contain finite numbers')
    return result


def number(value, label, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(label + ' must be a finite ' + ('positive number' if positive else 'number'))
    return float(value)


def load_registry(path=DEFAULT_REGISTRY):
    registry = read_json(path)
    if registry.get('schema') != SCHEMA or registry.get('status') != 'prepared_unvalidated':
        raise ValueError('Expected an explicitly unqualified scenario registry')
    if registry.get('physics_accepted') is not False or registry.get('sampler') != SAMPLER:
        raise ValueError('Registry has an unsupported acceptance or sampler declaration')
    splits = registry['split_seeds']
    if set(splits) != {'development', 'evaluation'}:
        raise ValueError('Registry must declare development and evaluation splits')
    seen = set()
    for seeds in splits.values():
        if not isinstance(seeds, list) or len(seeds) < 3:
            raise ValueError('Every split requires at least three candidate seeds')
        for seed in seeds:
            if type(seed) is not int or not 0 <= seed < 2**32 or seed in seen:
                raise ValueError('Split seeds must be distinct unsigned 32-bit integers')
            seen.add(seed)
    for scene_id, scene in registry['scenes'].items():
        if not re.fullmatch(r'[a-z][a-z0-9_]{1,63}', scene_id):
            raise ValueError('Unsafe scenario ID')
        if scene['status'] != 'prepared_unvalidated' or scene['native_qualified'] is not False:
            raise ValueError('A template cannot claim native qualification')
        if not set(scene['supported_tasks']) <= set(SIMPLE_TASKS):
            raise ValueError('This generator only supports the four reviewed cube tasks')
        if scene['supported_tasks'] or 'target_pose' in scene:
            pose = scene.get('target_pose')
            allowed = {'xy_m', 'xy_jitter_m', 'rpy_rad', 'yaw_jitter_rad'}
            if not isinstance(pose, dict) or 'xy_m' not in pose or set(pose) - allowed:
                raise ValueError('target_pose may only configure placement, never target identity or physics')
        names = [item['id'] for item in scene['instances']]
        if len(names) != len(set(names)) or 'cube' in names:
            raise ValueError('Scenario instance IDs conflict')
        for item in scene['instances']:
            if not re.fullmatch(r'[a-z][a-z0-9_]{1,63}', item['id']):
                raise ValueError('Unsafe instance ID')
            if item['kind'] not in ('objects', 'fixtures') or item['catalog'] not in registry['catalogs']:
                raise ValueError('Unknown instance kind/catalog')
    root = Path(registry['source_release']['path'])
    if digest(root / 'release_manifest.json') != registry['source_release']['sha256']:
        raise ValueError('Scenario source release changed')
    pinned_catalog = read_json(root / 'release_manifest.json').get('asset_catalog')
    target_catalog = registry['catalogs'].get('target_cube')
    if (not isinstance(pinned_catalog, dict) or not isinstance(target_catalog, dict)
            or Path(pinned_catalog['path']).resolve() != (Path(target_catalog['root'])/'manifest.json').resolve()
            or pinned_catalog['sha256'] != target_catalog['manifest_sha256']):
        raise ValueError('Target asset catalog changed from the frozen source release')
    registry['_registry_path'] = str(Path(path).resolve())
    registry['_registry_sha256'] = digest(path)
    return registry


def _asset(reference, registry):
    catalog = registry['catalogs'][reference['catalog']]
    root = Path(catalog['root'])
    if digest(root / 'manifest.json') != catalog['manifest_sha256']:
        raise ValueError('Scenario asset catalog changed: ' + reference['catalog'])
    return load_task_asset(reference['asset_id'], root, verify_hashes=True)


def asset_bounds(asset, *, include_collision=True):
    bounds = asset.get('bounds_m', asset.get('aabb_m'))
    if bounds is None:
        bounds = [asset['bounds_min_m'], asset['bounds_max_m']]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError('Asset bounds must have two corners')
    low, high = (vector(row, 3, 'asset bounds') for row in bounds)
    if np.any(high <= low) or np.any(high - low > .65):
        raise ValueError('Expected positive tabletop-sized asset bounds')
    result = np.array([low, high])
    if include_collision:
        for part in asset.get('collision_parts', []):
            geometry = part.get('geometry')
            if geometry in ('convex_mesh', 'mesh'):
                vertices, _ = load_obj_mesh(part['mesh_path'])
                vertices = vertices * vector(part.get('scale', [1., 1., 1.]), 3, 'mesh scale')
                if not np.isfinite(vertices).all() or len(vertices) < 4:
                    raise ValueError('Invalid collision mesh')
                part_bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
            elif geometry == 'box':
                half = vector(part['half_extents_m'], 3, 'collision box')
                if np.any(half <= 0):
                    raise ValueError('Invalid collision box dimensions')
                part_bounds = np.array([-half, half])
            elif geometry in ('cylinder', 'sphere'):
                radius = number(part['radius_m'], 'collision radius', positive=True)
                half = np.array([radius, radius, number(part['half_height_m'], 'cylinder height', positive=True)
                                                 if geometry == 'cylinder' else radius])
                part_bounds = np.array([-half, half])
            elif geometry in ('usd_convexHull', 'usd_convexDecomposition'):
                if asset.get('backend') != 'usd_reference':
                    raise ValueError('Native USD geometry requires the USD asset reader')
                continue  # Pinned USD catalog bounds; cooked hulls require native verification.
            else:
                raise ValueError('Unsupported collision bounds geometry: ' + str(geometry))
            q = vector(part.get('quaternion_xyzw', [0., 0., 0., 1.]), 4, 'collision quaternion')
            transformed = rotated_bounds(part_bounds, [q[3], *q[:3]])
            transformed += vector(part.get('position_m', [0., 0., 0.]), 3, 'collision translation')
            result = np.array([np.minimum(result[0], transformed[0]), np.maximum(result[1], transformed[1])])
    if np.any(result[1]-result[0] > .65):
        raise ValueError('Collision envelope exceeds tabletop-sized bounds')
    return result


def rotated_bounds(bounds, quaternion):
    q = vector(quaternion, 4, 'quaternion')
    if abs(float(np.linalg.norm(q)) - 1.) > 1e-7:
        raise ValueError('Quaternion must have unit length')
    corners = np.array(list(itertools.product(*zip(bounds[0], bounds[1], strict=True))))
    transformed = corners @ quaternion_matrix(q).T
    return np.array([transformed.min(axis=0), transformed.max(axis=0)])


def _quaternion(rpy):
    roll, pitch, yaw = vector(rpy, 3, 'rpy') / 2
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return [float(cr*cp*cy+sr*sp*sy), float(sr*cp*cy-cr*sp*sy),
            float(cr*sp*cy+sr*cp*sy), float(cr*cp*sy-sr*sp*cy)]


def _sample(scene_id, split, seed, field, center, half_range):
    center = number(center, field)
    half_range = number(half_range, field + ' range')
    if half_range < 0:
        raise ValueError('Sampling ranges cannot be negative')
    key = json.dumps([SAMPLER, scene_id, split, seed, field], separators=(',', ':')).encode()
    uniform = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
    return center + (2 * uniform - 1) * half_range


def _place(reference, asset, table_z, scene_id, split, seed):
    xy = vector(reference['xy_m'], 2, 'placement XY')
    jitter = vector(reference.get('xy_jitter_m', [0., 0.]), 2, 'placement jitter')
    rpy = vector(reference.get('rpy_rad', [0., 0., 0.]), 3, 'placement rotation')
    yaw = _sample(scene_id, split, seed, reference['id'] + '.yaw', rpy[2], reference.get('yaw_jitter_rad', 0.))
    q = _quaternion([rpy[0], rpy[1], yaw])
    local = rotated_bounds(asset_bounds(asset), q)
    position = [_sample(scene_id, split, seed, reference['id'] + f'.xy{axis}', xy[axis], jitter[axis])
                for axis in range(2)]
    drop = .001 if reference['kind'] == 'objects' else 0.
    position.append(table_z - float(local[0, 2]) + drop)
    return {'id': reference['id'], 'asset_id': reference['asset_id'], 'asset_root': asset['asset_root'],
            'position_m': position, 'quaternion_wxyz': q, 'friction': .65}


def _gap(a, b, dimensions=2):
    return float(np.linalg.norm(np.maximum(0., np.maximum(a[0, :dimensions]-b[1, :dimensions],
                                                         b[0, :dimensions]-a[1, :dimensions]))))


def validate_scene(scene, *, min_clearance_m=.015, table_margin_m=.03):
    """Reject overlap, table overflow and malformed physical specifications.

    Pair separation uses conservative world AABBs, including cloth. Robot-base
    exclusion is checked; full robot collision and stable settling remain native
    checks and are explicitly not claimed by this function.
    """
    number(min_clearance_m, 'clearance', positive=True)
    number(table_margin_m, 'table margin', positive=True)
    table_position = vector(scene['table']['position_m'], 3, 'table position')
    table_half = vector(scene['table']['half_extents_m'], 3, 'table extents')
    if np.any(table_half <= 0) or np.any(table_half > 2):
        raise ValueError('Invalid table extents')
    low, high = table_position-table_half, table_position+table_half
    table_z = high[2]
    base = vector(scene['robot']['base_position_m'], 3, 'robot base')
    keepout = np.array([base[:2] - [.18, .18], base[:2] + [.18, .18]])
    boxes, evidence, dependencies = {}, {}, {}
    for kind in ('objects', 'fixtures'):
        for item in scene.get(kind, []):
            name = item['id']
            if not re.fullmatch(r'[a-z][a-z0-9_]{1,63}', name) or name in boxes or name == 'cloth':
                raise ValueError('Duplicate or unsafe scene instance ID')
            asset = load_task_asset(item['asset_id'], item['asset_root'], verify_hashes=True)
            bounds = rotated_bounds(asset_bounds(asset), item.get('quaternion_wxyz', [1., 0., 0., 0.]))
            bounds += vector(item['position_m'], 3, 'object position')
            if not 0 <= number(item['friction'], 'friction') <= 5:
                raise ValueError('Invalid friction')
            if not asset.get('collision_parts'):
                raise ValueError('Asset has no native collision parts')
            if kind == 'objects':
                number(asset['mass_kg'], 'dynamic mass', positive=True)
                inertia = asset.get('inertia_diag_kg_m2')
                native_computed = inertia is None and asset.get('backend') == 'usd_reference'
                if not native_computed and np.any(vector(inertia, 3, 'inertia') <= 0):
                    raise ValueError('Dynamic inertia must be positive')
            if (np.any(bounds[0, :2] < low[:2]+table_margin_m)
                    or np.any(bounds[1, :2] > high[:2]-table_margin_m)):
                raise ValueError('Instance extends beyond the usable table: ' + name)
            bottom_gap = float(bounds[0, 2] - table_z)
            if not -.000001 <= bottom_gap <= .002:
                raise ValueError('Instance is not initialized at table support: ' + name)
            if _gap(bounds, keepout) < min_clearance_m:
                raise ValueError('Instance overlaps the robot-base keepout: ' + name)
            boxes[name] = bounds
            evidence[name] = {'world_aabb_m': bounds.tolist(), 'table_gap_m': bottom_gap,
                              'asset_id': item['asset_id'], 'manifest_sha256': asset['manifest_sha256'],
                              'inertia_source': 'native_computed_from_source_collision' if asset.get('inertia_diag_kg_m2') is None
                                                else 'declared_catalog_inertia'}
            for path, expected in asset_dependencies(asset).items():
                if path in dependencies and dependencies[path] != expected:
                    raise ValueError('Conflicting asset dependency hashes')
                dependencies[path] = expected
    cloth = scene.get('cloth', {})
    if cloth.get('enabled'):
        size = vector(cloth['size_m'], 2, 'cloth size')
        if np.any(size <= 0) or np.any(size > .6):
            raise ValueError('Invalid cloth size')
        cells = cloth['cells_xy']
        if len(cells) != 2 or any(type(x) is not int or not 1 <= x <= 100 for x in cells):
            raise ValueError('Invalid cloth grid')
        number(cloth['mass_kg'], 'cloth mass', positive=True)
        bounds = rotated_bounds(np.array([[0., 0., 0.], [*size, 0.]]), cloth['quaternion_wxyz'])
        bounds += vector(cloth['origin_m'], 3, 'cloth origin')
        if (np.any(bounds[0, :2] < low[:2]+table_margin_m)
                or np.any(bounds[1, :2] > high[:2]-table_margin_m)
                or not table_z+.002 <= bounds[0, 2] <= bounds[1, 2] <= table_z+.01):
            raise ValueError('Cloth must start flat just above the usable table')
        if _gap(bounds, keepout) < min_clearance_m:
            raise ValueError('Cloth overlaps the robot-base keepout')
        boxes['cloth'] = bounds
        evidence['cloth'] = {'world_aabb_m': bounds.tolist(), 'native_stability_verified': False}
    gaps = []
    for a, b in itertools.combinations(sorted(boxes), 2):
        gap = _gap(boxes[a], boxes[b])
        if gap < min_clearance_m:
            raise ValueError(f'Initial instance clearance below {min_clearance_m} m: {a}, {b}')
        gaps.append({'instances': [a, b], 'xy_aabb_clearance_m': gap})
    return {'schema': 'bench_scene_geometry_check.v1', 'passed': True,
            'checks': ['finite_metric_bounds', 'unit_quaternions', 'dynamic_mass_inertia',
                       'collision_inventory', 'table_support_gap', 'table_edges',
                       'pairwise_conservative_xy_clearance', 'robot_base_keepout', 'asset_hashes'],
            'instances': evidence, 'pair_clearances': gaps, 'asset_dependencies_sha256': dependencies,
            'physics_accepted': False, 'native_stability_verified': False,
            'limitations': ['No native settling, full robot collision, reachability or task success evidence.',
                            'Referenced USD assets use pinned source bounds; cooked hull envelopes require native checks.']}


def asset_dependencies(asset):
    """Freeze every runtime asset file actually referenced by either loader."""
    result = {asset['manifest_path']: asset['manifest_sha256']}
    if 'mesh_path' in asset:
        result[asset['mesh_path']] = asset['sha256']
    result.update(asset.get('source_dependencies_sha256', {}))
    for part in asset.get('visual_parts', []) + asset.get('collision_parts', []):
        if 'mesh_path' in part:
            result[part['mesh_path']] = part['sha256']
    for path, expected in result.items():
        if digest(path) != expected:
            raise ValueError('Asset dependency changed: ' + path)
    return result


def generate(scene_id, split, seed, *, task_id=None, registry_path=DEFAULT_REGISTRY):
    registry = load_registry(registry_path)
    if scene_id not in registry['scenes']:
        raise ValueError('Unknown scenario: ' + str(scene_id))
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    if split not in ('development', 'evaluation', 'diagnostic'):
        raise ValueError('Choose an explicit development/evaluation/diagnostic split')
    if split != 'diagnostic' and seed not in registry['split_seeds'][split]:
        raise ValueError('Seed is not a frozen member of the selected split')
    spec = registry['scenes'][scene_id]
    if task_id is None and spec['supported_tasks']:
        task_id = spec['supported_tasks'][0]
    if task_id is not None and task_id not in spec['supported_tasks']:
        raise ValueError('Scene/task combination has no supported evaluator adapter')
    template_task = task_id or '01_cube_push'
    entry, scene, task, initial = task_inputs(Path(registry['source_release']['path']), template_task)
    if initial:
        raise ValueError('Scenario generator requires a reset defined entirely by its scene')
    scene = deepcopy(scene)
    source_targets = [item for item in scene['objects'] if item['id'] == 'cube']
    scene['objects'], scene['fixtures'] = [], []
    scene['workcell_details']['enabled'] = False
    scene['workbench']['props'] = False
    scene['cloth']['enabled'] = False
    table_z = scene['table']['position_m'][2] + scene['table']['half_extents_m'][2]
    if task_id:
        cfg = task['evaluation']
        if cfg['object'] != 'cube' or cfg['mode'] not in ('push', 'pick_place', 'lift_hold', 'grasp_hold'):
            raise ValueError('Only the unchanged reviewed cube asset and simple task semantics are supported')
        target_reference = {**spec['target_pose'], 'id': 'cube', 'catalog': 'target_cube',
                            'asset_id': 'collab_cube_small', 'kind': 'objects'}
        target_asset = _asset(target_reference, registry)
        if (len(source_targets) != 1 or source_targets[0]['asset_id'] != target_reference['asset_id']
                or Path(source_targets[0]['asset_root']).resolve() != Path(target_asset['asset_root']).resolve()):
            raise ValueError('Scenario target asset must match the frozen source release identity')
        if not np.allclose(np.diff(asset_bounds(target_asset, include_collision=False), axis=0)[0], [.032, .040, .060], atol=1e-6):
            raise ValueError('Target dimensions differ from the frozen task description')
        cube = _place(target_reference, target_asset, table_z, scene_id, split, seed)
        cube['friction'] = source_targets[0]['friction']
        scene['objects'].append(cube)
        old_initial = vector(cfg['initial_position_m'], 3, 'old task initial pose')
        cfg['initial_position_m'] = list(cube['position_m'])
        if cfg['mode'] in ('lift_hold', 'grasp_hold'):
            delta = np.array(cube['position_m']) - old_initial
            cfg['goal_position_m'] = (np.array(cfg['goal_position_m']) + delta).tolist()
            task['target_position_m'] = list(cfg['goal_position_m'])
            task['public_goal']['target_position_m'] = list(cfg['goal_position_m'])
            for marker in scene['goal_markers']:
                marker['position_m'][:2] = cfg['goal_position_m'][:2]
        else:
            minimum = float(cfg['minimum_displacement_m'])
            if np.linalg.norm(np.array(cfg['goal_position_m'])[:2] - np.array(cube['position_m'])[:2]) <= minimum+.01:
                raise ValueError('Variant starts too close to its required displacement goal')
    else:
        task = None
        scene['goal_markers'] = []
    for reference in spec['instances']:
        asset = _asset(reference, registry)
        scene[reference['kind']].append(_place(reference, asset, table_z, scene_id, split, seed))
    if 'cloth' in spec:
        scene['cloth'].update(deepcopy(spec['cloth']))
        for axis in range(2):
            scene['cloth']['origin_m'][axis] = _sample(scene_id, split, seed, f'cloth.origin{axis}',
                spec['cloth']['origin_m'][axis], .012)
    identity = {'registry_version': registry['version'], 'registry_sha256': registry['_registry_sha256'],
                'scene_id': scene_id, 'split': split, 'seed': seed, 'sampler': SAMPLER,
                'task_id': task_id, 'status': 'prepared_unvalidated', 'physics_accepted': False}
    scene['scenario'] = identity
    scene['scene_variant'] = {'id': 'original_v1', 'ranking_stratum': f"scenario:{registry['version']}:{scene_id}:{split}:{seed}",
                              'frozen_source_mutated': False, 'scenario_initialization_frozen': True}
    scene['description'] = spec['description_zh']
    # These fields are legacy provenance in the source; new exports describe
    # the native backend they actually target rather than inheriting old labels.
    scene['physics'] = {'dt_s': scene['isaacsim']['physics_dt_s'],
                        'gravity_m_s2': [0., 0., -9.81], 'backend': 'isaacsim_native_physx'}
    if task is not None:
        task['scenario'] = deepcopy(identity)
    validation = validate_scene(scene, **registry['validation'])
    if task is not None:
        goal = vector(task['evaluation']['goal_position_m'], 3, 'task goal')
        if (list(goal) != task['target_position_m']
                or list(goal) != task['public_goal']['target_position_m']):
            raise ValueError('Private/public goal coordinates disagree')
        goal_bounds = rotated_bounds(asset_bounds(target_asset), cube['quaternion_wxyz']) + goal
        table = scene['table']
        table_low = np.array(table['position_m']) - table['half_extents_m']
        table_high = np.array(table['position_m']) + table['half_extents_m']
        margin = registry['validation']['table_margin_m']
        if (np.any(goal_bounds[0, :2] < table_low[:2]+margin)
                or np.any(goal_bounds[1, :2] > table_high[:2]-margin)):
            raise ValueError('Goal object footprint extends beyond usable table')
        for name, record in validation['instances'].items():
            if name != 'cube' and _gap(goal_bounds, np.array(record['world_aabb_m'])) < registry['validation']['min_clearance_m']:
                raise ValueError('Goal overlaps a context object or fixture: ' + name)
        validation['goal_workspace_clear'] = True
    request_defaults = ({'task_id': task_id, 'seed': seed, 'scene_variant': 'original_v1',
                         'robot_id': 'sharpa_w01_right'} if task is not None else None)
    return {'scene': scene, 'task': task, 'identity': identity, 'validation': validation,
            'episode_request_defaults': request_defaults,
            'source_release': deepcopy(registry['source_release']),
            'source_task': deepcopy(entry), 'registry': {k: deepcopy(v) for k, v in registry.items() if not k.startswith('_')}}
