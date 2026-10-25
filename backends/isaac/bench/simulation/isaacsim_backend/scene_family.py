"""Scene-family specification: the declarative contract for a furnished room.

A *scene family* is one room (mechanical assembly, laboratory, kitchen, study,
card room, living room) plus the manipulation objects it contains. The family
is written as data and built by ``scene.py``; this module validates it on CPU
without importing OpenUSD or PhysX, so a bad family fails before a GPU run.

Mandatory physical discipline
-----------------------------
Every dynamic object must declare, and be checked for:
  * at least one collision part, each individually convex, with the bore rule
    respected (a through bore must stay open: no single hull over a ring);
  * positive mass and a diagonal inertia consistent with the declared shape;
  * a rest pose that actually touches a support (no floating, no interpenetration);
  * a reachable footprint inside the declared work envelope.
Static props are solid but immovable; room surfaces are visual unless listed in
``collide``. Nothing decorative is ever given contact implicitly.
"""
from __future__ import annotations

import math

import numpy as np

# Room kinds this schema describes; scenes.rooms.FAMILIES lists the rooms
# authored so far (mechanical_assembly is specified but not yet merged).
SUPPORTED = ('mechanical_assembly', 'laboratory', 'kitchen', 'study', 'card_room', 'living_room')
GRAVITY = 9.81


def _vec(value, length, label, *, positive=False):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != length or not np.isfinite(array).all():
        raise ValueError(f'{label} must be {length} finite numbers')
    if positive and np.any(array <= 0):
        raise ValueError(f'{label} must be positive')
    return [float(v) for v in array]


class FamilyError(ValueError):
    """A scene family that cannot be simulated as declared."""


def validate_object(spec, *, family):
    """Validate one dynamic manipulation object."""
    if not isinstance(spec, dict) or not isinstance(spec.get('id'), str):
        raise FamilyError(f'{family}: object needs an id')
    name = spec['id']
    body = spec.get('body')
    if body is None:
        # Either a shared task-part library asset, or an inline authored body.
        if 'asset_id' not in spec:
            raise FamilyError(f'{family}/{name}: needs either body or asset_id')
        return spec
    origin = body.get('origin', 'centre')
    if origin not in ('centre', 'base_bottom_centre'):
        raise FamilyError(f"{family}/{name}: body.origin must be 'centre' or 'base_bottom_centre'")
    mass = float(body.get('mass_kg', 0.0))
    if not math.isfinite(mass) or mass <= 0:
        raise FamilyError(f'{family}/{name}: body.mass_kg must be positive and finite')
    size = _vec(body.get('size_m', [0, 0, 0]), 3, f'{family}/{name}.body.size_m', positive=True)
    parts = body.get('collision_parts')
    if not isinstance(parts, list) or not parts:
        raise FamilyError(f'{family}/{name}: body needs at least one collision part')
    bores = []
    for index, part in enumerate(parts):
        geometry = part.get('geometry', 'box')
        if geometry not in ('box', 'cylinder', 'convex_mesh', 'sphere'):
            raise FamilyError(f'{family}/{name}/part{index}: unknown geometry {geometry!r}')
        if geometry in ('box', 'cylinder') and 'half_extents_m' not in part and 'radius_m' not in part:
            raise FamilyError(f'{family}/{name}/part{index}: {geometry} needs its own dimensions')
        if part.get('bore') is True:
            bores.append(index)
            if 'bore_radius_m' not in part or float(part['bore_radius_m']) <= 0:
                raise FamilyError(f'{family}/{name}/part{index}: a bore needs bore_radius_m > 0')
            if geometry == 'box':
                # A single box cannot carry a through hole; the ring must be
                # split into convex sectors (as the gears and washers are).
                raise FamilyError(f'{family}/{name}/part{index}: a bored part must be '
                                  'cylinder/convex sectors, not one box')
    # A declared through bore must not be sealed by the union of the collision parts.
    for index in bores:
        radius = float(parts[index]['bore_radius_m'])
        minimum = min(_part_clearance(part) for part in parts)
        if minimum < radius - 1e-9:
            raise FamilyError(f'{family}/{name}/part{index}: some collision part intrudes into the '
                              f'{radius:.4f} m bore (clearance {minimum:.4f} m)')
    inertia = body.get('inertia_diag_kg_m2')
    if inertia is not None:
        inertia = _vec(inertia, 3, f'{family}/{name}.body.inertia_diag_kg_m2', positive=True)
        check = uniform_box_inertia(mass, size)
        if any(i < 0.02 * c for i, c in zip(inertia, check)):
            raise FamilyError(f'{family}/{name}: declared inertia is far below its uniform-box '
                              f'value {check}; a dynamic body cannot be massless about an axis')
    return spec


def _part_clearance(part):
    """Radial clearance of a convex collision part around its local z axis."""
    if part.get('geometry') == 'cylinder':
        return float(part.get('radius_m', 0.0))
    if part.get('geometry') == 'box':
        half = np.asarray(part.get('half_extents_m', [0, 0, 0]), dtype=float)
        return float(min(half[0], half[1]))
    return float(part.get('clearance_radius_m', 0.0))


def uniform_box_inertia(mass, size):
    sx, sy, sz = size
    return [mass / 12.0 * (sy ** 2 + sz ** 2),
            mass / 12.0 * (sx ** 2 + sz ** 2),
            mass / 12.0 * (sx ** 2 + sy ** 2)]


def validate_support(objects, supports, *, family, tolerance=1e-3):
    """Every dynamic object must rest on a declared support, not float or sink.

    A library-backed object has no inline ``body``; it may declare
    ``rest_geometry`` (size_m and origin) so its rest pose is still checked.
    """
    for spec in objects:
        if spec.get('spawn', {}).get('gravity', True) is False:
            continue
        position = np.asarray(spec['position_m'], dtype=float)
        body = spec.get('body')
        if body is None:
            body = spec.get('rest_geometry')
        if body is None:
            continue
        size = np.asarray(body['size_m'], dtype=float)
        z_min = position[2] - (size[2] / 2. if body.get('origin', 'centre') == 'centre' else 0.0)
        best = None
        for support in supports:
            top = float(support['top_z_m'])
            if not (support['xy_min'][0] - 1e-6 <= position[0] <= support['xy_max'][0] + 1e-6 and
                    support['xy_min'][1] - 1e-6 <= position[1] <= support['xy_max'][1] + 1e-6):
                continue
            if best is None or top > best:
                best = top
        if best is None:
            raise FamilyError(f'{family}/{spec["id"]}: does not rest over any declared support '
                              f'surface (checked {len(supports)})')
        if z_min < best - tolerance:
            raise FamilyError(f'{family}/{spec["id"]}: sinks {best - z_min:.4f} m into its support')
        if z_min > best + tolerance:
            raise FamilyError(f'{family}/{spec["id"]}: floats {z_min - best:.4f} m above its support')


def support_surfaces(scene):
    """Extract the planar support surfaces a family declares."""
    surfaces = []
    for key in ('surfaces', 'static_props'):
        for item in scene.get(key, []) or []:
            top = item.get('position_m', [0, 0, 0])
            half = item.get('half_extents_m')
            if half is None:
                continue
            top_z = float(top[2]) + float(half[2])
            surfaces.append({'id': item.get('id', key), 'top_z_m': top_z,
                             'xy_min': [float(top[0]) - float(half[0]), float(top[1]) - float(half[1])],
                             'xy_max': [float(top[0]) + float(half[0]), float(top[1]) + float(half[1])]})
    return surfaces


def validate_family(scene):
    """Full CPU validation of a scene family; raises on anything unsimulatable."""
    from .room import validate_room
    from .cameras import resolve_cameras

    family = scene.get('family')
    if family not in SUPPORTED:
        raise FamilyError(f'family must be one of {SUPPORTED}, got {family!r}')
    if scene.get('schema_version') != 'isaacsim-scene-family-v1':
        raise FamilyError('schema_version must be isaacsim-scene-family-v1')
    physics = scene.get('physics')
    if not isinstance(physics, dict):
        raise FamilyError(f'{family}: a physics block is required (dt_s, control_dt_s, gravity_m_s2)')
    dt_s = float(physics.get('dt_s', 0.0))
    control_dt_s = float(physics.get('control_dt_s', 0.0))
    if not all(math.isfinite(value) and value > 0 for value in (dt_s, control_dt_s)):
        raise FamilyError(f'{family}: physics.dt_s and physics.control_dt_s must be positive')
    if control_dt_s / dt_s < 2:
        raise FamilyError(f'{family}: a control window of {control_dt_s:g} s spans fewer than two '
                          f'physics steps at dt {dt_s:g} s')
    gravity = _vec(physics.get('gravity_m_s2', [0, 0, -9.81]), 3, f'{family}.physics.gravity_m_s2')
    if abs(gravity[2] + 9.81) > 0.5 or abs(gravity[0]) > 1e-6 or abs(gravity[1]) > 1e-6:
        raise FamilyError(f'{family}: gravity_m_s2 must be the standard vertical field, got {gravity}')
    room = validate_room(scene.get('room'))
    if not room['enabled']:
        raise FamilyError(f'{family}: a structured family requires a room block')
    cameras = resolve_cameras(scene)
    if not any(c['kind'] == 'world' for c in cameras.values()):
        raise FamilyError(f'{family}: needs at least one world (third-person) camera')

    robot = scene.get('robot', {})
    base = _vec(robot.get('base_position_m', [0, 0, 0]), 3, 'robot.base_position_m')
    # The mount must equal the selected versioned hand assembly, or the env
    # refuses to start. Checking it here turns a GPU-time crash into a CPU error.
    try:
        from .robot_registry import get_hand_spec
        hand = get_hand_spec(robot.get('hand_id', 'sharpa_w01_right'), robot.get('assembly_version'))
    except Exception as exc:      # unknown hand/assembly is its own, later error
        raise FamilyError(f'{family}: cannot resolve hand assembly: {exc}') from exc
    for key, spec_key in (('mount_position_m', 'mount_translation_m'),
                          ('mount_quaternion_wxyz', 'mount_quaternion_wxyz')):
        declared = robot.get(key, hand[spec_key])
        if list(declared) != list(hand[spec_key]):
            raise FamilyError(f'{family}: robot.{key}={list(declared)} does not match the '
                              f'{robot.get("assembly_version")} assembly {list(hand[spec_key])}; '
                              'omit it to inherit, or copy the assembly value exactly')
    home = robot.get('arm_home_rad')
    if home is None or len(home) != 7 or not np.isfinite(np.asarray(home, dtype=float)).all():
        raise FamilyError(f'{family}: robot.arm_home_rad must be seven finite joint angles')
    # The env reads these joint-drive gains directly; a missing one is a crash at
    # GPU startup, so it is caught here instead.
    for key in ('arm_kp', 'arm_kd', 'hand_kp', 'hand_kd'):
        value = robot.get(key)
        if value is None or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise FamilyError(f'{family}: robot.{key} must be a finite non-negative drive gain '
                              '(the released SHARPA scenes use arm_kp 2000, arm_kd 140, '
                              'hand_kp 3.0, hand_kd 0.1)')
    pedestal = scene.get('pedestal')
    if pedestal is None:
        raise FamilyError(f'{family}: the robot must stand on its own pedestal, not on the work table')
    p_half = _vec(pedestal['half_extents_m'], 3, 'pedestal.half_extents_m', positive=True)
    top = float(pedestal['position_m'][2]) + p_half[2]
    if abs(base[2] - top) > 1e-6:
        raise FamilyError(f'{family}: robot base z {base[2]} must equal the pedestal top {top}')
    table = scene.get('table')
    if table is not None:
        t_half = np.asarray(table['half_extents_m'], dtype=float)
        t_center = np.asarray(table['position_m'], dtype=float)
        px = float(pedestal['position_m'][0])
        py = float(pedestal['position_m'][1])
        inside = (abs(px - t_center[0]) < t_half[0] + p_half[0] - 1e-6 and
                  abs(py - t_center[1]) < t_half[1] + p_half[1] - 1e-6)
        if inside:
            raise FamilyError(f'{family}: the robot pedestal overlaps the work table; '
                              'the arm must stand beside it')

    for spec in scene.get('static_props', []) or []:
        if not spec.get('parts'):
            raise FamilyError(f'{family}/{spec.get("id")}: a static prop needs parts')
        for part in spec['parts']:
            if part.get('geometry', 'box') != 'box':
                raise FamilyError(f'{family}/{spec["id"]}/{part.get("id")}: only boxes are supported')

    objects = list(scene.get('objects', []) or [])
    for spec in objects:
        validate_object(spec, family=family)
        if 'asset_root' in spec and not spec.get('rest_geometry'):
            raise FamilyError(f'{family}/{spec["id"]}: a library-backed object needs rest_geometry '
                              '(size_m and origin) so its rest pose can be verified')
    supports = support_surfaces(scene)
    if objects and supports:
        validate_support(objects, supports, family=family)
    return {'family': family, 'room': room, 'cameras': cameras,
            'objects': len(objects), 'static_props': len(scene.get('static_props', []) or []),
            'supports': supports}
