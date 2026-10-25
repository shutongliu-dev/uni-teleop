"""Room shell, structured work surfaces and environment lighting for scenes.

The room is a real, closed solid: the floor, four walls and ceiling are built as
textured visual surfaces *and* as primitive box colliders, so the robot cannot
pass through a wall and nothing can fall out of the room. Collision is declared
explicitly per surface group (``room.collide``), and every solid is named in the
returned ``solid_paths`` so contact evidence always traces to a declared body
rather than to decoration.

Visual surfaces are meshes (they may be cut around a door or window); colliders
are primitive ``UsdGeom.Cube`` children, because PhysX's GPU contact filter only
supports primitive colliders and a mesh collider silently drops out of the
filtered contact view. A collider carries ``visibility=invisible`` so it is not
drawn over the textured surface it represents; that does not disable it, since
PhysX ignores UsdGeom imageability (the task-asset colliders rely on the same
fact).

The old ``add_workbench`` in ``scene.py`` remains the default when a scene has
no ``room`` block, which is what keeps every frozen release byte-identical.
"""
from __future__ import annotations

import numpy as np

from .visuals import bind, textured_material, validate_material, uv_box, uv_quad


# Walls named by world direction; ``extent_m`` spans the wall, ``thickness_m`` is
# its own depth. Openings are axis-aligned rectangles in the wall's local frame.
WALL_FACES = ('front', 'back', 'left', 'right')


def validate_room(spec):
    """CPU-only check of a room block; returns a normalised copy."""
    spec = dict(spec or {})
    if not spec.get('enabled', False):
        return {'enabled': False}
    interior = [float(v) for v in spec.get('size_m', [4.0, 4.0, 2.6])]
    if len(interior) != 3 or any(not np.isfinite(v) or v <= 0 for v in interior):
        raise ValueError('room.size_m must be three positive interior dimensions')
    origin = [float(v) for v in spec.get('origin_m', [0., 0., 0.])]
    if len(origin) != 3 or not all(np.isfinite(v) for v in origin):
        raise ValueError('room.origin_m must be three finite numbers')
    thickness = float(spec.get('wall_thickness_m', .12))
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError('room.wall_thickness_m must be positive')
    surfaces = spec.get('surfaces', {})
    unknown = set(surfaces) - {'floor', 'wall', 'ceiling', 'baseboard'}
    if unknown:
        raise ValueError(f'room.surfaces has unknown keys: {sorted(unknown)}')
    openings = []
    for index, opening in enumerate(spec.get('openings', [])):
        wall = opening.get('wall')
        if wall not in WALL_FACES:
            raise ValueError(f'room.openings[{index}].wall must be one of {WALL_FACES}')
        center = [float(v) for v in opening.get('center_m', [0., 0.])]
        size = [float(v) for v in opening.get('size_m', [0., 0.])]
        if len(center) != 2 or len(size) != 2 or any(not np.isfinite(v) for v in center + size):
            raise ValueError(f'room.openings[{index}] needs finite center_m and size_m pairs')
        if min(size) <= 0:
            raise ValueError(f'room.openings[{index}].size_m must be positive')
        openings.append({'id': opening.get('id', f'opening_{index:02d}'), 'wall': wall,
                         'center_m': center, 'size_m': size,
                         'kind': opening.get('kind', 'door')})
    lights = []
    for index, light in enumerate(spec.get('ceiling_lights', [])):
        position = [float(v) for v in light.get('position_m', [0., 0., interior[2] - .1])]
        extent = [float(v) for v in light.get('extent_m', [.6, .6])]
        intensity = float(light.get('intensity', 6000.))
        if len(position) != 3 or len(extent) != 2 or not all(np.isfinite(v) for v in position + extent):
            raise ValueError(f'room.ceiling_lights[{index}] needs finite position_m and extent_m')
        if intensity <= 0:
            raise ValueError(f'room.ceiling_lights[{index}].intensity must be positive')
        lights.append({'id': light.get('id', f'ceiling_{index:02d}'), 'position_m': position,
                       'extent_m': extent, 'intensity': intensity,
                       'color': [float(c) for c in light.get('color', [1., .98, .94])],
                       'shape': light.get('shape', 'rect')})
    dome = spec.get('dome', {})
    normalised_dome = {'enabled': bool(dome.get('enabled', True)),
                       'intensity': float(dome.get('intensity', 300.)),
                       'color': [float(c) for c in dome.get('color', [.85, .9, 1.])],
                       'hdri': dome.get('hdri'), 'hdri_visible': bool(dome.get('hdri_visible', False))}
    if normalised_dome['hdri'] is not None and not str(normalised_dome['hdri']).endswith(('.hdr', '.exr')):
        raise ValueError('room.dome.hdri must be an .hdr or .exr file')
    return {'enabled': True, 'size_m': interior, 'origin_m': origin,
            'wall_thickness_m': thickness,
            'ceiling_height_m': float(spec.get('ceiling_height_m', interior[2])),
            'baseboard_m': float(spec.get('baseboard_m', 0.)),
            'uv_scale': {str(k): [float(x) for x in v] for k, v in (spec.get('uv_scale') or {}).items()},
            'collide': spec.get('collide'),
            'surfaces': {key: str(value) for key, value in surfaces.items()},
            'openings': openings, 'ceiling_lights': lights, 'dome': normalised_dome,
            'title_zh': spec.get('title_zh'),
            'description_zh': spec.get('description_zh')}


def _wall_geometry(room, face):
    """Return the wall's plane, outward normal and the two in-plane axes."""
    sx, sy, sz = room['size_m']
    height = room['ceiling_height_m']
    ox, oy, oz = room['origin_m']
    half_x, half_y = sx / 2., sy / 2.
    if face == 'front':          # +y wall, spans x
        return {'center': np.array([ox, oy + half_y, oz + height / 2.]),
                'normal': np.array([0., 1., 0.]),
                'u_axis': np.array([1., 0., 0.]), 'v_axis': np.array([0., 0., 1.]),
                'half': np.array([half_x, height / 2.])}
    if face == 'back':           # -y wall, spans x
        return {'center': np.array([ox, oy - half_y, oz + height / 2.]),
                'normal': np.array([0., -1., 0.]),
                'u_axis': np.array([-1., 0., 0.]), 'v_axis': np.array([0., 0., 1.]),
                'half': np.array([half_x, height / 2.])}
    if face == 'right':          # +x wall, spans y
        return {'center': np.array([ox + half_x, oy, oz + height / 2.]),
                'normal': np.array([1., 0., 0.]),
                'u_axis': np.array([0., -1., 0.]), 'v_axis': np.array([0., 0., 1.]),
                'half': np.array([half_y, height / 2.])}
    if face == 'left':           # -x wall, spans y
        return {'center': np.array([ox - half_x, oy, oz + height / 2.]),
                'normal': np.array([-1., 0., 0.]),
                'u_axis': np.array([0., 1., 0.]), 'v_axis': np.array([0., 0., 1.]),
                'half': np.array([half_y, height / 2.])}
    raise ValueError(f'Unknown wall face {face!r}')


def _wall_quads(wall, openings):
    """Split a wall rectangle into quads around its openings (axis-aligned).

    Openings are subtracted as rectangles; the remaining area is expressed as
    up to four quads per opening in wall-local (u, v) coordinates.
    """
    hu, hv = float(wall['half'][0]), float(wall['half'][1])
    pending = [(-hu, hu, -hv, hv)]
    for opening in openings:
        ou, ov = opening['center_m']
        su, sv = (s / 2. for s in opening['size_m'])
        olo_u, ohi_u, olo_v, ohi_v = ou - su, ou + su, ov - sv, ov + sv
        remaining = []
        for lo_u, hi_u, lo_v, hi_v in pending:
            if ohi_u <= lo_u or olo_u >= hi_u or ohi_v <= lo_v or olo_v >= hi_v:
                remaining.append((lo_u, hi_u, lo_v, hi_v))
                continue
            if olo_v > lo_v:                                    # below the opening
                remaining.append((lo_u, hi_u, lo_v, min(olo_v, hi_v)))
            if ohi_v < hi_v:                                    # above the opening
                remaining.append((lo_u, hi_u, max(ohi_v, lo_v), hi_v))
            if olo_u > lo_u:                                    # left of the opening
                remaining.append((lo_u, min(olo_u, hi_u), max(lo_v, olo_v), min(hi_v, ohi_v)))
            if ohi_u < hi_u:                                    # right of the opening
                remaining.append((max(ohi_u, lo_u), hi_u, max(lo_v, olo_v), min(hi_v, ohi_v)))
        pending = remaining
    return [(lo_u, hi_u, lo_v, hi_v) for lo_u, hi_u, lo_v, hi_v in pending
            if hi_u - lo_u > 1e-6 and hi_v - lo_v > 1e-6]


def box_collider_matrix(center, half, basis=None):
    """The 4x4 matrix taking the unit cube (-1..1) onto a collider box; pure numpy.

    Row-vector convention, as Gf uses (p' = p * M): scale the unit cube, rotate it
    into the collider frame, then translate it to the world-space ``center``. The
    rotation's rows are the frame axes, so local x maps to u. A box is symmetric
    through its own centre, so the third axis is taken as u x v: that keeps the
    transform a proper rotation even for a left-handed frame -- every wall frame
    from ``_wall_geometry`` is one (its outward normal is -(u x v)).
    """
    matrix = np.diag([*(float(x) for x in half), 1.])
    if basis is not None:
        u, v = (np.asarray(axis, dtype=float) for axis in basis[:2])
        rotation = np.eye(4)
        rotation[0, :3], rotation[1, :3], rotation[2, :3] = u, v, np.cross(u, v)
        matrix = matrix @ rotation
    translation = np.eye(4)
    translation[3, :3] = [float(x) for x in center]
    return matrix @ translation


def _box_collider(stage, path, center, half, *, basis=None, physics_material=None,
                  contact_offset=.002):
    """Add an immovable solid box collider as a *primitive*, not a mesh.

    PhysX's GPU contact filter supports primitive colliders only - a
    ``UsdGeom.Mesh`` collider logs "GPU contact filter ... is not supported" and
    drops out of the filtered contact view, so contacts against a support would
    silently go unreported. Visual surfaces stay textured meshes; the collider is
    a separate cube child (invisible) with the same box, in either case.

    ``basis`` is the collider frame's (u, v, n) axes in world coordinates (only u
    and v are used; see below); the default is world-aligned. ``center`` is the
    world-space centre and ``half`` the half extents along u, v and n.
    """
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    if path is None:
        return None
    cube = UsdGeom.Cube.Define(stage, path + '/collider')
    cube.CreateSizeAttr(2.)
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    matrix = Gf.Matrix4d(*box_collider_matrix(center, half, basis).ravel().tolist())
    xform.AddTransformOp().Set(matrix)
    # Hidden from the renderer only: PhysX ignores UsdGeom visibility, exactly as
    # with the task-asset colliders in add_task_asset. Leaving it visible would
    # draw an untextured grey box over (or z-fighting with) the textured surface.
    UsdGeom.Imageable(cube).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx.CreateContactOffsetAttr(float(contact_offset))
    physx.CreateRestOffsetAttr(0.)
    if physics_material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            physics_material, UsdShade.Tokens.weakerThanDescendants, 'physics')
    return prim


def add_room(stage, room, materials, *, physics_material=None, collide=None):
    """Author floor, walls (with openings), ceiling, baseboard and lights.

    ``materials`` maps a surface key to an already-built USD material.
    ``collide`` selects which surfaces are solid; the default makes the shell
    solid so the robot cannot pass through a wall, and every solid surface is
    reported in the returned record.
    """
    from pxr import Gf, UsdGeom, UsdLux

    collide = ({'floor': True, 'walls': True, 'ceiling': True} if collide is None
               else {'floor': bool(collide.get('floor', True)),
                     'walls': bool(collide.get('walls', True)),
                     'ceiling': bool(collide.get('ceiling', False))})
    solid = []

    sx, sy, sz = room['size_m']
    ox, oy, oz = room['origin_m']
    height = room['ceiling_height_m']
    thickness = room['wall_thickness_m']
    scale = {'floor': [1., 1.], 'wall': [1., 1.], 'ceiling': [1., 1.], 'baseboard': [1., 1.]}
    for key, value in room.get('uv_scale', {}).items():
        scale[key] = value

    root = UsdGeom.Xform.Define(stage, '/World/Room').GetPrim()
    geometry = []

    floor_center = (ox, oy, oz - thickness / 2.)
    floor_prim = uv_box(stage, '/World/Room/Floor',
                        floor_center, (sx / 2. + thickness, sy / 2. + thickness, thickness / 2.),
                        scale=scale['floor'])
    if 'floor' in materials:
        bind(stage, floor_prim, materials['floor'])
    if collide['floor']:
        _box_collider(stage, '/World/Room', floor_center,
                      (sx / 2. + thickness, sy / 2. + thickness, thickness / 2.),
                      physics_material=physics_material)
        solid.append('/World/Room/Floor')
    geometry.append('/World/Room/Floor')

    ceiling_center = (ox, oy, oz + height + thickness / 2.)
    ceiling_prim = uv_box(stage, '/World/Room/Ceiling',
                          ceiling_center, (sx / 2. + thickness, sy / 2. + thickness, thickness / 2.),
                          scale=scale['ceiling'])
    if 'ceiling' in materials:
        bind(stage, ceiling_prim, materials['ceiling'])
    if collide['ceiling']:
        # The collider is named per surface so the record names the solid that exists.
        _box_collider(stage, '/World/Room/CeilingCollider', ceiling_center,
                      (sx / 2. + thickness, sy / 2. + thickness, thickness / 2.),
                      physics_material=physics_material)
        solid.append('/World/Room/CeilingCollider')
    geometry.append('/World/Room/Ceiling')

    by_face = {}
    for opening in room['openings']:
        by_face.setdefault(opening['wall'], []).append(opening)

    for face in WALL_FACES:
        wall = _wall_geometry(room, face)
        quads = _wall_quads(wall, by_face.get(face, []))
        for index, (lo_u, hi_u, lo_v, hi_v) in enumerate(quads):
            corners = []
            for u, v in ((lo_u, lo_v), (hi_u, lo_v), (hi_u, hi_v), (lo_u, hi_v)):
                corners.append(wall['center'] + u * wall['u_axis'] + v * wall['v_axis'])
            path = f'/World/Room/Wall_{face}_{index:02d}'
            prim = uv_quad(stage, path, corners,
                           u_axis=wall['u_axis'], v_axis=wall['v_axis'],
                           scale=scale['wall'])
            if 'wall' in materials:
                bind(stage, prim, materials['wall'])
            if collide['walls']:
                # A wall quad is a mesh (it may be cut around an opening), so its
                # collider is a primitive box the depth of the wall, placed *behind*
                # the visible surface: its inner face coincides with the wall the
                # camera sees, so the robot meets the wall exactly where it is drawn.
                collider_center = (np.asarray(corners).mean(axis=0)
                                   + wall['normal'] * (thickness / 2.))
                half_u = 0.5 * float(np.linalg.norm(np.asarray(corners[1]) - np.asarray(corners[0])))
                half_v = 0.5 * float(np.linalg.norm(np.asarray(corners[3]) - np.asarray(corners[0])))
                _box_collider(stage, path, collider_center,
                              (half_u, half_v, thickness / 2.),
                              basis=(wall['u_axis'], wall['v_axis'], wall['normal']),
                              physics_material=physics_material)
                solid.append(path)
            geometry.append(path)

    # Baseboard: a low strip around the interior.
    baseboard = float(room.get('baseboard_m', 0.))
    if baseboard > 0 and 'baseboard' in materials:
        for face in WALL_FACES:
            wall = _wall_geometry(room, face)
            hu = float(wall['half'][0])
            lower = -float(wall['half'][1])
            corners = []
            for u, v in ((-hu, lower), (hu, lower), (hu, lower + baseboard), (-hu, lower + baseboard)):
                corners.append(wall['center'] + u * wall['u_axis'] + v * wall['v_axis'])
            prim = uv_quad(stage, f'/World/Room/Baseboard_{face}', corners,
                           u_axis=wall['u_axis'], v_axis=wall['v_axis'],
                           scale=[1., 1.], semantic='unit')
            bind(stage, prim, materials['baseboard'])
            geometry.append(f'/World/Room/Baseboard_{face}')

    lights = []
    for light in room['ceiling_lights']:
        path = '/World/Room/Lights/' + light['id']
        if light['shape'] == 'rect':
            lamp = UsdLux.RectLight.Define(stage, path)
            lamp.CreateWidthAttr(float(light['extent_m'][0]))
            lamp.CreateHeightAttr(float(light['extent_m'][1]))
            # A RectLight emits along its own -Z, which is already world-down when
            # the light is unrotated. Rotating it here would aim it at a wall.
            if light.get('tilt_rpy'):
                UsdGeom.Xformable(lamp).AddRotateXYZOp().Set(
                    Gf.Vec3f(*[float(v) for v in light['tilt_rpy']]))
        else:
            lamp = UsdLux.SphereLight.Define(stage, path)
            lamp.CreateRadiusAttr(float(light['extent_m'][0]))
        lamp.CreateIntensityAttr(float(light['intensity']))
        lamp.CreateColorAttr(Gf.Vec3f(*light['color']))
        UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3f(*light['position_m']))
        lights.append(path)

    dome = room['dome']
    dome_path = None
    if dome['enabled']:
        dome_path = '/World/Lights/Ambient'
        light = UsdLux.DomeLight.Define(stage, dome_path)
        light.CreateIntensityAttr(float(dome['intensity']))
        light.CreateColorAttr(Gf.Vec3f(*dome['color']))
        if dome['hdri']:
            light.CreateTextureFileAttr().Set(dome['hdri'])
            light.CreateTextureFormatAttr().Set('latlong')
        # The dome is left visible: in Omniverse a light whose UsdGeom visibility
        # is ``invisible`` is excluded from lighting entirely, not merely hidden.
        # A fully enclosed room occludes it from the camera anyway, so there is
        # nothing to hide.

    return {'paths': geometry, 'lights': lights, 'dome': dome_path,
            'solid_paths': solid, 'collide': collide, 'collision_enabled': bool(solid),
            'note': 'Surfaces named in solid_paths are static colliders; the remaining shell is visual.'}


def add_pedestal(stage, spec, materials, *, prefix='/World/Room'):
    """A small stand for the robot, kept separate from the work table.

    Unlike the decorative shell this *is* solid: the robot's base plate rests
    on it and the contact must exist, so it gets a collider and a physics
    material. It is immovable (kinematic) because it is furniture.
    """
    from pxr import UsdGeom

    position = np.asarray([float(v) for v in spec['position_m']], dtype=float)
    half = np.asarray([float(v) for v in spec['half_extents_m']], dtype=float)
    if np.any(half <= 0):
        raise ValueError('pedestal half_extents_m must be positive')

    path = f'{prefix}/RobotPedestal'
    top = uv_box(stage, path, position, half, scale=spec.get('uv_scale', [1., 1.]))
    if 'top' in materials:
        bind(stage, top, materials['top'])
    physics_material = spec.get('physics_material')
    if physics_material is None:
        from .scene import rigid_material
        physics_material = rigid_material(stage, 'pedestal', spec.get('friction', .8))
    _box_collider(stage, path, position, half, physics_material=physics_material)

    skirt = float(spec.get('skirt_height_m', 0.))
    parts = [{'path': path, 'half_extents_m': half.tolist()}]
    if skirt > 0:
        skirt_half = np.array([half[0], half[1], skirt / 2.])
        skirt_center = position - np.array([0., 0., half[2] + skirt / 2.])
        skirt_path = f'{prefix}/RobotPedestalSkirt'
        skirt_prim = uv_box(stage, skirt_path, skirt_center, skirt_half,
                            scale=spec.get('uv_scale', [1., 1.]))
        if 'side' in materials:
            bind(stage, skirt_prim, materials['side'])
        _box_collider(stage, skirt_path, skirt_center, skirt_half,
                      physics_material=physics_material)
        parts.append({'path': skirt_path, 'half_extents_m': skirt_half.tolist()})

    return {'path': path, 'parts': parts,
            'top_z_m': float(position[2] + half[2]),
            'body_mode': 'kinematic_fixture',
            'note': 'Robot stand; solid furniture, separate from the work table.'}


def room_materials(stage, definitions, prefix='room'):
    """Build every material a room references, returning name -> USD material."""
    built, specs = {}, {}
    for name, raw in (definitions or {}).items():
        spec = validate_material(raw, name)
        specs[name] = {k: v for k, v in spec.items() if k != 'textures'}
        built[name] = textured_material(stage, f'{prefix}_{name}', spec)
    return built, specs
