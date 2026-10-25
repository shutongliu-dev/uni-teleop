"""Textured, physically-based appearance and room-level geometry for structured scenes.

This module authors *appearance only* plus the non-physical shell of a room
(floor, walls, ceiling, camera-visible trim). It never creates a collider, a
rigid body or a physics material: anything that must be solid is declared in
the scene JSON as a fixture or a static prop and built by ``scene.py``.

Everything here is opt-in. A scene without the ``materials`` and ``room``
blocks produces exactly the stage earlier releases produced, which is what
keeps frozen releases byte-identical.

Texture paths are absolute local files recorded in the shared asset manifest.
Nothing is fetched at build time and no simulator import is needed to validate
a scene: the loader only checks that each declared file exists and is readable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# --- material specification -------------------------------------------------

# Texture maps use ``<slot>_map`` keys; the scalar of the same slot (``color``,
# ``roughness``, ``metallic``, ``opacity``) is the fallback when no map is given,
# so a scalar and a map can never collide on one key. ``base_color`` names the
# albedo map. The earlier bare names (``normal``, ``occlusion``, ``emissive``, and
# ``roughness``/``metalness``/``opacity`` when they hold a path) remain accepted.
TEXTURE_SLOTS = {
    'base_color': ('base_color', 'base_color_map'),
    'roughness': ('roughness_map',),
    'metalness': ('metalness_map', 'metallic_map'),
    'normal': ('normal_map', 'normal'),
    'occlusion': ('occlusion_map', 'ao_map', 'occlusion'),
    'opacity': ('opacity_map',),
    'emissive': ('emissive_map', 'emissive'),
}
# Bare slot names that are a map when given a path and a scalar when given a number.
DUAL_KEYS = {'roughness': 'roughness', 'metalness': 'metalness', 'opacity': 'opacity'}
TEXTURE_KEYS = tuple(TEXTURE_SLOTS)
_KNOWN_MAP_KEYS = {key for keys in TEXTURE_SLOTS.values() for key in keys}


def _resolve(path):
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f'Scene material texture is missing: {resolved}')
    return resolved


def validate_material(spec, name):
    """Check a material spec without importing OpenUSD; returns a normalised copy."""
    if not isinstance(spec, dict):
        raise ValueError(f'Material {name!r} must be an object')
    kind = spec.get('kind', 'color')
    if kind == 'color':
        color = spec.get('color', [.55, .60, .65])
        if len(color) != 3 or not all(np.isfinite(float(c)) for c in color):
            raise ValueError(f'Material {name!r} color must be three finite numbers')
        return {'kind': 'color', 'name': name, 'color': [float(c) for c in color],
                'roughness': float(spec.get('roughness', .55)),
                'metallic': float(spec.get('metallic', 0.)), 'textures': {}}
    if kind != 'textured':
        raise ValueError(f'Unknown material kind {kind!r} for {name!r}')
    unknown = sorted(key for key in spec if key.endswith('_map') and key not in _KNOWN_MAP_KEYS)
    if unknown:
        # A misspelt map key would otherwise be dropped silently.
        raise ValueError(f'Material {name!r} has unknown texture keys {unknown}; '
                         f'use one of {sorted(_KNOWN_MAP_KEYS)}')
    textures = {}
    for slot, keys in TEXTURE_SLOTS.items():
        for key in keys:
            if spec.get(key):
                textures[slot] = str(_resolve(spec[key]))
                break
        else:
            legacy = DUAL_KEYS.get(slot)
            if legacy and isinstance(spec.get(legacy), str):
                textures[slot] = str(_resolve(spec[legacy]))
    if not textures:
        raise ValueError(f'Textured material {name!r} declares no texture files')

    def scalar(key, default):
        value = spec.get(key, default)
        return float(default if isinstance(value, str) else value)

    scale = spec.get('uv_scale_m', spec.get('uv_scale', [1., 1.]))
    if len(scale) != 2 or not all(np.isfinite(float(s)) and float(s) > 0 for s in scale):
        raise ValueError(f'Material {name!r} uv scale must be two positive numbers')
    return {'kind': 'textured', 'name': name, 'textures': textures,
            'uv_scale': [float(s) for s in scale],
            'uv_scale_semantics': spec.get('uv_scale_semantics', 'tiles_per_meter'),
            'roughness': scalar('roughness', .5),
            'metallic': scalar('metallic', 0.),
            'opacity': scalar('opacity', 1.),
            'normal_strength': float(spec.get('normal_strength', 1.)),
            'color': [float(c) for c in spec.get('color', [1., 1., 1.] if 'base_color' in textures
                                                  else [.6, .6, .6])],
            'source': spec.get('source'), 'license': spec.get('license')}


# --- USD appearance ---------------------------------------------------------

def textured_material(stage, name, spec):
    """Build a UsdPreviewSurface material with texture inputs.

    UsdPreviewSurface is used rather than a raw MDL so the appearance stays
    renderer-portable and identical to the flat-colour path the frozen
    releases already use.
    """
    from pxr import Gf, Sdf, UsdShade

    path = '/World/Looks/' + name
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, path + '/Surface')
    surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*spec['color']))
    surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(spec['roughness']))
    surface.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(float(spec['metallic']))
    surface.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(
        1.0 if spec['kind'] == 'color' else float(spec.get('opacity', 1.)))

    if spec['kind'] == 'textured':
        reader = _st_reader(stage, path + '/stReader')
        surface.CreateInput('st', Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(
            reader.ConnectableAPI(), 'result')
        _bind_textures(stage, path, surface, spec, reader)

    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
    return material


def _st_reader(stage, path):
    from pxr import Sdf, UsdShade
    reader = UsdShade.Shader.Define(stage, path)
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    fallback = reader.CreateInput('fallback', Sdf.ValueTypeNames.Float2)
    fallback.Set((0., 0.))
    return reader


def _texture_shader(stage, path, file_path, color_space, *, wrap='repeat'):
    from pxr import Sdf, UsdShade
    shader = UsdShade.Shader.Define(stage, path)
    shader.CreateIdAttr('UsdUVTexture')
    shader.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(file_path)
    shader.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(color_space)
    shader.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set(wrap)
    shader.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set(wrap)
    return shader


def _bind_textures(stage, base, surface, spec, reader):
    from pxr import Gf, Sdf

    def texture(slot, space):
        shader = _texture_shader(stage, f'{base}/{slot}', spec['textures'][slot], space)
        shader.CreateInput('st', Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(
            reader.ConnectableAPI(), 'result')
        return shader

    def connect_float(slot, target_input, channel='r'):
        shader = texture(slot, 'raw')
        shader.CreateOutput(channel, Sdf.ValueTypeNames.Float)
        surface.CreateInput(target_input, Sdf.ValueTypeNames.Float).ConnectToSource(
            shader.ConnectableAPI(), channel)

    textures = spec['textures']
    if 'base_color' in textures:
        shader = texture('base_color', 'sRGB')
        shader.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
            shader.ConnectableAPI(), 'rgb')
    if 'opacity' in textures:
        connect_float('opacity', 'opacity')
    if 'roughness' in textures:
        connect_float('roughness', 'roughness')
    if 'metalness' in textures:
        connect_float('metalness', 'metallic')
    if 'occlusion' in textures:
        connect_float('occlusion', 'occlusion')
    if 'emissive' in textures:
        shader = texture('emissive', 'sRGB')
        shader.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        surface.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
            shader.ConnectableAPI(), 'rgb')
    if 'normal' in textures:
        # OpenGL-convention (+Y) tangent-space map, remapped from [0, 1] to [-1, 1].
        strength = float(spec.get('normal_strength', 1.))
        shader = texture('normal', 'raw')
        shader.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(2. * strength, 2. * strength, 2., 1.))
        shader.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(-strength, -strength, -1., 0.))
        shader.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        surface.CreateInput('normal', Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            shader.ConnectableAPI(), 'rgb')


def bind(stage, prim, material):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


# --- UV geometry ------------------------------------------------------------

def _uv(prim, coordinates, *, scale=(1., 1.), semantic='tiles_per_meter', interpolation='faceVarying'):
    """Author an ``st`` primvar; coordinates are a list matching the vertex order."""
    from pxr import Sdf, UsdGeom, Vt
    array = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    if semantic == 'tiles_per_meter':
        array = array * np.asarray(scale, dtype=np.float32)
    primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        'st', Sdf.ValueTypeNames.TexCoord2fArray, interpolation)
    primvar.Set(Vt.Vec2fArray.FromNumpy(array.reshape(-1, 2)))


def uv_quad(stage, path, corners, *, u_axis, v_axis, scale=(1., 1.), semantic='tiles_per_meter',
            normal=(0., 0., 1.), double_sided=False):
    """A single textured quad with explicit UVs; ``corners`` are four xyz points.

    ``u_axis``/``v_axis`` give the world-space direction of increasing U and V,
    so tiling stays consistent when a wall is mirrored or rotated.
    """
    from pxr import Gf, UsdGeom, Vt
    corners = np.asarray(corners, dtype=float).reshape(4, 3)
    u_axis = np.asarray(u_axis, dtype=float)
    v_axis = np.asarray(v_axis, dtype=float)
    origin = corners[0]
    coordinates = []
    for corner in corners:
        delta = corner - origin
        coordinates.append([float(np.dot(delta, u_axis)), float(np.dot(delta, v_axis))])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(corners.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.array([4], dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.array([0, 1, 2, 3], dtype=np.int32)))
    if not double_sided:
        mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(corners.astype(np.float32)))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(bool(double_sided))
    _uv(mesh.GetPrim(), coordinates, scale=scale, semantic=semantic)
    return mesh.GetPrim()


def uv_box(stage, path, position, half, *, scale=(1., 1.), semantic='tiles_per_meter'):
    """A 24-vertex textured box: every face gets independent, unwrapped UVs.

    USD's ``Cube`` has no UVs, so a tiling texture on it would stretch. Each
    face is authored as its own quad with metric UVs derived from its own axes.
    """
    from pxr import UsdGeom, Vt
    position = np.asarray(position, dtype=float)
    half = np.asarray(half, dtype=float)
    lo, hi = position - half, position + half
    faces = []
    axes = {}
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for sign in (-1, 1):
            corners = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                point = position.copy()
                point[axis] = hi[axis] if sign > 0 else lo[axis]
                point[other[0]] = lo[other[0]] + du * (hi[other[0]] - lo[other[0]])
                point[other[1]] = lo[other[1]] + dv * (hi[other[1]] - lo[other[1]])
                corners.append(point)
            direction = np.zeros(3)
            direction[axis] = float(sign)
            axes[len(faces)] = (direction, [np.zeros(3) for _ in range(2)])
            u_axis = np.zeros(3)
            u_axis[other[0]] = 1.
            v_axis = np.zeros(3)
            v_axis[other[1]] = 1.
            faces.append((corners, u_axis, v_axis))

    points, counts, indices, coordinates = [], [], [], []
    for corners, u_axis, v_axis in faces:
        base = len(points)
        origin = np.asarray(corners[0])
        for corner in corners:
            points.append(corner.tolist())
            delta = np.asarray(corner) - origin
            coordinates.append([float(np.dot(delta, u_axis)), float(np.dot(delta, v_axis))])
        indices.extend([base, base + 1, base + 2, base + 3])
        counts.append(4)

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(counts, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(indices, dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray([lo, hi], dtype=np.float32)))
    _uv(mesh.GetPrim(), coordinates, scale=scale, semantic=semantic)
    return mesh.GetPrim()
