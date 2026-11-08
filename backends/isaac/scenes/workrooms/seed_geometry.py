"""CPU-only authoring helpers for immutable workroom asset variants.

No simulator imports, live object writes, release selection or policy logic.
Original collision APIs and physical materials remain attached to each prim.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bench.paths import load_json, portable
# Keep author-only USD helpers lazy; case composition uses only JSON/hash math.
def affine_mass(*args, **kwargs):
    from scenes.workrooms.living_room.seed_geometry import affine_mass as implementation
    return implementation(*args, **kwargs)


def collision_bounds(*args, **kwargs):
    from scenes.workrooms.living_room.seed_geometry import collision_bounds as implementation
    return implementation(*args, **kwargs)


def mass_inventory(*args, **kwargs):
    from scenes.workrooms.living_room.seed_geometry import mass_inventory as implementation
    return implementation(*args, **kwargs)


def pose(*args, **kwargs):
    from scenes.workrooms.living_room.seed_geometry import pose as implementation
    return implementation(*args, **kwargs)


def rotation(*args, **kwargs):
    from scenes.workrooms.living_room.seed_geometry import rotation as implementation
    return implementation(*args, **kwargs)


def set_mass(*args, **kwargs):
    from scenes.workrooms.living_room.seed_geometry import set_mass as implementation
    return implementation(*args, **kwargs)


def transform_group(*args, **kwargs):
    from scenes.workrooms.living_room.seed_geometry import transform_group as implementation
    return implementation(*args, **kwargs)



def file_hash(path):
    from bench.isaacsim.validation_epoch import file_sha256
    return file_sha256(path)


def canonical_hash(value):
    # Identities hash data:// references, never this machine's data root.
    return hashlib.sha256(json.dumps(portable(value), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(portable(value), indent=2, sort_keys=True, allow_nan=False)+'\n')


def checked_seed(seed):
    if type(seed) is not int or not 0 <= seed <= 9:
        raise ValueError('A physical case requires an integer seed in 0..9')
    return seed


def check_parent_release(parent):
    """Verify the frozen task/source files before deriving new inputs."""
    parent = Path(parent).resolve()
    manifest = load_json(parent/'release_manifest.json')
    if not manifest.get('files_sha256'):
        raise ValueError('Parent release has no file identity inventory')
    for relative, expected in manifest['files_sha256'].items():
        path = (parent/relative).resolve()
        if not path.is_relative_to(parent) or file_hash(path) != expected:
            raise ValueError('Parent release file differs: '+relative)
    return manifest


def planar_offset(seed, key, *, xy_m=(.012, .012), yaw_deg=7.):
    """Version-independent deterministic bounded offset; zero is canonical."""
    checked_seed(seed)
    if seed == 0:
        return [0., 0.], 0.
    raw = hashlib.sha256(('workroom_seed_geometry.v1:'+key+':'+str(seed)).encode()).digest()
    values = [int.from_bytes(raw[i:i+4], 'big')/(2**32-1)*2-1 for i in (0, 4, 8)]
    return [values[0]*xy_m[0], values[1]*xy_m[1]], math.radians(values[2]*yaw_deg)


def bake_body_scale(stage, body_path, scale):
    """Bake an affine map about a body origin, retaining child transforms.

    Each mesh is transformed by T^-1 A T in its own frame. Thus existing
    conditioning scales, rotations and nesting are applied exactly once. This
    supports mesh geometry (and nonphysical curves); primitive geometry fails
    closed instead of silently changing a collision approximation.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Three positive finite axis scales required')
    body = stage.GetPrimAtPath(body_path)
    if not body:
        raise ValueError('Missing authored body '+body_path)
    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(body).GetInverse()
    a = np.diag([*scale, 1.])
    for prim in Usd.PrimRange(body):
        if prim == body:
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Nested rigid body must be authored independently')
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.BasisCurves):
            geom = UsdGeom.Mesh(prim) if prim.IsA(UsdGeom.Mesh) else UsdGeom.BasisCurves(prim)
            t = np.array(cache.GetLocalToWorldTransform(prim)*inverse, dtype=float).T
            local = np.linalg.inv(t)@a@t
            points = np.asarray(geom.GetPointsAttr().Get(), dtype=float)
            mapped = points@local[:3, :3].T+local[:3, 3]
            geom.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mapped.astype(np.float32)))
            if geom.GetNormalsAttr().HasAuthoredValue():
                normals = np.asarray(geom.GetNormalsAttr().Get(), dtype=float)@np.linalg.inv(local[:3, :3])
                lengths = np.linalg.norm(normals, axis=1)
                if not (lengths > 0).all():
                    raise ValueError('Invalid source normals')
                normals /= lengths[:, None]
                geom.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
            geom.CreateExtentAttr([Gf.Vec3f(*mapped.min(axis=0)), Gf.Vec3f(*mapped.max(axis=0))])
        elif prim.IsA(UsdGeom.Gprim):
            raise ValueError('Explicit mesh bake required for '+str(prim.GetPath()))
    if body.HasAPI(UsdPhysics.RigidBodyAPI):
        api = UsdPhysics.MassAPI(body)
        mass, com, diagonal = api.GetMassAttr().Get(), api.GetCenterOfMassAttr().Get(), api.GetDiagonalInertiaAttr().Get()
        if mass is None or com is None or diagonal is None:
            raise ValueError('Scale requires original explicit mass, COM and inertia')
        r = rotation(api.GetPrincipalAxesAttr().Get())
        set_mass(body, *affine_mass(float(mass), np.asarray(com), r@np.diag(diagonal)@r.T, scale))


def transform_body(stage, path, translation_xy_m, yaw_rad=0., *, pivot_m=None):
    if pivot_m is None:
        pivot_m = pose(stage.GetPrimAtPath(path))['position_m']
    transform_group(stage, {'pivot_m': pivot_m, 'translation_xy_m': translation_xy_m,
                           'yaw_rad': yaw_rad, 'prim_paths': [path]})


def finish_texture(output, seed):
    """Project-authored subtle periodic finish, preserving category colours."""
    from PIL import Image
    checked_seed(seed)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:128, :128]
    mode = (seed-1) % 3
    if mode == 0:
        value = .88+.12*(.5+.5*np.sin(2*np.pi*x/16))
    elif mode == 1:
        value = .88+.12*((x//8+y//8) % 2)
    else:
        value = .88+.12*(.5+.5*np.cos(2*np.pi*(x+y)/16))
    rgba = np.repeat(np.rint(value[..., None]*255).astype(np.uint8), 3, axis=2)
    path = output/'surface_finish.png'; Image.fromarray(rgba, 'RGB').save(path)
    (output/'TEXTURE_LICENSE.txt').write_text('Original procedural periodic surface finish. CC0-1.0.\nNo physical friction or restitution changes.\n')
    return path


def apply_surface_texture(stage, body_paths, texture_path, seed, *, material_root=None):
    """Bind texture to render materials only; retain labels and physics binding."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    import posixpath
    body_paths = list(body_paths)
    if not body_paths:
        raise ValueError('At least one authored body is required')
    common = posixpath.commonpath(body_paths)
    if common in body_paths:
        common = posixpath.dirname(common)
    material_root = material_root or common + '/SeedVariantMaterials'
    if material_root in ('/SeedVariantMaterials', '/World/SeedVariantMaterials'):
        raise ValueError('Materials must live inside the referenced task subtree')
    materials = {}; changed = []
    for body_path in body_paths:
        body = stage.GetPrimAtPath(body_path)
        for prim in Usd.PrimRange(body):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            name = str(prim.GetPath()).lower()
            if any(token in name for token in ('label', 'mark', 'number', 'glyph', 'icon', 'print', 'tick', 'visualdetails')):
                continue
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            source = bound.ComputeSurfaceSource()[0] if bound else None
            display = UsdGeom.Mesh(prim).GetDisplayColorAttr().Get()
            color = source.GetInput('diffuseColor').Get() if source else (display[0] if display else None)
            if color is None or (source and source.GetInput('diffuseColor').HasConnectedSource()):
                continue
            key = str(bound.GetPath()) if source else 'display:'+str(list(color))
            if key not in materials:
                path = material_root + '/finish_'+str(len(materials))
                material = UsdShade.Material.Define(stage, path)
                shader = UsdShade.Shader.Define(stage, path+'/Surface'); shader.CreateIdAttr('UsdPreviewSurface')
                for field, default in (('roughness', .4), ('metallic', 0.), ('opacity', 1.)):
                    value = source.GetInput(field).Get() if source else None
                    shader.CreateInput(field, Sdf.ValueTypeNames.Float).Set(default if value is None else value)
                uv = UsdShade.Shader.Define(stage, path+'/UV'); uv.CreateIdAttr('UsdPrimvarReader_float2')
                uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
                tex = UsdShade.Shader.Define(stage, path+'/Texture'); tex.CreateIdAttr('UsdUVTexture')
                tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(texture_path).resolve())))
                tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw')
                tex.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
                tex.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
                tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color, 1.))
                tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
                shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
                materials[key] = material
            mesh = UsdGeom.Mesh(prim); points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            spans = np.ptp(points, axis=0); axes = np.argsort(spans)[-2:]
            # Vertex UVs remain deterministic and continuous across triangulation.
            uv_values = points[:, axes]/.024
            pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex')
            pv.Set(Vt.Vec2fArray.FromNumpy(uv_values.astype(np.float32)))
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[key])
            changed.append(str(prim.GetPath()))
    return {'schema': 'workroom_procedural_finish.v1', 'variant': ('fine_stripe','woven_check','diagonal_weave')[(seed-1)%3],
            'texture_path': str(Path(texture_path).resolve()), 'texture_sha256': file_hash(texture_path),
            'license': 'CC0-1.0', 'category_colours_preserved': True,
            'physical_materials_changed': False, 'visual_mesh_paths': changed}


def feature(label, body_id, prim_path, local_position_m, *, group=None, direction=(0., 0., 1.),
            kind='actual_surface', required=True, reference_state='initial'):
    return {'label': label, 'body_id': body_id, 'prim_path': prim_path,
            'local_position_m': list(local_position_m), 'approach_direction_local': list(direction),
            'approach_distance_m': .10, 'type': kind, 'reference_state': reference_state,
            'operation_group': group or label, 'required': required,
            'scope': 'geometric_reference_not_grasp_center'}
