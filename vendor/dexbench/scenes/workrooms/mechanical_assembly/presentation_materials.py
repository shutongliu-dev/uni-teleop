"""Mechanical task presentation materials, object-space UVs and crease normals.

These functions author only shading in the caller's current USD edit target.
They never save layers, change geometry/transforms, or author physics settings.
Visible SDF meshes may also be colliders: their shading is intentionally included.
Use an independent stronger presentation layer over the original physical asset.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT, load_json

TEXTURE_ROOT = Path(f'{DATA_ROOT}/assets/materials/ambientcg/v1')
CREASE_DEGREES = 30.0
# Full-image RGB means measured from these hash-pinned 1024x1024 Color JPEGs:
# decode RGB/255 with the piecewise sRGB transfer, then average each linear channel.
# Keep this scan reference explicit; authoring does not decode images or require PIL.
COLOR_MAP_REFERENCE = {
    'Metal009': {
        'sha256': 'e066e240ad0eaab6a73fb0f40fa7caa94365160770331babc91de2e8ce6a986f',
        'linear_rgb_mean': (.2885627730724378, .2990684073113796, .314024546327425),
    },
    'Rubber004': {
        'sha256': '95f9182e8564db9a8c6510cd0c9e0656beec0e5a0212cc177e2abf8eb03820b9',
        'linear_rgb_mean': (.02359683715274355, .025271157370157917, .03202456328746486),
    },
}
COLOR_MEAN_METHOD = 'Full RGB/255 image; sRGB <= .04045: /12.92, else ((sRGB+.055)/1.055)^2.4; channel mean.'
COLOR_BLEND = {'Metal009': .8, 'Rubber004': .35}
COLOR_BLEND_OVERRIDES = {'worktop_laminate': .55}
# Scanned colour varies around the retained linear tint; bias limits dark patches.
# color, roughness, metallic, texture family, normal XY strength, UV repeat metres.
MATERIAL_SPECS = {
    'steel_brushed': ((.53, .57, .60), .32, 1., 'Metal009', .10, .08),
    'steel_polished': ((.64, .67, .70), .18, 1., 'Metal009', .025, .08),
    'rubber_graphite': ((.065, .074, .078), .65, 0., 'Rubber004', .15, .035),
    'polymer_teal': ((.045, .24, .29), .43, 0., 'Rubber004', .045, .035),
    'marking_light': ((.82, .86, .82), .60, 0., None, 0., .05),
    'anodized_red': ((.60, .13, .085), .37, .72, 'Metal009', .045, .08),
    'anodized_blue': ((.13, .32, .45), .37, .72, 'Metal009', .045, .08),
    'anodized_yellow': ((.70, .55, .12), .37, .72, 'Metal009', .045, .08),
    'anodized_green': ((.17, .40, .24), .37, .72, 'Metal009', .045, .08),
    'fixture_dark': ((.065, .085, .09), .56, 0., 'Rubber004', .055, .06),
    'worktop_laminate': ((.20, .24, .25), .54, 0., 'Rubber004', .025, .25),
    'polymer_red': ((.60, .13, .085), .46, 0., 'Rubber004', .045, .035),
    'polymer_blue': ((.13, .32, .45), .46, 0., 'Rubber004', .045, .035),
    'polymer_yellow': ((.70, .55, .12), .46, 0., 'Rubber004', .045, .035),
    'polymer_green': ((.17, .40, .24), .46, 0., 'Rubber004', .045, .035),
}


def required_texture_files(texture_root=TEXTURE_ROOT):
    """Shared source manifests and the exact maps referenced by these materials."""
    root = Path(texture_root).resolve()
    return [root / family / name for family in ('Metal009', 'Rubber004')
            for name in ('SOURCE.json', f'{family}_1K-JPG_Color.jpg', f'{family}_1K-JPG_NormalGL.jpg',
                         f'{family}_1K-JPG_Roughness.jpg')]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _texture_evidence(texture_root):
    root = Path(texture_root).resolve()
    evidence = {}
    for family in ('Metal009', 'Rubber004'):
        source_path = root / family / 'SOURCE.json'
        source = load_json(source_path)
        if source.get('assetId') != family or 'CC0' not in source.get('license', ''):
            raise ValueError(f'Unexpected texture provenance: {source_path}')
        expected = {entry['file']: entry['sha256'] for entry in source['files']}
        files = {}
        for suffix in ('Color', 'NormalGL', 'Roughness'):
            path = root / family / f'{family}_1K-JPG_{suffix}.jpg'
            actual = _sha(path)
            if expected.get(path.name) != actual:
                raise ValueError(f'Texture does not match source manifest: {path}')
            files[suffix] = {'path': str(path), 'sha256': actual}
            if suffix == 'Color':
                reference = COLOR_MAP_REFERENCE[family]
                if actual != reference['sha256']:
                    raise ValueError(f'Color map does not match measured linear mean: {path}')
                files[suffix].update(linear_rgb_mean=list(reference['linear_rgb_mean']),
                                     mean_method=COLOR_MEAN_METHOD)
        evidence[family] = {'source': str(source_path), 'source_sha256': _sha(source_path),
            'license': source['license'], 'files': files}
    return evidence


def build_materials(stage, looks_root, texture_root=TEXTURE_ROOT):
    """Return role → material with tinted sRGB colour, raw normals and roughness.

    ``texture_root`` is the shared ambientcg/v1 directory. No textures are copied.
    Besides the requested metal/paint roles, colored polymer variants allow
    ``role_hint='preserve'`` to retain non-metallic red/blue/yellow/green finishes.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    evidence = _texture_evidence(texture_root)  # Validate before authoring.
    root = Sdf.Path(str(looks_root))
    if not root.IsAbsolutePath() or not root.IsPrimPath() or root == Sdf.Path.absoluteRootPath:
        raise ValueError('looks_root must be an absolute non-root prim path')
    UsdGeom.Scope.Define(stage, root)
    result = {}
    for role, (color, roughness, metallic, family, strength, repeat) in MATERIAL_SPECS.items():
        path = root.AppendChild(role)
        material = UsdShade.Material.Define(stage, path)
        surface = UsdShade.Shader.Define(stage, path.AppendChild('Surface'))
        surface.CreateIdAttr('UsdPreviewSurface')
        surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
        surface.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
        surface.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.46)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
        provenance = {'role': role, 'texture_repeat_m': repeat, 'normal_xy_strength': strength,
                      'scope': 'Shading only; no displacement or physics material changes.'}
        if family is not None:
            provenance['texture_source'] = evidence[family]
            blend = COLOR_BLEND_OVERRIDES.get(role, COLOR_BLEND[family])
            provenance['color_blend'] = blend
            provenance['color_tint_linear'] = list(color)
            st = UsdShade.Shader.Define(stage, path.AppendChild('ST'))
            st.CreateIdAttr('UsdPrimvarReader_float2')
            st.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            st.CreateOutput('result', Sdf.ValueTypeNames.Float2)
            for label, target, channel in [('Color', 'diffuseColor', 'rgb'),
                                           ('NormalGL', 'normal', 'rgb'), ('Roughness', 'roughness', 'r')]:
                tex = UsdShade.Shader.Define(stage, path.AppendChild(label))
                tex.CreateIdAttr('UsdUVTexture')
                tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(evidence[family]['files'][label]['path'])
                tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB' if label == 'Color' else 'raw')
                tex.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
                tex.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
                tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), 'result')
                if label == 'Color':
                    # UsdUVTexture decodes sRGB before scale/bias. Expected linear
                    # map colour therefore returns the original tint exactly.
                    mean = evidence[family]['files']['Color']['linear_rgb_mean']
                    scale = tuple(tint * blend / average for tint, average in zip(color, mean)) + (1.,)
                    bias = tuple(tint * (1. - blend) for tint in color) + (0.,)
                    output_type, input_type = Sdf.ValueTypeNames.Float3, Sdf.ValueTypeNames.Color3f
                elif label == 'NormalGL':
                    # Decode [0,1] to tangent normals; reduce XY, retain the original Z.
                    scale, bias = (2 * strength, 2 * strength, 2., 1.), (-strength, -strength, -1., 0.)
                    output_type, input_type = Sdf.ValueTypeNames.Float3, Sdf.ValueTypeNames.Normal3f
                else:
                    scale, bias = (.08, .08, .08, 1.), (roughness - .04,) * 3 + (0.,)
                    output_type = input_type = Sdf.ValueTypeNames.Float
                tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
                tex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
                tex.CreateOutput(channel, output_type)
                surface.CreateInput(target, input_type).ConnectToSource(tex.ConnectableAPI(), channel)
        material.GetPrim().SetCustomDataByKey('mechanicalPresentation', json.dumps(provenance, sort_keys=True))
        result[role] = material
    return result


def _original_finish(prim):
    from pxr import UsdGeom, UsdShade
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    values = {'color': None, 'metallic': 0., 'roughness': .5, 'material': str(material.GetPath()) if material else None}
    if material:
        shader = material.ComputeSurfaceSource()[0]
        if shader:
            for name, target in [('diffuseColor', 'color'), ('metallic', 'metallic'), ('roughness', 'roughness')]:
                inp = shader.GetInput(name)
                if inp and inp.Get() is not None:
                    value = inp.Get()
                    values[target] = list(value) if target == 'color' else float(value)
    if values['color'] is None:
        color = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        values['color'] = list(color[0]) if color else [.55, .55, .55]
    return values


def _preserve_role(finish):
    color = np.asarray(finish['color'], float)
    if color.shape != (3,) or not np.isfinite(color).all():
        raise ValueError('Invalid original material color')
    if color.max() - color.min() > .085 and color.max() > .16:
        names = ('red', 'blue', 'yellow', 'green')
        normalized = color / max(np.linalg.norm(color), 1.e-12)
        name = min(names, key=lambda n: np.linalg.norm(normalized -
            np.asarray(MATERIAL_SPECS['anodized_' + n][0]) /
            np.linalg.norm(MATERIAL_SPECS['anodized_' + n][0])))
        return ('anodized_' if finish['metallic'] >= .45 else 'polymer_') + name
    if color.max() < .16:
        return 'rubber_graphite' if finish['roughness'] >= .7 else 'fixture_dark'
    if finish['metallic'] >= .45:
        return 'steel_polished' if finish['roughness'] < .24 else 'steel_brushed'
    return 'marking_light'


def _role_hint(prim, root, hint, finish):
    from pxr import Sdf
    if isinstance(hint, Mapping):
        matches = []
        for key, value in hint.items():
            if key == '*':
                matches.append((0, value)); continue
            path = Sdf.Path(str(key))
            if not path.IsAbsolutePath():
                path = root.GetPath().AppendPath(path)
            if prim.GetPath().HasPrefix(path):
                matches.append((len(str(path)), value))
        if not matches:
            raise ValueError(f'No presentation role for {prim.GetPath()}')
        hint = max(matches, key=lambda pair: pair[0])[1]
    if hint == 'preserve':
        return _preserve_role(finish)
    if hint not in MATERIAL_SPECS:
        raise ValueError(f'Unknown presentation role: {hint}')
    return hint


def _feed_hash(h, value):
    if isinstance(value, dict):
        for key in sorted(value):
            _feed_hash(h, key); _feed_hash(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(f'{type(value).__name__}:{len(value)}:'.encode())
        for item in value:
            _feed_hash(h, item)
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            array = None
        if array is not None and array.dtype.kind in 'biufc' and array.ndim > 0:
            h.update(str((array.dtype.str, array.shape)).encode()); h.update(array.tobytes())
        else:
            h.update(repr(value).encode())
    h.update(b'\x00')


def _protected_digest(root):
    """All composed non-presentation properties, including every time sample."""
    from pxr import Usd
    h = hashlib.sha256()
    for prim in Usd.PrimRange(root):
        _feed_hash(h, (str(prim.GetPath()), prim.GetTypeName(),
            [a for a in prim.GetAppliedSchemas() if a != 'MaterialBindingAPI']))
        for attr in prim.GetAttributes():
            if attr.GetName() in ('normals', 'primvars:st', 'primvars:st:indices'):
                continue
            _feed_hash(h, (attr.GetName(), str(attr.GetTypeName()), attr.Get(),
                           attr.GetAllMetadata(), attr.GetConnections()))
            for time in attr.GetTimeSamples():
                _feed_hash(h, (time, attr.Get(time)))
        for relationship in prim.GetRelationships():
            if relationship.GetName() == 'material:binding':
                continue
            _feed_hash(h, (relationship.GetName(), relationship.GetTargets(), relationship.GetAllMetadata()))
    return h.hexdigest()


def _physics_material(prim):
    from pxr import UsdShade
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    if not material:
        return None
    physics = material.GetPrim()
    attrs = [a for a in physics.GetAttributes() if a.GetName().startswith(('physics:', 'physx'))]
    schemas = [s for s in physics.GetAppliedSchemas() if 'Physics' in s or 'Physx' in s]
    if not attrs and not schemas:
        return None
    h = hashlib.sha256()
    _feed_hash(h, (str(physics.GetPath()), schemas))
    for attr in attrs:
        _feed_hash(h, (attr.GetName(), attr.Get(), attr.GetAllMetadata()))
    return h.hexdigest()


def _mesh_arrays(mesh):
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or counts.ndim != 1 or indices.ndim != 1 or not len(counts)
            or (counts < 3).any() or counts.sum() != len(indices)
            or (indices < 0).any() or (indices >= len(points)).any()):
        raise ValueError(f'Invalid presentation mesh topology: {mesh.GetPath()}')
    return points, counts, indices


def _face_vectors(points, counts, indices):
    faces = np.split(indices, np.cumsum(counts)[:-1])
    # Newell area vectors support the existing triangle and polygon meshes.
    # Subtract the first corner to avoid cancellation under a large translation.
    polygons = [points[face] - points[face[0]] for face in faces]
    vectors = np.array([np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0) for p in polygons])
    lengths = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all():
        raise ValueError('Nonfinite face cannot receive a reliable shading normal')
    valid = lengths >= 1.e-24
    normals = np.zeros_like(vectors)
    normals[valid] = vectors[valid] / lengths[valid, None]
    # Historical thread visuals contain repeated-index zero-area triangles.
    # Keep every face and vertex. An unrenderable zero-area face receives a
    # finite adjacent-face normal only; this is not a geometric repair.
    if not valid.all():
        incident = defaultdict(list)
        for fi, face in enumerate(faces):
            if valid[fi]:
                for vertex in face:
                    incident[int(vertex)].append(fi)
        for fi in np.flatnonzero(~valid):
            candidates = {other for vertex in faces[fi] for other in incident[int(vertex)]}
            normals[fi] = normals[max(candidates, key=lambda other: (lengths[other], -other))] if candidates else (0, 0, 1)
    return faces, vectors, normals


def _corner_normals(points, counts, indices):
    faces, areas, normals = _face_vectors(points, counts, indices)
    valid = np.linalg.norm(areas, axis=1) >= 1.e-24
    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        if not valid[fi]:
            continue
        for a, b in zip(face, np.roll(face, -1)):
            edge_faces[tuple(sorted((int(a), int(b))))].append(fi)
    adjacent = defaultdict(set)
    cutoff = math.cos(math.radians(CREASE_DEGREES))
    for (a, b), incident in edge_faces.items():
        if len(incident) != 2:
            continue  # Boundaries and non-manifold edges remain hard.
        first, second = incident
        if np.dot(normals[first], normals[second]) >= cutoff - 1.e-12:
            for vertex in (a, b):
                adjacent[(vertex, first)].add(second)
                adjacent[(vertex, second)].add(first)
    values = []
    for fi, face in enumerate(faces):
        if not valid[fi]:
            values.extend([normals[fi]] * len(face)); continue
        for vertex in face:
            connected, pending = {fi}, [fi]
            while pending:
                current = pending.pop()
                for other in adjacent[(int(vertex), current)]:
                    if other not in connected and np.dot(normals[fi], normals[other]) >= cutoff - 1.e-12:
                        connected.add(other); pending.append(other)
            normal = areas[sorted(connected)].sum(axis=0)
            values.append(normal / np.linalg.norm(normal))
    return np.asarray(values), normals, int((~valid).sum())


def _object_frame(stage, root, cache):
    from pxr import UsdGeom
    matrix = np.asarray(cache.GetLocalToWorldTransform(root), float)
    linear = matrix[:3, :3]
    if not np.isfinite(matrix).all() or np.linalg.det(linear) <= 1.e-18:
        raise ValueError('Presentation root has a singular/reflected transform')
    u, _, vt = np.linalg.svd(linear)
    rotation = u @ vt
    return matrix[3, :3], rotation, UsdGeom.GetStageMetersPerUnit(stage)


def _surface_arrays(mesh, cache, origin, rotation, metres, repeat):
    points, counts, indices = _mesh_arrays(mesh)
    matrix = np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()), float)
    linear = matrix[:3, :3] @ rotation.T * metres
    translation = (matrix[3, :3] - origin) @ rotation.T * metres
    if not np.isfinite(matrix).all() or np.linalg.det(linear) <= 1.e-24:
        raise ValueError('Presentation mesh has a singular/reflected transform')
    object_points = points @ linear + translation
    normals, face_normals, degenerate = _corner_normals(object_points, counts, indices)
    # USD normals are mesh-local; inverse-transpose rendering then restores these
    # physical-space crease angles even for nested nonuniform scales.
    normals = normals @ linear.T
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    if mesh.GetOrientationAttr().Get() == 'leftHanded':
        normals *= -1.
    faces = np.split(indices, np.cumsum(counts)[:-1])
    uv = []
    for face, normal in zip(faces, face_normals):
        axis = int(np.argmax(np.abs(normal)))
        # Face-varying box projection in ONE physical object frame; no per-piece
        # bounding-box normalization or per-face arbitrary offset/scale.
        axes = ((1, 2), (0, 2), (0, 1))[axis]
        uv.extend(object_points[face][:, axes] / repeat)
    return np.asarray(uv), normals, len(points), len(counts), degenerate


def style_visible_meshes(stage, object_root, role_hint, materials):
    """Style visible meshes and return JSON-safe inventory/preservation metadata.

    ``role_hint`` is one role, ``'preserve'``, or a path-prefix → role mapping
    (relative to object_root, with optional ``'*'`` fallback). Explicit roles do
    not invoke color heuristics. UVs use face-varying axis projection in a single
    object frame, measured in metres including all nested scales. Hidden and
    guide/proxy meshes are skipped. A mesh named ``collision`` is not excluded
    if it is also the visible default/render surface.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    root = stage.GetPrimAtPath(str(object_root))
    if not root:
        raise ValueError(f'Missing presentation root: {object_root}')
    before = _protected_digest(root)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    frame = _object_frame(stage, root, cache)
    prepared, skipped = [], []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        image = UsdGeom.Imageable(prim)
        if image.ComputeVisibility() == UsdGeom.Tokens.invisible or image.ComputePurpose() not in ('default', 'render'):
            skipped.append({'path': str(prim.GetPath()), 'reason': 'hidden_or_nonrender_purpose'}); continue
        if prim.IsInstanceProxy():
            raise ValueError(f'Cannot author presentation on instance proxy: {prim.GetPath()}')
        mesh = UsdGeom.Mesh(prim)
        if mesh.GetPointsAttr().GetNumTimeSamples() or UsdGeom.PrimvarsAPI(prim).HasPrimvar('normals'):
            raise ValueError(f'Animated points or existing normal primvar require explicit handling: {prim.GetPath()}')
        finish = _original_finish(prim)
        role = _role_hint(prim, root, role_hint, finish)
        if role not in materials or not materials[role] or materials[role].GetPrim().GetStage() != stage:
            raise ValueError(f'Missing material in current stage: {role}')
        uv, normals, vertices, faces, degenerate = _surface_arrays(mesh, cache, *frame, MATERIAL_SPECS[role][-1])
        physics = _physics_material(prim)
        if physics is not None:
            _, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            if relationship.GetName() == 'material:binding':
                raise ValueError('All-purpose binding also supplies physics; author an explicit physics binding before styling')
        prepared.append((mesh, role, finish, uv, normals, vertices, faces, degenerate, physics))
    # All topology, material, and physics checks finish before mutating meshes.
    records = []
    for mesh, role, finish, uv, normals, vertices, faces, degenerate, physics in prepared:
        prim = mesh.GetPrim()
        st = UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st.Set(uv.tolist()); st.BlockIndices()
        mesh.CreateNormalsAttr().Set(normals.tolist())
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[role])
        if _physics_material(prim) != physics:
            raise RuntimeError(f'Physics material resolution changed at {prim.GetPath()}')
        records.append({'path': str(prim.GetPath()), 'role': role, 'original_finish': finish,
            'vertices': vertices, 'faces': faces, 'uv_count': len(uv), 'normal_count': len(normals),
            'existing_zero_area_faces_retained': degenerate,
            'repeat_m': MATERIAL_SPECS[role][-1], 'physics_material_unchanged': True})
    after = _protected_digest(root)
    if before != after:
        raise RuntimeError('Presentation modified a protected geometry/transform/physics property')
    return {'schema': 'mechanical_presentation_materials.v1', 'object_root': str(root.GetPath()),
        'styled_meshes': records, 'skipped_meshes': skipped, 'styled_count': len(records),
        'uv_mapping': 'faceVarying physical object-frame axis projection; shared origin and scale across pieces',
        'crease_degrees': CREASE_DEGREES, 'protected_before_sha256': before,
        'protected_after_sha256': after, 'geometry_transforms_physics_unchanged': True,
        'scope': 'Presentation only; no geometry, displacement, collision or physics material authoring.'}
