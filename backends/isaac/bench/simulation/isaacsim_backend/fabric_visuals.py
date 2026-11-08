"""Read-only fabric render skin selected by a pinned visual profile.

A render-only alternative to ClosedClothVisual. The native midsurface is
Loop-subdivided by a fixed sparse operator, a closed skin of the unchanged
visual thickness is offset along smooth normals, and baked pattern-space
textures carry the fabric scan, garment construction, hems and wear. Render
vertices are split once per UV atlas panel so points, normals and texture
coordinates all stay vertex-interpolated. Like ClosedClothVisual it has no
physics schemas, mass, collider or attachment, and syncing it never writes the
simulation mesh.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .cloth_visuals import boundary_edges
from bench.paths import load_json

PROFILE_SCHEMA = 'fabric_visual_profile.v1'
TEXTURE_ROLES = ('albedo', 'normal', 'roughness', 'occlusion', 'metallic')
ATLASES = (None, 'front_back', 'two_sided')
OMNISURFACE_INPUTS = {'diffuse_reflection_roughness': (0., 1.), 'specular_reflection_weight': (0., 1.),
                      'specular_reflection_ior': (1., 3.), 'specular_retro_reflection_weight': (0., 1.),
                      'specular_retro_reflection_roughness': (0., 1.)}


def cloth_geometry_digest(rest_vertices, triangles):
    """Identity of the material rest mesh a profile was baked for."""
    rest = np.ascontiguousarray(rest_vertices, dtype=np.float64)
    tri = np.ascontiguousarray(triangles, dtype=np.int32)
    value = hashlib.sha256()
    value.update(json.dumps([list(rest.shape), list(tri.shape)]).encode())
    value.update(rest.tobytes())
    value.update(tri.tobytes())
    return value.hexdigest()


def _loop_level(triangles, count):
    """One Loop subdivision step as (operator, triangles), new points = operator @ points."""
    from scipy import sparse

    tri = np.asarray(triangles, dtype=np.int64)
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    opposite = np.concatenate([tri[:, 2], tri[:, 0], tri[:, 1]])
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                        return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    if np.any(counts > 2):
        raise ValueError('Loop subdivision requires a manifold triangle surface')
    edge_count = len(unique)
    boundary = counts == 1
    rows, cols, values = [], [], []
    for k in range(2):
        rows.append(count + np.arange(edge_count))
        cols.append(unique[:, k])
        values.append(np.where(boundary, .5, .375))
    interior = ~boundary[inverse]
    rows.append(count + inverse[interior])
    cols.append(opposite[interior])
    values.append(np.full(int(interior.sum()), .125))
    first = np.concatenate([unique[:, 0], unique[:, 1]])
    second = np.concatenate([unique[:, 1], unique[:, 0]])
    valence = np.bincount(first, minlength=count)
    rim = unique[boundary]
    on_rim = np.zeros(count, dtype=bool)
    on_rim[rim.ravel()] = True
    beta = np.where(valence > 3, 3. / (8. * np.maximum(valence, 1)), 3. / 16.)
    select = ~on_rim[first]
    rows.append(first[select])
    cols.append(second[select])
    values.append(beta[first[select]])
    inner = np.flatnonzero(~on_rim)
    rows.append(inner)
    cols.append(inner)
    values.append(1. - valence[inner] * beta[inner])
    rim_degree = np.bincount(rim.ravel(), minlength=count)
    outer = np.flatnonzero(on_rim)
    rows.append(outer)
    cols.append(outer)
    values.append(np.full(len(outer), .75))
    for a, b in ((0, 1), (1, 0)):
        rows.append(rim[:, a])
        cols.append(rim[:, b])
        values.append(.25 / rim_degree[rim[:, a]])
    operator = sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(count + edge_count, count))
    if not np.allclose(np.asarray(operator.sum(axis=1)).ravel(), 1., atol=1.e-12):
        raise RuntimeError('Loop subdivision rows must be affine')
    index = count + inverse.reshape(3, -1).T
    a, b, c = tri.T
    ab, bc, ca = index.T
    children = np.concatenate([np.stack([a, ab, ca], axis=1), np.stack([ab, b, bc], axis=1),
                               np.stack([ca, bc, c], axis=1), np.stack([ab, bc, ca], axis=1)])
    return operator, children.astype(np.int32)


def loop_subdivision(triangles, count, levels):
    """Fixed sparse operator mapping native midsurface points to the render midsurface."""
    from scipy import sparse

    if levels not in (0, 1, 2):
        raise ValueError('Fabric skin subdivision must be 0, 1 or 2 levels')
    operator = sparse.identity(count, format='csr')
    current = np.asarray(triangles, dtype=np.int32)
    for _ in range(levels):
        step, current = _loop_level(current, operator.shape[0])
        operator = (step @ operator).tocsr()
    return operator, current


def vertex_normals(points, triangles):
    normal = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                      points[triangles[:, 2]] - points[triangles[:, 0]])
    normals = np.zeros_like(points)
    for index in range(3):
        np.add.at(normals, triangles[:, index], normal)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1.e-14):
        raise ValueError('Cannot render a fabric skin vertex with a degenerate normal')
    return normals / lengths[:, None]


class SkinTopology:
    """Subdivided closed skin whose render vertices are split per UV atlas panel.

    A sewn two-panel garment ('front_back') maps faces owned by the rest z < 0
    panel to the right atlas half, mirrored in u, so both outer faces keep
    right-handed tangent frames. A single sheet ('two_sided') maps its bottom
    skin there, so the side shown after a fold is right-handed too. Normals are
    computed on the unsplit skin, so panel seams shade smoothly.
    """

    def __init__(self, triangles, vertex_count, levels, rest_vertices=None, atlas=None):
        if atlas not in ATLASES:
            raise ValueError('Unknown fabric UV atlas')
        self.operator, self.triangles = loop_subdivision(triangles, vertex_count, levels)
        self.count = n = self.operator.shape[0]
        edges = boundary_edges(self.triangles)
        side = np.empty((len(edges) * 2, 3), dtype=np.int32)
        a, b = edges.T
        side[::2] = np.stack([b, a, a + n], axis=1)
        side[1::2] = np.stack([b, a + n, b + n], axis=1)
        self.skin_faces = np.concatenate([self.triangles, self.triangles[:, ::-1] + n, side])
        m = len(self.triangles)
        panel = np.zeros(len(self.skin_faces), dtype=np.int64)
        if atlas == 'front_back':
            # Side walls belong to the triangle that owns their boundary edge.
            owner = {(int(p), int(q)): k for k, row in enumerate(self.triangles)
                     for p, q in ((row[0], row[1]), (row[1], row[2]), (row[2], row[0]))}
            owners = np.concatenate([np.arange(m), np.arange(m), np.repeat(
                np.asarray([owner[(int(p), int(q))] for p, q in edges], dtype=np.int64), 2)])
            rest_z = self.operator @ np.asarray(rest_vertices, dtype=np.float64)[:, 2]
            panel = (rest_z[self.triangles].mean(axis=1) < 0.).astype(np.int64)[owners]
        elif atlas == 'two_sided':
            panel[m:2 * m] = 1
        keys = self.skin_faces.astype(np.int64) * 2 + panel[:, None]
        unique, inverse = np.unique(keys.ravel(), return_inverse=True)
        self.faces = inverse.reshape(-1, 3).astype(np.int32)
        self.source = unique // 2
        self.panel = (unique % 2).astype(np.int8)
        self.atlas = atlas

    def skin(self, midsurface, thickness_m):
        """Unsplit closed skin points and smooth vertex normals."""
        mid = self.operator @ np.asarray(midsurface, dtype=np.float64)
        offset = vertex_normals(mid, self.triangles) * thickness_m / 2
        points = np.concatenate([mid + offset, mid - offset])
        return points, vertex_normals(points, self.skin_faces)

    def render(self, midsurface, thickness_m):
        points, normals = self.skin(midsurface, thickness_m)
        return points[self.source], normals[self.source]

    def uv(self, rest_vertices, frame):
        """Render-vertex UVs in the profile frame; panel 1 samples the mirrored right half."""
        origin = np.asarray(frame['origin_xy_m'], dtype=np.float64)
        size = np.asarray(frame['size_xy_m'], dtype=np.float64)
        rest = self.operator @ np.asarray(rest_vertices, dtype=np.float64)[:, :2]
        uv = (rest[self.source % self.count] - origin) / size
        if self.atlas is not None:
            uv[:, 0] = np.where(self.panel == 1, 1. - .5 * uv[:, 0], .5 * uv[:, 0])
        return uv.astype(np.float32)


def _finite_pair(value, positive=False):
    pair = np.asarray(value, dtype=np.float64)
    return pair.shape == (2,) and np.isfinite(pair).all() and (not positive or np.all(pair > 0))


def load_visual_profile(pin):
    """Read a hash-pinned profile and verify every texture it names.

    Texture names are relative to the profile directory; the result carries
    their absolute paths. Any changed byte, unknown role or out-of-range
    shader value is rejected instead of rendering an unpinned appearance.
    """
    from bench.isaacsim.release import checked

    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise ValueError('A fabric visual profile is pinned by exactly its path and sha256')
    path = Path(pin['path'])
    if not path.is_absolute():
        raise ValueError('Fabric visual profile path must be absolute')
    checked(path, pin['sha256'])
    profile = load_json(path)
    frame = profile.get('uv_frame', {})
    textures = profile.get('textures', {})
    if (profile.get('schema') != PROFILE_SCHEMA or profile.get('render_only') is not True
            or profile.get('physics_changed') is not False
            or profile.get('shader') not in ('preview', 'omnisurface')
            or profile.get('subdivision_levels') not in (0, 1, 2)
            or frame.get('atlas') not in ATLASES
            or not _finite_pair(frame.get('origin_xy_m')) or not _finite_pair(frame.get('size_xy_m'), True)):
        raise ValueError('Invalid fabric visual profile')
    if (not isinstance(textures, dict) or 'albedo' not in textures or not set(textures) <= set(TEXTURE_ROLES)
            or set(profile.get('texture_sha256', {})) != set(textures)):
        raise ValueError('Fabric visual profile needs hash-pinned albedo and known texture roles only')
    resolved = {}
    for role, name in textures.items():
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError('Fabric textures must sit beside their profile')
        resolved[role] = str(path.parent / relative)
        checked(resolved[role], profile['texture_sha256'][role])
    binding = profile.get('cloth_binding', {})
    dye = np.asarray(binding.get('dye_linear_rgb', []), dtype=np.float64)
    if (set(binding) != {'cloth_geometry_sha256', 'dye_linear_rgb'} or dye.shape != (3,)
            or not np.isfinite(dye).all() or len(binding['cloth_geometry_sha256']) != 64):
        raise ValueError('Fabric visual profile must bind one cloth geometry and dye')
    ior = profile.get('ior', 1.5)
    if not np.isfinite(ior) or not 1. <= ior <= 3.:
        raise ValueError('Fabric preview IOR must be finite and between 1 and 3')
    options = profile.get('omnisurface', {})
    for key, value in options.items():
        if key == 'specular_retro_reflection_color':
            color = np.asarray(value, dtype=np.float64)
            if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0) or np.any(color > 1):
                raise ValueError('Fabric sheen colour must be three values in [0, 1]')
        elif key not in OMNISURFACE_INPUTS or not (
                np.isfinite(value) and OMNISURFACE_INPUTS[key][0] <= value <= OMNISURFACE_INPUTS[key][1]):
            raise ValueError('Unknown or out-of-range OmniSurface fabric input: ' + key)
    return {**profile, 'textures': resolved, 'pin': dict(pin)}


class FabricClothVisual:
    """Independent world-space fabric skin driven only by solver readback."""

    def __init__(self, stage, sim_path, name, triangles, rest_vertices, thickness_m, profile,
                 initial_world_vertices, dye_linear_rgb=None):
        from pxr import Sdf, UsdGeom, UsdShade, Vt

        if not name.replace('_', '').isalnum():
            raise ValueError('Invalid cloth visual name')
        self.thickness_m = float(thickness_m)
        if not np.isfinite(self.thickness_m) or self.thickness_m <= 0:
            raise ValueError('Visible cloth thickness must be positive and finite')
        rest = np.asarray(rest_vertices, dtype=np.float64)
        binding = profile['cloth_binding']
        if cloth_geometry_digest(rest_vertices, triangles) != binding['cloth_geometry_sha256']:
            raise ValueError('Fabric visual profile was baked for a different cloth rest mesh')
        if dye_linear_rgb is not None and not np.array_equal(
                np.asarray(dye_linear_rgb, dtype=np.float64), np.asarray(binding['dye_linear_rgb'], dtype=np.float64)):
            raise ValueError('Fabric visual profile dye differs from the frozen cloth colour')
        frame = profile['uv_frame']
        self.topology = SkinTopology(triangles, len(rest), profile['subdivision_levels'],
                                     rest_vertices=rest, atlas=frame.get('atlas'))
        self.path = '/World/ClothVisuals/' + name
        UsdGeom.Xform.Define(stage, '/World/ClothVisuals')
        self.mesh = UsdGeom.Mesh.Define(stage, self.path)
        # Sibling graphics: the simulator's guide purpose cannot hide this skin.
        UsdGeom.Imageable(stage.GetPrimAtPath(sim_path)).CreatePurposeAttr('guide')
        self.mesh.CreateSubdivisionSchemeAttr('none')
        self.mesh.CreateDoubleSidedAttr(False)
        self.mesh.CreateFaceVertexCountsAttr([3] * len(self.topology.faces))
        self.mesh.CreateFaceVertexIndicesAttr(self.topology.faces.ravel().tolist())
        self.mesh.SetNormalsInterpolation('vertex')
        UsdGeom.PrimvarsAPI(self.mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(self.topology.uv(rest, frame)))
        if profile['shader'] == 'omnisurface':
            material = self._omnisurface(stage, name, profile['textures'], profile.get('omnisurface', {}))
        else:
            material = self._preview(stage, name, profile['textures'], profile.get('ior'))
        UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(material)
        self.sync(rest if initial_world_vertices is None else initial_world_vertices)
        self.metadata = {'path': self.path, 'thickness_m': self.thickness_m,
            'source': 'Loop-subdivided native midsurface readback plus normal offsets',
            'kind': 'fabric_visual_profile', 'profile_id': profile['id'], 'profile': dict(profile['pin']),
            'shader': profile['shader'], 'uv_atlas': self.topology.atlas,
            'subdivision_levels': profile['subdivision_levels'],
            'render_vertices': int(len(self.topology.source)), 'render_faces': int(len(self.topology.faces)),
            'texture_sha256': dict(profile['texture_sha256']), 'physics_schemas': False, 'collision': False}

    @staticmethod
    def _omnisurface(stage, name, textures, options):
        """OmniSurface MDL: textured colour, roughness, metalness and normal plus a cloth sheen lobe."""
        from pxr import Gf, Sdf, UsdShade
        mat = UsdShade.Material.Define(stage, '/World/Looks/fabric_skin_' + name)
        shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/Shader')
        shader.SetSourceAsset(Sdf.AssetPath('OmniSurface.mdl'), 'mdl')
        shader.SetSourceAssetSubIdentifier('OmniSurface', 'mdl')
        output = shader.CreateOutput('out', Sdf.ValueTypeNames.Token)
        for kind in ('Surface', 'Displacement', 'Volume'):
            getattr(mat, f'Create{kind}Output')('mdl').ConnectToSource(output)
        images = {'albedo': ('diffuse_reflection_color_image', 'sRGB'),
                  'roughness': ('specular_reflection_roughness_image', 'raw'),
                  'normal': ('geometry_normal_image', 'raw'),
                  'metallic': ('metalness_image', 'raw')}
        for role, filename in textures.items():
            if role not in images:
                continue
            target, space = images[role]
            image = shader.CreateInput(target, Sdf.ValueTypeNames.Asset)
            image.Set(Sdf.AssetPath(str(filename)))
            image.GetAttr().SetColorSpace(space)
            if role in ('roughness', 'metallic'):
                # OmniImage reads scalar maps from alpha by default and an
                # alpha-free single-channel image reads 1 there: use red (1).
                shader.CreateInput(target + '_alpha_mode', Sdf.ValueTypeNames.Int).Set(1)
        values = {'diffuse_reflection_weight': 1., 'diffuse_reflection_roughness': .5,
                  'specular_reflection_weight': .6, 'specular_reflection_ior': 1.35,
                  'specular_retro_reflection_weight': .45, 'specular_retro_reflection_roughness': .45}
        values.update({key: value for key, value in options.items() if key != 'specular_retro_reflection_color'})
        for key, value in values.items():
            shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(float(value))
        if 'specular_retro_reflection_color' in options:
            shader.CreateInput('specular_retro_reflection_color', Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*map(float, options['specular_retro_reflection_color'])))
        return mat

    @staticmethod
    def _preview(stage, name, textures, ior=None):
        """Portable UsdPreviewSurface with the same baked maps (no sheen lobe)."""
        from pxr import Gf, Sdf, UsdShade
        mat = UsdShade.Material.Define(stage, '/World/Looks/fabric_skin_' + name)
        shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        if 'metallic' not in textures:
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
        if ior is not None:
            shader.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(float(ior))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        reader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/ST')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
        roles = {'albedo': ('diffuseColor', Sdf.ValueTypeNames.Color3f, 'rgb', Sdf.ValueTypeNames.Float3, 'sRGB'),
                 'normal': ('normal', Sdf.ValueTypeNames.Normal3f, 'rgb', Sdf.ValueTypeNames.Float3, 'raw'),
                 'roughness': ('roughness', Sdf.ValueTypeNames.Float, 'r', Sdf.ValueTypeNames.Float, 'raw'),
                 'occlusion': ('occlusion', Sdf.ValueTypeNames.Float, 'r', Sdf.ValueTypeNames.Float, 'raw'),
                 'metallic': ('metallic', Sdf.ValueTypeNames.Float, 'r', Sdf.ValueTypeNames.Float, 'raw')}
        for role, filename in textures.items():
            target, target_type, output, output_type, space = roles[role]
            tex = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/' + role)
            tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(filename))
            tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(space)
            tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
            for direction in ('wrapS', 'wrapT'):
                tex.CreateInput(direction, Sdf.ValueTypeNames.Token).Set('clamp')
            if role == 'normal':
                tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                tex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
            # Explicit standard output types, as in ClosedClothVisual.
            tex.CreateOutput(output, output_type)
            shader.CreateInput(target, target_type).ConnectToSource(tex.ConnectableAPI(), output)
        return mat

    def sync(self, world_vertices):
        from pxr import Gf, Sdf, Vt
        world = np.asarray(world_vertices, dtype=np.float64)
        if world.shape != (self.topology.operator.shape[1], 3) or not np.isfinite(world).all():
            raise ValueError('Fabric skin needs the complete finite native cloth surface')
        points, normals = self.topology.render(world, self.thickness_m)
        points = points.astype(np.float32)
        # Same atomic points/normals re-authoring as ClosedClothVisual: a moving
        # sheet with unchanged normals must still dirty Hydra's normal buffer.
        with Sdf.ChangeBlock():
            self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
            normals_attr = self.mesh.CreateNormalsAttr()
            normals_attr.Clear()
            normals_attr.Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
            self.mesh.CreateExtentAttr().Set([Gf.Vec3f(*map(float, points.min(axis=0))),
                                              Gf.Vec3f(*map(float, points.max(axis=0)))])
