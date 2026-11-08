"""Read-only render skin for a physically thick surface cloth.

The solver owns the midsurface, material thickness and collision offsets. This
closed skin follows native positions, with two faces separated by the declared
thickness and a stitched-looking perimeter. It has no physics schemas, mass,
collider or attachment. Updating it must never write the simulation mesh.
"""
from pathlib import Path

import numpy as np


def linear_albedo_mean(filename):
    """Mean source reflectance used to retain the declared cloth dye color."""
    from PIL import Image

    with Image.open(filename) as image:
        pixels = np.asarray(image.convert('RGB'), dtype=np.float64) / 255.
    linear = np.where(pixels <= .04045, pixels / 12.92,
                      ((pixels + .055) / 1.055) ** 2.4)
    mean = linear.mean(axis=(0, 1))
    if not np.isfinite(mean).all() or np.any(mean <= 1.e-6):
        raise ValueError('Albedo has a zero mean channel; cannot preserve declared dye colors')
    return mean


def boundary_edges(triangles):
    """Return consistently directed boundary edges of an oriented manifold."""
    triangles = np.asarray(triangles, dtype=np.int32)
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                            triangles[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Cloth skin requires a manifold triangle surface')
    return edges[counts[inverse] == 1]


def closed_skin(vertices, triangles, thickness_m):
    """Offset a native midsurface along its area-weighted vertex normals.

    Returns closed triangles and face-varying normals (sharp side seams). The
    supplied physical state is never modified. Thickness is a visual envelope,
    not a replacement for FEM thickness or native contact/rest offsets.
    """
    points = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int32)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0
            or tri.min() < 0 or tri.max() >= len(points)):
        raise ValueError('Invalid cloth skin topology or nonfinite native positions')
    if not np.isfinite(thickness_m) or thickness_m <= 0:
        raise ValueError('Visible cloth thickness must be positive and finite')
    normal = np.cross(points[tri[:, 1]] - points[tri[:, 0]],
                      points[tri[:, 2]] - points[tri[:, 0]])
    if np.any(np.linalg.norm(normal, axis=1) <= 1.e-14):
        raise ValueError('Cannot render a degenerate cloth triangle')
    normals = np.zeros_like(points)
    for index in range(3):
        np.add.at(normals, tri[:, index], normal)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1.e-14):
        raise ValueError('Cannot offset a cloth vertex with a degenerate normal')
    normals /= lengths[:, None]
    n = len(points)
    skin = np.concatenate([points + normals * thickness_m / 2,
                           points - normals * thickness_m / 2])
    edges = boundary_edges(tri)
    side = np.empty((len(edges) * 2, 3), dtype=np.int32)
    a, b = edges.T
    side[::2] = np.stack([b, a, a + n], axis=1)
    side[1::2] = np.stack([b, a + n, b + n], axis=1)
    bottom = tri[:, ::-1] + n
    faces = np.concatenate([tri, bottom, side])
    side_normals = np.cross(skin[side[:, 1]] - skin[side[:, 0]],
                           skin[side[:, 2]] - skin[side[:, 0]])
    side_normals /= np.maximum(np.linalg.norm(side_normals, axis=1)[:, None], 1.e-14)
    face_normals = np.concatenate([normals[tri], -normals[tri[:, ::-1]],
                                   np.repeat(side_normals[:, None, :], 3, axis=1)])
    return skin.astype(np.float32), faces, face_normals.reshape(-1, 3).astype(np.float32)


class ClosedClothVisual:
    """Independent world-space graphics mesh driven only by solver readback."""

    def __init__(self, stage, sim_path, name, triangles, rest_vertices,
                 thickness_m, color, texture_paths=None, initial_world_vertices=None,
                 texture_tile_size_m=None, visual_face_groups=None):
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        if not name.replace('_', '').isalnum():
            raise ValueError('Invalid cloth visual name')
        self.triangles = np.asarray(triangles, dtype=np.int32).copy()
        self.thickness_m = float(thickness_m)
        self.rest = np.asarray(rest_vertices, dtype=np.float32).copy()
        tile = np.asarray((.20, .20) if texture_tile_size_m is None
                          else texture_tile_size_m, dtype=np.float32)
        if tile.shape != (2,) or not np.isfinite(tile).all() or np.any(tile <= 0):
            raise ValueError('Cloth texture tile size must be a positive finite XY pair in metres')
        self.path = '/World/ClothVisuals/' + name
        UsdGeom.Xform.Define(stage, '/World/ClothVisuals')
        self.mesh = UsdGeom.Mesh.Define(stage, self.path)
        # Sibling graphics: setting the simulator's purpose to guide cannot
        # hide this skin by inherited visibility or purpose.
        UsdGeom.Imageable(stage.GetPrimAtPath(sim_path)).CreatePurposeAttr('guide')
        self.mesh.CreateSubdivisionSchemeAttr('none')
        self.mesh.CreateDoubleSidedAttr(False)
        points = self.rest if initial_world_vertices is None else initial_world_vertices
        vertices, faces, normals = closed_skin(points, self.triangles, self.thickness_m)
        self.faces = faces
        self.mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        self.mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        self.mesh.SetNormalsInterpolation('vertex')
        self.sync(points)
        # Texture coordinates remain attached to material rest coordinates.
        uv = self.rest[:, :2] / tile
        uv = np.concatenate([uv, uv])
        UsdGeom.PrimvarsAPI(self.mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
        material = self._material(stage, name, color, texture_paths or {})
        UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(material)
        # Narrow edge cells use a quieter weave/color, approximating the sewn
        # hem without altering geometry or adding fake physical reinforcement.
        spans = np.ptp(self.rest[:, :2], axis=0)
        local = self.rest[:, :2] - self.rest[:, :2].min(axis=0)
        near = np.minimum(local, spans - local).min(axis=1)
        edge_faces = np.any(near[self.triangles] < 1.e-6, axis=1)
        if visual_face_groups is not None:
            # Arbitrary garment silhouettes include collars, cuffs and crotch
            # boundaries; a rectangular bounding box cannot describe their hem.
            edge_faces = np.any(np.isin(self.triangles,
                np.unique(boundary_edges(self.triangles))), axis=1)
            used = set()
            for group_name, group in visual_face_groups.items():
                if not group_name.replace('_', '').isalnum():
                    raise ValueError('Invalid cloth visual face group name')
                indices = np.asarray(group.get('face_indices', []))
                dye = np.asarray(group.get('color', []), dtype=float)
                if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                        or len(indices) == 0 or indices.min() < 0
                        or indices.max() >= len(self.triangles)
                        or len(np.unique(indices)) != len(indices)
                        or dye.shape != (3,) or not np.isfinite(dye).all()
                        or np.any(dye < 0) or np.any(dye > 1)
                        or used.intersection(indices.tolist())):
                    raise ValueError('Invalid or overlapping cloth visual face groups')
                used.update(indices.tolist())
                edge_faces[indices] = False
                group_mat = self._material(stage, name + '_' + group_name, dye, {})
                group_faces = np.r_[indices, indices + len(self.triangles)]
                group_subset = UsdShade.MaterialBindingAPI(self.mesh).CreateMaterialBindSubset(
                    group_name, Vt.IntArray(group_faces.tolist()))
                UsdShade.MaterialBindingAPI.Apply(group_subset.GetPrim()).Bind(group_mat)
        ids = np.flatnonzero(np.concatenate([edge_faces, edge_faces,
                                              np.ones(len(faces) - 2 * len(edge_faces), dtype=bool)]))
        hem = self._material(stage, name + '_hem', np.asarray(color) * .90, {})
        subset = UsdShade.MaterialBindingAPI(self.mesh).CreateMaterialBindSubset(
            'Hem', Vt.IntArray(ids.tolist()))
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(hem)
        self.metadata = {'path': self.path, 'thickness_m': self.thickness_m,
            'source': 'native midsurface readback plus normal offsets',
            'physics_schemas': False, 'collision': False,
            'texture_paths': {k: str(v) for k, v in (texture_paths or {}).items()}}
        if texture_tile_size_m is not None:
            self.metadata['texture_tile_size_m'] = list(map(float, texture_tile_size_m))
        if visual_face_groups is not None:
            self.metadata['face_groups'] = {key: len(value['face_indices'])
                                          for key, value in visual_face_groups.items()}

    @staticmethod
    def _material(stage, name, color, textures):
        from pxr import Gf, Sdf, UsdShade
        mat = UsdShade.Material.Define(stage, '/World/Looks/cloth_skin_' + name)
        shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.96)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        if textures:
            uv = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/ST')
            uv.CreateIdAttr('UsdPrimvarReader_float2')
            uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            uv.CreateOutput('result', Sdf.ValueTypeNames.Float2)
            for role, filename in textures.items():
                if role not in ('normal', 'roughness', 'albedo'):
                    raise ValueError('Cloth texture role must be normal, roughness or albedo')
                if not Path(filename).is_file():
                    raise FileNotFoundError(filename)
                tex = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/' + role)
                tex.CreateIdAttr('UsdUVTexture')
                tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(filename))
                tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(
                    'sRGB' if role == 'albedo' else 'raw')
                tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
                for direction in ('wrapS', 'wrapT'):
                    tex.CreateInput(direction, Sdf.ValueTypeNames.Token).Set('repeat')
                if role == 'normal':
                    tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                    tex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                elif role == 'albedo':
                    # Decode before averaging: normalizing the sRGB mean would
                    # alter the cloth's declared linear blue/ochre dye colors.
                    scale = np.asarray(color, dtype=float) / linear_albedo_mean(filename)
                    tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(
                        Gf.Vec4f(*map(float, scale), 1.))
                # UsdUVTexture rgb is float3, even when consumed by normal3f.
                # Without an explicit output, ConnectToSource authors the
                # destination's normal3f type on the source shader instead.
                tex.CreateOutput('rgb' if role != 'roughness' else 'r',
                                 Sdf.ValueTypeNames.Float3 if role != 'roughness'
                                 else Sdf.ValueTypeNames.Float)
                target = 'diffuseColor' if role == 'albedo' else role
                value_type = {'normal': Sdf.ValueTypeNames.Normal3f,
                              'roughness': Sdf.ValueTypeNames.Float,
                              'albedo': Sdf.ValueTypeNames.Color3f}[role]
                shader.CreateInput(target, value_type).ConnectToSource(tex.ConnectableAPI(),
                    'r' if role == 'roughness' else 'rgb')
        return mat

    def sync(self, world_vertices):
        from pxr import Gf, Sdf, Vt
        points, faces, _ = closed_skin(world_vertices, self.triangles, self.thickness_m)
        if not np.array_equal(faces, self.faces):
            raise RuntimeError('Cloth render topology unexpectedly changed')
        # Dynamic native Hydra consumed face-corner buffers as vertex primvars
        # despite USD's faceVarying metadata. Author actual vertex normals:
        # area-weighted closed-shell faces give a smooth fabric edge and keep
        # the buffer valid under that renderer's vertex interpretation.
        normal = np.cross(points[faces[:, 1]]-points[faces[:, 0]],
                          points[faces[:, 2]]-points[faces[:, 0]])
        normals = np.zeros_like(points)
        for index in range(3):
            np.add.at(normals, faces[:, index], normal)
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 1.e-14):
            raise ValueError('Cannot render a cloth shell vertex with a degenerate normal')
        normals /= lengths[:, None]
        points_attr = self.mesh.GetPointsAttr()
        normals_attr = self.mesh.CreateNormalsAttr()
        extent_attr = self.mesh.CreateExtentAttr()
        self.mesh.SetNormalsInterpolation('vertex')
        point_values = Vt.Vec3fArray.FromNumpy(points)
        normal_values = Vt.Vec3fArray.FromNumpy(normals)
        extent = [Gf.Vec3f(*map(float, points.min(axis=0))),
                  Gf.Vec3f(*map(float, points.max(axis=0)))]
        # A flat sheet can move while its normals remain identical. USD elides
        # a same-value Set, leaving Hydra with only a points dirty notice and
        # causing its "without updating normal" fallback. Clear/re-author the
        # normals in the same change block as points so observers receive both
        # updates together and never see the temporary missing normal value.
        with Sdf.ChangeBlock():
            points_attr.Set(point_values)
            normals_attr.Clear()
            normals_attr.Set(normal_values)
            extent_attr.Set(extent)
