"""Original metric kitchen solids and USD authoring, with open physical cavities.

The generated meshes are the contact geometry, not illustrations of a different
hidden shape. Small decorative features are explicitly visual only. No force,
controller, solver or scene settings are changed by this module.
"""
from __future__ import annotations

import math
import numpy as np
import trimesh
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt


def closed_mesh(vertices, faces):
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    if not mesh.is_watertight or mesh.volume <= 0:
        raise ValueError('Kitchen contact geometry must be a closed positive solid')
    return mesh


def lathe(profile, segments=96, scale_xy=(1., 1.)):
    """Revolve an ordered (radius, z) material boundary, including pole caps."""
    vertices, rings, faces = [], [], []
    for radius, z in profile:
        if radius == 0:
            rings.append([len(vertices)])
            vertices.append((0., 0., z))
        else:
            rings.append(list(range(len(vertices), len(vertices)+segments)))
            vertices.extend((radius*math.cos(i*2*math.pi/segments)*scale_xy[0],
                             radius*math.sin(i*2*math.pi/segments)*scale_xy[1], z)
                            for i in range(segments))
    for a, b in zip(rings, rings[1:]):
        if len(a) == len(b) == 1:
            continue
        for i in range(segments):
            j = (i+1) % segments
            if len(a) == 1:
                faces.append((a[0], b[j], b[i]))
            elif len(b) == 1:
                faces.append((a[i], a[j], b[0]))
            else:
                faces.extend(((a[i], a[j], b[j]), (a[i], b[j], b[i])))
    return closed_mesh(vertices, faces)


def rounded_loop(half_x, half_y, radius, segments=12):
    points = []
    for sx, sy, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for angle in np.linspace(start, start+90, segments, endpoint=False):
            theta = math.radians(angle)
            points.append((sx*(half_x-radius)+radius*math.cos(theta),
                           sy*(half_y-radius)+radius*math.sin(theta)))
    return np.asarray(points)


def rounded_loft(profile):
    """Closed tray or board: (half_x, half_y, corner_radius, z), poles use x=0."""
    vertices, rings, faces = [], [], []
    for x, y, radius, z in profile:
        if x == 0:
            rings.append([len(vertices)])
            vertices.append((0., 0., z))
        else:
            points = rounded_loop(x, y, radius)
            rings.append(list(range(len(vertices), len(vertices)+len(points))))
            vertices.extend((px, py, z) for px, py in points)
    for a, b in zip(rings, rings[1:]):
        n = max(len(a), len(b))
        for i in range(n):
            j = (i+1) % n
            if len(a) == 1:
                faces.append((a[0], b[j], b[i]))
            elif len(b) == 1:
                faces.append((a[i], a[j], b[0]))
            else:
                faces.extend(((a[i], a[j], b[j]), (a[i], b[j], b[i])))
    return closed_mesh(vertices, faces)


def board(size, bevel=.0015):
    x, y, z = np.asarray(size)/2
    return rounded_loft([(0, 0, 0, 0), (x-bevel, y-bevel, .008, 0),
        (x, y, .0095, bevel), (x, y, .0095, 2*z-bevel),
        (x-bevel, y-bevel, .008, 2*z), (0, 0, 0, 2*z)])


def tray(size, wall=.004, floor=.004, radius=.014):
    x, y, h = size[0]/2, size[1]/2, size[2]
    return rounded_loft([(0, 0, 0, 0), (x-.001, y-.001, radius-.001, 0),
        (x, y, radius, .001), (x, y, radius, h-.001),
        (x-.001, y-.001, radius-.001, h),
        (x-wall, y-wall, radius-wall, h),
        (x-wall, y-wall, radius-wall, floor+.001),
        (x-wall-.001, y-wall-.001, radius-wall-.001, floor),
        (0, 0, 0, floor)])


def box(size, center):
    mesh = trimesh.creation.box(size)
    mesh.apply_translation(center)
    return mesh


def ellipsoid(radii, center, *, subdivisions=3):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.)
    mesh.apply_scale(radii)
    mesh.apply_translation(center)
    return mesh


def tube(points, radius, sections=12):
    points = np.asarray(points, dtype=float)
    vertices, faces = [], []
    for i, point in enumerate(points):
        direction = points[min(i+1, len(points)-1)]-points[max(0, i-1)]
        direction /= np.linalg.norm(direction)
        seed = np.array([0., 0., 1.]) if abs(direction[2]) < .95 else np.array([1., 0., 0.])
        a = np.cross(direction, seed)
        a /= np.linalg.norm(a)
        b = np.cross(direction, a)
        vertices.extend(point+radius*(a*math.cos(t)+b*math.sin(t))
                        for t in np.linspace(0, 2*math.pi, sections, endpoint=False))
    for ring in range(len(points)-1):
        for j in range(sections):
            a, b = ring*sections+j, ring*sections+(j+1)%sections
            c, d = a+sections, b+sections
            faces.extend(((a, b, d), (a, d, c)))
    vertices.extend((points[0], points[-1]))
    for j in range(sections):
        k = (j+1) % sections
        faces.extend(((len(vertices)-2, k, j),
                      (len(vertices)-1, (len(points)-1)*sections+j, (len(points)-1)*sections+k)))
    return closed_mesh(vertices, faces)


def translate(mesh, xyz):
    result = mesh.copy()
    result.apply_translation(xyz)
    return result


def extruded_outline(points, height, bottom=0.):
    """One watertight nonconvex solid, retaining fork slots and handle outline."""
    points = [tuple(p) for p in points]
    signed = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1]))
    if signed < 0:
        points.reverse()
    indices = list(range(len(points)))
    triangles = []
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    while len(indices) > 3:
        for j, current in enumerate(indices):
            prev, nxt = indices[j-1], indices[(j+1) % len(indices)]
            a, b, c = points[prev], points[current], points[nxt]
            if cross(a, b, c) <= 1e-14:
                continue
            if any(cross(a, b, points[k]) >= -1e-14 and
                   cross(b, c, points[k]) >= -1e-14 and
                   cross(c, a, points[k]) >= -1e-14
                   for k in indices if k not in (prev, current, nxt)):
                continue
            triangles.append((a, b, c))
            indices.pop(j)
            break
        else:
            raise ValueError('Invalid or degenerate utensil outline')
    triangles.append(tuple(points[i] for i in indices))
    vertices, faces = [], []
    for xy in triangles:
        offset = len(vertices)
        vertices.extend((x, y, bottom) for x, y in xy)
        vertices.extend((x, y, bottom+height) for x, y in xy)
        faces.extend(((offset+2, offset+1, offset), (offset+3, offset+4, offset+5)))
    for a, b in zip(points, list(points[1:])+[points[0]]):
        n = len(vertices)
        vertices.extend(((a[0], a[1], bottom), (b[0], b[1], bottom),
                         (b[0], b[1], bottom+height), (a[0], a[1], bottom+height)))
        faces.extend(((n, n+1, n+2), (n, n+2, n+3)))
    return closed_mesh(vertices, faces)


def schema(prim, name):
    values = prim.GetMetadata('apiSchemas')
    applied = list(values.GetAppliedItems()) if values else []
    if name not in applied:
        applied.append(name)
    prim.SetMetadata('apiSchemas', Sdf.TokenListOp.Create(prependedItems=applied))


def attribute(prim, name, kind, value):
    prim.CreateAttribute(name, kind, custom=False).Set(value)


def material(stage, path, color, roughness=.45, metallic=0., *, vertex_color=False, albedo_texture=None):
    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    if albedo_texture:
        reader = UsdShade.Shader.Define(stage, path+'/UV')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
        texture = UsdShade.Shader.Define(stage, path+'/Albedo')
        texture.CreateIdAttr('UsdUVTexture')
        texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(albedo_texture))
        texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        texture.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
        texture.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
        texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
        texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        shader.GetInput('diffuseColor').ConnectToSource(texture.ConnectableAPI(), 'rgb')
    elif vertex_color:
        reader = UsdShade.Shader.Define(stage, path+'/Color')
        reader.CreateIdAttr('UsdPrimvarReader_float3')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('displayColor')
        reader.CreateInput('fallback', Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*color))
        reader.CreateOutput('result', Sdf.ValueTypeNames.Float3)
        shader.GetInput('diffuseColor').ConnectToSource(reader.ConnectableAPI(), 'result')
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return result


def write_mesh(stage, path, geometry, mat, *, colors=None, texcoords=None, collision=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = np.asarray(geometry.vertices, dtype=np.float32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr([3]*len(geometry.faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(geometry.faces).reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0).tolist()), Gf.Vec3f(*points.max(0).tolist())])
    if collision:
        # Keep physical prim data identical; only visible shading is refined.
        normals, interpolation = geometry.vertex_normals, 'vertex'
    else:
        normals, interpolation = visual_corner_normals(geometry), 'faceVarying'
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, dtype=np.float32)))
    mesh.SetNormalsInterpolation(interpolation)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    if colors is not None:
        mesh.CreateDisplayColorPrimvar('vertex').Set(Vt.Vec3fArray.FromNumpy(np.asarray(colors, dtype=np.float32)))
    if texcoords is not None:
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(np.asarray(texcoords, dtype=np.float32)))
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(collision)
        schema(mesh.GetPrim(), 'PhysxCollisionAPI')
        attribute(mesh.GetPrim(), 'physxCollision:contactOffset', Sdf.ValueTypeNames.Float, .0002)
        attribute(mesh.GetPrim(), 'physxCollision:restOffset', Sdf.ValueTypeNames.Float, 0.)
        if collision == 'sdf':
            schema(mesh.GetPrim(), 'PhysxSDFMeshCollisionAPI')
            attribute(mesh.GetPrim(), 'physxSDFMeshCollision:sdfResolution', Sdf.ValueTypeNames.Int, 384)
            attribute(mesh.GetPrim(), 'physxSDFMeshCollision:sdfSubgridResolution', Sdf.ValueTypeNames.Int, 6)
            attribute(mesh.GetPrim(), 'physxSDFMeshCollision:sdfMargin', Sdf.ValueTypeNames.Float, .01)
            attribute(mesh.GetPrim(), 'physxSDFMeshCollision:sdfNarrowBandThickness', Sdf.ValueTypeNames.Float, .02)
    return mesh.GetPrim()


def visual_corner_normals(mesh):
    """Smooth curved parts while keeping broad planes and sharp edges planar.

    Shared vertex normals blend a thin board's top into its rim, producing an
    artificial diagonal X over a flat face. Face-corner normals separate those
    shading domains without changing any point, face or physical geometry.
    """
    face_normals = np.asarray(mesh.face_normals)
    incident = np.asarray(mesh.vertex_faces)[np.asarray(mesh.faces)]
    safe = np.maximum(incident, 0)
    neighbors = face_normals[safe]
    aligned = np.sum(neighbors*face_normals[:, None, None, :], axis=-1) >= math.cos(math.radians(40))
    weights = np.asarray(mesh.area_faces)[safe]*((incident >= 0) & aligned)
    normals = np.sum(neighbors*weights[..., None], axis=2)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.repeat(face_normals[:, None, :], 3, axis=1),
                        where=lengths > 1e-15)
    # Large coplanar groups use their actual plane normal at every corner.
    # The relative area threshold leaves finely faceted round vessels smooth.
    for facet, area in zip(mesh.facets, mesh.facets_area):
        if area >= .02*mesh.area:
            normals[facet] = face_normals[facet, None, :]
    return normals.reshape(-1, 3)


def wood_texcoords(mesh):
    """Metric box projection per face corner; top triangles share one plane.

    Coordinates are independent of tessellation density. Side projections
    prevent zero-area UVs on the vertical rim and divider faces without adding
    vertices or changing the contact mesh.
    """
    triangle = np.asarray(mesh.triangles)
    axis = np.argmax(np.abs(np.asarray(mesh.face_normals)), axis=1)
    uv = triangle[:, :, :2].copy()
    for normal_axis, projection in ((0, (1, 2)), (1, (0, 2))):
        selected = axis == normal_axis
        uv[selected] = triangle[selected][:, :, projection]
    uv /= np.array([.40, .28])
    return uv.reshape(-1, 2)


def natural_rest_transform(mesh):
    """Rigidly orient an existing solid onto its nearest stable hull facet.

    Candidate support polygons must contain the true integrated COM, with a
    positive edge margin. This changes local coordinates only: no vertex is
    flattened, no support is invented, and contact material is untouched.
    """
    from scipy.spatial import ConvexHull, QhullError
    hull = mesh.convex_hull
    candidates = []
    facets = list(hull.facets)
    grouped = set(int(i) for facet in facets for i in facet)
    facets.extend(np.array([i]) for i in range(len(hull.faces)) if i not in grouped)
    for facet in facets:
        normal = hull.face_normals[facet[0]]
        if normal[2] > -.7:
            continue
        down = np.array([0., 0., -1.])
        axis = np.cross(normal, down)
        angle = math.acos(float(np.clip(np.dot(normal, down), -1., 1.)))
        transform = (np.eye(4) if angle < 1e-10 else
                     trimesh.transformations.rotation_matrix(angle, axis))
        vertices = trimesh.transform_points(hull.vertices, transform)
        bottom = float(vertices[:, 2].min())
        support = vertices[vertices[:, 2] <= bottom+1e-8]
        if len(support) < 3:
            continue
        try:
            polygon = ConvexHull(support[:, :2])
        except QhullError:
            continue
        center = trimesh.transform_points(mesh.center_mass.reshape(1, 3), transform)[0]
        margin = float(-np.max(polygon.equations[:, :2]@center[:2]+polygon.equations[:, 2]))
        if margin < .00025:
            continue
        transform[2, 3] = -bottom
        receipt = {'method': 'Rigid coordinate-frame change onto an existing convex-hull facet; no shape deformation.',
            'rotation_angle_rad': angle, 'support_vertex_count': len(support),
            'support_polygon_area_m2': float(polygon.volume),
            'center_of_mass_support_edge_margin_m': margin,
            'source_to_rest_transform': transform.tolist()}
        candidates.append((round(angle, 10), -margin, transform, receipt))
    if not candidates:
        raise ValueError('No nearby natural resting facet contains the solid center of mass')
    _, _, transform, receipt = min(candidates, key=lambda item: item[:2])
    return transform, receipt


def mass_properties(meshes, mass):
    """Integrate declared collision solids; equal effective density is explicit."""
    solid = trimesh.util.concatenate(meshes)
    properties = solid.mass_properties
    inertia = np.asarray(properties.inertia)*mass/properties.mass
    eigenvalues, rotation = np.linalg.eigh(inertia)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    # USD Gf uses row vectors; principal axes rows are world eigenvectors.
    orientation = Gf.Matrix3d(*rotation.T.reshape(-1)).ExtractRotation().GetQuat()
    return {'center_of_mass_m': np.asarray(properties.center_mass).tolist(),
            'diagonal_inertia_kg_m2': eigenvalues.tolist(),
            'principal_axes_xyzw': [*list(orientation.GetImaginary()), orientation.GetReal()],
            'collision_volume_m3': float(properties.volume),
            'effective_density_kg_m3': float(mass/properties.volume),
            'inertia_method': 'Uniform effective density integrated over closed contact solids; detailed decoration mass included in the declared total mass.'}
