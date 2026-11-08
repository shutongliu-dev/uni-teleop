"""Coplanar contact tessellation; the visible tray and physical solid stay fixed."""
from collections import Counter

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import trimesh


def _boundary(faces):
    counts = Counter(tuple(sorted((int(a), int(b)))) for face in faces
                     for a, b in zip(face, np.roll(face, -1)))
    return {edge for edge, count in counts.items() if count == 1}


def subdivide_floor(mesh, floor_height, spacing=.020):
    """Retriangulate a convex flat floor with a pose-independent metric grid.

    Preserve every original vertex, boundary edge and nonfloor triangle.
    Only interior points are added; no support surface or solid is deformed.
    """
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    triangles = vertices[faces]
    selected = ((np.ptp(triangles[:, :, 2], axis=1) < 1e-10)
                & (np.abs(triangles[:, 0, 2]-floor_height) < 1e-8)
                & (mesh.face_normals[:, 2] > 1.-1e-10))
    old_faces = faces[selected]
    if not len(old_faces):
        raise ValueError('No existing upward flat tray floor')
    plane_ids = np.unique(old_faces)
    plane = vertices[plane_ids]
    polygon = ConvexHull(plane[:, :2])
    lower, upper = plane[:, :2].min(0), plane[:, :2].max(0)
    axes = [np.arange(np.ceil(lower[i]/spacing), np.floor(upper[i]/spacing)+1)*spacing
            for i in range(2)]
    added = []
    for x in axes[0]:
        for y in axes[1]:
            xy = np.array([x, y])
            if (np.max(polygon.equations[:, :2]@xy+polygon.equations[:, 2]) < -1e-9
                    and np.min(np.linalg.norm(plane[:, :2]-xy, axis=1)) > 1e-10):
                added.append([x, y, plane[0, 2]])
    if not added:
        raise ValueError('Metric floor grid added no interior samples')
    # Author the actual float32 coordinates that USD will retain.
    added = np.asarray(added, dtype=np.float32).astype(float)
    points = np.vstack((plane, added))
    ids = np.r_[plane_ids, np.arange(len(vertices), len(vertices)+len(added))]
    new_faces = ids[Delaunay(points[:, :2]).simplices]
    new_vertices = np.vstack((vertices, added))
    normal_z = np.cross(new_vertices[new_faces[:, 1]]-new_vertices[new_faces[:, 0]],
                        new_vertices[new_faces[:, 2]]-new_vertices[new_faces[:, 0]])[:, 2]
    new_faces[normal_z < 0] = new_faces[normal_z < 0, ::-1]
    if _boundary(old_faces) != _boundary(new_faces):
        raise ValueError('Floor tessellation changed an original boundary edge')
    result = trimesh.Trimesh(new_vertices, np.vstack((faces[~selected], new_faces)), process=False)
    if not result.is_watertight or not result.is_winding_consistent:
        raise ValueError('Floor tessellation broke the closed tray solid')
    if not np.array_equal(result.bounds, mesh.bounds):
        raise ValueError('Floor tessellation changed collision bounds')
    if not np.isclose(result.volume, mesh.volume, rtol=1e-12, atol=1e-15):
        raise ValueError('Floor tessellation changed the physical volume')
    boundary_ids = set(i for edge in _boundary(old_faces) for i in edge)
    interior_faces = [face for face in new_faces if not any(int(i) in boundary_ids for i in face)]
    edge = new_vertices[np.asarray(interior_faces)]
    maximum_interior_edge = float(np.linalg.norm(edge-np.roll(edge, -1, axis=1), axis=2).max())
    return result, {
        'method': f'{spacing*1000:g} mm Cartesian interior grid, Delaunay triangulation of the same flat floor.',
        'grid_spacing_m': spacing, 'old_floor_triangle_count': len(old_faces),
        'new_floor_triangle_count': len(new_faces), 'added_interior_vertex_count': len(added),
        'original_boundary_edge_count': len(_boundary(old_faces)),
        'maximum_interior_triangle_edge_m': maximum_interior_edge,
        'floor_local_z_m': float(plane[0, 2]),
        'volume_before_m3': float(mesh.volume), 'volume_after_m3': float(result.volume),
        'original_vertices_unchanged': True, 'nonfloor_faces_unchanged': True,
        'boundary_edges_unchanged': True, 'collision_bounds_unchanged': True,
        'scope': 'Contact tessellation only; native stability must be validated separately.'}


def refine_sorting_tray_floor(author, *, spacing=.020):
    """Update only the invisible Collision/shell, preserving visual USD data."""
    from pxr import UsdGeom, Vt
    row = author.meta['fixture_metadata']['cutlery_tray']
    path = row['body_path']+'/Collision/shell'
    usd = UsdGeom.Mesh(author.stage.GetPrimAtPath(path))
    original = trimesh.Trimesh(np.asarray(usd.GetPointsAttr().Get()),
        np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False)
    mesh, receipt = subdivide_floor(original, row['floor_top_local_z_m'], spacing=spacing)
    usd.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    usd.GetFaceVertexCountsAttr().Set([3]*len(mesh.faces))
    usd.GetFaceVertexIndicesAttr().Set(mesh.faces.reshape(-1).tolist())
    usd.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, dtype=np.float32)))
    row['contact_tessellation'] = {'collider_path': path, **receipt}


def refine_source_pad_top(author, *, spacing=.002):
    """Refine the existing static triangle-mesh pad top, retaining its bevel."""
    from pxr import UsdGeom, Vt
    fixture, mesh_name = ('source_pad', 'pad') if author.meta['task_id'] == 'utensil_sorting' else ('source_mat', 'mat')
    row = author.meta['fixture_metadata'][fixture]
    path = row['body_path']+'/Collision/'+mesh_name
    usd = UsdGeom.Mesh(author.stage.GetPrimAtPath(path))
    if usd.GetPrim().GetAttribute('physics:approximation').Get() != 'none':
        raise ValueError('Source support must retain its actual triangle-mesh collision')
    original = trimesh.Trimesh(np.asarray(usd.GetPointsAttr().Get()),
        np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False)
    mesh, receipt = subdivide_floor(original, original.bounds[1, 2], spacing=spacing)
    usd.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    usd.GetFaceVertexCountsAttr().Set([3]*len(mesh.faces))
    usd.GetFaceVertexIndicesAttr().Set(mesh.faces.reshape(-1).tolist())
    usd.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, dtype=np.float32)))
    row['contact_tessellation'] = {'collider_path': path, **receipt}
