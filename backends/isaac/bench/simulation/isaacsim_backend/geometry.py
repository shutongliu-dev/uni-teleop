"""Engine-independent geometry needed to preserve the frozen task topology."""
import re

import numpy as np


def cloth_grid(spec):
    """Match the source x-fast vertex IDs and anti-diagonal triangles exactly."""
    if spec.get('mesh_type') == 'custom_surface' or any(
            key in spec for key in ('rest_vertices_m', 'triangles')):
        return custom_cloth_mesh(spec)
    nx, ny = spec['cells_xy']
    sx, sy = spec['size_m']
    if nx < 1 or ny < 1 or sx <= 0 or sy <= 0:
        raise ValueError('Cloth requires positive dimensions and grid cells')
    v = np.array([[i * sx / nx, j * sy / ny, 0.] for j in range(ny + 1) for i in range(nx + 1)], dtype=np.float32)
    f = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            f.extend([[a, a + 1, a + nx + 1], [a + 1, a + nx + 2, a + nx + 1]])
    return v, np.asarray(f, dtype=np.int32)


def custom_cloth_mesh(spec):
    """Validate an oriented connected material surface before native cooking.

    Explicit topology is opt-in; the legacy rectangular grid remains unchanged.
    Boundary openings are allowed, but disconnected patches, duplicate faces,
    inconsistent winding and unused nodes are not physical garment topology.
    """
    if spec.get('mesh_type') != 'custom_surface':
        raise ValueError('Explicit cloth topology requires mesh_type=custom_surface')
    vertices = np.asarray(spec.get('rest_vertices_m', []), dtype=np.float64)
    raw = np.asarray(spec.get('triangles', []))
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3
            or not np.isfinite(vertices).all() or raw.ndim != 2
            or raw.shape[1:] != (3,) or not len(raw)
            or not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError('Custom cloth requires finite Nx3 rest vertices and integer Mx3 triangles')
    if raw.min() < 0 or raw.max() >= len(vertices):
        raise ValueError('Custom cloth triangle index outside the rest mesh')
    triangles = raw.astype(np.int32)
    if (len(np.unique(triangles)) != len(vertices)
            or len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles)):
        raise ValueError('Custom cloth has unused nodes or duplicate triangles')
    normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                       vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    area = np.linalg.norm(normals, axis=1) / 2
    if np.any(area <= 1.e-12):
        raise ValueError('Custom cloth has degenerate rest triangles')
    declared = spec.get('rest_area_m2', float(area.sum()))
    if not np.isfinite(declared) or not np.isclose(declared, area.sum(), rtol=1.e-5, atol=1.e-10):
        raise ValueError('Custom cloth rest_area_m2 differs from its triangles')
    directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                      return_inverse=True, return_counts=True)
    orientation = np.zeros(len(edges), dtype=int)
    np.add.at(orientation, inverse, np.where(directed[:, 0] < directed[:, 1], 1, -1))
    if np.any(counts > 2) or np.any(orientation[counts == 2]):
        raise ValueError('Custom cloth must be an oriented manifold surface')
    boundary = edges[counts == 1]
    if len(boundary) and np.any(np.unique(boundary, return_counts=True)[1] != 2):
        raise ValueError('Custom cloth boundary must have manifold loops')
    neighbors = [[] for _ in vertices]
    for a, b in edges:
        neighbors[a].append(b); neighbors[b].append(a)
    seen, pending = {0}, [0]
    while pending:
        for index in neighbors[pending.pop()]:
            if index not in seen:
                seen.add(index); pending.append(index)
    if len(seen) != len(vertices):
        raise ValueError('Custom cloth must have one connected material surface')
    return vertices.astype(np.float32), triangles


def named_cloth_specs(scene):
    """Validate only the opt-in collection; frozen single-cloth specs are untouched.

    Each item keeps the legacy grid/pose conventions, but has its own identity,
    constitutive material and reset state. In particular an initial folded mesh
    does not replace its flat stress-free rest shape.
    """
    collection = scene.get('cloths', [])
    if not isinstance(collection, list):
        raise ValueError('scene.cloths must be a list of named surface cloth specs')
    specs = {}
    occupied = {item['id'] for key in ('objects', 'fixtures', 'soft_bodies')
                for item in scene.get(key, [])}
    for spec in collection:
        if not isinstance(spec, dict):
            raise ValueError('Each named cloth must be a specification mapping')
        name = spec.get('id')
        if not isinstance(name, str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) is None:
            raise ValueError('Named cloth ids must be USD-safe identifiers')
        if name in specs or name in occupied:
            raise ValueError(f'Duplicate or conflicting named cloth id: {name}')
        if type(spec.get('enabled', True)) is not bool:
            raise ValueError(f'Named cloth {name} enabled must be a bool')
        if not spec.get('enabled', True):
            raise ValueError(f'Disabled named cloth {name} must be omitted from scene.cloths')
        cells = spec.get('cells_xy', [])
        size = np.asarray(spec.get('size_m', []), dtype=float)
        grid = spec.get('mesh_type', 'grid') == 'grid'
        if (spec.get('mesh_type', 'grid') not in ('grid', 'custom_surface')
                or (grid and (len(cells) != 2 or any(type(n) is not int or n < 1 for n in cells)))
                or size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0)):
            raise ValueError(f'Named cloth {name} requires positive size and integer grid cells')
        mass, friction = spec.get('mass_kg'), spec.get('friction', .45)
        if (isinstance(mass, bool) or mass is None or not np.isfinite(mass) or mass <= 0
                or isinstance(friction, bool) or not np.isfinite(friction) or friction < 0):
            raise ValueError(f'Named cloth {name} requires finite positive mass and nonnegative friction')
        if type(spec.get('self_contact', True)) is not bool:
            raise ValueError(f'Named cloth {name} self_contact must be a bool')
        if not isinstance(spec.get('isaacsim', {}), dict):
            raise ValueError(f'Named cloth {name} isaacsim overrides must be a mapping')
        if any(not key.startswith('cloth_') for key in spec.get('isaacsim', {})):
            raise ValueError(f'Named cloth {name} may only override cloth_* options')
        options = {**scene.get('isaacsim', {}), **spec.get('isaacsim', {})}
        from .cloth_materials import surface_damping_properties
        from .cloth_sleep import surface_settling_properties
        surface_damping_properties(options)
        surface_settling_properties(options)
        thickness = options.get('cloth_thickness_m', .001)
        contact, rest = options.get('cloth_contact_offset_m', .003), options.get('cloth_rest_offset_m', .0015)
        if (not np.isfinite([thickness, contact, rest]).all() or thickness <= 0
                or rest < 0 or contact <= rest):
            raise ValueError(f'Named cloth {name} requires thickness > 0 and contact_offset > rest_offset >= 0')
        vertices, _ = cloth_grid(spec)
        cloth_initial_vertices(spec, vertices)
        specs[name] = spec
    return specs


def cloth_initial_vertices(spec, rest_vertices):
    """World-space initial nodes, independently validated against the rest mesh."""
    origin = np.asarray(spec.get('origin_m', []), dtype=float)
    q = np.asarray(spec.get('quaternion_wxyz', [1, 0, 0, 0]), dtype=float)
    if (origin.shape != (3,) or q.shape != (4,) or not np.isfinite(origin).all()
            or not np.isfinite(q).all() or np.linalg.norm(q) == 0):
        raise ValueError('Named cloth requires a finite origin and nonzero quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    points = np.asarray(spec.get('initial_vertices_m', np.asarray(rest_vertices) @ rotation.T + origin),
                        dtype=np.float32)
    if points.shape != np.asarray(rest_vertices).shape or not np.isfinite(points).all():
        raise ValueError('Named cloth initial_vertices_m must match its finite world-space Nx3 rest topology')
    return points.copy()


# Six tetrahedra per cube (Kuhn split along the 0-6 diagonal); conforming between
# neighbouring cells because every cell uses the same diagonal orientation.
_KUHN_TETS = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))


def box_tet_grid(size_m, cells_xyz):
    """Regular box tetrahedralization centred on the origin (volume soft bodies).

    Returns vertices (x-fast, then y, then z) and positively oriented tetrahedra.
    """
    nx, ny, nz = (int(n) for n in cells_xyz)
    sx, sy, sz = (float(s) for s in size_m)
    if min(nx, ny, nz) < 1 or min(sx, sy, sz) <= 0:
        raise ValueError('Soft box requires positive dimensions and cells')
    index = lambda i, j, k: (k * (ny + 1) + j) * (nx + 1) + i
    v = np.array([[i * sx / nx - sx / 2, j * sy / ny - sy / 2, k * sz / nz - sz / 2]
                  for k in range(nz + 1) for j in range(ny + 1) for i in range(nx + 1)], dtype=np.float32)
    tets = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                corner = [index(i, j, k), index(i + 1, j, k), index(i + 1, j + 1, k), index(i, j + 1, k),
                          index(i, j, k + 1), index(i + 1, j, k + 1), index(i + 1, j + 1, k + 1), index(i, j + 1, k + 1)]
                for tet in _KUHN_TETS:
                    a, b, c, d = (corner[n] for n in tet)
                    if np.dot(np.cross(v[b] - v[a], v[c] - v[a]), v[d] - v[a]) < 0:
                        b, c = c, b
                    tets.append([a, b, c, d])
    return v, np.asarray(tets, dtype=np.int32)


def box_tet_grid_symmetric(size_m, cells_xyz):
    """Mirror-symmetric box tetrahedralization: 24 tets per cell around the cell
    centre and its six face centres (shared with neighbours, so conforming).

    The Kuhn split shares one diagonal direction in every cell; under PhysX
    deformable contact noise (dynamic friction only) that anisotropy made a
    resting foam block creep (~0.9 mm/s) and spread. This pattern has no
    preferred diagonal. Vertices: the x-fast corner grid first, then centres.
    """
    nx, ny, nz = (int(n) for n in cells_xyz)
    sx, sy, sz = (float(s) for s in size_m)
    if min(nx, ny, nz) < 1 or min(sx, sy, sz) <= 0:
        raise ValueError('Soft box requires positive dimensions and cells')
    index = lambda i, j, k: (k * (ny + 1) + j) * (nx + 1) + i
    points = [[i * sx / nx - sx / 2, j * sy / ny - sy / 2, k * sz / nz - sz / 2]
              for k in range(nz + 1) for j in range(ny + 1) for i in range(nx + 1)]
    extra = {}

    def centre(key, members):
        if key not in extra:
            extra[key] = len(points)
            points.append(np.mean([points[m] for m in members], axis=0).tolist())
        return extra[key]
    tets = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = {(a, b, d): index(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)}
                middle = centre(('cell', i, j, k), list(c.values()))
                faces = {}
                for axis in range(3):
                    for side in (0, 1):
                        corners = [key for key in c if key[axis] == side]
                        # Order the four face corners around the face.
                        u, v = [n for n in range(3) if n != axis]
                        ring = sorted(corners, key=lambda key: np.arctan2(key[v] - .5, key[u] - .5))
                        cell = [i, j, k]
                        cell[axis] += side
                        faces[(axis, side)] = (('face', axis, *cell), [c[key] for key in ring])
                for (key, ring) in faces.values():
                    face = centre(key, ring)
                    for n in range(4):
                        tet = [ring[n], ring[(n + 1) % 4], face, middle]
                        A, B, C, D = (np.asarray(points[x]) for x in tet)
                        if np.dot(np.cross(B - A, C - A), D - A) < 0:
                            tet[0], tet[1] = tet[1], tet[0]
                        tets.append(tet)
    return np.asarray(points, dtype=np.float32), np.asarray(tets, dtype=np.int32)


TET_PATTERNS = {'kuhn_6': box_tet_grid, 'symmetric_24': box_tet_grid_symmetric}


def tet_surface_triangles(tets):
    """Boundary faces of a tetrahedral mesh (each listed once), outward oriented."""
    faces = {}
    for a, b, c, d in np.asarray(tets):
        for tri, opposite in (((a, c, b), d), ((a, b, d), c), ((a, d, c), b), ((b, c, d), a)):
            key = tuple(sorted(tri))
            faces[key] = None if key in faces else tri
    return np.asarray([tri for tri in faces.values() if tri is not None], dtype=np.int32)
