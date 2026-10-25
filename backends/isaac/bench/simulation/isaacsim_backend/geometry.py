"""Engine-independent geometry needed to preserve the frozen task topology."""
import numpy as np


def cloth_grid(spec):
    """Match the source x-fast vertex IDs and anti-diagonal triangles exactly."""
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
