"""Read-only CUDA BVH audit of cloth midsurface intersections.

Every supplied frame is checked, including structural neighbor triangles.
Shared edges/vertices alone are allowed; crossings away from the structural
boundary are not. This detects sampled midsurface crossings, not contact force,
finite-thickness compression, or continuous motion between native steps.
"""
import numpy as np
import warp as wp


@wp.func
def _min3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def _max3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.func
def _away(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, shared: int):
    if shared == 0:
        return True
    e = b - a
    sq = wp.dot(e, e)
    t = wp.float64(0.)
    if sq > wp.float64(1.e-20):
        t = wp.clamp(wp.dot(p - a, e) / sq, wp.float64(0.), wp.float64(1.))
    return wp.length(p - (a + t * e)) > wp.float64(1.e-7)


@wp.func
def _segment_hit(p: wp.vec3d, q: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d,
                 shared_start: wp.vec3d, shared_end: wp.vec3d, shared: int):
    d, e1, e2 = q - p, b - a, c - a
    h = wp.cross(d, e2)
    det = wp.dot(e1, h)
    scale = wp.length(e1) * wp.length(e2) * wp.length(d)
    if wp.abs(det) <= wp.max(scale * wp.float64(1.e-12), wp.float64(1.e-30)):
        return False
    inv = wp.float64(1.) / det
    s = p - a
    u = inv * wp.dot(s, h)
    z = wp.cross(s, e1)
    v = inv * wp.dot(d, z)
    t = inv * wp.dot(e2, z)
    eps = wp.float64(1.e-10)
    if u >= -eps and v >= -eps and u + v <= wp.float64(1.) + eps and t >= -eps and t <= wp.float64(1.) + eps:
        return _away(p + t * d, shared_start, shared_end, shared)
    return False


@wp.func
def _sat(a: wp.mat33d, b: wp.mat33d, axis: wp.vec3d, shared: int):
    length = wp.length(axis)
    if length <= wp.float64(1.e-30):
        return True
    axis = axis / length
    aa = wp.vec3d(wp.dot(a[0], axis), wp.dot(a[1], axis), wp.dot(a[2], axis))
    bb = wp.vec3d(wp.dot(b[0], axis), wp.dot(b[1], axis), wp.dot(b[2], axis))
    overlap = wp.min(wp.max(aa[0], wp.max(aa[1], aa[2])), wp.max(bb[0], wp.max(bb[1], bb[2]))) - wp.max(wp.min(aa[0], wp.min(aa[1], aa[2])), wp.min(bb[0], wp.min(bb[1], bb[2])))
    if shared > 0:
        return overlap > wp.float64(1.e-9)
    return overlap >= wp.float64(-1.e-12)


@wp.func
def _on_edge_away(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d,
                  start: wp.vec3d, end: wp.vec3d, shared: int):
    edge = b - a
    length = wp.dot(edge, edge)
    if length <= wp.float64(1.e-20):
        return False
    t = wp.dot(p - a, edge) / length
    if t >= wp.float64(0.) and t <= wp.float64(1.) and wp.length(p - a - t * edge) <= wp.float64(1.e-9):
        return _away(p, start, end, shared)
    return False


@wp.func
def _intersection(a: wp.mat33d, b: wp.mat33d, ia: wp.vec3i, ib: wp.vec3i):
    shared = int(0)
    first, last = wp.vec3d(0.), wp.vec3d(0.)
    for i in range(3):
        equal = ia[i] == ib[0] or ia[i] == ib[1] or ia[i] == ib[2]
        if equal:
            if shared == 0:
                first = a[i]
            last = a[i]
            shared += 1
    for i in range(3):
        j = (i + 1) % 3
        # A nonparallel segment with a topologically shared endpoint can
        # intersect the other face's plane only at that endpoint. Avoid
        # reconstructing it through a nearly singular determinant: roundoff
        # can move the result outside _away's unchanged boundary tolerance.
        # Do not skip the face pair: opposite edges and the coplanar branch
        # below must still detect overlap away from the shared boundary.
        a_start_shared = ia[i] == ib[0] or ia[i] == ib[1] or ia[i] == ib[2]
        a_end_shared = ia[j] == ib[0] or ia[j] == ib[1] or ia[j] == ib[2]
        b_start_shared = ib[i] == ia[0] or ib[i] == ia[1] or ib[i] == ia[2]
        b_end_shared = ib[j] == ia[0] or ib[j] == ia[1] or ib[j] == ia[2]
        if not a_start_shared and not a_end_shared:
            if _segment_hit(a[i], a[j], b[0], b[1], b[2], first, last, shared):
                return True
        if not b_start_shared and not b_end_shared:
            if _segment_hit(b[i], b[j], a[0], a[1], a[2], first, last, shared):
                return True
    na = wp.cross(a[1] - a[0], a[2] - a[0])
    nb = wp.cross(b[1] - b[0], b[2] - b[0])
    la, lb = wp.length(na), wp.length(nb)
    if la <= wp.float64(1.e-14) or lb <= wp.float64(1.e-14):
        return False
    cosine = wp.abs(wp.dot(na, nb) / (la * lb))
    distance = wp.abs(wp.dot(b[0] - a[0], na)) / la
    if cosine > wp.float64(1.) - wp.float64(1.e-10) and distance < wp.float64(1.e-9):
        # A shared vertex does not license a second, remote edge touch. SAT
        # with strictly positive interior overlap alone misses that boundary.
        if shared > 0:
            for i in range(3):
                for j in range(3):
                    if _on_edge_away(a[i], b[j], b[(j + 1) % 3], first, last, shared):
                        return True
                    if _on_edge_away(b[i], a[j], a[(j + 1) % 3], first, last, shared):
                        return True
        for i in range(3):
            if not _sat(a, b, wp.cross(na, a[(i + 1) % 3] - a[i]), shared):
                return False
            if not _sat(a, b, wp.cross(na, b[(i + 1) % 3] - b[i]), shared):
                return False
        return True
    return False


@wp.func
def _point_face_distance(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    e, f, w = b - a, c - a, p - a
    ee, ef, ff = wp.dot(e, e), wp.dot(e, f), wp.dot(f, f)
    den = ee * ff - ef * ef
    if den > 1.e-20:
        v = (ff * wp.dot(w, e) - ef * wp.dot(w, f)) / den
        u = (ee * wp.dot(w, f) - ef * wp.dot(w, e)) / den
        if u >= 0. and v >= 0. and u + v <= 1.:
            return wp.length(w - v * e - u * f)
    return float(1.e10)


@wp.func
def _triangle_distance(a: wp.mat33d, b: wp.mat33d):
    distance = float(1.e10)
    for i in range(3):
        distance = wp.min(distance, _point_face_distance(wp.vec3(a[i]), wp.vec3(b[0]), wp.vec3(b[1]), wp.vec3(b[2])))
        distance = wp.min(distance, _point_face_distance(wp.vec3(b[i]), wp.vec3(a[0]), wp.vec3(a[1]), wp.vec3(a[2])))
        for j in range(3):
            result = wp.closest_point_edge_edge(wp.vec3(a[i]), wp.vec3(a[(i + 1) % 3]),
                wp.vec3(b[j]), wp.vec3(b[(j + 1) % 3]), 1.e-12)
            distance = wp.min(distance, result[2])
    return distance


@wp.kernel
def _audit(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), triangles: wp.array(dtype=wp.vec3i),
           owners: wp.array(dtype=int), thickness: wp.array(dtype=float), max_thickness: float,
           counts: wp.array(dtype=int), first_pairs: wp.array2d(dtype=int), penetration: wp.array(dtype=float), pair_capacity: int):
    face = wp.tid()
    ia = triangles[face]
    v0, v1, v2 = points[ia[0]], points[ia[1]], points[ia[2]]
    a = wp.matrix_from_rows(wp.vec3d(v0), wp.vec3d(v1), wp.vec3d(v2))
    if wp.length(wp.cross(a[1] - a[0], a[2] - a[0])) <= wp.float64(1.e-14):
        wp.atomic_add(counts, 2, 1)
    lower = _min3(_min3(v0, v1), v2) - wp.vec3(max_thickness + 1.e-8)
    upper = _max3(_max3(v0, v1), v2) + wp.vec3(max_thickness + 1.e-8)
    query = wp.mesh_query_aabb(mesh, lower, upper)
    for other in query:
        if other > face:
            ib = triangles[other]
            b = wp.matrix_from_rows(wp.vec3d(points[ib[0]]), wp.vec3d(points[ib[1]]), wp.vec3d(points[ib[2]]))
            hit = _intersection(a, b, ia, ib)
            if hit:
                kind = int(0)
                if owners[face] != owners[other]:
                    kind = 1
                index = wp.atomic_add(counts, kind, 1)
                if index < pair_capacity:
                    first_pairs[kind * pair_capacity + index, 0] = face
                    first_pairs[kind * pair_capacity + index, 1] = other
            # A common material vertex/edge has inherently zero separation.
            # All other face pairs receive the full finite-thickness check.
            structural = False
            for i in range(3):
                if ia[i] == ib[0] or ia[i] == ib[1] or ia[i] == ib[2]:
                    structural = True
            if not structural:
                separation = float(0.)
                if not hit:
                    separation = _triangle_distance(a, b)
                envelope = (thickness[owners[face]] + thickness[owners[other]]) * .5
                wp.atomic_max(penetration, 0, wp.max(0., envelope - separation))


class ClothIntersectionAudit:
    """One reusable CUDA BVH for independently named, immutable topologies."""

    def __init__(self, topology_by_id, vertex_counts, *, thickness_by_id=None, device='cuda:0',
                 intersection_pair_capacity=8):
        wp.init()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError('Full-stream cloth audits require a CUDA device')
        self.names = tuple(topology_by_id)
        if type(intersection_pair_capacity) is not int or not 8 <= intersection_pair_capacity <= 65536:
            raise ValueError('Intersection pair capacity must be an integer from 8 to 65536')
        self.pair_capacity = intersection_pair_capacity
        if not self.names or set(self.names) != set(vertex_counts):
            raise ValueError('Exact named cloth topology and vertex counts required')
        self.counts_by_id = dict(vertex_counts)
        thickness_by_id = thickness_by_id or {name: .003 for name in self.names}
        if set(thickness_by_id) != set(self.names):
            raise ValueError('Exact named cloth physical thicknesses required')
        thickness = np.asarray([thickness_by_id[name] for name in self.names], dtype=np.float32)
        if not np.isfinite(thickness).all() or (thickness <= 0).any():
            raise ValueError('Physical thickness must be positive and finite')
        self.max_thickness = float(thickness.max())
        self.thickness = wp.array(thickness, dtype=float, device=self.device)
        triangles, owners, offset = [], [], 0
        for owner, name in enumerate(self.names):
            tri = np.asarray(topology_by_id[name], dtype=np.int32)
            if tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0 or tri.min() < 0 or tri.max() >= vertex_counts[name]:
                raise ValueError('Invalid cloth topology')
            triangles.append(tri + offset)
            owners.extend([owner] * len(tri))
            offset += vertex_counts[name]
        tri = np.concatenate(triangles)
        self.triangle_count = len(tri)
        self.points = wp.zeros(offset, dtype=wp.vec3, device=self.device)
        self.indices = wp.array(tri.ravel(), dtype=int, device=self.device)
        self.triangles = wp.array(tri, dtype=wp.vec3i, device=self.device)
        self.owners = wp.array(owners, dtype=int, device=self.device)
        self.mesh = wp.Mesh(points=self.points, indices=self.indices)
        self.counts = wp.zeros(3, dtype=int, device=self.device)
        self.first_pairs = wp.zeros((2*self.pair_capacity, 2), dtype=int, device=self.device)
        self.penetration = wp.zeros(1, dtype=float, device=self.device)

    def check(self, vertices_by_id):
        if set(vertices_by_id) != set(self.names):
            raise ValueError('Cloth identity differs from audit topology')
        rows = []
        for name in self.names:
            value = np.asarray(vertices_by_id[name], dtype=np.float32)
            if value.shape != (self.counts_by_id[name], 3) or not np.isfinite(value).all():
                raise ValueError('Incomplete or nonfinite cloth geometry')
            rows.append(value)
        self.points.assign(np.concatenate(rows))
        self.mesh.refit()
        self.counts.zero_()
        self.penetration.zero_()
        wp.launch(_audit, dim=self.triangle_count,
            inputs=[self.mesh.id, self.points, self.triangles, self.owners, self.thickness,
                    self.max_thickness, self.counts, self.first_pairs, self.penetration, self.pair_capacity],
            device=self.device)
        counts = self.counts.numpy()
        pairs = self.first_pairs.numpy() if counts[:2].any() else None
        result = {'self_intersections': int(counts[0]), 'intercloth_intersections': int(counts[1]),
            'degenerate_triangles': int(counts[2]),
            'first_self_pairs': pairs[:min(8, counts[0])].tolist() if counts[0] else [],
            'first_intercloth_pairs': pairs[self.pair_capacity:self.pair_capacity+min(8, counts[1])].tolist() if counts[1] else [],
            'method': 'CUDA BVH with double-precision segment/triangle and coplanar SAT; topology-shared segment endpoints excluded from nonparallel tests; structural boundary tolerance 1e-7m',
            'continuous_between_steps': False, 'contact_force_measured': False,
            'finite_thickness_compression_checked': True,
            'max_cloth_envelope_penetration_m': float(self.penetration.numpy()[0]),
            'thickness_method': 'minimum vertex/face and edge/edge distance; excludes structurally shared-vertex faces',
            'crease_thickness_certified': False}
        if self.pair_capacity != 8:
            result.update(intersection_pair_inventory_complete=bool(np.all(counts[:2] <= self.pair_capacity)),
                intersection_pair_capacity=self.pair_capacity,
                self_intersection_pairs=(pairs[:min(self.pair_capacity, counts[0])].tolist() if counts[0] else []),
                intercloth_intersection_pairs=(pairs[self.pair_capacity:self.pair_capacity+min(self.pair_capacity, counts[1])].tolist() if counts[1] else []))
        return result
