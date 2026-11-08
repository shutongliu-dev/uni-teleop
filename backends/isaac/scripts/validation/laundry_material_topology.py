"""Original local material connectivity shared by capture and closure stopping."""
import numpy as np


def face_edge_neighbors(triangles):
    by_edge = {}
    for face, ids in enumerate(triangles):
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            by_edge.setdefault(tuple(sorted((int(a), int(b)))), []).append(face)
    neighbors = [set() for _ in triangles]
    for faces in by_edge.values():
        for face in faces:
            neighbors[face].update(set(faces) - {face})
    return neighbors


def local_material_patch(triangles, a, b, *, face_neighbors=None):
    """Return a shared edge, or an explicitly enabled unique one-face bridge.

    World/material diameters and actual contact remain the caller's checks.
    An absent neighbor inventory preserves the original shared-edge policy.
    """
    shared = set(triangles[a]) & set(triangles[b])
    if len(shared) == 2:
        return {'relation': 'shared_edge', 'bridge_triangle_id': None,
                'support_triangle_ids': [a, b]}
    if face_neighbors is None or len(shared) != 1:
        return None
    bridges = (face_neighbors[a] & face_neighbors[b]) - {a, b}
    if len(bridges) != 1:
        return None
    bridge = int(next(iter(bridges)))
    if len(np.unique(triangles[[a, bridge, b]])) != 5:
        return None
    return {'relation': 'single_bridge_triangle', 'bridge_triangle_id': bridge,
            'support_triangle_ids': [a, bridge, b]}
