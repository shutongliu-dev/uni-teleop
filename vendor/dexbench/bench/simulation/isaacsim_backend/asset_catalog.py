"""Dispatch shared JSON/OBJ and native USD catalogs without physics imports."""
from __future__ import annotations
import hashlib
from copy import deepcopy
import math
from pathlib import Path
from bench.paths import load_json
from bench.simulation.assets import task_parts

SOURCE_PATH = Path(task_parts.__file__).resolve()


def collision_mesh_coordinate_scale(spec):
    """Opt-in local cooking coordinates; default keeps the legacy USD stage."""
    factor = spec.get('collision_mesh_coordinate_scale', 1.)
    if type(factor) not in (int, float) or not math.isfinite(factor) or not 1. <= factor <= 1000.:
        raise ValueError('Collision mesh coordinate scale must be finite and in [1, 1000]')
    return float(factor)


def condition_collision_mesh(vertices, factor):
    """Return conditioned points and inverse prim scale, preserving world size.

    This changes cooking precision, not declared geometry or mass properties.
    Native cooking fidelity must still be validated for each opted-in task.
    """
    import numpy as np
    factor = collision_mesh_coordinate_scale({'collision_mesh_coordinate_scale': factor})
    if factor == 1.:
        return vertices, None
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Finite Nx3 collision mesh vertices required')
    conditioned = (points*factor).astype(np.float32)
    if not np.isfinite(conditioned).all():
        raise ValueError('Conditioned collision vertices overflowed float32')
    return conditioned, [1./factor]*3


def collision_offsets(spec):
    """Explicit task mesh offsets; omitted values preserve the legacy stage."""
    contact = spec.get('contact_offset_m', .001)
    rest = spec.get('rest_offset_m', 0.)
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (contact, rest))
            or contact <= 0 or rest >= contact):
        raise ValueError('Task collision offsets require finite contact > 0 and rest < contact')
    return {'contact_offset': float(contact), 'rest_offset': float(rest)}


def validate_triangle_mesh_role(asset, *, dynamic):
    """Raw triangle meshes are an explicit fixture representation, never a dynamic fallback."""
    if any(part.get('geometry') == 'triangle_mesh' for part in asset['collision_parts']):
        if dynamic or asset.get('dynamic') is not False:
            raise ValueError('Raw triangle_mesh collision is supported only for explicit non-dynamic fixtures; dynamic meshes require authored SDF')


def prepare_triangle_mesh(vertices, faces, scale=(1., 1., 1.)):
    """Validate explicit raw triangles before authoring and return owned arrays.

    Check the actual scaled float32 points sent to USD, including collapses or
    overflow introduced by representation. Open surfaces are valid fixtures;
    this does not require a closed volume or change legacy convex cooking.
    """
    import numpy as np
    points, triangles, scaling = np.asarray(vertices), np.asarray(faces), np.asarray(scale)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
            or points.dtype.kind not in 'iuf' or not np.isfinite(points).all()):
        raise ValueError('triangle_mesh requires finite Nx3 vertices with at least three points')
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0
            or triangles.dtype.kind not in 'iu'
            or np.any(triangles < 0) or np.any(triangles >= len(points))
            or np.any(triangles > np.iinfo(np.int32).max)):
        raise ValueError('triangle_mesh requires nonempty integer Mx3 indices within the vertex array')
    if (scaling.shape != (3,) or scaling.dtype.kind not in 'iuf'
            or any(isinstance(value, (bool, np.bool_)) for value in np.asarray(scale, dtype=object).flat)
            or not np.isfinite(scaling).all() or np.any(scaling <= 0)):
        raise ValueError('triangle_mesh scale must have three finite positive numbers')
    # Preserve the existing loader's float32 point/scale multiplication.
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        authored = points.astype(np.float32) * scaling.astype(np.float32)
    if not np.isfinite(authored).all():
        raise ValueError('triangle_mesh scaled float32 vertices must remain finite')
    indices = triangles.astype(np.int32)
    corners = authored[indices].astype(np.float64)
    cross = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    if np.any(np.all(cross == 0., axis=1)):
        raise ValueError('triangle_mesh triangles must be nondegenerate after float32 scaling')
    return authored, indices


def inertia_principal_axes(asset):
    """Return the optional body-local mass-frame rotation in USD wxyz order."""
    q = asset.get('principal_axes_xyzw', [0., 0., 0., 1.])
    if (not isinstance(q, (list, tuple)) or len(q) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in q)
            or abs(sum(v*v for v in q)-1.) > 1e-6):
        raise ValueError('Task inertia principal axes must be a finite unit xyzw quaternion')
    return [float(q[3]), *map(float, q[:3])]


def load_task_asset(asset_id, asset_root=None, *, verify_hashes=False):
    """Dispatch versioned USD catalogs without importing OpenUSD or a GPU."""
    if asset_root is not None:
        root = Path(asset_root).resolve()
        path = root / 'manifest.json'
        catalog = load_json(path)
        if catalog.get('schema') == 'collaborator_native_usd_catalog.v1':
            result = deepcopy(catalog['assets'][asset_id])
            result.update(asset_id=asset_id, asset_root=str(root), manifest_path=str(path),
                          manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            dependencies = dict(result['source_dependencies_sha256'])
            for part in result['collision_parts']:
                if 'mesh_path' in part:
                    part['mesh_path'] = str(root / part['mesh_path'])
                    dependencies[part['mesh_path']] = part['sha256']
            if verify_hashes:
                for name, expected in dependencies.items():
                    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                        raise ValueError(f'Task USD dependency hash mismatch: {name}')
            return result
    return task_parts.load_task_asset(asset_id, asset_root, verify_hashes=verify_hashes)


def load_task_catalog(*args, **kwargs):
    return task_parts.load_task_catalog(*args, **kwargs)


def load_obj_mesh(*args, **kwargs):
    return task_parts.load_obj_mesh(*args, **kwargs)


def provenance():
    return {'source_path': str(SOURCE_PATH), 'source_sha256': hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            'loading': 'Pure bench.simulation.assets JSON/OBJ reader; no physics runtime imports.'}
