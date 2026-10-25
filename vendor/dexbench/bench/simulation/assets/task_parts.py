"""Load original, metre-scale task parts from the versioned shared asset library.

This module intentionally does not import Newton or initialize a GPU. Visual
meshes and convex collision meshes are separate; a gear must never be replaced
by the convex hull of its complete visual mesh, which would seal its bore.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path

from bench.paths import DATA_ROOT, load_json


DEFAULT_ASSET_ROOT = Path(f"{DATA_ROOT}/assets/task_parts/sequence_workcell/v1")


def asset_root(root=None):
    return Path(root or os.environ.get("SEQUENCE_TASK_ASSETS", DEFAULT_ASSET_ROOT)).resolve()


def load_task_catalog(root=None):
    """Return the manifest with its location and digest, without changing paths."""
    root = asset_root(root)
    path = root / "manifest.json"
    result = load_json(path)
    result["manifest_path"] = str(path)
    result["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["asset_root"] = str(root)
    return result


def load_task_asset(asset_id, asset_root=None, *, verify_hashes=False):
    """Return one independent asset dictionary with absolute mesh paths.

    All positions and dimensions are metres and quaternions use xyzw order.
    ``visual_parts`` are non-colliding display meshes. ``collision_parts`` are
    individually convex and carry local transforms. Their union, not their
    global convex hull, is the physical shape. Geometry is one of ``box``,
    ``cylinder``, and ``convex_mesh``. Explicit non-dynamic fixtures may instead
    use ``triangle_mesh`` to preserve an original surface without convex cooking.
    Dynamic asset origins coincide with the
    declared COM; static origins are at the bottom centre of their base.
    """
    catalog = load_task_catalog(asset_root)
    try:
        result = deepcopy(catalog["assets"][asset_id])
    except KeyError:
        raise KeyError(f"Unknown task asset {asset_id!r}; available: {sorted(catalog['assets'])}") from None
    root = Path(catalog["asset_root"])
    result.update(asset_id=asset_id, asset_root=str(root),
                  manifest_path=catalog["manifest_path"],
                  manifest_sha256=catalog["manifest_sha256"])
    result["mesh_path"] = str(root / result["mesh_path"])
    combined = Path(result["mesh_path"])
    if not combined.is_file():
        raise FileNotFoundError(combined)
    if verify_hashes and hashlib.sha256(combined.read_bytes()).hexdigest() != result["sha256"]:
        raise ValueError(f"Task asset hash mismatch: {combined}")
    for part in result["visual_parts"] + result["collision_parts"]:
        if "mesh_path" in part:
            path = root / part["mesh_path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != part["sha256"]:
                raise ValueError(f"Task asset hash mismatch: {path}")
            part["mesh_path"] = str(path)
    return result


def load_obj_mesh(path):
    """Read the generator's triangulated OBJ into NumPy arrays, without GPU use."""
    import numpy as np
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            face = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
            if len(face) != 3:
                raise ValueError(f"Expected triangular OBJ face in {path}")
            faces.append(face)
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)
