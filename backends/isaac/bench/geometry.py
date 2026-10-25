"""Small dependency-free rigid-transform routines (SI, column vectors, wxyz)."""
from __future__ import annotations
import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def rotvec_matrix(v):
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-10:
        return np.eye(3) + skew(v)
    k = skew(v / angle)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected finite nonzero wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def matrix_quaternion(r):
    r = np.asarray(r, dtype=float)
    # Symmetric eigen decomposition is stable near pi and identity.
    xx, xy, xz = r[0]
    yx, yy, yz = r[1]
    zx, zy, zz = r[2]
    k = np.array([[xx-yy-zz, xy+yx, xz+zx, zy-yz],
                  [xy+yx, yy-xx-zz, yz+zy, xz-zx],
                  [xz+zx, yz+zy, zz-xx-yy, yx-xy],
                  [zy-yz, xz-zx, yx-xy, xx+yy+zz]]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def matrix_rotvec(r):
    q = matrix_quaternion(r)
    n = np.linalg.norm(q[1:])
    return 2 * q[1:] if n < 1e-10 else q[1:] * (2*np.arctan2(n, q[0]) / n)


def transform(position=(0, 0, 0), rotation=None):
    out = np.eye(4)
    out[:3, :3] = np.eye(3) if rotation is None else rotation
    out[:3, 3] = position
    return out


def rpy_matrix(rpy):
    r, p, y = rpy
    return rotvec_matrix([0, 0, y]) @ rotvec_matrix([0, p, 0]) @ rotvec_matrix([r, 0, 0])


def rotation_distance(a, b):
    return float(np.linalg.norm(matrix_rotvec(np.asarray(a) @ np.asarray(b).T)))
