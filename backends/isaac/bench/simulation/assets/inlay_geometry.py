"""Small CPU mesh recipes for turned pieces and flush furniture inlays.

These functions author appearance only. They do not generate collision shapes
or estimate mass properties. All dimensions are metres; polygon outlines are
convex and counterclockwise as viewed from above.
"""
from __future__ import annotations

import math

import numpy as np


def _mesh(vertices, faces):
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float),
                           faces=np.asarray(faces, dtype=np.int64), process=True)
    mesh.fix_normals(multibody=True)
    return mesh


def convex_outline(points):
    """Return the 2-D convex hull without a triangulation dependency."""
    values = sorted(set(tuple(map(float, p)) for p in points))

    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    lower, upper = [], []
    for point in values:
        while len(lower) > 1 and cross(lower[-2], lower[-1], point) <= 1.e-14:
            lower.pop()
        lower.append(point)
    for point in reversed(values):
        while len(upper) > 1 and cross(upper[-2], upper[-1], point) <= 1.e-14:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def offset_outline(outline, distance):
    """Parallel offset of a convex CCW outline; positive distances expand it."""
    outline = np.asarray(outline, dtype=float)
    edges = np.roll(outline, -1, axis=0) - outline
    edges /= np.linalg.norm(edges, axis=1)[:, None]
    normals = np.c_[edges[:, 1], -edges[:, 0]]
    previous = np.roll(normals, 1, axis=0)
    return outline + distance * (previous + normals) / (
        1. + np.sum(previous * normals, axis=1))[:, None]


def rounded_rectangle(size, radius, segments=12):
    half = np.asarray(size, dtype=float)/2
    if radius <= 0 or radius >= half.min():
        raise ValueError('Corner radius must lie inside the rectangle')
    points = []
    for center, angle in ((half-radius, 0.),
                          ([-half[0]+radius, half[1]-radius], math.pi/2),
                          (-half+radius, math.pi),
                          ([half[0]-radius, -half[1]+radius], 3*math.pi/2)):
        for theta in np.linspace(angle, angle+math.pi/2, segments+1):
            points.append(np.asarray(center) + radius*np.array([math.cos(theta), math.sin(theta)]))
    return np.asarray(points)


def _loft(rings):
    rings = np.asarray(rings)
    count = rings.shape[1]
    vertices = rings.reshape(-1, 3).tolist()
    faces = []
    for k in range(len(rings)-1):
        for j in range(count):
            a, b = k*count+j, k*count+(j+1)%count
            faces.extend([[a, b, b+count], [a, b+count, a+count]])
    for k, reverse in ((0, True), (len(rings)-1, False)):
        middle = len(vertices)
        vertices.append(rings[k].mean(axis=0).tolist())
        for j in range(count):
            a, b = k*count+j, k*count+(j+1)%count
            faces.append([middle, b, a] if reverse else [middle, a, b])
    return _mesh(vertices, faces)


def convex_prism(outline, bottom, top, bevel=0.):
    """Closed convex prism; any bevel is cut inward from the given envelope."""
    if top <= bottom or bevel < 0 or bevel*2 >= top-bottom:
        raise ValueError('Prism needs a positive height and an internal bevel')
    outline = np.asarray(outline, dtype=float)
    profile = [(outline, bottom), (outline, top)]
    if bevel:
        inset = offset_outline(outline, -bevel)
        profile = [(inset, bottom), (outline, bottom+bevel),
                   (outline, top-bevel), (inset, top)]
    return _loft([np.c_[xy, np.full(len(xy), z)] for xy, z in profile])


def rounded_box(size, corner, bevel):
    sx, sy, sz = size
    return convex_prism(rounded_rectangle([sx, sy], corner), -sz/2, sz/2, bevel)


def outline_ring(outline, width, bottom, top):
    """A narrow closed ring, entirely inside its outer outline."""
    outer = np.asarray(outline, dtype=float)
    inner = offset_outline(outer, -width)
    count = len(outer)
    vertices = np.concatenate([np.c_[xy, np.full(count, z)]
                               for z in (bottom, top) for xy in (outer, inner)])
    faces = []
    for i in range(count):
        j = (i+1)%count
        # Outer wall, inner wall, top and bottom, each with two triangles.
        for a, b, c, d in ((i, j, 2*count+j, 2*count+i),
                            (count+j, count+i, 3*count+i, 3*count+j),
                            (2*count+i, 2*count+j, 3*count+j, 3*count+i),
                            (j, i, count+i, count+j)):
            faces.extend([[a, b, c], [a, c, d]])
    return _mesh(vertices, faces)


def rounded_strip(a, b, width, bottom, top, segments=12):
    """Capsule-shaped flush inlay with semicircular, rather than square, ends."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    angle = math.atan2(*(b-a)[::-1])
    points = []
    for center, start in ((b, angle-math.pi/2), (a, angle+math.pi/2)):
        for theta in np.linspace(start, start+math.pi, segments+1):
            points.append(center + width/2*np.array([math.cos(theta), math.sin(theta)]))
    return convex_prism(points, bottom, top)


def rounded_cylinder(radius, height, edge_radius, segments=160, arc_steps=8):
    """Closed turned cylinder with small quarter-rounds inside its envelope."""
    if not 0 < edge_radius < min(radius, height/2):
        raise ValueError('Turned edge radius must fit inside the cylinder')
    profile = []
    for center_z, angles in ((-height/2+edge_radius, np.linspace(-math.pi/2, 0., arc_steps+1)),
                             (height/2-edge_radius, np.linspace(0., math.pi/2, arc_steps+1))):
        for angle in angles:
            profile.append((radius-edge_radius+edge_radius*math.cos(angle),
                            center_z+edge_radius*math.sin(angle)))
    angles = np.arange(segments)*2*math.pi/segments
    return _loft([np.c_[r*np.cos(angles), r*np.sin(angles), np.full(segments, z)]
                  for r, z in profile])
