"""Original CC0 render geometry for kitchen props, independent of collision.

The sponge finish replaces its complete visual shell. Its pores are shallow
concave geometry, not ellipsoids placed on top of a solid visible surface. This
does not implement foam deformation or alter the task's rigid collision body.
Dimensions here use the historical 100 x 65 x 45 mm local envelope; a caller
may scale both the finish and the independently authored contact geometry.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from . import geometry as g


SPONGE_DETAIL_SEED = 20261023
SPONGE_MAX_INSET_M = .00015


def _sponge_shell():
    # The extra 4 mm ring is a material boundary, not a geometric protrusion.
    return g.rounded_loft([
        (0, 0, 0, 0), (.050, .0325, .003, 0),
        (.050, .0325, .003, .004), (.050, .0325, .003, .022),
        (.033, .020, .003, .038), (.031, .019, .003, .045),
        (0, 0, 0, .045),
    ])


def _surface_samples(mesh, count, rng):
    """Area-weighted deterministic pore centres on the original outer skin."""
    indices = rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)
    triangle = np.asarray(mesh.triangles)[indices]
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1.] = 1. - uv[uv.sum(axis=1) > 1.]
    return (triangle[:, 0] + uv[:, :1] * (triangle[:, 1] - triangle[:, 0])
            + uv[:, 1:] * (triangle[:, 2] - triangle[:, 0]))


def sponge_visual_geometry():
    """Return ``(name, mesh, palette_key)`` meshes replacing every old visual.

    Two shared-boundary meshes describe yellow porous foam and the full green
    scouring layer. All detail is inset by at most 0.15 mm before scaling (at
    most 0.18 mm for the selected 1.2 x 1.2 x 1 enlargement). The original
    underside stays flat. The caller must not keep the old visible foam mesh
    underneath these meshes, as that would occlude the recessed pores.
    """
    source = _sponge_shell()
    vertices, faces = trimesh.remesh.subdivide_to_size(
        source.vertices, source.faces, max_edge=.0018, max_iter=9)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    before = np.array(mesh.vertices, copy=True)
    normals = np.asarray(mesh.vertex_normals)
    rng = np.random.default_rng(SPONGE_DETAIL_SEED)
    centres = _surface_samples(source, 1100, rng)
    # Leave the flat bottom and the material seam free of holes. This is visual
    # porosity only: the green scouring layer and foam remain one rigid body.
    centres = centres[centres[:, 2] > .005]
    radii = rng.uniform(.00085, .00175, len(centres))
    depths = rng.uniform(.00007, SPONGE_MAX_INSET_M, len(centres))
    distance, index = cKDTree(centres).query(before, k=4)
    profile = np.maximum(0., 1. - (distance / radii[index]) ** 2) ** 2
    inset = np.max(depths[index] * profile, axis=1)

    # Seven shallow grooves on the raised grip are cut into the foam itself.
    # They do not widen the grasp envelope or add hidden contact geometry.
    top = (before[:, 2] > .04499) & (np.abs(before[:, 1]) < .014)
    groove_x = np.linspace(-.024, .024, 7)
    groove_distance = np.min(np.abs(before[:, :1] - groove_x), axis=1)
    groove = .00012 * np.maximum(0., 1. - (groove_distance / .0006) ** 2) ** 2
    inset[top] = np.maximum(inset[top], groove[top])

    # Fine crossed channels suggest the nonwoven scouring side. They are
    # shallow cuts, all above the unchanged lower support plane.
    band = (before[:, 2] > .00025) & (before[:, 2] < .00375)
    weave = (.5 + .5 * np.sin((before[:, 0] + before[:, 1]) * 6200.)) * (
        .5 + .5 * np.sin((before[:, 0] - before[:, 1]) * 4900.))
    inset[band] = np.maximum(inset[band], .000065 * weave[band])
    inset[before[:, 2] < .00001] = 0.
    mesh.vertices = before - normals * inset[:, None]
    # Classify using original coordinates so the split cannot shift as a
    # consequence of a depression near the layer boundary.
    scouring = before[np.asarray(mesh.faces), 2].mean(axis=1) < .004
    # Shallow cavities need diffuse variation as well as normals under the
    # kitchen's broad, soft lighting. This is appearance pigmentation; it does
    # not increase the geometric cavity depth or create foam compliance.
    colour_rng = np.random.default_rng(SPONGE_DETAIL_SEED + 1)
    micro = colour_rng.uniform(-.025, .025, len(before))
    foam_tone = np.clip(.96 - .46 * np.sqrt(inset / SPONGE_MAX_INSET_M) + micro, .43, .99)
    strand = (.5 + .5 * np.sin(before[:, 0] * 2700 + before[:, 1] * 1900
                              + before[:, 2] * 2100)) ** 5
    cross_strand = (.5 + .5 * np.sin(before[:, 0] * 1300 - before[:, 1] * 3100
                                    + before[:, 2] * 1700)) ** 5
    scouring_tone = np.clip(.69 + .23 * strand + .18 * cross_strand + micro, .64, 1.08)
    vertex_index = cKDTree(mesh.vertices)
    result = []
    for name, selection, material in (
        ('porous_foam', ~scouring, 'crumb'),
        ('green_scouring_layer', scouring, 'leaf'),
    ):
        part = mesh.submesh([np.flatnonzero(selection)], append=True, repair=False)
        distance, indices = vertex_index.query(part.vertices)
        if np.max(distance) != 0.:
            raise ValueError('Sponge visual colour transfer changed surface vertices')
        base = np.array([.91, .68, .33] if material == 'crumb' else [.11, .25, .038])
        tone = foam_tone[indices] if material == 'crumb' else scouring_tone[indices]
        rgb = np.clip(base[None, :] * tone[:, None], 0., 1.)
        part.visual.vertex_colors = np.column_stack((np.rint(rgb * 255).astype(np.uint8),
                                                    np.full(len(rgb), 255, dtype=np.uint8)))
        part.metadata.update(
            visual_only=True, collision=False, original_license='CC0-1.0',
            pore_geometry='Recessed surface dimples; no protruding pore spheres.',
            max_local_inset_m=float(np.max(inset)),
            declared_max_local_inset_m=SPONGE_MAX_INSET_M,
            seed=SPONGE_DETAIL_SEED, pore_count=int(len(centres)),
            physical_approximation='Rigid dry sponge; no simulated foam compliance.',
            vertex_colour_profile='sponge_pore_and_nonwoven_colour_v2',
            vertex_colour_basis='Actual geometric inset plus fixed-seed pigmentation; no physical material change.',
        )
        result.append((name, part, material))
    return result


def sponge_detail_geometry():
    """Compatibility spelling for the complete visual replacement, not overlay."""
    return sponge_visual_geometry()
