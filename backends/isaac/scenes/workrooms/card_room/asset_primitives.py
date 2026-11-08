"""Shared tabletop piece mass, lettering, frame and pedestal primitives.

These original CC0 recipes retain their established geometry and material
priors. Game-specific boards and token appearance are authored in game_assets.
"""
from __future__ import annotations

import math

import numpy as np

from bench.simulation.assets.scene_parts import Asset, box_mesh
from bench.simulation.assets.inlay_geometry import offset_outline, rounded_box, rounded_rectangle

TIMBER = [.105, .061, .036]
APPEARANCE = 'hex_heritage_v3'


_LETTERING = {
    'A': [[(0, 0), (.5, 1), (1, 0)], [(.2, .38), (.8, .38)]],
    'B': [[(0, 0), (0, 1), (.65, 1), (1, .82), (1, .65), (.65, .5), (0, .5)],
          [(.65, .5), (1, .35), (1, .18), (.65, 0), (0, 0)]],
    'C': [[(1, .85), (.75, 1), (.25, 1), (0, .75), (0, .25), (.25, 0), (.75, 0), (1, .15)]],
    'D': [[(0, 0), (0, 1), (.6, 1), (1, .75), (1, .25), (.6, 0), (0, 0)]],
    'E': [[(1, 1), (0, 1), (0, 0), (1, 0)], [(0, .5), (.8, .5)]],
    'F': [[(1, 1), (0, 1), (0, 0)], [(0, .5), (.8, .5)]],
    'G': [[(1, .85), (.75, 1), (.25, 1), (0, .75), (0, .25), (.25, 0), (.75, 0),
           (1, .25), (1, .48), (.55, .48)]],
    '1': [[(.15, .72), (.55, 1), (.55, 0)], [(.15, 0), (.95, 0)]],
    '2': [[(0, .8), (.2, 1), (.75, 1), (1, .8), (1, .6), (0, 0), (1, 0)]],
    '3': [[(0, 1), (1, 1), (.5, .55), (.8, .55), (1, .35), (1, .2), (.75, 0), (.15, 0), (0, .15)]],
    '4': [[(.78, 0), (.78, 1), (0, .35), (1, .35)]],
    '5': [[(1, 1), (0, 1), (0, .55), (.75, .55), (1, .35), (1, .2), (.75, 0), (.15, 0), (0, .15)]],
    '6': [[(.9, .9), (.7, 1), (.25, 1), (0, .7), (0, .2), (.2, 0), (.75, 0), (1, .2),
           (1, .45), (.75, .6), (0, .6)]],
    '7': [[(0, 1), (1, 1), (.3, 0)]],
}


def _translated(mesh, position):
    result = mesh.copy()
    result.apply_translation(position)
    return result


def token_mass_properties(geometry):
    g = geometry
    cylinders = [(g.piece_radius_m, g.piece_base_height_m, 0.),
                 (g.handle_radius_m, g.handle_height_m,
                  (g.piece_base_height_m + g.handle_height_m)/2)]
    volume = sum(math.pi*r*r*h for r, h, _ in cylinders)
    parts = [(r, h, z, g.piece_mass_kg*math.pi*r*r*h/volume) for r, h, z in cylinders]
    center = sum(z*m for _, _, z, m in parts)/g.piece_mass_kg
    ix = sum(m*(3*r*r+h*h)/12 + m*(z-center)**2 for r, h, z, m in parts)
    iz = sum(m*r*r/2 for r, _, _, m in parts)
    return {'center_of_mass_m': [0., 0., center], 'inertia_diag_kg_m2': [ix, ix, iz],
            'inertia_method': 'Analytic uniform-density two-cylinder union about COM; parallel-axis theorem.',
            'design_density_kg_m3': g.piece_mass_kg/volume,
            'mass_status': 'Simulation design prior, not a measured manufactured piece',
            'origin': 'Centre of circular base; COM is explicitly offset along +Z'}


def _strip(a, b, width, z):
    import trimesh
    a, b = np.array(a), np.array(b)
    direction = b-a
    length = np.linalg.norm(direction)
    mesh = trimesh.creation.box([length, width, .0001])
    angle = math.atan2(direction[1], direction[0])
    matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    matrix[:3, 3] = [*(a+b)/2, z]
    mesh.apply_transform(matrix)
    return mesh


def _frame_pinstripe(geometry):
    """Fine mitred frame with intentional bottom-edge gaps for legible letters."""
    import trimesh
    g = geometry
    sx, sy = g.board_size_xy_m
    outline = offset_outline(rounded_rectangle([sx-.012, sy-.012], .005), -.000325)
    letters = [g.cell_local_xy(q, 0)[0] for q in range(7)]
    strips = []
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        if abs(a[1]-b[1]) < 1.e-12 and a[1] < -sy/2+.007:
            left, right = sorted([a[0], b[0]])
            for x in letters:
                if left < x-.0065 < right:
                    strips.append(_strip([left, a[1]], [x-.0065, a[1]], .00065,
                                         g.board_thickness_m+.000055))
                    left = x+.0065
            if right > left:
                strips.append(_strip([left, a[1]], [right, a[1]], .00065,
                                     g.board_thickness_m+.000055))
        else:
            strips.append(_strip(a, b, .00065, g.board_thickness_m+.000055))
    return trimesh.util.concatenate(strips)


def stand(sign, base_height=.95):
    """Cantilever column stays outside the table; cap supports the real base."""
    color = 'red' if sign < 0 else 'blue'
    asset = Asset('hex_stand_'+color, dynamic=False, mass_kg=0, friction=.4)
    # Origin is room world XY origin. The same four exact solid box colliders
    # remain; modest edge roundovers and wood panels are appearance only.
    for name, xyz, size, shade in (
        ('foot', [.65, sign*.74, .025], [.40, .38, .05], [.065, .053, .039]),
        ('column', [.65, sign*.74, (base_height-.04)/2], [.19, .19, base_height-.04], [.145, .085, .046]),
        ('cantilever', [.65, sign*.625, base_height-.035], [.22, .41, .03], [.075, .060, .042]),
        ('mount_plate', [.65, sign*.52, base_height-.0125], [.22, .20, .025], [.17, .18, .16]),
    ):
        asset.add_display(_translated(rounded_box(size, .003, .0008), xyz), shade, name,
                          roughness=.45 if name == 'column' else .39,
                          metallic=0. if name == 'column' else .35)
        asset.add_box_collision(np.asarray(size)/2, name, xyz)
    # Flush panel veneers break up the engineered column without changing its
    # silhouette or making a slender visual hide a large invisible collider.
    panels, pinstripes = [], []
    panel_height = base_height-.23
    for axis in (0, 1):
        for side in (-1, 1):
            center = np.array([.65, sign*.74, (base_height-.04)/2])
            center[axis] += side*.095045
            size = np.array([.142, .142, panel_height])
            size[axis] = .00007
            panels.append(_translated(box_mesh(size), center))
            for offset in (-.073, .073):
                line = center.copy()
                line[1-axis] += offset
                size = np.array([.0011, .0011, panel_height+.012])
                size[axis] = .00007
                pinstripes.append(_translated(box_mesh(size), line))
    import trimesh
    asset.add_display(trimesh.util.concatenate(panels), TIMBER, 'recessed_timber_panels', roughness=.49, metallic=0.)
    asset.add_display(trimesh.util.concatenate(pinstripes), [.28, .22, .13], 'panel_pinstripes', roughness=.55, metallic=.2)
    asset.data['installation'] = 'Ideal rigid anchored pedestal; column clears table edge, original robot geometry unchanged.'
    asset.data['appearance'] = {'version': APPEARANCE, 'finish': 'Walnut panels over dark bronze support',
                               'maximum_veneer_offset_m': .00008,
                               'collision_shape': 'Unchanged four solid boxes'}
    return asset
