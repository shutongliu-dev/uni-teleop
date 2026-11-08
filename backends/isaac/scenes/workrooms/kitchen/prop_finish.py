"""Small noncollision finish details for the household-size kitchen recipe."""
import math

import numpy as np

from . import geometry as g


def _ring(radius, z, width=.00012):
    return g.lathe([(radius+width*math.cos(t), z+width*math.sin(t))
                    for t in np.linspace(0, math.tau, 9)], 128)


def refine_visuals(author):
    """Use the original metric recipe before enlargement; never add labels."""
    counts = {}
    for row in author.meta['objects']:
        name, details = row['id'], []
        if name == 'sponge':
            from .prop_details import sponge_visual_geometry
            author.stage.RemovePrim(row['body_path']+'/Visual')
            details = sponge_visual_geometry()
        elif name in ('plate', 'dinner_plate', 'side_plate'):
            ratio = {'plate': 1., 'dinner_plate': 1.075, 'side_plate': .85}[name]
            details = [('glazed_rim', _ring(.067*ratio, .0149), 'blue'),
                       ('inner_rim', _ring(.066*ratio, .0149, .00006), 'brass')]
        elif name in ('cup', 'mug'):
            details = [('rolled_glaze_lip', _ring(.0305, .08005), 'cream'),
                       ('foot_glaze', _ring(.028, .0016, .00008), 'blue')]
        elif name == 'tea_cup':
            details = [('rolled_glaze_lip', _ring(.0325, .05803), 'cream')]
        elif name.startswith(('spoon', 'fork', 'knife')):
            z = .00808 if name.startswith('spoon') else .00608
            for i in range(7):
                details.append((f'brushed_handle_{i}',
                    g.box((.00007, .037, .00004), ((i-3)*.0008, .052, z)), 'brass'))
            if name.startswith('spoon') and 'natural_rest' in row:
                # These seven meshes are authored after the spoon's physical
                # rest transform. Original handle engravings were transformed
                # with the body already; do not touch them again here.
                transform = np.asarray(row['natural_rest']['source_to_rest_transform'])
                for _, detail, _ in details:
                    detail.apply_transform(transform)
        elif name in ('rice_carton', 'pasta_carton'):
            size = (.058, .046, .112) if name == 'rice_carton' else (.056, .043, .128)
            for side in (-1, 1):
                details.append((f'fold_crease_{side+1}', g.tube([
                    (side*(size[0]/2-.003), -size[1]/2+.0001, .006),
                    (side*(size[0]/2-.003), -size[1]/2+.0001, size[2]-.015)], .00012, 6), 'toast'))
        for label, mesh, material in details:
            colours = None
            visual_material = author.materials[material]
            if name == 'sponge':
                # Dedicated visual-only shaders leave shared crumb/leaf and
                # the bound physics material exactly as before.
                from .assets import PALETTE
                colour, roughness, metallic = PALETTE[material]
                visual_material = g.material(author.stage,
                    author.root+'/Materials/SpongeFinish_'+label,
                    colour, roughness, metallic, vertex_color=True)
                colours = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.
            g.write_mesh(author.stage, row['body_path']+'/Visual/finish_'+label,
                         mesh, visual_material, colors=colours)
        if details:
            counts[name] = len(details)
    author.meta['visual_finish'] = {'profile': 'household_finish_v1', 'detail_mesh_counts': counts,
        'sponge_approximation': 'Rigid collision envelope; two-layer porous visuals do not simulate compressibility or water absorption.',
        'scope': 'Render-only glaze lip, tableware trim, brushed cutlery and folded carton creases; no added text or collision.'}
    if 'sponge' in counts:
        author.meta['visual_finish']['sponge_colour_profile'] = 'sponge_pore_and_nonwoven_colour_v2'
