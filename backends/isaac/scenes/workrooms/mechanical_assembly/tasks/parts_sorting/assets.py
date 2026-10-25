"""Original metre-scale sorting parts and open trays, with physical mass data."""
from pathlib import Path
import hashlib
import json

import numpy as np
import trimesh

from bench.paths import DATA_ROOT
from bench.simulation.assets.scene_parts import Asset, write_library

ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/parts_sorting/v2')
DENSITY_KG_M3 = 2700.
PARTS = {
    'block': {'geometry': 'box', 'size_m': [.060, .060, .045], 'color': 'plastic_blue'},
    'cylinder': {'geometry': 'cylinder', 'radius_m': .030, 'height_m': .045,
                 'sections': 96, 'color': 'plastic_yellow'},
    'ring': {'geometry': 'annulus', 'outer_radius_m': .0375, 'inner_radius_m': .0175,
             'height_m': .035, 'segments': 96, 'sector_count': 96, 'bevel_m': 0.,
             'bore': True, 'bore_radius_m': .0175, 'color': 'plastic_red'},
}


def prepare_assets(output=ASSET_ROOT):
    """Write a new library; density is an explicit simulation prior, not a measurement."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    assets = []
    for name, part in PARTS.items():
        asset = Asset('sorting_'+name, dynamic=True, mass_kg=1., friction=.4)
        asset.add_part(part, label=name)
        if name == 'cylinder':
            # Preserve all 96 circle vertices while keeping each cooked hull
            # below the GPU contact limits (64 vertices, 32 polygon faces).
            asset.collision.clear()
            for quadrant in range(4):
                angles = np.arange(quadrant*24, (quadrant+1)*24+1)*2*np.pi/96
                xy = np.vstack(([0., 0.], .03*np.c_[np.cos(angles), np.sin(angles)]))
                vertices = np.vstack([np.c_[xy, np.full(len(xy), z)] for z in (-.0225, .0225)])
                mesh = trimesh.convex.convex_hull(vertices)
                asset.add_collision(mesh, f'cylinder_sector_{quadrant}', geometry='convex_mesh', position_m=[0., 0., 0.])
        # Collision sectors are disjoint except shared boundaries, so their
        # signed volume integrals add without overlap or a bore-closing hull.
        solid = trimesh.util.concatenate([mesh for _, mesh, _ in asset.collision])
        if not solid.is_watertight or solid.volume <= 0:
            raise ValueError('Invalid sorting collision solid: '+name)
        props = solid.mass_properties
        inertia = np.asarray(props['inertia']) * DENSITY_KG_M3
        com = np.asarray(props['center_mass'])
        if np.max(np.abs(inertia-np.diag(np.diag(inertia)))) > 1e-10:
            raise ValueError('Sorting body axes must be principal axes')
        asset.mass_kg = float(props['volume'] * DENSITY_KG_M3)
        asset.center_of_mass_m = com.tolist()
        asset.data.update(inertia_diag_kg_m2=np.diag(inertia).tolist(),
            inertia_method='Exact uniform-density closed-polyhedron collision-piece volume integrals.',
            density_kg_m3=DENSITY_KG_M3,
            density_source='Synthetic homogeneous aluminium-density simulation prior; visual colors identify shape, not material.',
            collision_volume_m3=float(props['volume']),
            geometry_design=part,
            visual_collision_difference='Ring display bevel 0.6 mm; physics has flat 96-sector rim and the declared clear bore.' if name == 'ring' else None)
        assets.append((asset.asset_id, asset))
    for name in PARTS:
        tray = Asset('sorting_tray_'+name, dynamic=False, mass_kg=0, friction=.4)
        tray.add_part({'geometry': 'box', 'size_m': [.142, .142, .006], 'color': 'steel'},
                      label='floor', offset=[0, 0, .003])
        for sign in (-1, 1):
            tray.add_part({'geometry': 'box', 'size_m': [.006, .142, .018], 'color': 'steel'},
                          label='wall_x_'+str(sign), offset=[sign*.068, 0, .015])
            tray.add_part({'geometry': 'box', 'size_m': [.130, .006, .018], 'color': 'steel'},
                          label='wall_y_'+str(sign), offset=[0, sign*.068, .015])
        # Shape icons are engraved/color cues only and never seal the tray.
        icon = Asset('icon', dynamic=False, mass_kg=0)
        icon.add_part(PARTS[name], label='icon')
        symbol = trimesh.util.concatenate([mesh for _, mesh, _ in icon.visual])
        symbol.apply_scale([.4, .4, .002])
        symbol.apply_translation([0, 0, .0061])
        tray.add_display(symbol, PARTS[name]['color'], label='shape_icon')
        tray.data.update(interior_xy_m=[.130, .130], floor_top_z_m=.006,
                         wall_height_above_floor_m=.018, matching_shape=name)
        assets.append((tray.asset_id, tray))
    result = write_library(output, assets, generator=__name__+'.prepare_assets',
        extra={'physics_prior': {'static_friction': .4, 'dynamic_friction': .4,
            'restitution': 0., 'friction_combine': 'average',
            'contact_offset_m': .0001, 'rest_offset_m': 0.,
            'scope': 'Explicit synthetic task contact parameters, uncalibrated.'}})
    source = {'source_kind': 'original_procedural', 'license': 'CC0-1.0',
        'authoring_source': str(Path(__file__).resolve()),
        'authoring_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dimensions_source': 'Existing parts_sorting/definition.py task design',
        'manifest_sha256': result['manifest_sha256'], 'source_assets_modified': False,
        'native_dynamics_accepted': False}
    (output/'SOURCE.json').write_text(json.dumps(source, indent=2)+'\n')
    (output/'LICENSE.txt').write_text('Original procedural geometry dedicated under CC0-1.0.\nhttps://creativecommons.org/publicdomain/zero/1.0/\n')
    return result
