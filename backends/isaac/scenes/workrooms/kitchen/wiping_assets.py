"""Original rigid sponge and noncollision surface soil, CC0-1.0.

The soil is a declared contact-distance cleaning approximation. Its material
change is render-only and never changes a collider or a rigid-body transform.
"""
import math

from . import geometry as g


def surface_wiping(a, presentation):
    from .assets import _pose, BAR_SUPPORT
    tray_pose = _pose(.25, -.67, .9001)
    tray = a.body('sponge_tray', 'sponge_return_tray', tray_pose,
        [('shell', g.tray((.17, .14, .018), wall=.005, floor=.004), 'blue', 'none')], 0,
        floor_top_local_z_m=.004, interior_bounds_xy_m=[[-.079, -.064], [.079, .064]],
        public_description='Blue open sponge tray; place the sponge here after wiping')
    initial = _pose(.25, -.67, .9045)
    # One convex collision body, with a full flat lower wiping face and gently
    # tapered upper grip. Decorative pores/ribs never create hidden colliders.
    sponge = g.rounded_loft([
        (0, 0, 0, 0), (.050, .0325, .003, 0),
        (.050, .0325, .003, .022), (.033, .020, .003, .038),
        (.031, .019, .003, .045), (0, 0, 0, .045)])
    details = [('scouring_band', g.box((.092, .0004, .005), (0, -.0326, .004)), 'leaf')]
    for i in range(5):
        details.append((f'grip_rib_{i}', g.box((.002, .030, .001),
                        ((i-2)*.011, 0, .0451)), 'cream'))
    for i in range(20):
        x, z = -.041+(i % 10)*.009, .011+(i//10)*.006
        details.append((f'foam_pore_{i}', g.ellipsoid((.0012, .00035, .0008),
                        (x, -.0326, z), subdivisions=1), 'bread'))
    a.body('sponge', 'rigid_cleaning_sponge', initial,
        [('foam', sponge, 'crumb', 'convexHull')], .055, details=details,
        wiping_face_bounds_xy_m=[[-.046, -.0285], [.046, .0285]],
        wiping_face_local_z_m=0.,
        public_description='Yellow rigid sponge with green scouring edge and raised grip; its flat underside wipes the counter',
        category='cleaning_tool', assigned_fixture='sponge_tray',
        physical_approximation='Rigid dry sponge; no foam compliance, fluid or detergent model.')
    patches, cells, bindings = [], [], []
    dirty = {'diffuse_color_rgb': [.30, .105, .035], 'emissive_color_rgb': [0., 0., 0.],
             'roughness': .63, 'metallic': 0.}
    # Exact material values of the pinned room's untextured kitchen_quartz.
    clean = {'diffuse_color_rgb': [.83, .82, .76], 'emissive_color_rgb': [0., 0., 0.],
             'roughness': .27, 'metallic': 0.}
    for index, center in enumerate(((.48, -.74), (.65, -.76), (.82, -.72))):
        name = 'soil_'+str(index+1)
        ids = []
        for ix in range(10):
            for iy in range(8):
                x, y = (ix-4.5)*.010, (iy-3.5)*.010
                if (x/.052)**2+(y/.042)**2 > 1.:
                    continue
                identifier = name+f'_{ix}_{iy}'
                world = [center[0]+x, center[1]+y, .90008]
                # Thin irregular soil flecks are visible but carry no physics.
                radius = .0051*(.93+.07*math.sin(ix*11+iy*7))
                a.decoration(identifier, g.ellipsoid((radius, radius, .000035), world,
                              subdivisions=1), 'toast')
                cells.append({'id': identifier, 'patch': name, 'center_world_m': world,
                              'area_m2': .0001})
                ids.append(identifier)
                bindings.append({'id': identifier, 'target_prim_path': a.root+'/Markings/'+identifier,
                    'state_path': ['detail', 'cleaned_cells', identifier],
                    'false_material': dict(dirty), 'true_material': dict(clean)})
        patches.append({'id': name, 'cell_ids': ids, 'description': 'Brown food residue patch '+str(index+1)})
    a.meta['goals'] = {'body_poses_xyzw': {'sponge': initial},
        'support_by_actor': {'sponge': 'sponge_tray'},
        'wiping_support_paths': [BAR_SUPPORT], 'wiping_surface_z_m': .90,
        'soil_patches': patches, 'soil_cells': cells,
        'cleaning_model': 'Each equal-area cell clears after actual loaded planar contact travel; not a chemical or fluid model.'}
    a.meta['source_support_by_actor'] = {'sponge': 'sponge_tray'}
    a.meta['visual_state_bindings'] = {'schema': 'boolean-material-bindings-v1', 'bindings': bindings}
    a.meta['public_goal'] = {
        'objects': [{'object_id': 'sponge', 'description': '黄色海绵，绿色擦面边缘、上部握持区；用下侧平面擦拭',
                     'category': 'cleaning_tool', 'destination': 'sponge_tray'}],
        'destinations': [{'destination_id': 'sponge_tray', 'description': '左侧蓝色浅盘，擦完后放回海绵',
                          'category': 'tool_return'}]+[
            {'destination_id': p['id'], 'description': ('左侧', '中间', '右侧')[i]+'棕色污渍，可重复擦拭',
             'category': 'surface_soil'} for i, p in enumerate(patches)],
    }
    a.meta['source'].update(source_revision='kitchen_surface_wiping_v1',
        cleaning_approximation='Rigid sponge and finite visible soil cells; actual contact and tangential travel drive clearing.')
    a.meta['layout_profile'] = 'surface_wiping_inset_4cm_v1'
