"""Task balance housing and native-state-driven, noncollision display surfaces.

Existing generic boolean material bindings drive the segments at camera sync;
no live transform, body parameter or contact field is written here.
"""
from copy import deepcopy


DISPLAY_CENTER_M = (0., -.0675, .052)
DISPLAY_TILT_DEG = 55.
DISPLAY_SIZE_M = (.164, .042)


def weighing_camera_overrides():
    """Task4's ordinary rear-worktop view; overview retains both work surfaces."""
    return {'eye': [1.05, 1.56, 1.38], 'target': [.71, 2.05, .946],
            'focal_length_mm': 32., 'title_zh': '称量视角（电子秤与后工作台）'}


def balance_platform_parts():
    """Fixed apparatus only; retain the original named base and pan exactly.

    The raised console stays ahead of the pan, leaving its whole vertical load
    corridor open. These meshes join the ordinary fixture collision authoring;
    they are not render-only obstacles or an additional weighing mechanism.
    """
    import math
    import trimesh
    from scenes.workrooms.laboratory.object_geometry import box
    parts = [('base', box((.135, .125, .022), (0., 0., .011))),
             ('pan', box((.100, .090, .003), (0., 0., .0235))),
             ('housing', box((.220, .145, .016), (0., -.0015, .010)))]
    for ix, x in enumerate((-.093, .093)):
        for iy, y in enumerate((-.055, .055)):
            parts.append((f'foot_{ix}_{iy}', box((.022, .020, .004), (x, y, .002))))
    for index, x in enumerate((-.070, .070)):
        parts.append((f'console_support_{index}', box((.012, .018, .019), (x, -.064, .0255))))
    frame = box((.180, .052, .006), (0., 0., -.0035))
    frame.apply_transform(trimesh.transformations.rotation_matrix(math.radians(DISPLAY_TILT_DEG), (1, 0, 0)))
    frame.apply_translation(DISPLAY_CENTER_M)
    parts.append(('display_frame', frame))
    return parts


def add_weighing_display(stage, metadata):
    from pxr import Gf, UsdGeom, UsdShade, Sdf
    if metadata['task_id'] != 'solid_weighing':
        raise ValueError('The balance readout belongs to the solid weighing asset')
    root = metadata['fixtures']['balance_platform'] + '/Readout'
    if stage.GetPrimAtPath(root):
        raise ValueError('Balance readout is already authored')
    UsdGeom.Xform.Define(stage, root)
    material_root = root + '/Materials'
    colors = {'screen': (.006, .014, .018), 'label': (.67, .76, .72), 'off': (.018, .032, .029),
              'on': (.18, .82, .55)}
    materials = {}
    for name, rgb in colors.items():
        material = UsdShade.Material.Define(stage, material_root + '/' + name)
        shader = UsdShade.Shader.Define(stage, material_root + '/' + name + '/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.48)
        if name == 'on':
            shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.20, 1.25, .70))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        materials[name] = material

    def cube(path, center, size, material):
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1.)
        shape.AddTranslateOp().Set(Gf.Vec3d(*center))
        shape.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(materials[material])

    # Tiny geometric labels need no font/texture files and survive freezing.
    font = {'N': ('101','111','111','111','101'), 'E': ('111','100','110','100','111'),
        'T': ('111','010','010','010','010'), 'g': ('111','101','111','001','111'),
        'O': ('111','101','101','101','111'), 'M': ('101','111','111','101','101'),
        'I': ('111','010','010','010','111'), 'A': ('010','101','111','101','101'),
        'L': ('100','100','100','100','111')}
    def label(parent, word, x, y, pitch=.00065):
        for index, letter in enumerate(word):
            for row, bits in enumerate(font[letter]):
                for col, bit in enumerate(bits):
                    if bit == '1':
                        cube(f'{parent}/Label_{word}_{index}_{row}_{col}',
                             (x+(index*4+col)*pitch, y+(2-row)*pitch, .00010),
                             (pitch*.78, pitch*.78, .00010), 'label')

    off = dict(diffuse_color_rgb=[.018,.032,.029], emissive_color_rgb=[0.,0.,0.], roughness=.4, metallic=0.)
    on = dict(diffuse_color_rgb=[.18,.82,.55], emissive_color_rgb=[.20,1.25,.70], roughness=.4, metallic=0.)
    bindings = []
    panels = []
    for name, position, rotation, size, scale in (
            ('Top', DISPLAY_CENTER_M, DISPLAY_TILT_DEG, DISPLAY_SIZE_M, 2.8),
            ('Front', (0., -.0741, .010), 90., (.112, .015), 1.)):
        parent = root + '/' + name
        panel = UsdGeom.Xform.Define(stage, parent)
        panel.AddTranslateOp().Set(Gf.Vec3d(*position))
        panel.AddRotateXOp().Set(rotation)
        cube(parent+'/Screen', (0.,0.,0.), (*size,.00012), 'screen')
        if name == 'Top':
            label(parent, 'NET', -.076, .006, pitch=.0012)
            label(parent, 'NOMINAL', -.076, -.005, pitch=.0008)
            label(parent, 'g', .057, -.005, pitch=.002)
        else:
            label(parent, 'NET', -.051, .002)
            label(parent, 'NOMINAL', -.051, -.003, pitch=.00052)
            label(parent, 'g', .041, 0., pitch=.001)
        centers = {'a': (0.,.0041), 'b': (.003,.00205), 'c': (.003,-.00205),
                   'd': (0.,-.0041), 'e': (-.003,-.00205), 'f': (-.003,.00205), 'g': (0.,0.)}
        segments = {}
        for digit in range(3):
            x = (-.009 + digit*.011)*scale if name == 'Top' else -.019 + digit*.016
            for segment, (sx, sy) in centers.items():
                key = f'digit_{digit}_{segment}'
                extent = (.0053,.0008) if segment in 'adg' else (.0008,.0033)
                segments[key] = (x+sx*scale, sy*scale, (extent[0]*scale,extent[1]*scale,.00012))
        segments['decimal'] = ((.0093 if name == 'Top' else .0043)*scale,-.0037*scale,
                               (.0011*scale,.0011*scale,.00012))
        segments['stable'] = ((.073,.014,(.003,.003,.00012)) if name == 'Top'
                              else (.049,.0044,(.002,.002,.00012)))
        for key, (x, y, extent) in segments.items():
            path = parent + '/' + key
            # Author the honest unknown state for static previews too. Runtime
            # camera sync still replaces every segment from the native reader.
            cube(path, (x,y,.00014), extent, 'on' if key.endswith('_g') else 'off')
            bindings.append({'id': 'balance_'+name.lower()+'_'+key, 'target_prim_path': path,
                'state_path': ['detail','balance_display','segments',key],
                'false_material': deepcopy(off), 'true_material': deepcopy(on)})
        panels.append({'id': name.lower(), 'screen_prim_path': parent+'/Screen',
                       'panel_prim_path': parent, 'size_m': list(size),
                       'digit_height_m': .009*scale, 'primary': name == 'Top'})
    metadata['visual_state_bindings'] = {'schema': 'boolean-material-bindings-v1', 'bindings': bindings}
    metadata['balance_display'] = {'root_prim_path': root, 'unit': 'g', 'precision_g': .1,
        'tare_mass_g': 28., 'minimum_stable_s': .3, 'invalid_text': '---',
        'measurement': 'Nominal net granule inventory, not calibrated force sensing.',
        'observer_state_path': ['detail','balance_display'], 'collision_enabled': False,
        'panels': panels, 'default_text': '---', 'primary_panel': 'top'}
    return metadata
