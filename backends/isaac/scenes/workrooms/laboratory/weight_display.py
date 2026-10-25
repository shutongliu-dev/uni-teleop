"""Two noncollision seven-segment screens on the task-owned balance base.

Existing generic boolean material bindings drive the segments at camera sync;
no live transform, body parameter or contact field is written here.
"""
from copy import deepcopy


def add_weighing_display(stage, metadata):
    from pxr import Gf, UsdGeom, UsdShade, Sdf
    if metadata['task_id'] != 'solid_weighing':
        raise ValueError('The balance readout belongs to the solid weighing asset')
    root = metadata['fixtures']['balance_platform'] + '/Readout'
    if stage.GetPrimAtPath(root):
        raise ValueError('Balance readout is already authored')
    UsdGeom.Xform.Define(stage, root)
    material_root = root + '/Materials'
    colors = {'screen': (.006, .014, .018), 'label': (.67, .76, .72), 'off': (.018, .032, .029)}
    materials = {}
    for name, rgb in colors.items():
        material = UsdShade.Material.Define(stage, material_root + '/' + name)
        shader = UsdShade.Shader.Define(stage, material_root + '/' + name + '/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.48)
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
    for name, position, rotation in (('Top', (0., -.0538, .02235), 0.),
                                      ('Front', (0., -.0627, .011), 90.)):
        parent = root + '/' + name
        panel = UsdGeom.Xform.Define(stage, parent)
        panel.AddTranslateOp().Set(Gf.Vec3d(*position))
        panel.AddRotateXOp().Set(rotation)
        cube(parent+'/Screen', (0.,0.,0.), (.112,.015,.00012), 'screen')
        label(parent, 'NET', -.051, .002)
        label(parent, 'NOMINAL', -.051, -.003, pitch=.00052)
        label(parent, 'g', .041, 0., pitch=.001)
        centers = {'a': (0.,.0041), 'b': (.003,.00205), 'c': (.003,-.00205),
                   'd': (0.,-.0041), 'e': (-.003,-.00205), 'f': (-.003,.00205), 'g': (0.,0.)}
        segments = {}
        for digit in range(3):
            x = -.019 + digit*.016
            for segment, (sx, sy) in centers.items():
                key = f'digit_{digit}_{segment}'
                size = (.0053,.0008,.00012) if segment in 'adg' else (.0008,.0033,.00012)
                segments[key] = (x+sx, sy, size)
        segments['decimal'] = (.0043,-.0037,(.0011,.0011,.00012))
        segments['stable'] = (.049,.0044,(.002,.002,.00012))
        for key, (x, y, size) in segments.items():
            path = parent + '/' + key
            cube(path, (x,y,.00014), size, 'off')
            bindings.append({'id': 'balance_'+name.lower()+'_'+key, 'target_prim_path': path,
                'state_path': ['detail','balance_display','segments',key],
                'false_material': deepcopy(off), 'true_material': deepcopy(on)})
    metadata['visual_state_bindings'] = {'schema': 'boolean-material-bindings-v1', 'bindings': bindings}
    metadata['balance_display'] = {'root_prim_path': root, 'unit': 'g', 'precision_g': .1,
        'tare_mass_g': 28., 'minimum_stable_s': .3, 'invalid_text': '---',
        'measurement': 'Nominal net granule inventory, not calibrated force sensing.',
        'observer_state_path': ['detail','balance_display'], 'collision_enabled': False}
    return metadata
