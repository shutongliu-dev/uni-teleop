"""Versioned task-owned workshop edge supplies; never scoring or movable props."""
from pathlib import Path
import json
from bench.isaacsim.release import checked, digest
from bench.paths import DATA_ROOT, load_json

ROOT = '/World/MechanicalDesktop'
ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/desktop_dressing/v1')


def build_assets(output=ASSET_ROOT):
    from pxr import Usd, UsdGeom
    from scenes.workrooms.mechanical_assembly.nist_assets import Author
    from scenes.workrooms.laboratory.object_geometry import box, cylinder, pose, tray_parts
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Never overwrite a desktop dressing version')
    output.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(output/'desktop.usda'))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    a = Author(stage, 'workshop_edge_supplies', ROOT)
    groups = []
    def prop(name, center, parts, color):
        item = a.body(name, pose(*center), parts, color=color)
        groups.append(item)
    # All context stays on the far x edge: the old near-half task assets and
    # both +/-Y robot approaches retain their authored working clearances.
    prop('tool_caddy', (.985, -.30, .78), tray_parts((.20,.18,.040)), 'teal')
    for i,y in enumerate((-.345,-.300,-.255)):
        parts = [cylinder(.012,.050,(0,0,.092)), cylinder(.003,.065,(0,0,.035))]
        prop('screwdriver_'+str(i), (.955,y,.784), parts, 'amber' if i != 1 else 'blue')
    prop('hex_key_rack', (1.035,-.30,.784), [box((.032,.14,.025),(0,0,.0125))], 'black')
    for i,y in enumerate((-.35,-.325,-.30,-.275,-.25)):
        prop('hex_key_'+str(i), (1.035,y,.809), [box((.004,.004,.055+i*.009),(0,0,(.055+i*.009)/2)),
             box((.024,.004,.004),(-.010,0,.055+i*.009))], 'steel')
    for i,y in enumerate((-.07,.07,.21)):
        prop('parts_bin_'+str(i), (.99,y,.78), tray_parts((.20,.12,.045)), 'blue' if i == 1 else 'teal')
        for j in range(4):
            x=.944+.029*j
            if i == 0:
                parts=[cylinder(.009,.012,(0,0,.006)), cylinder(.003,.022,(0,0,.019))]
            elif i == 1:
                from scenes.workrooms.laboratory.object_geometry import ring
                parts=ring(.006,.013,0,.004,16)
            else:
                parts=[box((.025,.018,.014),(0,0,.007))]
            prop(f'fixed_bin_content_{i}_{j}', (x,y,.784), parts, 'steel')
    prop('caliper_pad', (.99,.355,.78), [box((.23,.10,.003),(0,0,.0015))], 'black')
    prop('fixed_caliper', (.99,.355,.783), [box((.20,.013,.006),(0,0,.003)),
        box((.022,.045,.015),(-.055,.010,.0075)),
        box((.009,.048,.006),(-.093,.018,.003)),
        box((.009,.035,.006),(-.043,.017,.003))], 'steel')
    prop('caliper_readout', (.935,.365,.798), [box((.016,.021,.001),(0,0,.0005))], 'black')
    # Raised model number strips are visible structure; labels are not sensors.
    for i,y in enumerate((-.07,.07,.21)):
        prop('bin_label_'+str(i), (.889,y,.80), [box((.002,.055,.016))], 'white')
    stage.GetRootLayer().Save()
    metadata = {'schema':'mechanical_desktop_dressing.v1','root_prim_path':ROOT,
        'owner':'task','static_only':True,'scoring_support':False,'grasp_targets':False,
        'reserved_work_region_m':{'x':[.15,.80],'y':[-.50,.50]},
        'dressing_region_m':{'x':[.86,1.12],'y':[-.43,.43],'z':[.78,.97]},
        'groups':groups,'group_count':len(groups),'collider_count':sum(len(g['collider_paths']) for g in groups),
        'source':{'kind':'original_procedural','license':'CC0-1.0'},
        'note':'Fixed tool caddy, screwdrivers, hex keys, three stocked parts bins and caliper. Never task actors or scoring support.'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    manifest={'schema':'mechanical_desktop_dressing_asset.v1','usd_path':str(output/'desktop.usda'),
        'usd_sha256':digest(output/'desktop.usda'),'metadata_path':str(output/'metadata.json'),
        'metadata_sha256':digest(output/'metadata.json'),'root_prim_path':ROOT,'static_only':True,
        'dependencies':{},'physics_accepted':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def task_layer(manifest=ASSET_ROOT/'manifest.json'):
    path=Path(manifest).resolve()
    row=load_json(path)
    if row.get('schema') != 'mechanical_desktop_dressing_asset.v1' or row.get('static_only') is not True or row.get('root_prim_path') != ROOT:
        raise ValueError('Expected immutable static workshop dressing')
    checked(row['usd_path'],row['usd_sha256']); checked(row['metadata_path'],row['metadata_sha256'])
    return {'id':'mechanical_desktop_dressing','usd_path':row['usd_path'],'sha256':row['usd_sha256'],
        'source_prim_path':ROOT,'target_prim_path':ROOT,'rigid_bodies':{},'particle_sets':{},
        'dependencies_sha256':{str(path):digest(path),row['metadata_path']:row['metadata_sha256']}}


def scene_dependencies(scene):
    """Inventory an explicitly selected desktop layer; old releases stay identical."""
    reference = scene.get('mechanical_desktop_dressing')
    layers = [row for row in scene.get('task_layers', [])
              if row.get('id') == 'mechanical_desktop_dressing']
    if reference is None:
        if layers:
            raise ValueError('Desktop dressing requires its frozen manifest reference')
        return {}
    if not isinstance(reference, dict) or set(reference) != {'manifest_path', 'manifest_sha256'}:
        raise ValueError('Desktop dressing requires a manifest path and SHA256')
    checked(reference['manifest_path'], reference['manifest_sha256'])
    expected = task_layer(reference['manifest_path'])
    if layers != [expected]:
        raise ValueError('Desktop task layer differs from its frozen manifest')
    return {expected['usd_path']: expected['sha256'], **expected['dependencies_sha256']}
