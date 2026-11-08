"""Parameterized mechanical catalogs with the original runtime representation.

Objects keep their original OBJ/convex or USD/SDF loader, placement form and
contact actor identity. Only fresh catalog/texture/USD derivatives are written.
No solver, controller, fitting geometry or evaluator threshold is changed.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .spec import validate_parameters
from bench.paths import load_json
from scenes.workrooms.seed_geometry import file_hash, write_json

TASKS = ('parts_sorting', 'peg_insertion', 'shaft_spacer_assembly',
         'fastener_assembly', 'gear_train_assembly')


def _read(path):
    return load_json(Path(path))


def _absolute_meshes(value, root):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'mesh_path':
                value[key] = str((root/item).resolve())
            else:
                _absolute_meshes(item, root)
    elif isinstance(value, list):
        for item in value:
            _absolute_meshes(item, root)


def _scaled_catalog(spec, parameters, output):
    """Keep one catalog's exact geometry and source dependencies; scale once."""
    from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from .usd_authoring import apply_physics
    original = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
    root = Path(spec['asset_root']).resolve()
    catalog = _read(root/'manifest.json')
    _absolute_meshes(catalog, root)
    # Keep source catalog metadata/licenses, but only the selected asset is
    # published in this object-specific derivative; no unrelated asset changes.
    asset = deepcopy(catalog['assets'][spec['asset_id']])
    catalog['assets'] = {spec['asset_id']: asset}
    factor = parameters['physics']['mass_factor']
    if factor != parameters['physics']['density_factor']:
        raise ValueError('Mass/density must use one common factor')
    dependencies = {str(root/'manifest.json'): file_hash(root/'manifest.json')}
    native_path = '/World/Objects/'+spec['id']
    if original.get('backend') == 'usd_reference':
        from scenes.workrooms.mechanical_assembly.placement_variants import catalog_body_path
        native_path = catalog_body_path(spec)
        stage = Usd.Stage.Open(Usd.Stage.Open(original['usd_path']).Flatten())
        default = original['usd_default_prim']
        body_path = original['source_rigid_body_prim']
        body = stage.GetPrimAtPath(body_path)
        mass = UsdPhysics.MassAPI(body).GetMassAttr().Get()
        if mass is None or not math.isclose(float(mass), original['mass_kg'], rel_tol=2e-5, abs_tol=1e-8):
            raise ValueError('Original catalog mass differs from actual source USD')
        # Reproduce the original loader's explicit task/runtime fallback, never
        # mistake USD schema defaults for the source's native contact material.
        fallback = spec.get('usd_physics_material_fallback')
        fallback_values = ({k: fallback[k] for k in ('static_friction','dynamic_friction','restitution')}
                           if fallback else {'static_friction': spec.get('friction', .65),
                                             'dynamic_friction': spec.get('friction', .65), 'restitution': 0.})
        baseline = None
        for prim in Usd.PrimRange(body):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            binding = UsdShade.MaterialBindingAPI(prim)
            resolved, _ = binding.ComputeBoundMaterial('physics')
            has_material = bool(resolved and resolved.GetPrim().HasAPI(UsdPhysics.MaterialAPI))
            if 'friction' in spec or not has_material:
                if baseline is None:
                    baseline = UsdShade.Material.Define(stage, default+'/ParameterizedOriginalFallback')
                    api = UsdPhysics.MaterialAPI.Apply(baseline.GetPrim())
                    api.CreateStaticFrictionAttr(float(fallback_values['static_friction']))
                    api.CreateDynamicFrictionAttr(float(fallback_values['dynamic_friction']))
                    api.CreateRestitutionAttr(float(fallback_values['restitution']))
                    if fallback:
                        baseline.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
                        baseline.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode',Sdf.ValueTypeNames.Token).Set(fallback['friction_combine'])
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(baseline,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
        physics = apply_physics(stage, {spec['id']: body_path}, default, parameters)
        # Shared authoring also scales dormant collider materials once, while
        # preserving collisionEnabled=false. Never clone/scale that result again.
        disabled = [str(prim.GetPath()) for prim in Usd.PrimRange(body)
            if prim.HasAPI(UsdPhysics.CollisionAPI) and
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False]
        if (physics['material_physics'].get('collision_disabled_paths_preserved') != disabled or
                any(path not in physics['material_physics']['collider_material_bindings'] for path in disabled)):
            raise ValueError('Shared physics author omitted an original disabled collider')
        target = output/'physical.usda'
        stage.GetRootLayer().Export(str(target))
        row = physics['body_physics'][spec['id']]
        row['source_prim_path'] = row['prim_path']; row['prim_path'] = native_path
        asset['usd_path'] = str(target)
        asset['mass_kg'] = row['mass_kg']
        if row['inertia_mode'] == 'explicit_scaled' and 'inertia_diag_kg_m2' in asset:
            asset['inertia_diag_kg_m2'] = deepcopy(row['diagonal_inertia_kg_m2'])
        dependencies.update(original['source_dependencies_sha256'])
        dependencies[str(target)] = file_hash(target)
        # Flattened asset paths are absolute. Include every texture/source used
        # by the exported stage; retain the original dependencies as provenance.
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                if attr.GetTypeName() == Sdf.ValueTypeNames.Asset and attr.HasAuthoredValue():
                    value = attr.Get()
                    if value and value.path:
                        path = Path(value.resolvedPath or value.path).resolve()
                        dependencies[str(path)] = file_hash(path)
        asset['source_dependencies_sha256'] = {**asset['source_dependencies_sha256'], **dependencies}
        physics['baseline_loader_fallback'] = fallback_values
    else:
        before = {'mass_kg': original['mass_kg'], 'diagonal_inertia_kg_m2': deepcopy(original['inertia_diag_kg_m2'])}
        asset['mass_kg'] = original['mass_kg']*factor
        asset['inertia_diag_kg_m2'] = (np.asarray(original['inertia_diag_kg_m2'])*factor).tolist()
        for key in ('density_kg_m3', 'density'):
            if key in asset:
                asset[key] *= factor
        mu = float(spec.get('friction', .65))
        physics = {'body_physics': {spec['id']: {'prim_path': native_path,
            'before': before, 'mass_kg': asset['mass_kg'], 'mass_factor': factor,
            'diagonal_inertia_kg_m2': asset['inertia_diag_kg_m2'], 'inertia_mode': 'explicit_scaled'}},
            'material_physics': {'baseline_origin': 'original_catalog_loader_scene_friction',
                'baseline': {'static_friction': mu, 'dynamic_friction': mu, 'restitution': 0.},
                'applied': {'static_friction': mu*parameters['physics']['friction_factor'],
                            'dynamic_friction': mu*parameters['physics']['friction_factor'], 'restitution': 0.}},
            'mass_factor_applied_once': factor}
        for item in [original, *original['visual_parts'], *original['collision_parts']]:
            if 'mesh_path' in item:
                dependencies[item['mesh_path']] = item['sha256']
    catalog['parameterized_physics'] = {'schema': parameters['schema'],
        'parameter_instance_sha256': parameters['instance_sha256'],
        'parent_manifest': {'path': str(root/'manifest.json'), 'sha256': dependencies[str(root/'manifest.json')]},
        'geometry_dimensions_changed': False, 'native_qualified': False}
    write_json(output/'manifest.json', catalog)
    dependencies[str(output/'manifest.json')] = file_hash(output/'manifest.json')
    updated = deepcopy(spec); updated['asset_root'] = str(output)
    if original.get('backend') != 'usd_reference' or 'friction' in updated:
        updated['friction'] = float(spec.get('friction', .65))*parameters['physics']['friction_factor']
    return updated, physics, dependencies


def _presentation(spec, parameters, output):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    from bench.simulation.isaacsim_backend.presentation import presentation_stage, apply_presentation
    from scenes.workrooms.mechanical_assembly.placement_authoring import _catalog_stage
    from .usd_authoring import apply_appearance
    if spec.get('presentation'):
        raise ValueError('Unexpected original presentation requires explicit preservation review')
    stage, root, _, _ = _catalog_stage(spec)
    scratch = presentation_stage(stage)
    # Original SDF rings use a uniform displayColor without a shader. Bridge
    # that authored colour to a visual shader before applying the common finish.
    # Collision-only meshes without a display colour remain untouched.
    for index, prim in enumerate(list(Usd.PrimRange(scratch.GetPrimAtPath(root)))):
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            continue
        colors = UsdGeom.Mesh(prim).GetDisplayColorAttr().Get()
        if not colors:
            continue
        if any(color != colors[0] for color in colors):
            raise ValueError('Nonuniform displayColor requires an explicit appearance adapter')
        path = root+'/ParameterizedDisplayColor/Material_'+str(index)
        material = UsdShade.Material.Define(scratch, path)
        shader = UsdShade.Shader.Define(scratch, path+'/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(colors[0]))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    appearance = apply_appearance(scratch, {spec['id']:root}, root, parameters, output)
    path = output/'presentation.usda'
    layer = Sdf.Layer.CreateNew(str(path))
    Sdf.CopySpec(scratch.GetRootLayer(), root, layer, '/Asset'); layer.Save()
    deps = {str(p):file_hash(p) for p in output.rglob('*') if p.is_file() and p != path}
    descriptor = {'schema':'asset_presentation.v1','usd_path':str(path),'sha256':file_hash(path),
                  'prim_path':'/Asset','dependencies_sha256':deps}
    # Existing production preflight rejects geometry/physics/transform changes
    # and material targets outside the selected asset before exporting a pin.
    evidence = apply_presentation(stage, root, {'presentation':descriptor})
    return descriptor, {**appearance, 'runtime_presentation_preflight':evidence}, {**deps,str(path):file_hash(path)}


def _layout(scene, task, parameters):
    from scenes.workrooms.mechanical_assembly.placement_variants import plan_case, catalog_body_path
    from .usd_authoring import transform_point
    objects = {row['id']:row for row in scene['objects']}
    fixtures = {row['id']:row for row in scene['fixtures']}
    groups = ([{'id':'loose_supply','actor_ids':list(objects),'fixture_ids':[]}]
              if task['id']=='parts_sorting' else plan_case(scene,task,0)['groups'])
    records=[]
    for group in groups:
        selected=[(objects[name],True) for name in group['actor_ids']]+[(fixtures[name],False) for name in group['fixture_ids']]
        pivot=np.mean([row['position_m'] for row,_ in selected],axis=0).tolist()
        before={catalog_body_path(row,dynamic=dynamic):{'prim_path':catalog_body_path(row,dynamic=dynamic),
            'position_m':deepcopy(row['position_m']),'quaternion_wxyz':deepcopy(row.get('quaternion_wxyz',[1.,0.,0.,0.]))}
            for row,dynamic in selected}
        layout={'id':group['id'],'paths':list(before),'pivot_m':pivot,**deepcopy(parameters['layout']),
                'before':before,'actor_ids':group['actor_ids'],'fixture_ids':group['fixture_ids']}
        if not parameters['canonical']:
            c,s=math.cos(layout['yaw_rad']/2),math.sin(layout['yaw_rad']/2)
            for row,_ in selected:
                row['position_m']=transform_point(row['position_m'],layout)
                w,x,y,z=row.get('quaternion_wxyz',[1.,0.,0.,0.])
                row['quaternion_wxyz']=[c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]
        layout['after']={catalog_body_path(row,dynamic=dynamic):{'prim_path':catalog_body_path(row,dynamic=dynamic),
            'position_m':deepcopy(row['position_m']),'quaternion_wxyz':deepcopy(row.get('quaternion_wxyz',[1.,0.,0.,0.]))}
            for row,dynamic in selected}
        records.append(layout)
    return records


def _reset_metadata(scene,task,layout):
    from .usd_authoring import transform_point
    row=layout[0]
    task['evaluation']['initial_objects']={o['id']:{k:deepcopy(o.get(k,[1.,0.,0.,0.]))
        for k in ('position_m','quaternion_wxyz')} for o in scene['objects']}
    reset=task.get('reset_conditions',{})
    for name,p in reset.get('supply_positions_m',{}).items(): reset['supply_positions_m'][name]=transform_point(p,row)
    for key in ('bolt_supply_position_m','tool_supply_position_m'):
        if key in reset: reset[key]=transform_point(reset[key],row)
    for owner in (task,scene):
        definition=owner.get('task_layout',{}); lane=definition.get('supply_lane',{})
        for key in ('positions_m','part_positions_m'):
            for name,p in lane.get(key,{}).items():lane[key][name]=transform_point(p,row)
        if lane:
            lane['axis']='world_xy_with_yaw';lane['yaw_rad']=row['yaw_rad']
        for zone in definition.get('grasp_boundary',{}).get('part_zones',{}).values():
            if 'center_xy_m' in zone:zone['center_xy_m']=transform_point([*zone['center_xy_m'],0.],row)[:2]
    supply=task.get('supply_layout',{})
    if 'gear_supply_positions_xy_m' in supply:
        supply['gear_supply_positions_xy_m']=[transform_point([*p,0.],row)[:2] for p in supply['gear_supply_positions_xy_m']]
        supply['source_asset_geometry_and_manifest_unchanged']=False
        supply['source_asset_geometry_unchanged']=True


def author_catalog(values, parameters, output):
    """Return composed values plus a private, actual-parameter application record."""
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from scenes.workrooms.mechanical_assembly.placement_variants import catalog_body_path
    validate_parameters(parameters)
    result=deepcopy(values);scene,task=result['scene'],result['task']
    if task['id'] not in TASKS or parameters['task_id']!=task['id'] or parameters['family']!='mechanical_assembly' or scene.get('native_task_bundle'):
        raise ValueError('Expected one of the five original mechanical catalog tasks')
    output=Path(output).resolve();body_physics={};material_physics={};appearance={};deps={}
    layout=_layout(scene,task,parameters)
    if parameters['canonical']:
        # Return the canonical scene/task/projections byte-for-byte as data;
        # even a quaternion or omitted friction default is not normalized.
        return {**deepcopy(values),'applied':{'layout':layout,'canonical':True,
            'body_physics':{},'material_physics':{},'appearance':{'canonical':True},
            'geometry_dimensions_changed':False,'native_readback_verified':False}}
    output.mkdir(parents=True,exist_ok=False)
    for index,spec in enumerate(scene['objects']):
        destination=output/spec['id'];destination.mkdir()
        updated,physics,dependencies=_scaled_catalog(spec,parameters,destination)
        visual=destination/'appearance';visual.mkdir()
        presentation,style,visual_deps=_presentation(updated,parameters,visual)
        updated['presentation']=presentation;scene['objects'][index]=updated
        body_physics.update(physics['body_physics']);material_physics[spec['id']]=physics['material_physics']
        appearance[spec['id']]=style;deps.update(dependencies);deps.update(visual_deps)
        # Admission reads the real derived catalog again, not just a recipe.
        current=load_task_asset(updated['asset_id'],updated['asset_root'],verify_hashes=True)
        if current['mass_kg']!=body_physics[spec['id']]['mass_kg']:
            raise ValueError('Derived catalog and applied body mass differ')
    _reset_metadata(scene,task,layout)
    result['actual_body_poses_m']={o['id']:{'prim_path':catalog_body_path(o),'position_m':deepcopy(o['position_m']),
        'quaternion_wxyz':deepcopy(o.get('quaternion_wxyz',[1.,0.,0.,0.]))} for o in scene['objects']}
    for row in result['static_geometry']['bodies']:
        row['nominal_mass_kg']=body_physics[row['object_id']]['mass_kg']
    result['asset_files_sha256'].update(deps)
    result['applied']={'layout':layout,'body_physics':body_physics,'material_physics':{'objects':material_physics},
        'appearance':{'objects':appearance},'mass_factor_applied_once':parameters['physics']['mass_factor'],
        'geometry_dimensions_changed':False,'native_readback_verified':False,
        'representation':'Original catalog object/fixture loaders and original contact actor paths.',
        'scope':'New reset pose, true catalog/USD mass and inertia, local friction and visible appearance; no inherited robot/native qualification.'}
    return result


def validate_catalog(values, parameters, applied, data):
    """Pure JSON admission: recompute allowed edits and verify all asset pins.

    Actual USD properties are checked by authoring and the independent static /
    native validators. This admission neither imports pxr nor claims a native
    readback from JSON recipe fields.
    """
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from bench.simulation.isaacsim_backend.presentation import presentation_dependencies
    from scenes.workrooms.mechanical_assembly.placement_variants import catalog_body_path
    from .workroom import _patch
    validate_parameters(parameters)
    if values['task']['id'] not in TASKS or parameters['task_id'] != values['task']['id']:
        raise ValueError('Wrong mechanical task parameter identity')
    scene, task = deepcopy(values['scene']), deepcopy(values['task'])
    layout = _layout(scene, task, parameters)
    if applied['layout'] != layout or applied['geometry_dimensions_changed'] is not False:
        raise ValueError('Mechanical layout or geometry policy differs')
    if applied['native_readback_verified'] is not False:
        raise ValueError('Authoring recipe cannot claim native readback')
    actual_scene = _patch(values['scene'], data['scene_changes'])
    actual_task = _patch(values['task'], data['task_changes'])
    projections = ('actual_body_poses_m', 'static_geometry', 'interaction_targets', 'fixed_support_paths')
    if parameters['canonical']:
        if (actual_scene != values['scene'] or actual_task != values['task'] or
                any(data[key] != values[key] for key in projections) or
                applied != {'layout':layout,'canonical':True,'body_physics':{},
                    'material_physics':{},'appearance':{'canonical':True},
                    'geometry_dimensions_changed':False,'native_readback_verified':False}):
            raise ValueError('Canonical mechanical input changed')
        return {'validated': True, 'canonical': True, 'native_readback_verified': False}

    factor = parameters['physics']['mass_factor']
    friction = parameters['physics']['friction_factor']
    names = {row['id'] for row in scene['objects']}
    if (applied['mass_factor_applied_once'] != factor or
            set(applied['body_physics']) != names or
            set(applied['material_physics']['objects']) != names or
            set(applied['appearance']['objects']) != names or
            len(actual_scene['objects']) != len(scene['objects'])):
        raise ValueError('Mechanical physical object inventory differs')

    def close(actual, expected, label):
        if not np.allclose(actual, expected, rtol=2e-5, atol=1e-9, equal_nan=False):
            raise ValueError('Mechanical physical scaling differs: '+label)

    for index, old in enumerate(scene['objects']):
        new = actual_scene['objects'][index]
        if new['id'] != old['id'] or new['asset_id'] != old['asset_id']:
            raise ValueError('Mechanical asset identity differs')
        parent = Path(old['asset_root']).resolve()
        current = Path(new['asset_root']).resolve()
        if current == parent:
            raise ValueError('Noncanonical mechanical assets must be new derivatives')
        original = load_task_asset(old['asset_id'], parent, verify_hashes=True)
        derived = load_task_asset(new['asset_id'], current, verify_hashes=True)
        baseline = _read(parent/'manifest.json'); _absolute_meshes(baseline, parent)
        asset = deepcopy(baseline['assets'][old['asset_id']])
        baseline['assets'] = {old['asset_id']:asset}
        written = _read(current/'manifest.json')
        changed = written['assets'][old['asset_id']]
        close(changed['mass_kg'], original['mass_kg']*factor, 'catalog mass')
        asset['mass_kg'] = changed['mass_kg']
        if original.get('inertia_diag_kg_m2') is not None:
            close(changed['inertia_diag_kg_m2'], np.asarray(original['inertia_diag_kg_m2'])*factor, 'catalog inertia')
            asset['inertia_diag_kg_m2'] = changed['inertia_diag_kg_m2']
        for key in ('density_kg_m3','density'):
            if key in asset:
                asset[key] *= factor
        if original.get('backend') == 'usd_reference':
            if Path(changed['usd_path']).resolve() != current/'physical.usda':
                raise ValueError('Derived mechanical USD escaped its catalog')
            asset['usd_path'] = changed['usd_path']
            pins = changed['source_dependencies_sha256']
            if (any(pins.get(path) != sha for path,sha in original['source_dependencies_sha256'].items()) or
                    pins.get(str(parent/'manifest.json')) != file_hash(parent/'manifest.json') or
                    pins.get(changed['usd_path']) != file_hash(changed['usd_path'])):
                raise ValueError('Mechanical USD provenance differs')
            asset['source_dependencies_sha256'] = pins
        baseline['parameterized_physics'] = {'schema':parameters['schema'],
            'parameter_instance_sha256':parameters['instance_sha256'],
            'parent_manifest':{'path':str(parent/'manifest.json'),'sha256':file_hash(parent/'manifest.json')},
            'geometry_dimensions_changed':False,'native_qualified':False}
        if baseline != written:
            raise ValueError('Mechanical catalog changed geometry or unsupported metadata')
        row = applied['body_physics'][old['id']]
        if row['prim_path'] != catalog_body_path(old) or row['mass_factor'] != factor:
            raise ValueError('Mechanical body identity differs')
        close(row['before']['mass_kg'], original['mass_kg'], 'source mass')
        close(row['mass_kg'], row['before']['mass_kg']*factor, 'body mass')
        if row['mass_kg'] != derived['mass_kg']:
            raise ValueError('Mechanical static/catalog body mass differs')
        before_inertia = row['before']['diagonal_inertia_kg_m2']
        if row['inertia_mode'] == 'explicit_scaled':
            close(row['diagonal_inertia_kg_m2'], np.asarray(before_inertia)*factor, 'body inertia')
            close(derived['inertia_diag_kg_m2'], row['diagonal_inertia_kg_m2'], 'catalog body inertia')
        elif row['inertia_mode'] != 'native_geometry_computation' or row['diagonal_inertia_kg_m2'] != before_inertia:
            raise ValueError('Mechanical inertia policy differs')
        material = applied['material_physics']['objects'][old['id']]
        material_rows = list(material['materials'].values()) if 'materials' in material else [material]
        if not material_rows:
            raise ValueError('Mechanical collider material inventory is empty')
        if 'materials' in material and (not material['collider_material_bindings'] or
                set(material['collider_material_bindings'].values())- {v['path'] for v in material_rows}):
            raise ValueError('Mechanical collider bindings differ')
        for item in material_rows:
            for key in ('static_friction','dynamic_friction','restitution'):
                close(item['applied'][key], item['baseline'][key]*(1 if key=='restitution' else friction), 'contact '+key)
        if original.get('backend') != 'usd_reference':
            mu = float(old.get('friction', .65))
            if material['baseline'] != {'static_friction':mu,'dynamic_friction':mu,'restitution':0.}:
                raise ValueError('Mechanical catalog friction baseline differs')
        descriptor = new['presentation']
        if (Path(descriptor['usd_path']).resolve() != current/'appearance'/'presentation.usda' or
                descriptor['prim_path'] != '/Asset'):
            raise ValueError('Mechanical presentation escaped its catalog')
        presentation_dependencies(new)
        style = applied['appearance']['objects'][old['id']]
        evidence = style['runtime_presentation_preflight']
        if (not style['meshes'] or not style['artifacts'] or not evidence['physical_inputs_unchanged'] or
                evidence['usd_path'] != descriptor['usd_path'] or evidence['sha256'] != descriptor['sha256']):
            raise ValueError('Mechanical presentation authoring evidence differs')
        for artifact in style['artifacts']:
            if artifact['pattern_id'] != parameters['appearance']['pattern_id']:
                raise ValueError('Mechanical appearance parameter differs')
            for key, sha_key in (('path','sha256'),('roughness_path','roughness_sha256')):
                if (descriptor['dependencies_sha256'].get(artifact[key]) != artifact[sha_key] or
                        file_hash(artifact[key]) != artifact[sha_key]):
                    raise ValueError('Mechanical appearance dependency differs')
        old['asset_root'] = new['asset_root']; old['presentation'] = deepcopy(descriptor)
        if original.get('backend') != 'usd_reference' or 'friction' in old:
            old['friction'] = float(old.get('friction', .65))*friction
    _reset_metadata(scene, task, layout)
    if scene != actual_scene or task != actual_task:
        raise ValueError('Mechanical scene/task contains an unsupported edit')
    poses = {o['id']:{'prim_path':catalog_body_path(o),'position_m':deepcopy(o['position_m']),
        'quaternion_wxyz':deepcopy(o.get('quaternion_wxyz',[1.,0.,0.,0.]))} for o in scene['objects']}
    geometry = deepcopy(values['static_geometry'])
    for row in geometry['bodies']:
        row['nominal_mass_kg'] = applied['body_physics'][row['object_id']]['mass_kg']
    if (data['actual_body_poses_m'] != poses or data['static_geometry'] != geometry or
            any(data[key] != values[key] for key in ('interaction_targets','fixed_support_paths'))):
        raise ValueError('Mechanical frozen geometry/targets/projection differs')
    return {'validated':True,'canonical':False,'native_readback_verified':False}
