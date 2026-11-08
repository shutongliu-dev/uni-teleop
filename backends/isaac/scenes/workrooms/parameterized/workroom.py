"""Explicit parameterized publication bridge; legacy authors remain unchanged.

The canonical family author supplies the existing physical inventory. This
profile authors new USD opinions, numeric metadata and source-bound receipts.
Composition is JSON-only, so normal release admission remains pre-Kit.
"""
from copy import deepcopy
import importlib
import json
from pathlib import Path

from .spec import VERSION, canonical_hash, sample_parameters, validate_parameters, rejection_record
from bench.paths import load_json, undistributed_bytecode
from scenes.workrooms.seed_geometry import file_hash, write_json

PROJECTIONS = ('actual_body_poses_m', 'interaction_targets', 'static_geometry', 'fixed_support_paths')


def _author_files(family):
    return ('scenes/workrooms/parameterized/'+name+'.py' for name in
            ('spec','appearance','usd_authoring','workroom','workroom_rules','study_configuration')+
            (('mechanical_authoring',) if family=='mechanical_assembly' else ()))


def _source_identity(source, family):
    root=Path(source['snapshot_root']); path=root/'source_snapshot_manifest.json'
    manifest=load_json(path)
    files={str(root/name):manifest['files_sha256'][name] for name in _author_files(family)}
    if any(file_hash(p)!=sha for p,sha in files.items()): raise ValueError('Author source differs from complete snapshot')
    return {'snapshot_root':str(root),'manifest_path':str(path),'manifest_sha256':file_hash(path),'source_files_sha256':files}


def _verify_source(proof, family):
    root=Path(proof['snapshot_root']); path=root/'source_snapshot_manifest.json'
    if str(path)!=proof['manifest_path'] or file_hash(path)!=proof['manifest_sha256']:
        raise ValueError('Parameterized source snapshot identity differs')
    manifest=load_json(path)
    expected={str(root/name):manifest['files_sha256'][name] for name in _author_files(family)}
    if expected!=proof['source_files_sha256'] or any(file_hash(p)!=sha for p,sha in expected.items()):
        raise ValueError('Parameterized author source identity differs')


def recognizes(path):
    return load_json(Path(path)).get('generator_version') == VERSION


def _module(task_id):
    from scenes.workrooms.initial_randomization.release import adapter
    return adapter(task_id)


def _configuration(scene, task_id):
    return ('dual_same_side' if task_id == 'cooperative_insertion' else
            scene.get('robot_configuration', {}).get('id', 'single_arm'))


def _groups(values, family, task_id):
    scene, task = values['scene'], values['task']
    actors = {name: p['prim_path'] for name, p in values['actual_body_poses_m'].items()}
    md = task.get('asset_metadata', {}); fixtures = md.get('fixtures', {})
    if family == 'kitchen':
        if task_id == 'surface_wiping':
            ids = [('sponge', [*actors.values(), fixtures['sponge_tray']])]
        elif task_id == 'pot_lid_placement':
            ids = [('pot', [actors['pot'], fixtures['source_mat']]), ('lid', [actors['lid']])]
        elif any(name in fixtures for name in ('source_board', 'source_pad', 'source_mat')):
            source = next(fixtures[n] for n in ('source_board', 'source_pad', 'source_mat') if n in fixtures)
            ids = [('source_supply', [*actors.values(), source])]
        else:
            ids = [(name, [path]) for name, path in actors.items()]
    elif family == 'laboratory':
        from scenes.workrooms.laboratory.placement_variants import GROUPS
        all_paths = {**actors, **fixtures}
        if task_id in GROUPS:
            ids = [(name, [all_paths[n] for n in names]) for name, names in GROUPS[task_id].items()]
        elif task_id == 'vial_racking':
            ids = [('source_rack', [*actors.values(), fixtures['supply_cradle']])]
        else:
            ids = [(name, [path]) for name, path in actors.items()]
    elif family == 'study':
        from scenes.workrooms.study.placement_variants import MOVABLE
        if task_id == 'drawer_retrieval':
            from .study_configuration import DRAWER_FIXTURES
            ids=[('drawer_supply',[actors['drawer'],actors['notebook']]+['/World/StudyTask/Fixtures/'+n for n in DRAWER_FIXTURES])]
        else:
            names = [MOVABLE[task_id]] if task_id in MOVABLE else list(actors)
            ids = [(name, [actors[name]]) for name in names]
    else:
        from scenes.workrooms.mechanical_assembly.placement_variants import plan_case
        plan = plan_case(scene, task, 0)
        ids = [(g['id'], [actors[n] for n in g['actor_ids']]+[fixtures[n] for n in g['fixture_ids']])
               for g in plan['groups']]
    groups = [{'id': name, 'paths': paths} for name, paths in ids]
    for group in groups:
        if task_id=='drawer_retrieval' or (task_id == 'breakfast_serving' and group['id'] == 'tray') or (task_id == 'solid_weighing' and 'scoop' in group['id']):
            group['yaw_policy'] = 'fixed_axis_reference'
    return actors, groups


def _numeric_metadata(task, poses, body_physics, layout, parameters=None):
    """Update object-local nominal mass and actual reset poses, never target thresholds."""
    from .usd_authoring import transform_point
    by_path = {v['prim_path']: (name, v) for name, v in poses.items()}
    moves = {path: row for row in layout for path in row['paths']}

    def visit(value, key=None):
        if isinstance(value, dict):
            name = value.get('id', key)
            path = value.get('body_path', value.get('path'))
            if name not in poses and path in by_path:
                name = by_path[path][0]
            if name in poses:
                pose = poses[name]; q = pose['quaternion_wxyz']; xyzw = [*pose['position_m'], *q[1:], q[0]]
                if 'initial_pose_xyzw' in value: value['initial_pose_xyzw'] = xyzw
                if 'initial_position_m' in value: value['initial_position_m'] = deepcopy(pose['position_m'])
                if 'orientation_xyzw' in value: value['orientation_xyzw'] = [*q[1:], q[0]]
                if 'initial_pose' in value: value['initial_pose'] = {k: deepcopy(pose[k]) for k in ('position_m', 'quaternion_wxyz')}
                for field in ('mass_kg', 'mass'):
                    if field in value: value[field] = body_physics[name]['mass_kg']
                for field in ('diagonal_inertia_kg_m2', 'diagonal_inertia'):
                    if field in value and body_physics[name]['diagonal_inertia_kg_m2'] is not None:
                        value[field] = deepcopy(body_physics[name]['diagonal_inertia_kg_m2'])
                physical = value.get('mass_properties')
                if isinstance(physical, dict):
                    physical['mass_kg'] = body_physics[name]['mass_kg']
                    if 'diagonal_inertia_kg_m2' in physical:
                        physical['diagonal_inertia_kg_m2'] = deepcopy(body_physics[name]['diagonal_inertia_kg_m2'])
                    for part, mass in physical.get('component_mass_kg', {}).items():
                        physical['component_mass_kg'][part] = mass*body_physics[name]['mass_factor']
            elif path in moves:
                row = moves[path]['after'][path]; q = row['quaternion_wxyz']
                if 'initial_pose_xyzw' in value: value['initial_pose_xyzw'] = [*row['position_m'], *q[1:], q[0]]
                if 'initial_position_m' in value: value['initial_position_m'] = deepcopy(row['position_m'])
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(task)
    md = task.get('asset_metadata', {}); params = md.get('task_parameters', {})
    def actor_layout(name): return moves.get(poses[name]['prim_path'])
    def point(container, key, row):
        if key in container and row: container[key] = transform_point(container[key], row)
    task_id = task['id']
    if task_id in ('sample_sorting','vial_racking'):
        for label in md.get('visual_detail',{}).get('labels',[]):
            name=label.get('body_id'); row=next((r for r in md['objects'] if r['id']==name),None)
            if row and label['name'].startswith(('Identity_','VialIdentity_')):
                grams=format(row['mass_kg']*1000,'.6g')+' g'
                lines=([name[-2:]+' / '+row['category'].upper(),'SEALED',grams] if task_id=='sample_sorting'
                       else ['V'+name[-2:],grams,'SEALED'])
                label['text']='\n'.join(lines)
    elif task_id in ('cap_placement','solid_weighing'):
        for label in md.get('visual_detail',{}).get('labels',[]):
            name=label['body_id']; grams=(format(body_physics[name]['mass_kg']*1000,'.6g')+' g') if name in body_physics else None
            if task_id=='cap_placement' and label['name'].startswith('Identity_'):
                label['text']='SAMPLE\nB-01\nBODY '+grams
            elif label['name']=='CapIdentity':label['text']='TURN\n'+grams
            elif label['name'].startswith('BoatLabel'):label['text']='BOAT\nTARE '+grams
            elif label['name']=='HandleStamp':label['text']=grams
            elif label['name']=='GranuleLabel':
                label['text']='MODEL GRANULES\n24 X '+format(body_physics['pellet_00']['mass_kg']*1000,'.6g')+' g'
    if task_id == 'cap_placement':
        row = actor_layout('sample_bottle')
        point(params.get('bottle_home', {}), 'center_m', row)
        if row and 'thread' in params:
            p = params['thread']['axis_xy_m']; params['thread']['axis_xy_m'] = transform_point([*p, 0.], row)[:2]
    elif task_id == 'pipette_transfer':
        ret = params.get('tool_return', {})
        if 'initial_pose_xyzw' in ret:
            q = poses['pipette']['quaternion_wxyz']; ret['initial_pose_xyzw'] = [*poses['pipette']['position_m'], *q[1:], q[0]]
    elif task_id == 'tube_clamp_transfer':
        for key, name in [('source_position_m', 'tube'), ('tool_return_position_m', 'clamp_frame')]:
            point(params, key, actor_layout(name))
        for target in md.get('interaction_targets', []):
            if target.get('body_id') in poses:
                point(target, 'reference_position_m', actor_layout(target['body_id']))
    elif task_id == 'vial_racking':
        for hole in params.get('source_holes', []):
            row = actor_layout(hole['object_id']); point(hole, 'center_m', row)
            q = poses[hole['object_id']]['quaternion_wxyz']
            hole['initial_pose_xyzw'] = [*poses[hole['object_id']]['position_m'], *q[1:], q[0]]
    elif task_id == 'solid_weighing':
        pellets = params['pellet_ids']
        params['pellet_mass_kg'] = body_physics[pellets[0]]['mass_kg']
        params['initial_granule_mass_kg'] = sum(body_physics[n]['mass_kg'] for n in pellets)
        row = actor_layout('scoop')
        if row and 'scoop_return_bounds_m' in params:
            # This group keeps yaw zero because the existing return referee uses world-axis bounds.
            from itertools import product
            points = [transform_point(p, row) for p in product(*zip(*params['scoop_return_bounds_m']))]
            params['scoop_return_bounds_m'] = [[min(p[i] for p in points) for i in range(3)],
                                             [max(p[i] for p in points) for i in range(3)]]
    elif task_id == 'breakfast_serving':
        row = actor_layout('tray'); goals = md['goals']
        p = goals['source_center_xy_m']; goals['source_center_xy_m'] = transform_point([*p, 0.], row)[:2]
        q = poses['tray']['quaternion_wxyz']; goals['source_tray_pose_xyzw'] = [*poses['tray']['position_m'], *q[1:], q[0]]
        for p in goals['source_region_xy_m']:
            p[:] = transform_point([*p, 0.], row)[:2]
        goals['nominal_transport_distance_m'] = goals['destination_center_xy_m'][0]-goals['source_center_xy_m'][0]
    elif task_id == 'surface_wiping':
        import math
        row = actor_layout('sponge')
        c, s = math.cos(row['yaw_rad']/2), math.sin(row['yaw_rad']/2)
        for pose in md['goals']['body_poses_xyzw'].values():
            pose[:3] = transform_point(pose[:3], row)
            x, y, z, w = pose[3:]
            pose[3:] = [c*x-s*y, c*y+s*x, c*z+s*w, c*w-s*z]
        composite = md['elastic_composites']['sponge']; factor = body_physics['sponge']['mass_factor']
        for key in ('total_mass_kg', 'foam_mass_kg', 'pad_mass_kg', 'foam_density_kg_m3'):
            composite[key] *= factor
        for link in composite['links']:
            link['reduced_mass_kg'] *= factor
            link['reduced_angular_inertia_kg_m2'] = [x*factor for x in link['reduced_angular_inertia_kg_m2']]
        composite['material_prior']['mass_scaling'] = True
        composite['material_prior']['modal_damping_ratio'] /= math.sqrt(factor)
        composite['material_prior']['retained_drive_coefficients'] = True
        composite['total_mass_kg']=sum(body_physics[name]['mass_kg'] for name in composite['member_ids'])
        composite['pad_mass_kg']=body_physics['sponge']['mass_kg']
        composite['foam_mass_kg']=composite['total_mass_kg']-composite['pad_mass_kg']
        composite['foam_density_kg_m3']=composite['foam_mass_kg']/composite['foam_volume_m3']
        md['composite_objects']['sponge']['total_mass_kg']=composite['total_mass_kg']
        md['physics_prior']['sponge_elastic_lattice']=deepcopy(composite['material_prior'])
    if md and 'metadata' in task.get('evaluation', {}): task['evaluation']['metadata'] = deepcopy(md)
    if md and 'objects' in task.get('evaluation', {}): task['evaluation']['objects'] = deepcopy(md['objects'])
    config = task.get('evaluation', {}).get('configuration')
    if config is not None:
        from scenes.workrooms.study.rules import canonical_hash as config_hash
        if task_id in ('drawer_retrieval','reading_stand'):
            config['parameterized_physics']={'parameters':deepcopy(parameters),'layout':deepcopy(layout),
                'actual_body_poses_m':deepcopy(poses)}
        if task_id == 'drawer_retrieval':
            row=actor_layout('drawer'); mechanism=config['mechanism'];dx=row['translation_xy_m'][0]
            mechanism['closed_position_m']=transform_point(mechanism['closed_position_m'],row)
            for key in ('cabinet_front_x_m','notebook_extraction_max_x_m'):mechanism[key]+=dx
        config['configuration_sha256'] = config_hash({k: v for k, v in config.items() if k != 'configuration_sha256'})


def _refresh_targets(values):
    """Refresh world-space source/return references, retaining local grasp surfaces."""
    import numpy as np
    task=values['task']; md=task.get('asset_metadata',{}); p=md.get('task_parameters',{})
    points={}
    if task['id']=='cap_placement': points['bottle_return']=p['bottle_home']['center_m']
    elif task['id']=='solid_weighing':
        source=list(p['source_tray']['initial_position_m']); source[2]+=p['source_tray']['floor_top_local_z_m']
        points.update(source_granules_tool_access=source,scoop_return=np.mean(p['scoop_return_bounds_m'],axis=0).tolist())
    elif task['id']=='pipette_transfer':
        objects={r['id']:r for r in md['objects']}
        source=list(objects['source']['initial_position_m']);source[2]+=p['hydraulics']['containers']['source']['rim_z_m']
        points.update(source_mouth=source,pipette_cradle_return=p['tool_return']['initial_pose_xyzw'][:3])
    elif task['id']=='tube_clamp_transfer':
        source=list(p['source_position_m']);source[2]+=p['tube_length_m']/2
        points.update(tube_tool_access=source,clamp_return=p['tool_return_position_m'])
    elif task['id']=='surface_wiping': points['sponge_return']=md['goals']['body_poses_xyzw']['sponge'][:3]
    elif task['id']=='drawer_retrieval':
        row=task['evaluation']['configuration']['parameterized_physics']['layout'][0]
        from .usd_authoring import transform_point
        for feature in values['interaction_targets']:
            if feature['label'] in ('notebook_top_after_opening','drawer_handle_opened'):
                points[feature['label']]=transform_point(feature['local_position_m'],row)
    for feature in values['interaction_targets']:
        if feature['label'] in points: feature['local_position_m']=deepcopy(points[feature['label']])


def _evaluator_rules(values, public_mass_text_version=None):
    from .workroom_rules import laboratory_rules, study_rules
    laboratory_rules(values, public_mass_text_version=public_mass_text_version)
    study_rules(values)


def _diff(before, after, prefix=()):
    if before == after: return []
    if isinstance(before, dict) and isinstance(after, dict) and set(before) == set(after):
        return [row for key in before for row in _diff(before[key], after[key], (*prefix, key))]
    return [{'path': list(prefix), 'before': before, 'after': after}]



def _patch_fixtures(current, before, after):
    """Apply one shared task layout while retaining this installation's stand."""
    def partition(rows):
        if (not isinstance(rows,list) or any(not isinstance(row,dict) or
                not isinstance(row.get('id'),str) for row in rows) or
                len({row['id'] for row in rows})!=len(rows)):
            raise ValueError('Fixture patch requires unique explicit fixture ids')
        return ([row for row in rows if 'robot_stand' not in row['id']],
                [row for row in rows if 'robot_stand' in row['id']])
    task_current, _ = partition(current)
    task_before, stands_before = partition(before)
    task_after, stands_after = partition(after)
    if (stands_before != stands_after or task_current != task_before or
            [row['id'] for row in task_after] != [row['id'] for row in task_before]):
        raise ValueError('Fixture patch changed robot stands or mismatched original task fixtures')
    changed = {row['id']:row for row in task_after}
    return [deepcopy(changed.get(row['id'],row)) for row in current]


def _patch(value, changes):
    value = deepcopy(value)
    for change in changes:
        container = value
        for part in change['path'][:-1]: container = container[part]
        key = change['path'][-1]
        if change['path'] == ['fixtures']:
            container[key] = _patch_fixtures(container[key],change['before'],change['after'])
            continue
        if container[key] != change['before']:
            raise ValueError('Another installation has different task inputs: '+str(change['path']))
        container[key] = deepcopy(change['after'])
    return value


def _validate_numeric(values, manifest, data):
    """Recompute numeric task edits and group transforms before admitting a case."""
    import math
    import numpy as np
    from .usd_authoring import group_layout, transform_point
    params = manifest['parameter_instance']; applied = manifest['applied_parameters']
    actors, groups = _groups(values, manifest['family'], manifest['task_id'])
    receipts = applied['layout']
    if len(groups) != len(receipts): raise ValueError('Parameterized layout group inventory differs')
    moved = {}
    for group, row in zip(groups, receipts):
        expected = group_layout(params, group)
        if row['id'] != group['id'] or row['paths'] != group['paths'] or any(row[k] != v for k,v in expected.items()):
            raise ValueError('Parameterized layout differs from its deterministic group stream')
        for path in row['paths']:
            before, after = row['before'][path], row['after'][path]
            if not np.allclose(transform_point(before['position_m'], row), after['position_m'], atol=1e-10, rtol=0):
                raise ValueError('Parameterized transform readback differs')
            w,x,y,z = before['quaternion_wxyz']; c,s = math.cos(row['yaw_rad']/2),math.sin(row['yaw_rad']/2)
            q = np.array([c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w])
            a = np.array(after['quaternion_wxyz'])
            if min(np.max(abs(q-a)),np.max(abs(q+a))) > 1e-7:
                raise ValueError('Parameterized orientation readback differs')
            moved[path] = row
    masses = applied['body_physics']
    if set(masses) != set(actors) or set(data['actual_body_poses_m']) != set(actors):
        raise ValueError('Parameterized rigid actor inventory differs')
    old_mass = {r['object_id']:r['nominal_mass_kg'] for r in values['static_geometry']['bodies']}
    for name,path in actors.items():
        receipt = masses[name]; before = values['actual_body_poses_m'][name]
        expected = moved[path]['after'][path] if path in moved else before
        if data['actual_body_poses_m'][name] != expected:
            raise ValueError('Parameterized actor pose projection differs')
        if path in moved:
            start = moved[path]['before'][path]
            if not np.allclose(start['position_m'],before['position_m'],atol=1e-9,rtol=0):
                raise ValueError('Parameterized group starts from another canonical pose')
        if receipt['prim_path'] != path or receipt['mass_factor'] != params['physics']['mass_factor']:
            raise ValueError('Parameterized body identity/factor differs')
        if not math.isclose(receipt['before']['mass_kg'],old_mass[name],abs_tol=1e-7,rel_tol=1e-6):
            raise ValueError('Parameterized canonical mass differs')
        if not math.isclose(receipt['mass_kg'],receipt['before']['mass_kg']*receipt['mass_factor'],rel_tol=2e-7):
            raise ValueError('Parameterized mass was not scaled exactly once')
        old = receipt['before']['diagonal_inertia_kg_m2']
        if old and min(old)>0 and not np.allclose(receipt['diagonal_inertia_kg_m2'],np.array(old)*receipt['mass_factor'],rtol=2e-7,atol=1e-15):
            raise ValueError('Parameterized explicit inertia differs')
    expected_values = deepcopy(values); expected_task = expected_values['task']
    if params['seed']:
        _numeric_metadata(expected_task,data['actual_body_poses_m'],masses,receipts,params)
        _refresh_targets(expected_values)
    from .workroom_rules import checked_public_mass_text_version
    _evaluator_rules(expected_values, checked_public_mass_text_version(manifest))
    if _patch(values['task'],data['task_changes']) != expected_task:
        raise ValueError('Parameterized task changes exceed the declared physical recipe')
    expected_geometry = deepcopy(values['static_geometry'])
    for row in expected_geometry['bodies']: row['nominal_mass_kg'] = masses[row['object_id']]['mass_kg']
    if data['static_geometry'] != expected_geometry:
        raise ValueError('Parameterized nominal geometry differs')
    if data['interaction_targets'] != expected_values['interaction_targets']:
        raise ValueError('Parameterized interaction references differ')


def prepare_asset(parent_release, seed, output, *, profile=VERSION, attempt=0, layout_amplitude_scale=1.,
                  settled_state=None):
    if profile != VERSION: raise ValueError('Unknown physical parameter profile')
    from scenes.workrooms.initial_randomization import TASK_FAMILIES
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    from . import usd_authoring
    source = verify_if_snapshot(Path(__file__).resolve().parents[3], required_modules=(__name__, usd_authoring.__name__), entrypoint_file=Path(__file__))
    if not source or not source.get('verified'): raise ValueError('Parameterized author requires a complete frozen source')
    parent = Path(parent_release).resolve(); output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Never overwrite parameterized candidates')
    task = load_json(parent/'task.json'); scene = load_json(parent/'scene.json')
    task_id = task['id']; family = TASK_FAMILIES[task_id]; module = _module(task_id)
    from .workroom_rules import PUBLIC_MASS_TEXT_VERSION
    text_version = ({'public_mass_text_version': PUBLIC_MASS_TEXT_VERSION}
                    if task_id == 'solid_weighing' else {})
    cfg = _configuration(scene, task_id)
    settled = None
    if settled_state is not None:
        # Cable/granular only: a recorded robot-free settled state replaces the dynamic reset poses.
        from .settle_admission import checked_settled_state
        settled = checked_settled_state(settled_state, task_id=task_id, seed=seed, attempt=attempt,
                                        layout_amplitude_scale=layout_amplitude_scale)
    params = sample_parameters(family, task_id, cfg, seed, attempt, layout_amplitude_scale=layout_amplitude_scale,
                               settled_initial_state_sha256=None if settled is None else settled_state['sha256'])
    output.mkdir(parents=True)
    try:
        canonical_dir = output/'canonical'
        kwargs = {'profile': module.VERSION} if module.__name__ == 'scenes.workrooms.kitchen.seed_variants' else {}
        canonical = module.prepare_asset(parent, 0, canonical_dir, **kwargs)
        values = module.compose_case(parent, canonical, cfg)
        original_scene, original_task = deepcopy(values['scene']), deepcopy(values['task'])
        if scene.get('native_task_bundle'):
            bundle = values['scene']['native_task_bundle']; usd = bundle['usd_path']; root = bundle['root_prim_path']; kind = 'bundle'
        elif family == 'study':
            layer = next(r for r in values['scene']['task_layers'] if r['id'] == 'task')
            usd = layer['usd_path']; root = '/World/StudyTask'; kind = 'study'
        else:
            kind = 'catalog'
        if kind == 'catalog':
            if settled is not None:
                raise ValueError('A settled initial state is only supported for USD task stages')
            from .mechanical_authoring import author_catalog
            values = author_catalog(values, params, output/'catalog')
            applied = values.pop('applied')
        else:
            from pxr import Usd
            stage = Usd.Stage.Open(Usd.Stage.Open(usd).Flatten())
            actors, groups = _groups(values, family, task_id)
            layout = usd_authoring.apply_layout(stage, groups, params)
            physics = usd_authoring.apply_physics(stage, actors, root, params)
            appearance = usd_authoring.apply_appearance(stage, actors, root, params, output)
            settled_receipts = (None if settled is None else
                                usd_authoring.apply_settled_poses(stage, actors, settled['body_poses']))
            poses = {name: usd_authoring.world_pose(stage, path) for name, path in actors.items()}
            if seed:
                _numeric_metadata(values['task'], poses, physics['body_physics'], layout,params)
                _refresh_targets(values)
                if task_id in ('sample_sorting','vial_racking'):
                    from scenes.workrooms.laboratory.seed_variants import _label_masses
                    _label_masses(stage, values['task']['asset_metadata'])
                    values['task']['evaluation']['metadata'] = deepcopy(values['task']['asset_metadata'])
                elif task_id in ('cap_placement','solid_weighing'):
                    from scenes.workrooms.laboratory.visual_details import DetailAuthor
                    md=values['task']['asset_metadata']; detail=DetailAuthor(stage,md)
                    for label in md['visual_detail']['labels']:
                        if label['name'].startswith(('Identity_','CapIdentity','BoatLabel','HandleStamp','GranuleLabel')):
                            detail.text(label['body_id'],label['name'],label['text'],label['local_center_m'],label['size_m'],
                                facing=label['facing'],background=label['name']!='HandleStamp')
                target = output/'task.usda'; stage.GetRootLayer().Export(str(target))
                if kind == 'bundle':
                    values['scene']['native_task_bundle'].update(usd_path=str(target), usd_sha256=file_hash(target))
                    if 'elastic_composites' in values['scene']['native_task_bundle']:
                        values['scene']['native_task_bundle']['elastic_composites'] = deepcopy(values['task']['asset_metadata']['elastic_composites'])
                else:
                    layer = next(r for r in values['scene']['task_layers'] if r['id'] == 'task')
                    layer.update(usd_path=str(target), sha256=file_hash(target))
                    layer['dependencies_sha256'] = usd_authoring.task_layer_dependencies(layer)
                    usd_authoring.check_task_layer_admission(layer)
                    for name, digest in layer['dependencies_sha256'].items():
                        previous = values['asset_files_sha256'].get(name, digest)
                        if previous != digest:
                            raise ValueError('Authored task layer would replace a frozen dependency pin: '+name)
                        values['asset_files_sha256'][name] = digest
            values['actual_body_poses_m'] = poses
            for row in values['static_geometry']['bodies']:
                row['nominal_mass_kg'] = physics['body_physics'][row['object_id']]['mass_kg']
            # Body-local interaction surfaces remain exact after a rigid group transform.
            applied = {'layout': layout, **physics, 'appearance': appearance,
                       'geometry_dimensions_changed': False, 'native_readback_verified': False}
            if settled is not None:
                applied['settled_initial_state'] = {'record': settled_state, 'receipts': settled_receipts,
                    'scope': 'Initial state = recorded robot-free passive settled state; not authored layout poses.'}
        _evaluator_rules(values, text_version.get('public_mass_text_version'))
        if seed:
            # Robot-free physical reset admission; a rejection records its measurements.
            from .reset_admission import check_reset_admission
            write_json(output/'reset_admission.json',
                       check_reset_admission(values, applied['layout'], family=family, seed=seed))
        data = {'scene_changes': _diff(original_scene, values['scene']), 'task_changes': _diff(original_task, values['task']),
                **{key: deepcopy(values.get(key, [])) for key in PROJECTIONS}}
        write_json(output/'metadata.json', data)
        baseline = load_json(Path(canonical))
        author_source = _source_identity(source, family)
        deps = {**values['asset_files_sha256'], str(canonical): file_hash(canonical)}
        deps.update(author_source['source_files_sha256'])
        deps[author_source['manifest_path']]=author_source['manifest_sha256']
        deps.update({str(p): file_hash(p) for p in output.rglob('*') if p.is_file()})
        manifest = {'schema': 'workroom_seed_asset.v1', 'family': family, 'task_id': task_id, 'seed': seed,
            'generator_version': VERSION, 'variant_scope': VERSION, 'parameter_instance': params,
            **text_version,
            'applied_parameters': applied, 'canonical_asset': {'path': str(canonical), 'sha256': file_hash(canonical)},
            'parent_references': baseline['parent_references'], 'dependencies_sha256': deps,
            'metadata': str(output/'metadata.json'), 'metadata_sha256': file_hash(output/'metadata.json'),
            'source_snapshot': source, 'author_source': author_source, 'native_qualified': False, 'runtime_object_writes': False,
            **{key: deepcopy(data[key]) for key in PROJECTIONS}}
        if 'layout_amplitude_scale' in params:
            # Reduced-amplitude fallback; absent means the unchanged full amplitude.
            manifest['layout_amplitude_scale'] = params['layout_amplitude_scale']
        if settled is not None:
            manifest['settled_initial_state'] = {'record': settled_state, 'source': deepcopy(settled['source']),
                'hold_s': settled['hold_s'], 'body_count': len(settled['body_poses']),
                'scope': 'The published initial state is the recorded robot-free passive settled state of the source '
                         'candidate (cable/granular tasks only), explicitly not the authored layout poses.'}
        manifest['physical_instance_sha256'] = canonical_hash({'parameters': params, 'applied': applied,
                    'projections': {key: data[key] for key in PROJECTIONS}, 'dependencies': deps, **text_version})
        write_json(output/'seed_manifest.json', manifest)
        return output/'seed_manifest.json'
    except Exception as exc:
        write_json(output/'rejection.json', rejection_record(params, [type(exc).__name__+': '+str(exc)],
                                                             measurements=getattr(exc, 'measurements', None)))
        raise


def compose_case(parent_release, asset_manifest_path, configuration_id):
    path = Path(asset_manifest_path).resolve(); manifest = load_json(path)
    if manifest.get('generator_version') != VERSION or manifest.get('variant_scope') != VERSION:
        raise ValueError('Wrong parameterized asset identity')
    from .workroom_rules import checked_public_mass_text_version
    version = checked_public_mass_text_version(manifest)
    text_version = {} if version is None else {'public_mass_text_version': version}
    params = manifest['parameter_instance']; validate_parameters(params)
    _verify_source(manifest['author_source'], manifest['family'])
    if params['task_id'] != manifest['task_id'] or params['seed'] != manifest['seed'] or params['family'] != manifest['family']:
        raise ValueError('Parameterized candidate identity differs')
    for p, expected in {**manifest['parent_references'], **manifest['dependencies_sha256']}.items():
        if not undistributed_bytecode(p) and file_hash(p) != expected:
            raise ValueError('Changed frozen parameterized dependency '+p)
    canonical = manifest['canonical_asset']
    if file_hash(canonical['path']) != canonical['sha256']: raise ValueError('Canonical asset changed')
    values = _module(manifest['task_id']).compose_case(parent_release, canonical['path'], configuration_id)
    data = load_json(Path(manifest['metadata']))
    if file_hash(manifest['metadata']) != manifest['metadata_sha256']:
        raise ValueError('Parameterized metadata identity differs')
    from .study_appearance_successor import validate_successor
    validate_successor(manifest, data)
    expected = canonical_hash({'parameters': params, 'applied': manifest['applied_parameters'],
        'projections': {key: data[key] for key in PROJECTIONS}, 'dependencies': manifest['dependencies_sha256'], **text_version})
    if expected != manifest['physical_instance_sha256'] or any(data[k] != manifest[k] for k in PROJECTIONS):
        raise ValueError('Parameterized frozen projections disagree')
    if manifest['family'] == 'mechanical_assembly' and not values['scene'].get('native_task_bundle'):
        from .mechanical_authoring import validate_catalog
        validate_catalog(values, params, manifest['applied_parameters'], data)
    else:
        _validate_numeric(values,manifest,data)
    values['scene'] = _patch(values['scene'], data['scene_changes'])
    values['task'] = _patch(values['task'], data['task_changes'])
    _evaluator_rules(values, version)
    values.update({key: deepcopy(data[key]) for key in PROJECTIONS})
    values['parameterized_physics'] = {'parameters': deepcopy(params), 'applied': deepcopy(manifest['applied_parameters']),
                                     'physical_instance_sha256': expected, **text_version}
    values['asset_files_sha256'].update(manifest['dependencies_sha256'])
    values['asset_files_sha256'][str(path)] = file_hash(path)
    return values


def prepare_task_layer_dependency_successor(parent_release, asset_manifest_path, output, *, expected_sha256):
    """Publish only missing study dependency declarations; reuse all asset bytes.

    The prior asset remains intact. This does not resample any parameter or
    export USD/PNG files, and does not inherit its robot/native qualification.
    """
    from . import usd_authoring
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(Path(__file__).resolve().parents[3],
        required_modules=(__name__, usd_authoring.__name__), entrypoint_file=Path(__file__))
    if not source or not source.get('runtime_imports_verified'):
        raise ValueError('Binding successor requires a complete frozen author source')
    prior_path = Path(asset_manifest_path).resolve()
    if file_hash(prior_path) != expected_sha256:
        raise ValueError('Previous study asset differs from explicit pin')
    prior = load_json(prior_path)
    if prior.get('family') != 'study' or prior.get('parameter_instance', {}).get('canonical') is not False:
        raise ValueError('Only noncanonical parameterized study assets need this successor')
    parent = Path(parent_release).resolve()
    cfg = _configuration(load_json(parent/'scene.json'), prior['task_id'])
    values = compose_case(parent, prior_path, cfg)
    before_scene = deepcopy(values['scene'])
    layer = next(r for r in values['scene']['task_layers'] if r['id'] == 'task')
    old_pins = deepcopy(layer.get('dependencies_sha256', {}))
    layer['dependencies_sha256'] = usd_authoring.task_layer_dependencies(layer)
    if layer['dependencies_sha256'] == old_pins:
        raise ValueError('Prior task layer already pins every actual dependency')
    for name, digest in layer['dependencies_sha256'].items():
        if prior['dependencies_sha256'].get(name, digest) != digest:
            raise ValueError('Dependency successor would replace a previous global pin: '+name)
    admission = usd_authoring.check_task_layer_admission(layer)
    before_layer = next(r for r in before_scene['task_layers'] if r['id'] == 'task')
    expected_scene = deepcopy(before_scene)
    next(r for r in expected_scene['task_layers'] if r['id'] == 'task')['dependencies_sha256'] = deepcopy(layer['dependencies_sha256'])
    if values['scene'] != expected_scene or any(layer['dependencies_sha256'][p] != h for p,h in old_pins.items()):
        raise ValueError('Binding successor changed more than additional dependency pins')
    canonical = _module(prior['task_id']).compose_case(parent, prior['canonical_asset']['path'], cfg)
    data = load_json(Path(prior['metadata']))
    data['scene_changes'] = _diff(canonical['scene'], values['scene'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'metadata.json', data)
    author_source = _source_identity(source, prior['family'])
    deps = {**prior['dependencies_sha256'], **layer['dependencies_sha256'], str(prior_path): expected_sha256,
        **author_source['source_files_sha256'], author_source['manifest_path']: author_source['manifest_sha256'],
        str(output/'metadata.json'): file_hash(output/'metadata.json')}
    successor = deepcopy(prior)
    successor.update(metadata=str(output/'metadata.json'), metadata_sha256=file_hash(output/'metadata.json'),
        dependencies_sha256=deps, author_source=author_source, source_snapshot=source, native_qualified=False,
        dependency_binding_successor={'schema': 'study_task_layer_dependency_successor.v1',
            'previous_asset': {'path': str(prior_path), 'sha256': expected_sha256},
            'added_dependencies': {p:h for p,h in layer['dependencies_sha256'].items() if p not in old_pins},
            'usd': {'path': before_layer['usd_path'], 'sha256': before_layer['sha256']},
            'all_usd_and_image_files_reused': True, 'layout_and_physics_unchanged': True,
            'cpu_layer_admission': admission, 'native_qualified': False})
    text_version = ({'public_mass_text_version': prior['public_mass_text_version']}
                    if 'public_mass_text_version' in prior else {})
    successor['physical_instance_sha256'] = canonical_hash({'parameters': prior['parameter_instance'],
        'applied': prior['applied_parameters'], 'projections': {key:data[key] for key in PROJECTIONS},
        'dependencies': deps, **text_version})
    write_json(output/'seed_manifest.json', successor)
    if file_hash(prior_path) != expected_sha256:
        raise ValueError('Prior asset changed during successor publication')
    for name, digest in prior['dependencies_sha256'].items():
        if file_hash(name) != digest:
            raise ValueError('Previous asset dependency changed during successor publication: '+name)
    # Exercise the ordinary reader before returning a new reference.
    composed = compose_case(parent, output/'seed_manifest.json', cfg)
    if composed['scene'] != values['scene'] or any(composed[key] != values[key]
            for key in ('task', 'initial_state', *PROJECTIONS)):
        raise ValueError('Dependency successor changed task/initial/physical projections')
    return output/'seed_manifest.json'
