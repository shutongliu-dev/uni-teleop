"""Immutable paired vial/rack and bottle/tray physical reset variants.

Seed zero references original bytes. Nonzero cases change only task props;
room, robot, control roles, collider decomposition and scoring limits persist.
"""
from copy import deepcopy
import json
from pathlib import Path
import types

import numpy as np

from bench.paths import load_json, undistributed_bytecode
from scenes.workrooms.seed_geometry import (apply_surface_texture, bake_body_scale,
    canonical_hash, check_parent_release, checked_seed, collision_bounds, feature, file_hash,
    finish_texture, mass_inventory, planar_offset, pose, transform_body, write_json)

TASKS = ('sample_sorting', 'vial_racking')
VERSION = 'laboratory_seed_variants.v1'
MARKER = 'laboratory_seed_geometry.v1'


def _read(path):
    return load_json(Path(path))


def _check(path, expected):
    if undistributed_bytecode(path):
        return  # pinned bytecode cache, not shipped
    if file_hash(path) != expected:
        raise ValueError('Source asset identity differs: '+str(path))


def _label_masses(stage, metadata):
    # Re-author only original identity labels, before scaling their geometry.
    from .visual_details import DetailAuthor
    author = DetailAuthor(stage, metadata)
    for row in metadata['objects']:
        name = row['id']; grams = format(row['mass_kg']*1000, '.6g')+' g'
        for facing in ('front','rear'):
            if metadata['task_id']=='sample_sorting':
                author.text(name,'Identity_'+facing,
                    [name[-2:]+' / '+row['category'].upper(),'SEALED',grams],
                    (0.,0.,.026),(.026,.028),facing=facing,radius=.01316)
            else:
                author.text(name,'VialIdentity_'+facing,['V'+name[-2:],grams,'SEALED'],
                    (0.,0.,.047),(.018,.029),facing=facing,radius=.00916)
    replacement = {(r['body_id'],r['name']):r for r in author.labels}
    metadata['visual_detail']['labels'] = [replacement.get((r['body_id'],r['name']),r)
        for r in metadata['visual_detail']['labels']]


def _point(value, pivot, delta, scale):
    out=np.asarray(value,dtype=float).copy()
    out[:2]=(out[:2]-np.asarray(pivot)[:2])*scale+np.asarray(pivot)[:2]+delta
    return out.tolist()


def derive_stage(stage, original_metadata, seed):
    """Caller-owned CPU stage. Returns matching nominal reset metadata."""
    checked_seed(seed); metadata=deepcopy(original_metadata)
    task_id=metadata['task_id']
    if task_id not in TASKS: raise ValueError('Only laboratory Task1/Task2 are supported')
    if seed == 0:
        return metadata
    factor=1. if seed==0 else (.96 if seed%2 else 1.04)
    if seed:
        for row in metadata['objects']: row['mass_kg']*=factor**2
        _label_masses(stage,metadata)
        for path in [*metadata['actors'].values(),*metadata['fixtures'].values()]:
            bake_body_scale(stage,path,[factor,factor,1.])
    fixture_moves={}
    for name,path in metadata['fixtures'].items():
        pivot=original_metadata['fixture_metadata'][name]['initial_position_m']
        delta,_=planar_offset(seed,task_id+':'+name,yaw_deg=0.)
        fixture_moves[name]=(pivot,delta)
        if seed: transform_body(stage,path,delta,0.)
    for row in metadata['objects']:
        name=row['id'];path=row['body_path'];old=np.array(row['initial_position_m'])
        delta,yaw=planar_offset(seed,task_id+':'+name)
        if task_id=='vial_racking':
            pivot,group_delta=fixture_moves['supply_cradle']
            delta=(np.asarray(_point(old,pivot,group_delta,factor))[:2]-old[:2]).tolist()
        if seed: transform_body(stage,path,delta,yaw)
    p=metadata['task_parameters']
    if task_id=='sample_sorting':
        for target in p['targets'].values():
            pivot,delta=fixture_moves[target['fixture']]
            target['center_m']=_point(target['center_m'],pivot,delta,factor)
            target['goal_pose_xyzw'][:3]=_point(target['goal_pose_xyzw'][:3],pivot,delta,factor)
            target['half_extents_m'][:2]=[v*factor for v in target['half_extents_m'][:2]]
    else:
        for collection in ('holes','source_holes'):
            for hole in p[collection]:
                pivot,delta=fixture_moves[hole['fixture']]
                hole['center_m']=_point(hole['center_m'],pivot,delta,factor)
                hole['radius_m']*=factor
                key='goal_pose_xyzw' if collection=='holes' else 'initial_pose_xyzw'
                hole[key][:3]=_point(hole[key][:3],pivot,delta,factor)
        p['radial_clearance_m']*=factor
        p['source_rack']['base_size_m'][:2]=[v*factor for v in p['source_rack']['base_size_m'][:2]]
        p['source_rack']['number_plate_local_yz_m'][0]*=factor
    mass=mass_inventory(stage,metadata['actors'])
    for row in [*metadata['objects'],*metadata['fixture_metadata'].values()]:
        identity=pose(stage.GetPrimAtPath(row['body_path']));q=identity['quaternion_wxyz'];position=identity['position_m']
        if seed:
            row.update(initial_position_m=position,initial_pose_xyzw=[*position,*q[1:],q[0]],orientation_xyzw=[*q[1:],q[0]],
                local_bounds_m=collision_bounds(stage,row['body_path']))
            row['half_extents_m']=(np.ptp(row['local_bounds_m'],axis=0)/2).tolist()
            if row['id'] in mass:
                # Nominal density-scaled mass remains decimal; native float32 readback is separate.
                row.update({k:v for k,v in mass[row['id']].items() if k!='mass_kg'})
    if task_id=='vial_racking':
        for hole in p['source_holes']:
            hole['initial_pose_xyzw']=next(r['initial_pose_xyzw'] for r in metadata['objects'] if r['id']==hole['object_id'])
    if seed:
        metadata['geometry_variant']={'schema':MARKER,'seed':seed,'scale_xyz':[factor,factor,1.],
            'mass_policy':'Density-preserving affine transform of original explicit mass/COM/inertia',
            'paired_clearance_preserved':True,'z_dimensions_unchanged':True,
            'physical_materials_unchanged':True,'native_qualified':False}
    return metadata


def _observables(stage, metadata):
    poses={name:pose(stage.GetPrimAtPath(path)) for name,path in metadata['actors'].items()}
    static={'bodies':[]};targets=[]
    for row in metadata['objects']:
        name,path=row['id'],row['body_path'];bounds=collision_bounds(stage,path)
        static['bodies'].append({'object_id':name,'local_bounds_m':bounds,'nominal_mass_kg':row['mass_kg']})
        z=.055 if metadata['task_id']=='vial_racking' else .030
        for sign in (-1,1):
            targets.append(feature(name+'_side_'+str(sign),name,path,[0.,sign*abs(bounds[sign>0][1]),z],
                group=name+'_pickup',direction=(0.,float(sign),0.)))
    p=metadata['task_parameters']
    goals=list(p['targets'].values()) if metadata['task_id']=='sample_sorting' else p['holes']
    for index,g in enumerate(goals):
        fixture=g['fixture'];name=(list(p['targets'])[index] if 'targets'in p else g['object_id'])
        origin=metadata['fixture_metadata'][fixture]['initial_position_m']
        point=np.asarray(g['goal_pose_xyzw'][:3])-origin
        point[2]=g.get('rim_z_m',p.get('support_floor_z_m'))-origin[2]
        targets.append(feature(name+'_place',name,metadata['fixtures'][fixture],point.tolist(),kind='goal'))
    return poses,static,targets


def prepare_asset(parent_release:Path,seed:int,output:Path, *, profile=None, attempt=0)->Path:
    if profile == "parameterized_physics_v1":
        from scenes.workrooms.parameterized.workroom import prepare_asset as prepare_parameterized
        return prepare_parameterized(parent_release, seed, output, profile=profile, attempt=attempt)
    from pxr import Usd
    checked_seed(seed); parent_release=Path(parent_release).resolve();output=Path(output).resolve()
    check_parent_release(parent_release)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Never overwrite asset candidates: '+str(output))
    scene=_read(parent_release/'scene.json');task=_read(parent_release/'task.json')
    if task['id'] not in TASKS:raise ValueError('Wrong laboratory task')
    source=scene['laboratory_task_provenance']['asset_candidate'];refs={}
    for key in ('manifest','usd','metadata'):
        _check(source[key+'_path'],source[key+'_sha256']);refs[source[key+'_path']]=source[key+'_sha256']
    for name,sha in {**source.get('dependencies',{}),**source.get('auxiliary_files_sha256',{})}.items():
        _check(name,sha);refs[name]=sha
    for file in parent_release.rglob('*'):
        if file.is_file():refs[str(file)]=file_hash(file)
    output.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.Open(Usd.Stage.Open(source['usd_path']).Flatten())
    metadata=derive_stage(stage,task['asset_metadata'],seed)
    appearance={'variant':'canonical','physical_materials_changed':False}
    if seed:
        texture=finish_texture(output,seed)
        appearance=apply_surface_texture(stage,[*metadata['actors'].values(),*metadata['fixtures'].values()],texture,seed,
            material_root=metadata['root_path']+'/SeedVariantMaterials')
        usd=output/'task.usda';stage.GetRootLayer().Export(str(usd))
    else:usd=Path(source['usd_path'])
    poses,static,targets=_observables(stage,metadata)
    meta=output/'metadata.json';write_json(meta,metadata)
    (output/'ATTRIBUTION.txt').write_text('Original laboratory procedural assets and derived geometry/finish: CC0-1.0.\nParent source/license files are hash-bound in parent_references.\nNo robot/control or physical material changes.\n')
    deps={str(p):file_hash(p) for p in output.rglob('*') if p.is_file()}
    deps[str(usd)]=file_hash(usd)
    result={'schema':'workroom_seed_asset.v1','family':'laboratory','task_id':task['id'],'seed':seed,
        'generator_version':VERSION,'parent_references':refs,'dependencies_sha256':deps,
        'license':'CC0-1.0','initial_usd':str(usd),'initial_usd_sha256':file_hash(usd),
        'metadata':str(meta),'metadata_sha256':file_hash(meta),'actual_body_poses_m':poses,
        'static_geometry':static,'interaction_targets':targets,'visual_variant':appearance,
        'fixed_support_paths':list(metadata['fixtures'].values()),'native_qualified':False}
    path=output/'seed_manifest.json';write_json(path,result);return path


def _replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unsupported frozen evaluator source pattern: '+old)
    return text.replace(old,new)


def _sources(parent):
    sources={str(p.relative_to(parent/'evaluator')):p.read_text() for p in (parent/'evaluator').rglob('*.py')}
    rules=sources['rules.py']
    rules=_replace_once(rules,'def object_properties(task_id):','def object_properties(task_id, metadata=None):')
    marker="    basis = ("
    injection="""    if metadata is not None and metadata.get('geometry_variant', {}).get('schema') == 'laboratory_seed_geometry.v1':
        if task_id not in ('sample_sorting', 'vial_racking') or metadata.get('task_id') != task_id:
            raise ValueError('Physical seed nominal masses require a matching supported task')
        objects = {row['id']: row for row in metadata['objects']}
        if set(objects) != {row[0] for row in rows} or set(objects) != set(metadata['actors']):
            raise ValueError('Physical seed nominal mass inventory differs')
        if any(type(row['mass_kg']) not in (int, float) or not math.isfinite(row['mass_kg']) or row['mass_kg'] <= 0 for row in objects.values()):
            raise ValueError('Physical seed nominal masses must be finite and positive')
        rows = [(name, label, objects[name]['mass_kg']) for name, label, _ in rows]
"""
    rules=_replace_once(rules,marker,injection+marker)
    rules=_replace_once(rules,'def public_rules(task_id, seed=0):','def public_rules(task_id, seed=0, metadata=None):')
    rules=_replace_once(rules,"'object_properties': object_properties(task_id)","'object_properties': object_properties(task_id, metadata)")
    rules=_replace_once(rules,'rules = public_rules(task_id, seed)','rules = public_rules(task_id, seed, metadata)')
    sources['rules.py']=rules
    evaluator=sources['sequence_evaluator.py']
    evaluator=_replace_once(evaluator,"public_rules(task['id'], task['episode_seed'])","public_rules(task['id'], task['episode_seed'], c['metadata'])")
    evaluator=_replace_once(evaluator,"public_rules(self.c['task_id'], self.c['episode_seed'])","public_rules(self.c['task_id'], self.c['episode_seed'], self.c['metadata'])")
    sources['sequence_evaluator.py']=evaluator
    return sources


def compose_case(parent_release,asset_manifest_path,configuration_id):
    from scenes.workrooms.parameterized.workroom import recognizes, compose_case as compose_parameterized
    if recognizes(asset_manifest_path):
        return compose_parameterized(parent_release, asset_manifest_path, configuration_id)
    parent=Path(parent_release).resolve();manifest=_read(asset_manifest_path)
    check_parent_release(parent)
    if manifest['schema']!='workroom_seed_asset.v1' or manifest['family']!='laboratory':raise ValueError('Wrong seed asset family')
    for path,sha in {**manifest['parent_references'],**manifest['dependencies_sha256']}.items():_check(path,sha)
    if manifest['parent_references'].get(str(parent/'release_manifest.json'))!=file_hash(parent/'release_manifest.json'):
        # Geometry may be shared across parent robot configurations, but canonical asset identities must match.
        candidate=_read(parent/'scene.json')['laboratory_task_provenance']['asset_candidate']
        if manifest['parent_references'].get(candidate['usd_path'])!=candidate['usd_sha256']:raise ValueError('Parent task asset differs')
    scene=_read(parent/'scene.json');task=_read(parent/'task.json')
    if task['id']!=manifest['task_id']:raise ValueError('Task differs from seed asset')
    metadata=_read(manifest['metadata']);sources=_sources(parent)
    rules=types.ModuleType('_seed_laboratory_rules');exec(compile(sources['rules.py'],'<frozen-laboratory-rules>','exec'),rules.__dict__)
    task['episode_seed']=manifest['seed'];task['asset_metadata']=metadata
    task['public_rules']=rules.public_rules(task['id'],manifest['seed'],metadata)
    task['evaluation']=rules.evaluation_configuration(task['id'],metadata,manifest['seed'],
        task['evaluation']['physics_dt_s'],task.get('room_support_paths',{}))
    rules.bind_task_rules(task)
    bundle=scene['native_task_bundle'];bundle['usd_path']=manifest['initial_usd'];bundle['usd_sha256']=manifest['initial_usd_sha256']
    bundle['observer']['configuration']=deepcopy(task)
    files={k:__import__('hashlib').sha256(v.encode()).hexdigest() for k,v in sources.items()}
    bundle['observer']['files']=files;bundle['observer']['sha256']=files['sequence_evaluator.py']
    provenance=scene['laboratory_task_provenance']
    provenance['parent_asset_candidate']=deepcopy(provenance['asset_candidate'])
    provenance['asset_candidate']={'manifest_path':str(Path(asset_manifest_path).resolve()),
        'manifest_sha256':file_hash(asset_manifest_path),'usd_path':manifest['initial_usd'],
        'usd_sha256':manifest['initial_usd_sha256'],'metadata_path':manifest['metadata'],
        'metadata_sha256':manifest['metadata_sha256'],'root_prim_path':metadata['root_path'],
        'dependencies':deepcopy(manifest['dependencies_sha256'])}
    provenance['physical_reset_seeded']=True
    provenance['seed_scope']='Explicit immutable seed 0..9; zero canonical, bounded paired XY geometry and reset placement for others'
    return {'scene':scene,'task':task,'initial_state':deepcopy(task['initial_state']),
        'evaluator_sources':sources,'actual_body_poses_m':manifest['actual_body_poses_m'],
        'interaction_targets':manifest['interaction_targets'],'static_geometry':manifest['static_geometry'],
        'fixed_support_paths':manifest['fixed_support_paths'],
        'asset_files_sha256':{**manifest['parent_references'],**manifest['dependencies_sha256'],str(Path(asset_manifest_path).resolve()):file_hash(asset_manifest_path)}}
