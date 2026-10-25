"""Freeze task-layer candidates separately from native task-bundle releases."""
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil

from bench.isaacsim.release import checked, digest, dump
from .composition import TASK_IDS, AUTHORING_TASK_IDS, build_task_inputs, stand_usda
from .scene import environment_reference
from bench.paths import DATA_ROOT, load_json

SCHEMA='living_room_task_release.v1'
VERSION='isaacsim-living-room-release-v1'
DATASETS=Path(f'{DATA_ROOT}/datasets')


def asset_inventory(scene):
    from bench.simulation.isaacsim_backend.workroom import verify_reference
    from .dual_robot_layout import robot_asset_reference
    reference=scene['workroom'];verify_reference(reference)
    hashes={reference['usd_path']:reference['usd_sha256'],reference['dependency_manifest_path']:reference['dependency_manifest_sha256']}
    description=load_json(Path(reference['dependency_manifest_path']))
    hashes.update({p:r['sha256'] for p,r in description['dependencies'].items()})
    robot=robot_asset_reference()
    hashes.update({robot['root_usd']:robot['root_usd_sha256'],robot['source_manifest']:robot['source_manifest_sha256'],
        robot['import_manifest']:robot['import_manifest_sha256'],**robot['derived_files']})
    for layer in scene['task_layers']:
        hashes[layer['usd_path']]=layer['sha256'];hashes.update(layer['dependencies_sha256'])
    for p,sha in hashes.items():checked(p,sha)
    return {p:{'sha256':sha} for p,sha in sorted(hashes.items())}


def initial_state(scene,task):
    from bench.control.factory import public_robot_geometry,robot_configuration
    if (task.get('schema_version')!='living-room-task-v1' or task.get('id') not in TASK_IDS
        or task.get('task_id')!=task['id'] or scene.get('workroom')!=environment_reference()
        or scene.get('scenario',{}).get('task_id')!=task['id'] or scene.get('cloth',{}).get('enabled') is not False):
        raise ValueError('Living task identity or parent environment mismatch')
    from bench.simulation.isaacsim_backend.robot_registry import get_robot_spec
    import numpy as np
    spec=get_robot_spec(scene['robot'])
    for key in ('mount_position_m','mount_quaternion_wxyz'):
        if key in scene['robot'] and not np.allclose(scene['robot'][key],spec['mount_translation_m' if key=='mount_position_m' else key],rtol=0,atol=1.e-9):
            raise ValueError('Robot composite/source mount mismatch: '+key)
    model=public_robot_geometry(robot_configuration(scene));initial=deepcopy(task['initial_state'])
    if set(initial)!={'joint_positions_rad'} or set(initial['joint_positions_rad'])!=set(model.names):
        raise ValueError('Complete robot-only native reset required')

    if hasattr(model, 'sides'):
        for side, member in model.sides.items():
            member.checked_joints({name: initial['joint_positions_rad'][name] for name in member.names}, enforce_limits=True)
    else:
        model.checked_joints(initial['joint_positions_rad'], enforce_limits=True)
    # Reject explicit camera mounts before starting native physics. The backend
    # still treats the frozen declaration literally; no alias silently retargets it.
    from bench.simulation.isaacsim_backend.cameras import resolve_cameras
    links = set()
    for member in getattr(model, 'sides', {'single': model}).values():
        links.update(member.arm.forward(initial['joint_positions_rad']))
        links.update(member.hand.forward(initial['joint_positions_rad']))
    for camera in resolve_cameras(scene).values():
        if camera['kind'] == 'link' and camera['mount_link'] not in links:
            raise ValueError('Camera mounts on an unknown declared robot link: '+camera['mount_link'])
    from .composition import task_body_paths
    expected_bodies = task_body_paths(task['id'], task['evaluation']['candidate_metadata'])
    if (scene['task_layers'][0]['rigid_bodies'] != expected_bodies
            or task['evaluation']['body_paths'] != expected_bodies):
        raise ValueError('Task body mapping changed')
    if task['evaluation'].get('contact_actor_paths', {}) != task['evaluation']['candidate_metadata'].get('contact_actor_paths', {}):
        raise ValueError('Frozen tray contact mapping changed')
    if scene['task_layers'][0]['target_prim_path']!='/World/LivingTask':raise ValueError('Task subtree changed')
    dt = scene['isaacsim']['physics_dt_s']
    profile = scene.get('living_task_provenance', {}).get('physics_profile', {})
    if not isinstance(profile, dict):raise ValueError('Unexpected integration profile')
    half_step_profile = 'tower_extraction_tgs_external_forces_v2'
    if dt == .00025:
        if (task['id'] != 'tower_extraction' or profile.get('profile_id') != half_step_profile
                or profile.get('isaacsim_options') != {'physics_dt_s': .00025,
                    'enable_external_forces_every_iteration': True}
                or scene['isaacsim'].get('enable_external_forces_every_iteration') is not True
                or profile.get('integrator', {}).get('control_dt_s') != .02):
            raise ValueError('Unexpected integration profile')
    elif dt != .0005 or profile.get('profile_id') == half_step_profile:
        raise ValueError('Unexpected integration profile')
    if (scene.get('physics', {}).get('dt_s') != dt or task['evaluation']['physics_dt_s'] != dt
            or profile.get('physics_dt_s') != dt or task['evaluation']['sample_period_s'] != .02):
        raise ValueError('Inconsistent task integration clocks')
    return initial


def _freeze_evaluator(destination):
    source=Path(__file__).parent;destination.mkdir()
    for name,dest in [('evaluation.py','sequence_evaluator.py'),('native_evidence.py','native_evidence.py'),
                      ('tower_geometry.py','tower_geometry.py'),('scene.py','scene.py'),('dual_robot_layout.py','dual_robot_layout.py'),
                      ('height_public.py','height_public.py')]:
        shutil.copy2(source/name,destination/dest)
    if (source/'tower_trays.py').is_file():
        shutil.copy2(source/'tower_trays.py', destination/'tower_trays.py')
    # Preserve the original reference algorithms, binding their family-local
    # imports to this frozen package instead of mutable scene source.
    for task in ('tower_building','tower_extraction'):
        folder=destination/'tasks'/task;folder.mkdir(parents=True)
        for name in ('task.py','rules.py','evaluator.py','freeform.py'):
            if not (source/'tasks'/task/name).is_file():
                continue
            text=(source/'tasks'/task/name).read_text()
            text=text.replace('from scenes.workrooms.living_room.','from ...')
            (folder/name).write_text(text)


def validate_release(root,manifest,*,assets=True):
    from bench.isaacsim.release import _frozen_evaluator_module
    root=Path(root).resolve()
    if (manifest.get('schema')!=SCHEMA or manifest.get('schema_version')!=VERSION
        or manifest.get('status')!='prepared_unvalidated' or manifest.get('physics_accepted') is not False
        or manifest.get('task_policy_success_claimed') is not False or manifest.get('original_release_modified') is not False):
        raise ValueError('Living engineering release must preserve unqualified status')
    files=manifest['files'];required={'scene.json','task.json','initial_state.json','rules.json','asset_file_hashes.json',
        'physics_profile.json','source_provenance.json','evaluator/sequence_evaluator.py','evaluator/native_evidence.py','robot_stands.usda'}
    if not required<=files.keys() or manifest['files_sha256']!={p:r['sha256'] for p,r in files.items()}:
        raise ValueError('Incomplete frozen release inventory')
    for relative,row in files.items():
        path=root/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Release path escapes frozen directory')
        checked(path,row['sha256'])
    scene,task,initial,rules,profile=(load_json(root/n) for n in
        ('scene.json','task.json','initial_state.json','rules.json','physics_profile.json'))
    if (initial_state(scene,task)!=initial or task['public_rules']!=rules or profile['isaacsim_options']!=scene['isaacsim']
            or profile.get('profile_selection') != scene['living_task_provenance']['physics_profile']):
        raise ValueError('Frozen reset/rules/profile mismatch')
    entries=manifest['tasks']
    if len(entries)!=1 or entries[0]['id']!=task['id'] or manifest['task_id']!=task['id']:raise ValueError('Task entry mismatch')
    for key in ('scene','task','initial_state'):
        if entries[0][key]!=key+'.json' or entries[0][key+'_sha256']!=files[key+'.json']['sha256']:
            raise ValueError('Task entry input mismatch')
    if (manifest.get('physics_profile_sha256')!=files['physics_profile.json']['sha256']
        or manifest.get('physics_profile')!='physics_profile.json'
        or manifest.get('task_kind')!=entries[0].get('task_kind')
        or entries[0].get('task_kind')!=task.get('task_kind','single_agent')):
        raise ValueError('Frozen profile or task-kind identity mismatch')
    inventory=load_json(root/'asset_file_hashes.json')
    if manifest['asset_files_sha256']!={p:r['sha256'] for p,r in inventory.items()}:raise ValueError('Asset index mismatch')
    if assets and inventory!=asset_inventory(scene):raise ValueError('Incomplete asset dependency inventory')
    _frozen_evaluator_module(root).SequenceEvaluator(task)
    return manifest


def prepare_task_release(task_id,output,*,seed=0):
    if task_id not in AUTHORING_TASK_IDS:
        raise ValueError('Use the canonical task release entrypoint; legacy remote authoring is removed')
    output=Path(output).resolve()
    if not output.is_relative_to(DATASETS) or output.exists():raise ValueError('Choose a fresh directory under shared datasets')
    output.mkdir(parents=True)
    try:
        (output/'robot_stands.usda').write_text(stand_usda(task_id))
        values=build_task_inputs(task_id,seed=seed,stand_path=output/'robot_stands.usda')
        scene,task=values['scene'],values['task'];initial=initial_state(scene,task);inventory=asset_inventory(scene)
        for name,value in [('scene.json',scene),('task.json',task),('initial_state.json',initial),('rules.json',task['public_rules']),('asset_file_hashes.json',inventory)]:dump(output/name,value)
        dump(output/'physics_profile.json',{'backend':'Isaac Sim 6.1 native PhysX GPU',
            'isaacsim_options':deepcopy(scene['isaacsim']),'material_calibrated':False,'native_task_success_claimed':False,
            'profile_selection':scene['living_task_provenance']['physics_profile']})
        _freeze_evaluator(output/'evaluator')
        source=Path(__file__).parent
        dump(output/'source_provenance.json',{'sources_sha256':{str(p):digest(p) for p in sorted(source.rglob('*.py'))},
            'parent_environment':scene['workroom'],'candidate':scene['living_task_provenance']['candidate'],
            'frozen_reference_import_rebinding':'Family-local tower imports made relative to immutable evaluator package',
            'original_assets_modified':False,'original_releases_modified':False,'physics_accepted':False})
        files={str(p.relative_to(output)):{'sha256':digest(p)} for p in sorted(output.rglob('*')) if p.is_file()}
        entry={k:task[k] for k in ('id','title_zh','difficulty','cloth')};entry['task_kind']=task.get('task_kind','single_agent')
        for name in ('scene','task','initial_state'):entry.update({name:name+'.json',name+'_sha256':files[name+'.json']['sha256']})
        candidate=scene['living_task_provenance']['candidate']
        manifest={'schema':SCHEMA,'schema_version':VERSION,'task_id':task_id,'release_id':task_id+'_'+output.name,
            'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'status':'prepared_unvalidated','source_release':None,
            'original_release_modified':False,'task_kind':entry['task_kind'],'tasks':[entry],'files':files,'files_sha256':{p:r['sha256'] for p,r in files.items()},
            'asset_files_sha256':{p:r['sha256'] for p,r in inventory.items()},'physics_profile':'physics_profile.json',
            'physics_profile_sha256':files['physics_profile.json']['sha256'],
            'asset_catalog':{'path':candidate['manifest_path'],'sha256':candidate['manifest_sha256']},
            'physics_accepted':False,'task_policy_success_claimed':False,'qualification_scope':'Executable engineering candidate; no native robot success or physics acceptance.'}
        dump(output/'release_manifest.json',manifest);validate_release(output,manifest)
        return manifest
    except BaseException:
        shutil.rmtree(output)
        raise


def task_inputs(root,task_id,manifest,*,seed=0,phase='nominal'):
    if phase!='nominal':raise ValueError('Living tasks allow only the physical nominal reset')
    root=Path(root).resolve()
    validate_release(root,manifest,assets=True)
    entry=next((r for r in manifest['tasks'] if r['id']==task_id),None)
    if entry is None:raise ValueError('Task is absent from release')
    scene=load_json(root/entry['scene']);task=load_json(root/entry['task'])
    initial=load_json(root/entry['initial_state'])
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Seed must be unsigned 32-bit integer')
    task['seed']=seed;scene['scenario']['seed']=seed
    if task_id=='remote_operation':task['public_goal']['button_sequence']=['power','confirm'] if seed%2==0 else ['confirm','power']
    if task_id=='tower_extraction':
        from bench.isaacsim.release import _frozen_evaluator_module
        frozen=_frozen_evaluator_module(root)
        import importlib
        factory=importlib.import_module(frozen.__package__+'.tasks.tower_extraction.task').task_configuration
        c=task['evaluation']['reference_configuration'];r=c['rules']
        task['evaluation']['reference_configuration']=factory(c['geometry'],seed=seed,
            max_turns_per_player=r['max_turns_per_player'],protected_bottom_layers=r['protected_bottom_layers'],thresholds=r['thresholds'],
            **({'collection_trays': c['collection_trays']} if 'collection_trays' in c else {}))
    task['evaluation_phase']=phase
    return deepcopy(entry),scene,task,initial
