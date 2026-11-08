"""Single-Agent FR3 cooperation with asset-bound hand layouts and frozen task inputs."""
from copy import deepcopy
import json
from pathlib import Path
from bench.isaacsim.release import checked,digest
from .task_scene import empty_task_scene
from .cooperative_assets import TASK_ID,ROOT,ASSET_ROOT
from .desktop_dressing import task_layer
from bench.paths import load_json

TASK_IDS=(TASK_ID,)
TASK_SCHEMA='mechanical-cooperative-task-v1'
PHYSICS_DT=.0005


def evaluator_sources(task_id=TASK_ID):
    if task_id!=TASK_ID:raise ValueError('Unknown cooperative task')
    root=Path(__file__).parent
    return {'sequence_evaluator.py':root/'cooperative_evaluation.py','rules.py':root/'cooperative_rules.py','native_evidence.py':root/'nist_native_evidence.py'}


def load_asset_manifest(task_id,asset_manifest):
    if task_id!=TASK_ID:raise ValueError('Unknown cooperative task')
    path=Path(asset_manifest['manifest_path'] if isinstance(asset_manifest,dict) else asset_manifest).resolve()
    if isinstance(asset_manifest,dict):checked(path,asset_manifest['manifest_sha256'])
    manifest=load_json(path)
    if manifest.get('schema')!='mechanical-cooperative-asset.v1' or manifest.get('task_id')!=TASK_ID or manifest.get('root_prim_path')!=ROOT or manifest.get('presentation') is not False:raise ValueError('Expected cooperative reset asset')
    checked(manifest['usd_path'],manifest['usd_sha256']);checked(manifest['metadata_path'],manifest['metadata_sha256'])
    metadata=load_json(Path(manifest['metadata_path']))
    return {'manifest_path':str(path),'manifest_sha256':digest(path),'usd_path':manifest['usd_path'],
        'usd_sha256':manifest['usd_sha256'],'metadata_path':manifest['metadata_path'],
        'metadata_sha256':manifest['metadata_sha256'],'root_prim_path':ROOT,'dependencies':manifest['dependencies']},metadata


def robot_inputs(layout_id='mechanical_opposed_dual_right_sharpa_v1'):
    from .cooperative_robot_layout import robot_inputs as configured_inputs
    return configured_inputs(layout_id)


def bind_episode_task(task,seed,rules):
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Unsigned 32-bit seed required')
    task=deepcopy(task);task['episode_seed']=seed
    task['public_rules']=rules.public_rules(TASK_ID,seed)
    task['evaluation']=rules.evaluation_configuration(TASK_ID,task['asset_metadata'],seed,PHYSICS_DT,task['room_support_paths'])
    task['public_goal']={'operation':TASK_ID,'frame_id':'world_m_z_up','description':task['public_rules']['instruction_zh']}
    return rules.bind_task_rules(task)


def build_task_inputs(task_id=TASK_ID,*,asset_manifest=ASSET_ROOT/'manifest.json',seed=0):
    from . import cooperative_rules as rules
    from scenes.workrooms.observations import camera_specs
    candidate,metadata=load_asset_manifest(task_id,asset_manifest)
    layout_id=metadata.get('robot_layout_id','mechanical_opposed_dual_right_sharpa_v1')
    scene,_,_=empty_task_scene();robot,initial,ready=robot_inputs(layout_id)
    description=load_json(Path(scene['workroom']['dependency_manifest_path']))
    scene.update(robot=robot,fixtures=[],cameras=camera_specs(description,robot=robot))
    if robot['hand_id']=='dual_fr3_sharpa_w01':
        wrist=scene['cameras'].pop('wrist')
        for side in ('left','right'):
            scene['cameras'][side+'_wrist']={**deepcopy(wrist),'mount_link':side+'_hand_flange',
                'title_zh':('左' if side=='left' else '右')+'腕部相机'}
    scene['cameras']['task'].update(eye=[1.75,-1.05,1.80],target=[.65,0,.85],focal_length_mm=32.)
    if layout_id in ('mechanical_same_side_dual_right_sharpa_v1','mechanical_same_side_left_right_sharpa_v1'):
        # The overview retains both complete arms; this view resolves the
        # actual socket opening and independent peg without surface labels.
        scene['cameras']['task'].update(eye=[1.25,-.82,1.42],target=[.59,-.015,.83],focal_length_mm=40.)
    if metadata.get('visual_detail',{}).get('profile')=='v2':
        # Resolve the actual mouth, gripping surfaces and tray in the task
        # view while keeping the other four public camera baselines intact.
        scene['cameras']['task'].update(eye=[.99,-.53,1.28],target=[.61,-.015,.81],
            focal_length_mm=42.,resolution=[1440,1080])
    scene['isaacsim']['physics_dt_s']=PHYSICS_DT
    scene['physics']={'dt_s':PHYSICS_DT,'control_dt_s':.02,'gravity_m_s2':[0,0,-9.81],'backend':'native_isaacsim_physx_gpu'}
    scene.pop('scenario',None);scene.pop('mechanical_task_provenance',None)
    scene['description']='机械装配室 Task10：同一Agent双臂协作插接。'
    scene['task_layers']=[task_layer()]
    task={'schema_version':TASK_SCHEMA,'id':TASK_ID,'task_id':TASK_ID,'title_zh':'双臂协作插接','difficulty':10,
        'cloth':False,'initial_state':deepcopy(initial),'asset_metadata':metadata,'room_support_paths':description['support_colliders'],
        'steps':[{'id':TASK_ID,'instruction_zh':rules.INSTRUCTION}],
        'recommended_budgets':{'sim_time_s':300.,'wall_time_s':3600.,'max_actions':15000},
        'training_policy_included':False,'physics_accepted':False,'task_policy_success_claimed':False,
        'known_limitations':metadata['known_limitations']}
    task=bind_episode_task(task,seed,rules);sources=evaluator_sources()
    scene['native_task_bundle']={'schema':'native-usd-task-bundle-v1','usd_path':candidate['usd_path'],
        'usd_sha256':candidate['usd_sha256'],'root_prim_path':ROOT,'rigid_bodies':deepcopy(metadata['actors']),
        'joints':{},'particle_sets':{},'observer':{'path':'evaluator/sequence_evaluator.py',
        'sha256':digest(sources['sequence_evaluator.py']),'files':{n:digest(p) for n,p in sources.items()},
        'class_name':'NativeTaskObserver','configuration':deepcopy(task)}}
    scene['mechanical_cooperative_provenance']={'asset_candidate':candidate,'ready_design':ready,
        'layout_id':layout_id,'parent_environment_unchanged':True,
        'shared_controller_unchanged':True,'physical_reset_seeded':False,
        'task_object_pose_writes_after_reset':False,'same_agent_simultaneous_control':True}
    return {'scene':scene,'task':task,'initial_state':initial}
