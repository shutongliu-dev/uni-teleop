"""Freeze an independent, small manipulation release using collaborator USDs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from bench.isaacsim.release import DEFAULT as ORIGINAL_RELEASE, PROFILE, digest, dump, task_inputs, validate
from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
PREVIOUS_SIMPLE_RELEASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/v3')
DEFAULT = PREVIOUS_SIMPLE_RELEASE.parent / 'v7_contact_substeps_base'
ASSET_ROOT = Path(f'{DATA_ROOT}/assets/objects/hand_system0/bench_catalog/v1')
ROBOT_ASSET_ROOT = Path(f'{DATA_ROOT}/assets')

HUMANLIKE_INSTRUCTION = (
    '抓持姿态必须接近自然人手：掌面朝向方块，用拇指与食指或中指形成对指接触，'
    '手指自然屈曲，避免反折、手背夹持和挤压穿透。'
)
LIFT_GRASP_INSTRUCTION = (
    '在抬高达到4厘米前先维持合规抓持至少0.12秒；方块抬高达到4厘米且脱离桌面时必须已经抓稳，'
    '此后到桌面重新承托之前持续保持合规抓持，途中违反后不能靠最后摆正补救。'
)


def specifications(asset_root=ASSET_ROOT, robot_asset_root=ROBOT_ASSET_ROOT):
    """No physics launch. Return source scene/task documents for pre-freeze review."""
    asset_root = Path(asset_root).resolve()
    from .humanlike import default_humanlike_config
    humanlike = default_humanlike_config(Path(robot_asset_root).resolve(), object_id='cube', physics_dt_s=PROFILE['physics_dt_s'])
    catalog = load_json(asset_root / 'manifest.json')
    assets = catalog['assets']
    _, original, _, _ = task_inputs(ORIGINAL_RELEASE, '01_ordered_kitting')
    scene = deepcopy(original)
    scene['schema_version'] = 'isaacsim-simple-manipulation-scene-v1'
    scene['description'] = 'FR3灵巧手桌面基础操作；协作者原始USD物品和灰色工作台。'
    scene['robot']['self_collisions'] = True
    scene['cloth']['enabled'] = False
    scene['fixtures'] = []
    scene['objects'] = []
    positions = [('cube','collab_cube_small',[.45,.10]),
                 ('context_blue_cup','collab_blue_cup',[.74,.38]),
                 ('context_spoon','collab_bench_spoon',[.75,.57]),
                 ('context_peg','collab_peg',[.58,.57])]
    for name, asset_id, xy in positions:
        bottom = float(assets[asset_id]['bounds_min_m'][2])
        scene['objects'].append({'id':name,'asset_id':asset_id,'asset_root':str(asset_root),
            'position_m':[*xy,.751-bottom],'quaternion_wxyz':[1,0,0,0],'friction':.65})
    initial = scene['objects'][0]['position_m']
    resting_z = initial[2] - .001
    modes = [
        ('01_cube_push','桌面推移方块','push',[.45,.20,resting_z],
         '将方块沿桌面推到世界坐标 x=0.45、y=0.20 米附近，移开手，让方块稳定停在桌面。'),
        ('02_cube_lift_hold','抓起方块并保持','lift_hold',[.45,.10,initial[2]+.08],
         '用手抓住方块，将它从初始位置抬高至少8厘米，保持方块大致直立并稳定悬空。'),
        ('03_cube_pick_place','抓起搬运并放置方块','pick_place',[.52,.20,resting_z],
         '抓起方块并抬高至少8厘米，稳定抓持后搬到世界坐标 x=0.52、y=0.20 米附近，再放回桌面并移开手。'),
        ('04_cube_grasp_hold','在桌面上自然抓稳方块','grasp_hold',[.45,.10,resting_z],
         '在方块初始位置附近，以拇指对食指或中指自然抓稳方块并稳定保持；不需要抬起或搬运，抓稳保持阶段方块必须有桌面承托。'),
    ]
    output=[]
    for task_id,title,mode,goal,instruction in modes:
        evaluation={'mode':mode,'object':'cube','initial_position_m':initial,
            'goal_position_m':goal,'initial_tolerance_m':.005,'goal_xy_radius_m':.03,
            'goal_z_tolerance_m':.006,'minimum_displacement_m':.05,'minimum_lift_m':.08,
            'hold_s':.30,'lift_stage_hold_s':.12,'maximum_speed_m_s':.04,
            'maximum_angular_speed_rad_s':.3,'maximum_motion_speed_m_s':1.5,
            'maximum_tilt_rad':.7,'minimum_contact_duration_s':.04,'max_observation_gap_s':.041,
            'physics_dt_s':PROFILE['physics_dt_s'],'control_period_s':.02}
        evaluation['humanlike_grasp'] = deepcopy(humanlike)
        evaluation['humanlike_grasp_role'] = 'sustained_candidate_grasp_constraint' if mode == 'push' else 'required'
        if mode != 'push':
            instruction += HUMANLIKE_INSTRUCTION
        else:
            evaluation['push_grasp_violation_hold_s'] = .12
            instruction += ('推移允许单指接触，无需形成对指抓持；若拇指和食指或中指在同一观测窗口内分别接触目标，'
                '连续0.12秒不满足下列抓持姿态与对向接触约束会判定抓持违规，短暂闭合和松开的过渡不会单独判失败。'
                + HUMANLIKE_INSTRUCTION)
        if mode in ('lift_hold', 'pick_place'):
            evaluation['humanlike_lift_start_m'] = .04
            instruction += LIFT_GRASP_INSTRUCTION
        constraints = {'humanlike_grasp_required': mode != 'push',
            'description': HUMANLIKE_INSTRUCTION,
            'scope': 'Simulation pose/contact proxy, not a learned human motion score; SHARPA W01 right hand only.',
            'push_grasp_scoring': 'A one-finger push needs no grasp. A thumb/primary-finger grasp candidate that continuously fails the pose/opposed-contact constraint for 0.12 s causes an irreversible violation.',
            'criteria': {key: deepcopy(humanlike[key]) for key in (
                'hold_s', 'joint_limit_tolerance_rad', 'maximum_dip_minus_pip_rad',
                'minimum_contact_finger_flexion_sum_rad', 'maximum_normal_direction_cosine',
                'minimum_opposed_substep_fraction', 'maximum_robot_penetration_m',
                'wrist_approach_axis_local', 'wrist_approach_direction_world',
                'maximum_wrist_approach_angle_rad', 'object_in_wrist_box_min_m',
                'object_in_wrist_box_max_m', 'maximum_tip_direction_cosine',
                'minimum_tip_span_m', 'maximum_tip_span_m', 'maximum_tip_object_distance_m',
                'maximum_object_to_tip_segment_distance_m')}}
        if mode in ('lift_hold', 'pick_place'):
            constraints.update({'stable_grasp_before_lift_s': .12, 'continuous_grasp_from_lift_m': .04,
                                'continuous_grasp_until': 'table_support_returns', 'violation_is_irreversible': True})
        task={'schema_version':'simple-manipulation-task-v1','id':task_id,'title_zh':title,
            'difficulty':'elementary','cloth':False,'target_position_m':goal,
            'steps':[{'id':mode,'instruction_zh':instruction}], 'evaluation':evaluation,
            'public_goal':{'object_id':'cube','operation':mode,'target_position_m':goal,
                'frame_id':'world','description':instruction, 'grasp_constraints': constraints},
            'recommended_budgets':{'sim_time_s':45.,'wall_time_s':1800.,'max_actions':2250},
            'evaluation_scope':'Solver hand contact, measured object motion and stable terminal condition; grasp/lift/place additionally require frozen humanlike pose and opposed-contact evidence. Lift and transport retain any grasp violation. Humanlike grasp dwell precedes, and does not replace, the original task dwell. Pick-place requires ordered lift, in-hand transport and released table support.',
            'training_policy_included':False}
        specific=deepcopy(scene)
        specific['description']=instruction
        specific['goal_markers']=[{'id':'cube_target','position_m':[*goal[:2],.7505],
            'half_extents_m':[.035,.035,.0002],'color':[.12,.65,.28],
            'semantics':'Public target display only; noncolliding. Marker extent is not a scoring threshold.'}]
        output.append((specific,task))
    return output


def prepare(output=DEFAULT, asset_root=ASSET_ROOT, robot_asset_root=ROBOT_ASSET_ROOT):
    output, asset_root = Path(output).resolve(), Path(asset_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new release version; frozen output already exists: ' + str(output))
    specs=specifications(asset_root, robot_asset_root)
    catalog=asset_root/'manifest.json'
    catalog_data=load_json(catalog)
    # Pin every file in the compact adapter, including dependency inventories and
    # derived collision geometry. Original USDs are pinned by those inventories.
    asset_files={str(p.resolve()):{'sha256':digest(p)} for p in sorted(asset_root.rglob('*')) if p.is_file()}
    selected={obj['asset_id'] for scene,_ in specs for obj in scene['objects']}
    for _, task in specs:
        source = task['evaluation']['humanlike_grasp']['hand_urdf']
        asset_files[str(Path(source['path']).resolve())] = {'sha256': source['sha256']}
    for asset_id in sorted(selected):
        asset=catalog_data['assets'][asset_id]
        dependencies=asset['source_dependencies_sha256']
        if not dependencies or asset['usd_path'] not in dependencies:
            raise ValueError('Selected USD asset lacks its source dependency inventory: '+asset_id)
        for path, expected in dependencies.items():
            asset_files[str(Path(path).resolve())]={'sha256':expected}
        for part in asset['collision_parts']:
            if 'mesh_path' in part:
                path=(asset_root/part['mesh_path']).resolve()
                asset_files[str(path)]={'sha256':part['sha256']}
    # The adapter's dependency schema is authoritative; load verifies original
    # external USD references even when its compact manifest uses another layout.
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    for scene,_ in specs:
        for obj in scene['objects']:
            load_task_asset(obj['asset_id'],asset_root=asset_root,verify_hashes=True)
    for path,record in asset_files.items():
        if digest(path)!=record['sha256']:
            raise ValueError('Source asset changed while preparing release: '+path)
    output.mkdir(parents=True,exist_ok=True)
    files={}
    def save(relative,value):
        dump(output/relative,value)
        files[relative]={'sha256':digest(output/relative)}
    entries=[]
    for scene,task in specs:
        entry={key:task[key] for key in ('id','title_zh','difficulty','cloth')}
        for kind,value in [('scene',scene),('task',task)]:
            relative=f'tasks/{task["id"]}/{kind}.json'
            save(relative,value)
            entry[kind]=relative
            entry[kind+'_sha256']=files[relative]['sha256']
        entries.append(entry)
    evaluator=PROJECT/'scenes/manipulation/evaluator.py'
    destination=output/'evaluator/sequence_evaluator.py'
    destination.parent.mkdir()
    shutil.copy2(evaluator,destination)
    files['evaluator/sequence_evaluator.py']={'sha256':digest(destination),'source':str(evaluator)}
    humanlike_source = PROJECT / 'scenes/manipulation/humanlike.py'
    humanlike_destination = output / 'evaluator/humanlike.py'
    shutil.copy2(humanlike_source, humanlike_destination)
    files['evaluator/humanlike.py'] = {'sha256': digest(humanlike_destination), 'source': str(humanlike_source)}
    save('asset_file_hashes.json',asset_files)
    save('source_provenance.json',{
        'generator':{'path':str(Path(__file__)),'sha256':digest(__file__)},
        'layout_template_release':str(ORIGINAL_RELEASE),
        'layout_template_release_sha256':digest(ORIGINAL_RELEASE/'release_manifest.json'),
        'previous_simple_release':str(PREVIOUS_SIMPLE_RELEASE),
        'previous_simple_release_sha256':digest(PREVIOUS_SIMPLE_RELEASE/'release_manifest.json'),
        'previous_simple_release_modified':False,
        'original_three_task_physical_thresholds_modified':False,
        'added_grasp_contract':'Frozen SHARPA W01 pose and opposed solver-contact proxy; no human-motion realism claim.',
        'original_frozen_tasks_modified':False,'task_and_evaluator_independent':True,
        'collaborator_asset_catalog':str(catalog),'license_status':'See collaborator asset provenance; no redistribution license is inferred.'})
    profile={'backend':'Isaac Sim 6.1 native PhysX GPU','isaacsim_options':deepcopy(PROFILE),
        'material_calibrated':False,'cross_engine_material_equivalence_claimed':False}
    dump(output/'physics_profile.json',profile)
    manifest={'schema_version':'isaacsim-simple-manipulation-release-v1',
        'release_id':'isaacsim_simple_manipulation_'+output.name,
        'status':'prepared_unvalidated','prepared_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_release':None,'original_release_modified':False,'tasks':entries,'files':files,
        'physics_profile':'physics_profile.json','physics_profile_sha256':digest(output/'physics_profile.json'),
        'asset_catalog':{'path':str(catalog),'sha256':digest(catalog)},
        'physics_accepted':False,'task_policy_success_claimed':False,
        'purpose':'Independent elementary tasks; never pool success with the frozen sequential workcell.'}
    dump(output/'release_manifest.json',manifest)
    validate(output)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--asset-root',type=Path,default=ASSET_ROOT)
    parser.add_argument('--robot-asset-root',type=Path,default=ROBOT_ASSET_ROOT)
    args=parser.parse_args()
    result=prepare(args.output,args.asset_root,args.robot_asset_root)
    print(json.dumps({'release':str(args.output),'tasks':[row['id'] for row in result['tasks']]},indent=2))


if __name__=='__main__':
    main()
