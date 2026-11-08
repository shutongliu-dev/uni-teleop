"""Practical dual-arm insertion rules: retries allowed, real cooperation required."""
from copy import deepcopy
import hashlib,json,math
TASK_ID='cooperative_insertion'
INSTRUCTION='由同一个Agent同时控制两台机械臂配合：一手扶住可移动套筒，另一手将插销插入；确认配合后把装配件放入蓝色收集托盘，松手放稳。两臂角色可以互换，允许重抓、退出重插和调整。'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def thresholds(task_id=TASK_ID):
    if task_id!=TASK_ID:raise ValueError('Unknown cooperative task')
    return dict(stable_s=.3,pose_window_s=.02,maximum_linear_speed_m_s=.05,
        maximum_angular_speed_rad_s=.6,position_drift_m=.006,orientation_drift_rad=.18,
        minimum_movement_m=.012,minimum_contact_impulse_ns=1e-10,
        maximum_penetration_m=.006,maximum_mating_gap_m=.0005,support_memory_s=.05,
        placement_tolerance_m=.006,fallen_height_m=.10,cooperation_time_s=.08,
        max_time_s=300.,max_actions=15000)


def public_rules(task_id=TASK_ID,seed=0):
    if task_id!=TASK_ID or type(seed) is not int or not 0<=seed<2**32:raise ValueError('Known task and uint32 seed required')
    reasons={'reset_baseline_not_scored':'初始化只建立基准，不计完成。','task_in_progress':'可以继续调整和重试。',
        'cooperation_confirmed':'双手分工及真实插入进展已确认，可以将装配件放回托盘。',
        'waiting_for_release_and_stability':'装配件在托盘中，等待松手稳定。','success':'双臂配合、实际插接和托盘稳定均满足。',
        'object_fell':'任务物体实际落地。','physics_safety_stop':'明显物理穿透，与任务失败分开记录。',
        'native_evidence_unknown':'原生证据缺失或不一致，本次不评分。','budget_exhausted':'预算耗尽，任务未完成。'}
    referee={'schema':'physical_task_referee_contract.v1','task_id':task_id,'instruction_zh':INSTRUCTION,
        'phases':{'working':'两臂抓取和插接。','placing':'将已配合装配件移入托盘。','settling':'松手等待稳定。'},
        'reason_descriptions':reasons,'thresholds':thresholds(),
        'success_conditions':['套筒离开初始位置至少1.5厘米，两只不同机械手分别接触套筒和插销，并在真实配合期间产生至少8毫米相对插入进展。',
            '双手配合接触累计至少0.08秒；允许中途换手、重抓和重新插入。',
            '插销进入套筒至少3厘米，具有真实接触；允许正常间隙，不要求压装或精确受力。',
            '两物体保持装配关系且由蓝色托盘实际承托，完全松手稳定0.3秒。'],
        'recovery':['台面上错放、退出、重抓、短暂接触中断均可恢复，不扣尝试次数。',
            '允许套筒在操作中接触台面辅助支撑，但另一只手必须实际参与扶持。',
            '物体掉到台面可以拾回；实际落地记失败，预算耗尽记未完成。'],
        'outcomes':{'ongoing':'继续操作。','success':'完成。','task_failure':'物体实际落地。',
            'evaluator_invalid':'证据异常不评分。','safety_stop':'物理异常停止。','budget_exhaustion':'未完成。','cancelled':'取消。'},
        'physical_actions':'同一Agent通过普通episodes同时控制任务声明的两台FR3及其机械手；每只手的名称和实际左右手型以get_robot为准，无轮次竞争。',
        'observation_scope':'公开阶段与规则；私有原生裁判读取完整位姿和真实接触，不接受Agent自报完成。'}
    return {'schema':'mechanical_cooperative_rules.v1','task_id':task_id,'task_number':10,
        'instruction_zh':INSTRUCTION,'referee':referee,'reset_credit':False,
        'randomization':'Frozen authored physical reset; seed records identity only.',
        'background_props':'远端工具架、零件盒和量具为固定环境道具，不是操作目标或评分支撑。'}


def evaluation_configuration(task_id,metadata,seed=0,physics_dt_s=.0005,room_support_paths=None):
    rules=public_rules(task_id,seed)
    if not 0<physics_dt_s<=.002 or abs(.02/physics_dt_s-round(.02/physics_dt_s))>1e-8:raise ValueError('Invalid native period')
    if metadata.get('task_id')!=task_id or set(metadata['actors'])!={'sleeve','peg'}:raise ValueError('Two independent native parts required')
    objects={r['id']:r for r in metadata['objects']}
    if set(objects)!=set(metadata['actors']) or len(metadata['objects'])!=2:raise ValueError('Object metadata differs')
    for name,row in objects.items():
        if row['body_path']!=metadata['actors'][name] or row['mass_kg']<=0:raise ValueError('Invalid physical part')
        if len(row['initial_pose_xyzw'])!=7 or not all(math.isfinite(x) for x in row['initial_pose_xyzw']):raise ValueError('Invalid reset')
    sides=metadata['task_parameters']['robot_instances']
    if sides not in (['front','rear'],['left','right']):raise ValueError('Two declared robot hands required')
    scopes=metadata['task_parameters'].get('robot_stage_scopes')
    if (sides==['left','right'] and scopes!={'left':'Left','right':'Right'}) or (sides==['front','rear'] and scopes is not None):
        raise ValueError('Physical hand namespaces differ from the declared robot layout')
    return {'schema':'mechanical_cooperative_evaluation.v1','task_id':task_id,'episode_seed':seed,
        'physics_dt_s':float(physics_dt_s),'metadata':deepcopy(metadata),
        'room_support_paths':deepcopy(room_support_paths or {}),'thresholds':thresholds(),
        'public_rules_sha256':digest(rules)}


def bind_task_rules(task):
    task['public_rules_sha256']=digest(task['public_rules'])
    task['public_referee_contract_sha256']=digest(task['public_rules']['referee'])
    task['public_rules_file']='rules.json'
    return task
