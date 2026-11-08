"""Practical reading stand: a true hinge and a load-bearing sliding support pin."""
from copy import deepcopy
import math
from .rules import canonical_hash, OBSERVATION_SCHEMA, PHASES, REASONS

TASK_ID = 'reading_stand'
HINGE = [.30, 0., .81]
ANGLE_DEG = 55.
THETA = math.radians(ANGLE_DEG)
PIN_CENTER = [HINGE[0]+.25*math.cos(THETA)+.016*math.sin(THETA), 0.,
              HINGE[2]+.25*math.sin(THETA)-.016*math.cos(THETA)]
JOINT_PATHS = ['/World/StudyTask/PanelHinge', '/World/StudyTask/SupportPinSlide']
SUPPORT_PATHS = {'table':'/World/Room/Geometry/worktop/Collider',
                'floor':'/World/Room/Geometry/room_floor/Collider'}
THRESHOLDS = {'physics_dt_s':.0005, 'minimum_stable_s':.4,
    'minimum_panel_angle_deg':50., 'maximum_panel_angle_deg':60.,
    'maximum_pin_center_y_m':.020, 'maximum_book_panel_angle_rad':math.radians(10),
    'maximum_linear_speed_m_s':.025, 'maximum_angular_speed_rad_s':.15,
    'maximum_position_drift_m':.003, 'maximum_orientation_drift_rad':.06,
    'support_gap_m':.004, 'maximum_penetration_m':.002,
    'contact_point_tolerance_m':.004, 'minimum_support_impulse_ns':1.e-10,
    'initial_position_tolerance_m':.0001, 'initial_orientation_tolerance_rad':.002}


def panel_pose(angle_deg=ANGLE_DEG):
    a=math.radians(angle_deg)/2
    return {'position_m':list(HINGE),'quaternion_wxyz':[math.cos(a),0.,-math.sin(a),0.]}


def on_panel(local, angle_deg=ANGLE_DEG):
    a=math.radians(angle_deg);x,y,z=local
    return [HINGE[0]+x*math.cos(a)-z*math.sin(a),y,HINGE[2]+x*math.sin(a)+z*math.cos(a)]


def public_rules(configuration=None):
    instruction='展开桌面阅读支架，把背板调到约55度并滑入侧面的支撑销；把书放到背板与前沿托唇上，全部松手放稳。允许重抓、调整，动作顺序不限。'
    reasons={**REASONS,'panel_angle':'背板尚未在50至60度范围。',
        'pin_not_inserted':'把支撑销推入，使背板由销真实承托。',
        'panel_not_pin_supported':'背板尚未真实压在支撑销上。',
        'book_not_on_panel':'书本须完整落在阅读面内，由背板及前沿托唇承托。',
        'mechanism_not_operated':'须实际展开背板、移动支撑销并将书搬到支架。',
        'mechanism_off_rail':'机构离开其真实铰链或滑轨位置。'}
    contract={'schema':'physical_task_referee_contract.v1','task_id':TASK_ID,
        'instruction_zh':instruction,'phases':deepcopy(PHASES),'reason_descriptions':reasons,
        'success_conditions':[
            '初态背板折平、支撑销退出，书在桌面；初始摆放不计完成。',
            '背板相对桌面为50至60度，侧面支撑销进入中心20毫米范围；背板须有真实向上承托接触落到支撑销。',
            '书完整位于阅读面内，朝向与背板误差不超过10度；背板及前沿托唇提供真实承托，不允许悬空或由手托住。',
            '全部任务刚体与机器人完全分离后稳定至少400毫秒；真实铰链和滑轨可以承载。',
            '允许重抓、重新展开、抽销再插和暂放桌面；不要求一次成功或固定双手动作次序。'],
        'thresholds':deepcopy(THRESHOLDS),'recovery':['240秒内可以调整和重试；落到地板为失败。'],
        'outcomes':{'ongoing':'可继续','success':'机构锁定并真实承托书本','task_failure':'物体落地',
            'evaluator_invalid':'原生证据缺失','budget_exhaustion':'未完成','cancelled':'未完成','safety_stop':'物理异常停止'},
        'physical_actions':'仅机器人关节动作；禁止改写物体、隐藏焊接、吸附或直接施加任务物体外力。',
        'observation_scope':'公开目标、规则和机构尺寸；精确运行时姿态和接触是私有裁判证据。'}
    return {'schema':'study_reading_stand_rules.v1','task_id':TASK_ID,'referee':contract,
        'thresholds':deepcopy(THRESHOLDS),'observation_schema':OBSERVATION_SCHEMA,
        'floor_drop_is_terminal':True,'qualification':'Object engineering probes do not qualify robot manipulation.'}


def configuration():
    pin_initial=[PIN_CENTER[0],.28,PIN_CENTER[2]]
    objects={
        'panel':{'category':'panel','title_zh':'铰链阅读背板','local_bounds_m':[[0.,-.13,-.006],[.32,.13,.045]],'initial_pose':panel_pose(0.)},
        'support_pin':{'category':'pin','title_zh':'横向支撑销','local_bounds_m':[[-.018,-.14,-.018],[.018,.22,.018]],
            'initial_pose':{'position_m':pin_initial,'quaternion_wxyz':[1.,0.,0.,0.]}},
        'book':{'category':'book','title_zh':'绿色硬壳阅读书','local_bounds_m':[[-.070,-.085,-.008],[.070,.085,.008]],
            'initial_pose':{'position_m':[.49,-.31,.7882],'quaternion_wxyz':[1.,0.,0.,0.]}}}
    targets=[]
    def target(label,p,approach='above'):
        targets.append({'label':label,'reference_position_m':p,'approach':approach,
            'approach_direction_world':([0.,1.,0.] if approach=='from_positive_y' else [-math.sin(THETA),0.,math.cos(THETA)] if approach=='normal_to_reading_panel' else [0.,0.,1.]),
            'approach_distance_m':.06 if approach=='from_positive_y' else .045})
    target('panel_handle_folded',on_panel([.304,0.,.026],0.))
    target('panel_handle_deployed',on_panel([.304,0.,.026]))
    target('support_pin_handle_withdrawn',[PIN_CENTER[0],.48,PIN_CENTER[2]],'from_positive_y')
    target('support_pin_handle_inserted',[PIN_CENTER[0],.20,PIN_CENTER[2]],'from_positive_y')
    target('book_source',[.49,-.31,.805])
    target('book_reading_target',on_panel([.11,0.,.03]),'normal_to_reading_panel')
    for target in targets:
        if target['label'].startswith('panel_'):
            target.update(body_id='panel',local_position_m=[.304,0.,.026])
        elif target['label'].startswith('support_pin_'):
            target.update(body_id='support_pin',local_position_m=[0.,.20,0.])
        else:
            target.update(body_id='book',local_position_m=[0.,0.,.016])
    c={'schema':'study_reading_stand_configuration.v1','task_id':TASK_ID,'physics_dt_s':.0005,
       'objects':objects,'supports':{'table':{'bounds_m':[[.2,-1.,.70],[1.1,1.,.78]]}},
       'targets':{},'mechanism':{'hinge_world_m':list(HINGE),'pin_center_world_m':list(PIN_CENTER),
          'panel_angle_deg':ANGLE_DEG,'joint_paths':list(JOINT_PATHS),'pin_travel_m':.28,
          'load_path':'Book -> real panel/ledge contacts -> real support pin contact + world-anchored hinge; pin carried by physical prismatic guide.',
          'panel_book_region_local_m':[[.014,-.120,.002],[.302,.120,.045]]},
       'interaction_targets':targets,'budget':{'max_sim_time_s':240.,'max_robot_actions':None}}
    c['rules']=public_rules(c);c['configuration_sha256']=canonical_hash(c)
    return c


def validate_configuration(value):
    expected=configuration()
    if value!=expected: raise ValueError('Reading stand configuration differs from its frozen authored mechanism')
    return deepcopy(expected)
