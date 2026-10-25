"""L2 task composition and reset-only engineering contact probes."""
from copy import deepcopy
import hashlib
from pathlib import Path
import math

from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scenes.workrooms.mechanical_assembly.task_scene import DEFAULT_STAND_ROOT, empty_task_scene
from scenes.workrooms.mechanical_assembly.assembly_geometry import physical_spec
from scenes.workrooms.mechanical_assembly.referee import bind_task_rules
from scenes.workrooms.mechanical_assembly.rules import thresholds, public_rules
from .assets import ASSET_IDS, DEFAULT_ASSET_ROOT, PARAMETERS

TASK_ID='peg_insertion'
WORKPLATE_POSITION_M=[.40,0.,.78]


def build_task_inputs(*, asset_root=DEFAULT_ASSET_ROOT, stand_root=DEFAULT_STAND_ROOT, solver_type=None):
    """Prepare free supply pegs, fixed sockets and the unchanged shared robot."""
    asset_root=Path(asset_root).resolve();p=PARAMETERS
    for asset_id in ASSET_IDS.values(): load_task_asset(asset_id,asset_root,verify_hashes=True)
    scene,initial,_=empty_task_scene(stand_root,solver_type=solver_type)
    scene['fixtures'].append(physical_spec(ASSET_IDS['workplate'],asset_root,'socket_workplate',WORKPLATE_POSITION_M))
    target_centers={};supply_positions={}
    for index,y in enumerate([-.25,.25]):
        object_id=f'peg_{index}';cradle_id=f'supply_cradle_{index}'
        cradle=[.25,y,.78]
        scene['fixtures'].append(physical_spec(ASSET_IDS['cradle'],asset_root,cradle_id,cradle))
        position=[cradle[0],cradle[1],cradle[2]+p['supply_cradle_floor_height']]
        scene['objects'].append(physical_spec(ASSET_IDS['peg'],asset_root,object_id,position))
        supply_positions[object_id]=deepcopy(position)
        target_centers[object_id]=[WORKPLATE_POSITION_M[0],(-.5 if index==0 else .5)*p['socket_center_spacing'],
                                   WORKPLATE_POSITION_M[2]+p['workplate_height']]
    for spec in [*scene['objects'],*scene['fixtures']]:
        if spec['id']!='robot_stand':
            spec['collision_mesh_coordinate_scale']=1000.
    scene['description']='Two independent pegs in open supply cradles; fixed chamfered blind sockets.'
    scene['scenario']={'family':'mechanical_assembly','task_id':TASK_ID,'seed':0,'scope':'Original unassembled task reset.'}
    scene['mechanical_task_provenance'].update(task_id=TASK_ID,
        asset_manifest_sha256=hashlib.sha256((asset_root/'manifest.json').read_bytes()).hexdigest(),
        collision_mesh_coordinate_scale=1000.,
        native_task_dynamics_verified=False)
    evaluation={'actors':{**{f'peg_{i}':f'/World/Objects/peg_{i}' for i in range(2)},
                          'fixture':'/World/Fixtures/socket_workplate'},
        'robot_actor_prefix':'/World/Robot',
        'initial_objects':{o['id']:{k:deepcopy(o[k]) for k in ('position_m','quaternion_wxyz')} for o in scene['objects']},
        'target_bottom_centers_m':target_centers,'peg_height_m':p['peg_length'],
        'radial_clearance_m':(p['socket_bore_diameter']-p['peg_diameter'])/2,
        'maximum_axis_offset_m':.00325,'maximum_bottom_height_error_m':.0005,
        'maximum_axis_tilt_rad':math.radians(10.),'minimum_stable_s':.5,
        'maximum_stable_linear_speed_m_s':.005,'maximum_stable_angular_speed_rad_s':.05,
        'maximum_reported_penetration_m':.0005,
        'required_support_pairs':[[f'peg_{i}','fixture'] for i in range(2)],
        'must_be_released':True,'threshold_status':'Engineering candidates; require native validation.'}
    evaluation.update(thresholds(TASK_ID))
    task={'schema_version':'mechanical-peg-insertion-task-v1','id':TASK_ID,'task_id':TASK_ID,
        'title_zh':'粗销浅孔放置','difficulty':'guided_single_arm','cloth':False,
        'public_goal':{'operation':TASK_ID,'frame_id':'world','description':'将两根粗销分别放入标记孔位，放到底后松手。'},
        'steps':[{'id':'place','instruction_zh':'将两根销放入对应孔位，确认落在底座上。'},
                 {'id':'release','instruction_zh':'松手并保持稳定至少0.5秒。'}],
        'evaluation':evaluation,'public_rules':public_rules(TASK_ID),'initial_state':deepcopy(initial),
        'reset_conditions':{'kind':'unassembled_supply','supply_positions_m':supply_positions},
        'physics_accepted':False,'task_policy_success_claimed':False,'training_policy_included':False,
        'qualification_scope':'Authored physical candidate. Geometry/CPU checks alone do not qualify native insertion.',
        'recommended_budgets':{'sim_time_s':90.,'wall_time_s':1800.,'max_actions':4500}}
    bind_task_rules(task)
    return {'scene':scene,'task':task,'initial_state':initial}


def build_probe_inputs(*, asset_root=DEFAULT_ASSET_ROOT, stand_root=DEFAULT_STAND_ROOT,
                       lateral_offset_m=.008, case='guided_drop', solver_type=None):
    """New diagnostic reset: gravity must traverse the funnel and reach a seat.

    No running state writes or body forces are implied. The task's original
    supply reset remains separate and cannot receive credit for this probe.
    """
    if case not in ('guided_drop','outside_socket'):
        raise ValueError('Unknown peg contact probe')
    if not math.isfinite(lateral_offset_m) or abs(lateral_offset_m)>.010:
        raise ValueError('Guided offset must lie within +/-10 mm')
    result=build_task_inputs(asset_root=asset_root,stand_root=stand_root,solver_type=solver_type)
    top=WORKPLATE_POSITION_M[2]+PARAMETERS['workplate_height']+PARAMETERS['straight_seating_depth']+PARAMETERS['socket_guide_depth']
    offset=.045 if case=='outside_socket' else lateral_offset_m
    for index,obj in enumerate(result['scene']['objects']):
        target=result['task']['evaluation']['target_bottom_centers_m'][obj['id']]
        obj['position_m']=[target[0]+offset*(1 if index==0 else -1),target[1],top+.003]
    diagnostic={'kind':'gravity_socket_guidance','case':case,'reset_only':True,
        'lateral_offset_m':offset,'release_above_rim_m':.003,
        'minimum_downward_travel_m':.025,'expected_functional_success':case=='guided_drop',
        'requires_guide_contact_when_offset_exceeds_bore_clearance':abs(offset)>.003,
        'requires_positive_seat_support':True,'task_success_credit':False,
        'original_task_reset_modified':False}
    result['scene']['scenario']['scope']='Separate privileged engineering reset; not original task execution.'
    result['scene']['mechanical_task_provenance']['engineering_probe']=deepcopy(diagnostic)
    result['task']['engineering_probe']=diagnostic
    return result
