"""L3 free annular parts, fixed guided shaft, and gravity assembly diagnostics."""
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
from .collision_candidates import bind_dense_rings, validate_selection
from .layout import (PART_ORDER as LAYOUT_PART_ORDER, cpu_reachability_targets,
                     layout_contract, privileged_phase_contract, validate_layout,
                     validate_privileged_contract)

TASK_ID='shaft_spacer_assembly'
SHAFT_POSITION_M=[.40,0.,.78]
PART_ORDER=LAYOUT_PART_ORDER


def build_task_inputs(*, asset_root=DEFAULT_ASSET_ROOT, stand_root=DEFAULT_STAND_ROOT,
                      solver_type=None, ring_collision_candidate=None):
    """Return the unassembled task without changing the shared room or robot."""
    validate_selection(ring_collision_candidate, solver_type)
    asset_root=Path(asset_root).resolve();p=PARAMETERS
    for asset_id in ASSET_IDS.values(): load_task_asset(asset_id,asset_root,verify_hashes=True)
    scene,initial,_=empty_task_scene(stand_root,solver_type=solver_type)
    scene['fixtures'].append(physical_spec(ASSET_IDS['shaft_fixture'],asset_root,'shaft_fixture',SHAFT_POSITION_M))
    supplies={};targets={};stack_z=SHAFT_POSITION_M[2]+p['workplate_height']
    for name,y in zip(PART_ORDER,[-.27,0.,.27]):
        rest=[.25,y,.78]
        scene['fixtures'].append(physical_spec(ASSET_IDS['supply_rest'],asset_root,name+'_supply_rest',rest))
        position=[rest[0],rest[1],rest[2]+p['supply_rest_height']]
        scene['objects'].append(physical_spec(ASSET_IDS[name],asset_root,name,position))
        supplies[name]=position;targets[name]=[SHAFT_POSITION_M[0],SHAFT_POSITION_M[1],stack_z]
        stack_z+=p[name+'_outer_diameter_height'][1]
    # Keep authored world geometry; condition only local convex cooking coordinates.
    for spec in [*scene['objects'],*scene['fixtures']]:
        if spec['id']!='robot_stand':
            spec['collision_mesh_coordinate_scale']=1000.
    scene['description']='Three free bored parts on separate rests; fixed shaft and lower shoulder.'
    scene['scenario']={'family':'mechanical_assembly','task_id':TASK_ID,'seed':0,'scope':'Original unassembled task reset.'}
    task_layout = layout_contract()
    scene['task_layout']=deepcopy(task_layout)
    scene['mechanical_task_provenance'].update(task_id=TASK_ID,
        asset_manifest_sha256=hashlib.sha256((asset_root/'manifest.json').read_bytes()).hexdigest(),
        collision_mesh_coordinate_scale=1000.,
        native_task_dynamics_verified=False,
        task_layout_id=task_layout['layout_id'])
    if ring_collision_candidate is not None:
        bind_dense_rings(scene)
    evaluation={'actors':{**{n:'/World/Objects/'+n for n in PART_ORDER},'fixture':'/World/Fixtures/shaft_fixture'},
        'robot_actor_prefix':'/World/Robot',
        'initial_objects':{o['id']:{k:deepcopy(o[k]) for k in ('position_m','quaternion_wxyz')} for o in scene['objects']},
        'part_order':list(PART_ORDER),'target_bottom_centers_m':targets,
        'part_heights_m':{n:p[n+'_outer_diameter_height'][1] for n in PART_ORDER},
        'radial_clearance_m':(p['part_bore_diameter']-p['shaft_diameter'])/2,
        'maximum_axis_offset_m':.00425,'maximum_bottom_height_error_m':.0005,
        'maximum_axis_tilt_rad':math.radians(5.),'minimum_stable_s':.5,
        'maximum_stable_linear_speed_m_s':.005,'maximum_stable_angular_speed_rad_s':.05,
        'maximum_reported_penetration_m':.0005,
        'required_support_pairs':[['spacer','fixture'],['wide_ring','spacer'],['top_ring','wide_ring']],
        'must_be_released':True,'threshold_status':'Engineering candidates; require native validation.'}
    evaluation.update(thresholds(TASK_ID))
    layout_check = validate_layout(scene, {'task_id': TASK_ID, 'evaluation': evaluation,
                                           'task_layout': task_layout})
    task={'schema_version':'mechanical-shaft-spacer-task-v1','id':TASK_ID,'task_id':TASK_ID,
        'title_zh':'轴上隔套顺序装配','difficulty':'guided_single_arm','cloth':False,
        'public_goal':{'operation':TASK_ID,'frame_id':'world','description':'依次把隔套、宽环和顶环装到固定轴上，落座后松手。'},
        'steps':[{'id':n,'instruction_zh':f'按顺序装入 {n}，确认被下方零件或肩台承托后松手。'} for n in PART_ORDER],
        'evaluation':evaluation,'public_rules':public_rules(TASK_ID),'initial_state':deepcopy(initial),
        'reset_conditions':{'kind':'unassembled_supply','supply_positions_m':supplies},
        'task_layout':deepcopy(task_layout),
        'cpu_contract': {
            'layout_check': layout_check,
            'reachability_targets': cpu_reachability_targets(),
            'recovery': deepcopy(task_layout['recovery']),
            'reset_only': True,
            'native_privileged_entrypoint': 'scripts.validation.validate_mechanical_shaft_spacer:run',
            'ordinary_agent_validation': False,
        },
        'privileged_validation': privileged_phase_contract(),
        'physics_accepted':False,'task_policy_success_claimed':False,'training_policy_included':False,
        'qualification_scope':'Authored physical candidate. CPU geometry does not qualify native guidance or stack support.',
        'recommended_budgets':{'sim_time_s':120.,'wall_time_s':2400.,'max_actions':6000}}
    bind_task_rules(task)
    validate_privileged_contract()
    return {'scene':scene,'task':task,'initial_state':initial}


def build_probe_inputs(*, asset_root=DEFAULT_ASSET_ROOT, stand_root=DEFAULT_STAND_ROOT,
                       lateral_offset_m=.007, case='guided_stack_drop', solver_type=None):
    """Release unmounted parts above the tip; gravity must guide and stack them."""
    if case not in ('guided_stack_drop','misordered_stack_drop'):
        raise ValueError('Unknown shaft contact probe')
    if not math.isfinite(lateral_offset_m) or abs(lateral_offset_m)>.010:
        raise ValueError('Guided offset must lie within +/-10 mm')
    result=build_task_inputs(asset_root=asset_root,stand_root=stand_root,solver_type=solver_type)
    order=PART_ORDER if case=='guided_stack_drop' else tuple(reversed(PART_ORDER))
    bottom=SHAFT_POSITION_M[2]+PARAMETERS['workplate_height']+PARAMETERS['shaft_exposed_height']+.003
    by_id={obj['id']:obj for obj in result['scene']['objects']}
    for name in order:
        by_id[name]['position_m']=[SHAFT_POSITION_M[0]+lateral_offset_m,SHAFT_POSITION_M[1],bottom]
        bottom+=PARAMETERS[name+'_outer_diameter_height'][1]+.010
    diagnostic={'kind':'gravity_shaft_guidance_and_stack','case':case,'reset_only':True,
        'lateral_offset_m':lateral_offset_m,'release_above_tip_m':.003,'release_order':list(order),
        'minimum_downward_travel_m':.070,'expected_functional_success':case=='guided_stack_drop',
        'requires_guide_contact_when_offset_exceeds_bore_clearance':abs(lateral_offset_m)>.004,
        'requires_positive_layer_support':True,'task_success_credit':False,
        'original_task_reset_modified':False}
    result['scene']['scenario']['scope']='Separate privileged engineering reset; not original task execution.'
    result['scene']['mechanical_task_provenance']['engineering_probe']=deepcopy(diagnostic)
    result['task']['engineering_probe']=diagnostic
    return result
