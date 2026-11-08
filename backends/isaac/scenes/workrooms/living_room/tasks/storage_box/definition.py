"""Storage-box task entry; qualifications distinguish object physics from policy."""
from bench.paths import DATA_ROOT
from scenes.workrooms.living_room.task_design import robot_requirements
from scenes.workrooms.living_room.scene_candidates import scene_candidate
from .rules import public_rules, INSTRUCTION

TASK={
    'schema':'workroom_task_definition.v1','family':'living_room','task_id':'storage_box',
    'task_number':5,'difficulty':3,'title':'Task5 · 卡扣收纳盒取物','status':'implemented',
    'user_confirmed_task':True,'entrypoint':'scenes.workrooms.living_room.tasks.storage_box.task',
    'preparation_state':'frozen_executable_candidate',
    'release_schema':'living_room_manipulation_task_release.v1',
    'release_schema_version':'isaacsim-living-room-manipulation-release-v1',
    'scene_candidate':scene_candidate('storage_box'),
    'objects':{'intended_roles':['dynamic_box','passive_hinged_lid','two_physical_rotary_catches','blue_item','collection_tray'],
        'asset_root':f'{DATA_ROOT}/assets/task_parts/living_room/storage_box/v2',
        'asset_bindings':'Versioned USD, metadata and dependencies pinned by release hashes.'},
    'layout':{'frame':'world','workspace':'parent_scene_table','parent_scene':'living_room/scene.py',
        'same_environment_for_all_tasks':True,'initialization':'Closed lid and catches, item in box, empty tray.'},
    'robot_requirements':{**robot_requirements(1),'ownership':'One agent; canonical single right hand. Optional left/right dual configurations preserve the object contract.'},
    'rules':{'intent':INSTRUCTION,'implementation':'scenes.workrooms.living_room.tasks.storage_box.rules','contract':public_rules()},
    'evaluator':{'implementation':'scenes.workrooms.living_room.tasks.storage_box.evaluator',
        'reference_implementation':'scenes.workrooms.living_room.tasks.storage_box.evaluator.SequenceEvaluator',
        'frozen_path':'evaluator/sequence_evaluator.py',
        'scope':'Consecutive native transforms and contacts establish actual unlocking, lid opening, item support, reclosure and 300 ms released stability.'},
    'source_release':{'path':f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/storage_box/v3',
        'manifest_sha256':'c1c5c032c9d5dd34b9ef78506f7ca7fd534bfca73cc6ab88285a4c49fe39c163'},
    'qualification':{'status':'candidate_object_interactions_passed','native_preflight':False,'manipulation':False,
        'physics_accepted':False,'public_http_smoke':False,'public_functions_smoke':False,
        'object_interaction_evidence':{
            'path':f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/living/qualification_exact.json',
            'sha256':'e715dd685e6b11d24edae14da13a7193b0b0c100d34649777ee7d4b6261e2de3',
            'passed':True,
            'scope':'Bounded engineering-load lock/open/transfer/close/release chain and negative cases; 31,002 native frames replay exactly in the original Isaac Python.'},
        'scope':'Object interactions passed; robot manipulation, canonical HTTP and configuration IK remain separate qualifications.'},
}

# Post-run metadata only; the frozen release and its preparation flags are unchanged.
TASK['qualification'].update({'native_preflight': True,
 'agent_bench_interface': True,
 'public_http_smoke': True,
 'public_functions_smoke': True,
 'object_interactions': True,
 'discrete_workspace': True,
 'nonadjacent_arm_self_collision': True,
 'configuration_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/configuration_qualification.json',
                            'sha256': '670335af5e31afb4ee76ffe4988b0ee397c695b6a59e2abfe9d77df40901d08e'},
 'manipulation': False,
 'physics_accepted': False,
 'scope': 'Native object mechanisms, exact selected Bench/Agent interfaces and recorded-stage discrete '
          'approaches verified separately. Complete robot manipulation and continuous paths remain '
          'unqualified.'})


from scenes.workrooms.living_room.randomization.catalog import bind_definition

TASK = bind_definition(TASK)
