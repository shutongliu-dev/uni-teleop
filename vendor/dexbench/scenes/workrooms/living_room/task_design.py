"""Living-room task metadata; rules readiness never implies a native release."""
from copy import deepcopy

from .scene import environment_reference
from .scene_candidates import scene_candidate
from bench.paths import DATA_ROOT


def robot_requirements(articulations):
    if articulations not in (1, 2):
        raise ValueError('Living tasks use one or two independent right-hand robots')
    sides = ['front'] if articulations == 1 else ['front', 'rear']
    return {
        'arm': 'FR3', 'hand_id': 'sharpa_w01_right', 'assembly_version': 'isaacsim_6_1_v2',
        'articulations': articulations, 'active_sides': sides,
        'configuration_module': 'scenes.workrooms.living_room.dual_robot_layout',
        'configuration_factory': 'robot_configurations',
        'base_positions_m': {side: [.65, -.76 if side == 'front' else .76, .78] for side in sides},
        'base_constraint': 'native_fixed_base_v1', 'geometry_scale': [1., 1., 1.],
        'ownership': 'One agent per active arm; shared bench controls and original source joints.',
        'qualification': 'Existing ready-pose/endpoint checks are not task-motion qualification.',
    }


PHYSICAL_CONTRACT = {
    'actions': 'Robot joint targets only after reset; scene code owns task reset and scoring.',
    'forbidden_shortcuts': ['object_pose_playback', 'grasp_weld', 'magnetic_snap',
                           'direct_object_force', 'hidden_fixture', 'automatic_object_reset'],
    'recovery': 'Regrasp, use the table, realign and retry through actual physical actions.',
    'materials': 'Freeze versioned simulation priors with assets; do not tune friction to manufacture success.',
    'evidence': 'Authoritative continuous physical observations; agent declarations are never score evidence.',
    'faults': 'Missing evidence, invalid reset or infrastructure faults have no competitive winner.',
    'learning': 'Reuse the common task-known learning policy; these rules do not change learning permissions.',
}


def planned_task(*, task_id, number, title, articulations, intent, objects,
                 design, rule_contract, reference_evaluator=None, confirmed=False):
    return {
        'schema': 'workroom_task_definition.v1', 'family': 'living_room',
        'task_id': task_id, 'task_number': number, 'difficulty': number, 'title': title,
        'status': 'planned', 'entrypoint': f'scenes.workrooms.living_room.tasks.{task_id}.task',
        'preparation_state': 'physical_scene_candidate',
        'scene_candidate': scene_candidate(task_id),
        'user_confirmed_task': confirmed, 'objects': deepcopy(objects),
        'layout': {'frame': 'world', 'workspace': 'parent_scene_table',
                   'parent_scene': 'living_room/scene.py', 'initialization': None,
                   'same_environment_for_all_tasks': True},
        'robot_requirements': robot_requirements(articulations),
        'rules': {'intent': intent, 'implementation': f'scenes.workrooms.living_room.tasks.{task_id}.rules',
                  'contract': deepcopy(rule_contract)},
        'evaluator': {'implementation': None, 'reference_implementation': reference_evaluator,
                      'thresholds': None,
                      'scope': 'CPU rule reference, where present; no native observation adapter or released scorer.'},
        'source_release': None,
        'design': {'schema': 'living_room_task_design.v1', **deepcopy(design),
                   'physical_contract': deepcopy(PHYSICAL_CONTRACT)},
        'qualification': {'native_preflight': False, 'manipulation': False, 'physics_accepted': False,
                          'scope': 'Rules and CPU checks do not establish a runnable physical task.'},
        'cannot_run_reason': 'Scene candidates exist, but native task observation/scoring/action adapters and a runnable frozen release are not available; candidate dimensions and task-specific physical criteria are not qualified.',
    }


def design_configuration(definition, *, seed=0):
    """Inspectable task plan; never load or impersonate a runnable release."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    return {'schema': 'living_room_task_plan.v1', 'task_id': definition['task_id'],
            'task_number': definition['task_number'], 'status': 'planned',
            'seed': seed, 'environment': environment_reference(),
            'robot_requirements': deepcopy(definition['robot_requirements']),
            'rules': deepcopy(definition['rules']['contract']),
            'design': deepcopy(definition['design']), 'initial_state': None,
            'native_execution_available': False, 'source_release': None,
            'cannot_run_reason': definition.get('cannot_run_reason', 'This design-only factory does not launch a task; load the pinned catalog release for native execution.')}


def bind_prepared_release(task, *, path, manifest_sha256):
    """Select a frozen runnable candidate without granting physical acceptance."""
    result = deepcopy(task)
    task_id = result['task_id']
    competitive = task_id == 'tower_extraction'
    result.update(
        status='implemented', preparation_state='frozen_engineering_candidate',
        runtime_entrypoint=('bench.runtime.turn_games:TurnGameSession' if competitive
                            else 'bench.runtime.service:EpisodeSession'),
        runtime_adapter='scenes.workrooms.living_room.runtime.build_runtime',
        release_entrypoint='scenes.workrooms.living_room.layer_release.task_inputs',
        task_kind='turn_game' if competitive else 'single_agent',
        release_schema='living_room_task_release.v1',
        release_schema_version='isaacsim-living-room-release-v1',
        source_release={'path': path, 'manifest_sha256': manifest_sha256},
        evaluator={'implementation': 'scenes.workrooms.living_room.evaluation.SequenceEvaluator',
            'frozen_path': 'evaluator/sequence_evaluator.py',
            'reference_implementation': result['evaluator']['reference_implementation'],
            'thresholds': 'Frozen task.json evaluation and rules.json; no runtime threshold changes.',
            'scope': 'Private complete native samples feed the release-frozen evaluator; Agent claims never score.'},
        qualification={'release_status': 'prepared_unvalidated', 'native_preflight': False,
            'manipulation': False, 'physics_accepted': False, 'task_policy_success_claimed': False,
            'integration_report': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_bench_integration_20261018/report.json',
            'scope': 'Runnable engineering candidate. Startup/API evidence is external and does not establish manipulation or quantitative physics acceptance.'})
    result.pop('cannot_run_reason', None)
    result['objects']['asset_bindings'] = {'frozen_scene': path+'/scene.json',
        'frozen_inventory': path+'/asset_file_hashes.json'}
    result['layout']['initialization'] = 'Pinned task USD subtree plus complete robot initial_state.json; nominal reset only.'
    result['rules']['frozen_contract'] = path+'/rules.json'
    result['rules']['contract_scope'] = 'Design overview only; exact runtime rules are the pinned frozen_contract.'
    result['design']['status'] = 'frozen_engineering_candidate'
    result['design']['dimensions_and_thresholds'] = 'Selected candidate geometry and scoring thresholds are frozen in this release; no physical calibration claim.'
    if not competitive:
        result['robot_requirements'].update(
            configuration_module='scenes.workrooms.living_room.robot_layout',
            configuration_factory='robot_configuration',
            base_positions_m={'front': [.65, -.88, .78]})
    else:
        result['robot_requirements'].update(composite_hand_id='fr3_sharpa_instances',
            composite_assembly_version='namespaced_v1',
            instance_source='Two unchanged sharpa_w01_right / isaacsim_6_1_v2 assemblies; front/rear namespaces.')
        result['players'] = [{'player_id': 'player_0', 'robot_id': 'front'},
                             {'player_id': 'player_1', 'robot_id': 'rear'}]
    if task_id == 'water_pouring':
        result['qualification']['water_load_transfer'] = {
            'fraction_of_expected_water_weight': 0.07510755263013265,
            'quantitative_liquid_load_qualified': False,
            'evidence': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_water_20261018/bounded_evidence01.json',
            'scope': 'Observed wet-minus-dry support reaction is only 7.51% of theoretical water weight; this is not a 7.51% relative error. Task success remains gated off.'}
    return result
