"""Five implemented task designs; runnable publication pins frozen releases."""
from copy import deepcopy
from .composition import CAPABILITIES, ROBOT_CAPABILITIES
from .rules import ALL_TASK_IDS as TASK_IDS, GARMENT_TASK_IDS, GARMENT_FOLDING_TASK_IDS, CURRENT_TASK_IDS, default_profile, TITLES, public_rules, thresholds, ENGINEERING_CONTROL, ROBOT_CONTROL
from .robot_layout import LAYOUT_ID
from bench.paths import DATA_ROOT

# Historical input pins remain available by canonical ID; historical outcomes stay external.
HISTORICAL_ROBOT_RELEASES = {'towel_flattening': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laundry_five_tasks_20261023/releases/towel_flattening/dual_v1',
                      'manifest_sha256': '93ed09981afa3d85d57afdd65c35eb145c6aedffea2b8d4ef6aa3475a3198b2c'},
 'towel_folding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laundry_five_tasks_20261023/releases/towel_folding/dual_v1',
                   'manifest_sha256': 'a854d68bdb0f320c3cb0938c3d3e430878c4ba23f60811bd4d344ce1dc5b4d9d'},
 'towel_stacking': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laundry_five_tasks_20261023/releases/towel_stacking/dual_v1',
                    'manifest_sha256': 'aa2df9fbba6d874203794ae931cdbfa82776e2209112ac5ebda720a5d4ecb530'},
 'shirt_stacking': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laundry_five_tasks_20261023/releases/shirt_stacking/dual_v3',
                    'manifest_sha256': 'ca8835b887d70685fb835a96ce96579140c6c0cc64556a88f37e62b71d66e10a'},
 'trousers_stacking': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laundry_five_tasks_20261023/releases/trousers_stacking/dual_v2',
                       'manifest_sha256': '62c491d328320642b36b0830883b0c735bd3c6cd6a29db22f24377dffcdbb1cc'},
 'trousers_folding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laundry_single_garment_folding_20261023/releases/trousers_folding/outer_yaw70_v4',
                      'manifest_sha256': '54b0226fe348296039932e7dee76842037fa13848c1f6fce62c9a51a2a0136ff'},
 'shirt_folding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laundry_single_garment_folding_20261023/releases/shirt_folding/flat_settled_v4',
                   'manifest_sha256': 'e092645e068429d30466e6830895de19f6b9e159dc1dfc72f423add8bef52a50'}}
from .robot_configuration_catalog import RELEASES as CONFIGURED_RELEASES
ROBOT_RELEASES = deepcopy(HISTORICAL_ROBOT_RELEASES)
ROBOT_QUALIFICATION = {}
for _task_id, _options in CONFIGURED_RELEASES.items():
    ROBOT_RELEASES[_task_id] = deepcopy(_options['base_release'])
    ROBOT_QUALIFICATION[_task_id] = deepcopy(
        _options['configurations'][_options['default_configuration_id']]['qualification'])
RELEASES = dict(ROBOT_RELEASES)
# The selected release owns its rule version even when new authoring defaults
# advance. Update these bindings only when selecting newly prepared releases.
ROBOT_RULE_PROFILES = {task: 'laundry_single_garment_folding_task.v4' for task in GARMENT_FOLDING_TASK_IDS}
for _task_id, _options in CONFIGURED_RELEASES.items():
    _selected = _options['configurations'][_options['default_configuration_id']]
    ROBOT_RULE_PROFILES[_task_id] = _selected['rules']['contract']['rule_profile']
SUPERSEDED_TASKS = {'shirt_stacking': 'shirt_folding', 'trousers_stacking': 'trousers_folding'}


def task_definition(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown laundry task')
    release = RELEASES.get(task_id)
    implemented = release is not None
    robot = task_id in ROBOT_RELEASES or task_id in GARMENT_FOLDING_TASK_IDS
    number = (CURRENT_TASK_IDS.index(task_id)+1 if task_id in CURRENT_TASK_IDS else TASK_IDS.index(task_id)+1)
    mode = ROBOT_CONTROL if robot else ENGINEERING_CONTROL
    selected_profile = ROBOT_RULE_PROFILES.get(task_id, default_profile(task_id))
    result = {'schema': 'workroom_task_definition.v1', 'family': 'laundry_room',
        'task_id': task_id, 'task_number': number,
        'difficulty': number, 'title': TITLES[task_id],
        'status': 'implemented' if implemented else 'planned',
        'entrypoint': 'scenes.workrooms.laundry_room.tasks.'+task_id+'.entrypoint',
        'release_schema': 'laundry_cloth_task_release.v1',
        'release_schema_version': 'isaacsim-laundry-cloth-release-v1',
        'capabilities': dict(ROBOT_CAPABILITIES if robot else CAPABILITIES),
        'objects': {'roles': ['towel', 'towel_base'] if task_id.endswith('_stacking') else ['towel'],
            'model': 'live_surface_deformable', 'size_m': [.6, .4],
            'physical_thickness_m': .003, 'mass_kg': .108, 'properties_basis': 'simulation_priors'},
        'layout': {'parent_scene': 'laundry_room', 'same_environment': True,
            'primary_surface': 'folding_table', 'background_modified': False},
        'robot_requirements': {'mode': 'engineering_cloth_probe', 'robot_manipulation_supported': False},
        'rules': {'implementation': 'scenes.workrooms.laundry_room.rules:public_rules', 'contract': public_rules(task_id, profile=selected_profile, control_mode=mode)},
        'evaluator': {'implementation': 'scenes.workrooms.laundry_room.evaluation:SequenceEvaluator' if implemented else None,
            'reference_implementation': 'scenes.workrooms.laundry_room.evaluation:SequenceEvaluator',
            'frozen_evaluator': 'evaluator/sequence_evaluator.py', 'thresholds': thresholds(task_id, selected_profile)},
        'source_release': release,
        'qualification': {'native_preflight': False, 'engineering_cloth_operation': False,
            'robot_manipulation': False, 'physics_accepted': False,
            'scope': 'Task rules and frozen inputs; native engineering evidence is independently source-bound.'}}
    if task_id in (*GARMENT_TASK_IDS, *GARMENT_FOLDING_TASK_IDS):
        result['objects'].update(model='live_garment_surface_deformable',
            garment_kind='shirt' if task_id.startswith('shirt_') else 'trousers',
            physical_thickness_m=.0016 if task_id.startswith('shirt_') else .0018)
        result['objects'].pop('size_m')
        result['objects'].pop('mass_kg')
        if task_id in GARMENT_FOLDING_TASK_IDS:
            result['objects'].pop('physical_thickness_m')
            result['objects'].update(physical_thickness_range_m=[.0005, .002],
                                     properties_basis='versioned_single_garment_asset')
    if robot:
        result['robot_requirements'] = {'arm': 'fr3', 'hand_id': 'dual_fr3_sharpa_w01',
            'assembly_version': 'isaacsim_6_1_v2', 'articulations': 2,
            'active_sides': ['left', 'right'], 'layout_id': LAYOUT_ID}
        result['layout']['robot_installation'] = LAYOUT_ID
        result['layout']['control_scope'] = 'One Agent controls both arms and official left/right hands, 58 DOF.'
        result['qualification'].update(ROBOT_QUALIFICATION.get(task_id, {}))
        result['qualification']['scope'] = 'Robot installation and ordinary joint-action integration; manipulation and grasp qualification remain separate.'
    if task_id in SUPERSEDED_TASKS:
        result['superseded_by'] = SUPERSEDED_TASKS[task_id]
        result['historical_scope'] = 'Earlier folded-garment stacking task; retained by canonical ID, not a current numbered folding task.'
    if not implemented:
        result['cannot_run_reason'] = 'Task implementation exists; immutable release pointer is awaiting preparation and review.'
    return result
