"""Native task inputs referencing immutable living-room candidate subtrees."""
from copy import deepcopy
import importlib
from pathlib import Path

from bench.isaacsim.release import checked, digest, task_inputs
from bench.paths import load_json
from scenes.workrooms.registry import BASE_RELEASE, BASE_MANIFEST_SHA256, PHYSICS_OVERRIDES
from scenes.workrooms.observations import camera_specs
from .scene import environment_reference
from .scene_candidates import scene_candidate

TASK_IDS = ('remote_operation', 'water_pouring', 'tower_building', 'tower_extraction')
# Remote task layers remain readable for frozen historical releases only.
AUTHORING_TASK_IDS = ('water_pouring', 'tower_building', 'tower_extraction')
BODY_PATHS = {
    'remote_operation': {'remote_body': '/World/LivingTask/Remote/Body',
        'power_button': '/World/LivingTask/Remote/PowerButton',
        'confirm_button': '/World/LivingTask/Remote/ConfirmButton', 'holder': '/World/LivingTask/Holder'},
    'water_pouring': {'kettle': '/World/LivingTask/Kettle', 'cup': '/World/LivingTask/Cup'},
    **{task: {f'block_{i:02d}': f'/World/LivingTask/Blocks/block_{i:02d}' for i in range(18)}
       for task in ('tower_building', 'tower_extraction')},
}


def task_body_paths(task_id, metadata):
    """Bind tower inventory to the frozen asset configuration, including spares."""
    if task_id not in ('tower_building', 'tower_extraction'):
        return deepcopy(BODY_PATHS[task_id])
    configuration = metadata['configuration']
    names = (configuration['block_ids'] if task_id == 'tower_building'
             else configuration['blocks'])
    expected = {name: '/World/LivingTask/Blocks/' + name for name in names}
    authored = {row['block_id']: row['prim_path'] for row in metadata['objects']}
    if authored != expected or len(authored) != len(metadata['objects']):
        raise ValueError('Tower asset inventory differs from its frozen configuration')
    return expected


def candidate_inputs(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown living-room task')
    candidate = scene_candidate(task_id)
    checked(candidate['manifest_path'], candidate['manifest_sha256'])
    checked(candidate['initial_usd'], candidate['initial_usd_sha256'])
    manifest = load_json(Path(candidate['manifest_path']))
    metadata_path = Path(candidate['manifest_path']).with_name('metadata.json')
    checked(metadata_path, manifest['metadata_sha256'])
    metadata = load_json(metadata_path)
    dependencies = {p: r['sha256'] for p, r in manifest['dependencies'].items()}
    dependencies.update({candidate['manifest_path']: candidate['manifest_sha256'],
                         str(metadata_path): manifest['metadata_sha256']})
    for p, sha in dependencies.items():
        checked(p, sha)
    actual = {row['path'] for row in manifest['inventory']['bodies']}
    if actual != set(task_body_paths(task_id, metadata).values()) or any(
            not row['enabled'] or row['kinematic'] for row in manifest['inventory']['bodies']):
        raise ValueError('Candidate dynamic inventory changed')
    from .tray_contacts import validate_tray_contact_mapping
    validate_tray_contact_mapping(metadata, manifest['inventory'])
    return candidate, manifest, metadata, dependencies


def _robots(task_id):
    if task_id != 'tower_extraction':
        from .robot_layout import robot_configuration, ready_configuration, stand_geometry
        robot = robot_configuration()
        initial, ready = ready_configuration(robot)
        return robot, initial, ready, {'front': stand_geometry()}
    from .dual_robot_layout import robot_configurations, ready_configuration, stand_geometry
    robots = robot_configurations()
    robot = deepcopy(robots['front'])
    robot.update(hand_id='fr3_sharpa_instances', assembly_version='namespaced_v1',
                 base_position_m=[0., 0., 0.], base_quaternion_wxyz=[1., 0., 0., 0.])
    robot.pop('arm_home_rad', None)
    # The composite root has no physical flange mount. Each pinned original
    # source assembly retains its own unchanged mount inside its namespace.
    robot.pop('mount_position_m', None)
    robot.pop('mount_quaternion_wxyz', None)
    robot['instances'] = {side: {k: deepcopy(r[k]) for k in
        ('hand_id', 'assembly_version', 'base_position_m', 'base_quaternion_wxyz')}
        for side, r in sorted(robots.items())}
    robot['arm_home_rad_by_side'] = {side: deepcopy(r['arm_home_rad']) for side, r in robots.items()}
    initial, ready = {'joint_positions_rad': {}}, {}
    for side, r in sorted(robots.items()):
        state, design = ready_configuration(r)
        ready[side] = design
        for name, value in state['joint_positions_rad'].items():
            target = ('fr3_'+side+'_'+name[4:]) if name.startswith('fr3_') else side+'_'+name[6:]
            if not name.startswith(('fr3_', 'right_')):
                raise ValueError('Unsupported source joint name')
            initial['joint_positions_rad'][target] = value
    return robot, initial, ready, stand_geometry()


def stand_usda(task_id):
    """Exact original task stand boxes; no task-body physics overrides."""
    from .robot_layout import stand_geometry as single
    from .dual_robot_layout import stand_geometry as dual
    stands = dual() if task_id == 'tower_extraction' else {'front': single()}
    lines = ['#usda 1.0', '(metersPerUnit = 1; upAxis = "Z")',
             'def Xform "World" {', 'def Xform "RobotStands" {']
    for side, spec in sorted(stands.items()):
        for item in spec['boxes']:
            p, size = item['position_m'], item['size_m']
            lines.extend([f'def Cube "{side}_{item["name"]}" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{',
                'double size = 1', f'double3 xformOp:translate = ({p[0]}, {p[1]}, {p[2]})',
                f'double3 xformOp:scale = ({size[0]}, {size[1]}, {size[2]})',
                'uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                'color3f[] primvars:displayColor = [(0.15, 0.16, 0.17)]', '}'])
    return '\n'.join([*lines, '}', '}', ''])


def build_task_inputs(task_id, *, seed=0, stand_path=None):
    if task_id == 'remote_operation':
        raise ValueError('Task1 uses runtime_task.build_task_inputs; legacy remote authoring is removed')
    candidate, manifest, metadata, dependencies = candidate_inputs(task_id)
    bodies = task_body_paths(task_id, metadata)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be unsigned 32-bit integer')
    checked(BASE_RELEASE/'release_manifest.json', BASE_MANIFEST_SHA256)
    _, scene, _, _ = task_inputs(BASE_RELEASE, '01_cube_push')
    for key in ('room', 'scene_variant', 'static_props', 'surfaces', 'robot_pedestal', 'soft_bodies'):
        scene.pop(key, None)
    robot, initial, ready, stands = _robots(task_id)
    reference = environment_reference()
    checked(reference['usd_path'], reference['usd_sha256'])
    checked(reference['dependency_manifest_path'], reference['dependency_manifest_sha256'])
    description = load_json(Path(reference['dependency_manifest_path']))
    scene.update(robot=robot, objects=[], fixtures=[], goal_markers=[], cloth={'enabled': False},
        workbench={'enabled': False}, workcell_details={'enabled': False}, workroom=reference,
        cameras=camera_specs(description, robot=robot), scenario={'family': 'living_room', 'task_id': task_id, 'seed': seed})
    if metadata.get('profile') == 'trays_v1':
        scene['cameras']['task'].update(eye=[1.65, -.95, 1.75], target=[.65, 0., .98],
                                        focal_length_mm=30.)
    scene['isaacsim'].update(deepcopy(PHYSICS_OVERRIDES))
    scene['isaacsim']['physics_dt_s'] = .0005
    physics_profile = {
        'physics_dt_s': .0005,
        'prior_water_probe_dt_s': 1/960 if task_id == 'water_pouring' else None,
        'qualification_transferred': False,
        'selection': 'Explicit common bench integration profile',
    }
    if task_id == 'tower_extraction':
        from .tasks.tower_extraction.physics_profile import task_physics_profile
        physics_profile.update(task_physics_profile())
        scene['isaacsim'].update(physics_profile['isaacsim_options'])
        scene['physics']['dt_s'] = scene['isaacsim']['physics_dt_s']
    scene['task_layers'] = [{'id': 'task', 'usd_path': candidate['initial_usd'],
        'sha256': candidate['initial_usd_sha256'], 'source_prim_path': '/World/LivingTask',
        'target_prim_path': '/World/LivingTask', 'dependencies_sha256': dependencies,
        'rigid_bodies': deepcopy(bodies), 'particle_sets': {}}]
    if stand_path is None or not Path(stand_path).is_file():
        raise ValueError('A frozen stand layer path is required')
    scene['task_layers'].append({'id': 'robot_stands', 'usd_path': str(Path(stand_path).resolve()),
        'sha256': digest(stand_path), 'source_prim_path': '/World/RobotStands',
        'target_prim_path': '/World/RobotStands', 'dependencies_sha256': {},
        'rigid_bodies': {}, 'particle_sets': {}})
    if task_id == 'water_pouring':
        water = metadata['water']
        scene['task_layers'][0]['particle_sets']['water'] = {
            'path': water['particles_path'], 'system_path': water['system_path'],
            'count': water['particle_count'], 'total_mass_kg': water['total_mass_kg'],
            'position_attribute': 'points'}
    if any(layer.get('particle_sets') for layer in scene['task_layers']):
        # Explicit data output selection; GPU physics and its parameters stay intact.
        scene['native_readback_profile'] = 'gpu_solver_usd_host_v1'
    module = importlib.import_module(f'.tasks.{task_id}.rules', __package__)
    if task_id.startswith('tower_'):
        configuration = deepcopy(metadata['configuration'])
        if task_id == 'tower_extraction':
            from .tasks.tower_extraction.task import task_configuration
            r = configuration['rules']
            configuration = task_configuration(configuration['geometry'], seed=seed,
                max_turns_per_player=r['max_turns_per_player'],
                protected_bottom_layers=r['protected_bottom_layers'], thresholds=r['thresholds'],
                **({'collection_trays': configuration['collection_trays']}
                   if 'collection_trays' in configuration else {}))
        rules = deepcopy(configuration['rules'])
    else:
        configuration = None
        rules = module.public_rules()
    rules['runtime_status']='engineering_candidate'
    rules['native_manipulation_qualified']=False
    task = {'schema_version': 'living-room-task-v1', 'id': task_id, 'task_id': task_id,
        'title_zh': {'remote_operation': '操作遥控器', 'water_pouring': '倒水',
                     'tower_building': '搭积木塔', 'tower_extraction': '轮流抽积木'}[task_id],
        'difficulty': 'engineering_candidate', 'cloth': False, 'seed': seed,
        'initial_state': initial, 'public_rules': rules,
        'public_goal': {'operation': task_id}, 'steps': [{'id': task_id, 'instruction_zh': '按公开规则完成任务。'}],
        'evaluation': {'schema': 'living_native_evidence_v1', 'body_paths': deepcopy(bodies),
            'candidate_metadata': metadata, 'reference_configuration': configuration,
            'physics_dt_s': scene['isaacsim']['physics_dt_s'], 'sample_period_s': .02, 'support_force_min_n': 1.e-6,
            'stable_duration_s': 1., 'maximum_stable_linear_speed_m_s': .01,
            'maximum_stable_angular_speed_rad_s': .05},
        'runtime_adapter': 'living_room_v1', 'physics_accepted': False,
        'task_policy_success_claimed': False, 'recommended_budgets': {
            'sim_time_s': 120., 'wall_time_s': 1800., 'max_actions': 6000}}
    if metadata.get('contact_actor_paths'):
        task['evaluation']['contact_actor_paths'] = deepcopy(metadata['contact_actor_paths'])
    if configuration and configuration.get('scoring_mode') == 'freeform_height':
        from .height_public import bind_height_rules
        bind_height_rules(task)
    if task_id == 'tower_extraction':
        task['task_kind'] = 'turn_game'
        task['players'] = [{'player_id': 'player_0', 'robot_id': 'front'}, {'player_id': 'player_1', 'robot_id': 'rear'}]
        scene['turn_game_task'] = {'game_id': task_id, 'players': deepcopy(task['players'])}
    task['qualification'] = {'native_manipulation_qualified': False,
        'candidate_evidence_scope': candidate['evidence_scope'],
        'water_load_transfer_fraction_of_expected_weight': .07510755263013265 if task_id == 'water_pouring' else None,
        'water_load_transfer_relative_shortfall': 1.-.07510755263013265 if task_id == 'water_pouring' else None,
        'water_coupling_qualified': False if task_id == 'water_pouring' else None}
    scene['living_task_provenance'] = {'candidate': candidate, 'original_assets_modified': False,
        'original_robot_dynamics_and_gains_preserved': True, 'stand_geometry': stands,
        'task_layer_scope': 'Original task subtree only; preview robots excluded',
        'physics_profile': physics_profile,
        'ready_design': ready}
    return {'scene': scene, 'task': task, 'initial_state': deepcopy(initial)}
