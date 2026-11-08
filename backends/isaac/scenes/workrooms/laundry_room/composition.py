"""Explicit engineering or dual-robot inputs in the unchanged laundry v3 room."""
from copy import deepcopy
import json
from pathlib import Path

from .assets import candidate_metadata, load_asset_manifest
from .rules import ALL_TASK_IDS as TASK_IDS, GARMENT_TASK_IDS, GARMENT_FOLDING_TASK_IDS, CURRENT_TASK_IDS, default_profile, TASK_SCHEMA, TITLES, public_rules, thresholds, digest, bind_task_rules, PRACTICAL_PROFILE, LEGACY_PROFILE
from .rules import ENGINEERING_CONTROL, ROBOT_CONTROL
from .scene import environment_reference
from bench.paths import load_json

PHYSICS_DT = .0005
PHYSICS_PRIORS = {'physics_dt_s': PHYSICS_DT, 'cloth_iterations': 64,
    'cloth_linear_damping': .10, 'cloth_thickness_m': .003,
    'cloth_youngs_modulus_pa': 2.e6, 'cloth_poissons_ratio': .3,
    'cloth_surface_bend_stiffness': 2.e4,
    'cloth_contact_offset_m': .003, 'cloth_rest_offset_m': .0015}
CAPABILITIES = {'task_inputs': True, 'engineering_cloth_probe': True,
    'robot_episodes': False, 'robot_grasping_qualified': False,
    'engineering_entrypoint': 'scripts.validation.laundry_cloth_interactions'}
ROBOT_CAPABILITIES = {'task_inputs': True, 'engineering_cloth_probe': False,
    'robot_episodes': True, 'robot_grasping_qualified': False}
# Exact Task10 rigid robot options only. Room contact materials, the static
# mesh mode and every cloth solver/material value retain the towel profile.
ROBOT_PHYSICS_OPTIONS = {
    'robot_self_collisions': True, 'filter_thumb_cmc_structure': True,
    'arm_gravity_compensation': 'physx_feedforward_v1',
    'hand_actuator_model': 'sharpa_official_position_mode_v1',
    'robot_contact_materials': {'pad_friction': .5, 'shell_friction': .5},
    'robot_material_binding': 'collider_instance_v1',
    'robot_base_constraint': 'native_fixed_base_v1'}


def capabilities_for(mode):
    if mode == ENGINEERING_CONTROL:
        return deepcopy(CAPABILITIES)
    if mode == ROBOT_CONTROL:
        return deepcopy(ROBOT_CAPABILITIES)
    raise ValueError('Unknown laundry control mode')


def evaluator_sources(robot=False):
    root = Path(__file__).parent
    result = {'sequence_evaluator.py': root/'evaluation.py', 'rules.py': root/'rules.py'}
    if robot:
        result['native_observer.py'] = root/'native_observer.py'
    return result


def build_task_inputs(task_id, *, asset_manifest=None, seed=0, physics_options=None,
                      rule_profile=None, robot_layout=None, configuration_id=None,
                      stand_manifest=None, public_material_contract=None,
                      cloth_visual_profiles=None):
    from .robot_layout import LAYOUT_ID
    installation = None
    if configuration_id is not None:
        from . import robot_configurations as configured
        if robot_layout is not None or stand_manifest is None or task_id not in CURRENT_TASK_IDS:
            raise ValueError('Choose one current-task configuration and its stand manifest, without a legacy layout')
        installation = configured.installation(configuration_id, stand_manifest)
        robot_layout = installation['layout_id']
    elif stand_manifest is not None or robot_layout not in (None, LAYOUT_ID):
        raise ValueError('Unknown laundry robot layout')
    if task_id not in TASK_IDS:
        raise ValueError('Unknown laundry task')
    rule_profile = default_profile(task_id) if rule_profile is None else rule_profile
    mode = ENGINEERING_CONTROL if robot_layout is None else ROBOT_CONTROL
    rules = (public_rules(task_id, seed, rule_profile) if robot_layout is None else
             public_rules(task_id, seed, rule_profile, control_mode=mode))
    if task_id in GARMENT_FOLDING_TASK_IDS:
        from .garment_folding_assets import candidate_metadata as folding_metadata, load_assets
        candidate, metadata = (load_assets(task_id, asset_manifest) if asset_manifest else
                               (None, folding_metadata(task_id)))
    else:
        candidate, metadata = (load_asset_manifest(task_id, asset_manifest) if asset_manifest else
                               (None, candidate_metadata(task_id)))
    from .rules import ASSET_MATERIAL_CONTRACT
    if public_material_contract not in (None, ASSET_MATERIAL_CONTRACT) or (public_material_contract and configuration_id is None):
        raise ValueError('Public material declarations require their explicit configured contract')
    if configuration_id is not None:
        from .rules import configuration_rules
        rules = configuration_rules(task_id, seed, rule_profile, configuration_id,
            asset_metadata=metadata if public_material_contract else None)
    physics = {**PHYSICS_PRIORS, **metadata.get('physics_options', {}), **(physics_options or {})}
    # Room support materials are an explicit scene option, not cloth mechanics.
    # Keep the settled-reset comparison and frozen cloth profile unchanged.
    from bench.simulation.isaacsim_backend.workroom_materials import material_options
    support_materials = material_options(physics.pop('workroom_surface_materials', None))
    if robot_layout is not None and set(physics) & set(ROBOT_PHYSICS_OPTIONS):
        raise ValueError('Cloth physics options must not override the separately pinned robot profile')
    if ('settled_fold_reset' in metadata
            and physics != metadata['settled_fold_reset']['physics_options']):
        raise ValueError('Derived stack reset must keep the successful fold physics profile')
    nominal_thickness = metadata['cloths'][0]['physical_thickness_m']
    if (physics.get('cloth_thickness_m') != nominal_thickness
            or any(row['physical_thickness_m'] != nominal_thickness for row in metadata['cloths'])
            or not 0 < physics['physics_dt_s'] <= .002
            or abs(.02/physics['physics_dt_s']-round(.02/physics['physics_dt_s'])) > 1.e-8):
        raise ValueError('Preserve the asset physical thickness and compatible native time step')
    cloths = []
    for row in metadata['cloths']:
        cloths.append({'id': row['id'], 'enabled': True, **({'cells_xy': row['cells_xy']} if 'cells_xy' in row else {}),
            'size_m': row['size_m'], 'origin_m': [0., 0., 0.],
            'quaternion_wxyz': [1., 0., 0., 0.], 'mass_kg': row['mass_kg'],
            'friction': row.get('friction', .45) if task_id in GARMENT_FOLDING_TASK_IDS else .45,
            'self_contact': True,
            'visual_thickness_m': row['physical_thickness_m'],
            'visual_texture_paths': deepcopy(row['visual_texture_paths']),
            'initial_vertices_m': row['initial_vertices_m'],
            'isaacsim': {'cloth_thickness_m': row['physical_thickness_m'],
                        'cloth_visual_color': row['display_color']}})
        if task_id in (*GARMENT_TASK_IDS, *GARMENT_FOLDING_TASK_IDS):
            cloths[-1]['mesh_type'] = 'custom_surface'
            for key in ('rest_vertices_m', 'triangles', 'visual_face_groups', 'rest_area_m2'):
                if key in row:
                    cloths[-1][key] = deepcopy(row[key])
        if 'visual_texture_tile_size_m' in row:
            cloths[-1]['visual_texture_tile_size_m'] = deepcopy(row['visual_texture_tile_size_m'])
    # Render-only appearance: hash pins of baked fabric profiles, per cloth id.
    # Omission keeps every historical scene byte-identical.
    if cloth_visual_profiles is not None:
        if not cloth_visual_profiles or set(cloth_visual_profiles) - {row['id'] for row in cloths}:
            raise ValueError('Fabric visual profiles must name existing cloth ids')
        for row in cloths:
            if row['id'] in cloth_visual_profiles:
                row['visual_profile'] = deepcopy(cloth_visual_profiles[row['id']])
    initial = {'cloths_vertices_m': {row['id']: deepcopy(row['initial_vertices_m']) for row in cloths}}
    scene = {'schema_version': 'laundry-engineering-scene-v1',
        'description': 'Task-owned live towels in pinned laundry v3; no robot manipulation claim.',
        'workroom': environment_reference(), 'robot': {'enabled': False},
        'cloth': {'enabled': False}, 'cloths': cloths, 'objects': [], 'fixtures': [],
        'goal_markers': [], 'workbench': {'enabled': False}, 'workcell_details': {'enabled': False},
        'table': {'position_m': [.65, -.55, .875], 'half_extents_m': [.9, .45, .025], 'friction': .45},
        'physics': {'dt_s': physics['physics_dt_s'], 'control_dt_s': .02,
            'gravity_m_s2': [0., 0., -9.81], 'backend': 'native_isaacsim_physx_gpu'},
        'isaacsim': physics, 'laundry_task_provenance': {'asset_candidate': candidate,
            'parent_environment_unchanged': True, 'control_mode': 'engineering_local_grasp',
            'material_calibrated': False, 'labels_added': False}}
    task = {'schema_version': TASK_SCHEMA, 'id': task_id, 'task_id': task_id,
        'title_zh': TITLES[task_id], 'difficulty': (CURRENT_TASK_IDS.index(task_id)+1 if task_id in CURRENT_TASK_IDS else TASK_IDS.index(task_id)+1), 'cloth': True,
        'episode_seed': seed, 'capabilities': deepcopy(CAPABILITIES),
        'initial_state': initial, 'asset_metadata': metadata,
        'public_rules': rules, 'public_goal': {'operation': task_id, 'frame_id': 'world_m_z_up',
            'description': rules['instruction_zh'], 'target_center_xy_m': metadata['target_center_xy_m']},
        'evaluation': {'schema': 'laundry_evaluation.v1', 'task_id': task_id,
            'metadata': deepcopy(metadata), 'physics_dt_s': physics['physics_dt_s'],
            'thresholds': thresholds(task_id, rule_profile), 'public_rules_sha256': digest(rules)},
        'steps': [{'id': task_id, 'instruction_zh': rules['instruction_zh']}],
        'recommended_budgets': {'sim_time_s': 120., 'wall_time_s': 1800., 'max_actions': 6000},
        'training_policy_included': False, 'physics_accepted': False,
        'task_policy_success_claimed': False, 'robot_manipulation_supported': False,
        'qualification_scope': 'Task-only native engineering inputs; explicitly local cloth driving followed by complete release.',
        'known_limitations': ['Material, thickness, mass and solver values are simulation priors requiring native evidence.',
            'No robot installation or ordinary robot episode is supplied by this release.',
            'Cloth contact force and tension are not measured; support uses gravity, actual cloth geometry and unforced stability.',
            'Static room towels and appliances remain decorative and are excluded from task scores.']}
    if rule_profile != LEGACY_PROFILE:
        task['evaluation']['rule_profile'] = rule_profile
    if public_material_contract is not None:
        task['public_material_contract'] = public_material_contract
    if robot_layout is not None:
        from bench.isaacsim.release import checked
        from scenes.workrooms.observations import camera_specs
        from .robot_layout import robot_inputs, STAND_ROOT, STAND_MANIFEST_SHA256, workspace_contract

        if installation is None:
            checked(STAND_ROOT/'manifest.json', STAND_MANIFEST_SHA256)
            robot, robot_initial, ready = robot_inputs()
            sides = ['left', 'right']
            fixture = {'id': 'robot_stand', 'asset_id': 'laundry_dual_robot_stand',
                       'asset_root': str(STAND_ROOT), 'position_m': [0., 0., 0.]}
            stand_sha = STAND_MANIFEST_SHA256
            workspace = workspace_contract()
        else:
            robot, robot_initial, ready = configured.robot_inputs(configuration_id,
                version=installation.get('layout_version', 1))
            sides = installation['active_sides']
            fixture = installation['fixture']
            stand_sha = installation['stand_manifest']['sha256']
            workspace = configured.workspace_contract(configuration_id,
                version=installation.get('layout_version', 1))
        initial.update(deepcopy(robot_initial))
        description = load_json(Path(scene['workroom']['dependency_manifest_path']))
        cameras = camera_specs(description, robot=robot)
        if len(sides) == 2:
            wrist = cameras.pop('wrist')
            for side in sides:
                cameras[side+'_wrist'] = {**deepcopy(wrist), 'mount_link': side+'_hand_flange',
                    'title_zh': ('左' if side == 'left' else '右')+'腕部相机'}
        cameras['overview'].update(eye=[2.35, -2.10, 2.20], target=[.75, -.55, 1.03], focal_length_mm=24.)
        cameras['task'].update(eye=[1.75, -1.30, 1.65], target=[.75, -.55, .94], focal_length_mm=34.)
        scene.update(schema_version='laundry-robot-scene-v1',
            description='同一Agent控制官方左右SHARPA双FR3，操作原生柔性毛巾；洗衣房v3保持固定。',
            robot=robot, cameras=cameras, named_cloth_initialization='author-only',
            fixtures=[fixture])
        scene['isaacsim'] = {**deepcopy(physics), **deepcopy(ROBOT_PHYSICS_OPTIONS)}
        scene['laundry_task_provenance'].update(control_mode=mode, robot_layout_id=robot_layout,
            ready_design=ready, stand_manifest_sha256=stand_sha,
            robot_physics_options=deepcopy(ROBOT_PHYSICS_OPTIONS), cloth_physics_options=deepcopy(physics),
            shared_controller_unchanged=True, same_agent_simultaneous_control=True,
            local_cloth_attachments=False, cloth_state_writes_after_native_initialization=False)
        task.update(task_kind='single_agent', active_sides=sides,
            capabilities=capabilities_for(mode), robot_manipulation_supported=True,
            qualification_scope='Ordinary single-Agent dual-arm task candidate with native cloth and read-only evidence; grasp and complete operation qualification remain separate.',
            workspace_contract=workspace)
        task['evaluation']['control_mode'] = mode
        task['known_limitations'][1] = 'Dual-arm robot input and read-only native cloth evidence are supplied; full hand-cloth operation is not qualified.'
        task['known_limitations'].append('The original downward near-cloth IK samples do not prove a collision-free grasp or folding trajectory.')
        if installation is not None:
            scene['robot_configuration'] = deepcopy(installation)
            scene['description'] = '单Agent使用独立发布的机器人安装操作原生衣物；洗衣房v3保持固定。'
            task['robot_configuration_id'] = configuration_id
            task['qualification_scope'] = 'Versioned ordinary robot installation; contact, workspace and complete manipulation qualification are independently recorded.'
            task['known_limitations'][1] = 'The selected robot and native cloth observer are supplied; initialization and interface tests do not qualify complete manipulation.'
    if support_materials is not None:
        scene['isaacsim']['workroom_surface_materials'] = support_materials
    return {'scene': deepcopy(scene), 'task': bind_task_rules(task), 'initial_state': deepcopy(initial)}
