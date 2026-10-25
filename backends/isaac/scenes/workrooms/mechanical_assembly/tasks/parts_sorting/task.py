"""Three independent sorting parts and three fixed, open, shape-marked trays."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import trimesh

from bench.isaacsim.release import digest
from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from ...task_scene import empty_task_scene, DEFAULT_STAND_ROOT
from scenes.workrooms.mechanical_assembly.referee import bind_task_rules
from ...rules import public_rules, thresholds

TASK_ID = 'parts_sorting'
ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/parts_sorting/v2')
PART_NAMES = ('block', 'cylinder', 'ring')
CONTACT = {'friction': .4, 'contact_offset_m': .0001, 'rest_offset_m': 0.,
           'collision_mesh_coordinate_scale': 1000.}


def _bind_sdf_ring(spec, source, ring_asset_root):
    """Select an explicit source-preserving ring without changing the task reset."""
    candidate = load_task_asset('sorting_ring_sdf1024', ring_asset_root, verify_hashes=True)
    material = {'static_friction': CONTACT['friction'], 'dynamic_friction': CONTACT['friction'],
                'restitution': 0., 'friction_combine': 'average'}
    if (candidate.get('backend') != 'usd_reference'
            or any(candidate.get(key) != source[key] for key in
                   ('mass_kg', 'center_of_mass_m', 'inertia_diag_kg_m2'))
            or [candidate.get('bounds_min_m'), candidate.get('bounds_max_m')] != source['aabb_m']
            or candidate.get('material_prior') != material
            or candidate.get('contact_offset_m') != CONTACT['contact_offset_m']
            or candidate.get('rest_offset_m') != CONTACT['rest_offset_m']
            or candidate.get('source_dependencies_sha256', {}).get(source['mesh_path']) != digest(source['mesh_path'])
            or candidate.get('source_dependencies_sha256', {}).get(source['manifest_path']) != source['manifest_sha256']):
        raise ValueError('SDF ring must preserve the selected source geometry and physics')
    result = deepcopy(spec)
    result.pop('friction')
    result.pop('collision_mesh_coordinate_scale')
    result.update(asset_id='sorting_ring_sdf1024', asset_root=str(Path(ring_asset_root).resolve()),
                  usd_physics_policy='preserve_authored_v1', usd_physics_material_fallback=material)
    return result


def build_task_inputs(*, asset_root=ASSET_ROOT, stand_root=DEFAULT_STAND_ROOT, solver_type=None,
                      ring_asset_root=None):
    scene, initial, ready = empty_task_scene(stand_root, solver_type=solver_type)
    goals = {}; geometry = {}
    for name, y in zip(PART_NAMES, (-.17, 0., .17)):
        asset = load_task_asset('sorting_'+name, asset_root, verify_hashes=True)
        low = asset['aabb_m'][0][2]
        # Freeze actual collision vertices, including authored part transforms.
        # The evaluator only transforms these numbers, without loading meshes.
        points = []
        for part in asset['collision_parts']:
            mesh = trimesh.load(part['mesh_path'], force='mesh', process=False)
            qx,qy,qz,qw = part['quaternion_xyzw']
            r = trimesh.transformations.quaternion_matrix([qw,qx,qy,qz])[:3,:3]
            points.append((np.asarray(mesh.vertices)*part['scale']) @ r.T + part['position_m'])
        geometry[name] = np.unique(np.concatenate(points), axis=0).tolist()
        spec = {'id': name, 'asset_id': 'sorting_'+name,
            'asset_root': str(Path(asset_root).resolve()),
            'position_m': [.27, y, .782-low], 'quaternion_wxyz': [1., 0., 0., 0.], **CONTACT}
        if name == 'ring' and ring_asset_root is not None:
            spec = _bind_sdf_ring(spec, asset, ring_asset_root)
        scene['objects'].append(spec)
        scene['fixtures'].append({'id': 'tray_'+name, 'asset_id': 'sorting_tray_'+name,
            'asset_root': str(Path(asset_root).resolve()), 'position_m': [.55, y, .78], **CONTACT})
        goals[name] = {'tray_id': 'tray_'+name, 'center_xy_m': [.55, y],
            'inner_half_extents_xy_m': [.065, .065], 'floor_top_z_m': .786,
            'expected_center_z_m': .786-low,
            'object_actor': '/World/Objects/'+name,
            'support_collider': '/World/Fixtures/tray_'+name+'/collision_000'}
    scene['scenario'] = {'family': 'mechanical_assembly', 'task_id': TASK_ID, 'seed': 0}
    task = {'schema_version': 'mechanical-assembly-task-v1', 'id': TASK_ID, 'task_id': TASK_ID,
        'title_zh': '大零件分类', 'difficulty': 'guided_single_arm', 'cloth': False,
        'steps': [{'id': 'sort', 'instruction_zh': '把三个零件分别放入对应形状图标的浅盘，松手稳定。'}],
        'initial_state': deepcopy(initial), 'goals': goals,
        'public_goal': {'operation': TASK_ID, 'frame_id': 'world',
                        'description': '将三个零件分别放入对应形状图标的浅盘，松手后保持稳定。'},
        'public_rules': public_rules(TASK_ID),
        'evaluation': {**thresholds(TASK_ID), 'must_be_released': True,
            'robot_actor_prefix': '/World/Robot',
            'actors': {**{name: goal['object_actor'] for name, goal in goals.items()},
                       **{'tray_'+name: goal['support_collider'] for name, goal in goals.items()}},
            'initial_objects': {obj['id']: {key: deepcopy(obj[key]) for key in ('position_m','quaternion_wxyz')}
                                for obj in scene['objects']},
            'required_support_pairs': [[name, 'tray_'+name] for name in PART_NAMES],
            'geometry_vertices_local_m': geometry,
            'geometry_scope': 'All source collision mesh vertices, with authored part transforms and exact duplicate removal; frozen as local numerical geometry.'},
        'physics_accepted': False, 'task_policy_success_claimed': False,
        'recommended_budgets': {'sim_time_s': 120., 'wall_time_s': 1800., 'max_actions': 6000}}
    bind_task_rules(task)
    return {'scene': scene, 'task': task, 'initial_state': initial, 'ready_design': ready}


def contact_probe_inputs(*, asset_root=ASSET_ROOT, offset_xy_m=(0., 0.), release_height_m=.03,
                         solver_type=None):
    """Reset-only free drops into real bins; no sorting/manipulation score."""
    value = build_task_inputs(asset_root=asset_root, solver_type=solver_type)
    for obj in value['scene']['objects']:
        goal = value['task']['goals'][obj['id']]
        obj['position_m'] = [goal['center_xy_m'][0]+offset_xy_m[0],
            goal['center_xy_m'][1]+offset_xy_m[1], goal['expected_center_z_m']+release_height_m]
    value['diagnostic'] = {'kind': 'gravity_bin_contact_probe',
        'reset_only': True, 'offset_xy_m': list(offset_xy_m),
        'release_height_m': release_height_m, 'manipulation_score_claimed': False}
    return value
