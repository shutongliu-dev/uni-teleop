"""Build an explicit, unqualified three-installation index from frozen inputs."""
from copy import deepcopy
import json
from pathlib import Path

from bench.isaacsim.release import digest, validate
from .rules import CURRENT_TASK_IDS
from .robot_configurations import CONFIGURATION_IDS
from bench.paths import load_json


def material_identity(scene, task):
    """World placement may differ; material geometry and physics may not."""
    keys = ('id', 'rest_vertices_m', 'triangles', 'rest_area_m2', 'mass_kg',
            'physical_thickness_m', 'fixed_vertices', 'dynamic_model', 'friction')
    referee = deepcopy(task['public_rules']['referee'])
    referee.pop('physical_actions', None)
    referee['phases'].pop('settling', None)
    return {'physics': scene['isaacsim'], 'environment': scene['workroom'],
        'cloths': [{key: deepcopy(row[key]) for key in keys if key in row}
                   for row in task['asset_metadata']['cloths']],
        'thresholds': task['evaluation']['thresholds'], 'referee_except_operator': referee}


def configuration_row(task_id, releases, *, default_configuration_id='dual_opposed'):
    if task_id not in CURRENT_TASK_IDS or set(releases) != set(CONFIGURATION_IDS):
        raise ValueError('One current task and all three independent installations required')
    if default_configuration_id not in CONFIGURATION_IDS:
        raise ValueError('Unknown default installation')
    options = {}
    identity = None
    for identifier in CONFIGURATION_IDS:
        root = Path(releases[identifier]).resolve()
        manifest = validate(root)
        scene, task = [load_json(root/name) for name in ('scene.json', 'task.json')]
        if (manifest['task_id'] != task_id or manifest.get('configuration_id') != identifier
                or task.get('robot_configuration_id') != identifier):
            raise ValueError('Frozen configuration identity differs from the requested index')
        value = material_identity(scene, task)
        if identity is not None and value != identity:
            raise ValueError('Configurations must preserve the same task material, physical profile and numeric rules')
        identity = value
        layout = task['workspace_contract']
        metadata = task['asset_metadata']
        options[identifier] = {
            'source_release': {'path': str(root), 'manifest_sha256': digest(root/'release_manifest.json')},
            'release_schema': manifest['schema'], 'release_schema_version': manifest['schema_version'],
            'robot_requirements': {'hand_id': scene['robot']['hand_id'],
                'assembly_version': scene['robot']['assembly_version'],
                'articulations': len(task['active_sides']), 'active_sides': task['active_sides']},
            'layout': {key: deepcopy(layout[key]) for key in ('robot_layout_id', 'placements',
                'reference_surface', 'frame', 'units', 'control_scope')},
            'objects': {'roles': [row['id'] for row in metadata['cloths']],
                'model': 'live_surface_deformable', 'material_profile': metadata.get('material_profile'),
                'cloths': [{key: deepcopy(row[key]) for key in ('id', 'mass_kg', 'physical_thickness_m')}
                           for row in metadata['cloths']],
                'properties_basis': 'simulation_priors', 'material_calibrated': False},
            'rules': {'implementation': 'scenes.workrooms.laundry_room.rules:configuration_rules',
                      'contract': task['public_rules']},
            'qualification': {'status': 'prepared_unvalidated', 'native_preflight': False,
                'agent_bench_interface': False, 'workspace_ik': False, 'workspace_clearance': False,
                'robot_manipulation': False, 'physics_accepted': False,
                'scope': 'Prepared immutable installation only; layout, native interface, contact and operation evidence are independently bound.'}}
    if len({row['source_release']['manifest_sha256'] for row in options.values()}) != 3:
        raise ValueError('Robot choices must have distinct immutable identities')
    return {'family': 'laundry_room', 'task_id': task_id,
        'default_configuration_id': default_configuration_id,
        'base_release': deepcopy(options[default_configuration_id]['source_release']),
        'configurations': options}
