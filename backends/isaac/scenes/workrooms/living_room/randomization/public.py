"""Strict public projection of static case dimensions/goals, never body truth."""
from copy import deepcopy

from bench.protocol.living_reset import PublicResetCase

from .common import canonical_hash, selection
from ..public_task import select

SCHEMAS = {
    'remote_operation': {'dimensions_m': {'remote': ['number'], 'holder_opening': ['number']},
                         'goal': {'operation': str}},
    'water_pouring': {'dimensions_m': {'kettle_inner_profile': [['number']], 'cup_inner_profile': [['number']]},
                     'goal': {'operation': str, 'return_position_m': ['number'], 'target_interval_ml': ['number']}},
    'storage_box': {'dimensions_m': {'item': ['number']},
                    'goal': {'operation': str, 'tray_inside_xy_m': [['number']], 'tray_floor_z_m': 'number'}},
}
for _schema in SCHEMAS.values():
    _schema['bodies'] = [{'object_id': str, 'local_bounds_m': [['number']], 'nominal_mass_kg': 'number'}]
SCHEMAS['water_pouring']['handle_geometry'] = {'capsule_radius_m': 'number'}


def project_reset_case(task, projection):
    case = task['reset_case']
    selection(task['id'], case['seed'], case['configuration_id'])
    if (case['schema'] != 'living_room_reset_case.v1' or case['task_id'] != task['id']
            or case['case_sha256'] != canonical_hash({k: v for k, v in case.items() if k != 'case_sha256'})
            or not isinstance(projection, dict) or projection.get('task_id') != task['id']):
        raise ValueError('Public reset-case identity differs from the loaded task')
    result = deepcopy(projection)
    result['reset_case'] = PublicResetCase.model_validate({'schema': 'living_room_public_reset_case.v1', 'seed': case['seed'],
        'configuration_id': case['configuration_id'], 'case_sha256': case['case_sha256'],
        'static_geometry': select(case['static_geometry'], SCHEMAS[task['id']]),
        'scope': 'Static authored dimensions and goals for this episode; live object poses and native evidence remain private.'
    }).model_dump(mode='json', by_alias=True)
    return result
