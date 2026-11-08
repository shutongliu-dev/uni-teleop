"""Only nominal object geometry is projected out of private reset inputs."""
from copy import deepcopy

from bench.protocol.workroom_reset import PublicWorkroomResetCase, SCOPE
from . import TASK_FAMILIES, selection
from .common import canonical_hash


def project_reset_case(task, projection):
    case = task.get('reset_case')
    if case is None:
        return projection
    if case.get('schema') == 'living_room_reset_case.v1':
        from scenes.workrooms.living_room.randomization.public import project_reset_case as living
        return living(task, projection)
    selection(task['id'], case['seed'], case['configuration_id'])
    if (case.get('schema') != 'workroom_reset_case.v1' or case['task_id'] != task['id']
            or case['family'] != TASK_FAMILIES[task['id']]
            or case['case_sha256'] != canonical_hash({k: v for k, v in case.items() if k != 'case_sha256'})):
        raise ValueError('Private reset input identity differs')
    if projection is None or projection.get('task_id') != task['id']:
        raise ValueError('Seeded ordinary task lacks a matching public task contract')
    result = deepcopy(projection)
    result['reset_case'] = PublicWorkroomResetCase.model_validate({
        'schema': 'workroom_public_reset_case.v1', 'family': case['family'], 'task_id': task['id'],
        'seed': case['seed'], 'configuration_id': case['configuration_id'],
        'case_sha256': case['case_sha256'], 'static_geometry': case['static_geometry'],
        'scope': SCOPE}).model_dump(mode='json', by_alias=True)
    return result
