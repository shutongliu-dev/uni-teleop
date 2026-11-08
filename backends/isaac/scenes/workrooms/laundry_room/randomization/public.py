"""Whitelist projection of finite case identity, without private cloth nodes."""
from copy import deepcopy

from bench.protocol.laundry_reset import (PublicLaundryResetCase, SCOPE, VARIATION,
                                         PARAMETERIZED_SCOPE, PARAMETERIZED_VARIATION)
from . import RESET_SCHEMA, PARAMETERIZED_VERSION, selection
from .common import canonical_hash


def project_reset_case(task, projection):
    case = task.get('reset_case')
    if (not isinstance(case, dict) or case.get('schema') != RESET_SCHEMA
            or case.get('case_sha256') != canonical_hash({k: v for k, v in case.items() if k != 'case_sha256'})):
        raise ValueError('Unbound private laundry reset case')
    selection(case.get('task_id'), case.get('seed'), case.get('configuration_id'))
    if (not isinstance(projection, dict) or projection.get('task_id') != case['task_id']
            or case.get('family') != 'laundry_room' or case['task_id'] != task.get('id')
            or case['configuration_id'] != task.get('robot_configuration_id')
            or case['seed'] != task.get('episode_seed')
            or case.get('parent_release_sha256') != case.get('parent_release', {}).get('manifest_sha256')):
        raise ValueError('Laundry case and runtime task identities disagree')
    public = {key: case[key] for key in ('family', 'task_id', 'configuration_id', 'seed',
               'case_sha256', 'case_fingerprint', 'parent_release_sha256')}
    parameterized = case.get('generator_version') == PARAMETERIZED_VERSION
    public.update(schema='laundry_public_reset_case.v1',
        variation=PARAMETERIZED_VARIATION if parameterized else VARIATION,
        scope=PARAMETERIZED_SCOPE if parameterized else SCOPE)
    result = deepcopy(projection)
    result['reset_case'] = PublicLaundryResetCase.model_validate(public).model_dump(mode='json')
    return result
