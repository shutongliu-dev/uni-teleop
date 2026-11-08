"""Select a frozen public episode case without reading private task inputs."""
from copy import deepcopy
import re

from bench.protocol.living_reset import PublicResetCase
from bench.protocol.workroom_reset import PublicWorkroomResetCase
from bench.protocol.laundry_reset import PublicLaundryResetCase

from .bench_http import BenchHTTPError
from .scene_tasks import published_task

RESET_SCHEMA = 'living_room_public_reset_case.v1'
WORKROOM_RESET_SCHEMA = 'workroom_public_reset_case.v1'
LAUNDRY_RESET_SCHEMA = 'laundry_public_reset_case.v1'
RESET_MODELS = {RESET_SCHEMA: PublicResetCase, WORKROOM_RESET_SCHEMA: PublicWorkroomResetCase,
                LAUNDRY_RESET_SCHEMA: PublicLaundryResetCase}


def checked_reset_case(value, *, seed, configuration_id=None, task_id=None):
    """Validate public fields and their episode identity; no host asset access."""
    try:
        model = RESET_MODELS.get(value.get('schema')) if isinstance(value, dict) else None
        if model is None:
            raise ValueError('Unsupported reset schema')
        parsed = model.model_validate(value)
    except (ValueError, TypeError):
        raise BenchHTTPError('Invalid public frozen reset case', category='invalid_response') from None
    if (type(seed) is not int or parsed.seed != seed
            or (configuration_id is not None and parsed.configuration_id != configuration_id)):
        raise BenchHTTPError('Public reset case differs from requested seed or configuration',
                             category='invalid_response')
    if isinstance(parsed, (PublicWorkroomResetCase, PublicLaundryResetCase)):
        if task_id is not None and parsed.task_id != task_id:
            raise BenchHTTPError('Public reset case differs from the selected task', category='invalid_response')
    else:
        operation = {'remote_operation': 'remote_return_to_holder', 'water_pouring': 'pour_and_return',
                     'storage_box': 'collect_and_close'}.get(task_id)
        if task_id is not None and (operation is None or parsed.static_geometry.goal.operation != operation):
            raise BenchHTTPError('Public reset geometry differs from the selected task', category='invalid_response')
    return parsed.model_dump(mode='json', exclude_none=True)


def select_episode_case(bench, selection, published, seed):
    """Return (selected public task, effective seed) before episode creation.

    Only the explicitly advertised finite reset schema defaults None to zero.
    Historical seeded goals, including task0, retain their previous behavior.
    """
    original = published.get('reset_case')
    if original is None:
        return deepcopy(published), seed
    if not isinstance(original, dict) or original.get('schema') not in RESET_MODELS:
        raise BenchHTTPError('Unsupported public reset-case schema', category='invalid_response')
    effective = 0 if seed is None else seed
    if type(effective) is not int or not 0 <= effective <= 9:
        raise ValueError('Frozen ordinary reset seed must be an integer in 0..9')
    configuration = selection.configuration_id
    baseline = checked_reset_case(original, seed=original.get('seed'), configuration_id=configuration,
                                  task_id=published['task_id'])
    release_hash = published.get('task_release_sha256')
    if not isinstance(release_hash, str) or not re.fullmatch('[0-9a-f]{64}', release_hash):
        raise BenchHTTPError('Frozen seed discovery omitted its release identity', category='invalid_response')
    kwargs = {'task_id': published['task_id'], 'seed': effective}
    if configuration is not None:
        kwargs['configuration_id'] = configuration
    selected, board = published_task(selection, bench.list_tasks(**kwargs))
    case = checked_reset_case(selected.get('reset_case'), seed=effective, configuration_id=configuration,
                              task_id=published['task_id'])
    if (board or selected.get('task_kind') != published.get('task_kind')
            or selected.get('task_release_sha256') != release_hash
            or selected.get('configuration_id') != published.get('configuration_id')
            or case['configuration_id'] != baseline['configuration_id']
            or case['schema'] != baseline['schema']
            or (case['schema'] == LAUNDRY_RESET_SCHEMA
                and case['parent_release_sha256'] != baseline['parent_release_sha256'])
            or (case['seed'] == baseline['seed'] and case != baseline)
            or selected.get('episode_seed', effective) != effective):
        raise BenchHTTPError('Frozen seed discovery changed the selected contract identity',
                             category='invalid_response')
    selected['reset_case'] = case
    return selected, effective
