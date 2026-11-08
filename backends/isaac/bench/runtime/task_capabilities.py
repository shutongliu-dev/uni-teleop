"""Admission checks shared by the public runner and its native worker."""
from collections.abc import Mapping
from copy import deepcopy
import re


class TaskCapabilityError(ValueError):
    """The selected task explicitly excludes ordinary robot episodes."""


def engineering_capabilities(task):
    """Project only explicit public opt-out fields, never arbitrary task data."""
    capabilities = task.get('capabilities')
    if not isinstance(capabilities, Mapping) or capabilities.get('robot_episodes') is not False:
        return None
    public = {'robot_episodes': False}
    for name in ('task_inputs', 'engineering_cloth_probe', 'robot_grasping_qualified'):
        if type(capabilities.get(name)) is bool:
            public[name] = capabilities[name]
    entrypoint = capabilities.get('engineering_entrypoint')
    if isinstance(entrypoint, str) and re.fullmatch(r'scripts\.validation\.[A-Za-z_][A-Za-z0-9_]*', entrypoint):
        public['engineering_entrypoint'] = entrypoint
    return public


def robot_episode_unavailable_reason(task):
    capabilities = engineering_capabilities(task)
    if capabilities is None:
        return None
    entrypoint = capabilities.get('engineering_entrypoint')
    guidance = (f'Use the engineering script: python -m {entrypoint}.' if entrypoint else
                'Use the engineering validation entrypoint documented for this task.')
    return f'This task does not support ordinary robot /episodes. {guidance}'


def project_engineering_task(task, task_id):
    capabilities = engineering_capabilities(task)
    if capabilities is None:
        return None
    result = {'task_id': task_id, 'task_kind': 'engineering',
        'instruction': task['title_zh']+'。'+'；'.join(step['instruction_zh'] for step in task['steps']),
        'capabilities': capabilities, 'robot_episodes': False,
        'engineering_entrypoint': capabilities.get('engineering_entrypoint'),
        'create_endpoint': None, 'runnable': False,
        'availability_scope': 'Engineering task inputs only; ordinary robot execution is unavailable.',
        'unavailable_reason': robot_episode_unavailable_reason(task)}
    from bench.protocol.task_referee import contract_sha256, validated_public_contract
    contract = validated_public_contract(task)
    if contract is not None:
        if contract['task_id'] != task_id:
            raise ValueError('Engineering discovery identity differs from its public rules')
        result.update(public_rules=contract, public_rules_sha256=contract_sha256(contract))
    return result


def restrict_engineering_interfaces(catalog, tasks):
    """Keep mixed robot releases usable, with explicit task-specific opt-outs."""
    excluded = {name: engineering_capabilities(task) for name, task in tasks.items()}
    excluded = {name: value for name, value in excluded.items() if value is not None}
    if not excluded:
        return catalog
    result = deepcopy(catalog)
    result['task_capabilities'] = excluded
    if len(excluded) != len(tasks):
        return result
    entrypoints = {value.get('engineering_entrypoint') for value in excluded.values()}
    result.update(robot_episodes=False, capabilities={'robot_episodes': False},
        engineering_entrypoint=next(iter(entrypoints)) if len(entrypoints) == 1 else None,
        availability_scope='Selected tasks support engineering inputs only; no ordinary robot action interface is available.')
    reason = robot_episode_unavailable_reason(next(iter(tasks.values())))
    result['geometry_policies'] = []
    for row in result['interfaces']:
        row.update(available=False, backend_available=False, unavailable_reason=reason,
                   client_command=None, executor=None)
        for key in ('supported_policy_ids', 'backend_policy_ids', 'policies', 'model_actions', 'action_kinds', 'tools'):
            if key in row:
                row[key] = []
    common = result['shared_contract']
    common.update(robot_episodes=False, create_endpoint=None)
    for key in tuple(common):
        if key.endswith('_endpoint') and key != 'function_schema_endpoint':
            common[key] = None
    return result


def require_robot_episode_capability(task):
    """Honor an explicit opt-out before constructing a robot or simulator.

    Historical tasks have no capability declaration and keep their existing
    admission behavior. The error is public: do not include the task document,
    evaluator details, or arbitrary release-provided explanatory text.
    """
    reason = robot_episode_unavailable_reason(task)
    if reason is not None:
        raise TaskCapabilityError(reason)
