"""Frozen development matrices; never a server EvaluationPlan or leaderboard run."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import math
import os
import stat
from pathlib import Path
import time
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from typing import get_args
from bench.paths import load_json
from bench.protocol.schemas import TerminationReason
from bench.protocol.allocations import (MAX_EPISODE_SIM_TIME_S, MAX_EPISODE_WALL_TIME_S,
                                        MAX_EPISODE_ACTIONS)
from scenes.manipulation import DEFAULT_TASK_RELEASE

from .bench_http import BenchHTTPClient, BenchHTTPError
from .outcomes import classify_report

PROJECT = Path(__file__).resolve().parents[2]
PRESETS = {
    'joints_action': ('joints', 'action'),
    'fingertip_action': ('fingertip', 'action'),
    'fingertip_tools': ('fingertip', 'tools'),
    'fingertip_code': ('fingertip', 'code'),
}
# Keep the historical four-mode default and callers of PRESETS unchanged.
# The free agent is a separate study, never a fifth cell in that comparison.
MODE_PRESETS = {**PRESETS, 'free_tool_agent': ('hybrid', 'free')}
RUNTIME_PINS = ('runtime_code_sha256', 'robot_asset_sha256', 'physics_profile_sha256',
                'sensor_config_sha256', 'execution_environment_sha256')
CORE_ARTIFACTS = ('run_report.json', 'agent_config.json', 'episode_ready.json')
# Exact pre-migration runner inventory (be0cee5), interpreted as archived data.
# Never import a frozen runner to discover its dependencies.
_LEGACY_CLIENT_SOURCE_DEPENDENCIES = ('bench/agent/schemas.py', 'bench/agent/validation.py',
    'bench/geometry.py', 'scripts/maintenance/freeze_project_sources.py')
_SCORING_CLIENT_SOURCE_DEPENDENCIES = frozenset({
    'bench/protocol/allocations.py', 'bench/protocol/task_scoring.py'})
Token = Annotated[str, Field(strict=True, min_length=1, max_length=160, pattern=r'^[A-Za-z0-9_.:-]+$')]
Digest = Annotated[str, Field(strict=True, pattern=r'^[a-f0-9]{64}$')]


class Matrix(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    protocol: Literal['agent_development_matrix_v1'] = 'agent_development_matrix_v1'
    description: Annotated[str, Field(strict=True, max_length=4096)] = 'Development interface comparison; no leaderboard claim.'
    bench_url: str = 'http://127.0.0.1:8765'
    parallel_bench_urls: list[Annotated[str, Field(strict=True)]] = Field(default_factory=list, max_length=8)
    task_release: str = str(DEFAULT_TASK_RELEASE)
    tasks: list[Token] = Field(default_factory=lambda: ['01_cube_push'], min_length=1)
    modes: list[Literal['joints_action', 'fingertip_action', 'fingertip_tools', 'fingertip_code', 'free_tool_agent']] = Field(default_factory=lambda: list(PRESETS), min_length=1)
    visual_inputs: list[Literal['rgb', 'rgbd']] = Field(default_factory=lambda: ['rgbd'], min_length=1)
    tactile_inputs: list[Literal['on', 'off']] = Field(default_factory=lambda: ['on'], min_length=1)
    seeds: list[Annotated[int, Field(strict=True, ge=0, le=2**32-1)]] = Field(default_factory=lambda: [0], min_length=1)
    repeat_ids: list[Token] = Field(default_factory=lambda: ['r0'], min_length=1)
    robot: Token = 'sharpa_w01_right'
    model: Token = 'gpt-6-astra'
    provider: Literal['gateway', 'micu'] = 'gateway'
    model_retries: Annotated[int, Field(strict=True, ge=0, le=3)] | None = None
    model_retry_policy: Literal['provider_default_v2', 'legacy_v1'] = 'provider_default_v2'
    reasoning_effort: Literal['max'] = 'max'
    max_decisions: Annotated[int, Field(strict=True, ge=1, le=256)] = 2
    max_actions: Annotated[int, Field(strict=True, ge=1, le=MAX_EPISODE_ACTIONS)] = 4
    max_tool_calls_per_decision: Annotated[int, Field(strict=True, ge=1, le=32)] = 8
    max_program_actions: Annotated[int, Field(strict=True, ge=1, le=8)] = 3
    max_output_tokens: Annotated[int, Field(strict=True, ge=1, le=131072)] | None = 16384
    sim_budget: Annotated[float, Field(strict=True, gt=0, le=MAX_EPISODE_SIM_TIME_S)] | None = 12.
    wall_budget: Annotated[float, Field(strict=True, gt=0, le=MAX_EPISODE_WALL_TIME_S)] | None = 600.
    model_timeout: Annotated[float, Field(strict=True, gt=0, le=1800)] | None = 300.
    http_timeout: Annotated[float, Field(strict=True, gt=0, le=1800)] | None = 300.
    program_timeout: Annotated[float, Field(strict=True, gt=0)] | None = 120.
    expected_runtime_pins: dict[Literal['runtime_code_sha256', 'robot_asset_sha256', 'physics_profile_sha256',
        'sensor_config_sha256', 'execution_environment_sha256'], Digest] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_matrix(self):
        dimensions = [self.tasks, self.modes, self.visual_inputs, self.tactile_inputs, self.seeds, self.repeat_ids]
        if any(len(values) != len(set(values)) for values in dimensions):
            raise ValueError('Matrix dimensions must not contain duplicates')
        if 'free_tool_agent' in self.modes and self.modes != ['free_tool_agent']:
            raise ValueError('free_tool_agent requires a separate matrix from fixed interface modes')
        if math.prod(map(len, dimensions)) > 256:
            raise ValueError('Development suite is limited to 256 slots; split larger studies explicitly')
        if not isinstance(self.task_release, str) or not Path(self.task_release).is_absolute():
            raise ValueError('task_release must be an absolute path')
        origins = []
        for value in [self.bench_url, *self.parallel_bench_urls]:
            try:
                url = urlsplit(value)
                valid = (url.scheme in ('http', 'https') and url.hostname and url.port != 0
                         and not url.username and not url.password and not url.query and not url.fragment
                         and url.path in ('', '/'))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError('bench URLs must be HTTP origins without credentials')
            # A spelling-only origin difference must not schedule two workers onto one service.
            host = url.hostname.lower()
            if host in ('localhost', '127.0.0.1', '::1'):
                host = 'loopback'
            origins.append((url.scheme.lower(), host, url.port or (443 if url.scheme == 'https' else 80)))
        if self.parallel_bench_urls:
            if origins[0] != origins[1]:
                raise ValueError('bench_url must identify the first parallel worker')
            if len(origins[1:]) != len(set(origins[1:])):
                raise ValueError('Parallel workers must use distinct service origins')
        return self


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return load_json(Path(path))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_pins():
    from .runner import CLIENT_SOURCE_DEPENDENCIES
    paths = {p for folder in ('bench', 'agent', 'scenes') for p in (PROJECT/folder).rglob('*.py')}
    paths.add(PROJECT/'scripts/isaac_python.sh')
    paths.update(PROJECT/name for name in CLIENT_SOURCE_DEPENDENCIES)
    return {str(path.relative_to(PROJECT)): digest(path) for path in sorted(paths)}


def release_pins(matrix):
    path = Path(matrix.task_release)/'release_manifest.json'
    manifest = read(path)
    if (manifest.get('schema') == 'robot_board_game_task_release.v1'
            or manifest.get('task_kind') in ('board_game', 'turn_game')):
        raise ValueError('Development suite supports ordinary episodes; '
                         'use agent.model_client.function_runner for authenticated games')
    tasks = {row['id'] for row in manifest['tasks']}
    if not set(matrix.tasks) <= tasks:
        raise ValueError('Requested tasks do not belong to this single release')
    evaluator = manifest['files']['evaluator/sequence_evaluator.py']['sha256']
    for name, row in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Invalid release manifest path')
        if digest(path.parent/relative) != row['sha256']:
            raise ValueError('Release file differs from its frozen manifest')
    return {'task_release_sha256': digest(path), 'evaluator_sha256': evaluator}


def _requires_policy(control_mode):
    return control_mode in ('fingertip', 'hybrid')


def _slot_track(slot):
    return ('free_tool_agent' if slot['control_mode'] == 'hybrid' else
            'system0' if slot['control_mode'] == 'fingertip' else 'direct_joints')


_TASK_INVENTORY_KEYS = {'policy': 'policies_by_task', 'free_interface': 'free_interfaces_by_task'}


def _task_inventory_value(inventory, name, task):
    """Select the task's frozen binding, never another task's first descriptor."""
    mapped_name = _TASK_INVENTORY_KEYS[name]
    if mapped_name not in inventory:
        return inventory.get(name)
    if name in inventory:
        raise ValueError(f'Inventory cannot contain both {name} and {mapped_name}')
    values = inventory[mapped_name]
    if not isinstance(values, dict) or task not in values:
        raise ValueError(f'Inventory {mapped_name} must bind every selected task')
    return values[task]


def _store_task_inventory(inventory, name, values):
    # Retain the archived shared representation only when all bindings agree.
    descriptors = list(values.values())
    if all(value == descriptors[0] for value in descriptors):
        inventory[name] = descriptors[0]
    else:
        inventory[_TASK_INVENTORY_KEYS[name]] = values


def _validate_inventory_bindings(inventory, tasks):
    if not isinstance(inventory, dict):
        raise ValueError('Invalid public service inventory')
    rows = inventory.get('tasks')
    if (not isinstance(rows, list) or any(not isinstance(row, dict) or
            not isinstance(row.get('task_id'), str) for row in rows)
            or sorted(row['task_id'] for row in rows) != sorted(tasks)):
        raise ValueError('Public service inventory must bind exactly the selected tasks')
    for name, mapped_name in _TASK_INVENTORY_KEYS.items():
        if mapped_name in inventory:
            values = inventory[mapped_name]
            if (name in inventory or not isinstance(values, dict) or set(values) != set(tasks)
                    or any(not isinstance(value, dict) for value in values.values())):
                raise ValueError(f'Inventory {mapped_name} must exclusively bind exactly the selected tasks')


def _scoring_contract(plan, task):
    return next(row.get('scoring_contract') for row in plan['service_inventory']['tasks']
                if row['task_id'] == task)


def _score_summaries(rows):
    """Never pool different tasks, units or objectives into one numeric score."""
    groups = defaultdict(list)
    for row in rows:
        if row['task_score_valid']:
            groups[(row['slot']['task'], row['task_score_unit'], row['task_score_objective'])].append(
                row['task_score'])
    return [{'task_id': task, 'score_unit': unit, 'objective': objective,
             'valid_episodes': len(scores), 'mean': sum(scores) / len(scores),
             'min': min(scores), 'max': max(scores)}
            for (task, unit, objective), scores in sorted(groups.items())]


def _expected_policy(plan, slot):
    return (_task_inventory_value(plan['service_inventory'], 'policy', slot['task'])
            if _requires_policy(slot['control_mode']) else None)


def _validate_free_inventory(inventory, robot):
    """Require the advertised mixed-control handoff, including on plan reload."""
    from .runner import select_geometry_policy
    from .free_interface import require_free_interface
    tasks = [row['task_id'] for row in inventory['tasks']]
    _validate_inventory_bindings(inventory, tasks)
    for task in tasks:
        policy = _task_inventory_value(inventory, 'policy', task)
        if not isinstance(policy, dict) or select_geometry_policy([policy], robot) != policy:
            raise ValueError('free_tool_agent requires a pinned compatible classical policy')
        contract = _task_inventory_value(inventory, 'free_interface', task)
        if not isinstance(contract, dict):
            raise ValueError('free_tool_agent requires a pinned public interface contract')
        # Reconstruct this task's catalog projection for the same admission
        # validation as the online runner, including its exact controller hash.
        normalized = require_free_interface({
            'protocol': 'agent_interface_catalog_v1',
            'interfaces': [contract.get('free_tool_agent')],
            'geometry_policies': [policy], 'shared_contract': contract.get('shared_contract'),
        }, policy)
        if normalized != contract:
            raise ValueError('free_tool_agent pinned public interface contract is inconsistent')


def service_inventory(matrix):
    """Read public inventories only; no episode or model is created."""
    from .runner import select_geometry_policy
    with BenchHTTPClient(matrix.bench_url, timeout_s=matrix.http_timeout) as client:
        tasks = client.list_tasks()
        if not isinstance(tasks, list) or any(not isinstance(row, dict) or not isinstance(row.get('task_id'), str) for row in tasks):
            raise ValueError('Invalid public task inventory')
        selected = [row for row in tasks if row['task_id'] in matrix.tasks]
        if sorted(row['task_id'] for row in selected) != sorted(matrix.tasks):
            raise ValueError('Every requested task must be uniquely advertised by this service')
        inventory = {'tasks': sorted(selected, key=lambda row: row['task_id'])}
        if any(_requires_policy(MODE_PRESETS[mode][0]) for mode in matrix.modes):
            policies = {task: select_geometry_policy(client.list_policies(task_id=task), matrix.robot)
                        for task in sorted(matrix.tasks)}
            _store_task_inventory(inventory, 'policy', policies)
        else:
            inventory['policy'] = None
        if matrix.modes == ['free_tool_agent']:
            from .free_interface import require_free_interface
            contracts = {task: require_free_interface(client.list_interfaces(task_id=task), policies[task])
                         for task in sorted(matrix.tasks)}
            _store_task_inventory(inventory, 'free_interface', contracts)
            _validate_free_inventory(inventory, matrix.robot)
    return inventory


def expand_slots(matrix):
    slots = []
    for index, (task, mode, visual, tactile, seed, repeat) in enumerate(itertools.product(
            matrix.tasks, matrix.modes, matrix.visual_inputs, matrix.tactile_inputs, matrix.seeds, matrix.repeat_ids)):
        control, interaction = MODE_PRESETS[mode]
        slots.append({'slot_id': f'slot_{index:04d}', 'task': task, 'mode': mode, 'control_mode': control,
                      'interaction_mode': interaction, 'visual_input': visual, 'tactile_input': tactile,
                      'seed': seed, 'repeat_id': repeat})
        if matrix.parallel_bench_urls:
            worker = index % len(matrix.parallel_bench_urls)
            slots[-1].update(worker_id=f'worker_{worker:02d}', bench_url=matrix.parallel_bench_urls[worker])
    return slots


def create_plan(config, output):
    matrix = Matrix.model_validate(config)
    output = Path(output)
    if output.exists():
        raise ValueError('Plan output directory must not already exist')
    release, service, sources = release_pins(matrix), service_inventory(matrix), source_pins()
    for url in matrix.parallel_bench_urls[1:]:
        if service_inventory(matrix.model_copy(update={'bench_url': url})) != service:
            raise ValueError('Parallel services must advertise identical selected tasks and controller')
    plan = {'protocol': 'agent_development_suite_plan_v1', 'created_at_utc': now(),
            'matrix': matrix.model_dump(), 'slots': expand_slots(matrix), 'release_pins': release,
            'service_inventory': service, 'source_pins': sources,
            'scope': 'Frozen development matrix; NOT server EvaluationPlan; NOT leaderboard eligible.',
            'seed_warning': 'repeat_id and seed numbers do not prove different initialization or generalization.'}
    plan['plan_sha256'] = canonical_hash(plan)
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in sources.items():
        destination = output/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        value = (PROJECT/name).read_bytes()
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError('Source changed while freezing plan; preserve this incomplete directory and make a new plan')
        destination.write_bytes(value)
    save(output/'plan.json', plan)
    return plan


def load_plan(path):
    plan = read(path)
    if not isinstance(plan, dict) or plan.get('protocol') != 'agent_development_suite_plan_v1':
        raise ValueError('Expected a development suite plan, not an EvaluationPlan')
    if plan.get('plan_sha256') != canonical_hash({key:value for key,value in plan.items() if key != 'plan_sha256'}):
        raise ValueError('Plan digest mismatch')
    # The default used to mean one bounded Micu retry. Never reinterpret an
    # archived plan as the new unlimited policy merely by loading newer code.
    matrix = Matrix.model_validate({'model_retry_policy':'legacy_v1', **plan['matrix']})
    _validate_inventory_bindings(plan['service_inventory'], matrix.tasks)
    if any(_requires_policy(MODE_PRESETS[mode][0]) for mode in matrix.modes):
        from .runner import select_geometry_policy
        for task in matrix.tasks:
            select_geometry_policy([_task_inventory_value(plan['service_inventory'], 'policy', task)], matrix.robot)
    if matrix.modes == ['free_tool_agent']:
        _validate_free_inventory(plan['service_inventory'], matrix.robot)
    if plan['slots'] != expand_slots(matrix):
        raise ValueError('Plan slots must match the entire frozen matrix')
    if not isinstance(plan['source_pins'], dict) or not plan['source_pins']:
        raise ValueError('Plan needs source pins')
    for name, expected in plan['source_pins'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Invalid source snapshot path')
        if digest(Path(path).parent/'source_snapshot'/relative) != expected:
            raise ValueError('Frozen source snapshot digest mismatch')
    return plan, matrix


class PinMismatch(ValueError):
    pass


def model_retry_contract(matrix):
    """Resolve the frozen plan generation without changing its archived bytes."""
    if matrix.provider == 'gateway':
        return {'model_retry_policy':'legacy_gateway_v1',
                'model_retries':0 if matrix.model_retries is None else matrix.model_retries}
    legacy = matrix.model_retry_policy == 'legacy_v1'
    value = {'model_retry_policy':('bounded_same_endpoint_model_request_v1' if legacy
                                  else 'wait_10s_same_endpoint_v2'),
             'model_retries':(1 if legacy else None) if matrix.model_retries is None else matrix.model_retries,
             'model_timeout_scope':'shared_call_budget' if legacy else 'per_attempt'}
    if not legacy:
        value['model_retry_delay_s'] = 10
    return value


def preflight(plan, matrix):
    if source_pins() != plan['source_pins']:
        raise PinMismatch('project_sources')
    if release_pins(matrix) != plan['release_pins']:
        raise PinMismatch('task_release')
    if service_inventory(matrix) != plan['service_inventory']:
        raise PinMismatch('public_task_or_policy_inventory')


def runner_args(matrix, slot, output, credentials=None):
    values = matrix.model_dump(include={'bench_url', 'robot', 'model', 'provider', 'model_retries', 'reasoning_effort', 'max_decisions',
        'max_actions', 'max_tool_calls_per_decision', 'max_program_actions', 'max_output_tokens',
        'sim_budget', 'wall_budget', 'model_timeout', 'http_timeout', 'program_timeout'})
    values.update({key:slot[key] for key in ('task', 'seed', 'visual_input', 'tactile_input', 'control_mode', 'interaction_mode')})
    values['bench_url'] = slot.get('bench_url', matrix.bench_url)
    values['model_retry_policy'] = model_retry_contract(matrix)['model_retry_policy']
    return argparse.Namespace(**values, credentials=credentials, output=Path(output))


def _client_source_layout(source_pins):
    """Select one pinned archive layout, without path aliases or code execution."""
    from .runner import CLIENT_SOURCE_DEPENDENCIES
    roots = [root for root in ('interface/model_client', 'agent/model_client')
             if any(name.startswith(root + '/') for name in source_pins)]
    if len(roots) != 1:
        raise PinMismatch('client_source_root_inventory')
    root = roots[0]
    dependencies = (_LEGACY_CLIENT_SOURCE_DEPENDENCIES if root == 'interface/model_client'
                    else CLIENT_SOURCE_DEPENDENCIES)
    if root == 'agent/model_client' and not _SCORING_CLIENT_SOURCE_DEPENDENCIES.intersection(source_pins):
        # Historical plans predate these modules and retain their exact SDK
        # inventory. A new/partial inventory must include both dependencies.
        dependencies = tuple(name for name in dependencies if name not in _SCORING_CLIENT_SOURCE_DEPENDENCIES)
    return root, dependencies


def _config_errors(plan, matrix, slot, output, config_sha256):
    errors = []
    settings_path = output/'agent_config.json'
    if not settings_path.exists():
        errors.append('missing_agent_config')
    else:
        settings = read(settings_path)
        if not isinstance(settings, dict):
            return ['invalid_agent_config_shape']
        if canonical_hash(settings) != config_sha256:
            errors.append('agent_config_sha256')
        if settings.get('controller') != _expected_policy(plan, slot):
            errors.append('agent_config_controller')
        if (slot['mode'] == 'free_tool_agent' and settings.get('interface_inventory') !=
                _task_inventory_value(plan['service_inventory'], 'free_interface', slot['task'])):
            errors.append('agent_config_interface_inventory')
        contract = model_retry_contract(matrix)
        expected_retries = contract['model_retries']
        legacy = matrix.model_retry_policy == 'legacy_v1'
        # Archives without provider selection used gateway. Missing retry
        # fields inherit only an explicitly legacy plan's historical defaults.
        if settings.get('provider', 'gateway') != matrix.provider:
            errors.append('agent_config_provider')
        recorded_retries = settings.get('model_retries', 1 if legacy and matrix.provider=='micu' else 0)
        if (recorded_retries != expected_retries or type(recorded_retries) is not type(expected_retries)
                or (not legacy and 'model_retries' not in settings)):
            errors.append('agent_config_model_retries')
        historical_policy = ('bounded_same_endpoint_model_request_v1' if matrix.provider=='micu'
                             else 'legacy_gateway_v1')
        recorded_policy = settings.get('model_retry_policy', historical_policy if legacy else None)
        if recorded_policy != contract['model_retry_policy']:
            errors.append('agent_config_model_retry_policy')
        gateway = settings.get('gateway')
        if matrix.provider == 'micu':
            if not isinstance(gateway, dict) or gateway.get('protocol') != 'responses':
                errors.append('agent_config_provider_protocol')
            else:
                endpoint = gateway.get('endpoint')
                try:
                    url = urlsplit(endpoint)
                    hostname = url.hostname or ''
                    micu = (url.scheme == 'https' and not url.username and not url.password
                            and not url.query and not url.fragment and url.path.endswith('/v1/responses')
                            and (hostname == 'micuapi.ai' or hostname.endswith('.micuapi.ai')))
                except (ValueError, TypeError, AttributeError):
                    micu = False
                if not micu:
                    errors.append('agent_config_micu_endpoint')
                if gateway.get('retry_policy', historical_policy if legacy else None) != contract['model_retry_policy']:
                    errors.append('agent_config_gateway_retry_policy')
                if not legacy:
                    if gateway.get('retry_delay_s') != 10:
                        errors.append('agent_config_retry_delay')
                    if gateway.get('timeout_scope') != 'per_attempt':
                        errors.append('agent_config_retry_timeout_scope')
                    if ('max_retries' not in gateway or gateway['max_retries'] != expected_retries
                            or type(gateway['max_retries']) is not type(expected_retries)):
                        errors.append('agent_config_gateway_model_retries')
                elif gateway.get('timeout_scope', 'shared_call_budget') != 'shared_call_budget':
                    errors.append('agent_config_retry_timeout_scope')
        elif isinstance(gateway, dict) and gateway.get('protocol') != 'chat/completions':
            errors.append('agent_config_provider_protocol')
        for key, expected_value in {'model':matrix.model, 'reasoning_effort':matrix.reasoning_effort,
                **{key:slot[key] for key in ('control_mode','interaction_mode','visual_input','tactile_input')},
                'max_decisions':matrix.max_decisions, 'max_actions':matrix.max_actions,
                'max_tool_calls_per_decision':matrix.max_tool_calls_per_decision,
                'max_program_actions':matrix.max_program_actions,
                'max_output_tokens':matrix.max_output_tokens,
                'model_timeout_s':matrix.model_timeout, 'http_timeout_s':matrix.http_timeout}.items():
            if key not in settings or settings[key] != expected_value:
                errors.append('agent_config_'+key)
        program_limits = settings.get('program_limits', {})
        if (not isinstance(program_limits, dict) or 'max_runtime_s' not in program_limits
                or program_limits['max_runtime_s'] != matrix.program_timeout):
            errors.append('agent_config_program_timeout')
        archived = settings.get('client_sources_sha256')
        if not isinstance(archived, dict) or not archived:
            errors.append('missing_client_source_pins')
        else:
            try:
                source_root, dependencies = _client_source_layout(plan['source_pins'])
            except PinMismatch as exc:
                errors.append(str(exc))
                return errors
            archived_dependencies = {name.removeprefix('dependencies/') for name in archived
                                     if name.startswith('dependencies/')}
            # Ordinary legacy runs historically checked each supplied source;
            # do not retroactively demand files absent from that archive format.
            # New runs must include every dependency extracted into the SDK.
            if (archived_dependencies - set(dependencies) or
                    (source_root == 'agent/model_client' and
                     (archived_dependencies != set(dependencies) or
                      any(name not in plan['source_pins'] for name in dependencies)))):
                errors.append('client_source_dependency_inventory')
            for name, value in archived.items():
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    errors.append('invalid_client_source_path')
                    continue
                project_name = (name.removeprefix('dependencies/') if name.startswith('dependencies/')
                                else source_root+'/'+name)
                copied = output/'client_source'/relative
                if plan['source_pins'].get(project_name) != value or not copied.is_file() or digest(copied) != value:
                    errors.append('client_source_'+project_name)
    return errors


def verify_run(plan, matrix, slot, report, output):
    """Post-run attestation, never retroactive preregistration of observed pins."""
    errors = []
    result = report.get('result') if isinstance(report, dict) else None
    spec = result.get('spec') if isinstance(result, dict) else None
    ready_path = output/'episode_ready.json'
    ready = read(ready_path) if ready_path.exists() else None
    if not isinstance(ready, dict):
        errors.append('missing_or_invalid_episode_ready')
        ready = {}
    if not isinstance(spec, dict):
        errors.append('missing_result_spec')
        spec = ready.get('spec') if isinstance(ready.get('spec'), dict) else {}
    if not isinstance(ready.get('spec'), dict) or ready['spec'] != spec:
        errors.append('episode_ready_result_spec_mismatch')
    episode_id = spec.get('episode_id')
    if (not isinstance(episode_id, str) or not episode_id
            or ready.get('episode_id') != episode_id
            or not isinstance(report, dict) or report.get('episode_id') != episode_id):
        errors.append('episode_identity_mismatch')
    expected = {**plan['release_pins'], **matrix.expected_runtime_pins, 'task_id':slot['task'],
                'robot_id':matrix.robot, 'seed':slot['seed'], 'scene_variant':'original_v1',
                'observation_profile':'vision_proprio', 'execution_mode':'pause_physics_during_agent',
                'track':_slot_track(slot)}
    for name, value in expected.items():
        if spec.get(name) != value:
            errors.append('spec_'+name)
    expected_policy = _expected_policy(plan, slot)
    if spec.get('system0') != expected_policy:
        errors.append('controller_descriptor')
    if any(not isinstance(spec.get(name), str) or len(spec[name]) != 64 for name in (*RUNTIME_PINS, 'scene_sha256')):
        errors.append('missing_runtime_pins')
    budgets = spec.get('budgets', {})
    for key, value in {'max_actions':matrix.max_actions, 'max_model_calls':matrix.max_decisions,
                       'sim_time_s':matrix.sim_budget, 'wall_time_s':matrix.wall_budget}.items():
        if key not in budgets or budgets[key] != value:
            errors.append('budget_'+key)
    errors.extend(_config_errors(plan, matrix, slot, output, spec.get('agent_config_sha256')))
    contract = _scoring_contract(plan, slot['task'])
    scoring = result.get('scoring') if isinstance(result, dict) else None
    if contract is not None or scoring is not None:
        from bench.protocol.task_scoring import TerminalTaskScoring
        try:
            if contract is None or not TerminalTaskScoring.model_validate(scoring).matches(contract):
                errors.append('terminal_scoring_contract')
        except (ValueError, TypeError):
            errors.append('terminal_scoring_contract')
    return {'passed': not errors, 'errors': errors, 'actual_spec': spec,
            'runtime_pre_pinned_fields': sorted(matrix.expected_runtime_pins),
            'note': 'Other actual runtime pins are grouped separately, not claimed preregistered.'}


def artifact_pins(output):
    """Detect changes while validating a closed regular-file archive.

    This binds local file/directory identities before and after validation; it
    does not claim an atomic snapshot against a privileged concurrent writer.
    """
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PinMismatch('duplicate_artifact_key')
            result[key] = value
        return result

    output = Path(output).absolute()
    if output.resolve() != output or not output.is_dir():
        raise PinMismatch('archive_root_alias_or_not_directory')

    def identity(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
                info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def inventory():
        root_info = output.lstat()
        if not stat.S_ISDIR(root_info.st_mode):
            raise PinMismatch('archive_root_changed_type')
        records = {'.': identity(root_info)}
        files = set()
        for path in output.rglob('*'):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise PinMismatch('artifact_alias')
            name = path.relative_to(output).as_posix()
            records[name] = identity(info)
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise PinMismatch('nonregular_or_linked_artifact')
            files.add(name)
        return records, files

    initial, actual = inventory()

    def read_file(name):
        # O_NONBLOCK prevents a post-inventory FIFO swap from blocking open;
        # O_NOFOLLOW and fstat reject aliases/types before reading any bytes.
        fd = os.open(output/name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                    identity(info) != initial.get(name)):
                raise PinMismatch('artifact_identity_changed_before_read')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                raw = stream.read()
            if identity(os.fstat(fd)) != initial[name]:
                raise PinMismatch('artifact_identity_changed_during_read')
            return raw
        finally:
            os.close(fd)

    raw = read_file('artifact_hashes.json')
    before = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(manifest, dict) or not set(CORE_ARTIFACTS) <= set(manifest):
        raise PinMismatch('incomplete_artifact_manifest')
    for name, expected in manifest.items():
        relative = Path(name)
        if (relative.is_absolute() or relative.as_posix() != name or
                any(part in ('', '.', '..') for part in name.split('/'))):
            raise PinMismatch('invalid_artifact_path')
        if (not isinstance(expected, str) or len(expected) != 64 or
                any(c not in '0123456789abcdef' for c in expected)):
            raise PinMismatch('invalid_artifact_digest')
    if actual != set(manifest) | {'artifact_hashes.json'} or 'artifact_hashes.json' in manifest:
        raise PinMismatch('artifact_inventory_mismatch')
    pins = {'artifact_hashes.json': before}
    for name, expected in manifest.items():
        if hashlib.sha256(read_file(name)).hexdigest() != expected:
            raise PinMismatch('artifact_manifest_content_mismatch')
        pins[name] = expected
    if hashlib.sha256(read_file('artifact_hashes.json')).hexdigest() != before or inventory() != (initial, actual):
        raise PinMismatch('artifact_archive_changed_during_check')
    return pins

def _setup_failure_candidate(report):
    """Only a known runner create failure with no client-observed episode/work."""
    zeros = ('model_call_attempts', 'model_responses', 'primitive_attempts',
             'submitted_actions', 'completed_actions', 'tool_calls', 'point_queries', 'read_calls')
    empty = {'model_input_observation_sequences': [], 'model_response_metadata': [],
             'decision_status_counts': {}, 'interface_selection_counts': {},
             'program_status_counts': {}, 'primitive_termination_reason_counts': {}}
    return (report.get('schema') in ('online_visual_agent_run.v6', 'online_visual_agent_run.v7')
            and 'episode_id' in report and report['episode_id'] is None
            and 'result' in report and report['result'] is None
            and report.get('termination') == 'client_error'
            and report.get('error_type') == 'BenchHTTPError'
            and 'model_error' in report and report['model_error'] is None
            and (report.get('task_success') is False if report.get('schema') == 'online_visual_agent_run.v6'
                 else report.get('task_success') is None and report.get('task_outcome') == 'unknown')
            and report.get('online_model_decision_loop_exercised') is False
            and report.get('program_multiple_actions_exercised') is False
            and all(type(report.get(k)) is int and report[k] == 0 for k in zeros)
            and all(k in report and report[k] == v for k,v in empty.items()))


def setup_failure_pins(output):
    """Exact failed-create archive, including every copied client source."""
    def unique(pairs):
        result = {}
        for k,v in pairs:
            if k in result:
                raise PinMismatch('duplicate_setup_artifact_key')
            result[k] = v
        return result
    output = Path(output).absolute()
    if output.resolve() != output or not output.is_dir():
        raise PinMismatch('setup_archive_root_alias_or_not_directory')
    actual = set()
    for p in output.rglob('*'):
        if p.is_symlink():
            raise PinMismatch('setup_artifact_alias')
        if p.is_dir():
            continue
        if not stat.S_ISREG(p.stat().st_mode) or p.stat().st_nlink != 1:
            raise PinMismatch('nonregular_or_linked_setup_artifact')
        actual.add(p.relative_to(output).as_posix())
    manifest_path = output/'artifact_hashes.json'
    before = digest(manifest_path)
    manifest = load_json(manifest_path, object_pairs_hook=unique)
    settings = read(output/'agent_config.json')
    for name in ('run_report.json', 'agent_config.json', 'create_request.json'):
        load_json(output/name, object_pairs_hook=unique)
    sources = settings.get('client_sources_sha256')
    if not isinstance(sources,dict) or not sources:
        raise PinMismatch('missing_setup_client_sources')
    expected = {'run_report.json','agent_config.json','create_request.json'} | {'client_source/'+name for name in sources}
    verification = settings.get('source_snapshot_verification')
    if verification is not None:
        if verification != 'runtime_imports.json':
            raise PinMismatch('invalid_setup_source_verification_reference')
        evidence = load_json(output/verification, object_pairs_hook=unique)
        if not isinstance(evidence, dict) or evidence.get('runtime_imports_verified') is not True:
            raise PinMismatch('invalid_setup_source_verification')
        expected.add(verification)
    if not isinstance(manifest,dict) or set(manifest) != expected:
        raise PinMismatch('incomplete_or_extra_setup_artifact_manifest')
    if actual != expected | {'artifact_hashes.json'}:
        raise PinMismatch('setup_artifact_inventory')
    pins = {'artifact_hashes.json': before}
    for name,wanted in manifest.items():
        relative = Path(name)
        if (relative.is_absolute() or relative.as_posix() != name or
                any(part in ('','.', '..') for part in name.split('/'))):
            raise PinMismatch('invalid_setup_artifact_path')
        if (not isinstance(wanted,str) or len(wanted)!=64 or
                any(c not in '0123456789abcdef' for c in wanted) or digest(output/relative)!=wanted):
            raise PinMismatch('setup_artifact_content_mismatch')
        pins[name] = wanted
    if digest(manifest_path) != before:
        raise PinMismatch('setup_manifest_changed_during_check')
    return pins


def verify_setup_failure(plan, matrix, slot, report, output):
    from .runner import AGENT_VERSION
    errors = []
    if not _setup_failure_candidate(report):
        errors.append('not_zero_work_episode_setup_failure')
    errors.extend(_config_errors(plan,matrix,slot,output,report.get('config_sha256')))
    settings = read(output/'agent_config.json')
    try:
        source_root, dependencies = _client_source_layout(plan['source_pins'])
    except PinMismatch:
        pass  # _config_errors already rejects an absent or mixed client root.
    else:
        expected_sources = {Path(name).relative_to(source_root).as_posix():value
                            for name,value in plan['source_pins'].items()
                            if name.startswith(source_root+'/') and name.endswith('.py')
                            and (report.get('schema') != 'online_visual_agent_run.v6'
                                 or Path(name).parent == Path(source_root))}
        expected_sources.update({'dependencies/'+name:plan['source_pins'][name]
                                for name in dependencies if name in plan['source_pins']})
        if any(name not in plan['source_pins'] for name in dependencies):
            errors.append('setup_client_dependency_inventory')
        if settings.get('client_sources_sha256') != expected_sources:
            errors.append('setup_client_source_inventory')
    expected_policy = _expected_policy(plan, slot)
    for key,value in {'controller':expected_policy,'model':matrix.model,'reasoning_effort':matrix.reasoning_effort,
            **{key:slot[key] for key in ('control_mode','interaction_mode','visual_input','tactile_input')}}.items():
        if report.get(key) != value or settings.get(key) != value:
            errors.append('setup_config_'+key)
    create = read(output/'create_request.json')
    expected = {'task_id':slot['task'],'robot_id':matrix.robot,'seed':slot['seed'],
        'scene_variant':'original_v1','agent_id':'online_visual_'+slot['control_mode']+'_'+slot['interaction_mode']+'_agent',
        'agent_version':AGENT_VERSION,'agent_config_sha256':report.get('config_sha256'), 'track':_slot_track(slot),
        'observation_profile':'vision_proprio','budgets':{'max_actions':matrix.max_actions,'max_model_calls':matrix.max_decisions,
            'sim_time_s':matrix.sim_budget,'wall_time_s':matrix.wall_budget,'max_action_sim_s':5.0}}
    if expected_policy:
        expected.update(policy_id=expected_policy['policy_id'],policy_version=expected_policy['version'])
    if create != expected or settings.get('budgets') != expected['budgets']:
        errors.append('setup_create_request')
    return {'passed':not errors,'errors':errors,'actual_spec':None,'episode_verified':False,
            'model_not_run':True if not errors else None,
            'note':'Client setup failed before an episode identity/model/action; not a valid recorded episode. Server resource cleanup is a separate check.'}


def _run_worker(plan, matrix, slots, root, credentials, run_one):
    """One sequential queue per independent service; never overlap its episodes."""
    for slot in slots:
        record_path = root/'attempts'/(slot['slot_id']+'.json')
        if record_path.exists():
            continue  # Even crashed, rejected or failed attempts are never silently replaced.
        output = root/'runs'/slot['slot_id']
        worker_matrix = matrix.model_copy(update={'bench_url':slot.get('bench_url', matrix.bench_url)})
        record = {'protocol':'agent_development_suite_attempt_v1', 'plan_sha256':plan['plan_sha256'],
                  'slot':slot, 'status':'started', 'started_at_utc':now(), 'run_directory':str(output),
                  'bench_url':worker_matrix.bench_url, 'worker_id':slot.get('worker_id', 'worker_00'),
                  'model_accounting':'client_observed_not_server_verified', 'leaderboard_eligible':False}
        save(record_path, record)
        start = time.monotonic()
        phase = 'preflight'
        try:
            preflight(plan, worker_matrix)
            record['source_pins_before_sha256'] = canonical_hash(plan['source_pins'])
            if output.exists():
                raise PinMismatch('unrecorded_output_directory_already_exists')
            phase = 'runner'
            args = runner_args(matrix, slot, output, credentials)
            args.expected_spec_pins = {**plan['release_pins'], **matrix.expected_runtime_pins}
            if slot['mode'] == 'free_tool_agent':
                args.expected_spec_pins['track'] = 'free_tool_agent'
            args.expected_controller = _expected_policy(plan, slot)
            if slot['mode'] == 'free_tool_agent':
                args.expected_interface_inventory = _task_inventory_value(
                    plan['service_inventory'], 'free_interface', slot['task'])
            report = run_one(args)
            if not isinstance(report, dict):
                raise ValueError('Runner did not return a report')
            record['runner_report'] = report
            phase = 'verification'
            setup_failed = _setup_failure_candidate(report)
            record['artifact_pins'] = setup_failure_pins(output) if setup_failed else artifact_pins(output)
            if read(output/'run_report.json') != report:
                raise PinMismatch('runner_report_return_vs_archive')
            record['verification'] = (verify_setup_failure(plan,matrix,slot,report,output) if setup_failed
                                      else verify_run(plan,matrix,slot,report,output))
            record['status'] = ('episode_setup_failed' if setup_failed else 'finished') if record['verification']['passed'] else 'pin_mismatch'
            if record['status'] == 'episode_setup_failed':
                record.update(failure_phase='episode_setup', model_not_run=True)
        except Exception as error:
            record.update(status=('pin_mismatch' if isinstance(error, PinMismatch) else
                                  'service_unavailable' if isinstance(error,BenchHTTPError) and phase=='preflight' else
                                  'preflight_failed' if phase=='preflight' else 'runner_error'),
                          error_type=type(error).__name__, failure_phase=phase)
            if isinstance(error,PinMismatch):
                record['verification']={'passed':False,'errors':[str(error)]}
            # Exception messages may contain model credentials or response bodies.
        if phase != 'preflight':
            try:
                after = source_pins()
                record['source_pins_after_sha256'] = canonical_hash(after)
                if after != plan['source_pins']:
                    raise PinMismatch('source_changed_during_run')
            except Exception:
                record['status'] = 'pin_mismatch'
                record.pop('model_not_run', None)
                verification = record.setdefault('verification', {'errors':[]})
                verification['passed'] = False
                verification.pop('model_not_run', None)
                verification['errors'].append('source_changed_or_unreadable_after_run')
        record.update(finished_at_utc=now(), actual_attempt_wall_s=time.monotonic()-start)
        save(record_path, record)
        print(json.dumps({'slot_id':slot['slot_id'], 'worker_id':record['worker_id'],
                          'status':record['status']}), flush=True)


def run_plan(path, *, credentials=None, run_one=None):
    plan, matrix = load_plan(path)
    if matrix.provider == 'micu' and matrix.model_retry_policy == 'legacy_v1':
        raise ValueError('Legacy Micu plans remain read-only under this runner; create a new plan '
                         'to select the 10-second retry policy. Do not resume or rewrite the old queue.')
    root = Path(path).parent
    (root/'attempts').mkdir(exist_ok=True)
    (root/'runs').mkdir(exist_ok=True)
    if run_one is None:
        from .runner import run as run_one
    with (root/'.run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('A suite runner already owns this plan') from None
        queues = defaultdict(list)
        for slot in plan['slots']:
            queues[slot.get('worker_id', 'worker_00')].append(slot)
        if len(queues) == 1:
            _run_worker(plan, matrix, plan['slots'], root, credentials, run_one)
        else:
            # The plan lock stays held until every queue stops. Each thread owns independent
            # HTTP/model clients and disjoint run/attempt directories. No model or native
            # worker objects are shared; the only shared data is the immutable frozen plan.
            with ThreadPoolExecutor(max_workers=len(queues), thread_name_prefix='bench-suite') as executor:
                futures = [executor.submit(_run_worker, plan, matrix, slots, root, credentials, run_one)
                           for slots in queues.values()]
                for future in futures:
                    future.result()
    return report_plan(path)


def _sum(rows, getter):
    values = [getter(row) for row in rows]
    return sum(values) if all(isinstance(v, (float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in values) else None


def model_statistics(report):
    metadata = report.get('model_response_metadata')
    calls = report.get('model_call_attempts')
    complete = (isinstance(metadata,list) and type(calls) is int and calls >= 0
                and len(metadata) == calls and all(isinstance(row,dict) for row in metadata))
    # Retried calls expose only the final response usage. An absent earlier
    # response cannot be assigned zero tokens, even when a later retry succeeds.
    usage_complete = complete and all(type(row.get('attempts', 1)) is int
                                     and row.get('attempts', 1) == 1 for row in metadata)
    def total(path, *, integer=False):
        if not complete or (path[0] == 'usage' and not usage_complete):
            return None
        values=[]
        for row in metadata:
            value=row
            actual_path = path
            if row.get('protocol') == 'responses' and path[0] == 'usage':
                names = {'prompt_tokens':'input_tokens', 'completion_tokens':'output_tokens',
                         'completion_tokens_details':'output_tokens_details'}
                actual_path = tuple(names.get(name, name) for name in path)
            for name in actual_path:
                value=value.get(name) if isinstance(value,dict) else None
            if (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
                    or value < 0 or (integer and type(value) is not int)):
                return None
            values.append(value)
        return sum(values)
    return {'model_latency_client_s':total(('duration_s',)),
            'gateway_prompt_tokens':total(('usage','prompt_tokens'),integer=True),
            'gateway_completion_tokens':total(('usage','completion_tokens'),integer=True),
            'gateway_total_tokens':total(('usage','total_tokens'),integer=True),
            'gateway_reasoning_tokens':total(('usage','completion_tokens_details','reasoning_tokens'),integer=True),
            'response_metadata_covers_all_attempts':complete,
            'token_usage_covers_all_http_attempts':usage_complete,
            'response_metadata_count':len(metadata) if isinstance(metadata,list) else None}


def report_plan(path):
    plan, matrix = load_plan(path)
    root = Path(path).parent
    rows, groups = [], defaultdict(list)
    for slot in plan['slots']:
        attempt_path = root/'attempts'/(slot['slot_id']+'.json')
        attempt = read(attempt_path) if attempt_path.exists() else None
        if attempt and (attempt.get('plan_sha256') != plan['plan_sha256'] or attempt.get('slot') != slot):
            raise ValueError('Attempt does not match its frozen plan slot')
        run_directory = root/'runs'/slot['slot_id']
        report = (attempt or {}).get('runner_report', {})
        if not isinstance(report, dict):
            report = {}
        archive_errors=[]
        fresh_verification=None
        if attempt and (attempt.get('runner_report') is not None or attempt.get('artifact_pins') is not None
                        or attempt.get('status')=='finished'):
            try:
                setup_failed = attempt.get('status') == 'episode_setup_failed'
                fresh_pins=setup_failure_pins(run_directory) if setup_failed else artifact_pins(run_directory)
                if fresh_pins != attempt.get('artifact_pins'):
                    archive_errors.append('archived_artifacts_changed')
                archived_report=read(run_directory/'run_report.json')
                if archived_report != report:
                    archive_errors.append('run_report_differs_from_attempt')
                if not isinstance(archived_report,dict):
                    raise ValueError('Invalid archived report shape')
                if setup_failed:
                    checked=verify_setup_failure(plan,matrix,slot,archived_report,run_directory)
                    wanted=canonical_hash(plan['source_pins'])
                    if (not checked['passed'] or attempt.get('verification') != checked
                            or attempt.get('source_pins_before_sha256') != wanted
                            or attempt.get('source_pins_after_sha256') != wanted):
                        archive_errors.append('setup_failure_attestation_invalid')
                else:
                    fresh_verification=verify_run(plan,matrix,slot,archived_report,run_directory)
                    # A cached success is only a claim. Recheck its archived
                    # config/spec and recorded source attestations every time.
                    # Existing failed attempts retain their original status.
                    if attempt.get('status') == 'finished':
                        wanted=canonical_hash(plan['source_pins'])
                        if not fresh_verification['passed']:
                            archive_errors.append('normal_attestation_invalid')
                        if canonical_hash(attempt.get('verification')) != canonical_hash(fresh_verification):
                            archive_errors.append('cached_verification_differs_from_fresh')
                        source_fields=('source_pins_before_sha256', 'source_pins_after_sha256')
                        if any(name not in attempt for name in source_fields):
                            archive_errors.append('source_attestation_missing')
                        if any(name in attempt and attempt[name] != wanted for name in source_fields):
                            archive_errors.append('source_pin_attestation_invalid')
                # Even invalid cached metadata must not hide the archived
                # native outcome or make the whole planned denominator crash.
                report=archived_report
            except (OSError,ValueError,TypeError):
                archive_errors.append('archived_artifacts_missing_or_invalid')
        result = report.get('result') or {}
        if not isinstance(result, dict):
            result = {}
        verification = (attempt or {}).get('verification', {})
        if not isinstance(verification, dict):
            verification = {}
        # Cohort identity comes from the report bound to the archive, never
        # from an independently editable cached verification.actual_spec.
        spec = result.get('spec') or {}
        if not isinstance(spec, dict):
            spec = {}
        actual_pins = {key:spec.get(key) for key in (*RUNTIME_PINS, 'task_release_sha256', 'evaluator_sha256',
            'agent_config_sha256', 'agent_version', 'execution_mode', 'observation_profile', 'scene_variant', 'system0')}
        condition = {'mode':slot['mode'], 'control_mode':slot['control_mode'], 'interaction_mode':slot['interaction_mode'],
                     'visual_input':slot['visual_input'], 'tactile_input':slot['tactile_input'],
                     'robot':matrix.robot, 'model':matrix.model, 'reasoning_effort':matrix.reasoning_effort,
                     'provider':matrix.provider,
                     'model_protocol':'responses' if matrix.provider=='micu' else 'chat/completions',
                     **model_retry_contract(matrix),
                     'budgets':matrix.model_dump(include={'max_decisions','max_actions','max_tool_calls_per_decision',
                         'max_program_actions','sim_budget','wall_budget','max_output_tokens','model_timeout','http_timeout','program_timeout'}),
                     'actual_runtime_pins':actual_pins}
        if slot['mode'] == 'free_tool_agent':
            condition['track'] = 'free_tool_agent'
            condition['public_interface_contract'] = _task_inventory_value(
                plan['service_inventory'], 'free_interface', slot['task'])
            actual_pins['track'] = spec.get('track')
        status = attempt['status'] if attempt else 'missing'
        if archive_errors:
            status='artifact_invalid'
        outcome = classify_report(report)
        contract = _scoring_contract(plan, slot['task'])
        continuous = contract is not None
        evidence_valid = (status=='finished' and fresh_verification is not None
                          and fresh_verification.get('passed') is True and result.get('physics_valid') is True)
        binary_valid = evidence_valid and not continuous and outcome['task_outcome'] != 'unknown'
        score_valid = evidence_valid and continuous and outcome.get('task_score_valid') is True
        valid = binary_valid or score_valid
        claimed_success=binary_valid and outcome['task_outcome']=='success'
        completion_sim=result.get('completion_sim_s') if claimed_success else None
        completion_wall=result.get('completion_wall_s') if claimed_success else None
        if claimed_success and any(isinstance(value,bool) or not isinstance(value,(int,float))
                or not math.isfinite(value) or value<0 or (budget is not None and value>budget)
                for value,budget in ((completion_sim,matrix.sim_budget),(completion_wall,matrix.wall_budget))):
            valid=False
            binary_valid=False
            archive_errors.append('invalid_success_completion_times')
            status='artifact_invalid'
        success=bool(claimed_success) if binary_valid else None
        completion_sim=completion_sim if success else None
        completion_wall=completion_wall if success else None
        row = {'slot':slot,'status':status, 'attempt_status':(attempt or {}).get('status', 'missing'),
               'valid_recorded_episode':valid,
               'task_scoring_mode': 'continuous' if continuous else 'binary',
               'task_score': outcome.get('task_score'),
               'task_score_unit': outcome.get('task_score_unit'),
               'task_score_objective': contract.get('objective') if isinstance(contract, dict) else None,
               'task_score_valid': score_valid,
               'result_valid_for_scoring': valid,
               'task_success':success,
               'task_outcome':outcome['task_outcome'] if valid else 'unknown',
               'task_outcome_reason':('invalid_archived_evidence' if archive_errors
                                     else 'unverified_episode_evidence' if not valid and outcome['task_outcome']!='unknown'
                                     else outcome['task_outcome_reason']),
               'run_status':('not_run' if status=='missing' else 'infra_error' if status=='service_unavailable'
                             else 'interrupted' if status=='started' else 'unknown_error' if not report
                             else outcome['run_status']),
               'outcome_policy':outcome['outcome_policy'],
               'outcome':result.get('outcome'), 'termination':report.get('termination'),
               'runner_error_type':report.get('error_type') or (attempt or {}).get('error_type'),
               'failure_phase':(attempt or {}).get('failure_phase'),
               'pin_errors':verification.get('errors', []), 'physics_valid':result.get('physics_valid'),
               'artifact_errors':archive_errors,
               'completion_sim_s':completion_sim,'completion_wall_s':completion_wall,
               'diagnostic_penalized_sim_cost_s':(None if matrix.sim_budget is None or not binary_valid
                                                  else completion_sim if success else matrix.sim_budget),
               'diagnostic_penalized_wall_cost_s':(None if matrix.wall_budget is None or not binary_valid
                                                   else completion_wall if success else matrix.wall_budget),
               'scene_sha256':spec.get('scene_sha256'), 'model_call_attempts':report.get('model_call_attempts'),
               'submitted_actions':report.get('submitted_actions'), 'completed_actions':report.get('completed_actions'),
               'client_primitive_attempts':report.get('primitive_attempts'),
               'tool_calls':report.get('tool_calls'),
               'server_action_attempts':result.get('action_attempts'), 'elapsed_sim_s':result.get('elapsed_sim_s'),
               'elapsed_wall_s':result.get('elapsed_wall_s'), 'attempt_wall_s':(attempt or {}).get('actual_attempt_wall_s'),
               'model_accounting':report.get('model_accounting', 'unavailable'),
               'result_model_accounting':result.get('model_accounting', 'unavailable'),
               'actual_runtime_pins':actual_pins}
        row['model_not_run'] = bool(status=='episode_setup_failed' and not archive_errors)
        row.update(model_statistics(report if not archive_errors else {}))
        row['decision_status_counts']=report.get('decision_status_counts')
        row['program_status_counts']=report.get('program_status_counts')
        # Keep each decision's action feedback without inferring failure from completed-count differences.
        events = []
        for execution in sorted((run_directory/'decisions').glob('*/execution.json')):
            try:
                value=read(execution)
            except (OSError,ValueError):
                value={'status':'unreadable_record'}
            events.append({'decision':execution.parent.name,'status':value.get('status')if isinstance(value,dict)else'unreadable_record'})
        row['decision_execution_status_counts'] = dict(Counter(event['status'] for event in events))
        primitive_requests = sorted((run_directory/'decisions').rglob('primitive_request.json'))
        primitive_status=[]
        primitive_reasons=[]
        for request in primitive_requests:
            try:
                value=read(request.with_name('execution.json'))
                primitive_status.append(value.get('status')if isinstance(value,dict)else'unreadable_record')
                reason=value.get('termination_reason') if isinstance(value,dict) else None
                primitive_reasons.append((reason if type(reason) is str and reason in get_args(TerminationReason) else 'not_recorded')
                    if isinstance(value,dict) else 'unreadable_record')
            except (OSError,ValueError):
                primitive_status.append('no_readable_terminal_record')
                primitive_reasons.append('no_readable_terminal_record')
        row['recorded_primitive_attempts'] = len(primitive_requests)
        row['primitive_execution_status_counts'] = dict(Counter(primitive_status))
        row['primitive_termination_reason_counts'] = dict(Counter(primitive_reasons))
        rows.append(row)
        groups[canonical_hash(condition)].append((condition,row))
    cohorts=[]
    for key, entries in sorted(groups.items()):
        condition=entries[0][0]; values=[entry[1]for entry in entries]
        valid=[row for row in values if row['valid_recorded_episode']]
        binary=[row for row in values if row['task_scoring_mode'] == 'binary']
        binary_valid=[row for row in binary if row['valid_recorded_episode']]
        successes=sum(row['task_success'] is True for row in values)
        per_task={}
        for task in sorted({row['slot']['task']for row in values}):
            task_rows=[row for row in values if row['slot']['task']==task]
            task_valid=sum(row['valid_recorded_episode']for row in task_rows)
            task_binary_valid=sum(row['valid_recorded_episode'] and row['task_scoring_mode'] == 'binary'
                                  for row in task_rows)
            task_successes=sum(row['task_success'] is True for row in task_rows)
            per_task[task]={'planned_slots_in_group':len(task_rows),'valid_recorded_episodes':task_valid,
                            'valid_binary_episodes':task_binary_valid,
                            'valid_continuous_episodes':sum(row['task_score_valid'] for row in task_rows),
                            'continuous_scores':_score_summaries(task_rows),
                            'successes':task_successes,
                            'success_rate_valid_episodes':task_successes/task_binary_valid if task_binary_valid else None,
                            'status_counts':dict(Counter(row['status']for row in task_rows))}
        rates=[row['success_rate_valid_episodes']for row in per_task.values()if row['success_rate_valid_episodes']is not None]
        sim_cost=_sum(binary, lambda row:row['diagnostic_penalized_sim_cost_s']) if binary else None
        wall_cost=_sum(binary, lambda row:row['diagnostic_penalized_wall_cost_s']) if binary else None
        cohorts.append({'cohort_sha256':key,'condition':condition,'planned_slots_in_group':len(values),
                        'valid_recorded_episodes':len(valid),'successes':successes,
                        'planned_binary_slots':len(binary),
                        'valid_binary_episodes':len(binary_valid),
                        'valid_continuous_episodes':sum(row['task_score_valid'] for row in values),
                        'continuous_scores':_score_summaries(values),
                        'unknown_task_outcomes':sum(row['task_outcome']=='unknown' for row in values),
                        'run_status_counts':dict(Counter(row['run_status'] for row in values)),
                        'task_outcome_counts':dict(Counter(row['task_outcome'] for row in values)),
                        'success_fraction_of_planned_slots':successes/len(binary) if binary else None,
                        'success_rate_valid_episodes':successes/len(binary_valid) if binary_valid else None,
                        'status_counts':dict(Counter(row['status']for row in values)),
                        'outcome_counts':dict(Counter(row['outcome'] or 'not_recorded'for row in values)),
                        'model_accounting_counts':dict(Counter(row['model_accounting']for row in values)),
                        'actual_elapsed_sim_s':_sum(values,lambda r:r['elapsed_sim_s']),
                        'actual_elapsed_wall_s':_sum(values,lambda r:r['elapsed_wall_s']),
                        'actual_attempt_wall_s':_sum(values,lambda r:r['attempt_wall_s']),
                        'model_call_attempts_client_observed':_sum(values,lambda r:r['model_call_attempts']),
                        'server_action_attempts':_sum(values,lambda r:r['server_action_attempts']),
                        **{field:_sum(values,lambda row,field=field:row[field]) for field in
                            ('model_latency_client_s','gateway_prompt_tokens','gateway_completion_tokens','gateway_total_tokens','gateway_reasoning_tokens')},
                        'diagnostic_penalized_sim_cost_s':sim_cost,'diagnostic_penalized_wall_cost_s':wall_cost,
                        'diagnostic_success_per_penalized_sim_second':successes/sim_cost if sim_cost is not None and sim_cost>0 else None,
                        'diagnostic_success_per_penalized_wall_second':successes/wall_cost if wall_cost is not None and wall_cost>0 else None,
                        'diagnostic_task_macro_success_rate':sum(rates)/len(rates)if rates else None,
                        'tasks_with_valid_episodes':sum(row['valid_recorded_episodes'] > 0 for row in per_task.values()),
                        'tasks_with_valid_binary_episodes':len(rates),
                        'decision_status_counts':dict(sum((Counter(row['decision_status_counts']or{})for row in values),Counter())),
                        'program_status_counts':dict(sum((Counter(row['program_status_counts']or{})for row in values),Counter())),
                        'primitive_execution_status_counts':dict(sum((Counter(row['primitive_execution_status_counts'])for row in values),Counter())),
                        'per_task':per_task,
                        'slots':[row['slot']['slot_id']for row in values], 'leaderboard_eligible':False})
    report={'protocol':'agent_development_suite_report_v1','plan_sha256':plan['plan_sha256'],
            'attestation_policy':'normal_archive_attestation_v2',
            'attestation_scope':'Finished episodes require fresh archived spec/config validation and matching before/after source attestations. Missing historical attestations mean insufficient evidence under this policy, not a changed native outcome or proof of damaged archive bytes.',
            'created_at_utc':now(),'scope':'Development matrix only; not a server EvaluationPlan or ranked evaluation.',
            'leaderboard_eligible':False,'model_call_budget_server_verified':False,
            'planned_slots':len(rows),'attempted_slots':sum(row['status']!='missing'for row in rows),
            'unattempted_slot_ids':[row['slot']['slot_id']for row in rows if row['status']=='missing'],
            'incomplete_attempt_ids':[row['slot']['slot_id']for row in rows if row['status']=='started'],
            'all_slots_attempted':all(row['status']not in ('missing','started')for row in rows),
            'all_slots_valid':all(row['valid_recorded_episode']for row in rows),
            'seed_warning':plan['seed_warning'],'cohorts':cohorts,'slots':rows,
            'definitions':{'planned_fraction':'Binary-task failures and missing/invalid slots remain visible in the binary planned denominator; continuous tasks are excluded. Unknown runtime pins form separate groups.',
                'valid_success_rate':'Descriptive binary-task rate only; denominator excludes continuous tasks and invalid/incomplete evidence. Always show planned fraction and status counts alongside.',
                'continuous_scores':'Scores require the frozen public scoring contract, valid archived evidence and healthy terminal scoring. Summaries separate task, unit and objective; zero is valid. Continuous tasks receive neither binary failure labels nor binary penalized costs.',
                'action_accounting':'Server action_attempts includes accepted and rejected primitive attempts; completed actions are not task successes.',
                'model_accounting':'Client-observed model attempts are not independently server-verified.',
                'tokens_and_latency':'Sum client response duration_s and gateway usage fields only when every attempted call supplies that field; missing or partial evidence yields null, not zero.',
                'task_outcomes':'Success/failure require valid terminal evidence; API interruptions, cleanup cancellations and missing/invalid evidence stay unknown. A later API error does not erase a verified terminal result. Planned-slot success fraction describes coverage and is not a failure label for unknown slots.',
                'diagnostic_efficiency':'Binary tasks only: for enabled time budgets, valid successes use verified completion times and assessed task failures consume the full frozen budget. Unknown binary outcomes and disabled dimensions yield null costs and null aggregate efficiency; actual completion/elapsed times remain recorded. Continuous tasks are excluded. These are diagnostic costs, not actual elapsed times or rankings.',
                'artifact_integrity':'Re-read final run_report/config/ready and all manifest-covered bytes against pinned final artifact manifest on each report. Changed/missing evidence is retained as invalid.',
                'runtime_pins':'Unknown or changed runtime/controller/observation conditions never share a cohort.',
                'seeds':'Explicit repeat IDs are repeated trials, not evidence of independent layouts or generalization.'}}
    save(root/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    create=commands.add_parser('plan'); create.add_argument('--matrix',type=Path,required=True); create.add_argument('--output',type=Path,required=True)
    run=commands.add_parser('run'); run.add_argument('--plan',type=Path,required=True); run.add_argument('--credentials',type=Path)
    report=commands.add_parser('report'); report.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='plan':
        value=create_plan(read(args.matrix),args.output)
        summary={'plan':str(args.output/'plan.json'),'slots':len(value['slots']),'plan_sha256':value['plan_sha256']}
    else:
        value=run_plan(args.plan,credentials=args.credentials)if args.command=='run'else report_plan(args.plan)
        summary={key:value[key]for key in('planned_slots','attempted_slots','all_slots_attempted','all_slots_valid','leaderboard_eligible')}
        summary['report']=str(args.plan.parent/'report.json')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
