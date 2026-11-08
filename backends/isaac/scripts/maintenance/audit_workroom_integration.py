"""Read-only CPU audit of current catalog, frozen releases and public discovery.

Run as a module from either a source checkout or an isolated wheel installation.
Shared frozen assets must be available. No episode, native worker or model is
started; this does not certify manipulation or physical qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent.model_client.scene_tasks import TaskSetupError, list_scene_tasks, published_task, select_task
from bench.control.factory import robot_configuration
from bench.isaacsim.release import task_inputs
from bench.protocol.task_referee import contract_sha256
from bench.runtime.controllers import default_registry
from bench.runtime.reset_cases import is_randomized_release
from bench.runtime.server import create_app
from bench.runtime.task_capabilities import (TaskCapabilityError, engineering_capabilities,
                                             require_robot_episode_capability)
from bench.runtime.worker import public_task_projection
from bench.simulation.isaacsim_backend.l_mount import public_mount
from scenes.workrooms import catalog


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(row, name, condition):
    row['checks'][name] = bool(condition)
    if not condition:
        row['errors'].append(name)


def rejects_release(selection, actual):
    try:
        selection.verify_release({'task_release_sha256': actual})
    except TaskSetupError as exc:
        return exc.code == 'release_mismatch'
    return False


def rejects_engineering_episode(action):
    try:
        action()
    except TaskSetupError as exc:
        return exc.code == 'robot_episodes_unavailable'
    except TaskCapabilityError:
        return True
    return False


def audit_engineering_task(row, definition, loaded, root, scene, task):
    """Audit explicit opt-outs through read-only discovery, without robot setup."""
    task_id, family = definition['task_id'], definition['family']
    capabilities = engineering_capabilities(task)
    check(row, 'engineering_capabilities_match', capabilities is not None and
          capabilities == engineering_capabilities(definition) == engineering_capabilities(loaded['task']))
    check(row, 'runtime_scene_matches_frozen', scene == loaded['scene'])
    entrypoint = capabilities.get('engineering_entrypoint') if capabilities else None
    check(row, 'engineering_entrypoint_exists', bool(entrypoint) and importlib.util.find_spec(entrypoint) is not None)
    check(row, 'ordinary_episode_capability_rejected',
          rejects_engineering_episode(lambda: require_robot_episode_capability(task)))
    # The actual registry must honor the capability before constructing CPU
    # robot geometry. No service.create or native worker is invoked here.
    registry = default_registry(root)
    check(row, 'no_robot_policy', registry.descriptors(task_id=task_id) == [])
    manager = SimpleNamespace(release=root, registry=registry, shutdown=lambda: None)
    with TestClient(create_app(service=manager)) as client:
        response = client.get('/tasks')
        check(row, 'tasks_http_status', response.status_code == 200)
        public_rows = response.json()
        matches = [item for item in public_rows if item['task_id'] == task_id]
        check(row, 'task_uniquely_published', len(matches) == 1)
        public = matches[0]
        check(row, 'public_engineering_capabilities', public.get('capabilities') == capabilities and
              public.get('engineering_entrypoint') == entrypoint and public.get('robot_episodes') is False)
        check(row, 'public_engineering_only', public.get('task_kind') == 'engineering' and
              public.get('runnable') is False and 'create_endpoint' in public and public['create_endpoint'] is None)
        response = client.get('/policies', params={'task_id': task_id})
        check(row, 'policies_http_contract', response.status_code == 200 and response.json() == [])
        response = client.get('/interfaces', params={'task_id': task_id})
        check(row, 'interfaces_http_status', response.status_code == 200)
        interfaces = response.json()
        row['interface_schema'] = interfaces.get('protocol')
        check(row, 'engineering_interfaces_capabilities', interfaces.get('robot_episodes') is False and
              interfaces.get('engineering_entrypoint') == entrypoint and
              interfaces.get('task_capabilities', {}).get(task_id) == capabilities)
        check(row, 'ordinary_interfaces_unavailable', bool(interfaces.get('interfaces')) and
              all(item.get('available') is False and item.get('backend_available') is False and
                  item.get('client_command') is None for item in interfaces['interfaces']) and
              interfaces.get('geometry_policies') == [])
        common = interfaces.get('shared_contract', {})
        check(row, 'ordinary_action_endpoints_unavailable',
              all(key in common and common[key] is None for key in
                  ('create_endpoint', 'action_endpoint', 'function_call_endpoint')))
    row.update(route=public.get('create_endpoint'), task_kind=public.get('task_kind'),
               capabilities=capabilities, engineering_entrypoint=entrypoint)
    number = row['task_number']
    row['aliases'] = list(dict.fromkeys([task_id, f'task{number}']))
    for alias in row['aliases']:
        selection = select_task(alias, family)
        metadata = selection.metadata()
        check(row, f'alias:{alias}', selection.task_id == task_id and selection.pin_release and
              selection.source_release == definition['source_release'])
        check(row, f'engineering_selection:{alias}', metadata['session_kind'] == 'engineering' and
              metadata['runnable'] is False and metadata['practice_rsi_allowed'] is False and
              metadata.get('capabilities') == capabilities)
        check(row, f'ordinary_episode_selection_rejected:{alias}',
              rejects_engineering_episode(selection.require_runnable))
        check(row, f'published_episode_selection_rejected:{alias}',
              rejects_engineering_episode(lambda: published_task(selection, public_rows)))
        selection.verify_release({'task_release_sha256': row['release']['manifest_sha256']})
        check(row, f'reject_wrong_release:{alias}', rejects_release(selection, '0' * 64))
    row['agent_selection'] = select_task(task_id, family).metadata()


def audit_task(definition):
    family, task_id = definition['family'], definition['task_id']
    number = definition.get('task_number', definition.get('difficulty'))
    row = {'family': family, 'task_id': task_id, 'task_number': number,
           'status': definition['status'], 'source_release': definition['source_release'],
           'checks': {}, 'errors': []}
    try:
        loaded = catalog.load_task(family, task_id)
        check(row, 'catalog_full_hash_and_evaluator_validation', True)
        root = Path(definition['source_release']['path'])
        manifest = loaded['release_manifest']
        original_hash = file_sha256(root / 'release_manifest.json')
        row['release'] = {'path': str(root), 'manifest_sha256': original_hash,
                          'schema': manifest['schema'],
                          'file_hash_count': len(manifest['files_sha256']),
                          'asset_hash_count': len(manifest.get('asset_files_sha256', {}))}
        check(row, 'manifest_pin', original_hash == definition['source_release']['manifest_sha256'])
        row['parent_environment'] = loaded['scene']['workroom']
        check(row, 'parent_environment_exact', row['parent_environment'] ==
              loaded['environment']['environment'])

        # Frozen finite reset cases accept only their published seeds; seed 0 is
        # the canonical case that discovery publishes. Others bind seed 71.
        seed = 0 if is_randomized_release(manifest) else 71
        entry, scene, task, initial = task_inputs(root, task_id, seed=seed)
        check(row, 'canonical_runtime_task_id', entry['id'] == task_id)
        if engineering_capabilities(task) is not None or engineering_capabilities(definition) is not None:
            audit_engineering_task(row, definition, loaded, root, scene, task)
            check(row, 'release_unchanged', file_sha256(root / 'release_manifest.json') == original_hash)
            row['passed'] = not row['errors']
            return row
        check(row, 'runtime_robot_matches_frozen', scene['robot'] == loaded['scene']['robot'])
        configuration = robot_configuration(scene)
        registry = default_registry(root)
        descriptors = registry.descriptors(task_id=task_id)
        check(row, 'one_task_bound_policy', len(descriptors) == 1)
        descriptor = descriptors[0]
        policy = registry.resolve(descriptor.policy_id, descriptor.version, task_id=task_id)
        geometry = policy.geometry
        arms = list(geometry.sides.values()) if hasattr(geometry, 'sides') else [geometry]
        row['policy'] = descriptor.model_dump(mode='json')
        row['robot'] = {'requirements': definition['robot_requirements'],
                        'configuration': configuration, 'robot_id': geometry.robot_id,
                        'joint_count': len(geometry.names),
                        'arm_joint_count': sum(len(arm.arm.names) for arm in arms),
                        'hand_joint_count': sum(len(arm.hand.names) for arm in arms)}
        requirements = definition['robot_requirements']
        expected_robot = requirements.get('composite_hand_id', requirements.get('hand_id'))
        if expected_robot is not None:
            check(row, 'declared_robot_id', geometry.robot_id == expected_robot)
        expected_assembly = requirements.get('composite_assembly_version', requirements.get('assembly_version'))
        if expected_assembly is not None:
            check(row, 'declared_assembly', scene['robot'].get('assembly_version') == expected_assembly)
        if 'base_position_m' in requirements:
            check(row, 'declared_base', configuration['base_position_m'] == requirements['base_position_m'])
        if 'base_positions_m' in requirements:
            bases = requirements['base_positions_m']
            actual = ({side: item['base_position_m'] for side, item in configuration['instances'].items()}
                      if 'instances' in configuration else {next(iter(bases)): configuration['base_position_m']})
            check(row, 'declared_bases', bases == actual)
        check(row, 'declared_articulation_count', len(geometry.names) == 29 * requirements['articulations'])
        check(row, 'reset_joint_inventory', set(initial['joint_positions_rad']) == set(geometry.names))
        for arm in arms:
            arm.checked_joints({name: initial['joint_positions_rad'][name] for name in arm.names},
                               enforce_limits=True)
        check(row, 'reset_joint_limits', True)
        check(row, 'policy_robot', geometry.robot_id in descriptor.supported_robot_ids)
        check(row, 'control_period', descriptor.control_period_s == .02)
        check(row, 'policy_configuration', policy.robot_configuration == configuration)

        # Only read public discovery endpoints. This manager cannot create an
        # episode or a native process; default_registry constructs CPU geometry.
        manager = SimpleNamespace(release=root, registry=registry, shutdown=lambda: None)
        with TestClient(create_app(service=manager)) as client:
            response = client.get('/tasks')
            check(row, 'tasks_http_status', response.status_code == 200)
            public_rows = response.json()
            matches = [item for item in public_rows if item['task_id'] == task_id]
            check(row, 'task_uniquely_published', len(matches) == 1)
            public = matches[0]
            response = client.get('/policies', params={'task_id': task_id})
            check(row, 'policies_http_contract', response.status_code == 200 and
                  response.json() == [descriptor.model_dump(mode='json')])
            response = client.get('/interfaces', params={'task_id': task_id})
            check(row, 'interfaces_http_status', response.status_code == 200)
            row['interface_schema'] = response.json().get('protocol')
        row['route'] = public['create_endpoint']
        row['task_kind'] = public['task_kind']
        row['budgets'] = task.get('recommended_budgets')
        frozen_rules = task.get('public_rules', {})
        public_rules = public.get('public_rules', {})
        row['rules'] = {'frozen_schema': frozen_rules.get('schema'),
                        'frozen_sha256': task.get('public_rules_sha256'),
                        'public_schema': public_rules.get('schema'),
                        'public_sha256': public.get('public_rules_sha256'),
                        'public_computed_sha256': contract_sha256(public_rules),
                        'object_property_count': len(public_rules.get('object_properties', []))}
        if 'public_rules_sha256' in public:
            check(row, 'public_rules_hash', public['public_rules_sha256'] == contract_sha256(public_rules))
        if task.get('public_rules_sha256') is not None:
            check(row, 'frozen_rules_hash', task['public_rules_sha256'] == contract_sha256(frozen_rules))
        if public['create_endpoint'] == 'POST /episodes':
            projected = public_task_projection(task)
            # Discovery also names the selected release, installation and
            # physical mount; the worker publishes only its episode task.
            discovery_only = {'robot_mount', 'configuration_id', 'task_release_sha256'}
            mount = public_mount(scene['robot'])
            if mount is not None:
                check(row, 'discovery_robot_mount', public.get('robot_mount') == mount)
            if is_randomized_release(manifest):
                check(row, 'discovery_release_identity', public.get('task_release_sha256') == original_hash
                      and public.get('configuration_id') == manifest['configuration_id'])
            # task0 adds episode-bound placement targets only after creation.
            if task_id == 'task0' and family == 'card_room':
                check(row, 'worker_public_identity', projected['task_id'] == public['task_id'] and
                      {key: value for key, value in projected['public_rules'].items()
                       if key != 'target_cells'} == public['public_rules'])
                check(row, 'task0_discovery_does_not_pin_episode_targets',
                      'target_cells' not in public['goal'] and 'target_cells' not in public_rules and
                      len(projected['goal']['target_cells']) == 3 and projected['episode_seed'] == seed)
            else:
                # Water-pouring publishes its bound seed with the public task;
                # discovery describes the frozen seed-zero template.
                if 'episode_seed' in projected or 'episode_seed' in public:
                    check(row, 'worker_public_episode_seed', projected.get('episode_seed') == task.get('episode_seed'))
                    check(row, 'discovery_template_seed', public.get('episode_seed') == loaded['task'].get('episode_seed'))
                check(row, 'worker_public_contract',
                      {key: value for key, value in projected.items() if key != 'episode_seed'} ==
                      {key: value for key, value in public.items()
                       if key != 'episode_seed' and key not in discovery_only})
        if family == 'laboratory':
            check(row, 'laboratory_static_properties', bool(public_rules.get('object_properties')) and
                  public_rules['object_properties'] == frozen_rules['referee']['object_properties'])

        aliases = list(dict.fromkeys([task_id, f'task{number}'] +
                                    ([f'L{number}'] if family == 'mechanical_assembly' else [])))
        row['aliases'] = aliases
        for alias in aliases:
            selection = select_task(alias, family)
            selection.require_runnable()
            check(row, f'alias:{alias}', selection.task_id == task_id and selection.pin_release and
                  selection.source_release == definition['source_release'])
            selected, board = published_task(selection, public_rows)
            check(row, f'published:{alias}', selected == public and
                  board == (public['create_endpoint'] == 'POST /board-games'))
            selection.verify_release({'task_release_sha256': original_hash})
            check(row, f'reject_wrong_release:{alias}', rejects_release(selection, '0' * 64))
        row['agent_selection'] = select_task(task_id, family).metadata()
        check(row, 'release_unchanged', file_sha256(root / 'release_manifest.json') == original_hash)
    except Exception as exc:
        row['errors'].append(f'{type(exc).__name__}: {exc}')
        row['traceback'] = traceback.format_exc()
    row['passed'] = not row['errors']
    return row


def audit(*, expected_count=None, source_commit=None, expected_package_root=None):
    package_root = Path(catalog.__file__).resolve().parents[2]
    if expected_package_root is not None and package_root != Path(expected_package_root).resolve():
        raise ValueError(f'Imported catalog from unexpected package root: {package_root}')
    rows = catalog.list_tasks(include_planned=True)
    report = {'schema': 'workroom_integration_audit.v1',
              'at_utc': datetime.now(timezone.utc).isoformat(), 'source_commit': source_commit,
              'package_root': str(package_root), 'python': sys.executable,
              'python_version': sys.version, 'script_sha256': file_sha256(__file__),
              'catalog_sha256': file_sha256(catalog.__file__),
              'scope': 'CPU frozen loading and discovery only; no episode or physical qualification.',
              'native_runs': 0, 'model_calls': 0, 'checks': {}, 'errors': [], 'tasks': []}
    check(report, 'expected_task_count', expected_count is None or len(rows) == expected_count)
    check(report, 'unique_family_task_ids', len({(r['family'], r['task_id']) for r in rows}) == len(rows))
    check(report, 'all_current_tasks_implemented', all(r['status'] == 'implemented' for r in rows))
    discovered = [row for family in catalog.CURRENT_FAMILIES for row in list_scene_tasks(family)]
    check(report, 'agent_catalog_task_inventory',
          Counter((row['family'], row['task_id']) for row in discovered) ==
          Counter((row['family'], row['task_id']) for row in rows))
    for definition in rows:
        row = audit_task(definition)
        report['tasks'].append(row)
        print(f'{row["family"]}/{row["task_id"]}: {"PASS" if row["passed"] else row["errors"]}', flush=True)
    report['family_counts'] = dict(Counter(row['family'] for row in rows))
    report['status_counts'] = dict(Counter(row['status'] for row in rows))
    report['route_counts'] = dict(Counter(row.get('route', 'unverified') for row in report['tasks']))
    report['summary'] = {'tasks': len(rows), 'passed_tasks': sum(r['passed'] for r in report['tasks']),
                         'checks': len(report['checks']) + sum(len(r['checks']) for r in report['tasks']),
                         'passed_checks': sum(report['checks'].values()) +
                         sum(sum(r['checks'].values()) for r in report['tasks'])}
    report['passed'] = not report['errors'] and all(r['passed'] for r in report['tasks'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-count', type=int)
    parser.add_argument('--source-commit', help='Source revision label, also used for a wheel audit')
    parser.add_argument('--expected-package-root', type=Path,
                        help='Fail if imports escape this checkout or isolated installation')
    args = parser.parse_args()
    source_commit = args.source_commit
    if source_commit is None:
        result = subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'],
                                text=True, capture_output=True, check=False)
        source_commit = result.stdout.strip() if result.returncode == 0 else None
    report = audit(expected_count=args.expected_count, source_commit=source_commit,
                   expected_package_root=args.expected_package_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report['summary']), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
