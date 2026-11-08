"""Operator-pinned task variants; selection changes the release, never a scene in place."""
from copy import deepcopy
from pathlib import Path
import json

from bench.paths import load_json
from bench.runtime.reset_cases import is_randomized_release
from bench.isaacsim.release import digest
from bench.runtime.release_cache import validate_cached as validate
from bench.protocol.robot_configurations import project_robot_configurations


def discover_configuration_bindings(release):
    """Only the exact current catalog pin may advertise current alternatives.

    Historical releases retain their own contract and never inherit variants
    because their task ID happens to match a current task.
    """
    from scenes.workrooms.catalog import list_tasks
    root = Path(release).resolve()
    manifest = validate(root, assets=False)
    current_hash = digest(root/'release_manifest.json')
    task_ids = {row['id'] for row in manifest['tasks']}
    result = {}
    # Both physical robot profiles have operator-pinned entrypoints. An old
    # release still cannot inherit choices by task ID alone: the exact release
    # path and digest below must match one profile's current catalog.
    definitions = [row for profile in ('original', 'l_mount')
                   for row in list_tasks(include_planned=False, robot_profile=profile)]
    for definition in definitions:
        options = definition.get('robot_configurations')
        if definition['task_id'] not in task_ids or not options:
            continue
        matches = [key for key, option in options.items()
                   if Path(option['source_release']['path']).resolve() == root
                   and option['source_release']['manifest_sha256'] == current_hash]
        if not matches:
            continue
        if len(matches) != 1 or len(task_ids) != 1:
            raise ValueError('Robot configuration release must bind one unique task and configuration')
        public = project_robot_configurations(options)
        default = definition.get('default_configuration_id', 'single_arm')
        if default not in options or options[default]['source_release'] != definition['source_release']:
            raise ValueError('Default robot configuration differs from the task release')
        if definition.get('rules', {}).get('contract', {}).get('competitive'):
            raise ValueError('Robot configuration selection is ordinary single-agent only')
        candidate = {
            'task_id': definition['task_id'],
            'default_configuration_id': matches[0],
            'scene_description_zh': definition.get('scene_description_zh', definition['title']),
            'robot_configurations': public,
            'releases': {key: deepcopy(option['source_release']) for key, option in options.items()},
        }
        prior = result.get(definition['task_id'])
        if prior is not None and prior != candidate:
            raise ValueError('One release has conflicting physical robot configuration menus')
        result[definition['task_id']] = candidate
    return result


def checked_configuration_release(binding, configuration_id):
    if configuration_id not in binding['releases']:
        raise ValueError('Unknown robot configuration for this task')
    reference = binding['releases'][configuration_id]
    root = Path(reference['path']).resolve()
    if digest(root/'release_manifest.json') != reference['manifest_sha256']:
        raise ValueError('Robot configuration release differs from its catalog pin')
    manifest = validate(root, assets=False)
    variant = manifest.get('schema') == 'workroom_robot_configuration_release.v1'
    configured_cloth = (manifest.get('schema') == 'laundry_cloth_task_release.v1'
                        and 'configuration_id' in manifest)
    configured_randomized = (is_randomized_release(manifest)
                             or manifest.get('schema') == 'workroom_robot_revision_release.v1')
    legacy_revision_single = False
    if (manifest.get('schema') == 'workroom_robot_revision_release.v1'
            and configuration_id == 'single_arm' and manifest.get('configuration_id') is None):
        from scenes.workrooms.robot_revision.release import legacy_single_menu
        # Full wrapper validation above already verified the exact parent and
        # these files. Public menu selection must not invent a private ID.
        parent = Path(manifest['base_release']['path'])
        legacy_revision_single = legacy_single_menu(
            load_json(parent/'release_manifest.json'),
            load_json(parent/'scene.json'))
    if configured_cloth or configured_randomized:
        if manifest['configuration_id'] != configuration_id and not legacy_revision_single:
            raise ValueError('Frozen release belongs to another robot installation')
    elif ((configuration_id == 'single_arm' and variant)
            or (configuration_id != 'single_arm' and
                (not variant or manifest.get('configuration_id') != configuration_id))):
        raise ValueError('Frozen release belongs to another robot installation')
    entries = manifest['tasks']
    if len(entries) != 1 or entries[0]['id'] != binding['task_id']:
        raise ValueError('Robot configuration release belongs to another task')
    scene = load_json(root/entries[0]['scene'])
    expected = binding['robot_configurations'][configuration_id]['robot_requirements']
    if any(scene['robot'].get(key) != expected[key] for key in ('hand_id', 'assembly_version')):
        raise ValueError('Robot configuration release has another robot identity')
    return root


def public_configuration_binding(binding):
    return {key: deepcopy(binding[key]) for key in
            ('default_configuration_id', 'scene_description_zh', 'robot_configurations')}
