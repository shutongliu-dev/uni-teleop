"""One task catalog with explicit original and L-mount robot profiles."""
from copy import deepcopy
from pathlib import Path

from scenes.workrooms.robot_revision.assets import checked_file, file_hash
from scenes.workrooms.robot_revision.catalog import (
    bind_definition as bind_legacy_revision, _requirements)
from scenes.workrooms.initial_randomization import TASK_FAMILIES
from scenes.workrooms.living_room.randomization import TASK_IDS as LIVING_TASKS
from scenes.workrooms.laundry_room.randomization import TASK_IDS as LAUNDRY_TASKS
from .pairs import PROFILES, SCHEMA as PAIR_SCHEMA, read

SCHEMA = 'paired_parameterized_catalog.v1'
PINS = Path(__file__).with_name('catalog_pins.json')
TASKS = frozenset([family+'/'+task for task, family in TASK_FAMILIES.items()]
                  + ['living_room/'+task for task in LIVING_TASKS]
                  + ['laundry_room/'+task for task in LAUNDRY_TASKS])
QUALIFICATION = {
    'status': 'prepared_unvalidated', 'physics_accepted': False,
    'robot_manipulation': False, 'inherited_qualification': False,
    'scope': 'Parameterized task instance; geometry, native and interface evidence are separately bound.'}


# dexbench: the original robot (assembly isaacsim_6_1_v2) is the default profile; interface
# defaults to the L-mount robot. robot_profile='l_mount' (--robot-profile l_mount) selects it.
DEFAULT_PROFILE = 'original'


def checked_profile(profile):
    value = DEFAULT_PROFILE if profile is None else profile
    if value not in PROFILES:
        raise ValueError('robot_profile must be original or l_mount')
    return value


def _installation(manifest, pin):
    """The parent installation: declared, else the original fixed-task release's own robot.

    A task re-based onto new original releases has no seed publication over them,
    so its paired row names the original installations directly. Those predate
    configuration IDs; they are identified exactly as seed publication does.
    """
    if 'configuration_id' in manifest:
        return manifest['configuration_id']
    from scenes.workrooms.initial_randomization.release import _parent_configuration
    scene = read(checked_file(Path(pin['path'])/'scene.json', manifest['files_sha256']['scene.json']))
    return _parent_configuration(scene, manifest.get('task_id'))


def _option(definition, original, pair_ref, profile):
    if set(pair_ref) != {'path', 'sha256'}:
        raise ValueError('A robot pair requires an exact immutable reference')
    pair = read(checked_file(pair_ref['path'], pair_ref['sha256']))
    parent_pin = original['source_release']
    parent = read(checked_file(Path(parent_pin['path'])/'release_manifest.json',
                               parent_pin['manifest_sha256']))
    if (pair.get('schema') != PAIR_SCHEMA or pair.get('family') != definition['family']
            or pair.get('task_id') != definition['task_id']
            or pair.get('configuration_id') != _installation(parent, parent_pin)
            or pair.get('robot_assembly_versions') != PROFILES
            or set(pair.get('robot_profiles', {})) != set(PROFILES)
            or pair.get('seed_values') != list(range(10))
            or set(pair.get('physical_case_sha256', {})) != set(map(str, range(10)))):
        raise ValueError('Incomplete or wrong physical task/robot pair')
    manifests = {}
    for key, pin in pair['robot_profiles'].items():
        root = Path(pin['path'])
        manifests[key] = read(checked_file(root/'release_manifest.json', pin['manifest_sha256']))
        if (manifests[key].get('task_id') != definition['task_id']
                or manifests[key].get('configuration_id') != pair['configuration_id']
                or manifests[key].get('seed_values') != list(range(10))):
            raise ValueError('Robot pair points to a different task or installation')
    old, revised = manifests['original'], manifests['l_mount']
    if (revised.get('schema') != 'workroom_robot_revision_release.v1'
            or revised.get('base_release') != {
                **pair['robot_profiles']['original'], 'schema': old['schema']}):
        raise ValueError('L-mount task is not a robot-only derivative of its paired original')
    pin, manifest = pair['robot_profiles'][profile], manifests[profile]
    root = Path(pin['path'])
    scene = read(checked_file(root/'scene.json', manifest['files_sha256']['scene.json']))
    if scene['robot'].get('assembly_version') != PROFILES[profile]:
        raise ValueError('Physical robot differs from selected profile')
    result = deepcopy(original)
    result.update(source_release=deepcopy(pin), release_schema=manifest['schema'],
                  release_schema_version=manifest['schema_version'],
                  robot_requirements=_requirements(original['robot_requirements'], scene['robot']),
                  qualification=deepcopy(QUALIFICATION), robot_profile=profile,
                  physical_instance_pair=deepcopy(pair_ref))
    contract = (original.get('rules') or {}).get('contract')
    if contract is not None:
        # Where the parent definition serves exactly its release's public rules,
        # keep serving the selected release's rules. The parameterized release
        # may only re-describe its randomization; anything else is a new task.
        parent_task = read(checked_file(Path(parent_pin['path'])/'task.json', parent['files_sha256']['task.json']))
        if parent_task.get('public_rules') == contract:
            selected = read(checked_file(root/'task.json', manifest['files_sha256']['task.json']))['public_rules']
            changed = {key for key in set(contract) | set(selected) if contract.get(key) != selected.get(key)}
            if not changed <= {'randomization'}:
                raise ValueError('Parameterized release changed public rules beyond its randomization description')
            result['rules']['contract'] = deepcopy(selected)
    return result


def bind_definition(definition, robot_profile=None):
    profile = checked_profile(robot_profile)
    pins = read(PINS)
    if pins.get('schema') != SCHEMA or not isinstance(pins.get('tasks'), dict):
        raise ValueError('Invalid paired parameterized catalog')
    if pins['tasks'] and set(pins['tasks']) != TASKS:
        raise ValueError('Parameterized catalog must publish all 37 ordinary tasks atomically')
    identity = definition['family']+'/'+definition['task_id']
    if identity not in pins['tasks']:
        result = bind_legacy_revision(definition) if profile == 'l_mount' else deepcopy(definition)
        return result
    row = pins['tasks'][identity]
    options = definition.get('robot_configurations') or {'fixed': definition}
    if (set(row) != {'family', 'task_id', 'parent_releases', 'pairs'}
            or row['family'] != definition['family'] or row['task_id'] != definition['task_id']
            or row['parent_releases'] != {key: value['source_release'] for key, value in options.items()}
            or set(row['pairs']) != set(options)):
        raise ValueError('Parameterized task changed the original task/installation inventory')
    updated = {key: _option(definition, original, row['pairs'][key], profile)
               for key, original in options.items()}
    if 'robot_configurations' not in definition:
        return updated['fixed']
    result = deepcopy(definition)
    selected = updated[definition.get('default_configuration_id', 'single_arm')]
    result['robot_configurations'] = updated
    for key in ('source_release', 'release_schema', 'release_schema_version',
                'qualification', 'robot_requirements', 'robot_profile', 'physical_instance_pair'):
        result[key] = deepcopy(selected[key])
    return result


def publication_row(definition, pairs):
    options = definition.get('robot_configurations') or {'fixed': definition}
    if set(options) != set(pairs):
        raise ValueError('Pair inventory must preserve every original installation')
    for key, original in options.items():
        for profile in PROFILES:
            _option(definition, original, pairs[key], profile)
    return {'family': definition['family'], 'task_id': definition['task_id'],
            'parent_releases': {key: deepcopy(value['source_release']) for key, value in options.items()},
            'pairs': deepcopy(pairs)}
