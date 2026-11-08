"""Input-only publication pins; preparation or experiment outcomes never admit tasks."""
from copy import deepcopy
from pathlib import Path

from . import CONFIGURATION_IDS, SCHEMA, TASK_IDS, VERSION
from .common import checked_file, file_hash, read_json

PINS = Path(__file__).with_name('catalog_pins.json')
QUALIFICATION = {'status': 'prepared_unvalidated', 'native_preflight': False,
    'agent_bench_interface': False, 'workspace_ik': False, 'workspace_clearance': False,
    'robot_manipulation': False, 'physics_accepted': False,
    'scope': 'Immutable XY/albedo case inputs; validation evidence remains external and episode outcomes are not inherited.'}


def publication_row(task_id, releases):
    from ..robot_configuration_catalog import RELEASES
    from .release import validate_release
    if task_id not in TASK_IDS or set(releases) != set(CONFIGURATION_IDS):
        raise ValueError('Publish one current laundry task with all three original installations')
    original = deepcopy(RELEASES[task_id])
    for configuration, path in releases.items():
        root = Path(path).resolve(); manifest = read_json(root/'release_manifest.json')
        validate_release(root, manifest)
        if (manifest['task_id'] != task_id or manifest['configuration_id'] != configuration
                or manifest['base_release']['path'] != original['configurations'][configuration]['source_release']['path']):
            raise ValueError('Published seed release has another original task or installation')
        task = read_json(root/'task.json')
        option = original['configurations'][configuration]
        option.update(source_release={'path': str(root), 'manifest_sha256': file_hash(root/'release_manifest.json')},
                      release_schema=SCHEMA, release_schema_version=VERSION)
        option['rules']['contract'] = deepcopy(task['public_rules'])
        option['qualification'] = deepcopy(QUALIFICATION)
    original['base_release'] = deepcopy(original['configurations'][original['default_configuration_id']]['source_release'])
    return original


def bind_definition(definition):
    from ..robot_configuration_catalog import RELEASES
    if definition.get('family') != 'laundry_room' or definition.get('task_id') not in TASK_IDS:
        return definition
    pins = read_json(PINS)
    if (pins.get('schema') != 'laundry_seed_catalog.v1' or not isinstance(pins.get('tasks'), dict)
            or set(pins['tasks']) - set(TASK_IDS)):
        raise ValueError('Invalid laundry seed catalog pins')
    if not pins['tasks']:
        return definition
    if set(pins['tasks']) != set(TASK_IDS):
        raise ValueError('Laundry publication requires all five tasks, without partial selector changes')
    row = pins['tasks'][definition['task_id']]
    original = RELEASES[definition['task_id']]
    if (row.get('family') != 'laundry_room' or row.get('task_id') != definition['task_id']
            or set(row.get('configurations', {})) != set(CONFIGURATION_IDS)
            or row.get('default_configuration_id') != original['default_configuration_id']):
        raise ValueError('Laundry catalog task/installation identity differs')
    options = {}
    for configuration, option in row['configurations'].items():
        pin = option['source_release']
        path = checked_file(Path(pin['path'])/'release_manifest.json', pin['manifest_sha256'])
        manifest = read_json(path)
        if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
                or manifest.get('task_id') != definition['task_id']
                or manifest.get('configuration_id') != configuration or manifest.get('seed_values') != list(range(10))
                or option.get('release_schema') != SCHEMA or option.get('release_schema_version') != VERSION):
            raise ValueError('Laundry catalog pin names another release')
        old = original['configurations'][configuration]
        parent = old['source_release']
        if manifest.get('base_release') != {**parent, 'schema': old['release_schema']}:
            raise ValueError('Laundry catalog release differs from its original installation parent')
        checked_file(Path(parent['path'])/'release_manifest.json', parent['manifest_sha256'])
        task = read_json(checked_file(path.parent/'task.json', manifest['files_sha256'].get('task.json')))
        expected = deepcopy(old)
        expected.update(source_release=deepcopy(pin), release_schema=SCHEMA, release_schema_version=VERSION,
                        qualification=deepcopy(QUALIFICATION))
        expected['rules']['contract'] = deepcopy(task['public_rules'])
        if option != expected:
            raise ValueError('Laundry catalog must preserve original robot, layout, objects and unvalidated scope')
        attached = definition.get('robot_configurations', {}).get(configuration, {})
        if attached.get('source_release') != parent:
            raise ValueError('Laundry catalog must bind the attached original installation')
        options[configuration] = deepcopy(option)
        for key in ('label_zh', 'description_zh'):
            if not isinstance(attached.get(key), str) or not attached[key]:
                raise ValueError('Laundry installation is missing its original public description')
            options[configuration][key] = attached[key]
    selected = row['configurations'][row['default_configuration_id']]
    if row.get('base_release') != selected['source_release']:
        raise ValueError('Laundry default release differs from its selected installation')
    result = deepcopy(definition)
    result.update(default_configuration_id=row['default_configuration_id'],
                  robot_configurations=options,
                  source_release=deepcopy(selected['source_release']),
                  release_schema=SCHEMA, release_schema_version=VERSION)
    for key in ('robot_requirements', 'objects', 'rules', 'qualification'):
        result[key] = deepcopy(selected[key])
    result['layout'].update(deepcopy(selected['layout']))
    return result
