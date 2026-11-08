"""Published input pins only; validation outcomes never enter task admission."""
from copy import deepcopy
from pathlib import Path

from . import TASK_FAMILIES, SCHEMA, VERSION, configurations
from .common import checked_file, read_json

PINS = Path(__file__).with_name('catalog_pins.json')
UNVALIDATED = {'status': 'prepared_unvalidated', 'native_preflight': False,
              'manipulation': False, 'physics_accepted': False,
              'public_http_smoke': False, 'public_functions_smoke': False,
              'scope': 'Frozen randomized inputs; independent development evidence is not runtime admission data.'}


def bind_definition(definition):
    task_id = definition['task_id']
    if task_id not in TASK_FAMILIES or definition['family'] != TASK_FAMILIES[task_id]:
        return definition
    rows = read_json(PINS)
    if task_id not in rows:
        return definition
    row = rows[task_id]
    if (row['family'] != definition['family'] or row['parent_default_release'] != definition['source_release']
            or set(row['releases']) != set(configurations(task_id))):
        raise ValueError('Seed publication differs from the original task/installation set')
    result = deepcopy(definition)
    default = configurations(task_id)[0]
    originals = result.get('robot_configurations')
    if originals is not None:
        if set(originals) != set(row['releases']):
            raise ValueError('Randomization cannot change the original robot choices')
        for configuration_id, option in originals.items():
            if option['source_release'] != row['parents'][configuration_id]:
                raise ValueError('Seed publication has a different original robot release')
            option.update(source_release=deepcopy(row['releases'][configuration_id]),
                          release_schema=SCHEMA, release_schema_version=VERSION,
                          qualification=deepcopy(UNVALIDATED))
        # Keep host factory/base metadata, and publish physical hand chirality.
        originals['single_arm']['robot_requirements'] = {
            **deepcopy(definition['robot_requirements']),
            **originals['single_arm']['robot_requirements'], 'active_sides': ['right']}
    elif set(row['releases']) != {default}:
        raise ValueError('A fixed installation cannot gain implicit robot choices')
    result.update(source_release=deepcopy(row['releases'][default]),
                  release_schema=SCHEMA, release_schema_version=VERSION,
                  qualification=deepcopy(UNVALIDATED))
    result['layout']['initialization'] = 'Ten frozen physical initial cases selected before creation; seed 0 is canonical.'
    result['objects']['asset_bindings'] = 'Each seed pins actual geometry, colliders, appearance and mass inputs in its immutable release.'
    return result


def publication_row(definition, releases):
    from bench.isaacsim.release import validate
    from .common import file_hash
    task_id = definition['task_id']
    if definition['family'] != TASK_FAMILIES.get(task_id) or set(releases) != set(configurations(task_id)):
        raise ValueError('Publication needs the existing installations of an explicit ordinary task')
    originals = definition.get('robot_configurations') or {configurations(task_id)[0]: definition}
    pins = {}
    for configuration_id, path in releases.items():
        root = Path(path).resolve(); manifest = validate(root)
        expected = originals[configuration_id]['source_release']
        if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
                or manifest.get('task_id') != task_id or manifest.get('configuration_id') != configuration_id
                or manifest['base_release']['path'] != expected['path']
                or manifest['base_release']['manifest_sha256'] != expected['manifest_sha256']):
            raise ValueError('New input release differs from its exact original installation')
        checked_file(root/'release_manifest.json', file_hash(root/'release_manifest.json'))
        pins[configuration_id] = {'path': str(root), 'manifest_sha256': file_hash(root/'release_manifest.json')}
    return {'family': definition['family'], 'parent_default_release': deepcopy(definition['source_release']),
            'parents': {c: deepcopy(o['source_release']) for c, o in originals.items()}, 'releases': pins}
