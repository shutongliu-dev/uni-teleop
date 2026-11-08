"""Atomic current-catalog overlay; old definitions and releases remain immutable."""
from copy import deepcopy
from pathlib import Path

from . import SCHEMA, VERSION
from .assets import checked_file, file_hash, SUPPORTED_VERSIONS
from .inputs import canonical_hash
from .release import read, _configuration, legacy_single_menu

PINS = Path(__file__).with_name('catalog_pins.json')
QUALIFICATION = {'status': 'prepared_unvalidated', 'native_preflight': False,
    'workspace_ik': False, 'workspace_clearance': False, 'physics_accepted': False,
    'robot_manipulation': False, 'agent_bench_interface': False, 'inherited_qualification': False,
    'scope': 'New robot and ready inputs only. Historical task/robot validation and policy outcomes do not transfer.'}


def _options(definition):
    return definition.get('robot_configurations') or {'fixed': definition}


def _requirements(original, robot):
    result = deepcopy(original)
    if robot['hand_id'] == 'fr3_sharpa_instances':
        versions = {row['assembly_version'] for row in robot['instances'].values()}
        if len(versions) != 1 or not versions <= SUPPORTED_VERSIONS:
            raise ValueError('Composite robot catalog has mixed or unknown physical assembly versions')
        result.update(assembly_version=next(iter(versions)),
            instance_source='Two independently namespaced right SHARPA L-mount assemblies; original front/rear bases retained.')
    else:
        result.update(hand_id=robot['hand_id'], assembly_version=robot['assembly_version'])
    # Any qualification prose on the old robot is not evidence for this revision.
    if 'qualification' in result:
        result['qualification'] = 'New robot installation; validation evidence is external and not inherited.'
    return result


def _checked_option(definition, key, original, pin):
    path = checked_file(Path(pin['path'])/'release_manifest.json', pin['manifest_sha256'])
    value = read(path)
    parent = original['source_release']
    expected_parent = {**parent, 'schema': original['release_schema']}
    if (value.get('schema') != SCHEMA or value.get('schema_version') != VERSION
            or value.get('family') != definition['family'] or value.get('task_id') != definition['task_id']
            or value.get('base_release') != expected_parent
            or any(value.get(k) is not False for k in ('physics_accepted','task_policy_success_claimed','inherited_qualification'))):
        raise ValueError('Robot revision catalog differs from its exact original task/installation')
    checked_file(Path(parent['path'])/'release_manifest.json', parent['manifest_sha256'])
    old = read(Path(parent['path'])/'release_manifest.json')
    scene = read(checked_file(path.parent/'scene.json', value['files_sha256']['scene.json']))
    old_scene = read(checked_file(Path(parent['path'])/'scene.json', old['files_sha256']['scene.json']))
    expected_configuration = _configuration(old, old_scene)
    legacy_single = (key == 'single_arm' and expected_configuration is None
                     and legacy_single_menu(old, old_scene))
    if key != 'fixed' and expected_configuration != key and not legacy_single:
        raise ValueError('Original public menu differs from its internal installation identity')
    if value.get('configuration_id') != expected_configuration:
        raise ValueError('Robot revision changed a fixed or selectable installation identity')
    result = deepcopy(original)
    result.update(source_release=deepcopy(pin), release_schema=SCHEMA, release_schema_version=VERSION,
        qualification=deepcopy(QUALIFICATION), robot_requirements=_requirements(original['robot_requirements'],scene['robot']))
    return result


def publication_row(definition, releases):
    """Build input pins after full validation; no reports or success flags admitted."""
    from bench.isaacsim.release import validate
    originals = _options(definition)
    if set(releases) != set(originals):
        raise ValueError('Robot revision must preserve the original menu or fixed installation')
    pins = {}
    for key, root in releases.items():
        root = Path(root).resolve(); validate(root)
        pin = {'path':str(root),'manifest_sha256':file_hash(root/'release_manifest.json')}
        _checked_option(definition,key,originals[key],pin); pins[key]=pin
    return {'family':definition['family'],'task_id':definition['task_id'],
        'default_configuration_id':definition.get('default_configuration_id'),
        'parent_default_release':deepcopy(definition['source_release']),
        'parents':{key:deepcopy(row['source_release']) for key,row in originals.items()},'releases':pins}


def bind_definition(definition):
    pins = read(PINS)
    if pins.get('schema') != 'robot_revision_catalog.v1' or not isinstance(pins.get('tasks'),dict):
        raise ValueError('Invalid robot revision catalog')
    if not pins['tasks']:
        return definition
    if set(pins['tasks']) != set(CURRENT_TASKS):
        raise ValueError('Robot revision catalog must replace all 51 current tasks atomically')
    identity = definition['family']+'/'+definition['task_id']
    if identity not in CURRENT_TASKS:
        return definition  # Historical/superseded tasks never gain this overlay.
    row = pins['tasks'][identity]; originals = _options(definition)
    if (set(row) != {'family','task_id','default_configuration_id','parent_default_release','parents','releases'}
            or row.get('family') != definition['family'] or row.get('task_id') != definition['task_id']
            or row.get('default_configuration_id') != definition.get('default_configuration_id')
            or row.get('parent_default_release') != definition['source_release']
            or row.get('parents') != {k:v['source_release'] for k,v in originals.items()}
            or set(row.get('releases',{})) != set(originals)):
        raise ValueError('Robot revision changed original catalog selection semantics')
    updated = {k:_checked_option(definition,k,old,row['releases'][k]) for k,old in originals.items()}
    if 'robot_configurations' not in definition:
        return updated['fixed']
    result = deepcopy(definition)
    default = definition.get('default_configuration_id','single_arm')
    selected = updated[default]
    result['robot_configurations'] = updated
    for key in ('source_release','release_schema','release_schema_version','qualification','robot_requirements'):
        result[key] = deepcopy(selected[key])
    return result


def default_service_release(legacy):
    """Resolve the ordinary HTTP default only after a complete current overlay.

    Explicit operator releases never call this resolver. Empty pins retain the
    historical default; a partial/broken migration fails instead of falling back.
    Imports are delayed until server construction to avoid catalog import cycles.
    """
    pins = read(PINS)
    if pins.get('schema') != 'robot_revision_catalog.v1' or not isinstance(pins.get('tasks'), dict):
        raise ValueError('Invalid robot revision catalog')
    if not pins['tasks']:
        return Path(legacy)
    if set(pins['tasks']) != set(CURRENT_TASKS):
        raise ValueError('Default service requires the complete current robot overlay')
    from scenes.workrooms.catalog import task_definition
    from scenes.workrooms.parameterized.catalog import DEFAULT_PROFILE
    current = task_definition('card_room', 'task0')
    # dexbench: with the original robot as default profile, task0 is not a revision release.
    if ((DEFAULT_PROFILE == 'l_mount' and current.get('release_schema') != SCHEMA)
            or current.get('default_configuration_id', 'single_arm') != 'single_arm'
            or current.get('robot_configurations', {}).get('single_arm', {}).get('source_release') != current['source_release']):
        raise ValueError('Current default task0 is not the single-arm installation of the default robot')
    pin = current['source_release']
    checked_file(Path(pin['path'])/'release_manifest.json', pin['manifest_sha256'])
    return Path(pin['path'])


# Current task inventory at the explicit robot migration, not historical definitions.
CURRENT_TASKS = frozenset(('card_room/barrier_race', 'card_room/barrier_race_practice', 'card_room/connect_four_5x5', 'card_room/disk_curling', 'card_room/disk_curling_practice', 'card_room/hex7x7', 'card_room/memory_pairs', 'card_room/memory_pairs_practice', 'card_room/polyomino_territory', 'card_room/polyomino_territory_practice', 'card_room/task0', 'card_room/tic_tac_toe_3x3', 'kitchen/breakfast_serving', 'kitchen/pantry_stocking', 'kitchen/pot_lid_placement', 'kitchen/produce_transfer', 'kitchen/surface_wiping', 'kitchen/table_clearing', 'kitchen/utensil_sorting', 'laboratory/cap_placement', 'laboratory/pipette_transfer', 'laboratory/sample_sorting', 'laboratory/solid_weighing', 'laboratory/tube_clamp_transfer', 'laboratory/vial_racking', 'laundry_room/shirt_folding', 'laundry_room/towel_flattening', 'laundry_room/towel_folding', 'laundry_room/towel_stacking', 'laundry_room/trousers_folding', 'living_room/remote_operation', 'living_room/storage_box', 'living_room/tower_building', 'living_room/tower_extraction', 'living_room/water_pouring', 'mechanical_assembly/cooperative_insertion', 'mechanical_assembly/fastener_assembly', 'mechanical_assembly/gear_train_assembly', 'mechanical_assembly/nist_belt_drive_assembly', 'mechanical_assembly/nist_cable_routing', 'mechanical_assembly/nist_connector_insertion', 'mechanical_assembly/nist_wire_harness', 'mechanical_assembly/parts_sorting', 'mechanical_assembly/peg_insertion', 'mechanical_assembly/shaft_spacer_assembly', 'study/book_retrieval', 'study/book_shelving', 'study/book_stacking', 'study/drawer_retrieval', 'study/reading_stand', 'study/stationery_sorting'))
