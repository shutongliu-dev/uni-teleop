"""Author reset coordinates and appearance pins; no cloth solver or USD writes."""
from copy import deepcopy
from pathlib import Path

from . import ASSET_SCHEMA, GENERATOR_VERSION, PARAMETERIZED_VERSION, checked_generator_version, selection
from . import common
from .common import (canonical_hash, checked_file, checked_inventory, file_hash,
                     local_path, parent_inputs, read_json, verify_provenance, write_json)
from .cases import case_plan, transformed_metadata, validate_plan


def parent_visuals(scene):
    result = {row['id']: deepcopy(row['visual_profile']) for row in scene['cloths']}
    if len(result) != len(scene['cloths']):
        raise ValueError('Each original cloth needs one pinned visual profile')
    return result


def _visual_inventory(scene, profiles, seed, *, assets=True, parameters=None):
    from ..fabric_visual_release import check_profile_binding, profile_inventory
    from .appearance import validate_profile
    originals = parent_visuals(scene)
    if not isinstance(profiles, dict) or set(profiles) != set(originals):
        raise ValueError('Visual profile cloth inventory differs')
    if seed == 0:
        if profiles != originals:
            raise ValueError('Seed zero must preserve all canonical appearance pins')
        return profile_inventory(originals)
    check_profile_binding(scene, profiles)
    result = {}
    for name, pin in profiles.items():
        if not Path(pin['path']).resolve().is_relative_to(common.ASSET_ROOT):
            raise ValueError('New appearance must be in shared laundry randomization assets')
        result.update(validate_profile(pin, originals[name], seed, assets=assets, parameters=parameters))
    return result


def prepare_asset(parent_release, seed, output, *, visual_profiles=None,
                  generator_version=GENERATOR_VERSION, attempt=0):
    """Create one small immutable case, referencing original material assets."""
    parent = Path(parent_release).resolve()
    manifest, scene, task, _, references = parent_inputs(parent)
    configuration = manifest['configuration_id']; task_id = task['id']
    selection(task_id, seed, configuration)
    output = Path(output).resolve()
    if not output.is_relative_to(common.ASSET_ROOT) or output.exists():
        raise ValueError('Create a case only in a fresh shared laundry asset directory')
    parameterized = generator_version == PARAMETERIZED_VERSION
    provenance = common.source_provenance(parameterized=True) if parameterized else common.source_provenance()
    plan = case_plan(task_id, configuration, seed, task['asset_metadata'],
                     generator_version=generator_version, attempt=attempt)
    profiles = parent_visuals(scene) if visual_profiles is None else deepcopy(visual_profiles)
    visuals = _visual_inventory(scene, profiles, seed, parameters=plan.get('parameters'))
    metadata = transformed_metadata(task['asset_metadata'], plan)
    if parameterized:
        from .parameterized import material_metadata
        metadata = material_metadata(metadata, scene, plan)
    initial = {row['id']: row['initial_vertices_m'] for row in metadata['cloths']}
    output.mkdir(parents=True)
    write_json(output/'plan.json', plan)
    write_json(output/'initial_vertices.json', initial)
    value = {'schema': ASSET_SCHEMA, 'family': 'laundry_room', 'task_id': task_id,
             'configuration_id': configuration, 'seed': seed, 'generator_version': generator_version,
             'parent_release': {'path': str(parent), 'manifest_sha256': file_hash(parent/'release_manifest.json')},
             'parent_references': references,
             'dependencies_sha256': {**manifest['asset_files_sha256'], **visuals},
             'files_sha256': {n: file_hash(output/n) for n in ('plan.json', 'initial_vertices.json')},
             'visual_profiles': profiles,
             'parent_metadata_sha256': canonical_hash(task['asset_metadata']),
             'material_topology_and_physics_unchanged': not (parameterized and seed != 0),
             'target_coordinates_unchanged': True,
             'runtime_cloth_state_writes': False,
             'original_inputs_modified': False,
             'source_provenance': provenance,
             'license_scope': 'Original assets and texture license/source records remain hash-pinned; procedural albedo modulation is project-authored CC0.'}
    if parameterized:
        value['license_scope'] = 'Original assets and texture license/source records remain hash-pinned; procedural albedo and roughness prints are project-authored CC0.'
        value['parameterized_physics'] = {'parameters': deepcopy(plan['parameters']),
            'applied': deepcopy(metadata.get('parameterized_physics', {'canonical': True, 'applied': False})),
            'rest_mesh_topology_thickness_unchanged': True, 'solver_controller_unchanged': True}
    write_json(output/'asset_manifest.json', value)
    load_asset(parent, output/'asset_manifest.json', seed)
    return output/'asset_manifest.json'


def load_asset(parent_release, path, seed, *, assets=True, parent_values=None):
    parent, path = Path(parent_release).resolve(), Path(path).resolve()
    if not path.is_relative_to(common.ASSET_ROOT):
        raise ValueError('Case assets must be inside the shared laundry asset root')
    values = parent_values or parent_inputs(parent, assets=assets)
    original, scene, task, initial, references = values
    selection(task['id'], seed, original['configuration_id'])
    asset = read_json(path)
    checked_generator_version(task['id'], original['configuration_id'], asset.get('generator_version'))
    parameterized = asset['generator_version'] == PARAMETERIZED_VERSION
    expected_fields = {'schema', 'family', 'task_id', 'configuration_id', 'seed', 'generator_version',
        'parent_release', 'parent_references', 'dependencies_sha256', 'files_sha256', 'visual_profiles',
        'parent_metadata_sha256', 'material_topology_and_physics_unchanged', 'target_coordinates_unchanged',
        'runtime_cloth_state_writes', 'original_inputs_modified', 'source_provenance', 'license_scope'}
    if parameterized:
        expected_fields.add('parameterized_physics')
    if (set(asset) != expected_fields or asset.get('schema') != ASSET_SCHEMA
            or asset.get('family') != 'laundry_room' or asset.get('task_id') != task['id']
            or asset.get('configuration_id') != original['configuration_id']
            or type(asset.get('seed')) is not int or asset['seed'] != seed
            or asset['parent_release'] != {'path': str(parent), 'manifest_sha256': file_hash(parent/'release_manifest.json')}
            or asset['parent_references'] != references
            or asset['parent_metadata_sha256'] != canonical_hash(task['asset_metadata'])
            or asset['material_topology_and_physics_unchanged'] is not (not (parameterized and seed != 0))
            or asset['target_coordinates_unchanged'] is not True
            or asset['runtime_cloth_state_writes'] is not False or asset['original_inputs_modified'] is not False):
        raise ValueError('Invalid laundry seed asset identity or protected invariants')
    verify_provenance(asset['source_provenance'])
    if parameterized and set(asset['source_provenance']['author_sources_sha256']) != set(common.PARAMETERIZED_SOURCE_FILES):
        raise ValueError('Parameterized laundry requires its complete author source identity')
    if set(asset['files_sha256']) != {'plan.json', 'initial_vertices.json'}:
        raise ValueError('Unexpected laundry case asset files')
    for name, sha in asset['files_sha256'].items():
        checked_file(local_path(path.parent, name), sha)
    actual_files = {str(p.relative_to(path.parent)) for p in path.parent.rglob('*') if p.is_file()}
    if actual_files != {'asset_manifest.json', 'plan.json', 'initial_vertices.json'}:
        raise ValueError('Undeclared files in laundry case asset')
    plan = read_json(path.parent/'plan.json')
    validate_plan(plan, task['asset_metadata'])
    if plan['generator_version'] != asset['generator_version']:
        raise ValueError('Laundry asset and plan generator versions differ')
    if (plan['task_id'], plan['configuration_id'], plan['seed']) != (task['id'], original['configuration_id'], seed):
        raise ValueError('Case plan names another task/configuration/seed')
    metadata = transformed_metadata(task['asset_metadata'], plan)
    if parameterized:
        from .parameterized import material_metadata
        metadata = material_metadata(metadata, scene, plan)
        expected_physics = {'parameters': deepcopy(plan['parameters']),
            'applied': deepcopy(metadata.get('parameterized_physics', {'canonical': True, 'applied': False})),
            'rest_mesh_topology_thickness_unchanged': True, 'solver_controller_unchanged': True}
        if asset['parameterized_physics'] != expected_physics:
            raise ValueError('Parameterized laundry material recipe differs from actual inputs')
    positions = {row['id']: row['initial_vertices_m'] for row in metadata['cloths']}
    if read_json(path.parent/'initial_vertices.json') != positions:
        raise ValueError('Authored cloth reset differs from its exact declared transform')
    visuals = _visual_inventory(scene, asset['visual_profiles'], seed, assets=assets, parameters=plan.get('parameters'))
    dependencies = {**original['asset_files_sha256'], **visuals}
    if asset['dependencies_sha256'] != dependencies:
        raise ValueError('Case omitted or added an undeclared material/appearance dependency')
    checked_inventory(references, verify=assets)
    checked_inventory(dependencies, verify=assets)
    inventory = {**references, **dependencies, str(path): file_hash(path),
                 **{str(path.parent/n): h for n, h in asset['files_sha256'].items()}}
    return asset, plan, metadata, inventory
