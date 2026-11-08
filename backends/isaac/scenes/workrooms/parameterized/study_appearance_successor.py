"""Immutable cover-only successors of explicitly selected physical instances.

Historical authors and sampled parameters are retained. New cover recipes have
their own complete source identity. Runtime checks bind the frozen recipe and
files; expensive pixel and USD invariance audits belong to asset preparation
and independent development evidence, never ordinary episode startup.
"""
from copy import deepcopy
import json
from pathlib import Path

from bench.paths import load_json
from scenes.workrooms.seed_geometry import file_hash, write_json
from .spec import canonical_hash

VERSION = 'study_cover_appearance_successor.v1'
RECIPE_KEY = 'cover_background_successor'
TASKS = ('book_stacking', 'book_retrieval', 'book_shelving',
         'drawer_retrieval', 'reading_stand')
SOURCE_FILES = (
    'scenes/workrooms/parameterized/study_appearance_successor.py',
    'scenes/workrooms/parameterized/workroom.py',
    'scenes/workrooms/parameterized/spec.py',
    'scenes/workrooms/parameterized/appearance.py',
    'scenes/workrooms/parameterized/usd_authoring.py',
    'scenes/workrooms/study/book_cover_appearance.py',
    'scenes/workrooms/study/plain_book_cover_appearance.py',
)


def _merge_pins(*mappings):
    pins = {}
    for mapping in mappings:
        for path, digest in mapping.items():
            if path in pins and pins[path] != digest:
                raise ValueError('Cover successor would replace a frozen dependency pin: '+path)
            pins[path] = digest
    return pins


def _source_identity(root):
    root = Path(root).resolve(); path = root/'source_snapshot_manifest.json'
    manifest = load_json(path)
    files = {str(root/name): manifest['files_sha256'][name] for name in SOURCE_FILES}
    if any(file_hash(p) != h for p, h in files.items()):
        raise ValueError('Cover author differs from complete frozen source')
    return {'snapshot_root': str(root), 'manifest_path': str(path),
            'manifest_sha256': file_hash(path), 'source_files_sha256': files}


def _cover_pins(recipe):
    """All inputs and generated pixels referenced by the narrow adapter."""
    mappings = []
    for key in ('library_manifest', 'source_manifest', 'source_usd'):
        if key in recipe:
            mappings.append({recipe[key]['path']: recipe[key]['sha256']})
    for row in recipe['artifacts']:
        mappings.append({row['path']: row['sha256']})
        for path_key, hash_key in (('source_path', 'source_sha256'),
                ('font_path', 'font_sha256'), ('mask_path', 'mask_file_sha256')):
            if path_key in row:
                mappings.append({row[path_key]: row[hash_key]})
    return _merge_pins(*mappings)


def _metadata_successor(prior_data, before_layer, after_layer):
    expected = deepcopy(before_layer)
    for key in ('usd_path', 'sha256', 'dependencies_sha256'):
        expected[key] = deepcopy(after_layer[key])
    if expected != after_layer or after_layer['usd_path'] == before_layer['usd_path']:
        raise ValueError('Cover successor must change only the task USD and its pins')
    if any(after_layer['dependencies_sha256'].get(p) != h
           for p, h in before_layer['dependencies_sha256'].items()):
        raise ValueError('Cover successor dropped a prior task-layer pin')
    data = deepcopy(prior_data)
    rows = [r for r in data['scene_changes'] if r['path'] == ['task_layers']]
    if len(rows) != 1:
        raise ValueError('Expected the selected study task-layer metadata patch')
    layers = [r for r in rows[0]['after'] if r['id'] == 'task']
    if len(layers) != 1 or layers[0] != before_layer:
        raise ValueError('Prior task layer differs from selected metadata')
    layers[0].clear(); layers[0].update(deepcopy(after_layer))
    return data


def validate_successor(manifest, data):
    """JSON/source/file binding only; no PIL, USD or historical outcome gates."""
    marker = manifest.get('appearance_successor')
    recipe = manifest.get('applied_parameters', {}).get('appearance', {}).get(RECIPE_KEY)
    if marker is None:
        if recipe is not None or 'appearance_author_source' in manifest:
            raise ValueError('Orphan cover successor recipe/source')
        return
    if (marker.get('schema') != VERSION or manifest['family'] != 'study'
            or manifest['task_id'] not in TASKS or manifest['seed'] == 0
            or manifest['parameter_instance']['canonical'] is not False):
        raise ValueError('Unsupported cover successor identity')
    pin = marker['previous_asset']
    if file_hash(pin['path']) != pin['sha256']:
        raise ValueError('Cover predecessor changed')
    prior = load_json(Path(pin['path']))
    if 'appearance_successor' in prior:
        raise ValueError('Cover successor cannot silently stack another cover edit')
    source = manifest['appearance_author_source']
    if source != _source_identity(source['snapshot_root']):
        raise ValueError('Cover author source identity differs')
    if (not isinstance(recipe, dict) or recipe.get('canonical') is not False
            or recipe.get('instance_sha256') != manifest['parameter_instance']['instance_sha256']
            or recipe.get('pattern_id') != manifest['parameter_instance']['appearance']['pattern_id']
            or recipe.get('sampled_pattern_scale_m') != manifest['parameter_instance']['appearance']['pattern_scale_m']
            or recipe.get('pattern_scale_m_applied') is not False):
        raise ValueError('Cover recipe disagrees with the retained parameter instance')
    expected_schema = ('study_book_cover_background_v1' if manifest['task_id'] in TASKS[:3]
                       else 'study_plain_book_cover_background_v1')
    if recipe.get('schema') != expected_schema or not recipe.get('artifacts'):
        raise ValueError('Missing reviewed cover recipe')
    expected_applied = deepcopy(prior['applied_parameters'])
    if RECIPE_KEY in expected_applied['appearance']:
        raise ValueError('Predecessor already has a cover successor')
    expected_applied['appearance'][RECIPE_KEY] = deepcopy(recipe)
    if manifest['applied_parameters'] != expected_applied:
        raise ValueError('Cover successor changed prior physical or appearance parameters')
    # Preserve the original layout/physics author, parameters and every other
    # manifest field. The new appearance author is explicitly separate.
    expected_manifest = deepcopy(prior)
    for key in ('metadata', 'metadata_sha256', 'dependencies_sha256',
                'physical_instance_sha256', 'applied_parameters',
                'appearance_successor', 'appearance_author_source'):
        expected_manifest[key] = deepcopy(manifest[key])
    if expected_manifest != manifest or manifest.get('native_qualified') is not False:
        raise ValueError('Cover successor changed immutable predecessor fields')
    if file_hash(prior['metadata']) != prior['metadata_sha256']:
        raise ValueError('Cover predecessor metadata changed')
    prior_data = load_json(Path(prior['metadata']))
    if data != _metadata_successor(prior_data, marker['previous_task_layer'], marker['task_layer']):
        raise ValueError('Cover successor changed task, reset or physical metadata')
    required = _merge_pins(prior['dependencies_sha256'], {pin['path']: pin['sha256']},
        source['source_files_sha256'], {source['manifest_path']: source['manifest_sha256']},
        _cover_pins(recipe), {marker['task_layer']['usd_path']: marker['task_layer']['sha256']},
        marker['task_layer']['dependencies_sha256'], {manifest['metadata']: manifest['metadata_sha256']})
    if manifest['dependencies_sha256'] != required:
        raise ValueError('Cover successor dependency inventory differs')


def _value(value):
    """Stable comparison of complete USD property/metadata values."""
    from pxr import Sdf
    if isinstance(value, Sdf.AssetPath):
        return ('asset', value.path)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _value(v)) for k, v in value.items()))
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    try:
        return tuple(_value(v) for v in value)
    except TypeError:
        return str(value)


def stage_properties(stage):
    """Complete composed prim metadata, attributes, samples and relationships."""
    return {str(p.GetPath()): {
        'metadata': _value(p.GetAllMetadata()),
        'attributes': {a.GetName(): {'metadata': _value(a.GetAllMetadata()),
            'value': _value(a.Get()), 'connections': tuple(map(str, a.GetConnections())),
            'samples': tuple((t, _value(a.Get(t))) for t in a.GetTimeSamples())}
            for a in p.GetAttributes()},
        'relationships': {r.GetName(): {'metadata': _value(r.GetAllMetadata()),
            'targets': tuple(map(str, r.GetTargets()))} for r in p.GetRelationships()}}
        for p in [stage.GetPseudoRoot(), *stage.TraverseAll()]}


def check_only_cover_changes(before, after, recipe):
    """Author-time fail-closed check; never an episode admission outcome."""
    restored = deepcopy(after)
    if recipe['schema'] == 'study_book_cover_background_v1':
        for row in recipe['artifacts']:
            prim, name = row['texture_file_attribute'].rsplit('.', 1)
            expected = ('asset', row['path'])
            if restored[prim]['attributes'][name]['value'] != expected:
                raise ValueError('Cover texture binding differs from generated pixels')
            restored[prim]['attributes'][name]['value'] = before[prim]['attributes'][name]['value']
    else:
        material = recipe['new_material']
        added = set(after)-set(before)
        expected = {material, *(material+'/'+n for n in ('Surface', 'UV', 'UVScale', 'Albedo'))}
        if added != expected:
            raise ValueError('Plain cover added unexpected USD prims')
        for path in added:
            row = restored.pop(path)
            if any('physics' in name.lower() for name in row['attributes']):
                raise ValueError('Cover material introduced physical properties')
        for mesh in recipe['meshes']:
            rel = restored[mesh]['relationships']['material:binding']
            if rel['targets'] != (material,):
                raise ValueError('Plain cover binding differs')
            rel['targets'] = before[mesh]['relationships']['material:binding']['targets']
    if restored != before:
        raise ValueError('Cover edit changed nonappearance USD properties')


def prepare_asset_successor(parent_release, asset_manifest_path, output, *, expected_sha256):
    """Publish one nonzero seed, without resampling or overwriting any inputs."""
    from pxr import Usd
    from . import usd_authoring, workroom
    from scenes.workrooms.study import book_cover_appearance, plain_book_cover_appearance
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(Path(__file__).resolve().parents[3],
        required_modules=(__name__, workroom.__name__, usd_authoring.__name__,
            book_cover_appearance.__name__, plain_book_cover_appearance.__name__),
        entrypoint_file=Path(__file__))
    if not source or not source.get('runtime_imports_verified'):
        raise ValueError('Cover successor requires a complete frozen author source')
    prior_path = Path(asset_manifest_path).resolve()
    if file_hash(prior_path) != expected_sha256:
        raise ValueError('Selected prior asset differs from explicit pin')
    prior = load_json(prior_path)
    if (prior.get('family') != 'study' or prior.get('task_id') not in TASKS
            or prior.get('parameter_instance', {}).get('canonical') is not False
            or 'appearance_successor' in prior):
        raise ValueError('Expected an unmodified noncanonical study cover asset')
    parent = Path(parent_release).resolve()
    cfg = workroom._configuration(load_json(parent/'scene.json'), prior['task_id'])
    values = workroom.compose_case(parent, prior_path, cfg)
    layer = next(r for r in values['scene']['task_layers'] if r['id'] == 'task')
    before_layer = deepcopy(layer)
    stage = Usd.Stage.Open(Usd.Stage.Open(layer['usd_path']).Flatten())
    before = stage_properties(stage)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    actors, _ = workroom._groups(values, 'study', prior['task_id'])
    author = (book_cover_appearance.apply_book_cover_background if prior['task_id'] in TASKS[:3]
              else plain_book_cover_appearance.apply_plain_book_cover_background)
    recipe = author(stage, actors, prior['parameter_instance'], output/'covers')
    after = stage_properties(stage); check_only_cover_changes(before, after, recipe)
    target = output/'task.usda'; stage.GetRootLayer().Export(str(target))
    if stage_properties(Usd.Stage.Open(str(target))) != after:
        raise ValueError('Exported cover USD differs from isolated authored stage')
    layer.update(usd_path=str(target), sha256=file_hash(target))
    layer['dependencies_sha256'] = usd_authoring.task_layer_dependencies(layer)
    usd_authoring.check_task_layer_admission(layer)
    data = _metadata_successor(load_json(Path(prior['metadata'])), before_layer, layer)
    write_json(output/'metadata.json', data)
    appearance_source = _source_identity(source['snapshot_root'])
    deps = _merge_pins(prior['dependencies_sha256'], {str(prior_path): expected_sha256},
        appearance_source['source_files_sha256'],
        {appearance_source['manifest_path']: appearance_source['manifest_sha256']},
        _cover_pins(recipe), {str(target): file_hash(target)}, layer['dependencies_sha256'],
        {str(output/'metadata.json'): file_hash(output/'metadata.json')})
    successor = deepcopy(prior)
    successor['applied_parameters']['appearance'][RECIPE_KEY] = recipe
    successor.update(metadata=str(output/'metadata.json'), metadata_sha256=file_hash(output/'metadata.json'),
        dependencies_sha256=deps, appearance_author_source=appearance_source,
        appearance_successor={'schema': VERSION,
            'previous_asset': {'path': str(prior_path), 'sha256': expected_sha256},
            'previous_task_layer': before_layer, 'task_layer': deepcopy(layer)})
    text_version = ({'public_mass_text_version': prior['public_mass_text_version']}
                    if 'public_mass_text_version' in prior else {})
    successor['physical_instance_sha256'] = canonical_hash({'parameters': prior['parameter_instance'],
        'applied': successor['applied_parameters'], 'projections': {k: data[k] for k in workroom.PROJECTIONS},
        'dependencies': deps, **text_version})
    validate_successor(successor, data)
    write_json(output/'seed_manifest.json', successor)
    composed = workroom.compose_case(parent, output/'seed_manifest.json', cfg)
    if composed['scene'] != values['scene'] or any(composed[k] != values[k]
            for k in ('task', 'initial_state', *workroom.PROJECTIONS)):
        raise ValueError('Cover successor changed task, initial state or physical projections')
    return output/'seed_manifest.json'
