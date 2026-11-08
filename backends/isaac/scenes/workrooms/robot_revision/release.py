"""Immutable robot-only derivatives of exact current task publications."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from . import SCHEMA, VERSION
from .assets import file_hash, checked_file, read_asset_manifest, check_robot_binding
from .inputs import canonical_hash, derive_inputs, checked_revision
from .practice import derive_case, derive_practice_inputs
from bench.paths import DATA_ROOT, load_json, portable

SEED_SCHEMAS = frozenset({'living_room_randomized_release.v1', 'workroom_randomized_release.v1',
                           'laundry_randomized_release.v1'})
PRACTICE_SCHEMA = 'card_rsi_practice_release.v1'
PARENT_SCHEMAS = SEED_SCHEMAS | {PRACTICE_SCHEMA, 'card_game_expansion_release.v1',
    'robot_board_game_task_release.v1', 'card_room_placement_task_release.v1',
    'workroom_robot_configuration_release.v1', 'living_room_task_release.v1'}
SOURCE_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = Path(f"{DATA_ROOT}/datasets")
# Historical parents live in the data package's datasets/ (interface: the store's datasets/);
# RELEASE_ROOT is only where new revisions are published.
PARENT_ROOT = Path(f"{DATA_ROOT}/datasets")
SOURCE_FILES = tuple('scenes/workrooms/robot_revision/'+n+'.py' for n in
                     ('__init__', 'assets', 'inputs', 'practice', 'practice_runtime', 'release'))
OBSERVER_COMPATIBILITY_SOURCES = (
    'scenes/workrooms/robot_revision/laundry_observer.py',
    'scenes/workrooms/laundry_room/native_observer.py',
)


def read(path):
    return load_json(Path(path))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portable(value), indent=2, allow_nan=False)+'\n')


def local(root, name):
    root, relative = Path(root).resolve(), Path(name)
    path = root/relative
    if (relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents)):
        raise ValueError('Robot revision file escapes its immutable release')
    return path


def selection_kind(manifest):
    schema = manifest.get('schema')
    if schema in SEED_SCHEMAS: return 'finite_seed'
    if schema == PRACTICE_SCHEMA: return 'practice'
    if manifest.get('task_kind') == 'board_game': return 'fixed_game'
    return 'legacy_seed'


def _configuration(manifest, scene):
    if 'configuration_id' in manifest: return manifest['configuration_id']
    if manifest['task_id'] == 'cooperative_insertion': return 'dual_same_side'
    if (manifest.get('task_kind') in ('board_game','turn_game')
            or manifest.get('schema') in (PRACTICE_SCHEMA,'robot_board_game_task_release.v1','card_game_expansion_release.v1')):
        return None
    # Legacy task0/tower-building single releases never declared an internal
    # configuration. Their public single_arm menu is a separate catalog key.
    return scene.get('robot_configuration', {}).get('id')


def legacy_single_menu(manifest, scene):
    """A public single-arm option may wrap an undeclared legacy internal ID."""
    return (manifest.get('schema') in {'card_room_placement_task_release.v1', 'living_room_task_release.v1'}
        and 'configuration_id' not in manifest and 'robot_configuration' not in scene
        and manifest.get('task_kind') not in ('board_game', 'turn_game')
        and scene.get('robot', {}).get('hand_id') == 'sharpa_w01_right')


def _documents(parent, original, revision, formal_release):
    from bench.isaacsim.validation_epoch import memo_json
    # This narrowly reuses the pure all-seed JSON derivation. validate_release
    # continues to validate parent, local files, assets, source and inventories
    # before calling us, and compares every derived document afterwards.
    names = {'scene.json', 'task.json', 'initial_state.json'}
    kind = selection_kind(original)
    if kind == 'finite_seed':
        names.update(record[n] for record in original['cases'].values()
                     for n in ('scene', 'task', 'initial_state'))
    elif kind == 'practice':
        names.add(original['cases'])
    pins = {str(local(parent, name)): original['files_sha256'][name] for name in names}
    key = ['robot_revision_documents.v1', str(Path(parent).resolve()),
           original, revision, formal_release]
    return tuple(memo_json(key, pins,
        lambda: _documents_uncached(parent, original, revision, formal_release)))


def _documents_uncached(parent, original, revision, formal_release):
    """Deterministic JSON changes only; all other files are copied byte-for-byte."""
    scene, task, initial = [read(parent/(n+'.json')) for n in ('scene','task','initial_state')]
    checked_revision(revision, scene, initial)
    kind = selection_kind(original)
    docs, cases = {}, deepcopy(original.get('cases'))
    if kind == 'practice':
        if formal_release is None: raise ValueError('Practice revision needs its new formal parent')
        original_cases = read(local(parent, original['cases']))
        new_cases = {k:derive_case(c, task['asset_metadata'], revision_sha256=canonical_hash(revision),
            formal_release_sha256=formal_release['manifest_sha256']) for k,c in original_cases.items()}
        docs[original['cases']] = new_cases
        for key, case in new_cases.items(): docs['cases/'+key+'/case.json'] = case
        derived = derive_practice_inputs(scene, task, initial, revision, formal_release=formal_release)
    else:
        if formal_release is not None: raise ValueError('Only practice may select a formal parent revision')
        derived = derive_inputs(scene, task, initial, revision)
        if kind == 'finite_seed':
            if set(cases) != set(map(str, range(10))): raise ValueError('Original seed inventory is not exactly 0 through 9')
            for seed, record in cases.items():
                values = [read(local(parent, record[n])) for n in ('scene','task','initial_state')]
                result = derive_inputs(*values, revision)
                for name, value in zip(('scene','task','initial_state'), result): docs[record[name]] = value
                case = result[1]['reset_case']; docs[record['case']] = case
                record['case_sha256'], record['case_fingerprint'] = case['case_sha256'], case['case_fingerprint']
                if seed == '0' and canonical_hash(result) != canonical_hash(derived):
                    raise ValueError('Parent discovery inputs differ from its canonical seed zero')
    for name, value in zip(('scene','task','initial_state'), derived): docs[name+'.json'] = value
    return docs, cases


def _source_provenance():
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    proof = verify_if_snapshot(SOURCE_ROOT, required_modules=(__name__,
        'scenes.workrooms.robot_revision.inputs', 'scenes.workrooms.robot_revision.practice',
        'scenes.workrooms.robot_revision.assets'), entrypoint_file=Path(__file__))
    if not proof or proof.get('runtime_imports_verified') is not True:
        raise ValueError('Robot revision publication requires a complete frozen source')
    return {'schema':'robot_revision_source_provenance.v1', 'source_snapshot':str(SOURCE_ROOT),
        'source_manifest_sha256':file_hash(SOURCE_ROOT/'source_snapshot_manifest.json'),
        'sources_sha256':{p:file_hash(SOURCE_ROOT/p) for p in SOURCE_FILES},
        'original_assets_modified':False, 'original_releases_modified':False,
        'runtime_object_writes':False, 'inherited_qualification':False}


def _verify_source(value):
    root = Path(value['source_snapshot']).resolve()
    if value.get('schema') != 'robot_revision_source_provenance.v1': raise ValueError('Unknown robot revision author identity')
    path = checked_file(root/'source_snapshot_manifest.json', value['source_manifest_sha256'])
    snapshot = read(path)
    if (snapshot.get('schema') != 'complete_benchmark_source_snapshot_v1'
            or Path(snapshot.get('snapshot_root', '')).resolve() != root
            or set(value.get('sources_sha256', {})) != set(SOURCE_FILES)):
        raise ValueError('Robot revision author snapshot is incomplete or belongs to another root')
    for name, expected in value['sources_sha256'].items():
        if snapshot.get('files_sha256', {}).get(name) != expected:
            raise ValueError('Robot revision source is absent from its author snapshot')
        checked_file(local(root,name), expected)
    if any(value.get(k) is not False for k in ('original_assets_modified','original_releases_modified',
                                               'runtime_object_writes','inherited_qualification')):
        raise ValueError('Robot revision cannot modify old inputs or inherit qualification')


def _formal(reference, original, revision, *, assets):
    from bench.isaacsim.release import validate
    if original['schema'] != PRACTICE_SCHEMA:
        if reference is not None: raise ValueError('Unexpected formal parent revision')
        return
    if not isinstance(reference, dict) or set(reference) != {'path','manifest_sha256'}:
        raise ValueError('Practice requires a complete new formal release reference')
    root = Path(reference['path']).resolve(); checked_file(root/'release_manifest.json', reference['manifest_sha256'])
    formal = validate(root, assets=assets)
    if (formal.get('schema') != SCHEMA or formal.get('selection_kind') != 'fixed_game'
            or formal.get('task_id') != original['task_id'].removesuffix('_practice')
            or formal.get('robot_revision_sha256') != canonical_hash(revision)
            or formal.get('base_release',{}).get('manifest_sha256') != original['parent_release']['manifest_sha256']):
        raise ValueError('Practice must use the matching formal task, robot revision and original geometry')


def _observer_compatibility(parent, original, revision, version, provenance):
    """Derive one explicit frame-reader fix; historical observers stay exact."""
    from . import laundry_observer as compatibility
    from .assets import SUPPORTED_VERSIONS
    scene = read(parent/'scene.json')
    if (version != compatibility.VERSION or original['schema'] != 'laundry_randomized_release.v1'
            or scene.get('workroom', {}).get('family') != 'laundry_room'
            or revision['robot'].get('assembly_version') not in SUPPORTED_VERSIONS):
        raise ValueError('Unsupported native observer compatibility revision')
    name = compatibility.OBSERVER_PATH
    if name not in original['files_sha256']:
        raise ValueError('Original laundry observer is absent from its frozen inventory')
    source_root = Path(provenance['source_snapshot'])
    snapshot = read(checked_file(source_root/'source_snapshot_manifest.json',
                                 provenance['source_manifest_sha256']))
    sources = {}
    for relative in OBSERVER_COMPATIBILITY_SOURCES:
        expected = snapshot.get('files_sha256', {}).get(relative)
        if expected is None:
            raise ValueError('Observer compatibility author is absent from its frozen source')
        checked_file(local(source_root, relative), expected)
        sources[relative] = expected
    original_path = checked_file(local(parent, name), original['files_sha256'][name])
    patched = compatibility.patch_observer(original_path.read_bytes())
    # The generated file is precisely the frozen author's observer, never a
    # current mutable implementation silently substituted for that author.
    if hashlib.sha256(patched).hexdigest() != sources[OBSERVER_COMPATIBILITY_SOURCES[1]]:
        raise ValueError('Observer compatibility differs from its frozen author')
    record = {'schema': 'robot_revision_observer_compatibility.v1', 'version': version,
        'file': name, 'parent_sha256': original['files_sha256'][name],
        'derived_sha256': hashlib.sha256(patched).hexdigest(), 'sources_sha256': sources}
    return record, {name: patched}


def prepare_release(parent_release, output, *, revision, robot_asset_manifest,
                    configuration_id=None, formal_parent_revision=None, observer_compatibility=None):
    """Publish one installation only after robot references/ready map are explicit."""
    from bench.isaacsim.release import validate
    provenance = _source_provenance()
    parent, output = Path(parent_release).resolve(), Path(output).resolve()
    original = validate(parent)
    if original.get('schema') not in PARENT_SCHEMAS: raise ValueError('Unsupported or nested robot revision parent')
    if output.exists() or not output.is_relative_to(RELEASE_ROOT):
        raise ValueError('Robot revisions require a fresh immutable shared release directory')
    scene = read(parent/'scene.json')
    if configuration_id != _configuration(original, scene):
        raise ValueError('Robot revision differs from the original installation identity')
    asset_path = Path(robot_asset_manifest).resolve()
    asset = read_asset_manifest(asset_path, expected_sha256=revision['robot_asset_manifest_sha256'])
    check_robot_binding(revision,asset)
    _formal(formal_parent_revision, original, revision, assets=True)
    docs, cases = _documents(parent, original, revision, formal_parent_revision)
    compatibility, observer_files = None, {}
    if observer_compatibility is not None:
        compatibility, observer_files = _observer_compatibility(
            parent, original, revision, observer_compatibility, provenance)
    dependencies = {**original['asset_files_sha256'], **asset['files_sha256'],
                    str(asset_path):file_hash(asset_path), str(parent/'release_manifest.json'):file_hash(parent/'release_manifest.json')}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'-', dir=output.parent))
    try:
        for name, expected in original['files_sha256'].items():
            source = checked_file(local(parent,name), expected); target=local(staging,name)
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
        for name, value in docs.items(): write(local(staging,name),value)
        for name, value in observer_files.items(): local(staging,name).write_bytes(value)
        write(staging/'robot_revision.json',revision)
        write(staging/'robot_revision_source.json',provenance)
        if 'asset_file_hashes.json' in original['files_sha256']:
            write(staging/'asset_file_hashes.json',{p:{'sha256':h} for p,h in sorted(dependencies.items())})
        files={str(p.relative_to(staging)):file_hash(p) for p in sorted(staging.rglob('*')) if p.is_file()}
        entry=deepcopy(original['tasks'][0])
        for name in ('scene','task','initial_state'):
            entry[name]=name+'.json';entry[name+'_sha256']=files[name+'.json']
        manifest={'schema':SCHEMA,'schema_version':VERSION,'task_id':original['task_id'],
            'family':scene['workroom']['family'],'configuration_id':configuration_id,
            'task_kind':original.get('task_kind',entry.get('task_kind','single_agent')),
            'selection_kind':selection_kind(original),'parent_schema':original['schema'],
            'parent_schema_version':original.get('schema_version'),
            'base_release':{'path':str(parent),'manifest_sha256':file_hash(parent/'release_manifest.json'),'schema':original['schema']},
            'robot_revision':'robot_revision.json','robot_revision_sha256':canonical_hash(revision),
            'robot_asset_manifest':{'path':str(asset_path),'sha256':file_hash(asset_path)},
            'formal_parent_revision':deepcopy(formal_parent_revision),
            'source_provenance':'robot_revision_source.json','tasks':[entry],
            'files_sha256':files,'files':{n:{'sha256':h} for n,h in files.items()},
            'asset_files_sha256':dependencies,'status':'prepared_unvalidated',
            'physics_accepted':False,'task_policy_success_claimed':False,'inherited_qualification':False,
            'original_release_modified':False,'prepared_at_utc':datetime.now(timezone.utc).isoformat()}
        for key in ('runtime_identity','physics_profile','physics_profile_sha256','public_rules','frozen_evaluator'):
            if key in original: manifest[key]=deepcopy(original[key])
        if compatibility is not None: manifest['observer_compatibility'] = compatibility
        if cases is not None: manifest['cases']=cases
        if manifest['selection_kind']=='finite_seed': manifest['seed_values']=list(range(10))
        if manifest['selection_kind']=='practice': manifest['parent_release']=deepcopy(formal_parent_revision)
        write(staging/'release_manifest.json',manifest)
        validate_release(staging,manifest)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise


def validate_release(root, manifest, *, assets=True):
    return validate_release_with_parent(root, manifest, assets=assets)[0]


def validate_release_with_parent(root, manifest, *, assets=True):
    """validate_release() and {resolved parent root: the validate(parent) result it checked}."""
    from bench.isaacsim.release import validate
    root=Path(root).resolve()
    if (canonical_hash(read(root/'release_manifest.json')) != canonical_hash(manifest)
            or manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
            or manifest.get('status') != 'prepared_unvalidated'
            or any(manifest.get(k) is not False for k in ('physics_accepted','task_policy_success_claimed',
                                                         'inherited_qualification','original_release_modified'))):
        raise ValueError('Robot revision identity or qualification flags changed')
    base=manifest['base_release'];parent=Path(base['path']).resolve()
    if parent==root or base.get('schema') not in PARENT_SCHEMAS or not parent.is_relative_to(PARENT_ROOT):
        raise ValueError('Robot revision must bind an independent historical parent')
    checked_file(parent/'release_manifest.json',base['manifest_sha256']);original=validate(parent,assets=assets)
    if (base['schema']!=original['schema'] or manifest['parent_schema']!=original['schema']
            or manifest['parent_schema_version']!=original.get('schema_version')
            or manifest['task_id']!=original['task_id'] or manifest['selection_kind']!=selection_kind(original)
            or manifest['configuration_id']!=_configuration(original,read(parent/'scene.json'))):
        raise ValueError('Robot revision parent task/selection/installation differs')
    files=manifest['files_sha256']
    expected_names=set(original['files_sha256'])|{'robot_revision.json','robot_revision_source.json'}
    if (set(files)!=expected_names or manifest.get('files')!={n:{'sha256':h} for n,h in files.items()}
            or {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} != expected_names|{'release_manifest.json'}):
        raise ValueError('Robot revision file inventory is incomplete or undeclared')
    for name, expected in files.items(): checked_file(local(root,name),expected)
    if manifest.get('robot_revision')!='robot_revision.json' or manifest.get('source_provenance')!='robot_revision_source.json':
        raise ValueError('Robot revision must use its declared local derivation inputs')
    revision=read(root/'robot_revision.json')
    if canonical_hash(revision)!=manifest['robot_revision_sha256']: raise ValueError('Robot revision descriptor hash differs')
    provenance = read(root/'robot_revision_source.json')
    _verify_source(provenance)
    observer_files = {}
    if 'observer_compatibility' in manifest:
        record = manifest['observer_compatibility']
        if not isinstance(record, dict):
            raise ValueError('Observer compatibility requires an explicit provenance record')
        expected, observer_files = _observer_compatibility(
            parent, original, revision, record.get('version'), provenance)
        if canonical_hash(record) != canonical_hash(expected):
            raise ValueError('Observer compatibility provenance or file identity changed')
        for name, value in observer_files.items():
            if local(root, name).read_bytes() != value:
                raise ValueError('Observer compatibility file differs from its exact derivation')
    reference=manifest['robot_asset_manifest'];asset=read_asset_manifest(reference['path'],expected_sha256=reference['sha256'],assets=assets)
    if revision['robot_asset_manifest_sha256']!=reference['sha256']: raise ValueError('Robot descriptor and immutable asset pin differ')
    check_robot_binding(revision,asset)
    _formal(manifest.get('formal_parent_revision'),original,revision,assets=assets)
    docs,cases=_documents(parent,original,revision,manifest.get('formal_parent_revision'))
    dependencies={**original['asset_files_sha256'],**asset['files_sha256'],str(Path(reference['path']).resolve()):reference['sha256'],
                  str(parent/'release_manifest.json'):base['manifest_sha256']}
    if manifest['asset_files_sha256']!=dependencies: raise ValueError('Robot revision dependency closure differs')
    if assets:
        for p,h in dependencies.items():checked_file(p,h)
    if 'asset_file_hashes.json' in original['files_sha256']:
        docs['asset_file_hashes.json']={p:{'sha256':h} for p,h in sorted(dependencies.items())}
    for name,expected in docs.items():
        if canonical_hash(read(local(root,name)))!=canonical_hash(expected):
            raise ValueError('Robot revision changed protected inputs: '+name)
    for name,expected in original['files_sha256'].items():
        if name not in docs and name not in observer_files and files[name]!=expected:
            raise ValueError('Robot revision changed frozen rules/physics/assets: '+name)
    entry=deepcopy(original['tasks'][0])
    for name in ('scene','task','initial_state'):
        entry[name]=name+'.json';entry[name+'_sha256']=files[name+'.json']
    if (manifest['tasks']!=[entry] or manifest.get('cases')!=cases
            or manifest['family']!=read(parent/'scene.json')['workroom']['family']
            or manifest['task_kind']!=original.get('task_kind',entry.get('task_kind','single_agent'))):
        raise ValueError('Robot revision discovery/case inventory differs')
    for key in ('runtime_identity','physics_profile','physics_profile_sha256','public_rules','frozen_evaluator'):
        if manifest.get(key)!=original.get(key):raise ValueError('Robot revision changed original scoring/physics identity')
    if manifest['selection_kind']=='finite_seed' and manifest.get('seed_values')!=list(range(10)):
        raise ValueError('Robot revision lost an original frozen seed')
    if manifest['selection_kind']=='practice' and manifest.get('parent_release')!=manifest['formal_parent_revision']:
        raise ValueError('Practice public parent differs from the selected robot revision')
    return deepcopy(manifest), {str(parent): original}


def task_inputs(root,task_id,manifest,*,seed=0,phase='nominal',practice=None,parents=None):
    """parents optionally maps the resolved parent root to the validate(parent) result
    that this revision's own validate(root) produced; otherwise the parent is validated here."""
    from bench.isaacsim.release import task_inputs as original_inputs, validated_task_inputs
    if task_id!=manifest['task_id'] or phase!='nominal':raise ValueError('Robot revision requires its exact nominal task')
    root=Path(root).resolve();parent=Path(manifest['base_release']['path'])
    validated=(parents or {}).get(str(parent.resolve()))
    if validated is None:
        _,scene,task,initial=original_inputs(parent,task_id,phase,seed=seed,practice=practice)
    else:
        _,scene,task,initial=validated_task_inputs(parent,validated,task_id,phase,seed=seed,practice=practice)
    revision=read(root/'robot_revision.json')
    if manifest['selection_kind']=='practice':
        scene,task,initial=derive_practice_inputs(scene,task,initial,revision,formal_release=manifest['formal_parent_revision'])
        task['runtime_release_manifest_sha256']=file_hash(root/'release_manifest.json')
    else:scene,task,initial=derive_inputs(scene,task,initial,revision)
    return deepcopy(manifest['tasks'][0]),scene,task,initial


def load_practice_inputs(root):
    from bench.isaacsim.release import validate
    root=Path(root).resolve();manifest=validate(root)
    if manifest['schema']!=SCHEMA or manifest['selection_kind']!='practice':raise ValueError('Not a robot-revised practice release')
    return {'manifest':manifest,'cases':read(root/manifest['cases']),
        'parent_release':deepcopy(manifest['parent_release']),
        'source_metadata':{'release_root':str(root),'release_manifest_sha256':file_hash(root/'release_manifest.json'),
                           'sources_sha256':read(root/'task.json')['sources_sha256']}}
