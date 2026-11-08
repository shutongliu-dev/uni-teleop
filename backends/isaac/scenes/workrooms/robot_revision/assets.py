"""Immutable robot input manifests, independent of development qualification."""
import hashlib
import json
from pathlib import Path

from bench.paths import DATA_ROOT, load_json, undistributed_bytecode

VERSION = 'isaacsim_6_1_l_mount_v1'
SUPPORTED_VERSIONS = frozenset({VERSION, 'isaacsim_6_1_l_mount_v2', 'isaacsim_6_1_l_mount_v3'})
SCHEMA = 'fr3_sharpa_l_mount_revision.v1'
RUNTIME_SCHEMA = 'fr3_sharpa_l_mount_runtime_inputs.v1'
MANIFEST_SCHEMAS = {'revision_manifest.json': SCHEMA, 'runtime_inputs_manifest.json': RUNTIME_SCHEMA}
ASSET_ROOT = Path(f"{DATA_ROOT}/assets")
ASSEMBLIES = frozenset({'single_right', 'dual_left', 'dual_right', 'front', 'rear'})


def file_hash(path):
    from bench.isaacsim.validation_epoch import file_sha256
    return file_sha256(path)


def checked_file(path, expected):
    path = Path(path)
    if isinstance(expected, str) and len(expected) == 64 and undistributed_bytecode(path):
        return path  # pinned bytecode cache, not shipped; see bench.paths.undistributed_bytecode
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)
            or not path.is_file() or file_hash(path) != expected):
        raise ValueError('Robot input file differs from its immutable digest: '+str(path))
    return path


def read_asset_manifest(path, *, expected_sha256=None, assets=True):
    path = Path(path).resolve()
    if (not path.is_relative_to(ASSET_ROOT/'robots/assemblies')
            or path.name not in MANIFEST_SCHEMAS or path.parent.name not in SUPPORTED_VERSIONS):
        raise ValueError('Robot revision requires its versioned shared asset manifest')
    if expected_sha256 is not None:
        checked_file(path, expected_sha256)
    value = load_json(path)
    if (value.get('schema') != MANIFEST_SCHEMAS[path.name] or value.get('assembly_version') != path.parent.name
            or set(value.get('hands', {})) != {'left', 'right'}
            or set(value.get('assemblies', {})) != ASSEMBLIES):
        raise ValueError('Robot manifest has another version or incomplete hand/assembly inventory')
    if value['schema'] == RUNTIME_SCHEMA:
        # The archival record remains provenance. Historical reports, uploads
        # and their old dependency tree are not runtime input reads.
        provenance = value.get('provenance_manifest')
        if (not isinstance(provenance,dict) or not {'path','sha256'} <= set(provenance)
                or set(provenance)-{'path','sha256','scope'}
                or ('scope' in provenance and not isinstance(provenance['scope'],str))
                or not isinstance(provenance['path'],str) or not Path(provenance['path']).is_absolute()
                or not isinstance(provenance['sha256'],str) or len(provenance['sha256'])!=64
                or any(c not in '0123456789abcdef' for c in provenance['sha256'])):
            raise ValueError('Runtime robot inputs require a well-formed archival provenance reference')
    files = value.get('files_sha256')
    if not isinstance(files, dict) or not files:
        raise ValueError('Robot manifest must bind actual source and derived dependencies')
    for rows, fields in ((value['hands'], ('urdf', 'source')),
                         (value['assemblies'], ('urdf', 'source', 'usd_import'))):
        for row in rows.values():
            if not isinstance(row, dict):
                raise ValueError('Robot manifest entry is not an input reference')
            for name in fields:
                filename, expected = row.get(name+'_path'), row.get(name+'_sha256')
                if not isinstance(filename, str) or files.get(filename) != expected or expected is None:
                    raise ValueError('Robot manifest omits a declared source/import dependency')
    for filename, expected in files.items():
        if (not isinstance(filename, str) or not Path(filename).is_absolute()
                or not Path(filename).resolve().is_relative_to(ASSET_ROOT)):
            raise ValueError('Robot dependencies must resolve inside shared assets')
        if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('Robot dependency has an invalid digest')
    # The importer inventories every derived USD dependency; checking just the
    # top USD file would miss modified referenced geometry or material files.
    runtime_files = {}
    if value['schema'] == RUNTIME_SCHEMA:
        for rows, fields in ((value['hands'],('urdf','source')),
                             (value['assemblies'],('urdf','source','usd_import'))):
            for row in rows.values():
                runtime_files.update({row[n+'_path']:row[n+'_sha256'] for n in fields})
                source = load_json(checked_file(row['source_path'],row['source_sha256']))
                if (source.get('urdf_path') != row['urdf_path'] or source.get('urdf_sha256') != row['urdf_sha256']
                        or not isinstance(source.get('mesh_inputs'),dict)):
                    raise ValueError('Runtime source metadata differs from its actual public URDF')
                runtime_files.update(source['mesh_inputs'])
                runtime_files.update(source.get('license_sha256',{}))
                if 'import_urdf_path' in source:
                    runtime_files[source['import_urdf_path']]=source['import_urdf_sha256']
    for row in value['assemblies'].values():
        import_path = checked_file(row['usd_import_path'], row['usd_import_sha256'])
        imported = load_json(import_path)
        required = {imported['usd_path']: imported['usd_sha256'], **imported.get('derived_files', {})}
        if not imported.get('derived_files') or any(files.get(p) != h for p,h in required.items()):
            raise ValueError('Robot manifest omits actual imported USD dependencies')
        runtime_files.update(required)
    if value['schema'] == RUNTIME_SCHEMA and runtime_files != files:
        raise ValueError('Runtime robot inputs differ from actual physical/render/source dependencies')
    if assets:
        for filename,expected in files.items(): checked_file(filename,expected)
    return value


def check_robot_binding(revision, asset):
    """The descriptor may select only the exact immutable assembly and mounts."""
    from .inputs import canonical_hash
    robot = revision['robot']
    version = asset.get('assembly_version')
    if version not in SUPPORTED_VERSIONS:
        raise ValueError('Robot asset selects an unregistered assembly version')
    mounts = asset.get('mounts_by_side')
    if not isinstance(mounts,dict) or set(mounts) != {'left','right'}:
        raise ValueError('Robot asset must declare both physical mount frames')
    if robot['hand_id'] == 'fr3_sharpa_instances':
        if robot.get('assembly_version') != 'namespaced_v1' or set(robot.get('instances',{})) != {'front','rear'}:
            raise ValueError('Robot revision changed composite installation semantics')
        for member in robot['instances'].values():
            if member.get('assembly_version') != version or member.get('hand_id') != 'sharpa_w01_right':
                raise ValueError('Both front/rear assemblies must retain their original right hand')
        if 'mounts_by_side' in robot and robot['mounts_by_side'] != {s:mounts['right'] for s in ('front','rear')}:
            raise ValueError('Composite mount metadata differs from the right-hand asset')
        return
    if robot['hand_id'] not in ('sharpa_w01_right','dual_fr3_sharpa_w01') or robot.get('assembly_version') != version:
        raise ValueError('Robot descriptor must select the published L-mount assembly')
    selected = mounts if robot['hand_id']=='dual_fr3_sharpa_w01' else {'right':mounts['right']}
    if (canonical_hash(robot.get('mounts_by_side')) != canonical_hash(selected)
            or canonical_hash(robot.get('mount_position_m')) != canonical_hash(mounts['right']['translation_m'])
            or canonical_hash(robot.get('mount_quaternion_wxyz')) != canonical_hash(mounts['right']['quaternion_wxyz'])):
        raise ValueError('Robot descriptor mount metadata differs from the published assembly')
