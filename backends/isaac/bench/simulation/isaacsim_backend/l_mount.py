"""Pinned L-adapter frames and read-only access to separately prepared assets.

The hand frame is a massless interface on the adapter output face. The actual
palm begins 35 mm along its local Z axis. Historical centred mounts are separate
versions; selecting this version never invokes an implicit asset conversion.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from bench.paths import load_json

SUPPORTED_VERSIONS = ('isaacsim_6_1_l_mount_v1', 'isaacsim_6_1_l_mount_v2', 'isaacsim_6_1_l_mount_v3')
VERSION = SUPPORTED_VERSIONS[-1]
COLLISION_RECIPES = {
    SUPPORTED_VERSIONS[0]: 'fr3_sharpa_l_adapter_source_preserving_v1',
    SUPPORTED_VERSIONS[1]: 'fr3_sharpa_l_adapter_explicit_convex_v2',
    SUPPORTED_VERSIONS[2]: 'fr3_sharpa_l_adapter_explicit_only_v3',
}
CONVENTION = 'l_adapter_20261104_v1'
UPLOAD_SHA256 = '20c4b8d0bd0e637ca3739936b25be810959f6a484268870f4c8e04970ff53d22'
HAND_URDF = 'robots/sharpa/wave_01/derived/l_mount_20261104_v1/{side}_sharpa_l_mount.urdf'
MOUNTS = {
    'right': {
        'translation_m': [.045961940777125586, -.04596194077712559, .065],
        'quaternion_wxyz': [.27059805007309856, .6532814824381883, -.27059805007309856, .6532814824381882],
        'rpy_rad': [math.pi, -math.pi / 2, -math.pi / 4],
    },
    'left': {
        'translation_m': [.045961940777125586, .04596194077712559, .065],
        'quaternion_wxyz': [.27059805007309856, -.6532814824381883, -.27059805007309856, -.6532814824381882],
        'rpy_rad': [-math.pi, -math.pi / 2, math.pi / 4],
    },
}


def registry_override(*, bimanual=False):
    right = MOUNTS['right']
    return {
        'mount_convention': CONVENTION,
        'mount_translation_m': deepcopy(right['translation_m']),
        'mount_rpy_rad': deepcopy(right['rpy_rad']),
        'mount_quaternion_wxyz': deepcopy(right['quaternion_wxyz']),
        'mounts_by_side': deepcopy(MOUNTS if bimanual else {'right': right}),
        'urdf': HAND_URDF.format(side='right'),
        'hand_frame_description': 'Massless hand mounting interface on the L adapter output face; palm root is 35 mm along local +Z.',
        'robot_upload_sha256': UPLOAD_SHA256,
    }


def public_mount(robot):
    """Publish physical mounts for the selected revision, including instances."""
    from .robot_registry import get_robot_spec
    if robot.get('assembly_version') in SUPPORTED_VERSIONS:
        spec = get_robot_spec(robot)
        mounts = spec['mounts_by_side']
    elif robot.get('hand_id') == 'fr3_sharpa_instances':
        from .robot_instances import checked_instances
        instances = checked_instances(robot)
        if not all(item['assembly_version'] in SUPPORTED_VERSIONS for item in instances.values()):
            return None
        spec = get_robot_spec(robot)
        mounts = {name: deepcopy(MOUNTS['right']) for name in instances}
    else:
        return None
    return {'mount_convention': CONVENTION, 'mounts_by_side': deepcopy(mounts),
            'status': spec['mount_status'],
            'hand_frame_description': registry_override()['hand_frame_description']}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_source_inputs(source, assets_root, *, _parents=()):
    """Physical input closure, separate from the complete upload provenance.

    SOURCE records retain archived scripts and historical reports. They are
    provenance metadata, not files a robot startup needs to read or qualify.
    """
    assets = Path(assets_root).resolve()
    side = 'left' if source.get('side') == 'left' else 'right'
    hand = assets / HAND_URDF.format(side=side)
    hand_source = hand.parent / f'SOURCE_{side}.json'
    result = {source['urdf_path']: source['urdf_sha256'],
              source['import_urdf_path']: source['import_urdf_sha256'],
              **source['mesh_inputs'], **source['license_sha256']}
    for path in (hand, hand_source):
        expected = source['sources'].get(str(path))
        if expected is None:
            raise ValueError('Physical assembly omitted its actual controller hand input')
        result[str(path)] = expected
    if 'source_manifest' in source:
        parent = Path(source['source_manifest']).resolve()
        if parent in _parents or not parent.is_relative_to(assets / 'robots/assemblies'):
            raise ValueError('Invalid recursive robot instance source')
        expected = source['source_manifest_sha256']
        if _sha(parent) != expected:
            raise ValueError('Robot instance source metadata changed')
        result[str(parent)] = expected
        result.update(runtime_source_inputs(load_json(parent), assets,
                                            _parents=(*_parents, parent)))
    if any(not Path(path).resolve().is_relative_to(assets) for path in result):
        raise ValueError('Robot physical inputs must remain inside their shared asset root')
    return result


def prepared_assembly(assets_root, *, side=None, output=None, version=VERSION):
    if side not in (None, 'left', 'right'):
        raise ValueError('Unknown L-mount assembly side')
    if version not in SUPPORTED_VERSIONS:
        raise ValueError('Unknown L-mount assembly version')
    directory = Path(output) if output is not None else Path(assets_root) / 'robots/assemblies' / (
        'fr3_sharpa' if side is None else 'dual_fr3_sharpa_w01') / version
    path = directory / ('SOURCE.json' if side is None else f'SOURCE_{side}.json')
    source = load_json(path)
    if source.get('assembly_version') != version:
        raise ValueError('Prepared L-mount source belongs to another assembly version')
    urdf = Path(source['urdf_path'])
    if urdf.parent.resolve() != directory.resolve():
        raise ValueError('Prepared L-mount URDF is outside its versioned directory')
    inputs = runtime_source_inputs(source, assets_root)
    if any(_sha(p) != expected for p, expected in inputs.items()):
        raise ValueError('Prepared L-mount source dependency changed')
    mount = source['mount']
    expected_parent = 'fr3_link8' if side is None else f'fr3_{side}_link8'
    if mount['parent'] != expected_parent or mount['child'] != f'{side or "right"}_hand_flange':
        raise ValueError('Prepared L-mount frame names differ from its actual assembly')
    from math import isclose
    for key in ('translation_m', 'quaternion_wxyz'):
        expected = MOUNTS[side or 'right'][key]
        actual = mount[key]
        if len(actual) != len(expected) or not all(isclose(a, b, rel_tol=0., abs_tol=1e-12)
                                                 for a, b in zip(actual, expected)):
            raise ValueError('Prepared L-mount frame differs from the registry')
    return urdf, source


def prepared_import(urdf, *, side=None):
    path = Path(urdf).parent / ('USD_IMPORT.json' if side is None else f'USD_IMPORT_{side}.json')
    record = load_json(path)
    usd = Path(record['usd_path'])
    source_path = Path(urdf).parent / ('SOURCE.json' if side is None else f'SOURCE_{side}.json')
    source = load_json(source_path)
    version = source.get('source_assembly_version', source.get('assembly_version'))
    if version not in SUPPORTED_VERSIONS:
        raise ValueError('Unknown prepared L-mount import version')
    if (record['urdf_sha256'] != _sha(urdf) or record['usd_sha256'] != _sha(usd)
            or record.get('source_path') != str(source_path)
            or record.get('source_sha256') != _sha(source_path)
            or record.get('import_urdf_path') != source['import_urdf_path']
            or record.get('import_urdf_sha256') != source['import_urdf_sha256']
            or record['isaacsim_target_version'] != '6.1.0'
            or record.get('collision_recipe_id') != COLLISION_RECIPES[version]
            or record.get('import_urdf_sha256') != _sha(record['import_urdf_path'])
            or any(_sha(p) != expected for p, expected in record['derived_files'].items())):
        raise ValueError('Prepared L-mount native import changed or targets a different simulator')
    return usd
