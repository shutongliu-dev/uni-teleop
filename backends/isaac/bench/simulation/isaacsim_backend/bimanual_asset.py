"""Versioned dual FR3 + SHARPA W01 (left + right) assemblies: one URDF per arm.

PhysX limits one articulation to 64 links, and a single FR3 + SHARPA already
has 43 native bodies, so the bimanual robot is two articulations. Each side is
the pinned FR3 (all names prefixed fr3_<side>_) with the official SHARPA
<side>_sharpa_wave_with_flange hand at the flange mount of the selected assembly
version (v1: identity, flange_axis_centred_v1; v2: 180 deg about each flange axis,
flange_axis_centred_v2). Upstream files are never modified.
"""
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .robot_asset import DEFAULT_ASSETS, digest, write_once
from bench.paths import load_json

BIMANUAL_ID = 'dual_fr3_sharpa_w01'
VERSION = 'isaacsim_6_1_v1'
SIDES = ('left', 'right')
ARM_URDF = 'robots/franka/fr3/fr3_no_hand.absolute.urdf'
HAND_URDF = 'robots/sharpa/wave_01/derived/{side}_sharpa_wave_with_flange.urdf'


def arm_name(side, name):
    """fr3_link0 -> fr3_left_link0; the root 'base' -> fr3_left_base."""
    if name == 'base':
        return f'fr3_{side}_base'
    if not name.startswith('fr3_'):
        raise ValueError(f'Unexpected FR3 name {name!r}')
    return f'fr3_{side}_' + name[len('fr3_'):]


def assemble_side(side, assets_root=DEFAULT_ASSETS, output=None, version=VERSION):
    from .robot_registry import get_hand_spec
    if side not in SIDES:
        raise ValueError('side must be left or right')
    spec = get_hand_spec(BIMANUAL_ID, version)
    assets_root = Path(assets_root)
    output = Path(output or assets_root / 'robots/assemblies' / BIMANUAL_ID / version)
    output.mkdir(parents=True, exist_ok=True)
    sources = [assets_root / ARM_URDF, assets_root / HAND_URDF.format(side=side)]
    merged = ET.Element('robot', name=f'fr3_sharpa_{side}')
    meshes = {}
    for index, source in enumerate(sources):
        root = ET.parse(source).getroot()
        for original in root:
            if original.tag not in ('link', 'joint', 'material'):
                continue
            item = deepcopy(original)
            if index == 0:
                if item.get('name'):
                    item.set('name', arm_name(side, item.get('name')) if item.tag != 'material' else f'fr3_{side}_' + item.get('name'))
                if item.tag == 'joint':
                    for edge in ('parent', 'child'):
                        item.find(edge).set('link', arm_name(side, item.find(edge).get('link')))
                for material in item.findall('.//material'):
                    if material.get('name'):
                        material.set('name', f'fr3_{side}_' + material.get('name'))
            for mesh in item.findall('.//mesh'):
                path = Path(mesh.get('filename'))
                if not path.is_absolute():
                    path = (source.parent / path).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                mesh.set('filename', str(path))
                meshes[str(path)] = digest(path)
            for i, material in enumerate(item.findall('.//material')):
                if not material.get('name'):
                    material.set('name', f"{item.get('name', 'local')}_material_{i}")
            merged.append(item)
    mount = ET.SubElement(merged, 'joint', name=f'fr3_{side}_sharpa_mount', type='fixed')
    ET.SubElement(mount, 'parent', link=f'fr3_{side}_link8')
    ET.SubElement(mount, 'child', link=f'{side}_hand_flange')
    if spec['mount_translation_m'] != [0., 0., 0.]:
        raise ValueError('The SHARPA with_flange root is the flange interface; no translation')
    # v1 keeps its exact frozen text; other versions author the registry yaw.
    rpy = '0 0 0' if spec['mount_rpy_rad'] == [0., 0., 0.] else ' '.join(map(str, spec['mount_rpy_rad']))
    ET.SubElement(mount, 'origin', xyz='0 0 0', rpy=rpy)
    links = [link.get('name') for link in merged.findall('link')]
    if len(links) != len(set(links)):
        raise ValueError('Duplicate link names')
    joints = {joint.get('name'): {key: float(joint.find('limit').get(key)) for key in ('lower', 'upper', 'effort', 'velocity')}
              for joint in merged.findall('joint') if joint.get('type') == 'revolute'}
    if len(joints) != 29:
        raise ValueError(f'Expected 29 revolute joints per arm, got {len(joints)}')
    ET.indent(merged, space='  ')
    body = ET.tostring(merged, encoding='utf-8', xml_declaration=True) + b'\n'
    path = output / f'fr3_sharpa_{side}.urdf'
    write_once(path, body, f'Never overwrite a versioned assembly; select new output: {path}')
    licenses = [assets_root / 'robots/franka/fr3/franka_description/LICENSE', assets_root / 'robots/sharpa/wave_01/upstream/LICENSE.txt']
    record = {'schema_version': 'fr3-bimanual-side-isaacsim-source-v1', 'robot_id': BIMANUAL_ID, 'side': side,
              'urdf_path': str(path), 'urdf_sha256': digest(path), 'sources': {str(p): digest(p) for p in sources},
              'mesh_inputs': meshes, 'joint_limits': joints,
              'mount': ({'parent': f'fr3_{side}_link8', 'child': f'{side}_hand_flange', 'translation_m': [0, 0, 0],
                         'quaternion_wxyz': [1, 0, 0, 0], 'convention': 'official_sharpa_with_flange_identity',
                         'status': 'uncalibrated_simulation_assumption', 'adapter_mass_included': False}
                        if version == VERSION else
                        {'parent': f'fr3_{side}_link8', 'child': f'{side}_hand_flange', 'translation_m': [0, 0, 0],
                         'quaternion_wxyz': spec['mount_quaternion_wxyz'], 'rpy_rad': spec['mount_rpy_rad'],
                         'convention': spec['mount_convention'], 'assembly_version': version,
                         'status': 'uncalibrated_simulation_assumption', 'adapter_mass_included': False}),
              'licenses': [str(p) for p in licenses], 'license_sha256': {str(p): digest(p) for p in licenses},
              'modifications': ['FR3 link/joint/material names prefixed fr3_<side>_ (root base -> fr3_<side>_base).',
                                'Mesh paths resolved absolutely; upstream inertias and collisions preserved.'],
              'generator_sha256': digest(__file__)}
    manifest = output / f'SOURCE_{side}.json'
    if manifest.exists():
        existing = load_json(manifest)
        for key in ('urdf_sha256', 'sources', 'mesh_inputs', 'joint_limits', 'license_sha256'):
            if existing[key] != record[key]:
                raise ValueError('Existing assembly provenance mismatch: ' + key)
        return path, existing
    write_once(manifest, (json.dumps(record, indent=2) + '\n').encode(), f'Assembly provenance changed: {manifest}')
    return path, record


def assemble_bimanual(assets_root=DEFAULT_ASSETS, output=None, version=VERSION):
    return {side: assemble_side(side, assets_root, output, version) for side in SIDES}
