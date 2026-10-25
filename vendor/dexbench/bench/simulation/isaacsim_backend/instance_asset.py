"""Versioned identifier-only copies of pinned original robot assemblies."""
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .robot_asset import DEFAULT_ASSETS, digest, sharpa_assembly_dir, write_once
from .robot_instances import ASSEMBLY_VERSION, ROBOT_ID, checked_namespace, instance_name
from bench.paths import load_json


def namespaced_tree(root, namespace):
    """Preserve all physical XML values; change robot/link/joint identities only."""
    checked_namespace(namespace)
    tree = deepcopy(root)
    tree.set('name', f'fr3_sharpa_{namespace}')
    for element in tree:
        if element.tag in ('link', 'joint'):
            element.set('name', instance_name(namespace, element.get('name')))
        if element.tag == 'joint':
            for edge in ('parent', 'child'):
                node = element.find(edge)
                node.set('link', instance_name(namespace, node.get('link')))
            mimic = element.find('mimic')
            if mimic is not None:
                mimic.set('joint', instance_name(namespace, mimic.get('joint')))
    return tree


def assemble_instance(namespace, assets_root=DEFAULT_ASSETS, *, source_version, output=None):
    """Read an existing pinned assembly; never rebuild or overwrite the source."""
    from .robot_registry import get_hand_spec
    checked_namespace(namespace)
    spec = get_hand_spec('sharpa_w01_right', source_version)
    assets_root = Path(assets_root)
    source_path = sharpa_assembly_dir(assets_root, source_version) / 'SOURCE.json'
    original = load_json(source_path)
    source_urdf = Path(original['urdf_path'])
    if (original['mount']['translation_m'] != spec['mount_translation_m']
            or original['mount']['quaternion_wxyz'] != spec['mount_quaternion_wxyz']):
        raise ValueError('Frozen source assembly mount differs from its registry version')
    hashes = {str(source_urdf): original['urdf_sha256'], **original['sources'],
              **original['mesh_inputs'], **original['license_sha256']}
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError('Frozen source robot asset changed: ' + path)
    tree = namespaced_tree(ET.parse(source_urdf).getroot(), namespace)
    output = Path(output or assets_root / 'robots/assemblies' / ROBOT_ID /
                  ASSEMBLY_VERSION / source_version / namespace)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'robot.urdf'
    body = ET.tostring(tree, encoding='utf-8', xml_declaration=True) + b'\n'
    write_once(path, body, f'Never overwrite a versioned instance assembly: {path}')
    mount = deepcopy(original['mount'])
    for key in ('parent', 'child'):
        mount[key] = instance_name(namespace, mount[key])
    record = {'schema_version': 'fr3-source-instance-v1', 'robot_id': ROBOT_ID,
        'namespace': namespace, 'assembly_version': ASSEMBLY_VERSION,
        'source_hand_id': 'sharpa_w01_right', 'source_assembly_version': source_version,
        'source_manifest': str(source_path), 'source_manifest_sha256': digest(source_path),
        'urdf_path': str(path), 'urdf_sha256': digest(path),
        'sources': {**original['sources'], str(source_urdf): original['urdf_sha256']},
        'mesh_inputs': deepcopy(original['mesh_inputs']),
        'licenses': deepcopy(original['licenses']), 'license_sha256': deepcopy(original['license_sha256']),
        'joint_limits': {instance_name(namespace, name): deepcopy(value)
                         for name, value in original['joint_limits'].items()},
        'mount': mount, 'source_name_map': {
            node.get('name'): instance_name(namespace, node.get('name'))
            for node in ET.parse(source_urdf).getroot() if node.tag in ('link', 'joint')},
        'physical_values_changed': False,
        'generator_sha256': {'bench/simulation/isaacsim_backend/instance_asset.py': digest(__file__)}}
    write_once(output / 'SOURCE.json', (json.dumps(record, sort_keys=True, indent=2) + '\n').encode(),
               f'Never overwrite versioned instance provenance: {output}')
    return path, record


def import_instances(assets_root, instances):
    """Import one unchanged-physics assembly per declared instance."""
    from .robot_asset import _import_side
    from .robot_instances import checked_instances
    instances = checked_instances({'hand_id': ROBOT_ID, 'assembly_version': ASSEMBLY_VERSION,
                                   'instances': instances})
    usds, records, manifests = {}, {}, {}
    for name, item in instances.items():
        urdf, source = assemble_instance(name, assets_root, source_version=item['assembly_version'])
        manifest = urdf.parent / 'USD_IMPORT.json'
        usds[name] = _import_side(urdf, urdf.parent / 'usd', manifest,
            {f'fr3_{name}_joint.*': 2000., f'{name}_.*': 3.},
            {f'fr3_{name}_joint.*': 140., f'{name}_.*': .1})
        imported = load_json(manifest)
        if set(imported['revolute_joint_names']) != set(source['joint_limits']):
            raise ValueError('Native instance joint inventory differs from its source assembly')
        records[name], manifests[name] = source, str(manifest)
    combined = {'schema_version': 'fr3-source-instances-isaacsim-v1', 'robot_id': ROBOT_ID,
        'sides': records, 'instances': instances,
        'joint_limits': {name: value for record in records.values() for name, value in record['joint_limits'].items()},
        'sources': {path: sha for record in records.values() for path, sha in record['sources'].items()},
        'mount': {name: record['mount'] for name, record in records.items()},
        'assembly_urdfs': {record['urdf_path']: record['urdf_sha256'] for record in records.values()},
        'import_manifests': manifests, 'hand_spec': {'merge_fixed_joints': False}}
    return usds, combined
