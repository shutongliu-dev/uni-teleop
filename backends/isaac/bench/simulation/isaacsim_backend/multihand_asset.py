"""Build versioned FR3 + hand URDFs from pinned, unmodified upstream models."""
from copy import deepcopy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .robot_registry import get_hand_spec
from .robot_asset import DEFAULT_ASSETS, digest, write_once
from bench.paths import load_json


def assemble_hand_urdf(hand_id, assets_root=DEFAULT_ASSETS, output=None, assembly_version=None):
    spec = get_hand_spec(hand_id, assembly_version)
    assets_root = Path(assets_root)
    output = Path(output or assets_root / 'robots/assemblies' / ('fr3_' + hand_id) / spec.get('assembly_version', 'isaacsim_6_1_v2'))
    output.mkdir(parents=True, exist_ok=True)
    inputs = [assets_root / 'robots/franka/fr3/fr3_no_hand.absolute.urdf', assets_root / spec['urdf']]
    merged = ET.Element('robot', name='fr3_' + hand_id)
    meshes, inertias, modifications = {}, {}, []
    for index, source in enumerate(inputs):
        root = ET.parse(source).getroot()
        prefix = spec['prefix'] if index else ''
        removed = spec.get('remove_root') if index else None
        for original in root:
            if original.tag not in ('link', 'joint', 'material'):
                continue
            if removed and (original.tag == 'link' and original.get('name') == removed or
                            original.tag == 'joint' and original.find('parent').get('link') == removed):
                modifications.append('Removed external world link/joint: ' + str(original.get('name')))
                continue
            item = deepcopy(original)
            if item.get('name'):
                item.set('name', prefix + item.get('name'))
            if item.tag == 'joint':
                for edge in ('parent', 'child'):
                    item.find(edge).set('link', prefix + item.find(edge).get('link'))
            for material in item.findall('.//material'):
                material.set('name', prefix + (material.get('name') or item.get('name', 'material')))
            for mesh in item.findall('.//mesh'):
                uri = mesh.get('filename')
                if uri.startswith('package://'):
                    package, relative = uri[len('package://'):].split('/', 1)
                    if package not in spec['package_roots']:
                        raise ValueError(f'Unknown mesh package: {uri}')
                    path = assets_root / spec['package_roots'][package] / relative
                else:
                    path = Path(uri)
                    if not path.is_absolute():
                        path = source.parent / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                mesh.set('filename', str(path))
                meshes[str(path)] = digest(path)
            if item.tag == 'joint' and item.get('name') in spec['mimic_joints']:
                ET.SubElement(item, 'mimic', **{k: str(v) for k,v in spec['mimic_joints'][item.get('name')].items()})
            inertial = item.find('inertial')
            if inertial is not None:
                mass = float(inertial.find('mass').get('value'))
                v = {key: float(value) for key, value in inertial.find('inertia').attrib.items()}
                matrix = np.array([[v['ixx'], v['ixy'], v['ixz']], [v['ixy'], v['iyy'], v['iyz']], [v['ixz'], v['iyz'], v['izz']]])
                eigen = np.linalg.eigvalsh(matrix)
                fixed_point_mass = bool(np.all(eigen == 0) and item.find('collision') is None and any(j.get('type') == 'fixed' and j.find('child').get('link') == original.get('name') for j in root.findall('joint')))
                if mass <= 0 or (eigen[0] <= 0 and not fixed_point_mass) or not np.isfinite(matrix).all():
                    raise ValueError(f'Invalid upstream mass/inertia: {item.get("name")}')
                inertias[item.get('name')] = {'mass_kg': mass, 'fixed_point_mass_merged_into_parent': fixed_point_mass, 'inertia_eigenvalues_kg_m2': eigen.tolist(),
                                             'triangle_inequality': bool(eigen[-1] <= eigen[:2].sum() + 1e-10)}
            merged.append(item)
    adapter = add_adapter(merged, spec.get('adapter'))
    if adapter:
        inertias[adapter['link']] = {'mass_kg': adapter['mass_kg'], 'fixed_point_mass_merged_into_parent': False,
                                     'inertia_eigenvalues_kg_m2': sorted(adapter['inertia_diagonal_kg_m2']), 'triangle_inequality': True}
    # The adapter frame coincides with fr3_link8, so the registry mount stays
    # the hand-root pose in the flange frame with or without an adapter.
    mount = ET.SubElement(merged, 'joint', name='fr3_hand_mount', type='fixed')
    ET.SubElement(mount, 'parent', link=adapter['link'] if adapter else 'fr3_link8')
    ET.SubElement(mount, 'child', link=spec['prefix'] + spec['root_link'])
    ET.SubElement(mount, 'origin', xyz=' '.join(map(str, spec['mount_translation_m'])), rpy=' '.join(map(str, spec['mount_rpy_rad'])))
    joints = {}
    for joint in merged.findall('joint'):
        if joint.get('type') == 'revolute':
            joints[joint.get('name')] = {key: float(joint.find('limit').get(key)) for key in ('lower', 'upper', 'effort', 'velocity')}
    if len(joints) != spec['joint_count']:
        raise ValueError(f'Joint inventory differs from registry: {len(joints)} != {spec["joint_count"]}')
    links = [link.get('name') for link in merged.findall('link')]
    if len(links) != len(set(links)):
        raise ValueError('Duplicate link names')
    ET.indent(merged, space='  ')
    body = ET.tostring(merged, encoding='utf-8', xml_declaration=True) + b'\n'
    path = output / ('fr3_' + hand_id + '.urdf')
    manifest = output / 'SOURCE.json'
    write_once(path, body, f'Never overwrite a versioned assembly; select new output: {path}')
    licenses = [assets_root / 'robots/franka/fr3/franka_description/LICENSE', assets_root / spec['license']]
    record = {'schema_version': 'fr3-multihand-isaacsim-source-v3' if adapter or 'mount_convention' in spec else 'fr3-multihand-isaacsim-source-v2',
              'hand_spec': spec,
              'urdf_path': str(path), 'urdf_sha256': digest(path), 'sources': {str(p): digest(p) for p in inputs},
              'mesh_inputs': meshes, 'joint_limits': joints, 'upstream_mass_inertia_audit': inertias,
              'mass_kg_sum': sum(x['mass_kg'] for x in inertias.values()),
              'mount': {'parent': 'fr3_link8', 'child': spec['prefix'] + spec['root_link'],
                        'translation_m': spec['mount_translation_m'], 'quaternion_wxyz': spec['mount_quaternion_wxyz'],
                        'rpy_rad': spec['mount_rpy_rad'], 'status': 'uncalibrated_simulation_assumption',
                        'convention': spec.get('mount_convention'), 'adapter': adapter,
                        'adapter_mass_included': bool(adapter)},
              'licenses': [str(p) for p in licenses], 'license_sha256': {str(p): digest(p) for p in licenses},
              'modifications': modifications + ['Names prefixed to avoid collisions; mesh paths resolved absolutely; source inertias/collision geometry preserved.',
                                                'Simulator-specific tags and ROS transmission tags are not imported. See hand_spec.control_model.'],
              'fingertip_frames': {k: spec['prefix'] + v for k,v in spec['fingertip_links'].items()},
              'fingertip_frame_semantics': 'Named distal rigid-link origins; not measured contact points.',
              'generator_sha256': digest(__file__)}
    if manifest.exists():
        existing = load_json(manifest)
        for key in ('urdf_sha256', 'sources', 'mesh_inputs', 'joint_limits', 'license_sha256'):
            if existing[key] != record[key]:
                raise ValueError('Existing assembly provenance mismatch: ' + key)
        return path, existing
    write_once(manifest, (json.dumps(record, indent=2) + '\n').encode(), f'Assembly provenance changed: {manifest}')
    return path, record


def add_adapter(robot, adapter):
    """Append a solid aluminium flange disc between fr3_link8 and the hand.

    Its frame is fr3_link8 (bottom face on the flange face, axis along +z).
    The collision cylinder starts collision_clearance_m above the flange face,
    like the SHARPA flange root, so the welded link7 face never overlaps it.
    """
    if not adapter:
        return None
    L, r = float(adapter['thickness_m']), float(adapter['radius_m'])
    clearance, density = float(adapter['collision_clearance_m']), float(adapter['density_kg_m3'])
    if not (0 < clearance < L and r > 0 and density > 0):
        raise ValueError('Invalid adapter geometry')
    mass = density * math.pi * r * r * L
    ixx, izz = mass * (3 * r * r + L * L) / 12, mass * r * r / 2
    link = ET.SubElement(robot, 'link', name='fr3_hand_adapter')
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'origin', xyz=f'0 0 {L / 2!r}', rpy='0 0 0')
    ET.SubElement(inertial, 'mass', value=repr(mass))
    ET.SubElement(inertial, 'inertia', ixx=repr(ixx), ixy='0', ixz='0', iyy=repr(ixx), iyz='0', izz=repr(izz))
    visual = ET.SubElement(link, 'visual')
    ET.SubElement(visual, 'origin', xyz=f'0 0 {L / 2!r}', rpy='0 0 0')
    ET.SubElement(ET.SubElement(visual, 'geometry'), 'cylinder', radius=repr(r), length=repr(L))
    ET.SubElement(ET.SubElement(visual, 'material', name='fr3_hand_adapter_aluminium'), 'color', rgba='0.74 0.75 0.77 1')
    collision = ET.SubElement(link, 'collision')
    ET.SubElement(collision, 'origin', xyz=f'0 0 {(L + clearance) / 2!r}', rpy='0 0 0')
    ET.SubElement(ET.SubElement(collision, 'geometry'), 'cylinder', radius=repr(r), length=repr(L - clearance))
    joint = ET.SubElement(robot, 'joint', name='fr3_adapter_mount', type='fixed')
    ET.SubElement(joint, 'parent', link='fr3_link8')
    ET.SubElement(joint, 'child', link='fr3_hand_adapter')
    ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
    return {'link': 'fr3_hand_adapter', 'joint': 'fr3_adapter_mount', **adapter, 'mass_kg': mass,
            'inertia_diagonal_kg_m2': [ixx, ixx, izz], 'top_face_z_m': L}
