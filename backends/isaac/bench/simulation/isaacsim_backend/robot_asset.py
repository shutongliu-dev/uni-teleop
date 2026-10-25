"""Derive a single FR3 + right SHARPA URDF and import it with Isaac Sim.

The manufacturer source files remain unchanged. The fixed hand mount is an
explicit, uncalibrated simulation assumption, not a measured adapter model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from bench.paths import DATA_ROOT, load_json

DEFAULT_ASSETS = Path(f'{DATA_ROOT}/assets')
DERIVED_RELATIVE = Path('robots/assemblies/fr3_sharpa/isaacsim_6_1_v1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, body, changed_message):
    """Create a versioned assembly file once; never rewrite it.

    An existing file must already hold exactly ``body``. Until 2026-10-15 every import
    rewrote the identical bytes in place, so a concurrent import could hash a truncated
    file ('Existing assembly provenance mismatch'). The first write is atomic.
    """
    path = Path(path)
    if path.exists():
        if path.read_bytes() != body:
            raise FileExistsError(changed_message)
        return
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_bytes(body)
    os.replace(temporary, path)


# Source-mesh audit behind the opt-in LEAP thumb_pip/palm structural filter
# (CPU, pinned upstream collision meshes, full leap_12 x leap_13 range).
LEAP_THUMB_PIP_PALM_AUDIT = {
    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/distortion_repair_20261014T043445Z/'
            'runs/embodiment_grasp_02/leap_thumb_pip_gap_map.json',
    'sha256': 'a5fcfa121dd8b91c1312dc002008d5c9f10e89daf26498709577007d40f75b99',
    'finding': ('The source meshes are >= 1.2 mm apart over the joint-range interior and meet only at the '
                'leap_12 limit edges (-0.35 / >= 1.99 rad), where the joint limit already holds the thumb. '
                'Convex decomposition (assemblies v5 and v6) bridges the 1.2-1.5 mm gap in the opposition '
                'region, so the native pair is in persistent artificial contact.'),
}


def collision_configuration(scene):
    """Return explicit Isaac overrides, validating the one structural exclusion."""
    options = scene.get('isaacsim', {})
    enabled = options.get('robot_self_collisions', scene['robot'].get('self_collisions', False))
    filter_cmc = options.get('filter_thumb_cmc_structure', False)
    filter_leap = options.get('filter_leap_thumb_pip_palm', False)
    if not isinstance(enabled, bool) or not isinstance(filter_cmc, bool) or not isinstance(filter_leap, bool):
        raise ValueError('Self-collision options must be booleans')
    if filter_cmc and not enabled:
        raise ValueError('CMC structural exclusion requires enabled self collisions')
    pairs = []
    hand_id = scene['robot'].get('hand_id', 'sharpa_w01_right')
    from .robot_instances import ROBOT_ID, checked_instances, instance_name
    if hand_id == ROBOT_ID:
        instances = checked_instances(scene['robot'])
        # Reuse each original source's exact structural audit; only names change.
        for name, item in instances.items():
            single = deepcopy(scene)
            single['robot'] = {**scene['robot'], **item}
            single['robot'].pop('instances', None)
            single['robot'].pop('arm_base_poses', None)
            original = collision_configuration(single)
            pairs.extend([[instance_name(name, link) for link in pair]
                          for pair in original['excluded_link_pairs']])
        return {**original, 'excluded_link_pairs': pairs}
    if filter_leap and (not enabled or hand_id != 'leap_v1_right'):
        raise ValueError('The LEAP thumb_pip/palm structural exclusion requires the LEAP hand with self collisions')
    if filter_cmc and hand_id in ('sharpa_w01_right', 'dual_fr3_sharpa_w01'):
        for side in (('right',) if hand_id == 'sharpa_w01_right' else ('left', 'right')):
            source = Path(scene['robot'].get('assets_root', DEFAULT_ASSETS)) / f'robots/sharpa/wave_01/derived/{side}_sharpa_wave_with_flange.urdf'
            root = ET.parse(source).getroot()
            intermediate = root.find(f"link[@name='{side}_thumb_CMC_VL']")
            edges = {(j.find('parent').get('link'), j.find('child').get('link')) for j in root.findall('joint')}
            if intermediate is None or intermediate.find('collision') is not None or not {
                    (f'{side}_hand_C_MC', f'{side}_thumb_CMC_VL'), (f'{side}_thumb_CMC_VL', f'{side}_thumb_MC')} <= edges:
                raise ValueError('SHARPA CMC structural topology changed; exclusion requires re-audit')
            pairs.append([f'{side}_hand_C_MC', f'{side}_thumb_MC'])
    exclusion_source = None
    if enabled and scene['robot'].get('hand_id') == 'allegro_v5_right':
        # Four two-axis base structures are excluded by the official SRDF.
        # Keep every cross-finger and distal/palm collision enabled.
        assets = Path(scene['robot'].get('assets_root', DEFAULT_ASSETS))
        srdf = assets / 'robots/allegro/v5_80bd4a8/upstream/src/allegro_hand_moveit/config/allegro_hand.srdf'
        upstream_pairs = {frozenset((p.get('link1'), p.get('link2'))) for p in ET.parse(srdf).getroot().findall('disable_collisions')}
        selected = [('palm_link', f'link_{n}_0') for n in (1, 5, 9, 13)]
        if any(frozenset(pair) not in upstream_pairs for pair in selected):
            raise ValueError('Official Allegro structural exclusions changed')
        pairs = [['allegro_' + a, 'allegro_' + b] for a, b in selected]
        exclusion_source = {'path': str(srdf), 'sha256': digest(srdf), 'selection': 'Only four palm/proximal pairs across the second base rotation; broad Never and cross-finger SRDF pairs NOT imported.'}
    if enabled and scene['robot'].get('hand_id') == 'shadow_hand_e_right':
        assets = Path(scene['robot'].get('assets_root', DEFAULT_ASSETS))
        srdf = assets / 'robots/shadow/sr_interface_01aca16/sr_moveit_hand_config/config/shadowhand.srdf.xacro'
        root = ET.parse(srdf).getroot()
        upstream_pairs = set()
        for node in root.iter('disable_collisions'):
            for finger in ('ff', 'mf', 'rf', 'lf'):
                upstream_pairs.add(frozenset(node.get(k).replace('${prefix}', 'rh_').replace('${link_prefix}', finger) for k in ('link1','link2')))
        selected = [('rh_palm', f'rh_{finger}proximal') for finger in ('ff','mf','rf','lf')]
        selected += [('rh_palm','rh_thproximal'), ('rh_thproximal','rh_thmiddle'), ('rh_forearm','rh_palm')]
        if any(frozenset(pair) not in upstream_pairs for pair in selected):
            raise ValueError('Official Shadow structural exclusions changed')
        # merge_fixed_joints welds forearm into FR3 link7; filter its actual body.
        pairs = [[('fr3_link7' if a == 'rh_forearm' else a), b] for a,b in selected]
        exclusion_source = {'path': str(srdf), 'sha256': digest(srdf),
                            'selection': 'Seven structural pairs spanning two-axis bases; forearm is welded into fr3_link7. Cross-finger and broad Never pairs remain enabled.'}
    reason = None
    if filter_leap:
        # Opt-in per release (v9_embodiments_r4+): the thumb's second link and the
        # palm meet only at joint-limit edges; every other thumb/palm pair stays on.
        source = Path(scene['robot'].get('assets_root', DEFAULT_ASSETS)) / 'robots/leap/v1_150bc3d/upstream/assets/leap_hand/robot.urdf'
        edges = {(j.get('name'), j.find('parent').get('link'), j.find('child').get('link')) for j in ET.parse(source).getroot().findall('joint')}
        if not {('12', 'palm_lower', 'pip_4'), ('13', 'pip_4', 'thumb_pip')} <= edges:
            raise ValueError('LEAP thumb topology changed; the thumb_pip/palm exclusion requires re-audit')
        pairs = [['leap_palm_lower', 'leap_thumb_pip']]
        exclusion_source = {'path': str(source), 'sha256': digest(source), 'audit': dict(LEAP_THUMB_PIP_PALM_AUDIT),
                            'selection': 'One pair (palm_lower, thumb_pip), two joints apart; thumb_dip/fingertip-palm and cross-finger pairs remain enabled.'}
        reason = 'Source meshes meet only at joint-limit edges; the convex decomposition bridges their 1.2-1.5 mm working gap.'
    elif pairs:
        reason = ('Selected upstream SRDF structural overlaps across multi-axis base joints.' if exclusion_source
                  else 'Adjacent physical structures across the collisionless two-axis CMC intermediate link.')
    return {'enabled': enabled, 'source_scene_enabled': scene['robot'].get('self_collisions', False),
            'explicit_isaac_override': 'robot_self_collisions' in options,
            'excluded_link_pairs': pairs, 'cross_finger_exclusions': [], 'exclusion_source': exclusion_source,
            'exclusion_reason': reason,
            'limitation': None if enabled else 'Robot self collisions are disabled; unsuitable for qualified manipulation acceptance.',
            'physics_joint_adjacent_collision_default': 'PhysX disables collision between connected links unless explicitly enabled.'}


def sharpa_assembly_dir(assets_root=DEFAULT_ASSETS, assembly_version=None):
    """Versioned single-arm FR3 + SHARPA assembly directory (default: the v1 identity mount)."""
    return Path(assets_root) / DERIVED_RELATIVE.parent / (assembly_version or DERIVED_RELATIVE.name)


def assemble_urdf(assets_root=DEFAULT_ASSETS, output=None, assembly_version=None):
    """Combine pinned upstream models, resolving all mesh paths absolutely."""
    from .robot_registry import DEFAULT_HAND, get_hand_spec
    spec = get_hand_spec(DEFAULT_HAND, assembly_version)
    assets_root = Path(assets_root)
    output = Path(output or sharpa_assembly_dir(assets_root, spec['assembly_version']))
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [assets_root / 'robots/franka/fr3/fr3_no_hand.absolute.urdf',
                    assets_root / 'robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf']
    merged = ET.Element('robot', {'name': 'fr3_sharpa'})
    mesh_inputs = {}
    for source in source_paths:
        root = ET.parse(source).getroot()
        for original in root:
            # Simulator-specific options in the SHARPA source are not URDF.
            if original.tag not in ('link', 'joint', 'material'):
                continue
            item = deepcopy(original)
            for mesh in item.findall('.//mesh'):
                path = Path(mesh.attrib['filename'])
                if not path.is_absolute():
                    path = (source.parent / path).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                mesh.set('filename', str(path))
                mesh_inputs[str(path)] = digest(path)
            # Anonymous local visual materials must not collide by empty name.
            for i, material in enumerate(item.findall('.//material')):
                if not material.get('name'):
                    material.set('name', item.get('name', 'local') + f'_material_{i}')
            merged.append(item)
    joint = ET.SubElement(merged, 'joint', name='fr3_sharpa_mount', type='fixed')
    ET.SubElement(joint, 'parent', link='fr3_link8')
    ET.SubElement(joint, 'child', link='right_hand_flange')
    if spec['mount_translation_m'] != [0., 0., 0.]:
        raise ValueError('The SHARPA with_flange root is the flange interface; no translation')
    # v1 keeps its exact frozen text; other versions author the registry yaw.
    rpy = '0 0 0' if spec['mount_rpy_rad'] == [0., 0., 0.] else ' '.join(map(str, spec['mount_rpy_rad']))
    ET.SubElement(joint, 'origin', xyz='0 0 0', rpy=rpy)
    names = [e.get('name') for e in merged.findall('link')]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate link name in combined robot')
    joints = {}
    for e in merged.findall('joint'):
        if e.get('type') == 'revolute':
            limit = e.find('limit')
            joints[e.get('name')] = {key: float(limit.get(key)) for key in ('lower', 'upper', 'effort', 'velocity')}
    if len(joints) != 29:
        raise ValueError(f'Expected exactly 29 revolute joints, got {len(joints)}')
    path = output / 'fr3_sharpa.urdf'
    ET.indent(merged, space='  ')
    body = ET.tostring(merged, encoding='utf-8', xml_declaration=True) + b'\n'
    write_once(path, body, f'Derived robot changed; use a new version directory: {path}')
    license_paths = [assets_root / 'robots/franka/fr3/franka_description/LICENSE',
                     assets_root / 'robots/sharpa/wave_01/upstream/LICENSE.txt']
    for license_path in license_paths:
        if not license_path.is_file():
            raise FileNotFoundError(f'Upstream license missing: {license_path}')
    source = {'schema_version': 'fr3-sharpa-isaacsim-robot-source-v1',
              'urdf_path': str(path), 'urdf_sha256': digest(path),
              'sources': {str(p): digest(p) for p in source_paths},
              'mesh_inputs': mesh_inputs, 'joint_limits': joints,
              'mount': ({'parent': 'fr3_link8', 'child': 'right_hand_flange',
                         'translation_m': [0, 0, 0], 'quaternion_wxyz': [1, 0, 0, 0],
                         'status': 'uncalibrated_simulation_assumption', 'adapter_mass_included': False}
                        if spec['mount_rpy_rad'] == [0., 0., 0.] else
                        {'parent': 'fr3_link8', 'child': 'right_hand_flange', 'translation_m': [0, 0, 0],
                         'quaternion_wxyz': spec['mount_quaternion_wxyz'], 'rpy_rad': spec['mount_rpy_rad'],
                         'convention': spec['mount_convention'], 'assembly_version': spec['assembly_version'],
                         'status': 'uncalibrated_simulation_assumption', 'adapter_mass_included': False}),
              'licenses': [str(p) for p in license_paths],
              'license_sha256': {str(p): digest(p) for p in license_paths},
              'generator_sha256': digest(__file__)}
    manifest = output / 'SOURCE.json'
    if manifest.exists():
        existing = load_json(manifest)
        for key in ('urdf_sha256', 'sources', 'mesh_inputs', 'joint_limits', 'license_sha256'):
            if existing[key] != source[key]:
                raise ValueError('Existing assembly provenance mismatch: ' + key)
        return path, existing
    write_once(manifest, (json.dumps(source, indent=2) + '\n').encode(), f'Assembly provenance changed: {manifest}')
    return path, source


def import_robot(assets_root=DEFAULT_ASSETS, output=None, *, hand_id='sharpa_w01_right', assembly_version=None,
                 instances=None):
    """Run the actual Isaac Sim URDF importer; SimulationApp must be running.

    The bimanual robot returns ({side: usd}, combined source) instead of (usd, source).
    ``assembly_version`` selects a registered assembly (default: the hand's v1-era one).
    """
    import omni.kit.app

    from .robot_instances import ROBOT_ID, ASSEMBLY_VERSION
    if hand_id == ROBOT_ID:
        if output is not None or assembly_version != ASSEMBLY_VERSION:
            raise ValueError('Instance imports require their registered versioned asset directories')
        from .instance_asset import import_instances
        return import_instances(assets_root, instances)
    if instances is not None:
        raise ValueError('Instance declarations require the instance robot embodiment')
    if hand_id == 'dual_fr3_sharpa_w01':
        return import_bimanual(assets_root, output, assembly_version)
    if hand_id == 'sharpa_w01_right':
        source_path = Path(output or sharpa_assembly_dir(assets_root, assembly_version)) / 'SOURCE.json'
        if source_path.is_file():
            source = load_json(source_path)
            urdf = Path(source['urdf_path'])
            for path, expected in {str(urdf): source['urdf_sha256'], **source['sources'], **source['mesh_inputs'], **source['license_sha256']}.items():
                if digest(path) != expected:
                    raise ValueError('Frozen robot asset changed: ' + path)
        else:
            urdf, source = assemble_urdf(assets_root, output, assembly_version)
    else:
        from .multihand_asset import assemble_hand_urdf
        urdf, source = assemble_hand_urdf(hand_id, assets_root, output, assembly_version)
    manifest = urdf.parent / 'USD_IMPORT.json'
    if manifest.is_file():
        record = load_json(manifest)
        usd = Path(record['usd_path'])
        if record.get('urdf_sha256') == digest(urdf) and record.get('usd_sha256') == digest(usd):
            for path, expected in record['derived_files'].items():
                if digest(path) != expected:
                    raise ValueError('Cached robot USD dependency changed: ' + path)
            return usd, source
        raise ValueError('Existing derived USD does not match import manifest')
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.asset.importer.urdf', True)
    # Isaac Sim 6.1 replaced the old URDFParseAndImportFile Kit command with
    # this documented importer class. usd_path is a DIRECTORY, not a filename.
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    merge_fixed = bool(source.get('hand_spec', {}).get('merge_fixed_joints', False))
    config = URDFImporterConfig(
        urdf_path=str(urdf), usd_path=str(urdf.parent / 'usd'),
        merge_fixed_joints=merge_fixed, merge_mesh=False, collision_from_visuals=False,
        collision_type=source.get('hand_spec', {}).get('collision_type', 'Convex Hull'), allow_self_collision=False, fix_base=True,
        joint_drive_type='force', joint_target_type='position',
        override_joint_stiffness={'fr3_joint.*': 2000., '(right_|allegro_|leap_|rh_).*': 3.},
        override_joint_damping={'fr3_joint.*': 140., '(right_|allegro_|leap_|rh_).*': .1},
        run_asset_transformer=True, run_multi_physics_conversion=True)
    usd = Path(URDFImporter(config).import_urdf())
    if not usd.is_file():
        raise RuntimeError(f'Isaac URDF importer did not produce USD: {usd}')
    from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema
    stage = Usd.Stage.Open(str(usd))
    # The 6.1 multi-physics transformer emits an unselected Physics variant.
    # Match the official importer tests by explicitly selecting the PhysX
    # payload before inspecting articulation schemas or referencing the asset.
    physics_variant = stage.GetDefaultPrim().GetVariantSet('Physics')
    if 'physx' not in physics_variant.GetVariantNames():
        raise RuntimeError('Imported robot is missing the native PhysX variant')
    physics_variant.SetVariantSelection('physx')
    collision_overrides = []
    # Versioned per hand; assemblies before LEAP v6 used the first three values.
    decomposition_parameters = source.get('hand_spec', {}).get('convex_decomposition',
        {'max_convex_hulls': 64, 'hull_vertex_limit': 64, 'voxel_resolution': 300000})
    if source.get('hand_spec', {}).get('native_mesh_approximation') == 'convexDecomposition':
        prefix = source['hand_spec']['prefix']
        is_hand = lambda p: any(part.startswith(prefix) for part in str(p.GetPath()).split('/'))
        # Importer 6.1 collision_type only acts with collision_from_visuals=True.
        # Keep the source URDF collision meshes and explicitly author their
        # native approximation; otherwise the importer silently keeps hulls.
        for prim in list(stage.Traverse()):
            if is_hand(prim) and prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in stage.Traverse():
            if is_hand(prim) and prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
                decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                decomposition.CreateHullVertexLimitAttr(decomposition_parameters['hull_vertex_limit'])
                decomposition.CreateMaxConvexHullsAttr(decomposition_parameters['max_convex_hulls'])
                decomposition.CreateVoxelResolutionAttr(decomposition_parameters['voxel_resolution'])
                if 'shrink_wrap' in decomposition_parameters:
                    decomposition.CreateShrinkWrapAttr(bool(decomposition_parameters['shrink_wrap']))
                if 'error_percentage' in decomposition_parameters:
                    decomposition.CreateErrorPercentageAttr(float(decomposition_parameters['error_percentage']))
                collision_overrides.append(str(prim.GetPath()))
        if len(collision_overrides) != source['hand_spec']['hand_dof'] + 1:
            raise RuntimeError('LEAP native collision mesh inventory mismatch: ' + str(collision_overrides))
    stage.GetRootLayer().Save()
    articulation_paths = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    revolute = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    if len(articulation_paths) != 1 or set(revolute) != set(source['joint_limits']):
        raise RuntimeError(f'Imported articulation mismatch: roots={articulation_paths}, joints={revolute}')
    record = {'urdf_sha256': digest(urdf), 'usd_path': str(usd), 'usd_sha256': digest(usd),
              'importer': 'isaacsim.asset.importer.urdf', 'isaacsim_target_version': '6.1.0',
              'articulation_paths': articulation_paths, 'revolute_joint_names': revolute,
              'api_reference': 'https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.asset.importer.urdf/docs/index.html',
              'fixed_base': True, 'merge_fixed_joints': merge_fixed, 'self_collision': False,
              'native_convex_decomposition_meshes': collision_overrides,
              'native_convex_decomposition_parameters': decomposition_parameters if collision_overrides else None,
              'selected_physics_variant': 'physx', 'collision_type': source.get('hand_spec', {}).get('collision_type', 'Convex Hull'),
              'derived_files': {str(p): digest(p) for p in sorted((urdf.parent / 'usd').rglob('*')) if p.is_file()}}
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    return usd, source


def _import_side(urdf, usd_dir, manifest, stiffness, damping):
    """Import one bimanual arm URDF (cached by URDF/USD hashes like import_robot)."""
    if manifest.is_file():
        record = load_json(manifest)
        usd = Path(record['usd_path'])
        if record.get('urdf_sha256') == digest(urdf) and record.get('usd_sha256') == digest(usd):
            for path, expected in record['derived_files'].items():
                if digest(path) != expected:
                    raise ValueError('Cached robot USD dependency changed: ' + path)
            return usd
        raise ValueError('Existing derived USD does not match import manifest')
    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.asset.importer.urdf', True)
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    config = URDFImporterConfig(
        urdf_path=str(urdf), usd_path=str(usd_dir), merge_fixed_joints=False, merge_mesh=False,
        collision_from_visuals=False, collision_type='Convex Hull', allow_self_collision=False, fix_base=True,
        joint_drive_type='force', joint_target_type='position',
        override_joint_stiffness=stiffness, override_joint_damping=damping,
        run_asset_transformer=True, run_multi_physics_conversion=True)
    usd = Path(URDFImporter(config).import_urdf())
    if not usd.is_file():
        raise RuntimeError(f'Isaac URDF importer did not produce USD: {usd}')
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(usd))
    variant = stage.GetDefaultPrim().GetVariantSet('Physics')
    if 'physx' not in variant.GetVariantNames():
        raise RuntimeError('Imported robot is missing the native PhysX variant')
    variant.SetVariantSelection('physx')
    stage.GetRootLayer().Save()
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    revolute = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    if len(roots) != 1 or len(revolute) != 29:
        raise RuntimeError(f'Imported arm articulation mismatch: roots={roots}, joints={len(revolute)}')
    record = {'urdf_sha256': digest(urdf), 'usd_path': str(usd), 'usd_sha256': digest(usd),
              'importer': 'isaacsim.asset.importer.urdf', 'isaacsim_target_version': '6.1.0',
              'articulation_paths': roots, 'revolute_joint_names': revolute, 'fixed_base': True,
              'merge_fixed_joints': False, 'self_collision': False, 'selected_physics_variant': 'physx',
              'collision_type': 'Convex Hull',
              'derived_files': {str(p): digest(p) for p in sorted(usd_dir.rglob('*')) if p.is_file()}}
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    return usd


def import_bimanual(assets_root=DEFAULT_ASSETS, output=None, assembly_version=None):
    """Two articulations (left/right FR3 + SHARPA), one USD each, and a combined source."""
    from .bimanual_asset import BIMANUAL_ID, VERSION, assemble_bimanual
    usds, records, manifests = {}, {}, {}
    for side, (urdf, source) in assemble_bimanual(assets_root, output, assembly_version or VERSION).items():
        manifests[side] = urdf.parent / f'USD_IMPORT_{side}.json'
        usds[side] = _import_side(urdf, urdf.parent / f'usd_{side}', manifests[side],
                                  {f'fr3_{side}_joint.*': 2000., f'{side}_.*': 3.},
                                  {f'fr3_{side}_joint.*': 140., f'{side}_.*': .1})
        records[side] = source
    combined = {'schema_version': 'fr3-bimanual-isaacsim-source-v1', 'robot_id': BIMANUAL_ID, 'sides': records,
                'joint_limits': {**records['left']['joint_limits'], **records['right']['joint_limits']},
                'sources': {**records['left']['sources'], **records['right']['sources']},
                'mount': {side: records[side]['mount'] for side in records},
                'assembly_urdfs': {records[side]['urdf_path']: records[side]['urdf_sha256'] for side in records},
                'import_manifests': {side: str(path) for side, path in manifests.items()},
                'hand_spec': {'merge_fixed_joints': False}}
    return usds, combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--urdf-only', action='store_true')
    args = parser.parse_args()
    if args.urdf_only:
        path, _ = assemble_urdf(args.assets_root, args.output)
        print(path)
        return
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    try:
        path, _ = import_robot(args.assets_root, args.output)
        print(path, flush=True)
    finally:
        app.close()


if __name__ == '__main__':
    main()
