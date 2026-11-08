"""Read static original pad meshes; no scene, observation or task truth inputs."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from bench.geometry import matrix_quaternion, rpy_matrix, transform
from bench.paths import materialize
from bench.protocol.robot import PadGeometry


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=16)
def _mesh_vertices(raw, suffix):
    """Cache by source bytes, never by a mutable filename or caller object."""
    import trimesh
    mesh = trimesh.load(io.BytesIO(raw), file_type=suffix, force='mesh', process=False)
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError('Invalid source pad collision mesh')
    vertices.flags.writeable = False
    return vertices


@lru_cache(maxsize=16)
def _hand_tree(raw, namespace):
    tree = ET.fromstring(raw)
    if namespace:
        from bench.simulation.isaacsim_backend.instance_asset import namespaced_tree
        tree = namespaced_tree(tree, namespace)
    return tree


def _origin(element):
    origin = element.find('origin')
    def vector(name):
        value = np.fromstring(origin.get(name, '0 0 0') if origin is not None else '0 0 0', sep=' ')
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Invalid source pad origin')
        return value
    return transform(vector('xyz'), rpy_matrix(vector('rpy')))


def source_pad_geometries(model):
    """Project verified SHARPA fixed-tip source geometry into public tip frames.

    Non-SHARPA embodiments remain compatible by omitting this optional field.
    Original files are rehashed before cached bytes/meshes are reused; neither
    development reports nor native object/contact state are read.
    """
    if not getattr(model.embodiment, 'sharpa_joint_semantics', False):
        return {}
    source = Path(model.hand_path)
    raw = source.read_bytes()
    digest = _sha(raw)
    if digest != model.asset_sha256[str(source)]:
        raise ValueError('Source hand geometry differs from the pinned URDF')
    namespace = model.embodiment.side if model.embodiment.hand_id == 'fr3_sharpa_instances' else None
    tree = _hand_tree(raw, namespace)
    from bench.simulation.isaacsim_backend.l_mount import SUPPORTED_VERSIONS as L_MOUNT_VERSIONS
    l_mount = model.embodiment.assembly_version in L_MOUNT_VERSIONS
    if l_mount:
        assembly = Path(model.arm_path)
        manifest = assembly.parent / ('SOURCE.json' if namespace is not None or model.embodiment.hand_id == 'sharpa_w01_right'
                                      else f'SOURCE_{model.embodiment.side}.json')
        source_manifest_raw = manifest.read_bytes()
        record = materialize(json.loads(source_manifest_raw))  # data:// references -> local paths
        source_files = {**record['sources'], **record['mesh_inputs']}
        if source_files.get(str(source)) != digest:
            raise ValueError('L-mount hand file differs from the physical assembly inputs')
    else:
        source_root = source.parent.parent
        source_manifest_raw = (source_root / 'SOURCE.json').read_bytes()
        source_files = json.loads(source_manifest_raw)['files']
    links = {link.get('name'): link for link in tree.findall('link')}
    children = {joint.find('child').get('link'): joint for joint in tree.findall('joint')}
    model_children = {joint.child: joint for joint in model.hand.joints}
    output = {}
    for finger, tip in model.fingertip_links.items():
        joint, fixed = children.get(tip), model_children.get(tip)
        if (joint is None or fixed is None or joint.get('type') != 'fixed' or fixed.kind != 'fixed'
                or not fixed.parent.endswith('_elastomer') or joint.find('parent').get('link') != fixed.parent
                or finger in model.tip_offsets):
            # Do not infer a pad for an unfamiliar virtual-tip convention.
            continue
        pad = links[fixed.parent]
        pad_from_tip = _origin(joint)
        if not np.allclose(pad_from_tip, fixed.origin, atol=1e-12, rtol=0.):
            raise ValueError('Source pad transform differs from robot kinematics')
        tip_from_pad = np.linalg.inv(pad_from_tip)
        vertices, hashes = [], {'urdf': digest, 'source_manifest': _sha(source_manifest_raw)}
        collisions = pad.findall('collision')
        if not 1 <= len(collisions) <= 16:
            raise ValueError('Missing or excessive source pad collision geometry')
        for index, collision in enumerate(collisions):
            mesh = collision.find('geometry/mesh')
            if mesh is None or not mesh.get('filename') or '://' in mesh.get('filename'):
                raise ValueError('Unsupported source pad collision geometry')
            path = (source.parent / mesh.get('filename')).resolve(strict=True)
            mesh_raw = path.read_bytes()
            hashes[f'collision_mesh_{index}'] = _sha(mesh_raw)
            expected = (source_files.get(str(path)) if l_mount else
                        source_files[path.relative_to(source_root / 'upstream').as_posix()]['sha256'])
            if expected != hashes[f'collision_mesh_{index}']:
                raise ValueError('Source pad mesh differs from original robot input manifest')
            points = _mesh_vertices(mesh_raw, path.suffix.lstrip('.').lower())
            scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale == 0.):
                raise ValueError('Invalid source pad mesh scale')
            tip_from_mesh = tip_from_pad @ _origin(collision)
            vertices.append((tip_from_mesh @ np.c_[points * scale, np.ones(len(points))].T).T[:, :3])
        points = np.concatenate(vertices)
        low, high = points.min(0), points.max(0)
        output[finger] = PadGeometry(pad_link=fixed.parent,
            tip_from_pad={'frame_id': tip, 'position_m': tip_from_pad[:3, 3].tolist(),
                          'quaternion_wxyz': matrix_quaternion(tip_from_pad[:3, :3]).tolist()},
            bounds_in_tip_frame_m=[low.tolist(), high.tolist()],
            bounds_center_in_tip_frame_m=((low + high) / 2).tolist(), source_hashes=hashes
        ).model_dump(mode='json')
    return output
