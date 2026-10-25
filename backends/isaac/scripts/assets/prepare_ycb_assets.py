"""Acquire twelve pinned official YCB 16k scans and build CPU-checked assets.

The original archives, textures, license, and attribution remain under the YCB
source root. Derived metre-scale OBJ files use bounded convex slab compounds;
no source file is edited and no simulator is launched. Native physics acceptance
is a separate operation. Rebuilds require a new output version.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, HalfspaceIntersection
import trimesh

from bench.paths import DATA_ROOT, load_json


SOURCE_ROOT = Path(f'{DATA_ROOT}/assets/objects/ycb/google_16k/20261014')
OUTPUT_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/ycb_manipulation/v1')
INDEX_URL = 'https://ycb-benchmarks.s3.amazonaws.com/index.html'
BASE_URL = 'https://ycb-benchmarks.s3.amazonaws.com/data/google/'
MASS_URL = 'https://www.eng.yale.edu/grablab/pubs/Calli_RAM2015.pdf'
LICENSE_URL = 'https://creativecommons.org/licenses/by/4.0/legalcode.en'
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_EXTRACTED_BYTES = 128 * 1024 * 1024
MAX_COLLISION_VERTICES = 192

# Masses: Calli et al., IEEE RAM September 2015, Table 2, printed page 43.
# They describe the nominal YCB objects, not a new measurement of each mesh.
# Slabs are an explicit approximation; banana/hammer are never global hulls.
SELECTION = {
    '002_master_chef_can': (.414, 1, '2f8dcc63e243911c546058476c35276448601997f5e658296ccdf4a83af24c10'),
    '003_cracker_box': (.411, 1, '0f084b0212274700d416384b9bc290521257d2b15ab66f68768a6a22806226b6'),
    '004_sugar_box': (.514, 1, '38e5820b942e7db501ba105cfb42ba8b470f27540dbab95324105a52de8617c6'),
    '005_tomato_soup_can': (.349, 1, '264f28559475e8e548b20420a7b6db944b4b27471039a73733d2c08a4ba2cc3d'),
    '006_mustard_bottle': (.603, 4, '992621f78a4fa6267da71418e530cd547bd066775da0242e578ddfb561d7cb5d'),
    '007_tuna_fish_can': (.171, 1, 'b44bde265f9ebab879680ad843fafedb9c6549c7414f7325c66faa49207cd502'),
    '010_potted_meat_can': (.370, 1, '197f0741d40c740e94bccb11dec6be7a4d6473dd3d140f500a0b4357efcab604'),
    '011_banana': (.066, 8, 'bad03e98066dec02ea1536ff2fa58f068efa97e9dbc89fceb4418f1ec633c881'),
    '036_wood_block': (.729, 1, 'df2d86634812e5d3aeecfbfc18f134921538c0093176eaa8247524ea0e684075'),
    '040_large_marker': (.0158, 3, '393d956da5f6cfe50614829faf4755bdb786c7db95abc1488f3b60505b1e2740'),
    '048_hammer': (.665, 12, '4745093ebe83e71e4ec80f5fa438a42a7175a0df6e5e2c6964a23085bc6ae195'),
    '061_foam_brick': (.028, 1, '945de5118d459a02ae65db06725fc08fdb84c66ae3efefb97cffd599fef679fa'),
}
DOCUMENTS = {
    'source_index.html': (INDEX_URL, 'd90b97268e9fd4eaf501efc22d17e27d419e9d94f498c6f6b6f63defd2a49891'),
    'LICENSE_CC_BY_4_0.html': (LICENSE_URL, '58230517b7895aa219ff46a6ed63e67057a1582fb9ea054b16728a8c13525650'),
    'Calli_RAM2015.pdf': (MASS_URL, '9fa0bc337e72ff0ccd9e17d8592ef2c332596c21c1cb011f9adf51f03a4fb27f'),
}
ATTRIBUTION = (
    'YCB Object and Model Set, by Berk Calli, Arjun Singh, Aaron Walsman, '
    'Siddhartha Srinivasa, Pieter Abbeel, and Aaron Dollar. Dataset: CC BY 4.0. '
    'Official Google scanner 16k meshes, acquisition snapshot 2026-10-14. '
    'Changes: OBJ recentered at approximate COM; textures preserved in source, '
    'display reduced to one sampled mean color; convex collision slabs generated; '
    'mass assigned from the authors\' published table; approximate COM/inertia computed. '
    'No endorsement by the authors is implied. '
    + INDEX_URL + ' License: ' + LICENSE_URL + '\n'
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_or_identical(path, payload):
    path = Path(path)
    if isinstance(payload, str):
        payload = payload.encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError('Refusing to change a versioned file: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def dump(path, value):
    write_new_or_identical(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def fetch_verified(url, path, expected_sha256, *, max_bytes=MAX_ARCHIVE_BYTES):
    path = Path(path)
    if path.exists():
        if path.is_symlink() or sha256(path) != expected_sha256:
            raise ValueError('Existing download hash mismatch: ' + str(path))
        return {'path': str(path), 'url': url, 'bytes': path.stat().st_size, 'sha256': expected_sha256}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    if temporary.exists():
        raise FileExistsError('Incomplete download requires explicit inspection: ' + str(temporary))
    request = Request(url, headers={'User-Agent': 'YCB-benchmark-asset-preparation/1.0'})
    try:
        with urlopen(request, timeout=120) as response, temporary.open('xb') as stream:
            advertised = response.headers.get('Content-Length')
            if advertised is not None and int(advertised) > max_bytes:
                raise ValueError('Download exceeds byte budget')
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError('Download exceeds byte budget')
                stream.write(chunk)
            if advertised is not None and total != int(advertised):
                raise ValueError('Truncated download')
            metadata = {'etag': response.headers.get('ETag'), 'last_modified': response.headers.get('Last-Modified')}
        if sha256(temporary) != expected_sha256:
            raise ValueError('Upstream bytes differ from the pinned acquisition; use a reviewed new version')
        temporary.replace(path)
    except BaseException:
        # Only remove the partial file created by this invocation.
        temporary.unlink(missing_ok=True)
        raise
    return {'path': str(path), 'url': url, 'bytes': total, 'sha256': expected_sha256, **metadata}


def extract_verified(archive, root, object_id, *, max_bytes=MAX_EXTRACTED_BYTES):
    """Extract regular files only, with no links, traversal, or hidden overwrite."""
    root = Path(root)
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        seen = set()
        total = 0
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or '\\' in member.name
                    or not name.parts or name.parts[0] != object_id
                    or member.name in seen or not (member.isfile() or member.isdir())):
                raise ValueError('Unsafe or duplicate archive member: ' + member.name)
            seen.add(member.name)
            total += member.size
            if total > max_bytes:
                raise ValueError('Extracted archive exceeds byte budget')
        for member in members:
            target = root / member.name
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('Archive target escapes the source root')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                data = stream.extractfile(member).read()
                if len(data) != member.size:
                    raise ValueError('Truncated tar member')
                write_new_or_identical(target, data)
    return total


def acquire(source_root=SOURCE_ROOT, *, allow_download=False):
    source_root = Path(source_root)
    jobs = [(BASE_URL + name + '_google_16k.tgz', source_root / 'archives' / (name + '_google_16k.tgz'), row[2])
            for name, row in SELECTION.items()]
    jobs += [(url, source_root / name, digest) for name, (url, digest) in DOCUMENTS.items()]
    if not allow_download and any(not path.is_file() for _, path, _ in jobs):
        raise FileNotFoundError('Source files missing; explicitly pass --download to acquire them')
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda row: fetch_verified(*row), jobs))
    for name in SELECTION:
        extract_verified(source_root / 'archives' / (name + '_google_16k.tgz'), source_root / 'extracted', name)
    write_new_or_identical(source_root / 'ATTRIBUTION.txt', ATTRIBUTION)
    inventory = {str(path.relative_to(source_root)): sha256(path)
                 for path in sorted(source_root.rglob('*'))
                 if path.is_file() and path.name not in ('source_manifest.json', 'download_receipt.json')}
    manifest = {'schema_version': 'ycb-google16k-source-v1', 'acquisition_snapshot': '20261014',
                'objects': list(SELECTION), 'license': 'CC-BY-4.0', 'index_url': INDEX_URL,
                'source_version': 'Official Google16k archive bytes last modified in April 2016; pinned SHA256',
                'files_sha256': inventory,
                'archives': [{'object_id': name, 'url': BASE_URL + name + '_google_16k.tgz',
                              'sha256': row[2], 'bytes': (source_root / 'archives' / (name + '_google_16k.tgz')).stat().st_size}
                             for name, row in SELECTION.items()]}
    dump(source_root / 'source_manifest.json', manifest)
    # Separate receipt preserves optional live response headers without placing
    # access time or network cache state in the stable source identity.
    if not (source_root / 'download_receipt.json').exists():
        dump(source_root / 'download_receipt.json', {'acquired_utc': datetime.now(timezone.utc).isoformat(), 'files': receipts})
    return manifest


def directions(count=80):
    index = np.arange(count)
    z = 1 - 2 * (index + .5) / count
    angle = index * math.pi * (3 - math.sqrt(5))
    return np.vstack([np.eye(3), -np.eye(3), np.column_stack([np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z])])


def convex_proxy(points, slab_axis=None):
    """Conservative convex hull, bounded to the native convex vertex budget."""
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    exact = trimesh.convex.convex_hull(points)
    if len(exact.vertices) <= MAX_COLLISION_VERTICES:
        return exact, 0.
    normals = np.vstack([np.eye(3), -np.eye(3)])
    if slab_axis is not None:
        normals = np.vstack([normals, slab_axis, -slab_axis])
    support = np.max(points @ normals.T, axis=0)
    interior = points.mean(axis=0)
    planes = np.column_stack([normals, -support])
    if np.any(normals @ interior - support >= -1e-10):
        raise ValueError('Convex piece has no well-conditioned interior')
    # Adapt to the actual hull faces rather than imposing a low-resolution
    # sphere of normals. This preserves flat box/can faces and cuts the largest
    # remaining hull-plane violation first, under a strict convex vertex cap.
    exact_planes = ConvexHull(points).equations
    for _ in range(96 - len(planes)):
        vertices = HalfspaceIntersection(planes, interior).intersections
        violations = vertices @ exact_planes[:, :3].T + exact_planes[:, 3]
        face = int(np.unravel_index(np.argmax(violations), violations.shape)[1])
        if violations.max() <= .0002:
            break
        planes = np.vstack([planes, exact_planes[face]])
    proxy = trimesh.convex.convex_hull(HalfspaceIntersection(planes, interior).intersections)
    if len(proxy.vertices) > MAX_COLLISION_VERTICES:
        raise ValueError('Collision proxy exceeds convex vertex budget')
    probe = directions(512)
    error = float(np.max(np.max(proxy.vertices @ probe.T, axis=0) - np.max(points @ probe.T, axis=0)))
    return proxy, error


def collision_slabs(mesh, count):
    """Partition along the longest PCA axis, retaining crossing-edge vertices.

    Each hull conservatively encloses its clipped surface slab. Slab compounds
    retain large longitudinal bends. Fine notches and cavities remain unverified.
    """
    vertices = np.asarray(mesh.vertices)
    _, _, axes = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)
    axis = axes[0]
    if axis[np.argmax(abs(axis))] < 0:
        axis = -axis
    coordinate = vertices @ axis
    edges = mesh.edges_unique
    endpoints = vertices[edges]
    endpoint_coordinate = coordinate[edges]
    intervals = np.linspace(coordinate.min(), coordinate.max(), count + 1)
    output = []
    errors = []
    for low, high in zip(intervals[:-1], intervals[1:]):
        points = [vertices[(coordinate >= low - 1e-12) & (coordinate <= high + 1e-12)]]
        for plane in (low, high):
            a, b = endpoint_coordinate.T
            mask = ((a - plane) * (b - plane) < 0) & (abs(b-a) > 1e-12)
            t = (plane - a[mask]) / (b[mask] - a[mask])
            points.append(endpoints[mask, 0] + t[:, None] * (endpoints[mask, 1] - endpoints[mask, 0]))
        piece, error = convex_proxy(np.concatenate(points), axis)
        if not piece.is_volume or piece.volume <= 0:
            raise ValueError('Collision slab is not a positive closed volume')
        output.append(piece)
        errors.append(error)
    return output, {'method': 'PCA surface clipping followed by conservative convex hulls',
                    'slabs': count, 'axis_in_source_frame': axis.tolist(),
                    'maximum_sampled_directional_support_expansion_m': max(errors),
                    'directional_error_samples': 518, 'support_error_is_not_hausdorff_bound': True}


def compound_mass_properties(pieces, mass):
    volumes = np.array([piece.volume for piece in pieces])
    total = float(volumes.sum())
    if not math.isfinite(mass) or mass <= 0 or not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError('Mass and convex volumes must be finite and positive')
    centers = np.array([piece.center_mass for piece in pieces])
    com = np.average(centers, weights=volumes, axis=0)
    tensor = np.zeros((3, 3))
    for piece, volume, center in zip(pieces, volumes, centers):
        delta = center - com
        tensor += piece.moment_inertia * mass / total
        tensor += mass * volume / total * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    if not np.isfinite(tensor).all() or np.linalg.eigvalsh(tensor).min() <= 0:
        raise ValueError('Derived inertia is not positive definite')
    return com, tensor


def _export(mesh, path):
    write_new_or_identical(path, trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False))
    return {'mesh_path': str(path), 'sha256': sha256(path)}


def convert_asset(object_id, source_root, output_root):
    if object_id not in SELECTION:
        raise ValueError('Object is outside the reviewed twelve-object selection')
    mass, slab_count, archive_sha = SELECTION[object_id]
    source = Path(source_root) / 'extracted' / object_id / 'google_16k'
    source_mesh = source / 'nontextured.ply'
    mesh = trimesh.load_mesh(source_mesh, process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not np.isfinite(mesh.vertices).all() or len(mesh.faces) < 4:
        raise ValueError('Invalid source triangle mesh')
    if np.any(mesh.extents <= .002) or max(mesh.extents) > .5:
        raise ValueError('Source dimensions do not match metre-scale tabletop geometry')
    # Do not repair/fill source holes silently. Every selected scan is required
    # to form an oriented closed solid before mass/volume comparisons are made.
    if not mesh.is_volume:
        raise ValueError('Source is not a watertight, consistently oriented solid')
    pieces, collision_report = collision_slabs(mesh, slab_count)
    com, tensor = compound_mass_properties(pieces, mass)
    source_volume = float(mesh.volume)
    collision_volume = sum(float(piece.volume) for piece in pieces)
    volume_ratio = collision_volume / source_volume
    # A large overfill stays outside the consumable catalog. Nothing silently
    # converts a badly concave object into an apparently qualified rigid body.
    if volume_ratio > 1.35:
        raise ValueError(f'Collision volume inflation {volume_ratio:.3f} exceeds 1.35 candidate limit')
    mesh.apply_translation(-com)
    for piece in pieces:
        piece.apply_translation(-com)
    asset_id = 'ycb_' + object_id
    folder = Path(output_root) / asset_id
    folder.mkdir(parents=True, exist_ok=False)
    visual_path = folder / 'visual.obj'
    _export(mesh, visual_path)
    # The existing task_parts renderer has no UV texture contract. Keep all
    # originals and disclose a single display color rather than claiming texture.
    image = Image.open(source / 'texture_map.png').convert('RGB')
    image.thumbnail((128, 128))
    color = (np.asarray(image, dtype=float).mean(axis=(0, 1)) / 255.).tolist()
    collisions = []
    for index, piece in enumerate(pieces):
        path = folder / f'collision_{index:02d}.obj'
        _export(piece, path)
        collisions.append({'geometry': 'convex_mesh', 'mesh_path': str(path.relative_to(output_root)),
                           'sha256': sha256(path), 'position_m': [0., 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.]})
    original_bounds = (mesh.bounds + com).tolist()
    off_diagonal = tensor - np.diag(np.diag(tensor))
    source_inputs = {str(path.relative_to(source_root)): sha256(path) for path in sorted(source.iterdir()) if path.is_file()}
    collision_report.update({'source_volume_m3': source_volume, 'collision_volume_m3': collision_volume,
                             'collision_source_volume_ratio': volume_ratio,
                             'max_collision_vertices': max(len(piece.vertices) for piece in pieces),
                             'source_vertices': len(mesh.vertices), 'source_triangles': len(mesh.faces),
                             'source_watertight': mesh.is_watertight, 'source_winding_consistent': mesh.is_winding_consistent})
    asset = {'name': asset_id, 'description': 'Official YCB ' + object_id[4:].replace('_', ' ') + '; development rigid-body approximation',
             'default_motion': 'dynamic', 'mesh_path': str(visual_path.relative_to(output_root)), 'sha256': sha256(visual_path),
             'mass_kg': mass, 'center_of_mass_m': [0., 0., 0.], 'inertia_diag_kg_m2': np.diag(tensor).tolist(),
             'full_inertia_tensor_kg_m2': tensor.tolist(), 'bounds_m': mesh.bounds.tolist(),
             'dimensions_m': mesh.extents.tolist(), 'visual_parts': [{'geometry': 'mesh',
                 'mesh_path': str(visual_path.relative_to(output_root)), 'sha256': sha256(visual_path),
                 'position_m': [0., 0., 0.], 'color': color, 'roughness': .65, 'metallic': 0.}],
             'collision_parts': collisions, 'license': 'CC-BY-4.0', 'source_object_id': object_id,
             'source_root': str(Path(source_root).resolve()), 'source_archive_sha256': archive_sha,
             'source_files_sha256': source_inputs, 'source_bounds_m': original_bounds,
             'source_to_asset_translation_m': (-com).tolist(), 'source_to_asset_rotation_xyzw': [0., 0., 0., 1.],
             'mass_source': {'url': MASS_URL, 'table': 'Table 2, printed page 43 / PDF page 8',
                             'mass_kg': mass, 'scope': 'Published nominal object mass; scan instance not reweighed'},
             'units': {'mesh_coordinate_scale_to_m': 1., 'basis': 'Numeric source coordinates preserved; tabletop dimensions checked; OBJ/PLY has no explicit unit tag'},
             'inertia_derivation': 'Uniform density over nonoverlapping convex slabs; recentered COM. Diagonal approximation in original scan axes; full tensor retained for audit.',
             'dropped_offdiagonal_frobenius_ratio': float(np.linalg.norm(off_diagonal) / np.linalg.norm(tensor)),
             'collision_model': 'Union of separately convex slab proxies; conservative coarse external-contact approximation',
             'geometry_audit': collision_report, 'materials_calibrated': False,
             'appearance': {'texture_used_by_runtime': False, 'original_texture_preserved': True,
                            'runtime': 'Single mean texture color; labels/details are unavailable in the current OBJ renderer'},
             'qualification': {'status': 'cpu_checked_native_unvalidated', 'native_physics_accepted': False,
                               'scene_loader_compatible': True, 'supports_external_rigid_contact_candidate': True,
                               'fine_concavities_and_insertion_qualified': False,
                               'deformable_material_model': False,
                               'limitations': ['Friction/restitution are scene assumptions, not YCB measurements.',
                                   'Uniform density ignores filled containers and tool material distributions.',
                                   'Fine rims, marker clip, hammer claw and banana concavity need native task-specific review.',
                                   'Foam brick is treated as rigid; no compression or elasticity claim.']}}
    return asset


def verify_catalog(root):
    from bench.simulation.assets.task_parts import load_task_asset, load_obj_mesh
    root = Path(root)
    manifest = load_json(root / 'manifest.json')
    checks = []
    for asset_id in manifest['assets']:
        asset = load_task_asset(asset_id, root, verify_hashes=True)
        collision_meshes = []
        for part in asset['collision_parts']:
            vertices, faces = load_obj_mesh(part['mesh_path'])
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            # OBJ text and the existing loader's float32 projection introduce
            # coplanar-face roundoff. Bound it quantitatively, rather than
            # relying on trimesh's exact coplanarity predicate.
            convex_volume_relative_error = abs(mesh.convex_hull.volume / mesh.volume - 1.) if mesh.is_volume else math.inf
            if not mesh.is_volume or convex_volume_relative_error > 1e-5 or len(vertices) > MAX_COLLISION_VERTICES:
                raise ValueError('Invalid convex collision mesh for ' + asset_id)
            collision_meshes.append(mesh)
        vertices, faces = load_obj_mesh(asset['mesh_path'])
        if not np.isfinite(vertices).all() or np.min(asset['inertia_diag_kg_m2']) <= 0:
            raise ValueError('Invalid mesh or inertia for ' + asset_id)
        if not np.allclose(np.stack([vertices.min(axis=0), vertices.max(axis=0)]), asset['bounds_m'], atol=1e-7):
            raise ValueError('Bounds do not match loaded OBJ for ' + asset_id)
        checks.append({'asset_id': asset_id, 'collision_parts': len(collision_meshes), 'loader_hashes_verified': True,
                       'float32_convex_volume_relative_error_tolerance': 1e-5})
    return {'schema_version': 'ycb-catalog-cpu-verification-v1', 'native_physics_accepted': False,
            'assets': checks, 'count': len(checks)}


def build(source_root=SOURCE_ROOT, output_root=OUTPUT_ROOT):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError('Use --verify or a new output version; refusing to overwrite ' + str(output_root))
    source_manifest = acquire(source_root, allow_download=False)
    staging = output_root.with_name(output_root.name + f'.building-{os.getpid()}')
    staging.mkdir(parents=True, exist_ok=False)
    assets, excluded = {}, {}
    for object_id in SELECTION:
        try:
            assets['ycb_' + object_id] = convert_asset(object_id, source_root, staging)
        except ValueError as exc:
            excluded['ycb_' + object_id] = {'source_object_id': object_id, 'status': 'excluded_collision_review_required', 'reason': str(exc)}
    manifest = {'schema_version': 'ycb-manipulation-assets-v1', 'source': INDEX_URL,
                'source_root': str(source_root), 'source_manifest_sha256': sha256(source_root / 'source_manifest.json'),
                'generator_sha256': sha256(__file__), 'license': 'CC-BY-4.0', 'units': 'm, kg, s',
                'attribution': ATTRIBUTION.strip(), 'native_physics_accepted': False,
                'assets': assets, 'excluded_assets': excluded,
                'source_archive_bytes': sum(row['bytes'] for row in source_manifest['archives']),
                'build_scope': 'Twelve official scans; CPU geometry/schema validation only; no simulator/model calls'}
    dump(staging / 'manifest.json', manifest)
    write_new_or_identical(staging / 'ATTRIBUTION.txt', ATTRIBUTION)
    write_new_or_identical(staging / 'LICENSE_CC_BY_4_0.html', (source_root / 'LICENSE_CC_BY_4_0.html').read_bytes())
    report = verify_catalog(staging)
    dump(staging / 'audit.json', report)
    dump(staging / 'source_provenance.json', {'source_manifest_sha256': sha256(source_root / 'source_manifest.json'),
                                            'source_root': str(source_root), 'source_files_sha256': source_manifest['files_sha256']})
    staging.rename(output_root)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify_catalog(args.output), indent=2))
        return
    if args.download or args.download_only:
        acquire(args.source_root, allow_download=args.download)
    if args.download_only:
        print(json.dumps({'source_root': str(args.source_root), 'objects': list(SELECTION)}))
        return
    result = build(args.source_root, args.output)
    print(json.dumps({'output': str(args.output), 'available_cpu_candidates': list(result['assets']),
                      'excluded': result['excluded_assets'], 'native_physics_accepted': False}, indent=2))


if __name__ == '__main__':
    main()
