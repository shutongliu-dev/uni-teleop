"""Index collaborator USDs and prepare a read-only-reference native catalog.

Run with shared assets/tools/usd/26.8/venv/bin/python. No simulator/GPU starts.
Source USDs, textures, composition and collision approximations stay unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdPhysics

from bench.paths import DATA_ROOT, load_json

DEFAULT_SOURCE = Path(f'{DATA_ROOT}/assets/objects/hand_system0/object_usd_20261011')
DEFAULT_OUTPUT = DEFAULT_SOURCE.parent / 'bench_catalog/v1'
PREFERRED = {
    'Robotool/name/cube_small': 'collab_cube_small',
    'Robotool/name/blue_cup': 'collab_blue_cup',
    'Robotool/name/bench_spoon_20261010': 'collab_bench_spoon',
    'Robotool/name/peg': 'collab_peg',
    'Robotool/name/peg_hole': 'collab_peg_hole',
    'Robotool/name/rubber_mallet': 'collab_rubber_mallet',
    'SPIDER/name/apple': 'collab_apple',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(item):
    p = item['asset_path']
    return (0 if p.startswith('robotool/batch/') else 1 if p.startswith('spider/') else 2,
            len(p), p)


def json_value(value):
    if value is None or isinstance(value, (str, bool, float, int)):
        return value
    if isinstance(value, Gf.Quatf):
        return [float(value.GetReal()), *map(float, value.GetImaginary())]
    return [float(x) for x in value]


def collect_dependencies(stage):
    files, builtins = {}, set()
    for layer in stage.GetUsedLayers():
        if layer.realPath:
            files[str(Path(layer.realPath).resolve())] = digest(layer.realPath)
        for asset in layer.GetExternalReferences():
            if asset in ('OmniPBR.mdl', 'OmniPBR_Opacity.mdl'):
                builtins.add(asset)
            elif asset:
                resolved = str(Ar.GetResolver().Resolve(Sdf.ComputeAssetPathRelativeToLayer(layer, asset)))
                if not resolved:
                    raise ValueError(f'Unresolved layer dependency: {asset}')
                files[str(Path(resolved).resolve())] = digest(resolved)
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        for attr in prim.GetAuthoredAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                continue
            values = attr.Get()
            values = list(values or []) if attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray else [values]
            stack = attr.GetPropertyStack()
            layer = stack[0].layer if stack else stage.GetRootLayer()
            for asset in values:
                if asset is None or not asset.path:
                    continue
                if asset.path in ('OmniPBR.mdl', 'OmniPBR_Opacity.mdl'):
                    builtins.add(asset.path)
                    continue
                resolved = asset.resolvedPath or str(Ar.GetResolver().Resolve(Sdf.ComputeAssetPathRelativeToLayer(layer, asset.path)))
                if not resolved:
                    raise ValueError(f'Unresolved material dependency: {asset.path}')
                files[str(Path(resolved).resolve())] = digest(resolved)
    return dict(sorted(files.items())), sorted(builtins)


def inspect_asset(source, item, asset_id, output):
    path = source / item['asset_path']
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    if not root or not root.IsA(UsdGeom.Xform):
        raise ValueError('A default Xform prim is required')
    if str(UsdGeom.GetStageUpAxis(stage)) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError('Only explicitly meter-scale Z-up sources are accepted')
    if not Gf.IsClose(UsdGeom.Xformable(root).GetLocalTransformation(), Gf.Matrix4d(1), 1e-9):
        raise ValueError('Nonidentity source root transform requires an explicit frame adapter')
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    rigid = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(rigid) != 1 or rigid[0] != root:
        raise ValueError('Exactly one rigid body at the default prim is required')
    if any(p.HasAPI(UsdPhysics.ArticulationRootAPI) or p.IsA(UsdPhysics.Joint) for p in prims):
        raise ValueError('Articulated source needs explicit separate adaptation')
    mass_api = UsdPhysics.MassAPI(root)
    mass = mass_api.GetMassAttr().Get() if mass_api else None
    if mass is None or not 0 < mass < 100:
        raise ValueError('Explicit positive mass below 100 kg is required')
    dependencies, builtins = collect_dependencies(stage)
    cache = UsdGeom.XformCache()
    parts = []
    materials = []
    for prim in prims:
        if prim.GetTypeName() == 'Material':
            materials.append(str(prim.GetPath()))
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f'Non-mesh source collider requires explicit adapter: {prim.GetPath()}')
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        record = {'usd_prim_path': str(prim.GetPath()), 'approximation': str(approximation),
                  'geometry': 'usd_' + str(approximation), 'position_m': [0, 0, 0],
                  'quaternion_xyzw': [0, 0, 0, 1]}
        # Only an authored convex hull is independently represented as a hull.
        # A concave decomposition must never be replaced by its global hull.
        if approximation == 'convexHull':
            geometry = UsdGeom.Mesh(prim)
            transform = cache.GetLocalToWorldTransform(prim)
            points = [transform.Transform(Gf.Vec3d(*x)) for x in geometry.GetPointsAttr().Get()]
            indices = list(geometry.GetFaceVertexIndicesAttr().Get())
            counts = list(geometry.GetFaceVertexCountsAttr().Get())
            obj = output / 'audit_collision' / asset_id / f'{len(parts):03d}.obj'
            obj.parent.mkdir(parents=True, exist_ok=True)
            lines = ['# Source collider transformed into the asset root frame.']
            lines.extend('v %.12g %.12g %.12g' % tuple(p) for p in points)
            cursor = 0
            for count in counts:
                face = indices[cursor:cursor + count]
                for j in range(1, count - 1):
                    lines.append('f %d %d %d' % (face[0] + 1, face[j] + 1, face[j + 1] + 1))
                cursor += count
            obj.write_text('\n'.join(lines) + '\n')
            record.update(geometry='convex_mesh', mesh_path=str(obj.relative_to(output)), sha256=digest(obj))
        parts.append(record)
    if not parts:
        raise ValueError('No enabled colliders')
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                    UsdGeom.Tokens.proxy]).ComputeWorldBound(root).ComputeAlignedBox()
    inertia = mass_api.GetDiagonalInertiaAttr().Get()
    center = mass_api.GetCenterOfMassAttr().Get()
    # USD defaults (zero inertia or sentinel COM) ask PhysX to compute geometry.
    authored_inertia = json_value(inertia) if mass_api.GetDiagonalInertiaAttr().HasAuthoredValueOpinion() else None
    authored_center = json_value(center) if mass_api.GetCenterOfMassAttr().HasAuthoredValueOpinion() else None
    return {'backend': 'usd_reference', 'label': item['object_name_en'],
            'source_model_key': item['model_key'], 'source_label': item,
            'usd_path': str(path), 'usd_default_prim': str(root.GetPath()),
            'source_dependencies_sha256': dependencies, 'runtime_builtin_dependencies': builtins,
            'mass_kg': float(mass), 'center_of_mass_m': authored_center,
            'inertia_diag_kg_m2': authored_inertia,
            'bounds_min_m': list(bounds.GetMin()), 'bounds_max_m': list(bounds.GetMax()),
            'dimensions_m': list(bounds.GetSize()), 'collision_parts': parts,
            'visual_parts': [{'usd_reference': str(root.GetPath())}],
            'material_prim_paths': materials,
            'physics_assumptions': {'mass': 'Source authored mass, not experimentally calibrated.',
                                    'inertia': 'Source authored where present; otherwise PhysX computes from source collision geometry.',
                                    'root_frame': 'Unscaled source default-prim frame; not necessarily center of mass.',
                                    'collision': 'Source collision mesh composition and approximation retained.',
                                    'friction': 'Source physics material retained; unbound shapes use explicit fallback 0.65 unless scene overrides.',
                                    'independent_geometry_audit': 'Authored convexHull meshes exported individually; cooked convexDecomposition remains unsupported for cloth clearance certification.'},
            'qualification': 'CPU composition only; native qualification recorded separately.'}


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=True)
    labels = load_json(source / 'asset_labels.json')['assets']
    entries = [x for x in labels if x['usd_role'] == 'entry']
    groups = defaultdict(list)
    exact = defaultdict(list)
    for item in entries:
        groups[item['model_key']].append(item)
        exact[digest(source / item['asset_path'])].append(item['asset_path'])
    selection, assets, rejected = [], {}, []
    for key, candidates in sorted(groups.items()):
        candidates.sort(key=rank)
        asset_id = PREFERRED.get(key, 'collab_' + re.sub('[^a-z0-9]+', '_', key.lower()).strip('_'))
        failures = []
        for item in candidates:
            try:
                asset = inspect_asset(source, item, asset_id, output)
                assets[asset_id] = asset
                selected = item['asset_path']
                break
            except Exception as error:
                failures.append({'asset_path': item['asset_path'], 'reason': str(error)})
        else:
            selected = None
            rejected.append({'model_key': key, 'attempts': failures})
        selection.append({'model_key': key, 'asset_id': asset_id if selected else None,
                          'selected_entry': selected, 'entry_count': len(candidates),
                          'entries': [x['asset_path'] for x in candidates], 'unsupported_attempts': failures})
    index = {'schema': 'collaborator_usd_selection.v1', 'source_root': str(source),
             'usd_entry_count': len(entries), 'name_or_id_groups': len(groups),
             'exact_entry_byte_hash_groups': len(exact), 'available_asset_count': len(assets),
             'unique_physical_objects_claimed': False,
             'deduplication': 'One selectable compatible entry per collaborator model_key; exact file hashes are separately indexed, not a mesh uniqueness claim.',
             'groups': selection, 'exact_entry_hashes': dict(exact), 'rejected_groups': rejected}
    (output / 'selection_index.json').write_text(json.dumps(index, indent=2) + '\n')
    manifest = {'schema': 'collaborator_native_usd_catalog.v1',
                'created_at': datetime.now(timezone.utc).isoformat(), 'source_root': str(source),
                'source_labels_sha256': digest(source / 'asset_labels.json'),
                'selection_index_sha256': digest(output / 'selection_index.json'),
                'generator': {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)},
                'license': 'No separate redistribution license supplied. Source files are unchanged; no license inferred.',
                'source_provenance': str(source.parent / 'object_usd_20261011.provenance.json'),
                'assets': assets}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'manifest': str(output / 'manifest.json'), 'available': len(assets),
                      'entry_count': len(entries), 'groups': len(groups), 'rejected': rejected}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve())
