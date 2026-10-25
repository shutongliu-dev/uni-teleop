"""Prepare an immutable catalog referencing the exact original threaded assets.

CPU OpenUSD only. No USD or mesh is copied, rescaled, rewritten or simulated.
The catalog feeds the existing usd_reference loader, including child-body assets.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .threaded_assets import ASSEMBLY_ROOT, TOOL_ROOT, validated_threaded_assets


ASSET_IDS = {
    'screwdriver': 'mechanical_screwdriver_v1',
    'bolt': 'mechanical_bolt_m16',
    'fixture': 'mechanical_threaded_fixture_m16',
}
UPSTREAM_SCENE_RELATIVE = 'provenance/github/robobench/suites/assembly/scenes/allen_bolt_assembly.py'
UPSTREAM_SCENE_SHA256 = '09feabc1aff1af9f19fd39635d612ab5dfa15985708d93da83d0468b051185d8'


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_value(value):
    from pxr import Gf
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (Gf.Quatd, Gf.Quatf, Gf.Quath)):
        return [float(value.GetReal()), *map(float, value.GetImaginary())]
    return [_json_value(item) for item in value]


def _authored_physics(prim):
    return {attr.GetName(): _json_value(attr.Get()) for attr in prim.GetAuthoredAttributes()
            if attr.GetName().startswith(('physics:', 'physx'))}


def _authored_api_schemas(prim):
    # usd-core lacks PhysxSchema, so GetAppliedSchemas omits its unknown names.
    schemas = prim.GetMetadata('apiSchemas')
    return list(schemas.GetAppliedItems()) if schemas else []


def _identity(matrix, tolerance=1.e-8):
    return all(abs(matrix[i][j] - float(i == j)) <= tolerance
               for i in range(4) for j in range(4))


def _inspect_asset(binding, *, role, provenance_dependencies, source_scale=1.):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    from scripts.assets.prepare_collaborator_catalog import collect_dependencies
    from bench.simulation.isaacsim_backend.workroom import BUILTIN_MDL

    path = Path(binding['path'])
    if _digest(path) != binding['sha256']:
        raise ValueError('Threaded USD differs from the selected source')
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    if (not root or str(root.GetPath()) != binding['default_prim']
            or not root.IsA(UsdGeom.Xform)):
        raise ValueError('Threaded source default prim differs from its pinned contract')
    if UsdGeom.GetStageMetersPerUnit(stage) != 1. or str(UsdGeom.GetStageUpAxis(stage)) != 'Z':
        raise ValueError('Original threaded assets must remain metre-scale and Z-up')
    root_transform = UsdGeom.Xformable(root).GetLocalTransformation()
    expected_transform = Gf.Matrix4d(1.)
    expected_transform.SetScale(float(source_scale))
    if any(abs(root_transform[i][j] - expected_transform[i][j]) > 1.e-8
           for i in range(4) for j in range(4)):
        raise ValueError('Source default frame differs from the pinned uniform scale')
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise ValueError('Threaded reference must contain exactly one rigid body')
    if any(prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.IsA(UsdPhysics.Joint)
           or prim.IsA(UsdPhysics.Scene) for prim in prims):
        raise ValueError('A task part cannot import joints, articulations or a PhysicsScene')
    body = bodies[0]
    transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(body, root)
    if not _identity(transform):
        raise ValueError('Body must have the same origin, orientation and unit scale as the default prim')
    if UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is False:
        raise ValueError('Source body is disabled')
    gravity = body.GetAttribute('physxRigidBody:disableGravity')
    if role != 'fixture' and gravity and gravity.HasAuthoredValueOpinion() and gravity.Get():
        raise ValueError('Dynamic source disables gravity')
    mass_api = UsdPhysics.MassAPI(body)
    mass = mass_api.GetMassAttr().Get()
    if mass is None or not math.isfinite(mass) or mass <= 0:
        raise ValueError('Source body needs a positive authored mass')
    dependencies, builtins = collect_dependencies(stage)
    dependencies.update(provenance_dependencies)
    builtin_files = {}
    for name in builtins:
        if name not in BUILTIN_MDL:
            raise ValueError('Unknown runtime MDL dependency: ' + name)
        builtin_files[name] = {'path': name, 'provided_by': 'Isaac Sim core MDL library'}
    parts, collision_audit = [], []
    cache = UsdGeom.XformCache()
    for prim in prims:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError('Unexpected non-mesh collider: ' + str(prim.GetPath()))
        approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        if approximation not in ('convexHull', 'sdf', 'none'):
            raise ValueError('Unexpected collision approximation: ' + approximation)
        if role != 'fixture' and approximation == 'none':
            raise ValueError('Uncooked triangle-mesh dynamic collider is unsupported by this contract')
        geometry = UsdGeom.Mesh(prim)
        points = geometry.GetPointsAttr().Get()
        counts = geometry.GetFaceVertexCountsAttr().Get()
        indices = geometry.GetFaceVertexIndicesAttr().Get()
        if not points or not indices or sum(counts) != len(indices):
            raise ValueError('Invalid source collision topology')
        matrix = cache.GetLocalToWorldTransform(prim)
        transformed = [matrix.Transform(Gf.Vec3d(*point)) for point in points]
        bounds = [[min(point[axis] for point in transformed) for axis in range(3)],
                  [max(point[axis] for point in transformed) for axis in range(3)]]
        if not all(math.isfinite(value) for row in bounds for value in row):
            raise ValueError('Nonfinite collision geometry')
        signature = json.dumps({'points': _json_value(points), 'counts': list(counts),
                                'indices': list(indices), 'local_to_asset': _json_value(matrix)},
                               separators=(',', ':'), sort_keys=True).encode()
        material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        physics_material = (material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI))
        material_record = {
            'source_material_path': str(material.GetPath()) if material else None,
            'source_binding_path': str(relationship.GetPath()) if relationship else None,
            'has_physics_material_api': bool(physics_material),
            'authored_parameters': _authored_physics(material.GetPrim()) if physics_material else {},
            'runtime_expectation': 'Preserve original material' if physics_material else 'Explicit runtime fallback; source USD has no physics material',
        }
        record = {'usd_prim_path': str(prim.GetPath()), 'approximation': approximation,
                  'geometry': 'usd_' + approximation, 'position_m': [0., 0., 0.],
                  'quaternion_xyzw': [0., 0., 0., 1.]}
        parts.append(record)
        collision_audit.append({**record, 'point_count': len(points),
            'triangulated_face_count': sum(int(count) - 2 for count in counts),
            'bounds_m': bounds, 'geometry_and_transform_sha256': hashlib.sha256(signature).hexdigest(),
            'authored_api_schemas': _authored_api_schemas(prim),
            'authored_parameters': _authored_physics(prim), 'physics_material': material_record})
    if not parts:
        raise ValueError('Source has no enabled collision geometry')
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy']).ComputeWorldBound(root).ComputeAlignedRange()
    def authored(attr):
        return _json_value(attr.Get()) if attr.HasAuthoredValueOpinion() else None
    info = {
        'backend': 'usd_reference', 'label': role,
        'usd_path': str(path), 'usd_default_prim': str(root.GetPath()),
        'source_rigid_body_prim': str(body.GetPath()),
        'placement_form': 'root' if body == root else 'anchored',
        'mass_kg': float(mass), 'center_of_mass_m': authored(mass_api.GetCenterOfMassAttr()),
        'inertia_diag_kg_m2': authored(mass_api.GetDiagonalInertiaAttr()),
        'bounds_min_m': list(bounds.GetMin()), 'bounds_max_m': list(bounds.GetMax()),
        'dimensions_m': list(bounds.GetSize()), 'source_scale': [source_scale] * 3,
        'source_dependencies_sha256': dict(sorted(dependencies.items())),
        'runtime_builtin_dependencies': builtins, 'runtime_builtin_files': builtin_files,
        'collision_parts': parts, 'visual_parts': [{'usd_reference': str(root.GetPath())}],
        'material_prim_paths': [str(prim.GetPath()) for prim in prims if prim.GetTypeName() == 'Material'],
        'physics_assumptions': {
            'mass_and_inertia': 'Source simulation values, not measured. Unauthored COM/inertia remain for PhysX to compute.',
            'collision': 'Original composed geometry, SDF resolution and approximation preserved; no global convex-hull substitution.',
            'friction': 'Screwdriver source physics material retained. Bolt/fixture need a recorded runtime fallback; no upstream scene friction rewrite is imported.',
            'placement': 'Original metre-scale, Z-up, scale-1 frame; the single rigid body is at the wrapper origin.',
            'fixture': 'Fixture may only be used as a kinematic task fixture; its plate uses a triangle mesh.',
        },
        'qualification': 'CPU composition audit only; native cooking, contact and manipulation not validated.',
        'source_modified': False,
    }
    audit = {'source': deepcopy(binding), 'body_path': str(body.GetPath()),
             'body_authored_api_schemas': _authored_api_schemas(body),
             'body_authored_parameters': _authored_physics(body),
             'body_relative_to_default': _json_value(transform), 'colliders': collision_audit,
             'source_scale': [source_scale] * 3, 'placement_form': info['placement_form'],
             'native_physics_validated': False}
    return info, audit


def prepare_threaded_catalog(output, *, tool_root=TOOL_ROOT, assembly_root=ASSEMBLY_ROOT):
    """Create SOURCE.json/manifest.json in a NEW version directory, without USD copies."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new mechanical catalog version; output already exists: ' + str(output))
    contract = validated_threaded_assets(tool_root=tool_root, assembly_root=assembly_root)
    scene_path = Path(assembly_root) / UPSTREAM_SCENE_RELATIVE
    if _digest(scene_path) != UPSTREAM_SCENE_SHA256:
        raise ValueError('Fallback parameter provenance changed')
    assets, audits = {}, {}
    for role, binding in contract['assets'].items():
        source_name = 'tool' if role == 'screwdriver' else 'assembly'
        source = contract['source_manifests'][source_name]
        license_info = contract['licenses'][source_name]
        provenance = {source['path']: source['sha256'], license_info['path']: _digest(license_info['path']),
                      str(scene_path): UPSTREAM_SCENE_SHA256}
        if 'notice' in license_info:
            provenance[license_info['notice']] = _digest(license_info['notice'])
        info, audit = _inspect_asset(binding, role=role, provenance_dependencies=provenance)
        assets[ASSET_IDS[role]], audits[role] = info, audit
    options = {
        'usd_physics_policy': 'preserve_authored_v1',
        'usd_physics_fallbacks': {'contact_offset_m': .0002, 'rest_offset_m': 0.,
                                'solver_position_iterations': 192, 'solver_velocity_iterations': 1},
        'usd_physics_fallbacks_provenance': {
            'path': str(scene_path), 'sha256': UPSTREAM_SCENE_SHA256,
            'scope': 'Only absent USD fields use these upstream-scene simulation priors. Source-authored values take priority; no upstream controller/weld/friction code is imported.',
            'native_validated': False,
        },
    }
    report = {
        'schema': 'mechanical_threaded_catalog_source.v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'generator': {'path': str(Path(__file__).resolve()), 'sha256': _digest(__file__)},
        'source_manifests': contract['source_manifests'], 'licenses': contract['licenses'],
        'asset_ids': deepcopy(ASSET_IDS), 'assets': audits,
        'recommended_spec_options': options,
        'source_modified': False, 'native_physics_validated': False,
        'scope': 'CPU source geometry/provenance audit only; static stability, native collision cooking, preinserted reset and manipulation need separate rollouts.',
    }
    # No source USD, mesh, texture or license is duplicated into this directory.
    output.mkdir(parents=True, exist_ok=False)
    source_path = output / 'SOURCE.json'
    source_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for info in assets.values():
        info['source_dependencies_sha256'][str(source_path)] = _digest(source_path)
    manifest = {'schema': 'collaborator_native_usd_catalog.v1',
                'created_at': report['created_at_utc'], 'source_provenance': str(source_path),
                'source_provenance_sha256': _digest(source_path), 'assets': assets,
                'license': 'Per-source terms retained at the pinned license/NOTICE paths in SOURCE.json; no combined relicensing.',
                'generator': report['generator'], 'qualification': report['scope']}
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (output / 'SHA256SUMS').write_text(''.join(f'{_digest(path)}  {path.name}\n'
                                            for path in (source_path, manifest_path)))
    return {'manifest_path': str(manifest_path), 'manifest_sha256': _digest(manifest_path),
            'source_audit_path': str(source_path), 'source_audit_sha256': _digest(source_path),
            'asset_ids': deepcopy(ASSET_IDS), 'recommended_spec_options': deepcopy(options),
            'source_modified': False, 'native_physics_validated': False}
