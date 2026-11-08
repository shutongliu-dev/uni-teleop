"""Offline geometry/IK adapter for frozen ordinary workroom seed cases.

Reuses the living diagnostic's endpoint/clearance algorithm, without inheriting
its task list, actor assumptions or qualification. Never starts native physics.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path
import sys

import numpy as np

from bench.isaacsim.release import digest
from scenes.workrooms.initial_randomization import TASK_FAMILIES, SEEDS, configurations, selection
from scenes.workrooms.initial_randomization.common import read_json, canonical_hash
from scenes.workrooms.initial_randomization.physical_inputs import object_actor_path
from scripts.validation import living_randomized_workspace as living

require = living.require
schema_prefix = 'workroom_randomized'


def read_request(path):
    request = read_json(path)
    require(isinstance(request, dict) and set(request) <= {'schema', 'cases', 'duration_s'}
            and request.get('schema') == 'workroom_randomized_validation_request.v1', 'Unexpected request schema')
    cases = request.get('cases')
    maximum = sum(len(configurations(task)) for task in TASK_FAMILIES) * len(SEEDS)
    require(isinstance(cases, list) and 1 <= len(cases) <= maximum,
            f'Require 1–{maximum} explicit ordinary cases')
    seen = set()
    for case in cases:
        require(isinstance(case, dict) and {'task_id', 'configuration_id', 'seed', 'release'} <= set(case)
                and set(case) <= {'family', 'task_id', 'configuration_id', 'seed', 'release', 'release_manifest_sha256'},
                'Unexpected case fields')
        selection(case['task_id'], case['seed'], case['configuration_id'])
        case.setdefault('family', TASK_FAMILIES[case['task_id']])
        require(case['family'] == TASK_FAMILIES[case['task_id']], 'Task family differs')
        require(isinstance(case['release'], str) and Path(case['release']).is_absolute(), 'Absolute release required')
        if 'release_manifest_sha256' in case:
            value = case['release_manifest_sha256']
            require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                    'Invalid explicit release manifest SHA256')
        key = (case['family'], case['task_id'], case['configuration_id'], case['seed'])
        require(key not in seen, 'Duplicate case'); seen.add(key)
    duration = request.get('duration_s', .8)
    require(type(duration) in (int, float) and math.isfinite(duration) and .6 <= duration <= 3.
            and math.isclose(round(duration/.02)*.02, duration, abs_tol=1e-9), 'Invalid passive duration')
    return {**request, 'duration_s': float(duration)}


def case_name(case):
    return case['family']+'/'+living.case_name(case)


def validate_case(case, scene, task):
    reset = task.get('reset_case', {})
    require(reset.get('schema') == 'workroom_reset_case.v1', 'Missing workroom reset case')
    require(all(reset.get(k) == case[k] for k in ('family', 'task_id', 'seed', 'configuration_id'))
            and TASK_FAMILIES.get(task['id']) == case['family'], 'Case selector differs from frozen inputs')
    require(reset.get('case_sha256') == canonical_hash({k: v for k, v in reset.items() if k != 'case_sha256'}),
            'Reset case hash differs')
    require(scene.get('workroom', {}).get('family') == case['family'], 'Wrong fixed parent room')
    if 'release_manifest_sha256' in case:
        require(digest(Path(case['release'])/'release_manifest.json') == case['release_manifest_sha256'],
                'Selected release manifest differs from the request')


def installation(scene, initial, family, task_id, configuration_id):
    """Retain the old menu, with one explicit fixed cooperative installation."""
    if task_id != 'cooperative_insertion':
        return living.installation(scene, initial, family, task_id, configuration_id)
    from bench.control.factory import public_robot_geometry, robot_configuration
    from scenes.workrooms.mechanical_assembly import cooperative_robot_layout as layout
    require(family == 'mechanical_assembly' and configuration_id == 'dual_same_side'
            and 'robot_configuration' not in scene, 'Cooperative insertion retains its fixed original installation')
    robot, expected_initial, _ = layout.robot_inputs(layout.MIRRORED_LAYOUT_ID)
    require(scene['robot'] == robot and initial['joint_positions_rad'] == expected_initial['joint_positions_rad'],
            'Cooperative robot or ready joints differ from its original fixed installation')
    models = public_robot_geometry(robot_configuration(scene)).sides
    poses = layout.placements(layout.MIRRORED_LAYOUT_ID)
    require(set(models) == set(poses) == {'left', 'right'}, 'Original cooperative hand inventory differs')
    for side, model in models.items():
        require(np.allclose(model.arm_base[:3, 3], poses[side]['base_position_m'], atol=1e-9, rtol=0),
                'Actual cooperative arm base differs')
    return models, layout.stand_geometry(layout.MIRRORED_LAYOUT_ID), {
        side: pose['base_yaw_rad'] for side, pose in poses.items()}


def body_paths(scene, task):
    recorded = task['reset_case']['actual_body_poses_m']
    require(isinstance(recorded, dict) and recorded, 'Missing frozen body poses')
    paths = {name: row['prim_path'] for name, row in recorded.items()}
    declared = {}
    for layer in [*scene.get('task_layers', []), scene.get('native_task_bundle', {})]:
        bodies = layer.get('rigid_bodies', {})
        require(not set(declared).intersection(bodies), 'Duplicate dynamic object ID')
        declared.update(bodies)
    for obj in scene.get('objects', []):
        require(obj['id'] not in declared, 'Duplicate dynamic object ID')
        declared[obj['id']] = object_actor_path(obj)
    require(paths == declared and len(set(paths.values())) == len(paths),
            'Frozen dynamic poses differ from actual scene declarations')
    geometry = task['reset_case']['static_geometry']['bodies']
    require(len(geometry) == len(paths) and {row['object_id'] for row in geometry} == set(paths),
            'Static geometry inventory differs from dynamic actors')
    return paths


def _catalog_reference_body(stage, root, asset):
    """CPU projection of usd_asset.placement_form/anchored placement, unchanged source ops."""
    from pxr import Usd, UsdGeom, UsdPhysics
    root.GetReferences().AddReference(asset['usd_path'], asset['usd_default_prim'])
    local = np.asarray(UsdGeom.Xformable(root).GetLocalTransformation())
    if root.HasAPI(UsdPhysics.RigidBodyAPI) and np.allclose(local, np.eye(4), atol=1e-9, rtol=0):
        return root
    root.GetReferences().ClearReferences()
    anchor_path = str(root.GetPath())+'/'+asset['usd_default_prim'].rstrip('/').rsplit('/', 1)[-1]
    anchor = UsdGeom.Xform.Define(stage, anchor_path).GetPrim()
    anchor.GetReferences().AddReference(asset['usd_path'], asset['usd_default_prim'])
    bodies = [p for p in Usd.PrimRange(anchor) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    require(len(bodies) == 1, 'Anchored source requires one original rigid body')
    body = bodies[0]
    relative, _ = UsdGeom.XformCache().ComputeRelativeTransform(body, root)
    matrix = np.asarray(relative); scales = np.linalg.norm(matrix[:3, :3], axis=1)
    require(np.all(scales > 0) and np.max(scales)-np.min(scales) <= 1e-6
            and np.max(abs(matrix[:3, :3]/scales[:, None]-np.eye(3))) <= 1e-6
            and np.linalg.norm(matrix[3, :3]) <= 1e-6,
            'Anchored body must retain original loader uniform scale and zero relative rotation/translation')
    return body


def catalog_body_path(spec):
    """Actual original catalog actor path, including the scaled fastener child."""
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    asset = load_task_asset(spec['asset_id'], asset_root=spec.get('asset_root'), verify_hashes=True)
    path = '/World/Objects/'+spec['id']
    if asset.get('backend') != 'usd_reference':
        return path
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory(); root = UsdGeom.Xform.Define(stage, path).GetPrim()
    return str(_catalog_reference_body(stage, root, asset).GetPath())


def verify_static_geometry(stage, paths, geometry, authored_mass=None):
    """Independently read enabled collision geometry and authored actor mass.

This is the source collider envelope, not cooked convex/SDF contact geometry.
Mesh vertices and analytic primitives are evaluated in the actual body frame;
visual extents, cached extent attributes and the published bounds are not used
to compute the readback. Native tensor mass/COM/inertia remain a separate gate.
"""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    rows = geometry.get('bodies', [])
    require(len(rows) == len(paths) and {r['object_id'] for r in rows} == set(paths),
            'Published geometry inventory differs from actual actors')
    expected = {r['object_id']: r for r in rows}
    masses = living.authored_mass_properties(stage, paths) if authored_mass is None else authored_mass
    cache = UsdGeom.XformCache(); result = {}
    for name, path in paths.items():
        body = stage.GetPrimAtPath(path)
        require(body and body.HasAPI(UsdPhysics.RigidBodyAPI)
                and UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is not False
                and UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get() is not True,
                'Expected an enabled dynamic actor: '+name)
        world = cache.GetLocalToWorldTransform(body); basis = np.asarray(world)[:3, :3]
        scales = np.linalg.norm(basis, axis=1)
        require(np.all(scales > 0) and np.linalg.det(basis) > 0 and np.max(scales)-np.min(scales) <= 2e-6
                and np.allclose((basis/scales[:, None])@(basis/scales[:, None]).T, np.eye(3), atol=2e-6, rtol=0),
                'Dynamic body has nonuniform scale or shear: '+name)
        # Native body poses carry rigid rotation/translation. Ancestor uniform
        # scale remains in the collision shapes (original fastener 1.5x anchor).
        rigid_world = np.array(world); rigid_world[:3, :3] /= scales[:, None]
        inverse = Gf.Matrix4d(rigid_world.tolist()).GetInverse(); bounds = []; colliders = []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            require(prim == body or not prim.HasAPI(UsdPhysics.RigidBodyAPI),
                    'Nested rigid body makes collider ownership ambiguous: '+name)
            if (not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
                continue
            matrix = cache.GetLocalToWorldTransform(prim)*inverse
            linear = np.asarray(matrix)[:3, :3].T
            center = np.asarray(matrix.ExtractTranslation())
            if prim.IsA(UsdGeom.Mesh):
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                require(points is not None and len(points) > 0, 'Collider mesh lacks source vertices')
                bounds.extend(np.asarray(matrix.Transform(Gf.Vec3d(*p))) for p in points)
            elif prim.IsA(UsdGeom.Cube):
                half = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
                bounds.extend(np.asarray(matrix.Transform(Gf.Vec3d(*p))) for p in product((-half, half), repeat=3))
            elif prim.IsA(UsdGeom.Sphere):
                extent = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())*np.linalg.norm(linear, axis=1)
                bounds.extend([center-extent, center+extent])
            elif prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
                shape = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(prim)
                axis = {'X': 0, 'Y': 1, 'Z': 2}[str(shape.GetAxisAttr().Get())]
                radial = linear if prim.IsA(UsdGeom.Capsule) else np.delete(linear, axis, axis=1)
                extent = (float(shape.GetRadiusAttr().Get())*np.linalg.norm(radial, axis=1)
                          + float(shape.GetHeightAttr().Get())/2*abs(linear[:, axis]))
                bounds.extend([center-extent, center+extent])
            else:
                raise ValueError('Unsupported actual collider readback: '+str(prim.GetPath()))
            colliders.append({'prim_path': str(prim.GetPath()), 'type': prim.GetTypeName()})
        points = np.asarray(bounds)
        require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(),
                'Missing/invalid actual collision envelope: '+name)
        actual = np.array([points.min(0), points.max(0)])
        nominal = np.asarray(expected[name]['local_bounds_m'], dtype=float)
        require(nominal.shape == (2, 3) and np.isfinite(nominal).all() and np.all(nominal[1] > nominal[0]),
                'Invalid nominal collision bounds: '+name)
        error = float(abs(actual-nominal).max())
        require(error <= 2e-6, f'Actual collision bounds differ from published geometry: {name}, max_error_m={error:.9g}')
        mass = masses[name]['mass_kg']; nominal_mass = expected[name]['nominal_mass_kg']
        require(type(nominal_mass) in (int, float) and math.isfinite(nominal_mass) and nominal_mass > 0
                and math.isclose(mass, nominal_mass, rel_tol=2e-6, abs_tol=1e-9),
                'Actual authored mass differs from published geometry: '+name)
        result[name] = {'prim_path': path, 'actual_local_bounds_m': actual.tolist(),
            'nominal_local_bounds_m': nominal.tolist(), 'maximum_bounds_error_m': error,
            'actual_authored_mass_kg': mass, 'nominal_mass_kg': nominal_mass,
            'mass_error_kg': abs(mass-nominal_mass), 'enabled_colliders': colliders}
        result[name]['source_body_uniform_scale'] = float(scales[0])
    return {'passed': True, 'bounds_tolerance_m': 2e-6, 'mass_relative_tolerance': 2e-6,
            'mass_absolute_tolerance_kg': 1e-9, 'bodies': result,
            'scope': 'Actual source collision envelope and authored mass, independently compared with public nominal geometry; not a cooked-SDF contact or manipulation test.'}


def verify_case_geometry(stage, scene, task, authored_mass=None):
    from pxr import Usd, UsdPhysics
    paths = body_paths(scene, task)
    namespaces = ['/World/Objects', '/World/Fixtures']
    namespaces += [x.get('target_prim_path', x['source_prim_path']) for x in scene.get('task_layers', [])]
    if scene.get('native_task_bundle'):
        namespaces.append(scene['native_task_bundle']['root_prim_path'])
    actual = set()
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not any(path == root or path.startswith(root+'/') for root in namespaces):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(prim)
            if api.GetRigidBodyEnabledAttr().Get() is not False and api.GetKinematicEnabledAttr().Get() is not True:
                actual.add(path)
    require(actual == set(paths.values()), 'Actual stage dynamic actors differ from frozen inventory')
    result = verify_static_geometry(stage, paths, task['reset_case']['static_geometry'], authored_mass)
    result['actual_dynamic_actor_paths'] = sorted(actual)
    return result


def _place(prim, spec):
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*spec['position_m']))
    q = spec.get('quaternion_wxyz', [1., 0., 0., 0.])
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))


def mount_catalog_geometry(stage, spec, dynamic, sources):
    """Read-only geometric projection of the original catalog loader.

Plain collision parts and the original root/anchored USD placement forms are
projected. Uses the original float32 mesh conditioning helper. It authors no
solver/material defaults and does not claim native cooking equivalence.
"""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
    from bench.simulation.isaacsim_backend.asset_catalog import (
        load_task_asset, load_obj_mesh, condition_collision_mesh, collision_mesh_coordinate_scale,
        inertia_principal_axes, prepare_triangle_mesh, validate_triangle_mesh_role,
    )
    asset = load_task_asset(spec['asset_id'], asset_root=spec.get('asset_root'), verify_hashes=True)
    validate_triangle_mesh_role(asset, dynamic=dynamic)
    if collision_mesh_coordinate_scale(spec) != 1. and any(
            p.get('geometry','convex_mesh') not in ('mesh','convex_mesh') for p in asset['collision_parts']):
        raise ValueError('Collision mesh conditioning requires explicit mesh collision parts')
    sources[asset['manifest_path']] = asset['manifest_sha256']
    sources.update(asset.get('source_dependencies_sha256', {}))
    path = '/World/'+('Objects/' if dynamic else 'Fixtures/')+spec['id']
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    if asset.get('backend') == 'usd_reference':
        body = _catalog_reference_body(stage, root, asset)
        api = UsdPhysics.RigidBodyAPI(body)
        api.CreateRigidBodyEnabledAttr(True); api.CreateKinematicEnabledAttr(not dynamic)
        sources[asset['usd_path']] = digest(Path(asset['usd_path']))
        _place(root, spec)
        return
    _place(root, spec)
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = UsdPhysics.MassAPI.Apply(root); mass.CreateMassAttr(float(asset['mass_kg']))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*asset['center_of_mass_m']))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*asset['inertia_diag_kg_m2']))
        q = inertia_principal_axes(asset); mass.CreatePrincipalAxesAttr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    for i, part in enumerate(asset['collision_parts']):
        child = path+f'/collision_{i:03d}'; kind = part.get('geometry', 'convex_mesh'); scale = None
        if kind in ('convex_mesh', 'mesh', 'triangle_mesh'):
            p = Path(part['mesh_path']); require(digest(p) == part['sha256'], 'Catalog mesh changed')
            sources[str(p)] = part['sha256']
            vertices, faces = load_obj_mesh(str(p))
            if kind == 'triangle_mesh':
                vertices, faces = prepare_triangle_mesh(vertices, faces, part.get('scale', [1, 1, 1]))
            else:
                vertices *= np.asarray(part.get('scale', [1, 1, 1]), dtype=np.float32)
                vertices, scale = condition_collision_mesh(vertices, collision_mesh_coordinate_scale(spec))
            shape = UsdGeom.Mesh.Define(stage, child)
            shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
            shape.CreateFaceVertexCountsAttr([3]*len(faces)); shape.CreateFaceVertexIndicesAttr(np.asarray(faces).ravel().tolist())
            shape.CreateSubdivisionSchemeAttr('none')
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('none' if kind == 'triangle_mesh' else 'convexHull')
        elif kind == 'box':
            shape = UsdGeom.Cube.Define(stage, child); shape.CreateSizeAttr(2.); scale = part['half_extents_m']
        elif kind == 'sphere':
            shape = UsdGeom.Sphere.Define(stage, child); shape.CreateRadiusAttr(float(part['radius_m']))
        elif kind == 'cylinder':
            shape = UsdGeom.Cylinder.Define(stage, child)
            shape.CreateRadiusAttr(float(part['radius_m'])); shape.CreateHeightAttr(2*part['half_height_m']); shape.CreateAxisAttr('Z')
        else:
            raise ValueError('Unsupported catalog collision projection: '+kind)
        q = part.get('quaternion_xyzw', [0, 0, 0, 1])
        _place(shape.GetPrim(), {'position_m': part.get('position_m', [0, 0, 0]), 'quaternion_wxyz': [q[3], *q[:3]]})
        if scale is not None:
            UsdGeom.Xformable(shape).AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)


def composed_support_stage(scene):
    """Immutable room, task layers/bundle, and catalog actors at native paths."""
    from pxr import Usd, UsdGeom
    from bench.simulation.isaacsim_backend.workroom import verify_reference
    sources = verify_reference(scene['workroom'])
    stage = Usd.Stage.CreateInMemory(); UsdGeom.Xform.Define(stage, '/World')
    root = UsdGeom.Xform.Define(stage, '/World/Room').GetPrim()
    root.GetReferences().AddReference(scene['workroom']['usd_path'], scene['workroom']['source_prim'])
    stage.OverridePrim('/World/Room/Robot').SetActive(False)
    layers = list(scene.get('task_layers', []))
    if scene.get('native_task_bundle'):
        bundle = scene['native_task_bundle']
        layers.append({'usd_path': bundle['usd_path'], 'sha256': bundle['usd_sha256'],
                       'source_prim_path': bundle['root_prim_path'], 'target_prim_path': bundle['root_prim_path']})
    for layer in layers:
        path = layer['usd_path']; expected = layer.get('sha256', layer.get('usd_sha256'))
        require(digest(Path(path)) == expected, 'Actual task USD changed')
        sources[path] = expected
        for dep, sha in layer.get('dependencies_sha256', {}).items():
            require(digest(Path(dep)) == sha, 'Actual task dependency changed'); sources[dep] = sha
        root = UsdGeom.Xform.Define(stage, layer.get('target_prim_path', layer['source_prim_path'])).GetPrim()
        root.GetReferences().AddReference(path, layer['source_prim_path'])
    for dynamic, values in ((True, scene.get('objects', [])), (False, scene.get('fixtures', []))):
        for spec in values:
            if 'stand' not in spec['id']:
                mount_catalog_geometry(stage, spec, dynamic, sources)
    return stage, sources


class CollisionBounds(living.RecordedCollisionBounds):
    """Original robot shapes plus fully composed final task collider bounds."""
    def __init__(self, scene, robot, models, initial, stands):
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics
        minimal = deepcopy(scene); minimal['objects'] = []; minimal['task_layers'] = []
        minimal.pop('native_task_bundle', None)
        minimal['fixtures'] = [x for x in scene.get('fixtures', []) if 'stand' in x['id']]
        super().__init__(minimal, robot, models, initial, stands)
        original_stage = self.stage
        self.stage, sources = composed_support_stage(scene); self.dependencies.update(sources)
        # ArmSelfCollision reads these original mesh prims, not only FK boxes.
        require(Sdf.CopySpec(original_stage.GetRootLayer(), '/Robots', self.stage.GetRootLayer(), '/Robots'),
                'Could not retain original robot collision references')
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
        transforms = UsdGeom.XformCache(); corners, names = [], []
        for prim in Usd.PrimRange(self.stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if str(prim.GetPath()).startswith('/Robots/'):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Boundable):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            if 'robot_stand' in str(prim.GetPath()).lower():
                continue
            bound = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
            points = np.array([(*p, 1.) for p in product(*zip(bound.GetMin(), bound.GetMax()))])
            world = np.asarray(transforms.GetLocalToWorldTransform(prim)).T
            points = (world@points.T).T[:, :3]
            require(np.isfinite(points).all(), 'Nonfinite actual collision shape')
            corners.append(points); names.append(str(prim.GetPath()))
        self.obstacle_corners = np.asarray(corners); self.obstacle_names = names
        self.obstacles = np.asarray([[p.min(0), p.max(0)] for p in corners])


def run_case(case):
    return living.run_case(case, adapter=sys.modules[__name__])


def augment_interaction_checks(report, **kwargs):
    from scripts.validation.workroom_interaction_endpoints import TASKS, augment
    if report['case']['task_id'] in TASKS:
        augment(report, **kwargs)


def source_identity():
    root = Path(__file__).resolve().parents[2]
    manifest = root/'source_snapshot_manifest.json'
    return {'source_manifest_sha256': digest(manifest) if manifest.is_file() else None,
        'validator_sources_sha256': {name: digest(root/name) for name in
            ('scripts/validation/workroom_randomized_workspace.py',
             'scripts/validation/living_randomized_workspace.py',
             'scripts/validation/workroom_interaction_endpoints.py',
             'scripts/validation/workroom_source_interference.py',
             'scripts/diagnostics/check_interaction_workspace.py')}}


def main():
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    root = Path(__file__).resolve().parents[2]
    required = ('scripts.validation.workroom_randomized_workspace',
                'scripts.validation.living_randomized_workspace', 'bench.isaacsim.release')
    first = verify_if_snapshot(root, required_modules=required, entrypoint_file=Path(__file__))
    require(first is not None and Path.cwd().resolve() == root, 'Use a complete frozen source root')
    try:
        return living.main(adapter=sys.modules[__name__])
    finally:
        require(verify_if_snapshot(root, required_modules=required, entrypoint_file=Path(__file__)) is not None,
                'Final frozen validator source verification missing')


if __name__ == '__main__':
    raise SystemExit(main())
