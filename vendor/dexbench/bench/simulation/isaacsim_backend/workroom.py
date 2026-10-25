"""Opt-in composition of immutable furnished rooms, without their parked robot.

OpenUSD is imported only by composition/preflight helpers. No SimulationApp,
physics stepping, source-asset edits or scene-authoring builder is used here.
"""
from __future__ import annotations

import hashlib
from fnmatch import fnmatchcase
from pathlib import Path

from bench.paths import load_json

SCHEMA = 'isaacsim-workroom-reference-v1'
ROOM_PATH = '/World/Room'
SUPPORT_PATHS = {
    'table': ROOM_PATH + '/Geometry/worktop/Collider',
    'floor': ROOM_PATH + '/Geometry/room_floor/Collider',
}
BUILTIN_MDL = {'OmniGlass.mdl', 'OmniPBR.mdl'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def checked(path, expected):
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f'Workroom dependency must be an existing absolute file: {path}')
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Workroom dependency hash mismatch: {path}')
    return actual


def verify_reference(spec):
    """Check the immutable source and its published direct dependency inventory."""
    if spec.get('schema') != SCHEMA or spec.get('source_prim') != '/World':
        raise ValueError('Workroom requires the v1 reference schema and /World source prim')
    hashes = {str(spec['usd_path']): checked(spec['usd_path'], spec['usd_sha256'])}
    manifest_path = str(spec['dependency_manifest_path'])
    hashes[manifest_path] = checked(manifest_path, spec['dependency_manifest_sha256'])
    manifest = load_json(Path(manifest_path))
    if manifest.get('family') != spec['family']:
        raise ValueError('Workroom family differs from the pinned dependency inventory')
    for name, record in manifest['dependencies'].items():
        hashes[name] = checked(name, record['sha256'])
    return hashes


def add_workroom(stage, spec, *, surface_materials=None):
    """Reference only /World; keep bench-owned physics and robot outside it."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from .workroom_materials import apply_materials, material_options
    material_options(surface_materials)
    hashes = verify_reference(spec)
    source = Usd.Stage.Open(spec['usd_path'])
    if (UsdGeom.GetStageMetersPerUnit(source) != 1.0
            or UsdGeom.GetStageUpAxis(source) != UsdGeom.Tokens.z):
        raise ValueError('Workroom source must be metre-scale and Z-up')
    if not source.GetPrimAtPath('/World/Robot'):
        raise ValueError('Workroom source lacks the expected parked Robot')
    if stage.GetPrimAtPath(ROOM_PATH):
        raise ValueError(f'Workroom destination already exists: {ROOM_PATH}')
    root = UsdGeom.Xform.Define(stage, ROOM_PATH).GetPrim()
    root.GetReferences().AddReference(str(spec['usd_path']), '/World')
    # A local stronger opinion: the published USD and its robot stay immutable.
    stage.OverridePrim(ROOM_PATH + '/Robot').SetActive(False)
    selection = spec.get('geometry_selection')
    selected = None
    if selection is not None:
        if not isinstance(selection, list) or not selection or any(
                not isinstance(pattern, str) or '/' in pattern for pattern in selection):
            raise ValueError('geometry_selection must contain Geometry child-name patterns')
        children = list(stage.GetPrimAtPath(ROOM_PATH + '/Geometry').GetChildren())
        kept, removed = [], []
        for child in children:
            if any(fnmatchcase(child.GetName(), pattern) for pattern in selection):
                kept.append(str(child.GetPath()))
            else:
                removed.append(str(child.GetPath()))
                child.SetActive(False)
        selected = {'kept': kept, 'removed': removed}
    solids = []
    primitive_filters = []
    collider_types = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        # Imported SimReady decorations retain disabled joint/body schemas.
        # Their authored enabled flags, rather than schema presence, determine
        # whether they would participate in native dynamics.
        rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False
        joint = prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False
        articulation = (prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                        and prim.GetAttribute('physxArticulation:articulationEnabled').Get() is not False)
        if prim.IsA(UsdPhysics.Scene) or rigid or joint or articulation:
            raise ValueError(f'Workroom environment contains active dynamics: {prim.GetPath()}')
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                path = str(prim.GetPath())
                solids.append(path)
                collider_types[path] = prim.GetTypeName()
                # Static triangle meshes still collide physically, but cannot
                # serve as filters in the GPU tensor contact view. Do not
                # silently claim complete scenery contact observability.
                if prim.GetTypeName() in {'Cube', 'Sphere', 'Capsule', 'Cylinder', 'Cone'}:
                    primitive_filters.append(path)
    for name, path in SUPPORT_PATHS.items():
        if path not in solids:
            raise ValueError(f'Workroom lacks its {name} collider: {path}')
    material_record = apply_materials(stage, SUPPORT_PATHS, surface_materials)
    return {**({'surface_materials': material_record} if material_record is not None else {}),
            'kind': 'referenced_workroom', 'family': spec['family'], 'path': ROOM_PATH,
            'source': dict(spec), 'source_file_hashes': hashes,
            'geometry_selection': selected,
            'disabled_source_robot': ROOM_PATH + '/Robot',
            'solid_paths': sorted(solids), 'supports': dict(SUPPORT_PATHS),
            'collider_types': collider_types, 'contact_filter_paths': sorted(primitive_filters),
            'contact_unobserved_paths': sorted(set(solids) - set(primitive_filters)),
            'contact_scope': 'Primitive static filters only; excluded mesh colliders remain physically enabled.',
            'scope': 'Static scenery; dynamic diagnostics are separately created bench objects.'}


def apply_runtime_lighting(stage, spec, config=None):
    """Opt-in light overrides on the referenced runtime layer, never the asset."""
    if config is None:
        return None
    if (not isinstance(config, dict) or set(config) != {'schema', 'family', 'profile'}
            or config['schema'] != 'workroom-lighting-v1'):
        raise ValueError('workroom_lighting requires schema workroom-lighting-v1, family and profile')
    if config['family'] != spec['family']:
        raise ValueError('workroom_lighting family differs from its referenced room')
    from bench.simulation.assets.workrooms.common import apply_lighting_profile
    return apply_lighting_profile(stage, config['family'], config['profile'], light_root=ROOM_PATH + '/Lights')


def preflight(spec, diagnostics=None, *, surface_materials=None):
    """CPU USD composition, recursive dependency and diagnostic clearance checks.

This establishes no native physics or manipulation acceptance. Built-in MDL
names are left to the local Isaac Sim installation to resolve.
"""
    from pxr import Usd, UsdGeom, UsdUtils

    stage = Usd.Stage.CreateInMemory()
    record = add_workroom(stage, spec, surface_materials=surface_materials)
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(spec['usd_path']))
    paths = {Path(layer.realPath) for layer in layers if layer.realPath}
    paths.update(Path(path) for path in assets)
    builtins = {}
    for name in unresolved:
        if name not in BUILTIN_MDL:
            raise ValueError(f'Unresolved workroom dependency: {name}')
        builtins[name] = name
    for path in paths:
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f'Missing workroom dependency: {path}')
    # Include invisible physics shapes; visual bounds alone miss the colliders.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'], False, True)
    boxes = {}
    for path in record['solid_paths']:
        bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise ValueError(f'Workroom collider has no bounds: {path}')
        boxes[path] = (list(bounds.GetMin()), list(bounds.GetMax()))
    for name, probe in (diagnostics or {}).items():
        x, y, z = probe['initial_position_m']
        hx, hy, hz = probe['half_extents_m']
        support = probe['expected_support_path']
        lo, hi = boxes[support]
        top = float(probe['support_top_z_m'])
        if (abs(hi[2] - top) > 1e-5 or x - hx < lo[0] or x + hx > hi[0]
                or y - hy < lo[1] or y + hy > hi[1] or z - hz <= top):
            raise ValueError(f'Diagnostic {name} does not start above its declared support')
        for path, (low, high) in boxes.items():
            if path == support:
                continue
            if (low[0] < x + hx and high[0] > x - hx
                    and low[1] < y + hy and high[1] > y - hy
                    and low[2] < z + hz and high[2] > top + 1e-5):
                raise ValueError(f'Diagnostic {name} fall path intersects {path}')
    return {**record, 'status': 'cpu_preflight_passed', 'physics_accepted': False,
            'dependency_file_hashes': {str(path): digest(path) for path in sorted(paths)},
            'runtime_builtin_dependencies': builtins,
            'diagnostic_clearance_checked': sorted(diagnostics or {})}
