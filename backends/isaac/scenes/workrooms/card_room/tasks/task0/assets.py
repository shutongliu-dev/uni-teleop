"""Explicit, hash-bound token collision candidates for task0 composition.

This binds declared invariants only; native geometry/contact evidence is separate.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path


def validate_token_asset_selection(selection):
    if (not isinstance(selection, dict)
            or set(selection) != {'asset_root', 'asset_id', 'manifest_sha256'}
            or not all(type(v) is str and v for v in selection.values())
            or not Path(selection['asset_root']).is_absolute()
            or len(selection['manifest_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in selection['manifest_sha256'])):
        raise ValueError('Token collision asset requires an absolute root, id and SHA256')


def pinned_usd_dependency(asset):
    """Require the referenced USD itself in the verified dependency inventory."""
    name = asset.get('usd_path')
    if not isinstance(name, str) or not Path(name).is_absolute() or not Path(name).is_file():
        raise ValueError('Token USD reference requires an existing absolute path')
    path = Path(name).resolve()
    dependencies = asset.get('source_dependencies_sha256', {})
    expected = dependencies.get(name)
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Token USD reference must have a matching dependency hash')
    return {str(path): expected}


def bind_token_collision_asset(scene, selection):
    """Hash-bound, reset-only candidate composition for the three task0 tokens.

    Only the collision representation may change. Geometry fidelity and native
    material/shape readback require independent validation; these manifest
    checks deliberately do not claim native equivalence.
    """
    validate_token_asset_selection(selection)
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    path = Path(selection['asset_root'])/'manifest.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != selection['manifest_sha256']:
        raise ValueError('Candidate token manifest hash mismatch')
    asset = load_task_asset(selection['asset_id'], selection['asset_root'], verify_hashes=True)
    if asset.get('backend') != 'usd_reference' or asset.get('placement_form') != 'root':
        raise ValueError('Candidate token requires a root-body USD reference')
    pinned_usd_dependency(asset)
    objects = scene.get('objects', [])
    if len(objects) != 3 or {s['id'] for s in objects} != {'black_00', 'black_01', 'black_02'}:
        raise ValueError('Token diagnostic requires exactly the three task0 objects')
    reports = []
    for spec in objects:
        if spec['asset_id'] != 'token_black' or 'usd_physics_policy' in spec:
            raise ValueError('Token diagnostic requires the original plain black token')
        original = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        unchanged = ('mass_kg', 'center_of_mass_m', 'inertia_diag_kg_m2')
        if any(asset.get(k) != original.get(k) or k not in asset for k in unchanged):
            raise ValueError('Candidate changes original token mass, inertia or dimensions')
        parts = original['collision_parts']
        if len(parts) != 2 or any(p['geometry'] != 'cylinder'
                or p.get('scale', [1, 1, 1]) != [1, 1, 1]
                or p['quaternion_xyzw'] != [0., 0., 0., 1.] for p in parts):
            raise ValueError('Token diagnostic requires original unscaled Z cylinders')
        extents = [[p['radius_m'], p['radius_m'], p['half_height_m']] for p in parts]
        lower = [min(p['position_m'][i]-e[i] for p, e in zip(parts, extents)) for i in range(3)]
        upper = [max(p['position_m'][i]+e[i] for p, e in zip(parts, extents)) for i in range(3)]
        bounds = {'bounds_min_m': lower, 'bounds_max_m': upper,
                  'dimensions_m': [hi-lo for lo, hi in zip(lower, upper)]}
        if any(not isinstance(asset.get(k), list) or len(asset[k]) != 3
                or any(not math.isclose(a, b, rel_tol=0., abs_tol=1e-12)
                       for a, b in zip(asset[k], values)) for k, values in bounds.items()):
            raise ValueError('Candidate changes original token dimensions')
        if asset.get('principal_axes_xyzw', [0., 0., 0., 1.]) != original.get('principal_axes_xyzw', [0., 0., 0., 1.]):
            raise ValueError('Candidate changes original token inertia axes')
        friction = spec.get('friction', .65)
        expected_material = {'static_friction': friction, 'dynamic_friction': friction,
                             'restitution': 0., 'friction_combine': 'average'}
        if (asset.get('material_prior') != expected_material
                or asset.get('contact_offset_m') != spec.get('contact_offset_m', .001)
                or asset.get('rest_offset_m') != spec.get('rest_offset_m', 0.)):
            raise ValueError('Candidate changes original token material or offsets')
        before = deepcopy(spec)
        spec.update(asset_root=str(path.parent.resolve()), asset_id=selection['asset_id'],
                    usd_physics_policy='preserve_authored_v1')
        spec.pop('friction', None)  # Same value is explicitly authored in the candidate.
        reports.append({'path': 'objects.'+spec['id'], 'before': before, 'after': deepcopy(spec),
                        'changed': True, 'original_manifest_sha256': original['manifest_sha256'],
                        'candidate_manifest_sha256': asset['manifest_sha256'],
                        'manifest_properties_unchanged': {k: asset[k] for k in unchanged},
                        'original_cylinder_bounds_m': bounds,
                        'material_prior_unchanged': expected_material,
                        'scope': 'Manifest/dependency hashes and declared invariants; not native geometry certification.'})
    return reports


def validate_board_asset_selection(selection):
    try:
        validate_token_asset_selection(selection)
    except ValueError as exc:
        raise ValueError('Board collision asset requires an absolute root, id and SHA256') from exc


def bind_board_collision_asset(scene, selection):
    """Bind a pinned exact-surface board candidate without changing task semantics."""
    validate_board_asset_selection(selection)
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    path = Path(selection['asset_root'])/'manifest.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != selection['manifest_sha256']:
        raise ValueError('Candidate board manifest hash mismatch')
    asset = load_task_asset(selection['asset_id'], selection['asset_root'], verify_hashes=True)
    fixtures = [row for row in scene.get('fixtures', []) if row.get('id') == 'board']
    if len(fixtures) != 1 or fixtures[0].get('asset_id') != 'board':
        raise ValueError('Board candidate requires the original single board fixture')
    if scene.get('isaacsim', {}).get('static_mesh_contact_mode') != 'kinematic_mesh_v1':
        raise ValueError('Board triangle candidate requires existing fixed-mesh contact observability')
    spec = fixtures[0]
    if 'usd_physics_policy' in spec or asset.get('backend') == 'usd_reference':
        raise ValueError('Board triangle candidate requires the plain fixture loader')
    original = load_task_asset('board', spec['asset_root'], verify_hashes=True)
    original_parts = original['collision_parts']
    expected_box = {'geometry': 'box', 'position_m': [0, 0, .006], 'half_extents_m': [.23, .23, .006],
                    'quaternion_xyzw': [0., 0., 0., 1.], 'scale': [1, 1, 1]}
    if len(original_parts) != 1 or any(original_parts[0].get(k) != v for k, v in expected_box.items()):
        raise ValueError('Board source collision dimensions changed')
    unchanged = ('dynamic', 'mass_kg', 'center_of_mass_m', 'inertia_diag_kg_m2', 'inertia_method',
                 'origin', 'mesh_path', 'sha256', 'visual_parts')
    if any(asset.get(k) != original.get(k) for k in unchanged) or asset.get('dynamic') is not False:
        raise ValueError('Board candidate changes original fixture physics or visual geometry')
    bounds = {'bounds_min_m': [-.23, -.23, 0.], 'bounds_max_m': [.23, .23, .012],
              'dimensions_m': [.46, .46, .012]}
    if any(asset.get(k) != value for k, value in bounds.items()):
        raise ValueError('Board candidate changes original box exterior')
    parts = asset['collision_parts']
    if len(parts) != 1 or any(parts[0].get(k) != v for k, v in {
            'geometry': 'triangle_mesh', 'position_m': [0., 0., 0.],
            'quaternion_xyzw': [0., 0., 0., 1.], 'scale': [1, 1, 1]}.items()):
        raise ValueError('Board candidate requires one unscaled closed triangle surface')
    friction = spec.get('friction', .4)
    expected_material = {'static_friction': friction, 'dynamic_friction': friction,
                         'restitution': 0., 'friction_combine': 'average'}
    if (asset.get('material_prior') != expected_material
            or asset.get('contact_offset_m') != spec.get('contact_offset_m', .001)
            or asset.get('rest_offset_m') != spec.get('rest_offset_m', 0.)):
        raise ValueError('Board candidate changes original material or offsets')
    dependencies = asset.get('source_dependencies_sha256', {})
    if dependencies.get(original['manifest_path']) != original['manifest_sha256']:
        raise ValueError('Board candidate must pin its original source manifest')
    if dependencies.get(parts[0]['mesh_path']) != parts[0]['sha256']:
        raise ValueError('Board triangle surface must have an explicit dependency hash')
    for name, expected in dependencies.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Board source dependency hash mismatch: '+name)
    before = deepcopy(spec)
    spec.update(asset_root=str(path.parent.resolve()), asset_id=selection['asset_id'])
    return [{'path': 'fixtures.board', 'before': before, 'after': deepcopy(spec), 'changed': True,
             'original_manifest_sha256': original['manifest_sha256'],
             'candidate_manifest_sha256': asset['manifest_sha256'],
             'original_box_bounds_m': bounds, 'material_prior_unchanged': expected_material,
             'contact_actor_path_unchanged': '/World/Fixtures/board/collision_000',
             'scope': 'Declared source surface and dependency invariants; native cooked mesh/contact validation is separate.'}]
