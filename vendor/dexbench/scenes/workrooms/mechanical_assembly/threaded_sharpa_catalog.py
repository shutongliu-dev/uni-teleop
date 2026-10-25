"""Reference the user-selected SHARPA v3 bundle and its recorded contact test.

Catalog construction is CPU-only and never rebuilds, copies or edits a USD.
Historical external-wrench acceptance does not qualify robot manipulation.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from .threaded_assets import ASSEMBLY_ROOT
from .threaded_catalog import _digest, _inspect_asset
from bench.paths import DATA_ROOT, load_json


SHARPA_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/threaded_sharpa/v3')
EVIDENCE_ROOT = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/threaded_asset_native_20261016')
SOURCE_SHA256 = '0c7762d73479438c7609c551dd26458cdbd5ba68ba103f1dd3f92a6b7113532f'
ASSET_IDS = {'screwdriver': 'mechanical_screwdriver_sharpa_v3',
             'bolt': 'mechanical_bolt_sharpa_v3', 'fixture': 'mechanical_threaded_fixture_sharpa_v3'}
BINDINGS = {
    'screwdriver': ('straight_hex_screwdriver_sharpa.usd', '/straight_hex_screwdriver_sharpa',
                   'ef8b809c9cbe6410c7b8a9eaaf87633a7225f1081bd05bba671460eabaf84f54', 1.),
    'bolt': ('sharpa_threaded_bolt.usda', '/sharpa_threaded_bolt',
             'e98784eae93da6bf848f4d0080e4de0d4aab90bfe1971b73f7e904a32beaec46', 1.5),
    'fixture': ('sharpa_threaded_fixture.usda', '/sharpa_threaded_fixture',
                'f7f8a957a8d9d9f4ac4f3afbda55dbe634f9604cfcaeb33b6a05c4538e7da789', 1.5),
}
EVIDENCE_FILES = {
    'screw_contact_acceptance.json': '726025545af41c53f08ef62493ac7baa7721530540c0f9e2b08a0a8c25c8b3aa',
    'runs/contact_validation_06/snapshot/test_threaded_dynamics.py': '6eb4d8242df83a4143565a01e6e1c1bc9508a6cd0cf43a66e1ba59bf07682b51',
    'runs/contact_validation_06/snapshot/config.json': '28b26ac96fc93bb5a111b7e6137ef771b0b2181ace146576f9f86d1c8212aaf5',
    'runs/contact_validation_06/result/report.json': 'f0c47f21c2e296f09c5723b1926533e10df3732c632f1b30568a40221ba4d136',
    'runs/contact_convergence_07/result/report.json': '049bf1bebe8ef58e42285ca5669f6bc4e55e79a11c6d9e1e25da84cc689e3d8d',
}


def threaded_sharpa_asset_bindings():
    """Import-safe exact references for planned tasks; no simulator imports."""
    return {role: {'path': str(SHARPA_ROOT / filename), 'default_prim': prim,
                   'sha256': digest, 'source_scale': [scale] * 3,
                   'selection': 'User-selected existing threaded_sharpa/v3; no additional scaling.',
                   'qualification': 'Recorded scripted contact dynamics only; robot manipulation remains unqualified.'}
            for role, (filename, prim, digest, scale) in BINDINGS.items()}


def _checked(path, expected):
    path = Path(path)
    if _digest(path) != expected:
        raise ValueError('Pinned threaded SHARPA provenance changed: ' + str(path))
    return str(path), expected


def _source_chain(root):
    """Validate every version's file inventory and its immutable parent chain."""
    dependencies = {}
    expected = SOURCE_SHA256
    current = Path(root)
    while True:
        path, digest = _checked(current / 'SOURCE.json', expected)
        dependencies[path] = digest
        source = load_json(Path(path))
        for row in source['files']:
            p, sha = _checked(current / row['path'], row['sha256'])
            dependencies[p] = sha
        parents = current / 'PARENT_DEPENDENCIES.json'
        if parents.exists():
            for p, sha in load_json(parents).items():
                dependencies.update([_checked(p, sha)])
        parent = source.get('parent_bundle')
        if parent is None:
            break
        current, expected = Path(parent['path']), parent['source_manifest_sha256']
    upstream = load_json(Path(root) / 'SOURCE.json')['upstream_manifest']
    dependencies.update([_checked(upstream['path'], upstream['sha256'])])
    upstream_root = Path(upstream['path']).parent
    original = load_json(Path(upstream['path']))
    for row in original['files']:
        dependencies.update([_checked(upstream_root / row['path'], row['sha256'])])
    return dependencies


def _profile(evidence_root):
    evidence = {str(Path(evidence_root) / name): sha for name, sha in EVIDENCE_FILES.items()}
    for path, sha in evidence.items():
        _checked(path, sha)
    accepted = load_json(Path(evidence_root) / 'screw_contact_acceptance.json')
    if accepted['contact_dynamics_accepted'] is not True or accepted['robot_skill_acceptance'] is not None:
        raise ValueError('Unexpected historical acceptance scope')
    run = load_json(Path(evidence_root) / 'runs/contact_validation_06/result/report.json')
    if run['robot_present'] or run['policy_calls'] or run['post_reset_state_writes']:
        raise ValueError('Historical contact harness scope changed')
    for role, key in [('screwdriver', 'driver_usd'), ('bolt', 'bolt_usd'), ('fixture', 'fixture_usd')]:
        if run['input_sha256'][key] != BINDINGS[role][2]:
            raise ValueError('Historical acceptance tested a different USD')
    cfg = run['config']
    expected_cfg = {'scale': 1.5, 'contact_offset': .0003, 'velocity_iterations': 16,
                    'max_depenetration_velocity_m_s': .05, 'dt': 1 / 480}
    if any(cfg.get(k) != value for k, value in expected_cfg.items()):
        raise ValueError('Historical contact profile differs from its pinned configuration')
    provenance = {'report_path': str(Path(evidence_root) / 'screw_contact_acceptance.json'),
                  'report_sha256': EVIDENCE_FILES['screw_contact_acceptance.json'],
                  'frozen_script_path': str(Path(evidence_root) / 'runs/contact_validation_06/snapshot/test_threaded_dynamics.py'),
                  'frozen_script_sha256': EVIDENCE_FILES['runs/contact_validation_06/snapshot/test_threaded_dynamics.py'],
                  'scope': 'Historical v3 bounded external tool-wrench tests only. No robot grasp; new scene, placement and timestep require new acceptance.',
                  'historical_contact_validated': True, 'current_scene_validated': False}
    per_role = {}
    for role, friction in [('screwdriver', .6), ('bolt', .01), ('fixture', .75)]:
        per_role[role] = {
            'usd_physics_policy': 'preserve_authored_v1',
            'usd_physics_fallbacks': {'contact_offset_m': .0003, 'rest_offset_m': 0.,
                'solver_position_iterations': 192, 'solver_velocity_iterations': 16,
                'max_depenetration_velocity_m_s': .05, 'linear_damping': 0.,
                'angular_damping': 0., 'sleep_threshold': 0., 'enable_speculative_ccd': False},
            'usd_physics_material_fallback': {'static_friction': friction, 'dynamic_friction': friction,
                                            'restitution': 0., 'friction_combine': 'average'},
            'usd_physics_fallbacks_provenance': deepcopy(provenance),
        }
    return per_role, evidence, run, provenance


def prepare_threaded_sharpa_catalog(output, *, source_root=SHARPA_ROOT, evidence_root=EVIDENCE_ROOT):
    """Create a NEW reference-only catalog; return per-role backend options."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new mechanical catalog version: ' + str(output))
    dependencies = _source_chain(source_root)
    options, evidence, run, acceptance = _profile(evidence_root)
    dependencies.update(evidence)
    bindings = threaded_sharpa_asset_bindings()
    assets, audits = {}, {}
    expected_shapes = {'screwdriver': ['convexHull'] * 3,
                       'bolt': ['sdf'] + ['convexHull'] * 7,
                       'fixture': ['sdf', 'convexHull', 'convexHull', 'sdf']}
    for role, binding in bindings.items():
        binding['path'] = str(Path(source_root) / BINDINGS[role][0])
        info, audit = _inspect_asset(binding, role=role, provenance_dependencies=dependencies,
                                     source_scale=BINDINGS[role][3])
        if [p['approximation'] for p in info['collision_parts']] != expected_shapes[role]:
            raise ValueError('v3 active collider contract differs: ' + role)
        for collider in audit['colliders']:
            authored = collider['authored_parameters']
            if abs(authored['physxCollision:contactOffset'] - .0003) > 1.e-9 or authored['physxCollision:restOffset'] != 0:
                raise ValueError('Source offsets conflict with historical contact profile')
            material = collider['physics_material']
            if material['has_physics_material_api']:
                values = material['authored_parameters']
                target = options[role]['usd_physics_material_fallback']
                if any(abs(values[key] - target[name]) > 1.e-6 for key, name in [
                    ('physics:staticFriction', 'static_friction'), ('physics:dynamicFriction', 'dynamic_friction'),
                    ('physics:restitution', 'restitution')]):
                    raise ValueError('Source material conflicts with historical contact profile')
        info['physics_assumptions'].update(
            friction='Pinned historical profile: source driver 0.6; absent bolt/fixture materials use explicit 0.01/0.75. No global 0.65 fallback.',
            placement='Existing v3 transforms retained: bolt/fixture already scale 1.5, driver uses its original v3 geometry. Never apply another scale.',
            fixture='Task fixture remains fixed kinematic; original v3 plate and thread insert SDF retained.')
        info['qualification'] = acceptance['scope']
        info['recommended_spec_options'] = deepcopy(options[role])
        historic_role = 'driver' if role == 'screwdriver' else role
        audit['historical_native_properties'] = deepcopy(run['native_properties'][0][historic_role])
        assets[ASSET_IDS[role]], audits[role] = info, audit
    report = {'schema': 'mechanical_threaded_sharpa_catalog_source.v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'generator': {'path': str(Path(__file__).resolve()), 'sha256': _digest(__file__)},
              'source_root': str(source_root), 'source_manifest_sha256': SOURCE_SHA256,
              'asset_ids': deepcopy(ASSET_IDS), 'assets': audits,
              'per_role_spec_options': options, 'historical_acceptance': acceptance,
              'evidence_sha256': evidence, 'source_assets_modified': False,
              'current_scene_native_validated': False, 'robot_manipulation_validated': False,
              'licenses': {'driver': {'path': str(SHARPA_ROOT.parent / 'v1/DRIVER_LICENSE.txt'), 'terms': 'CC0-1.0'},
                  'assembly': {'license': str(ASSEMBLY_ROOT / 'provenance/huggingface/LICENSE'),
                               'notice': str(ASSEMBLY_ROOT / 'provenance/huggingface/NOTICE'),
                               'terms': 'Separate upstream authored-part and NVIDIA terms; not relicensed as CC0.'}}}
    output.mkdir(parents=True, exist_ok=False)
    source_path = output / 'SOURCE.json'
    source_path.write_text(json.dumps(report, indent=2) + '\n')
    for info in assets.values():
        info['source_dependencies_sha256'][str(source_path)] = _digest(source_path)
    manifest = {'schema': 'collaborator_native_usd_catalog.v1', 'assets': assets,
                'source_provenance': str(source_path), 'source_provenance_sha256': _digest(source_path),
                'per_role_spec_options': deepcopy(options), 'historical_acceptance': acceptance,
                'source_assets_modified': False, 'current_scene_native_validated': False,
                'generator': report['generator']}
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'SHA256SUMS').write_text(''.join(f'{_digest(p)}  {p.name}\n' for p in (source_path, manifest_path)))
    return {'manifest_path': str(manifest_path), 'manifest_sha256': _digest(manifest_path),
            'source_audit_path': str(source_path), 'source_audit_sha256': _digest(source_path),
            'asset_ids': deepcopy(ASSET_IDS), 'per_role_spec_options': deepcopy(options),
            'source_assets_modified': False, 'current_scene_native_validated': False}
