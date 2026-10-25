"""Explicit source-preserving L3 collision candidates; no reset or rule changes."""
from copy import deepcopy
import hashlib
from pathlib import Path

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset

DENSE_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/'
                  'shaft_spacer_assembly/dense1024_v1')
DENSE_MANIFEST_SHA256 = 'b1a0156db9ed6b0d7337e67da9a97c18c7cf08f92a5fc66c993e9ee4ff3493f0'
PHYSICAL_FIELDS = ('mass_kg', 'center_of_mass_m', 'inertia_diag_kg_m2',
                   'material_prior', 'contact_offset_m', 'rest_offset_m',
                   'geometry_contract', 'geometry_features')


def validate_selection(candidate, solver_type):
    if candidate not in (None, 'dense1024_v1'):
        raise ValueError('Unknown shaft ring collision candidate')
    if candidate is not None and solver_type != 'PGS':
        raise ValueError('The dense1024_v1 candidate requires explicit solver_type=PGS')


def bind_dense_rings(scene):
    """Bind exactly the tested catalog while preserving canonical supply poses."""
    manifest = DENSE_ROOT/'manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != DENSE_MANIFEST_SHA256:
        raise ValueError('Pinned dense ring manifest changed')
    names = ('spacer', 'wide_ring', 'top_ring')
    if tuple(row['id'] for row in scene['objects']) != names:
        raise ValueError('Dense collision binding requires the three canonical rings')
    for spec in scene['objects']:
        original = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        asset_id = 'mechanical_l3_'+spec['id']+'_sdf1024'
        candidate = load_task_asset(asset_id, DENSE_ROOT, verify_hashes=True)
        source_hash = candidate['source_dependencies_sha256'].get(original['manifest_path'])
        if source_hash != original['manifest_sha256']:
            raise ValueError('Dense rings require their exact source asset catalog')
        if any(original[key] != candidate[key] for key in PHYSICAL_FIELDS):
            raise ValueError('Dense ring physical fields differ from source')
        if (candidate.get('backend') != 'usd_reference'
                or candidate.get('source_scale') != [1., 1., 1.]
                or candidate.get('placement_form') != 'root'
                or candidate.get('source_modified') is not False):
            raise ValueError('Dense ring authored geometry contract differs')
        material = candidate['material_prior']
        fallback = {key: deepcopy(material[key]) for key in
                    ('static_friction', 'dynamic_friction', 'restitution', 'friction_combine')}
        if (spec['contact_offset_m'] != candidate['contact_offset_m']
                or spec['rest_offset_m'] != candidate['rest_offset_m']
                or spec['friction'] != fallback['static_friction']
                or spec['friction'] != fallback['dynamic_friction']):
            raise ValueError('Dense ring contact parameters differ from source')
        spec.pop('friction')
        spec.pop('collision_mesh_coordinate_scale')
        spec.update(asset_id=asset_id, asset_root=str(DENSE_ROOT),
                    usd_physics_policy='preserve_authored_v1',
                    usd_physics_material_fallback=fallback)
    scene['mechanical_task_provenance']['ring_collision_candidate'] = {
        'id': 'dense1024_v1', 'manifest_path': str(manifest),
        'manifest_sha256': DENSE_MANIFEST_SHA256,
        'original_source_geometry_modified': False,
        'original_task_reset_modified': False,
        'task_policy_success_claimed': False,
        'scope': 'Explicit dense SDF1024 collision representation with the original '
                 'supply reset and rules. Native scene evidence is attached to the '
                 'frozen release separately; no robot manipulation qualification.',
    }
