"""Derive a source-preserving L1 ring candidate from a frozen sorting probe."""
from copy import deepcopy
from pathlib import Path

from bench.isaacsim.release import digest
from bench.paths import load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scripts.assets.prepare_mechanical_ring_sdf import prepare


def prepare_inputs(candidate_root, source_input):
    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.release import _asset_inventory
    source_input = Path(source_input).resolve()
    original = load_json(source_input)
    if (original['task']['id'] != 'parts_sorting'
            or [case['id'] for case in original['cases']] != ['nominal_supply','centered','offset','wrong_bins']
            or original['scene']['isaacsim'].get('solver_type','TGS') != 'TGS'
            or 'diagnostic_solver_type' in original):
        raise ValueError('The original four-case TGS sorting probe is required')
    spec = next(s for s in original['scene']['objects'] if s['id']=='ring')
    material = {'static_friction':spec['friction'],'dynamic_friction':spec['friction'],
        'restitution':0.,'friction_combine':'average'}
    # These are the actual plain-asset scene.rigid_material and collision
    # settings, made explicit because the older L1 catalog omits those fields.
    physics = {'material_prior':material,'contact_offset_m':spec['contact_offset_m'],
        'rest_offset_m':spec['rest_offset_m']}
    source = load_task_asset(spec['asset_id'],spec['asset_root'],verify_hashes=True)
    authored = prepare(candidate_root,spec['asset_root'],spec['asset_id'],physics_fallback=physics)
    value = deepcopy(original)
    ring = next(s for s in value['scene']['objects'] if s['id']=='ring')
    ring.pop('friction')
    ring.pop('collision_mesh_coordinate_scale')
    ring.update(asset_id=spec['asset_id']+'_sdf1024',asset_root=authored['asset_root'],
        usd_physics_policy='preserve_authored_v1',usd_physics_material_fallback=material)
    value['native_sdf_source_meshes'] = {'ring':source['mesh_path']}
    value['quantitative_weight_balance_required'] = True
    value['scene']['engineering_contact_force_crosscheck'] = True
    value['sdf_candidate_provenance'] = {'source_input':str(source_input),
        'source_input_sha256':digest(source_input),'candidate_manifest_sha256':authored['manifest_sha256'],
        'source_geometry_modified':False,'cases_and_task_rules_unchanged':True,
        'scope':'Same TGS scene, all four original gravity cases, one full-surface SDF ring. No task score or catalog activation.'}
    value['asset_files_sha256'] = {p:r['sha256'] for p,r in _asset_inventory(value['scene']).items()}
    value['asset_files_sha256'].update(original['asset_files_sha256'])
    value['asset_files_sha256'][str(source_input)] = digest(source_input)
    return value
