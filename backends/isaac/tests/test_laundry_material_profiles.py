"""New physical thickness is bound to mass, clearance and source physics."""
from copy import deepcopy
import json

import numpy as np
import pytest

from bench.isaacsim.release import digest
from scenes.workrooms.laundry_room import assets
from scenes.workrooms.laundry_room import garment_folding_assets as garments


@pytest.mark.parametrize('task,cells', [('towel_flattening', (30, 20)),
    ('towel_folding', (60, 40)), ('towel_stacking', (60, 40))])
def test_thick_towel_preserves_rest_area_density_and_real_layer_clearance(task, cells):
    old = assets.candidate_metadata(task, cells_xy=cells)
    new = assets.candidate_metadata(task, cells_xy=cells, material_profile='towel_4mm_density_v1')
    assert assets.validate_metadata(new) == new
    assert new['schema'] == 'laundry_towel_asset.v3'
    assert old['schema'] == 'laundry_towel_asset.v1' and 'material_profile' not in old
    for before, after in zip(old['cloths'], new['cloths']):
        assert before['rest_vertices_m'] == after['rest_vertices_m']
        assert before['triangles'] == after['triangles']
        assert after['mass_kg'] == .144 and after['physical_thickness_m'] == .004
        assert after['mass_kg']/(after['rest_area_m2']*.004) == pytest.approx(150.)
        q = np.asarray(after['initial_vertices_m'])
        assert q[:, 2].min() >= .902-1.e-10
        if task == 'towel_stacking':
            rest = np.asarray(after['rest_vertices_m'])
            lower, upper = q[rest[:, 0] < -.05], q[rest[:, 0] > .05]
            assert upper[:, 2].mean()-lower[:, 2].mean() == pytest.approx(.008)
    physics = new['physics_options']
    assert physics['cloth_thickness_m'] == .004
    assert physics['cloth_rest_offset_m'] == .002
    assert physics['cloth_contact_offset_m'] == .005
    assert physics['cloth_self_collision_filter_distance_m'] >= 2*physics['cloth_rest_offset_m']
    assert not new['native_physics_accepted']


@pytest.mark.parametrize('task,profile,thickness,mass', [
    ('shirt_folding', 'shirt_16mm_density_v1', .0016, .24),
    ('trousers_folding', 'trousers_18mm_density_v1', .0018, .39)])
def test_thicker_sewn_garment_keeps_topology_with_density_preserving_mass(task, profile, thickness, mass):
    options = {'mesh_profile': garments.FLAT_SHIRT_PROFILE} if task == 'shirt_folding' else {}
    old = garments.candidate_metadata(task, **options)
    new = garments.candidate_metadata(task, material_profile=profile, **options)
    assert garments.validate_metadata(new) == new
    a, b = old['cloths'][0], new['cloths'][0]
    assert a['rest_vertices_m'] == b['rest_vertices_m'] and a['triangles'] == b['triangles']
    assert b['mass_kg'] == mass and b['physical_thickness_m'] == thickness
    assert a['mass_kg']/(a['rest_area_m2']*.0012) == pytest.approx(mass/(b['rest_area_m2']*thickness))
    assert new['physics_options']['cloth_rest_offset_m'] == thickness/2
    assert new['physics_options']['cloth_self_collision_filter_distance_m'] == thickness
    assert new['goal_reference_status'] == 'provisional_projection_not_qualified'
    assert new['material_prior']['predicted_bending_thickness_factor'] == pytest.approx((thickness/.0012)**3)
    with pytest.raises(ValueError, match='Explicit thickness differs'):
        garments.candidate_metadata(task, material_profile=profile, physical_thickness_m=.0012, **options)
    changed = deepcopy(new)
    changed['cloths'][0]['mass_kg'] = a['mass_kg']
    with pytest.raises(ValueError):
        garments.validate_metadata(changed)


def test_material_source_rejects_old_thin_native_run_and_per_cloth_override(tmp_path):
    """The new reset profile must be backed by that material's actual scene."""
    profile = 'trousers_18mm_density_v1'
    metadata = garments.candidate_metadata('trousers_folding', material_profile=profile)
    metadata_path = tmp_path/'metadata.json'
    metadata_path.write_text(json.dumps(metadata))
    scene = {'isaacsim': metadata['physics_options'], 'cloths': [{
        'mass_kg': .39, 'isaacsim': {'cloth_thickness_m': .0018}}],
        'laundry_task_provenance': {'asset_candidate': {
            'manifest_path': str(tmp_path/'manifest.json'),
            'files_sha256': {str(metadata_path): digest(metadata_path)}}}}
    def source():
        scene_path = tmp_path/'scene.json';scene_path.write_text(json.dumps(scene))
        report_path = tmp_path/'native_report.json'
        report_path.write_text(json.dumps({'scene_input': {'path': str(scene_path), 'sha256': digest(scene_path)}}))
        return {'native_report': {'path': str(report_path), 'sha256': digest(report_path)}}
    bound = garments._bind_native_material_profile(source(), profile, 'trousers_folding')
    assert bound['material_profile'] == profile
    scene['cloths'][0]['isaacsim']['cloth_rest_offset_m'] = .0006
    with pytest.raises(ValueError, match='not simulated'):
        garments._bind_native_material_profile(source(), profile, 'trousers_folding')
    scene['cloths'][0]['isaacsim'] = {'cloth_thickness_m': .0012}
    scene['cloths'][0]['mass_kg'] = .26
    with pytest.raises(ValueError, match='not simulated'):
        garments._bind_native_material_profile(source(), profile, 'trousers_folding')


def test_new_towel_manifest_roundtrip_and_old_native_evidence_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, 'ASSETS', tmp_path)
    target = tmp_path/'towel4mm'
    manifest = assets.prepare_assets('towel_folding', target, cells_xy=(60, 40),
                                    material_profile='towel_4mm_density_v1')
    assert manifest['schema'] == 'laundry_towel_asset_manifest.v3'
    provenance, metadata = assets.load_asset_manifest('towel_folding', target/'manifest.json')
    assert metadata['physical_thickness_m'] == .004
    assert manifest['material_authoring_source']['path'] in provenance['files_sha256']
    with pytest.raises(ValueError, match='cannot inherit'):
        assets.prepare_assets('towel_stacking', tmp_path/'invalid',
            material_profile='towel_4mm_density_v1', source_fold_native_report='old.json',
            source_fold_audit='old_audit.json')


@pytest.mark.parametrize('task', ['towel_flattening', 'towel_folding', 'towel_stacking'])
def test_new_towel_placement_translates_every_initial_and_goal_without_changing_material(task):
    options = {'cells_xy': (60, 40), 'material_profile': 'towel_4mm_density_v1'}
    original = assets.candidate_metadata(task, **options)
    moved = assets.candidate_metadata(task, world_translation_xy_m=[.1, -.08], **options)
    assert assets.validate_metadata(moved) == moved
    assert 'world_translation_xy_m' not in original
    assert moved['physics_options'] == original['physics_options']
    assert moved['clear_rectangle_xy_m'] == original['clear_rectangle_xy_m']
    assert np.array_equal(np.asarray(moved['target_center_xy_m']),
                          np.asarray(original['target_center_xy_m'])+[.1, -.08])
    for a, b in zip(original['cloths'], moved['cloths']):
        assert a['rest_vertices_m'] == b['rest_vertices_m'] and a['triangles'] == b['triangles']
        assert a['mass_kg'] == b['mass_kg']
        assert np.array_equal(np.asarray(b['initial_vertices_m']),
                              np.asarray(a['initial_vertices_m'])+[.1, -.08, 0.])
    with pytest.raises(ValueError, match='requires the new towel material'):
        assets.candidate_metadata(task, world_translation_xy_m=[.1, -.08])


def test_partial_edge_fold_reset_retains_flat_material_and_needs_actual_unfolding(tmp_path, monkeypatch):
    options={'cells_xy':(60,40),'material_profile':'towel_4mm_density_v1'}
    original=assets.candidate_metadata('towel_flattening',**options)
    candidate=assets.candidate_metadata('towel_flattening',reset_profile='towel_partial_edge_fold_v1',**options)
    row=candidate['cloths'][0];rest=np.array(row['rest_vertices_m']);q=np.array(row['initial_vertices_m']);tri=np.array(row['triangles'])
    assert row['rest_vertices_m']==original['cloths'][0]['rest_vertices_m']
    assert np.ptp(rest[:,2])==0 and np.ptp(q[:,2])==pytest.approx(.016)
    edge=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0)
    ratios=np.linalg.norm(q[edge[:,1]]-q[edge[:,0]],axis=1)/np.linalg.norm(rest[edge[:,1]]-rest[edge[:,0]],axis=1)
    assert ratios.max()<1.000001 and ratios.min()>.90
    projected_area=np.abs(np.cross(q[tri[:,1]]-q[tri[:,0]],q[tri[:,2]]-q[tri[:,0]])[:,2]).sum()/2
    assert projected_area<row['rest_area_m2']  # Curved fold, not stretching a flat q target.
    assert candidate['physics_options']['cloth_surface_bend_stiffness']==100.
    assert original['physics_options']['cloth_surface_bend_stiffness']==1000.
    assert assets.validate_metadata(candidate)==candidate
    monkeypatch.setattr(assets,'ASSETS',tmp_path)
    assets.prepare_assets('towel_flattening',tmp_path/'partial',reset_profile='towel_partial_edge_fold_v1',**options)
    assert assets.load_asset_manifest('towel_flattening',tmp_path/'partial/manifest.json')[1]==candidate
    with pytest.raises(ValueError,match='Partial-edge reset'):
        assets.candidate_metadata('towel_flattening',material_profile='towel_4mm_density_v1',reset_profile='towel_partial_edge_fold_v1')
