"""Actual clothing material topology; CPU geometry is not native qualification."""
from copy import deepcopy

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.cloth_visuals import ClosedClothVisual, closed_skin
from bench.simulation.isaacsim_backend.geometry import cloth_grid, named_cloth_specs
from scenes.workrooms.laundry_room.garments import (
    NOMINAL_MASS_KG, TASK_KINDS, candidate_metadata, geometry_report, validate_metadata,
)


def scene_spec(row):
    return {**deepcopy(row), 'enabled': True, 'origin_m': [0., 0., 0.],
            'quaternion_wxyz': [1., 0., 0., 0.], 'self_contact': True, 'friction': .45,
            'isaacsim': {'cloth_thickness_m': row['physical_thickness_m']}}


@pytest.mark.parametrize('task', TASK_KINDS)
def test_real_pattern_topology_folds_without_intersection_or_gross_strain(task):
    metadata = candidate_metadata(task)
    assert validate_metadata(deepcopy(metadata)) == metadata
    assert [row['id'] for row in metadata['cloths']] == ['towel', 'towel_base']
    row = metadata['cloths'][0]
    rest, triangles, initial = (np.asarray(row[key]) for key in ('rest_vertices_m', 'triangles', 'initial_vertices_m'))
    # Neither a recolored rectangle nor a closed rigid garment prop.
    assert row['rest_area_m2'] < np.prod(row['size_m']) * .85
    assert row['dynamic_model'] == 'native_surface_deformable'
    assert row['fixed_vertices'] == []
    assert row['mass_kg'] == NOMINAL_MASS_KG[task]
    assert row['mass_kg'] == pytest.approx(row['rest_area_m2'] * row['areal_density_kg_m2'], abs=5.e-7)
    assert row['physical_thickness_m'] == (.0016 if task == 'shirt_stacking' else .0018)
    assert all(len(indices) >= 3 for indices in row['boundary_features'].values())
    assert all(len(group['face_indices']) > 0 for group in row['visual_face_groups'].values())
    report = geometry_report(task)
    assert report['self_intersection_pairs'] == []
    assert report['closed_skin_self_intersection_pairs'] == []
    assert report['topology']['connected_component_sizes'] == [len(rest)]
    assert report['topology']['nonmanifold_edges'] == 0
    assert report['closed_skin_topology']['boundary_edges'] == 0
    assert report['closed_skin_topology']['nonmanifold_edges'] == 0
    assert report['minimum_edge_ratio'] > .95
    assert report['maximum_edge_ratio'] < 1.00001
    assert report['minimum_triangle_area_ratio'] > .94
    assert report['maximum_triangle_area_ratio'] < 1.15
    assert report['folded_layer_pair_count'] > 500
    assert report['bottom_support_vertex_count'] > 300
    assert initial[:, 2].min() == pytest.approx(.90+row['physical_thickness_m']/2)
    assert np.ptp(initial[:, 2]) < .014
    assert not report['native_physics_accepted']


@pytest.mark.parametrize('task', TASK_KINDS)
def test_custom_mesh_enters_same_native_surface_path_and_preserves_material_ids(task):
    row = candidate_metadata(task)['cloths'][0]
    spec = scene_spec(row)
    before = deepcopy(spec)
    assert named_cloth_specs({'cloths': [spec]})['towel'] == before
    rest, triangles = cloth_grid(spec)
    np.testing.assert_allclose(rest, row['rest_vertices_m'], atol=2.e-8)
    np.testing.assert_array_equal(triangles, row['triangles'])
    assert not rest[:, 2].any()
    assert spec == before
    pairs = np.asarray(row['folded_layer_pairs'])
    reference = np.asarray(row['folded_reference_vertices_m'])
    assert (reference[pairs[:, 1], 2]-reference[pairs[:, 0], 2] >= row['physical_thickness_m']*.8-1.e-12).all()
    assert np.linalg.norm(reference[pairs[:, 1], :2]-reference[pairs[:, 0], :2], axis=1).max() < .00761


@pytest.mark.parametrize('task', TASK_KINDS)
def test_material_features_follow_closed_read_only_skin(task):
    from pxr import Usd, UsdGeom, Vt
    row = candidate_metadata(task)['cloths'][0]
    stage = Usd.Stage.CreateInMemory()
    native = UsdGeom.Mesh.Define(stage, '/World/Cloths/towel')
    initial = np.asarray(row['initial_vertices_m'], dtype=np.float32)
    native.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(initial))
    visual = ClosedClothVisual(stage, str(native.GetPath()), 'towel', row['triangles'],
        row['rest_vertices_m'], row['physical_thickness_m'], row['display_color'],
        initial_world_vertices=initial, visual_face_groups=row['visual_face_groups'])
    assert visual.mesh.GetNormalsInterpolation() == 'vertex'
    assert len(visual.mesh.GetNormalsAttr().Get()) == len(visual.mesh.GetPointsAttr().Get())
    assert visual.metadata['face_groups'] == {name: len(group['face_indices']) for name, group in row['visual_face_groups'].items()}
    assert not native.GetPrim().GetAppliedSchemas()
    assert not any('Physics' in schema for schema in visual.mesh.GetPrim().GetAppliedSchemas())
    shifted = initial+[.01, 0, .02]
    visual.sync(shifted)
    assert visual.mesh.GetNormalsInterpolation() == 'vertex'
    assert len(visual.mesh.GetNormalsAttr().Get()) == len(visual.mesh.GetPointsAttr().Get())
    np.testing.assert_array_equal(np.asarray(native.GetPointsAttr().Get()), initial)
    np.testing.assert_allclose(visual.mesh.GetPointsAttr().Get(), closed_skin(shifted, row['triangles'], row['physical_thickness_m'])[0])


@pytest.mark.parametrize('mutation', [
    lambda spec: spec.pop('mesh_type'),
    lambda spec: spec.update(triangles=[[0., 1., 2.]]),
    lambda spec: spec['triangles'].append(spec['triangles'][0]),
    lambda spec: spec['triangles'][0].__setitem__(0, 10**12),
    lambda spec: spec['rest_vertices_m'][0].__setitem__(0, float('nan')),
    lambda spec: spec['rest_vertices_m'].append([1., 1., 1.]),
    lambda spec: spec.update(rest_area_m2=.24),
    lambda spec: spec['triangles'][0].reverse(),
])
def test_custom_topology_errors_reject_before_native_start(mutation):
    spec = scene_spec(candidate_metadata('shirt_stacking')['cloths'][0])
    mutation(spec)
    with pytest.raises(ValueError):
        named_cloth_specs({'cloths': [spec]})


def test_separate_garments_do_not_share_mutable_features_or_layer_pairs():
    metadata = candidate_metadata('shirt_stacking')
    before = deepcopy(metadata['cloths'][1])
    metadata['cloths'][0]['boundary_features']['neckline'].clear()
    metadata['cloths'][0]['folded_layer_pairs'].clear()
    assert metadata['cloths'][1] == before


@pytest.mark.parametrize('task', TASK_KINDS)
def test_garment_physics_reuses_successful_towel_scale_without_claiming_validation(task):
    metadata = candidate_metadata(task)
    options = metadata['physics_options']
    thickness = metadata['cloths'][0]['physical_thickness_m']
    assert options == {'physics_dt_s': .0005, 'cloth_iterations': 64,
        'cloth_linear_damping': 2., 'cloth_thickness_m': thickness,
        'cloth_youngs_modulus_pa': 667000., 'cloth_poissons_ratio': .3,
        'cloth_surface_bend_stiffness': 100., 'cloth_collision_pair_update_frequency': 8,
        'cloth_collision_iteration_multiplier': 8, 'cloth_speculative_ccd': True,
        'cloth_self_collision_filter_distance_m': thickness,
        'cloth_contact_offset_m': .0018 if task == 'shirt_stacking' else .002,
        'cloth_rest_offset_m': thickness/2}
    assert not metadata['native_physics_accepted']
    assert not metadata['material_calibrated']
    assert 'Towel evidence does not qualify garment dynamics' in metadata['physics_prior_basis']


def test_garment_asset_manifest_binds_author_geometry_and_original_weave(tmp_path, monkeypatch):
    from scenes.workrooms.laundry_room import assets, garments
    monkeypatch.setattr(garments, 'ASSETS', tmp_path)
    output = tmp_path/'shirt_v1'
    manifest = assets.prepare_assets('shirt_stacking', output)
    provenance, metadata = assets.load_asset_manifest('shirt_stacking', output/'manifest.json')
    assert manifest['schema'] == garments.MANIFEST_SCHEMA
    assert metadata['procedural_texture_root'] == str(output)
    assert len(provenance['files_sha256']) == 7
    assert not manifest['native_physics_accepted']
    assert not manifest['material_calibrated']
    with pytest.raises(ValueError, match='fresh'):
        assets.prepare_assets('shirt_stacking', output)
    with (output/'woven_normal.png').open('ab') as handle:
        handle.write(b'modified')
    with pytest.raises(ValueError, match='provenance'):
        assets.load_asset_manifest('shirt_stacking', output/'manifest.json')


def test_omitted_source_center_preserves_both_original_v1_descriptors_byte_for_byte():
    import hashlib
    import json
    frozen = {'shirt_stacking': 'e93788696d1487bf7daf3d4b7d79da63549efe143092d2f248daf6edb4d798a3',
              'trousers_stacking': '8aabe7a70b591866290bfe589c591e3a3a2ffe465d4d261d2e85bea3cb7c5d46'}
    for task, sha in frozen.items():
        metadata = candidate_metadata(task, source_center_xy_m=None)
        assert 'source_center_xy_m' not in metadata
        assert hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == sha
        assert validate_metadata(metadata) == metadata


def test_explicit_shirt_source_center_only_translates_source_initial_and_reference():
    old = candidate_metadata('shirt_stacking')
    new = candidate_metadata('shirt_stacking', source_center_xy_m=[.40, -.55])
    assert validate_metadata(deepcopy(new)) == new
    assert new['source_center_xy_m'] == [.40, -.55]
    assert new['cloths'][1] == old['cloths'][1]
    for key in ('initial_vertices_m', 'folded_reference_vertices_m'):
        np.testing.assert_allclose(np.asarray(new['cloths'][0][key])-old['cloths'][0][key],
            np.broadcast_to([.10, 0., 0.], np.asarray(old['cloths'][0][key]).shape), rtol=0., atol=2.e-16)
        new['cloths'][0][key] = old['cloths'][0][key]
    new.pop('source_center_xy_m')
    assert new == old


def test_regularized_shirt_preserves_outline_crease_and_material_but_removes_tiny_edges():
    from bench.simulation.isaacsim_backend.cloth_visuals import boundary_edges
    from scenes.workrooms.laundry_room.garments import REGULARIZED_SHIRT_PROFILE
    old = candidate_metadata('shirt_stacking', source_center_xy_m=[.40, -.55])
    new = candidate_metadata('shirt_stacking', source_center_xy_m=[.40, -.55], mesh_profile=REGULARIZED_SHIRT_PROFILE)
    assert validate_metadata(deepcopy(old)) == old
    assert validate_metadata(deepcopy(new)) == new
    assert 'mesh_profile' not in old and new['mesh_profile'] == REGULARIZED_SHIRT_PROFILE
    assert old['design'] == new['design']
    assert old['physics_options'] == new['physics_options']
    for before, after in zip(old['cloths'], new['cloths']):
        rest_before, rest_after = np.asarray(before['rest_vertices_m']), np.asarray(after['rest_vertices_m'])
        boundary_before = rest_before[np.unique(boundary_edges(before['triangles']))]
        boundary_after = rest_after[np.unique(boundary_edges(after['triangles']))]
        assert set(map(tuple, boundary_before)) == set(map(tuple, boundary_after))
        assert len(rest_before)-len(rest_after) == 4
        lookup = {tuple(point): index for index, point in enumerate(rest_before)}
        retained = [lookup[tuple(point)] for point in rest_after]
        for key in ('initial_vertices_m', 'folded_reference_vertices_m'):
            np.testing.assert_array_equal(np.asarray(before[key])[retained], after[key])
        for key in ('mass_kg', 'physical_thickness_m', 'rest_area_m2', 'areal_density_kg_m2', 'size_m'):
            assert before[key] == after[key]
        for axis, hinge, _, radius in old['design']['fold_sequence_axis_hinge_sign_radius_m']:
            for coordinate in hinge+np.linspace(-np.pi*radius/2, np.pi*radius/2, 9):
                old_line = rest_before[np.isclose(rest_before[:, axis], coordinate, atol=1.e-10, rtol=0.)]
                new_line = rest_after[np.isclose(rest_after[:, axis], coordinate, atol=1.e-10, rtol=0.)]
                assert set(map(tuple, old_line)) == set(map(tuple, new_line))
    report = geometry_report('shirt_stacking', source_center_xy_m=[.40, -.55], mesh_profile=REGULARIZED_SHIRT_PROFILE)
    assert report['minimum_rest_edge_m'] >= .0004
    assert report['self_intersection_pairs'] == report['closed_skin_self_intersection_pairs'] == []
    assert report['minimum_edge_ratio'] > .99 and report['maximum_edge_ratio'] < 1.00001
    assert report['minimum_triangle_area_ratio'] > .99 and report['maximum_triangle_area_ratio'] < 1.00001


def test_shirt_mesh_profile_cannot_silently_change_trousers_or_accept_unknown_versions():
    from scenes.workrooms.laundry_room.garments import REGULARIZED_SHIRT_PROFILE
    for task, profile in [('trousers_stacking', REGULARIZED_SHIRT_PROFILE), ('shirt_stacking', 'unknown')]:
        with pytest.raises(ValueError, match='mesh profile'):
            candidate_metadata(task, mesh_profile=profile)
