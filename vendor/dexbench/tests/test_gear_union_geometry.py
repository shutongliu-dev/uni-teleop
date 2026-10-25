"""Physical union must preserve all teeth, the bore ledges and source integrals."""
import numpy as np
import pytest

from scripts.validation.gear_union_geometry import geometry_report, source_asset_audit


@pytest.mark.parametrize('input_hub', [False, True])
def test_single_surface_preserves_original_solid_and_source_boundary(input_hub):
    mesh, report = geometry_report(input_hub=input_hub)
    assert report['passed']
    assert report['original_physical_piece_count'] == 448
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.euler_number == 0 and mesh.body_count == 1
    assert report['source_boundary_witnesses'] >= 40000
    assert report['outward_face_centers_inside_source_union'] == 0
    assert report['inward_face_centers_outside_source_union'] == 0
    assert report['mass_absolute_error_kg'] < 1.e-12
    assert report['inertia_max_absolute_error_kg_m2'] < 1.e-15
    # The source's radial tooth-root relief must remain a real vertical face,
    # and the mismatched root/hub hole must have both upward/downward ledges.
    normal = mesh.face_normals
    center = mesh.triangles_center
    radius = np.linalg.norm(center[:, :2], axis=1)
    root_relief = (radius > .04375) & (radius < .04698) & (abs(normal[:, 2]) < 1.e-12)
    assert root_relief.sum() >= 80
    inner_ledge = (abs(center[:, 2]-.018) < 1.e-12) & (radius < .0121)
    assert np.any(normal[inner_ledge, 2] > .99)
    assert np.any(normal[inner_ledge, 2] < -.99)
    assert not np.any(radius < .01199)


@pytest.mark.parametrize('name', ['input_gear', 'idler_gear', 'output_gear'])
def test_immutable_source_files_match_generated_cells_within_obj_rounding(name):
    report = source_asset_audit(name)
    assert report['passed']
    assert report['physical_piece_count'] == 448
    assert len(report['source_files_sha256']) == 449
