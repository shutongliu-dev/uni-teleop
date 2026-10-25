"""The candidate may partition source geometry, never fill its bore or move it."""
import numpy as np
import pytest
from scripts.validation.gear_sector_geometry import (partition, sector, signed_surface_distance,
    yaw_matrix)
from scripts.validation.gear_union_geometry import gear_exterior


@pytest.fixture(scope='module')
def partitioned():
    source = gear_exterior(input_hub=True)
    pieces, report = partition(source)
    return source, pieces, report


def test_exact_union_partition_is_closed_and_integrals_preserved(partitioned):
    source, pieces, report = partitioned
    assert report['passed'] and all(report['checks'].values())
    assert len(pieces) == 8
    assert all(p.euler_number == 2 and p.is_watertight for p in pieces)
    assert report['max_source_face_area_partition_error_m2'] < 1.e-16
    # The central axis must be outside every solid sector at actual gear heights.
    axis = np.array([[0., 0., z] for z in [.001, .010, .020, .039]])
    assert all(np.all(signed_surface_distance(p, axis) > .01) for p in pieces)


def test_local_frame_preserves_vertices_and_each_axis_calibration(partitioned):
    source, pieces, report = partitioned
    for p, row in zip(pieces, report['sectors'], strict=True):
        r = yaw_matrix(row['local_frame_yaw_deg'])
        assert np.allclose((p.vertices@r)@r.T, p.vertices, atol=1.e-16, rtol=0.)
        assert max(np.ptp(p.vertices@r,axis=0))/1250 < 36.e-6
        c = row['calibration']
        normals = np.asarray(c['expected_gradient_body'])
        assert np.max(normals[:,2]) > .99 and np.min(normals[:,2]) < -.99
        assert np.linalg.matrix_rank(normals) == 3
        assert c['minimum_distance_to_cut_plane_m'] >= .0019
        assert np.allclose(c['cpu_signed_distance_m'], c['expected_signed_distance_m'], atol=1.e-10, rtol=0.)


def test_missing_or_undeclared_partition_fails():
    with pytest.raises(ValueError, match='declared'):
        sector(None, 0, 7)


def test_buried_cut_is_not_original_body_signed_distance(partitioned):
    source,pieces,_=partitioned
    points=np.array([[.03,0.,.009],[.04,0.,.0179],[.03,.0001,.009],[.03,-.0001,.009]])
    body=signed_surface_distance(source,points)
    minimum=np.min([signed_surface_distance(p,points) for p in pieces],axis=0)
    assert body[0]==pytest.approx(-.009,abs=1.e-10)
    assert body[1]==pytest.approx(-.0001,abs=1.e-10)
    assert np.all(abs(minimum[:2])<1.e-10)
    assert np.allclose(minimum[2:],-.0001,atol=1.e-10,rtol=0.)
