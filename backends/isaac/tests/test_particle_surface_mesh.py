"""Short geometric checks; these do not simulate liquid or qualify pouring."""

import hashlib
import os

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.particle_surface_mesh import reconstruct_particle_surface


GPU_UUID = 'GPU-0e25a0f7-5cfd-4039-a657-466bda03de44'


@pytest.fixture
def gpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        pytest.skip('Short geometry checks require explicit authorized GPU UUID visibility')
    wp = pytest.importorskip('warp')
    if not wp.is_cuda_available():
        pytest.skip('CUDA is unavailable')
    return 'cuda:0'


def assert_closed_outward(mesh):
    v, f = mesh.vertices, mesh.triangles
    assert len(v) and len(f)
    assert np.isfinite(v).all() and np.isfinite(mesh.normals).all()
    np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-12)
    edges = np.sort(np.concatenate((f[:, (0, 1)], f[:, (1, 2)], f[:, (2, 0)])), axis=1)
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    for group in mesh.metadata['groups']:
        assert group['visual_volume_m3'] > 0.0


@pytest.mark.parametrize('points, kwargs', [
    ([[0, 0]], {}),
    ([[0, 0, np.nan]], {}),
    ([[0, 0, np.inf]], {}),
    ([[0, 0, 0]], {'spacing_m': 0.0}),
    ([[0, 0, 0]], {'voxel_size_m': -1.0}),
    ([[0, 0, 0]], {'kernel_radius_m': 0.1}),
    ([[0, 0, 0]], {'max_grid_nodes': 0}),
    ([[0, 0, 0]], {'max_particles': True}),
    ([[0, 0, 0]], {'device': 'cpu'}),
])
def test_rejects_invalid_input_without_gpu(points, kwargs):
    args = {'spacing_m': 0.008, 'device': 'cuda:0', **kwargs}
    with pytest.raises(ValueError):
        reconstruct_particle_surface(points, **args)


def test_empty_surface_does_not_initialize_gpu():
    mesh = reconstruct_particle_surface(np.empty((0, 3)), 0.008, device='cuda:0')
    assert mesh.vertices.shape == mesh.triangles.shape == mesh.normals.shape == (0, 3)
    assert mesh.metadata['source_point_count'] == mesh.metadata['grid_nodes'] == 0


def test_grid_budget_is_checked_before_device_lookup():
    points = np.array([[0., 0., 0.], [0.008, 0., 0.]])
    with pytest.raises(ValueError, match='max_grid_nodes'):
        reconstruct_particle_surface(points, 0.008, device='cuda:999', max_grid_nodes=8)


def test_single_drop_has_outward_normals_and_expected_radius(gpu):
    point = np.array([[0.65, -0.1, 0.9]], dtype=np.float64)
    mesh = reconstruct_particle_surface(point, 0.008, device=gpu)
    assert_closed_outward(mesh)
    rays = mesh.vertices - point[0]
    lengths = np.linalg.norm(rays, axis=1)
    assert lengths.max() <= 0.004001
    assert lengths.min() > 0.0036
    assert np.all(np.einsum('ij,ij->i', rays, mesh.normals) > 0.0)
    sphere_volume = 4 / 3 * np.pi * 0.004**3
    assert 0.75 < mesh.metadata['visual_volume_m3'] / sphere_volume < 1.05


def test_two_distant_drops_remain_separate_without_room_sized_grid(gpu):
    points = np.array([[0., 0., 0.], [10., 20., 30.]])
    mesh = reconstruct_particle_surface(points, 0.008, device=gpu)
    assert_closed_outward(mesh)
    assert mesh.metadata['group_count'] == 2
    assert mesh.metadata['grid_nodes'] < 6_000
    distances = np.linalg.norm(mesh.vertices[:, None, :] - points[None, :, :], axis=2)
    assert np.max(distances.min(axis=1)) <= 0.004001


def test_points_on_opposite_sides_of_large_gap_are_not_joined(gpu):
    points = np.array([[0., 0., 0.], [0.024, 0., 0.]])
    mesh = reconstruct_particle_surface(points, 0.008, device=gpu)
    assert mesh.metadata['group_count'] == 2
    centroids = mesh.vertices[mesh.triangles].mean(axis=1)
    assert not np.any((centroids[:, 0] > 0.005) & (centroids[:, 0] < 0.019))


def test_scale_and_translation_equivariance(gpu):
    points = np.array([[0., 0., 0.], [0.008, 0., 0.], [0.008, 0.008, 0.]])
    original = reconstruct_particle_surface(points, 0.008, device=gpu)
    scale, offset = 2.5, np.array([17., -8., 3.])
    transformed = reconstruct_particle_surface(points * scale + offset, 0.008 * scale, device=gpu)
    np.testing.assert_array_equal(transformed.triangles, original.triangles)
    np.testing.assert_allclose(transformed.vertices, original.vertices * scale + offset, atol=2e-8, rtol=0)
    np.testing.assert_allclose(transformed.normals, original.normals, atol=2e-6, rtol=0)
    assert transformed.metadata['visual_volume_m3'] == pytest.approx(original.metadata['visual_volume_m3'] * scale**3, rel=1e-5)


def test_488_particle_bulk_keeps_input_and_separate_spill(gpu):
    coords = np.array(np.meshgrid(np.arange(-4, 5), np.arange(-4, 5), np.arange(-4, 5), indexing='ij')).reshape(3, -1).T
    order = np.argsort(np.einsum('ij,ij->i', coords, coords), kind='stable')
    points = np.asarray(coords[order[:487]] * 0.008 + [0.6, -0.1, 0.9], dtype=np.float32)
    points = np.vstack((points, np.array([[0.8, 0.2, 0.8]], dtype=np.float32)))
    before = points.copy()
    points.setflags(write=False)
    mesh = reconstruct_particle_surface(points, 0.008, device=gpu)
    assert_closed_outward(mesh)
    np.testing.assert_array_equal(points, before)
    assert mesh.metadata['source_sha256'] == hashlib.sha256(before.tobytes()).hexdigest()
    assert mesh.metadata['source_point_count'] == 488
    assert mesh.metadata['group_count'] == 2
    assert sorted(len(g['source_indices']) for g in mesh.metadata['groups']) == [1, 487]
    assert mesh.metadata['grid_nodes'] < 100_000
    assert mesh.metadata['visual_only'] and not mesh.metadata['visual_volume_is_mass_ledger']
