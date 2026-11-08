"""Exact cutlery convex cells preserve material and open slots, without overlaps."""
import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import trimesh

from scenes.workrooms.kitchen.assets import _utensil
from scenes.workrooms.kitchen.utensil_convex import partition_prismatic_mesh


@pytest.mark.parametrize('kind, maximum_parts', [('fork', 16), ('knife', 6)])
@pytest.mark.parametrize('metric_scale', [1., 1.15])
def test_convex_cutlery_is_exact_nonoverlapping_material_partition(kind, maximum_parts, metric_scale):
    mesh, _ = _utensil(kind)
    mesh.apply_scale(metric_scale)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32).astype(float)
    parts, receipt = partition_prismatic_mesh(mesh)
    assert len(parts) <= maximum_parts
    assert all(p.is_watertight and p.is_convex for p in parts)
    joined = trimesh.util.concatenate(parts)
    np.testing.assert_array_equal(joined.bounds, mesh.bounds)
    np.testing.assert_allclose(joined.center_mass, mesh.center_mass, atol=1e-10, rtol=0.)
    np.testing.assert_allclose(joined.moment_inertia, mesh.moment_inertia, atol=1e-16, rtol=1e-8)
    assert sum(p.volume for p in parts) == pytest.approx(mesh.volume, rel=1e-9)
    hulls = [ConvexHull(p.vertices).equations for p in parts]
    for i, a in enumerate(hulls):
        for b in hulls[i+1:]:
            equations = np.vstack((a, b))
            # An overlapping 3-D interior would contain a positive-radius ball.
            constraints = np.c_[equations[:, :3], np.linalg.norm(equations[:, :3], axis=1)]
            result = linprog([0., 0., 0., -1.], A_ub=constraints, b_ub=-equations[:, 3],
                bounds=[(None, None)]*3+[(0., None)], method='highs',
                options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
            assert result.status == 2 or (result.success and result.x[3] <= 1e-9)
    if kind == 'fork':
        for x in (-.0095, .0005, .0105):
            gap = np.array([x, -.070, .003])*metric_scale
            assert not any(np.max(eq[:, :3]@gap+eq[:, 3]) < -1e-9 for eq in hulls)
    assert not receipt['mass_inertia_material_pose_visual_changes']


def test_curved_spoon_cannot_be_accidentally_filled_as_a_prism():
    spoon, _ = _utensil('spoon')
    with pytest.raises(ValueError, match='two planar vertex layers'):
        partition_prismatic_mesh(spoon)
