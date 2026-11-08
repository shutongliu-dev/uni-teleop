"""Explicit CUDA checks; select an approved UUID before enabling this group."""
import os
from fractions import Fraction
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('RUN_CLOTH_CUDA_AUDIT') != '1',
                              reason='Requires explicit approved CUDA audit selection')


@pytest.mark.parametrize('vertices,faces,expected', [
    ([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], [[0,1,2],[1,3,2]], 0),
    ([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], [[0,1,2],[0,1,3]], 0),
    ([[0,0,0],[1,0,0],[0,1,0],[.2,.2,0]], [[0,1,2],[0,1,3]], 1),
    ([[0,0,0],[1,0,0],[0,1,0],[.3,.2,-1],[.3,.2,1]], [[0,1,2],[0,3,4]], 1),
    ([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]], [[0,1,2],[0,3,4]], 0),
    ([[0,0,0],[1,0,0],[0,1,0],[.2,.2,0],[.5,.2,0],[.2,.5,0]], [[0,1,2],[3,4,5]], 1),
    ([[0,0,0],[1,0,0],[0,1,0],[.2,.2,.00001],[.5,.2,.00001],[.2,.5,.00001]], [[0,1,2],[3,4,5]], 0),
    ([[0,0,0],[.01,0,0],[0,.01,0],[.002,.002,-1e-8],[.005,.002,1e-8],[.002,.005,1e-8]], [[0,1,2],[3,4,5]], 1),
])
def test_structural_neighbors_and_actual_crossings(vertices, faces, expected):
    from bench.evaluation.cloth.gpu_intersections import ClothIntersectionAudit
    from bench.evaluation.cloth.cloth_self_intersection_precomputed import self_intersections
    p, tri = np.array(vertices, dtype=np.float32), np.array(faces)
    audit = ClothIntersectionAudit({'towel': tri}, {'towel': len(p)})
    result = audit.check({'towel': p})
    reference, _ = self_intersections(p.astype(np.float64), tri)
    assert result['self_intersections'] == len(reference) == expected
    assert result['intercloth_intersections'] == 0


def test_intercloth_separation_compression_and_refit():
    from bench.evaluation.cloth.gpu_intersections import ClothIntersectionAudit
    p = np.array([[0,0,0], [.2,0,0], [0,.2,0]], dtype=np.float32)
    tri = np.array([[0,1,2]])
    audit = ClothIntersectionAudit({'top': tri, 'base': tri}, {'top': 3, 'base': 3})
    for gap, expected in [(.003, 0.), (.002, .001), (.001, .002), (.004, 0.)]:
        result = audit.check({'top': p + [0,0,gap], 'base': p})
        assert result['self_intersections'] == result['intercloth_intersections'] == 0
        assert result['max_cloth_envelope_penetration_m'] == pytest.approx(expected, abs=1.e-7)
    crossing = p.copy()
    crossing[:, 2] = [-.01, .01, .01]
    result = audit.check({'top': crossing, 'base': p})
    assert result['intercloth_intersections'] == 1
    assert result['max_cloth_envelope_penetration_m'] == pytest.approx(.003, abs=1.e-7)


def test_coplanar_distinct_cloths_and_nonfinite_state():
    from bench.evaluation.cloth.gpu_intersections import ClothIntersectionAudit
    p = np.array([[0,0,0], [.2,0,0], [0,.2,0]], dtype=np.float32)
    tri = np.array([[0,1,2]])
    audit = ClothIntersectionAudit({'a': tri, 'b': tri}, {'a': 3, 'b': 3})
    assert audit.check({'a': p, 'b': p})['intercloth_intersections'] == 1
    p[0,0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        audit.check({'a': p, 'b': p})


def test_shared_vertex_does_not_allow_remote_coplanar_edge_overlap():
    from bench.evaluation.cloth.gpu_intersections import ClothIntersectionAudit
    p = np.array([[0,0,0],[.02,0,0],[0,.02,0],[.01,0,0],[0,-.02,0]], dtype=np.float32)
    audit = ClothIntersectionAudit({'a': [[0,1,2],[0,3,4]]}, {'a': 5})
    assert audit.check({'a': p})['self_intersections'] == 1


@pytest.mark.parametrize('faces', [
    [[0,3,2],[1,4,3]],
    [[1,4,3],[0,3,2]],
    [[2,3,0],[3,4,1]],
    [[3,2,0],[4,3,1]],
])
def test_near_parallel_native_shared_endpoint_is_not_an_intersection(faces):
    from bench.evaluation.cloth.gpu_intersections import ClothIntersectionAudit
    # Native crease01, step 3490, faces 87/89, vertices 45/46/75/76/77.
    # Float32 readback values are preserved exactly. The old determinant
    # reconstructs t=.999993377 and a point 1.3246e-7 m from shared vertex 76.
    p = np.array([
        [.4314776360988617, -.7300000190734863, .9264999032020569],
        [.44976523518562317, -.7300000190734863, .9294998645782471],
        [.4138926565647125, -.7099766731262207, .9165875315666199],
        [.4314776062965393, -.7099999785423279, .9264999032020569],
        [.4497652053833008, -.7099997997283936, .9294998645782471],
    ], dtype=np.float32)

    # Independent exact arithmetic proves that segment 45->76 meets the
    # other triangle only at that shared endpoint, without a wider tolerance.
    exact = [[Fraction(float(x)) for x in row] for row in p]
    def sub(a, b):
        return [x - y for x, y in zip(a, b)]
    def cross(a, b):
        return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))
    d, e1, e2 = sub(exact[3], exact[0]), sub(exact[4], exact[1]), sub(exact[3], exact[1])
    h, s = cross(d, e2), sub(exact[0], exact[1])
    z, determinant = cross(s, e1), dot(e1, h)
    assert determinant == Fraction(-150993, 9444732965739290427392)
    assert (dot(s, h)/determinant, dot(d, z)/determinant, dot(e2, z)/determinant) == (0, 1, 1)

    audit = ClothIntersectionAudit({'towel': faces}, {'towel': len(p)})
    result = audit.check({'towel': p})
    assert result['self_intersections'] == 0
    assert 'topology-shared segment endpoints excluded' in result['method']
    assert 'structural boundary tolerance 1e-7m' in result['method']


@pytest.mark.parametrize('vertices,faces', [
    # Shared vertex, but the opposite edge really crosses the other face.
    ([[0,0,0],[1,0,0],[0,1,0],[.3,.2,-1],[.3,.2,1]], [[0,1,2],[0,3,4]]),
    # Only one edge lies in the other face's plane. The whole triangle pair
    # is noncoplanar; the retained edge 3->4 must detect its remote endpoint.
    ([[0,0,0],[2,0,0],[0,2,0],[1,0,0],[0,0,1]], [[0,1,2],[0,3,4]]),
    # B's opposite-edge plane crossing is outside A. A's opposite edge must
    # still detect the overlap; testing both directions is essential.
    ([[0,0,0],[1,0,0],[0,1,0],[2,.5,1],[2,.5,-1]], [[0,1,2],[0,3,4]]),
    # Shared edge with coplanar area overlap remains invalid.
    ([[0,0,0],[1,0,0],[0,1,0],[.2,.2,0]], [[0,1,2],[0,1,3]]),
    # A second geometrically coincident vertex has a different topology ID.
    # This is remote touch, not permission to enlarge the shared boundary.
    ([[0,0,0],[1,0,0],[0,1,0],[1,0,0],[1,-1,0]], [[0,1,2],[0,3,4]]),
])
def test_endpoint_exclusion_preserves_remote_intersections(vertices, faces):
    from bench.evaluation.cloth.gpu_intersections import ClothIntersectionAudit
    p = np.asarray(vertices, dtype=np.float32)
    for ordered in (faces, faces[::-1]):
        audit = ClothIntersectionAudit({'towel': ordered}, {'towel': len(p)})
        assert audit.check({'towel': p})['self_intersections'] == 1
