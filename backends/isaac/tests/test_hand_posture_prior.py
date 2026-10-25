import numpy as np
import pytest

from bench.control.hand_posture import HandPosturePrior


NAMES = ['right_index_MCP_FE', 'right_index_PIP', 'right_index_DIP']


def test_projection_uses_original_point_not_box_clipped_point():
    prior = HandPosturePrior()
    original = np.array([.2, -3., 3.])
    result = prior.project('index', original, [0., 0., 0.], [2., 2., 2.], NAMES)
    np.testing.assert_allclose(result, [.2, 0., .5], atol=1e-15)
    np.testing.assert_array_equal(original, [.2, -3., 3.])


def test_random_pair_projection_satisfies_optimality_against_feasible_points():
    rng = np.random.default_rng(74109)
    prior = HandPosturePrior()
    lo, hi = np.array([-.2, -.3, -.1]), np.array([1.3, 1.1, 1.7])
    for _ in range(250):
        original = rng.uniform(-5, 5, 3)
        result = prior.project('index', original, lo, hi, NAMES)
        assert np.all(result >= lo) and np.all(result <= hi)
        assert result[2] - result[1] <= .5
        # Projection onto a closed convex set: (x-Px)·(y-Px) <= 0
        # for every feasible y. Test independent feasible candidates.
        candidates = rng.uniform(lo, hi, (150, 3))
        candidates = candidates[candidates[:, 2] - candidates[:, 1] <= .5]
        assert np.all((candidates - result) @ (original - result) <= 1.e-12)


def test_thumb_is_only_box_projected_and_empty_pair_is_rejected():
    prior = HandPosturePrior()
    np.testing.assert_array_equal(prior.project('thumb', [-1., 3.], [0., 0.], [2., 2.], ['a', 'b']), [0., 2.])
    with pytest.raises(ValueError, match='empty intersection'):
        prior.project('index', [0., 0., 2.], [0., 0., 2.], [1., 1., 3.], NAMES)


@pytest.mark.parametrize('bad', [-.1, float('nan'), float('inf'), True])
def test_prior_rejects_invalid_limits(bad):
    with pytest.raises(ValueError):
        HandPosturePrior(bad)
