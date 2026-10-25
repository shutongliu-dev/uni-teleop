import itertools
import numpy as np
import pytest
from scripts.validation.mechanical_convex_geometry import convex_surface_error


def test_redundant_face_vertices_preserve_surface_but_shrinkage_does_not():
    cube = np.array(list(itertools.product((-.01, .01), repeat=3)))
    extra = np.vstack((cube, [[.01, 0., 0.]]))
    assert convex_surface_error(extra, cube) < 1e-12
    assert convex_surface_error(cube, cube*.9) == pytest.approx(.001)
    assert convex_surface_error(cube, cube*1.1) == pytest.approx(.001)


def test_translation_is_measured_and_invalid_geometry_fails():
    cube = np.array(list(itertools.product((-.01, .01), repeat=3)))
    assert convex_surface_error(cube, cube+[.0002, 0., 0.]) == pytest.approx(.0002)
    with pytest.raises(ValueError):
        convex_surface_error(cube, [[float('nan'), 0., 0.]])
