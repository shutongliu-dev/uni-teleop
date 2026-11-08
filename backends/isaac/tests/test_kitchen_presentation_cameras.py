from copy import deepcopy

import numpy as np

from scripts.media.render_kitchen_tasks import presentation_cameras


def test_detail_fits_tall_rear_and_near_foreground_props():
    # The old max-world-axis distance cropped both the tall rear bottle and
    # the closest foreground utensil, despite fitting their horizontal span.
    vertices = np.array([[-.2, .2, 0.], [-.2, .2, .3],
                         [.2, -.2, 0.], [.2, -.2, .1]])
    scene = {'cameras': {'task': {'eye': [1, 2, 3], 'target': [0, 0, 0]}}}
    before = deepcopy(scene)
    task = {'id': 'pantry_stocking', 'asset_metadata': {'objects': [
        {'initial_pose_xyzw': [0, 0, .9, 0, 0, 0, 1],
         'collision_support_points_m': vertices.tolist()}]}}
    for aspect in (4./3., 1., 2.):
        cameras = presentation_cameras(scene, task, aspect_ratio=aspect)
        assert scene == before
        assert cameras['task'] == scene['cameras']['task']
        camera = cameras['detail']
        forward = np.array(camera['target']) - camera['eye']
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = vertices + [0, 0, .9] - camera['eye']
        depth = relative @ forward
        assert np.all(depth > 0.)
        vertical_ndc = (relative @ up) / depth * camera['focal_length_mm'] / 13.5
        horizontal_ndc = (relative @ right) / depth * camera['focal_length_mm'] / (13.5 * aspect)
        assert np.max(np.abs(vertical_ndc)) <= .900001
        assert np.max(np.abs(horizontal_ndc)) <= .900001
