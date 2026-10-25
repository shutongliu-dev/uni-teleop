"""CPU controls for native visibility assessment, with independent image data.

The fixture is an analytic front-facing box: its image rectangle and optical-z
depth follow from similar triangles. Do not synthesize it with the validator's
box_depth helper, which would let the same projection error pass both sides.
"""
from copy import deepcopy
import math

import numpy as np
import pytest

from scenes.workrooms.observations import expected_camera_world
from scripts.validation.validate_workroom_observations import assess, instance_mask


@pytest.fixture
def analytic_frame():
    width, height, focal_px = 64, 48, 64.
    front_depth, half_width = 1.75, .25
    rows, columns = np.mgrid[:height, :width]
    # Identity camera looks down world -z; the cube center is at z=-2 m.
    x_front = (columns + .5 - width / 2) * front_depth / focal_px
    y_front = (rows + .5 - height / 2) * front_depth / focal_px
    target = (np.abs(x_front) <= half_width) & (np.abs(y_front) <= half_width)
    depth = np.full((height, width), 3., dtype=float)
    depth[target] = front_depth
    rgb = np.stack((40 + columns, 50 + rows, 60 + (columns + rows) % 100), axis=-1).astype(np.uint8)
    rgb[target] = [120, 150, 180]
    hand = np.zeros((height, width), dtype=bool)
    hand[2:6, 2:6] = True
    projection = [[2., 0., 0., 0.], [0., 8. / 3., 0., 0.],
                  [0., 0., -1.002, -1.], [0., 0., -.02002, 0.]]
    return {'rgb': rgb, 'depth': depth,
            'calibration': {'resolution': [width, height], 'projection': projection,
                            'world_to_view': np.eye(4).tolist(), 'meters_per_scene_unit': 1.},
            'pose': {'position_m': [0., 0., -2.], 'quaternion_wxyz': [1., 0., 0., 0.]},
            'half': [.25, .25, .25], 'target_mask': target,
            'hand_mask': hand, 'required': True}


def failures(report):
    return {key for key, passed in report['checks'].items() if not passed}


def test_analytic_box_has_correct_pixel_count_depth_and_exposure(analytic_frame):
    report = assess(**analytic_frame)
    assert report['passed'], report
    # Pixel centers give columns 23..40 and rows 15..32, exactly 18 x 18.
    assert report['predicted_target_pixels'] == 18 * 18
    assert report['visible_target_pixels'] == 18 * 18
    assert report['target_visible_fraction'] == 1.
    assert report['target_depth_coverage'] == 1.
    assert report['target_depth_p95_error_m'] == pytest.approx(0., abs=1e-12)
    assert report['target_median_luma'] == pytest.approx(145.788)
    assert report['hand_pixels'] == 16


@pytest.mark.parametrize('removed_columns, passes', [(6, True), (10, False)])
def test_partial_occlusion_is_measured_against_full_projected_box(analytic_frame, removed_columns, passes):
    occluded = analytic_frame['target_mask'].copy()
    occluded[:, :23] = False
    occluded[:, 23 + removed_columns:] = False
    analytic_frame['rgb'][occluded] = [75, 85, 95]
    analytic_frame['depth'][occluded] = 1.25  # Opaque foreground surface.
    analytic_frame['target_mask'][:, 23:23 + removed_columns] = False
    report = assess(**analytic_frame)
    assert report['passed'] is passes
    assert report['target_visible_fraction'] == pytest.approx((18 - removed_columns) / 18)
    assert failures(report) == (set() if passes else {'target_not_mostly_occluded'})


def test_fully_occluded_target_cannot_pass_on_detailed_background(analytic_frame):
    analytic_frame['depth'][analytic_frame['target_mask']] = 1.25
    analytic_frame['rgb'][analytic_frame['target_mask']] = [75, 85, 95]
    analytic_frame['target_mask'][:] = False
    report = assess(**analytic_frame)
    assert not report['passed']
    assert report['checks']['image_has_information']
    assert not report['checks']['target_resolved']
    assert not report['checks']['target_not_mostly_occluded']
    assert not report['checks']['target_depth_matches_physx']
    assert report['target_depth_p95_error_m'] is None
    assert report['target_median_luma'] is None


@pytest.mark.parametrize('old_surface_depth', [1.70, 1.80])
def test_previous_pose_depth_rejected_even_with_current_instance_mask(analytic_frame, old_surface_depth):
    # A 5 cm stale surface can have the same visible segmentation at this scale.
    analytic_frame['depth'][analytic_frame['target_mask']] = old_surface_depth
    report = assess(**analytic_frame)
    assert not report['passed']
    assert failures(report) == {'target_depth_matches_physx'}
    assert report['target_depth_p95_error_m'] == pytest.approx(.05)
    assert report['target_depth_coverage'] == 1.


@pytest.mark.parametrize('invalid_depth', [0., np.nan, np.inf])
def test_missing_depth_cannot_be_hidden_by_accurate_remaining_pixels(analytic_frame, invalid_depth):
    analytic_frame['depth'][18:30, 27:37] = invalid_depth
    report = assess(**analytic_frame)
    assert not report['passed']
    assert failures(report) == {'target_depth_coverage'}
    assert report['target_depth_coverage'] < .90
    assert report['target_depth_p95_error_m'] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize('target_value', [0, 255])
def test_target_exposure_is_required_despite_background_information(analytic_frame, target_value):
    analytic_frame['rgb'][analytic_frame['target_mask']] = target_value
    report = assess(**analytic_frame)
    assert not report['passed']
    assert failures(report) == {'target_exposure'}
    assert report['checks']['image_has_information']
    assert report['target_median_luma'] == pytest.approx(target_value)


def test_constant_colored_frame_has_no_spatial_information(analytic_frame):
    # Across-channel RGB std is large, despite every pixel being identical.
    analytic_frame['rgb'][:] = [30, 150, 220]
    assert analytic_frame['rgb'].std() > 50.
    report = assess(**analytic_frame)
    assert not report['passed']
    assert failures(report) == {'image_has_information'}
    assert report['luma_spatial_std'] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize('hand_pixels', [0, 15])
def test_missing_or_unresolved_hand_rejected_for_required_work_view(analytic_frame, hand_pixels):
    analytic_frame['hand_mask'][:] = False
    analytic_frame['hand_mask'].flat[:hand_pixels] = True
    report = assess(**analytic_frame)
    assert not report['passed']
    assert failures(report) == {'hand_visible'}


def test_optional_home_view_does_not_claim_target_visibility(analytic_frame):
    analytic_frame['target_mask'][:] = False
    analytic_frame['hand_mask'][:] = False
    analytic_frame['depth'][:] = 0.
    analytic_frame['required'] = False
    report = assess(**analytic_frame)
    assert report['passed']
    assert report['checks'] == {'image_has_information': True}
    assert report['target_required'] is False
    assert report['visible_target_pixels'] == 0
    assert report['target_depth_p95_error_m'] is None


def test_instance_paths_match_complete_prim_and_descendants_only():
    prefix = '/World/Objects/cube_table'
    labels = {'1': prefix, '2': prefix + '/mesh', '3': prefix + '/visual/mesh',
              '4': prefix + '_aux', '5': prefix + '2/mesh',
              '6': '/Other' + prefix, '7': '/World/Objects/cube', '0': '/'}
    ids = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 999]], dtype=np.uint32)
    expected = np.array([[False, True, True], [True, False, False], [False, False, False]])
    np.testing.assert_array_equal(instance_mask(ids, labels, prefix), expected)


def test_instance_mask_rejects_semantic_class_labels():
    # Accidentally attaching semantic segmentation must not mimic prim IDs.
    with pytest.raises(ValueError, match='prim-path labels'):
        instance_mask(np.array([[1]], dtype=np.uint32), {'1': {'class': 'cube_table'}},
                      '/World/Objects/cube_table')


@pytest.fixture
def native_mount_reference():
    camera = {'kind': 'link', 'aim': False, 'standoff_m': 0.,
              'offset_xyz_m': [.1, .2, .3], 'offset_rpy_rad': [math.pi / 2, 0., 0.]}
    wrist = {'position_m': [1., 2., 3.],
             'quaternion_wxyz': [math.sqrt(.5), 0., 0., math.sqrt(.5)]}
    # Wrist Rz(90) followed by local Rx(90); translated offset becomes (-.2,.1,.3).
    # This exact matrix is independent of all production transform helpers.
    world = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.], [1., 0., 0., 0.],
                      [.8, 2.1, 3.3, 1.]])
    return camera, wrist, world


def test_expected_mount_matches_analytic_native_and_rtx_row_convention(native_mount_reference):
    camera, wrist, expected = native_mount_reference
    before = deepcopy((camera, wrist))
    actual = expected_camera_world(camera, wrist)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    # A camera-space point transformed through world and back via RTX view.
    optical_point = np.array([.2, -.1, -1., 1.])
    np.testing.assert_allclose(optical_point @ actual @ np.linalg.inv(expected), optical_point, atol=1e-14)
    assert (camera, wrist) == before


@pytest.mark.parametrize('fault', ['lagged_wrist', 'wrong_wrist_rotation', 'wrong_mount', 'lagged_rtx'])
def test_native_mount_comparison_rejects_wrong_and_stale_transforms(native_mount_reference, fault):
    camera, wrist, native_rtx_world = native_mount_reference
    if fault == 'lagged_wrist':
        wrist['position_m'][0] -= .04
    elif fault == 'wrong_wrist_rotation':
        wrist['quaternion_wxyz'] = [1., 0., 0., 0.]
    elif fault == 'wrong_mount':
        camera['offset_xyz_m'][0] += .02
    else:
        native_rtx_world[3, 0] -= .04
    expected = expected_camera_world(camera, wrist)
    position_error = np.linalg.norm(expected[3, :3] - native_rtx_world[3, :3])
    angle_error = np.arccos(np.clip((np.trace(expected[:3, :3] @ native_rtx_world[:3, :3].T) - 1) / 2, -1., 1.))
    assert not (position_error < .001 and angle_error < .005)
    if fault == 'wrong_wrist_rotation':
        assert angle_error == pytest.approx(math.pi / 2)
    else:
        assert position_error == pytest.approx(.02 if fault == 'wrong_mount' else .04)


@pytest.mark.parametrize('non_fixed', [{'aim': True}, {'aim_target_m': [0., 0., 0.]}, {'standoff_m': .12}])
def test_native_mount_reference_rejects_aimed_camera_specs(native_mount_reference, non_fixed):
    camera, wrist, _ = native_mount_reference
    with pytest.raises(ValueError, match='fixed link camera'):
        expected_camera_world({**camera, **non_fixed}, wrist)
