"""Native-engineering motion candidates; CPU geometry is not cloth feasibility."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from scenes.workrooms.laundry_room.composition import build_task_inputs
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scenes.workrooms.laundry_room.garments import candidate_metadata, REGULARIZED_SHIRT_PROFILE
from scripts.validation.laundry_engineering_motion import build_motion, GARMENT_STACK_PROFILE


def inputs(task_id, source_x=None):
    options = {'source_center_xy_m': [source_x or (.4 if task_id == 'shirt_stacking' else .3), -.55]}
    if task_id == 'shirt_stacking':
        options['mesh_profile'] = REGULARIZED_SHIRT_PROFILE
    metadata = candidate_metadata(task_id, **options)
    with patch('scenes.workrooms.laundry_room.composition.candidate_metadata', return_value=metadata):
        return build_task_inputs(task_id)


@pytest.mark.parametrize('task_id', ['shirt_stacking', 'trousers_stacking'])
def test_garment_patches_and_entire_nominal_path_obey_frozen_locality(task_id):
    value = inputs(task_id)
    before = deepcopy(value)
    motion = build_motion(value['task'], value['scene'], garment_profile=GARMENT_STACK_PROFILE)
    assert value == before
    evaluator = SequenceEvaluator(value['task'])
    initial = {row['id']: np.asarray(row['initial_vertices_m'])
               for row in value['task']['asset_metadata']['cloths']}
    ids = sum(motion.patches, [])
    assert len(set(ids)) == len(ids) <= .1*len(initial['towel'])
    evaluator._check_local_mask('towel', ids)
    for t in np.linspace(0., motion.duration_s, 101):
        actual = {name: q.copy() for name, q in initial.items()}
        shifts = []
        for index, region in enumerate(motion.patches):
            position, rotation = motion.pose(index, t)
            assert rotation == [1., 0., 0., 0.]
            shift = np.asarray(position)-motion.origins[index]
            actual['towel'][region] += shift
            shifts.append(shift)
        np.testing.assert_allclose(shifts[0], shifts[1], atol=1.e-14)
        evaluator._check_world_grasp_regions(actual, {'towel': ids, 'towel_base': []})
    assert motion.metadata['garment_grasp']['base_controlled_nodes'] == 0
    assert not motion.metadata['success_assumed']
    assert not motion.metadata['target_shape_written']
    for region in motion.patches:
        assert np.ptp(initial['towel'][region, 2]) >= .003


@pytest.mark.parametrize('task_id, expected_dz', [('shirt_stacking', .0096), ('trousers_stacking', .015)])
def test_garment_transport_uses_actual_source_target_and_base_thickness(task_id, expected_dz):
    value = inputs(task_id, source_x=.43)
    motion = build_motion(value['task'], value['scene'], garment_profile=GARMENT_STACK_PROFILE)
    for index in range(2):
        np.testing.assert_allclose(motion.pose(index, 0.)[0], motion.origins[index])
        np.testing.assert_allclose(np.asarray(motion.pose(index, motion.release_time_s)[0])-motion.origins[index],
                                   [.52, 0., expected_dz], atol=1.e-12)
        np.testing.assert_allclose(np.asarray(motion.pose(index, motion.duration_s)[0])-
                                   motion.pose(index, motion.release_time_s)[0], [0., 0., .12])
    assert motion.duration_s == 14.5 and motion.metadata['total_native_steps'] == 29000
    assert motion.metadata['garment_profile'] == GARMENT_STACK_PROFILE


def test_garment_options_are_explicit_and_do_not_change_towel_defaults():
    value = inputs('shirt_stacking')
    for kwargs in ({}, {'garment_profile': 'unknown'},
                   {'garment_profile': GARMENT_STACK_PROFILE, 'stack_grip_span_m': [.04, .04]}):
        with pytest.raises(ValueError):
            build_motion(value['task'], value['scene'], **kwargs)
    towel = build_task_inputs('towel_stacking')
    ordinary = build_motion(towel['task'], towel['scene'])
    explicit = build_motion(towel['task'], towel['scene'], garment_profile=None)
    assert ordinary.metadata == explicit.metadata
    for t in (0., 2., 7.5, 8.5, 14.5):
        for index in range(2):
            assert ordinary.pose(index, t) == explicit.pose(index, t)
    with pytest.raises(ValueError, match='requires a garment'):
        build_motion(towel['task'], towel['scene'], garment_profile=GARMENT_STACK_PROFILE)


def test_garment_patch_rejects_nonlocal_material_even_when_world_points_overlap():
    value = inputs('shirt_stacking')
    motion = build_motion(value['task'], value['scene'], garment_profile=GARMENT_STACK_PROFILE)
    spec = value['task']['asset_metadata']['cloths'][0]
    # One selected vertex moves in rest/material coordinates only; the physical
    # clamp still looks local in world coordinates, which is insufficient.
    spec['rest_vertices_m'][motion.patches[0][0]][0] += .2
    with pytest.raises(ValueError, match='material fragment exceeds 75 mm'):
        build_motion(value['task'], value['scene'], garment_profile=GARMENT_STACK_PROFILE)


@pytest.mark.parametrize('passive', [.1, float('nan'), 120., .5001])
def test_garment_passive_interval_must_be_complete_and_on_native_steps(passive):
    value = inputs('trousers_stacking')
    with pytest.raises(ValueError):
        build_motion(value['task'], value['scene'], garment_profile=GARMENT_STACK_PROFILE,
                     passive_after_release_s=passive)
