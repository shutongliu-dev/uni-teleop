"""Material-thickness referee fixtures; no simulated or native grasp claim."""
import numpy as np
import pytest

from bench.protocol.task_referee import PublicContract
from scenes.workrooms.laundry_room.rules import (
    ALL_TASK_IDS, ENGINEERING_CONTROL, GARMENT_LEGACY_PROFILE, GARMENT_PROFILE,
    GARMENT_TASK_IDS, PRACTICAL_PROFILE, ROBOT_CONTROL, digest, public_rules,
    thresholds)
from test_laundry_garment_rules import garment_goal, garment_task, operated, released
from test_laundry_robot_rules import robot_frame


# Captured from the pre-v2 source. Explicit historical selection stays exact.
HISTORICAL_PUBLIC_DIGESTS = {
    'towel_flattening': ('211ece40b2a708cde3ccddd521f253ffbf41fc8c9b3c363f0c3974b9bba3d101',
                         'dca85a4bf2acfc8403e855a354a375c3666fc392771dc54fb62324387690c77d'),
    'towel_folding': ('9317116519c99d78d8f4ed2abcfd52db113d7b3f45d6915472b33e6582b57ceb',
                      'aa550019b750c5fc88255d7f7709026e217e3b8b3ef10c0177ab1de34e956454'),
    'towel_stacking': ('6c5722211c3c738bec37c2ac80baa07f1d8e2d03785a30558a93dad031968a01',
                       '5cd765a92f376b940d79f228052e07156509c50eb3cd4390665a234c37fe08c2'),
    'shirt_stacking': ('e93df93bc7613cff6c844d5da38dc782d07d5fb5729198d566741858e40ed70e',
                       '99337f30cbf0663510ce845d8ee9c02eeacb6e266d94a6817ea06fa8748f5b8d'),
    'trousers_stacking': ('fd00e1c226b220d5328bc48fec9c3624bcf3fecba55430930e54fe7b1673e136',
                          'f5f41f3255ca3e5b6668a72affdf7eb6ee4ec565d98dfeddab59c0e73a8a99c6'),
}


@pytest.mark.parametrize('task_id', tuple(HISTORICAL_PUBLIC_DIGESTS))
@pytest.mark.parametrize('mode_index, mode', list(enumerate((ENGINEERING_CONTROL, ROBOT_CONTROL))))
def test_previous_public_rules_and_new_contracts(task_id, mode_index, mode):
    previous = GARMENT_LEGACY_PROFILE if task_id in GARMENT_TASK_IDS else PRACTICAL_PROFILE
    assert digest(public_rules(task_id, profile=previous, control_mode=mode)) == HISTORICAL_PUBLIC_DIGESTS[task_id][mode_index]
    PublicContract.model_validate(public_rules(task_id, control_mode=mode)['referee'])


def _reposition(goal, thickness, table_z):
    for vertices in goal.values():
        vertices[:, 2] += table_z+thickness/2-vertices[:, 2].min()
    goal['towel'][:, 2] += goal['towel_base'][:, 2].max()+thickness-goal['towel'][:, 2].min()


@pytest.mark.parametrize('task_id', GARMENT_TASK_IDS)
@pytest.mark.parametrize('shape, accepted', [('material_contact', True), ('quarter_thickness', False), ('inverted_layers', False)])
def test_v2_material_contact_and_invalid_layers(task_id, shape, accepted):
    task = garment_task(task_id)
    goal = garment_goal(task)
    spec = task['asset_metadata']['cloths'][0]
    thickness = spec['physical_thickness_m']
    reference = np.array(spec['folded_reference_vertices_m'])
    pairs = np.array(spec['folded_layer_pairs'])
    minimum_reference_gap = np.min(reference[pairs[:, 1], 2]-reference[pairs[:, 0], 2])
    for vertices in goal.values():
        height = vertices[:, 2]-vertices[:, 2].min()
        if shape == 'material_contact':
            # The closest reference layers settle to one material thickness.
            height *= thickness/minimum_reference_gap
        elif shape == 'quarter_thickness':
            # Every material layer lies inside a quarter-thickness envelope.
            height *= .25*thickness/np.ptp(height)
        else:
            height *= -1
        vertices[:, 2] = height
    _reposition(goal, thickness, task['asset_metadata']['table_top_z_m'])
    scorer = operated(task, goal)
    result = released(task, scorer, goal)
    assert result['success'] is accepted, result
    assert (result['detail']['geometry']['towel']['ordered_layer_fraction'] >= .85) is accepted


@pytest.mark.parametrize('profile, accepted', [(GARMENT_LEGACY_PROFILE, False), (GARMENT_PROFILE, True)])
def test_shirt_nominal_scale_contact_replaces_reset_air_gap_floor(profile, accepted):
    task = garment_task('shirt_stacking', profile)
    goal = garment_goal(task)
    # Main 8 mm air gap settles to 1.64 mm, near the nominal 1.6 mm.
    # The sleeve region retains 0.82 mm, above the public 0.8 mm minimum.
    for vertices in goal.values():
        vertices[:, 2] = (vertices[:, 2]-vertices[:, 2].min())*.205
    spec = task['asset_metadata']['cloths'][0]
    _reposition(goal, spec['physical_thickness_m'], task['asset_metadata']['table_top_z_m'])
    result = released(task, operated(task, goal), goal)
    assert result['success'] is accepted, result


@pytest.mark.parametrize('task_id', GARMENT_TASK_IDS)
def test_v2_changes_only_the_layer_floor_and_keeps_crossings_recoverable(task_id):
    old = thresholds(task_id, GARMENT_LEGACY_PROFILE)
    current = thresholds(task_id, GARMENT_PROFILE)
    assert old.pop('garment_layer_gap_ratio_min') == .25
    assert current.pop('garment_layer_thickness_ratio_min') == .5
    assert current == old
    task = garment_task(task_id)
    goal = garment_goal(task)
    scorer = operated(task, goal)
    for step in range(3, 254):
        frame = robot_frame(task, step, goal)
        frame['physics']['self_intersections'] = 1
        result = scorer.observe(frame)
    assert not result['success']
    assert result['reason'] == 'contact_recovery_required'
