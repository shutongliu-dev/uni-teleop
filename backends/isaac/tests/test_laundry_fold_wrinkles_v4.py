"""Folded garments may retain air; historical rules and failure guards stay fixed."""
import numpy as np
import pytest

from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scenes.workrooms.laundry_room.rules import (
    GARMENT_FOLDING_AIR_PROFILE, GARMENT_FOLDING_PROFILE,
    digest, public_rules, thresholds,
)
from test_laundry_fold_air_compression import profile_task
from test_laundry_single_garment_folding import folding_task, operate, finish


@pytest.mark.parametrize('task_id', ['shirt_folding', 'trousers_folding'])
def test_v4_changes_only_the_fold_air_upper_tolerance(task_id):
    old = thresholds(task_id, GARMENT_FOLDING_AIR_PROFILE)
    new = thresholds(task_id, GARMENT_FOLDING_PROFILE)
    assert old['garment_layer_gap_error_m'] == .012
    assert new == {**old, 'garment_layer_gap_error_m': .025}
    conditions = public_rules(task_id)['referee']['success_conditions']
    assert any('25毫米' in x and 'XY' in x and '天然衣片' in x for x in conditions)


@pytest.mark.parametrize('task_id', ['shirt_folding', 'trousers_folding'])
def test_more_fold_air_cannot_excuse_stretching_the_material(task_id):
    task, goal = folding_task(task_id)
    q = goal['towel'].copy()
    q[:, 2] = q[:, 2].min() + (3 if task_id == 'shirt_folding' else 2)*(q[:, 2]-q[:, 2].min())
    # Geometry-only fixture isolates the changed gap decision. Its crease
    # stretch remains invalid and must fail the complete referee.
    new = SequenceEvaluator(task)
    old = SequenceEvaluator(profile_task(task, GARMENT_FOLDING_AIR_PROFILE))
    assert new._garment_fold({'towel': q})[0]
    assert not old._garment_fold({'towel': q})[0]
    assert new._mesh_metrics('towel', q)['max_edge_stretch_ratio'] > new.limits['max_edge_stretch_ratio']
    assert not finish(task, operate(task, {'towel': q}), {'towel': q})['success']


@pytest.mark.parametrize('task_id', ['shirt_folding', 'trousers_folding'])
@pytest.mark.parametrize('failure', ['unfolded', 'collapsed', 'high_bulge', 'wrong_outer_material'])
def test_v4_rejects_unfolded_collapse_bulges_and_wrong_material_fold(task_id, failure):
    task, goal = folding_task(task_id)
    q = goal['towel'].copy()
    if failure == 'unfolded':
        q = np.asarray(task['asset_metadata']['cloths'][0]['initial_vertices_m']).copy()
    elif failure == 'collapsed':
        q[:, 2] = q[:, 2].min()
    elif failure == 'high_bulge':
        center = q[:, :2].mean(axis=0)
        q[:, 2] += .10*np.exp(-np.sum((q[:, :2]-center)**2, axis=1)/.06**2)
    else:
        outer = np.flatnonzero(q[:, 0] <= np.quantile(q[:, 0], .40))
        q[outer, 1] += .12
    result = finish(task, operate(task, {'towel': q}), {'towel': q})
    assert not result['success']
    assert result['status'] != 'unknown'


@pytest.mark.parametrize('task_id,mode,profile,expected', [
    ('shirt_folding', 'engineering_local_grasp', 'v1', '7cb97594a6802ca82f6233a75099535946c7c3fddcf0e093cd489a0cb2eeb843'),
    ('shirt_folding', 'robot_joint_targets', 'v1', 'eca7a7fe5dddbc06c2f0e3b9af415d78a9cf54baf4491b7ec172ab4e0e8eed4a'),
    ('trousers_folding', 'engineering_local_grasp', 'v1', 'b9a27a9078c166dc50b514bdcd12b9ec8bb558cbc5899756100d0093078f6460'),
    ('trousers_folding', 'robot_joint_targets', 'v1', '8eda685464aef21ed916beaac848f92fd3b07fc558ad42804df22bf89d312ed5'),
    ('shirt_folding', 'engineering_local_grasp', 'v3', '3c92b2b284e82c4aa5885bbc5522df81bfc4a195188ea9234416118d40df12a0'),
    ('shirt_folding', 'robot_joint_targets', 'v3', '9d6df8b6941301916ae45db5813cc930d54e54bace6e278489ab71320bd586f4'),
    ('trousers_folding', 'engineering_local_grasp', 'v3', 'c62b72c73d350a5f176fb843e293877141a7bb6dd0e6d4728080babb22f0f5e0'),
    ('trousers_folding', 'robot_joint_targets', 'v3', 'd67eb711e854d2e9035475f8bd5270978ea4fa2b6c653f9a4168bddb052d3ddc'),
])
def test_v1_and_v3_public_contract_digests_are_preserved(task_id, mode, profile, expected):
    assert digest(public_rules(task_id, profile='laundry_single_garment_folding_task.'+profile,
                               control_mode=mode)) == expected
