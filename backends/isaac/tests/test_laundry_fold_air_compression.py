"""Reference air gaps are optional; physical wall thickness/order stay required."""
from copy import deepcopy

import numpy as np
import pytest

from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scenes.workrooms.laundry_room.rules import (
    GARMENT_FOLDING_SCALED_PROFILE, GARMENT_FOLDING_PROFILE, GARMENT_FOLDING_AIR_PROFILE,
    bind_task_rules, digest, public_rules, thresholds,
)
from test_laundry_single_garment_folding import folding_task, operate, finish


def profile_task(task, profile):
    task = deepcopy(task)
    task['public_rules'] = public_rules(task['id'], profile=profile,
                                      control_mode=task['evaluation']['control_mode'])
    task['evaluation'].update(rule_profile=profile, thresholds=thresholds(task['id'], profile),
                              public_rules_sha256=digest(task['public_rules']))
    return bind_task_rules(task)


def airy_reference():
    task, compact = folding_task()
    # Deliberately generous, synthetic reference bend air. Actual compact q
    # remains the same valid material fold; no physics state is manufactured.
    meta = task['asset_metadata']
    spec = meta['cloths'][0]
    q = np.asarray(spec['folded_reference_vertices_m'])
    q[:, 2] = q[:, 2].min() + 4*(q[:, 2]-q[:, 2].min())
    spec['folded_reference_vertices_m'] = q.tolist()
    task['evaluation']['metadata'] = deepcopy(meta)
    return profile_task(task, GARMENT_FOLDING_AIR_PROFILE), compact


def test_v3_accepts_natural_air_compression_but_v2_remains_unchanged():
    task, compact = airy_reference()
    assert finish(task, operate(task, compact), compact)['success']
    old = profile_task(task, GARMENT_FOLDING_SCALED_PROFILE)
    previous = finish(old, operate(old, compact), compact)
    assert not previous['success']
    assert previous['detail']['geometry']['towel']['ordered_layer_fraction'] < .85
    assert task['evaluation']['thresholds'] == old['evaluation']['thresholds']


@pytest.mark.parametrize('mode', ['zero_thickness', 'reverse_layers', 'excessive_air'])
def test_v3_still_rejects_collapsed_reversed_or_widely_separated_layers(mode):
    task, compact = airy_reference()
    q = compact['towel'].copy()
    bottom = q[:, 2].min()
    scale = {'zero_thickness': 0., 'reverse_layers': -1., 'excessive_air': 8.}[mode]
    q[:, 2] = bottom+scale*(q[:, 2]-bottom)
    e = SequenceEvaluator(task)
    passed, detail = e._garment_fold({'towel': q})
    assert not passed and detail['towel']['ordered_layer_fraction'] < .85


@pytest.mark.parametrize('task_id,mode,expected', [
    ('shirt_folding', 'engineering_local_grasp', 'cb03ec53b086dbc6a4248005ffd78686c1564007a22a9dcc4a16ec081c622257'),
    ('shirt_folding', 'robot_joint_targets', '466e3bbfbe7b823e415fc029db3b0ef2757ff2109a49acb253d0fbf1baed8b95'),
    ('trousers_folding', 'engineering_local_grasp', '80d29ef8d5b039f30fd84915053632cd9399166e949ad336101eb4ebd8364a09'),
    ('trousers_folding', 'robot_joint_targets', '7ffe740996c37c73853de2ef0a631a93cc60dba13ae633c42f4b563ba122d332'),
])
def test_frozen_v2_public_rules_remain_byte_equivalent(task_id, mode, expected):
    assert digest(public_rules(task_id, profile=GARMENT_FOLDING_SCALED_PROFILE, control_mode=mode)) == expected


def test_selected_release_keeps_its_explicit_rule_version():
    from scenes.workrooms.laundry_room.catalog_definition import task_definition
    from scenes.workrooms.catalog import load_task
    from agent.model_client.scene_tasks import select_task
    for task_id in ('shirt_folding', 'trousers_folding'):
        definition = task_definition(task_id)
        frozen = load_task('laundry_room', task_id)['task']
        assert definition['rules']['contract']['rule_profile'] == frozen['evaluation']['rule_profile']
        select_task(task_id, family='laundry_room').require_runnable()
    assert GARMENT_FOLDING_PROFILE != GARMENT_FOLDING_SCALED_PROFILE
