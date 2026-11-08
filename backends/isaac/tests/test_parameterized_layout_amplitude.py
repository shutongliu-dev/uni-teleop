"""Deterministic reduced-amplitude layout fallback of the parameter sampler."""
import hashlib
import math

import pytest

from scenes.workrooms.initial_randomization import TASK_FAMILIES
from scenes.workrooms.parameterized.spec import (CONFIGURATIONS, LAYOUT_AMPLITUDE_SCALES, canonical_hash,
                                                 sample_parameters, validate_parameters)

# Living and laundry tasks call the same sampler with their own family identity.
EXTRA_TASKS = [('remote_operation', 'living_room'), ('shirt_folding', 'laundry_room'), ('storage_box', 'living_room'),
               ('towel_flattening', 'laundry_room'), ('towel_folding', 'laundry_room'),
               ('towel_stacking', 'laundry_room'), ('trousers_folding', 'laundry_room'),
               ('water_pouring', 'living_room')]
# instance_sha256 digest of the sampler before the amplitude scale existed (commit 41754b0b).
PRE_CHANGE_DIGEST = 'f54deb5a2e4312a3a3cba57ed4198391205e5f64044c71e5ab9bab78dddf7b16'


def _digest(**kwargs):
    h, count = hashlib.sha256(), 0
    for task_id, family in sorted(TASK_FAMILIES.items()) + EXTRA_TASKS:
        for configuration in CONFIGURATIONS:
            for seed in range(10):
                for attempt in ((0,) if seed == 0 else (0, 1, 2, 7, 13, 48, 240, 9999)):
                    h.update(sample_parameters(family, task_id, configuration, seed, attempt, **kwargs)
                             ['instance_sha256'].encode())
                    count += 1
    return count, h.hexdigest()


def test_default_scale_is_byte_identical_for_every_task_seed_and_attempt():
    assert _digest() == (8103, PRE_CHANGE_DIGEST)
    assert _digest(layout_amplitude_scale=1.) == (8103, PRE_CHANGE_DIGEST)
    record = sample_parameters('laboratory', 'sample_sorting', 'single_arm', 1, 13)
    assert 'layout_amplitude_scale' not in record and validate_parameters(record)


@pytest.mark.parametrize('scale', [.75, .5, .25])
def test_scale_multiplies_only_translation_and_yaw(scale):
    full = sample_parameters('laboratory', 'sample_sorting', 'dual_opposed', 1, 13)
    reduced = sample_parameters('laboratory', 'sample_sorting', 'dual_opposed', 1, 13, layout_amplitude_scale=scale)
    assert reduced['layout_amplitude_scale'] == scale and validate_parameters(reduced)
    assert math.hypot(*reduced['layout']['translation_xy_m']) == pytest.approx(
        scale*math.hypot(*full['layout']['translation_xy_m']), abs=2e-12)
    for axis in range(2):
        assert reduced['layout']['translation_xy_m'][axis] == pytest.approx(
            scale*full['layout']['translation_xy_m'][axis], abs=2e-12)
    assert reduced['layout']['yaw_rad'] == pytest.approx(scale*full['layout']['yaw_rad'], abs=2e-12)
    for field in ('appearance', 'physics', 'optional_physics', 'parameter_specs', 'stream_sha256',
                  'family', 'task_id', 'configuration_id', 'seed', 'attempt', 'canonical', 'status'):
        assert reduced[field] == full[field]
    assert reduced['instance_sha256'] != full['instance_sha256']


def test_recorded_scale_is_part_of_the_validated_identity():
    reduced = sample_parameters('mechanical_assembly', 'shaft_spacer_assembly', 'single_arm', 3, 7,
                                layout_amplitude_scale=.5)
    for forged in ({**reduced, 'layout_amplitude_scale': .25},
                   {key: value for key, value in reduced.items() if key != 'layout_amplitude_scale'},
                   {**sample_parameters('mechanical_assembly', 'shaft_spacer_assembly', 'single_arm', 3, 7),
                    'layout_amplitude_scale': 1.}):
        forged = dict(forged)
        forged['instance_sha256'] = canonical_hash({k: v for k, v in forged.items() if k != 'instance_sha256'})
        with pytest.raises(ValueError, match='differ'):
            validate_parameters(forged)


def test_only_declared_scales_and_never_for_the_canonical_seed():
    assert LAYOUT_AMPLITUDE_SCALES == (1., .75, .5, .25)
    for scale in (.6, 0., 1.5, 1, -.5, float('nan')):
        with pytest.raises(ValueError, match='declared fallback'):
            sample_parameters('kitchen', 'pantry_stocking', 'single_arm', 2, 3, layout_amplitude_scale=scale)
    with pytest.raises(ValueError, match='Canonical seed 0'):
        sample_parameters('kitchen', 'pantry_stocking', 'single_arm', 0, layout_amplitude_scale=.5)
    canonical = sample_parameters('kitchen', 'pantry_stocking', 'single_arm', 0, layout_amplitude_scale=1.)
    assert canonical == sample_parameters('kitchen', 'pantry_stocking', 'single_arm', 0)


def test_study_evaluator_spec_keeps_the_published_frozen_bytes():
    """Published study evaluators embed the pre-fallback sampler; recomposition must reproduce it."""
    from pathlib import Path
    from scenes.workrooms.parameterized import study_evaluator_spec
    path = Path(study_evaluator_spec.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        'e4c3c50ddf035a57393603a619e908e2883d572aaad9f0b99b6d30f51287ab8a')
    for task_id, family in sorted(TASK_FAMILIES.items()) + EXTRA_TASKS:
        for seed, attempt in ((0, 0), (3, 2), (9, 13)):
            assert study_evaluator_spec.sample_parameters(family, task_id, 'dual_opposed', seed, attempt) == \
                sample_parameters(family, task_id, 'dual_opposed', seed, attempt)
