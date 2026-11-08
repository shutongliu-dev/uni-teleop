from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import load_json
from scripts.validation.laundry_captured_garment_fold import planned_duration
from scripts.validation.laundry_released_joint_plan import (
    OBSERVED_SLEEVE_PROFILE, phase_released_joint_plan, load_released_joint_plan,
    rebind_released_start, plan_observed_released_press)
from scripts.validation.laundry_single_garment_native import pin


def fixture(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return pin(path)
    inputs = {'scene.json': {'original': 'scene'}, 'task.json': {'id': 'shirt_folding'},
              'initial_state.json': {'joint_positions_rad': {'arm': 0., 'hand': 0.}}}
    samples = [{'label': 'actual_released_state_to_original_park' if i < 100 else 'press',
                'time_s': i * .02, 'joint_positions_rad': {'arm': 0., 'hand': 0.}}
               for i in range(151)]
    plan = {'input_pins': {n: save(n, v) for n, v in inputs.items()},
            'source_clearance': {'all_clear': True}, 'attachments': False,
            'cloth_state_writes': 0, 'planned_duration_s': 3., 'samples': samples}
    manifest = {'schema': 'laundry_checkpoint_ordinary_palm_press.v1', 'attachments': False,
                'runtime_cloth_state_writes': 0, 'plan': save('plan.json', plan),
                'independent_cpu_reviews': {'example': save('review.json', {'original': True})}}
    settings = {'profile': 'ordinary_released_press_v1', 'duration_s': 3.,
                'start_transition_s': 2., 'probe_manifest': save('manifest.json', manifest)}
    return settings, inputs, samples


def models(velocity=2.):
    def joint(name):
        return SimpleNamespace(name=name, lower=-1., upper=1., velocity=velocity)
    return {'right': SimpleNamespace(arm=SimpleNamespace(active=[joint('arm')]),
        hand=SimpleNamespace(active=[joint('hand')]),
        hand_posture_prior=SimpleNamespace(violation=lambda *args: False))}


def test_default_duration_and_only_selected_phases_change(tmp_path):
    settings, _, _ = fixture(tmp_path)
    options = {'phases': {'1': {'released_joint_plan': settings}}}
    assert phase_released_joint_plan({}, 0) is None
    assert planned_duration(1, options, 2.) == planned_duration(1, {}, 2.)
    assert planned_duration(2, options, 2.) == planned_duration(2, {}, 2.) + 3.
    assert phase_released_joint_plan(options, 1) == settings


@pytest.mark.parametrize('key,value', [('duration_s', True), ('duration_s', 3.01),
    ('duration_s', float('nan')), ('start_transition_s', 3.),
    ('start_transition_s', 0.), ('profile', 'cloth_playback')])
def test_invalid_explicit_settings_fail_even_in_unused_phase(tmp_path, key, value):
    settings, _, _ = fixture(tmp_path)
    settings[key] = value
    with pytest.raises(ValueError):
        planned_duration(1, {'phases': {'1': {'released_joint_plan': settings}}}, 2.)


def test_content_pins_original_inputs_and_park_endpoint(tmp_path):
    settings, inputs, expected = fixture(tmp_path)
    rows, provenance = load_released_joint_plan(settings, inputs, ['arm', 'hand'])
    assert rows == expected and provenance['probe_manifest'] == settings['probe_manifest']
    changed = deepcopy(inputs); changed['scene.json']['original'] = 'another scene'
    with pytest.raises(ValueError, match='inputs differ'):
        load_released_joint_plan(settings, changed, ['arm', 'hand'])
    (tmp_path / 'review.json').write_text('{}')
    with pytest.raises(ValueError, match='Pinned input changed'):
        load_released_joint_plan(settings, inputs, ['arm', 'hand'])


def test_actual_start_is_smooth_and_later_plan_is_identical(tmp_path):
    settings, _, rows = fixture(tmp_path)
    original = deepcopy(rows)
    actual = {'arm': .2, 'hand': .1}
    rebound, receipt = rebind_released_start(rows, settings, actual, models())
    assert rows == original
    assert rebound[0]['joint_positions_rad'] == actual
    assert rebound[100:] == original[100:]
    assert rebound[50]['joint_positions_rad'] == {'arm': .1, 'hand': .05}
    assert receipt['all_later_references_unchanged']
    assert receipt['attachments'] is False and receipt['cloth_state_writes'] == 0
    assert 0 < receipt['maximum_original_joint_velocity_fraction'] < 1
    # Native callback consumes one complete reference per 20ms, at either dt.
    for dt in (.0005, .002):
        iterator = iter(row['joint_positions_rad'] for row in rebound[:-1])
        commands = [next(iterator) for local in range(round(3. / dt))
                    if local % round(.02 / dt) == 0]
        assert commands == [r['joint_positions_rad'] for r in rebound[:-1]]
        with pytest.raises(StopIteration):next(iterator)


@pytest.mark.parametrize('delta,accepted', [(5.6e-17, True), (1.e-7, False)])
def test_park_identity_allows_only_interpolation_roundoff(tmp_path, delta, accepted):
    settings, inputs, _ = fixture(tmp_path)
    path = tmp_path / 'plan.json'
    plan = load_json(path)
    plan['samples'][100]['joint_positions_rad']['arm'] = delta
    path.write_text(json.dumps(plan))
    manifest_path = tmp_path / 'manifest.json'
    manifest = load_json(manifest_path); manifest['plan'] = pin(path)
    manifest_path.write_text(json.dumps(manifest)); settings['probe_manifest'] = pin(manifest_path)
    if accepted:
        rows, _ = load_released_joint_plan(settings, inputs, ['arm', 'hand'])
        assert rows[100]['joint_positions_rad']['arm'] == delta
    else:
        with pytest.raises(ValueError, match='original park'):
            load_released_joint_plan(settings, inputs, ['arm', 'hand'])


@pytest.mark.parametrize('actual', [{'arm': 2., 'hand': 0.}, {'arm': float('nan'), 'hand': 0.},
                                   {'arm': .1}, {'arm': .1, 'hand': 0., 'extra': 0.}])
def test_invalid_actual_robot_start_fails(tmp_path, actual):
    settings, _, rows = fixture(tmp_path)
    with pytest.raises(ValueError):rebind_released_start(rows, settings, actual, models())


def test_rebound_velocity_and_original_posture_limits_still_apply(tmp_path):
    settings, _, rows = fixture(tmp_path)
    with pytest.raises(ValueError, match='velocity'):
        rebind_released_start(rows, settings, {'arm': .2, 'hand': 0.}, models(.01))
    model = models()
    model['right'].hand_posture_prior.violation = lambda *args: True
    with pytest.raises(ValueError, match='posture prior'):
        rebind_released_start(rows, settings, {'arm': .2, 'hand': 0.}, model)


def test_observed_profile_cannot_use_old_blended_template(tmp_path):
    settings, _, rows = fixture(tmp_path)
    settings.update(profile=OBSERVED_SLEEVE_PROFILE, duration_s=17.)
    assert phase_released_joint_plan({'phases': {'1': {'released_joint_plan': settings}}}, 1) == settings
    with pytest.raises(ValueError, match='replan the complete path'):
        rebind_released_start(rows, settings, {'arm': .2, 'hand': 0.}, models())
    for changed in ({'duration_s': 20.5}, {'start_transition_s': 3.}):
        invalid = {**settings, **changed}
        with pytest.raises(ValueError, match='17-second'):
            phase_released_joint_plan({'phases': {'1': {'released_joint_plan': invalid}}}, 1)


@pytest.mark.parametrize('failure', [None, 'changed_input', 'changed_duration', 'missing_commands'])
def test_observed_execution_binds_fresh_plan_not_recorded_rows(tmp_path, monkeypatch, failure):
    settings, inputs, old_rows = fixture(tmp_path)
    _, provenance = load_released_joint_plan(settings, inputs, ['arm', 'hand'])
    settings.update(profile=OBSERVED_SLEEVE_PROFILE, duration_s=17.)
    actual = {'arm': .2, 'hand': .1}
    cloth = np.array([[0., 0., 1.]])
    masks = {'towel': np.array([0])}
    enabled = [False]
    checked = object()
    generated = {'planned_duration_s': 17., 'source_clearance': {'all_clear': True},
        'attachments': False, 'cloth_state_writes': 0, 'error': None,
        'samples': [{'label': 'fresh', 'time_s': i * .02, 'joint_positions_rad': dict(actual)}
                    for i in range(851)]}

    def planner(task, q, joints, initial, model, export, template, **kwargs):
        assert task == inputs['task.json'] and q is cloth and joints is actual
        assert initial == inputs['initial_state.json']
        assert kwargs['control_masks'] is masks and kwargs['assist_enabled'] is enabled
        assert kwargs['collision_preflight'] is checked
        assert template['samples'] == old_rows
        return generated

    monkeypatch.setitem(sys.modules, 'scripts.validation.laundry_observed_sleeve_press',
                        SimpleNamespace(plan_observed_sleeve_press=planner))
    if failure == 'changed_input':
        inputs = deepcopy(inputs); inputs['scene.json']['original'] = 'different physical scene'
    elif failure == 'changed_duration':
        generated['planned_duration_s'] = 18.
    elif failure == 'missing_commands':
        generated['samples'].pop(200)
    args = (settings, provenance, inputs, actual, models(), {}, cloth, masks, enabled, checked)
    if failure:
        with pytest.raises(ValueError):
            plan_observed_released_press(*args)
    else:
        rows, plan = plan_observed_released_press(*args)
        assert rows == generated['samples'] and plan is generated
        assert all(row['joint_positions_rad'] == actual for row in rows)
        assert len(rows) == 851 and old_rows[0]['joint_positions_rad'] != rows[0]['joint_positions_rad']
