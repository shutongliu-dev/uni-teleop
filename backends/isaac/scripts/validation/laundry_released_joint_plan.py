"""Explicit ordinary joint motions between completely released cloth phases.

The observed robot start is blended to the recorded park endpoint. Subsequent
references are unchanged. This is a new physical execution, never cloth replay
or inherited qualification of the adapted approach.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from scripts.validation.laundry_engineering_motion import minimum_jerk
from scripts.validation.laundry_physical_press_probe import command_rows, read_pinned

OBSERVED_SLEEVE_PROFILE = 'observed_sleeve_vertical_press_v1'


def phase_released_joint_plan(options, index):
    value = options.get('phases', {}).get(str(index), {}).get('released_joint_plan')
    if value is None:
        return None
    if (not isinstance(value, dict)
            or set(value) != {'profile', 'probe_manifest', 'duration_s', 'start_transition_s'}
            or value['profile'] not in ('ordinary_released_press_v1', OBSERVED_SLEEVE_PROFILE)):
        raise ValueError('Explicit released joint plan requires its profile and pinned manifest')
    pin = value['probe_manifest']
    if (not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}
            or not isinstance(pin['path'], str) or not Path(pin['path']).is_absolute()
            or not isinstance(pin['sha256'], str) or len(pin['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in pin['sha256'])):
        raise ValueError('Released motion requires an absolute content-pinned input')
    for key, bounds in [('duration_s', (1., 30.)), ('start_transition_s', (1., 5.))]:
        v = value[key]
        if (type(v) not in (int, float) or not np.isfinite(v) or not bounds[0] <= v <= bounds[1]
                or abs(v / .02 - round(v / .02)) > 1.e-9):
            raise ValueError('Released motion requires finite complete 20ms intervals')
    if value['start_transition_s'] >= value['duration_s']:
        raise ValueError('The released start transition must precede the rest of the motion')
    if value['profile'] == OBSERVED_SLEEVE_PROFILE and (
            value['duration_s'] != 17. or value['start_transition_s'] != 2.):
        raise ValueError('Observed sleeve pressing retains its reviewed 17-second vertical sequence')
    return deepcopy(value)


def load_released_joint_plan(settings, inputs, joint_names):
    manifest = read_pinned(settings['probe_manifest'])
    if (manifest.get('schema') != 'laundry_checkpoint_ordinary_palm_press.v1'
            or manifest.get('attachments') is not False or manifest.get('runtime_cloth_state_writes') != 0):
        raise ValueError('Only explicitly ordinary robot-only probe plans are accepted')
    plan = read_pinned(manifest['plan'])
    for reference in manifest['independent_cpu_reviews'].values():
        read_pinned(reference)
    # The probe authored a diagnostic reset; the full fold retains the original
    # release. Compare the plan's ORIGINAL inputs, never its derived initial q.
    for name in ('scene.json', 'task.json', 'initial_state.json'):
        if read_pinned(plan['input_pins'][name]) != inputs[name]:
            raise ValueError('Released motion physical/task inputs differ from this full run')
    if plan['planned_duration_s'] != settings['duration_s']:
        raise ValueError('Released motion duration differs from its pinned plan')
    rows = command_rows(plan, joint_names)
    transition = round(settings['start_transition_s'] / .02)
    park = inputs['initial_state.json']['joint_positions_rad']
    # Only the first explicitly marked park segment may be rebound to live q.
    if (any(r['label'] != 'actual_released_state_to_original_park' for r in rows[:transition])
            or set(rows[transition]['joint_positions_rad']) != set(park)
            or any(abs(rows[transition]['joint_positions_rad'][n] - park[n]) > 1.e-12 for n in park)):
        raise ValueError('Released motion must first reach the original park')
    return rows, {'probe_manifest': settings['probe_manifest'], 'plan': manifest['plan'],
                  'original_cpu_reviews': manifest['independent_cpu_reviews']}


def plan_observed_released_press(settings, provenance, inputs, actual_joints, models,
                                export, cloth_q, control_masks, assist_enabled,
                                collision_preflight):
    """Plan from fresh observations; the template supplies no new path credit."""
    if settings['profile'] != OBSERVED_SLEEVE_PROFILE:
        raise ValueError('Observed planning requires its explicit released-motion profile')
    from scripts.validation.laundry_observed_sleeve_press import plan_observed_sleeve_press
    template = read_pinned(provenance['plan'])
    for name in ('scene.json', 'task.json', 'initial_state.json'):
        if read_pinned(template['input_pins'][name]) != inputs[name]:
            raise ValueError('Observed press template differs from this full run')
    plan = plan_observed_sleeve_press(inputs['task.json'], cloth_q, actual_joints,
        inputs['initial_state.json'], models, export, template,
        control_masks=control_masks, assist_enabled=assist_enabled,
        collision_preflight=collision_preflight)
    if plan['planned_duration_s'] != settings['duration_s']:
        raise ValueError('Fresh observed press changed its declared duration')
    rows = command_rows(plan, list(actual_joints))
    return rows, plan


def rebind_released_start(rows, settings, actual_joints, models):
    if settings['profile'] != 'ordinary_released_press_v1':
        raise ValueError('Observed sleeve pressing must replan the complete path from actual cloth')
    names = set(rows[0]['joint_positions_rad'])
    if (set(actual_joints) != names or not np.isfinite(list(actual_joints.values())).all()):
        raise ValueError('Fresh actual joints must cover the full original robot')
    result = deepcopy(rows)
    end_index = round(settings['start_transition_s'] / .02)
    end = rows[end_index]['joint_positions_rad']
    for i in range(end_index):
        u = minimum_jerk(i * .02, 0., settings['start_transition_s'])
        result[i]['joint_positions_rad'] = {n: float(actual_joints[n] + u * (end[n] - actual_joints[n]))
                                            for n in actual_joints}
    joints = {j.name: j for model in models.values() for chain in (model.arm, model.hand)
              for j in chain.active}
    if set(joints) != names:
        raise ValueError('Ordinary plan joint inventory differs from original robot geometry')
    previous = None
    peak_velocity_fraction = 0.
    for row in result:
        q = row['joint_positions_rad']
        for name, joint in joints.items():
            if not joint.lower <= q[name] <= joint.upper:
                raise ValueError('Rebound released motion exceeds original joint limits')
            if previous is not None:
                speed = abs(q[name] - previous[name]) / .02
                if speed > joint.velocity + 1.e-9:
                    raise ValueError('Rebound released motion exceeds original joint velocity')
                if joint.velocity > 0:
                    peak_velocity_fraction = max(peak_velocity_fraction, speed / joint.velocity)
        if any(model.hand_posture_prior.violation(finger, q)
               for model in models.values() for finger in ('thumb', 'index', 'middle', 'ring', 'little')):
            raise ValueError('Released motion exceeds original hand posture prior')
        previous = q
    return result, {'actual_start_joint_positions_rad': dict(actual_joints),
        'original_plan_start_joint_positions_rad': rows[0]['joint_positions_rad'],
        'rebound_control_intervals': end_index,
        'maximum_original_joint_velocity_fraction': peak_velocity_fraction,
        'all_later_references_unchanged': result[end_index:] == rows[end_index:],
        'attachments': False, 'cloth_state_writes': 0,
        'scope': 'Fresh ordinary-joint execution with live start rebound to original park. '
                 'Original CPU collision qualification does not cover this changed approach; '
                 'actual collisions, cloth response and the full task require new evidence.'}
