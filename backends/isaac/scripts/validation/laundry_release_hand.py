"""Explicit original-joint release posture for captured-cloth diagnostics.

This changes position targets only. The cooked clearance planner and the live
opening use the same held-command-to-open interpolation. It does not establish
cloth separation or modify the original approach/closing hand presets.
"""
from __future__ import annotations

import numpy as np


def release_hand_profile(options):
    profile = options.get('release_hand_profile', 'pinch_open_v1')
    if profile not in ('pinch_open_v1', 'distal_extend_v1', 'full_spread_extend_v2'):
        raise ValueError('Unknown explicit release hand profile')
    if (profile in ('distal_extend_v1', 'full_spread_extend_v2')
            and options.get('release_profile', 'fixed_v1') not in ('cooked_hand_clearance_v2', 'cooked_opening_lift_v3')):
        raise ValueError('Distal extension requires the actual cooked opening clearance profile')
    if (options.get('release_profile') == 'cooked_opening_lift_v3'
            and profile != 'full_spread_extend_v2'):
        raise ValueError('Dynamic cooked opening requires the complete full-spread release')
    return profile


def distal_release_target(side, postures):
    if side not in ('left', 'right'):
        raise ValueError('An original hand side is required')
    opened = dict(postures[0]['hand_posture_rad'])
    overrides = {side+'_index_PIP': .1, side+'_index_DIP': .1, side+'_thumb_IP': .05}
    if (not set(overrides).issubset(opened) or
            any(not n.startswith(side+'_') or not np.isfinite(v) for n, v in opened.items())):
        raise ValueError('The complete original open hand posture must include all three distal joints')
    opened.update(overrides)
    return opened


def release_target(side, postures, profile, *, model=None):
    """Separate a full release opening from the chosen capture aperture."""
    if profile == 'distal_extend_v1':
        return distal_release_target(side, postures)
    if profile != 'full_spread_extend_v2' or model is None:
        raise ValueError('Full-spread release requires the original hand model')
    from scripts.validation.laundry_contact_pinch import pinch_postures
    # The first original inverse-kinematic solution is independent of the
    # closing samples. Do not reuse a narrow capture's MCP/CMC aperture here.
    _, full_open = pinch_postures(model, side, open_distance=.060,
                                  close_distance=.010, samples=2)
    opened = dict(postures[0]['hand_posture_rad'])
    if set(opened) != set(full_open[0]['hand_posture_rad']):
        raise ValueError('Full-spread release requires the complete original joint inventory')
    for name, value in full_open[0]['hand_posture_rad'].items():
        if name.startswith((side+'_thumb_', side+'_index_')):
            opened[name] = value
    return distal_release_target(side, [{'hand_posture_rad': opened}])


def interpolate_release_target(start_targets, opened, amount):
    if (not np.isfinite(amount) or not 0. <= amount <= 1. or
            any(n not in start_targets or not np.isfinite([start_targets[n], v]).all()
                for n, v in opened.items())):
        raise ValueError('Finite held/open original targets and opening fraction in [0, 1] required')
    return {n: float(start_targets[n]+amount*(v-start_targets[n])) for n, v in opened.items()}


def opening_postures(assist, postures, start_targets, profile, *, model=None):
    if profile == 'pinch_open_v1':
        return assist.opening_postures(postures, start_targets)
    if profile not in ('distal_extend_v1', 'full_spread_extend_v2'):
        raise ValueError('Unknown explicit release hand profile')
    opened = release_target(assist.side, postures, profile, model=model)
    sweep = [{'hand_posture_rad': interpolate_release_target(start_targets, opened, u)}
             for u in np.linspace(0., 1., 61)]
    if profile == 'full_spread_extend_v2':
        for row in sweep:
            posture = row['hand_posture_rad']
            if (any(not j.lower <= posture[j.name] <= j.upper for j in model.hand.active)
                    or any(model.hand_posture_prior.violation(f, posture) for f in ('thumb', 'index'))):
                raise ValueError('Full-spread release sweep violates the original hand bounds or posture prior')
    return sweep


def retain_released_targets(planned_targets, released_targets):
    """Keep only the explicitly released hands extended through exit and retreat."""
    if any(n not in planned_targets or not np.isfinite(v) for n, v in released_targets.items()):
        raise ValueError('Released hand targets must remain in the original target inventory')
    return {**planned_targets, **released_targets}
