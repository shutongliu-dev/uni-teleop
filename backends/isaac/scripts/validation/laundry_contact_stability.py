"""Bound original cooked contact-surface motion during a held finger command.

This measures sampled rigid-surface geometry relative to the actual wrist. It
does not measure normal force, friction retention, or between-sample contact.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from bench.evaluation.cloth.cloth_robot_collision import apply_transform, transform_xyzw


PROFILE = 'practical_surface_hold_v6'
REFERENCE_SCHEMA = 'laundry_original_cooked_contact_motion.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class CookedContactMotion:
    """Retain every outer vertex of each original distal contact body."""

    def __init__(self, reference, side, body_names):
        expected = {f'{side}_{finger}_{surface}' for finger in ('thumb', 'index')
                    for surface in ('DP', 'elastomer')}
        if (not isinstance(reference, dict) or reference.get('schema') != REFERENCE_SCHEMA
                or reference.get('side') != side
                or reference.get('body_names_sha256') != digest(list(body_names))
                or not isinstance(reference.get('cooked_export_sha256'), str)
                or len(reference['cooked_export_sha256']) != 64
                or set(reference.get('surfaces', {})) != expected
                or not expected <= set(body_names)):
            raise ValueError('Complete matching original cooked distal surfaces required')
        self.export_sha256 = reference['cooked_export_sha256']
        self.reference_sha256 = digest(reference)
        self.body_names = list(body_names)
        self.side = side
        self.wrist = self.body_names.index(side+'_hand_wrist')
        self.vertices = {}
        for name, points in reference['surfaces'].items():
            points = np.asarray(points, float)
            if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4
                    or not np.isfinite(points).all()
                    or np.linalg.matrix_rank(points-points[0]) < 3):
                raise ValueError('Finite complete cooked convex outer vertices required')
            self.vertices[name] = points.copy()
            self.vertices[name].setflags(write=False)

    def positions(self, variant, body_q):
        pair = variant['surface_pair']
        names = {finger: f'{self.side}_{finger}_{pair[finger]}'
                 for finger in ('thumb', 'index')}
        if (variant.get('cooked_export_sha256') != self.export_sha256
                or variant.get('contact_body_names') != names):
            raise ValueError('Contact witness and cooked motion reference differ')
        inverse_wrist = np.linalg.inv(transform_xyzw(body_q[self.wrist]))
        return {name: apply_transform(self.vertices[name], inverse_wrist @
                    transform_xyzw(body_q[self.body_names.index(name)]))
                for name in names.values()}


def surface_window_metrics(history, key):
    """Maximum displacement of corresponding vertices over the full window."""
    first = history[0]['candidates'][key]['contact_surface_local']
    maximum, speed = 0., 0.
    for name in first:
        points = np.asarray([row['candidates'][key]['contact_surface_local'][name]
                             for row in history])
        # Pairwise span also rejects motion between opposite sides of the
        # initial pose; an initial-point radius alone could hide a 4mm sweep.
        maximum = max(maximum, float(np.linalg.norm(
            points[:, None]-points[None, :], axis=-1).max()))
        if len(history) > 1:
            elapsed = np.diff([row['time'] for row in history])
            speed = max(speed, float((np.linalg.norm(np.diff(points, axis=0), axis=-1)
                                      / elapsed[:, None]).max()))
    return {'maximum_contact_surface_hand_span_m': maximum,
            'maximum_sampled_contact_surface_hand_speed_m_s': speed,
            'contact_bodies': sorted(first)}
