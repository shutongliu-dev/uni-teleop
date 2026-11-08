"""Explicit two-stage engineering hold; never a friction-capture certificate.

The original v6 gate supplies every geometric and motion check. Only its dwell
failure may allow a provisional local constraint after 20 ms. A fresh, separate
100 ms window must then confirm the same material under that declared aid.
"""
from __future__ import annotations

from copy import deepcopy
import math

from scripts.validation.laundry_contact_stability import PROFILE as PRACTICAL_PROFILE

PROFILE = 'contact_engaged_assist_v7'
ENGAGEMENT_DURATION_S = .02
CONFIRMATION_TIMEOUT_S = .5


def engagement_constraint_result(result):
    """Recognize only the explicit provisional result, keeping capture false."""
    original = result.get('full_v6_gate') or {}
    return (result.get('schema') == 'laundry_contact_engagement.v1'
            and result.get('capture_profile') == PROFILE
            and result.get('engagement_candidate') is True
            and result.get('capture_candidate') is False
            and result.get('stage') == 'engaged_unconfirmed'
            and result.get('pre_assistance_100ms_passed') is False
            and original.get('capture_profile') == PRACTICAL_PROFILE
            and original.get('capture_candidate') is False
            and original.get('reasons') == ['insufficient_sampled_stable_duration']
            and original.get('continuous_duration_s', -1.) >= ENGAGEMENT_DURATION_S - 1.e-9
            and original.get('vertex_ids') == result.get('vertex_ids'))


class ContactEngagement:
    def __init__(self):
        self.stage = 'waiting_contact'
        self.engaged_time_s = None
        self.vertex_ids = []
        self.confirmation = None
        self.failure_reason = None

    @property
    def confirmed(self):
        return self.stage == 'confirmed_assisted_hold'

    def expired(self, time_s):
        return (self.stage == 'engaged_unconfirmed'
                and float(time_s) > self.engaged_time_s + CONFIRMATION_TIMEOUT_S + 1.e-9)

    def fail(self, reason):
        self.stage = 'failed'
        self.failure_reason = str(reason)

    def observe(self, time_s, gate_result, *, latched, severe_penetration=False):
        t = float(time_s)
        if not math.isfinite(t):
            raise ValueError('Finite native engagement time required')
        if self.stage in ('failed', 'released'):
            raise RuntimeError('A finished provisional hold cannot be re-enabled')
        if self.confirmed:
            raise RuntimeError('Confirmed transport uses the original holding path')
        if (gate_result.get('capture_profile') != PRACTICAL_PROFILE
                or gate_result.get('thresholds', {}).get('minimum_duration_s') != .1):
            raise ValueError('Engagement must use the unchanged full v6 gate')
        reasons = list(gate_result['reasons'])
        ids = list(map(int, gate_result.get('vertex_ids', [])))
        locally_valid = reasons in ([], ['insufficient_sampled_stable_duration'])
        if self.stage == 'engaged_unconfirmed':
            if severe_penetration or not latched:
                self.fail('deep_penetration' if severe_penetration else 'closure_stop_lost')
                return 'fail', self.receipt(gate_result)
            if ids and set(ids) != set(self.vertex_ids):
                self.fail('confirmed_material_differs_from_engaged_patch')
                return 'fail', self.receipt(gate_result)
            first = gate_result.get('evidence', {}).get('first_time_s', -math.inf)
            if (gate_result['capture_candidate'] and locally_valid and ids
                    and first > self.engaged_time_s + 1.e-9
                    and t <= self.engaged_time_s + CONFIRMATION_TIMEOUT_S + 1.e-9):
                self.stage = 'confirmed_assisted_hold'
                self.confirmation = deepcopy(gate_result)
                return 'confirm', self.receipt(gate_result)
            if t >= self.engaged_time_s + CONFIRMATION_TIMEOUT_S - 1.e-9:
                self.fail('assisted_confirmation_timeout')
                return 'fail', self.receipt(gate_result)
            return 'wait', self.receipt(gate_result)
        duration = float(gate_result['continuous_duration_s'])
        if (latched and not severe_penetration and ids
                and not gate_result['capture_candidate']
                and reasons == ['insufficient_sampled_stable_duration']
                and duration >= ENGAGEMENT_DURATION_S - 1.e-9
                and gate_result.get('evidence', {}).get('sample_count', 0) >= 2):
            self.stage = 'engaged_unconfirmed'
            self.engaged_time_s = t
            self.vertex_ids = ids
            return 'engage', self.receipt(gate_result, candidate=True)
        return 'wait', self.receipt(gate_result)

    def receipt(self, gate_result=None, *, candidate=False):
        return {'schema': 'laundry_contact_engagement.v1', 'capture_profile': PROFILE,
                'stage': self.stage, 'capture_candidate': False,
                'engagement_candidate': bool(candidate), 'vertex_ids': list(self.vertex_ids),
                'engaged_time_s': self.engaged_time_s,
                'engagement_duration_s': ENGAGEMENT_DURATION_S,
                'confirmation_timeout_s': CONFIRMATION_TIMEOUT_S,
                'assisted_confirmation_passed': self.confirmed,
                'pre_assistance_100ms_passed': False,
                'friction_grasp_proven': False, 'failure_reason': self.failure_reason,
                'evidence': deepcopy((gate_result or {}).get('evidence', {})),
                'full_v6_gate': deepcopy(gate_result),
                'scope': 'Provisional local wrist holding after sampled20ms actual two-finger contact. '
                         'Only a separate fresh100ms same-material v6 window under this aid permits transport; '
                         'this never qualifies unassisted100ms retention or friction bearing.'}
