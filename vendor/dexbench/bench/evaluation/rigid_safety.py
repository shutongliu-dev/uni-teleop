"""Online guard for the existing solver-reported rigid penetration gate.

This inspects completed native windows; it never steps physics or changes drives.
The 1 mm task settlement tolerance is a separate scoring condition.
"""
from copy import deepcopy
import math


RIGID_CONTACT_DIAGNOSTIC_M = .002


def exceeds_rigid_penetration(separation):
    return separation < -RIGID_CONTACT_DIAGNOSTIC_M


class RigidContactSafetyGuard:
    def __init__(self, *, required=True):
        self.required = required
        self.trigger = None

    def assert_can_step(self):
        if self.trigger is not None:
            raise ValueError('Native rigid-contact safety stop is latched; further steps are forbidden')

    def observe(self, observation):
        if self.trigger is not None:
            return deepcopy(self.trigger)
        contacts = observation.get('contacts', {})
        sampling = contacts.get('sampling')
        # Historical/non-native worker doubles did not publish native receipts.
        # Current native workers must always provide a complete receipt.
        if sampling is None and not self.required:
            return None

        def number(value):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Invalid native rigid-contact safety receipt number')
            return value

        if (not isinstance(sampling, dict) or sampling.get('complete') is not True
                or sampling.get('protocol') != 'native_rigid_contact_sampling_v1'
                or contacts.get('source') != 'PhysX stage-bound rigid contact tensor; normal impulses'
                or contacts.get('errors') != []):
            raise ValueError('Native rigid-contact safety guard requires complete sampling')
        start, end, native_start, native_end, dt = [number(sampling.get(k)) for k in
            ('start_sim_time_s', 'end_sim_time_s', 'native_start_time_s', 'native_end_time_s', 'physics_dt_s')]
        first, last, count = [sampling.get(k) for k in
            ('native_start_step_index', 'native_end_step_index', 'sampled_substeps')]
        if (dt <= 0 or not all(type(v) is int and v >= 0 for v in (first, last, count))
                or count == 0 or last - first != count or abs(end - start - .02) > 1e-8
                or abs(native_end - native_start - .02) > 1e-8 or abs(count * dt - .02) > 1e-8
                or abs(number(observation.get('sim_time_s')) - end) > 1e-8):
            raise ValueError('Native rigid-contact safety receipt does not cover one complete control window')
        rows = contacts.get('rigid')
        if not isinstance(rows, list):
            raise ValueError('Native rigid-contact safety receipt lacks contact rows')
        violations = []
        for index, row in enumerate(rows):
            stamp, separation = number(row.get('time_s')), number(row.get('min_separation_m'))
            tick = round((stamp - start) / dt)
            if not 1 <= tick <= count or abs(stamp - start - tick * dt) > 1e-8:
                raise ValueError('Native rigid contact is outside its control window')
            if exceeds_rigid_penetration(separation):
                violations.append((stamp, index, row))
        if not violations:
            return None
        stamp, index, row = min(violations, key=lambda item: (item[0], item[1]))
        self.trigger = {'schema': 'native_rigid_safety_stop.v1',
            'reason': 'reported_rigid_penetration', 'threshold_m': RIGID_CONTACT_DIAGNOSTIC_M,
            'contact_time_s': stamp, 'window_start_sim_time_s': start, 'window_end_sim_time_s': end,
            'native_start_step_index': first, 'native_end_step_index': last,
            'contact_row_index': index, 'reported_pair_rows': len(rows), 'violating_pair_rows': len(violations),
            'first_separation_m': row['min_separation_m'],
            'minimum_separation_m': min(item[2]['min_separation_m'] for item in violations),
            'actor0': str(row.get('actor0', ''))[:512], 'actor1': str(row.get('actor1', ''))[:512],
            'scope': 'Solver-reported rigid pairs; completed window only, not independent mesh clearance.'}
        return deepcopy(self.trigger)
