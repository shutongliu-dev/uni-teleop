"""Read-only arrival receipts for contact-limited privileged hand phases."""
import math


DIGITS = ('thumb', 'index', 'middle', 'ring', 'pinky')
CONTACT_WINDOW_S = .1
CONTACT_BLOCK_S = .02
MIN_CONTACT_IMPULSE_NS = 1.e-8


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _digit(name):
    matches = {digit for component in name.split('/') for digit in DIGITS
               if component.startswith('right_'+digit+'_')}
    return next(iter(matches)) if len(matches) == 1 else None


def _within(actor, prefix):
    return actor == prefix or actor.startswith(prefix+'/')


def _contact_window(phase, native_rows, dt, native_step, robot_actor_paths):
    expected_samples = round(CONTACT_WINDOW_S/dt)
    block_samples = round(CONTACT_BLOCK_S/dt)
    rows = list(native_rows[-expected_samples:]) if native_rows is not None else []
    prefix = phase.get('hand_contact_actor_prefix')
    valid_prefix = (isinstance(prefix, str) and prefix.startswith('/World/Objects/')
                    and all(component and component not in ('.', '..')
                            for component in prefix.split('/')[1:]))
    complete = bool(valid_prefix and len(rows) == expected_samples
                    and type(native_step) is int)
    previous_time = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            complete = False
            break
        time_s = row.get('time_s')
        contacts = row.get('raw_contacts')
        if (row.get('phase') != phase['id'] or type(row.get('native_step')) is not int
                or type(native_step) is not int
                or row['native_step'] != native_step-len(rows)+index+1
                or not _finite_number(time_s)
                or (previous_time is not None and abs(time_s-previous_time-dt) > 1.e-8)
                or not isinstance(contacts, list)):
            complete = False
            break
        previous_time = time_s
        if any(not isinstance(contact, dict)
               or not isinstance(contact.get('actor0'), str)
               or not isinstance(contact.get('actor1'), str)
               or not _finite_number(contact.get('impulse_magnitude_sum_ns'))
               or contact['impulse_magnitude_sum_ns'] < 0.
               for contact in contacts):
            complete = False
            break
    block_count = round(CONTACT_WINDOW_S/CONTACT_BLOCK_S)
    receipts = {digit: {'passed': False,
                       'positive_contact_substeps_per_block': [0]*block_count,
                       'impulse_sum_ns_per_block': [0.]*block_count,
                       'robot_actor_paths': []}
                for digit in DIGITS}
    robot_paths = None if robot_actor_paths is None else frozenset(robot_actor_paths)
    if complete:
        for index, row in enumerate(rows):
            block_index = index//block_samples
            positive_digits = set()
            for contact in row['raw_contacts']:
                impulse = contact['impulse_magnitude_sum_ns']
                if impulse <= MIN_CONTACT_IMPULSE_NS:
                    continue
                for robot_actor, object_actor in ((contact['actor0'], contact['actor1']),
                                                  (contact['actor1'], contact['actor0'])):
                    if (not _within(object_actor, prefix)
                            or not robot_actor.startswith('/World/Robot/')
                            or (robot_paths is not None and robot_actor not in robot_paths)):
                        continue
                    digit = _digit(robot_actor)
                    if digit is None:
                        continue
                    receipt = receipts[digit]
                    positive_digits.add(digit)
                    receipt['impulse_sum_ns_per_block'][block_index] += impulse
                    if robot_actor not in receipt['robot_actor_paths']:
                        receipt['robot_actor_paths'].append(robot_actor)
            for digit in positive_digits:
                receipts[digit]['positive_contact_substeps_per_block'][block_index] += 1
        for receipt in receipts.values():
            receipt['passed'] = all(receipt['positive_contact_substeps_per_block'])
            receipt['robot_actor_paths'].sort()
    return receipts, {
        'complete': complete, 'duration_s': CONTACT_WINDOW_S,
        'block_duration_s': CONTACT_BLOCK_S, 'block_count': block_count,
        'expected_native_samples': expected_samples, 'observed_native_samples': len(rows),
        'endpoint_native_step': native_step,
        'robot_actor_inventory_checked': robot_paths is not None,
        'minimum_contact_impulse_ns': MIN_CONTACT_IMPULSE_NS,
    }


def assess_phase_endpoint(phase, observed_joint_positions_rad, native_rows, dt, *,
                          native_step, tolerance_rad=.02, robot_actor_paths=None):
    """Waive a finger residual only against its own sustained object contact.

    Arm and unrecognized joints always retain the ordinary endpoint tolerance.
    These receipts describe a blocked hand target, not grasp or task success.
    """
    if (not _finite_number(dt) or dt <= 0. or dt > CONTACT_BLOCK_S
            or abs(CONTACT_BLOCK_S/dt-round(CONTACT_BLOCK_S/dt)) > 1.e-8
            or abs(CONTACT_WINDOW_S/dt-round(CONTACT_WINDOW_S/dt)) > 1.e-8):
        raise ValueError('Endpoint contacts require a complete native 20 ms grid')
    if not _finite_number(tolerance_rad) or not 0. < tolerance_rad <= .02:
        raise ValueError('Endpoint tolerance must be positive and at most 0.02 rad')
    targets = phase['joint_positions_rad']
    if (not isinstance(targets, dict) or not targets
            or any(not isinstance(name, str) or not _finite_number(target)
                   or not _finite_number(observed_joint_positions_rad.get(name))
                   for name, target in targets.items())):
        raise ValueError('Endpoint assessment requires finite commanded and observed joints')
    residuals = {name: abs(observed_joint_positions_rad[name]-target)
                 for name, target in targets.items()}
    receipts, window = _contact_window(phase, native_rows, dt, native_step, robot_actor_paths)
    waived = sorted(name for name, error in residuals.items()
                    if error > tolerance_rad and _digit(name) in receipts
                    and receipts[_digit(name)]['passed'])
    unwaived_maximum = max((error for name, error in residuals.items() if name not in waived),
                          default=0.)
    return {
        'passed': unwaived_maximum <= tolerance_rad,
        'raw_max_error_rad': max(residuals.values()),
        'unwaived_max_error_rad': unwaived_maximum,
        'joint_errors_rad': residuals, 'waived_joint_names': waived,
        'endpoint_tolerance_rad': tolerance_rad,
        'hand_contact_actor_prefix': phase.get('hand_contact_actor_prefix'),
        'digit_receipts': receipts, 'native_contact_window': window,
        'task_success_credit': 0,
        'scope': 'Read-only endpoint/contact receipt; no grasp or task success credit.',
    }
