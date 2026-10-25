"""Identity-free, scalar fingertip proxy from recorded native rigid impulses.

This is neither a hardware tactile driver nor a visuotactile sensor simulator.
It projects only whole-elastomer rigid normal loads. No counterpart identity,
contact location, shear, torque, taxel image or cloth force crosses this API.
"""
from __future__ import annotations

import math


PROTOCOL = 'rigid_fingertip_contact_proxy_v1'
MODE = 'simulated_rigid_contact_proxy'
SOURCE = 'PhysX stage-bound rigid contact tensor; normal impulses'
NATIVE_SOURCE = 'stage-bound PhysX RigidContactView.get_contact_data(dt=1)'
FINGER_LINKS = {finger: f'right_{native}_elastomer' for finger, native in (
    ('thumb', 'thumb'), ('index', 'index'), ('middle', 'middle'), ('ring', 'ring'), ('little', 'pinky'))}
ELASTOMER_REGION = 'whole_elastomer_collision_body'
DISTAL_REGION = 'whole_distal_link_collision_body'
NORMAL_TOLERANCE = 32 * 2**-23


def finger_links(robot_id, hand_spec=None, side=None):
    """Sensor link per finger and its region: SHARPA elastomer pads, else the
    registered distal rigid bodies (the whole distal link is one channel). A
    bimanual robot is projected per hand (``side``)."""
    if robot_id == 'sharpa_w01_right' and side is None:
        return dict(FINGER_LINKS), ELASTOMER_REGION
    if robot_id in ('dual_fr3_sharpa_w01', 'fr3_sharpa_instances'):
        sides = ('left', 'right') if robot_id == 'dual_fr3_sharpa_w01' else tuple((hand_spec or {}).get('arms', {}))
        if side not in sides:
            return {}, None
        return {finger: f'{side}_' + link.split('_', 1)[1] for finger, link in FINGER_LINKS.items()}, ELASTOMER_REGION
    if side is not None:
        return {}, None
    if not isinstance(hand_spec, dict) or hand_spec.get('hand_id') != robot_id:
        return {}, None
    links = hand_spec.get('fingertip_links')
    prefix = hand_spec.get('prefix', '')
    if (not isinstance(links, dict) or not isinstance(prefix, str)
            or set(links) not in ({'thumb', 'index', 'middle', 'ring', 'little'}, {'thumb', 'index', 'middle', 'ring'})):
        return {}, None
    return {finger: prefix + link for finger, link in links.items()}, DISTAL_REGION


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def tactile_configuration(metadata, robot_id, side=None):
    """Build a private, hash-pinnable configuration from actual worker metadata.

    Availability is all-or-none for the hand's finger channels (five W01
    elastomer rigid bodies, or each registered hand's distal rigid bodies). The
    returned sensor_paths are private configuration, never public observations.
    Completeness relies on the existing native step contract: absent contacts
    do not independently prove that every physics substep was sampled.
    """
    config = {'protocol': PROTOCOL, 'mode': MODE, 'available': False, 'robot_id': robot_id,
              **({} if side is None else {'side': side}),
              'source': SOURCE, 'native_physics_dt_s': None, 'control_period_s': .02,
              'sensor_paths': {}, 'force_calibrated': False,
              'region': 'whole_elastomer_collision_body',
              'aggregation': 'sum of unique pair point-normal impulse magnitudes over the complete control window / window duration',
              'contact_impulse_threshold_ns': 0.0,
              'window_completeness': 'assumed_from_native_step_contract',
              'coverage': {'rigid': False, 'cloth': False, 'self_contacts': True,
                           'region': 'whole_elastomer_collision_body',
                           'taxels': False, 'shear': False, 'torque': False}}
    if not isinstance(metadata, dict):
        return config
    evidence = metadata.get('contact_evidence', {})
    hand = metadata.get('hand_spec', {})
    links, region = finger_links(robot_id, hand, side)
    if not links:
        return config
    config['region'] = config['coverage']['region'] = region
    scene_info = metadata.get('scene_info', {})
    if not all(isinstance(value, dict) for value in (evidence, hand, scene_info)):
        return config
    dt = metadata.get('physics_dt_s')
    period = metadata.get('control_dt_s')
    if (metadata.get('native_isaacsim_app') is not True or hand.get('hand_id') != robot_id
            or not _finite(dt) or dt <= 0 or not _finite(period) or abs(period - .02) > 1e-9
            or evidence.get('source') != NATIVE_SOURCE or evidence.get('impulse_components') != 'normal_only'
            or not isinstance(evidence.get('aggregation'), dict)
            or evidence['aggregation'].get('version') != 'float64_point_normal_impulses_v1'):
        return config
    steps = period / dt
    if not 1 <= steps <= 10000 or abs(steps - round(steps)) > 1e-8:
        return config
    sensors = evidence.get('sensor_paths')
    paths = scene_info.get('robot_link_paths')
    bodies = metadata.get('body_names')
    if (not isinstance(sensors, list) or not isinstance(paths, dict) or not isinstance(bodies, list)
            or not all(isinstance(value, str) for value in sensors + bodies)):
        return config
    mapping = {}
    for finger, link in links.items():
        path = paths.get(link)
        if (not isinstance(path, str) or not path.startswith('/World/Robot/')
                or path.rsplit('/', 1)[-1] != link or sensors.count(path) != 1 or bodies.count(link) != 1):
            return config
        mapping[finger] = path
    config.update(available=True, sensor_paths=mapping, native_physics_dt_s=float(dt))
    config['coverage']['rigid'] = True
    return config


def project_tactile(raw, configuration, control_period_s=.02):
    """Return only a versioned sensor projection; never a raw contact report.

    A valid zero means the native contract reports no solved normal impulse in
    the last complete control interval. At reset, missing data, malformed
    reports or unsupported coverage are represented explicitly rather than zero.
    Validity here checks structure and the native sampling contract, not force
    calibration or independent completeness of zero-contact substeps.
    """
    sim_time = raw.get('sim_time_s') if isinstance(raw, dict) else None
    if not _finite(sim_time) or sim_time < 0:
        raise ValueError('Tactile projection requires a finite nonnegative observation time')
    fingers = tuple(configuration.get('sensor_paths') or {}) or tuple(FINGER_LINKS)
    coverage = {'rigid': configuration.get('available') is True, 'cloth': False, 'self_contacts': True,
                'region': configuration.get('region', ELASTOMER_REGION), 'taxels': False, 'shear': False, 'torque': False}
    result = {'protocol': PROTOCOL, 'mode': MODE, 'status': 'unavailable', 'sim_time_s': float(sim_time),
              'window_start_sim_s': None, 'window_duration_s': None,
              'window_completeness': 'not_available', 'force_calibrated': False,
              'frame_id': 'scalar_invariant', 'coverage': coverage,
              'fingers': {finger: {'status': 'unavailable', 'rigid_contact_any': None,
                                   'normal_impulse_sum_ns': None, 'mean_normal_load_n': None}
                          for finger in fingers}}

    def status(value):
        result['status'] = value
        for finger in result['fingers'].values():
            finger['status'] = value
        return result

    if configuration.get('available') is not True:
        return result
    period, dt = control_period_s, configuration.get('native_physics_dt_s')
    paths = configuration.get('sensor_paths')
    if (not _finite(period) or period <= 0 or abs(period - configuration.get('control_period_s', 0.)) > 1e-9
            or not _finite(dt) or dt <= 0 or not isinstance(paths, dict) or set(paths) != set(fingers)):
        return status('invalid')
    steps = round(period / dt)
    if steps <= 0 or abs(steps * dt - period) > 1e-9:
        return status('invalid')
    contacts = raw.get('contacts')
    if (not isinstance(contacts, dict) or contacts.get('source') != SOURCE
            or contacts.get('errors') != [] or not isinstance(contacts.get('rigid'), list)):
        return status('invalid')
    duration = raw.get('last_duration_s')
    rows = contacts['rigid']
    if _finite(duration) and duration == 0 and sim_time == 0 and not rows:
        return status('not_sampled')
    if (not _finite(duration) or abs(duration - period) > 1e-9 or sim_time < period - 1e-8):
        return status('invalid')
    start = max(0., float(sim_time - period))
    by_path = {path: finger for finger, path in paths.items()}
    values = {finger: [] for finger in fingers}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            return status('invalid')
        timestamp, actor0, actor1 = (row.get(key) for key in ('time_s', 'actor0', 'actor1'))
        magnitude = row.get('impulse_magnitude_sum_ns')
        points, nonzero = row.get('point_count'), row.get('nonzero_impulse_normal_count')
        normal_error = row.get('normal_length_max_abs_error')
        vector = row.get('impulse_vector_ns')
        if (not _finite(timestamp) or not isinstance(actor0, str) or not isinstance(actor1, str)
                or actor0 == actor1 or not _finite(magnitude) or magnitude < 0
                or type(points) is not int or points <= 0 or type(nonzero) is not int or not 0 <= nonzero <= points
                or not _finite(normal_error) or not 0 <= normal_error <= NORMAL_TOLERANCE
                or row.get('impulse_components') != 'normal_only'
                or not isinstance(vector, list) or len(vector) != 3 or not all(_finite(x) for x in vector)):
            return status('invalid')
        if math.hypot(*vector) > magnitude + 1e-8 or (magnitude > 0 and nonzero == 0):
            return status('invalid')
        tick = round((timestamp - start) / dt)
        if not 1 <= tick <= steps or abs(timestamp - (start + tick * dt)) > 1e-8:
            return status('invalid')
        pair = (tick, *sorted((actor0, actor1)))
        if pair in seen:
            return status('invalid')
        seen.add(pair)
        # Each native pair occurs once; pad-pad contact contributes to each
        # involved finger once, without publishing the counterpart identity.
        for actor in (actor0, actor1):
            finger = by_path.get(actor)
            if finger is not None:
                values[finger].append(float(magnitude))
    try:
        sums = {finger: math.fsum(samples) for finger, samples in values.items()}
        if not all(math.isfinite(total / period) for total in sums.values()):
            return status('invalid')
    except (OverflowError, ValueError):
        return status('invalid')
    result.update(status='available', window_start_sim_s=start, window_duration_s=float(period),
                  window_completeness='assumed_from_native_step_contract')
    for finger, total in sums.items():
        result['fingers'][finger] = {'status': 'valid', 'rigid_contact_any': total > 0.,
                                     'normal_impulse_sum_ns': total, 'mean_normal_load_n': total / period}
    return result
