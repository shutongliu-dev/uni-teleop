"""Identity-free rigid-contact history across one robot action.

This consumes acknowledged native observations, never an evaluator or object
pose. Coactivity means contacts occurred in the same sampled substep; it does
not certify opposing normals, a common object, force closure, or human likeness.
The existing five-pad observation remains a distinct last-window measurement.
"""
from copy import deepcopy
from itertools import combinations
import math
import re

from bench.sensors.tactile import FINGER_LINKS, project_tactile, tactile_configuration

PROTOCOL = 'robot_rigid_contact_history_v1'
BIMANUAL_PROTOCOL = 'robot_rigid_contact_history_v2'
INSTANCE_PROTOCOL = 'robot_rigid_contact_history_v3'
SIDES = ('left', 'right')
FINGERS = tuple(FINGER_LINKS)
REGIONS = tuple(f + '_pad' for f in FINGERS) + tuple(f + '_shell' for f in FINGERS) + ('palm_mount', 'arm')
PAIRS = tuple(a + '+' + b for a, b in combinations(FINGERS, 2))
SEMANTICS = 'positive solved normal impulses; coactivity does not imply opposition or a common object; each unique pair counted once per involved region'
BIMANUAL_SEMANTICS = ('positive solved normal impulses; per-hand self_contact includes same-side robot pairs only; '
    'cross_side_contact includes any left-right robot pair, including arms; global self_contact counts every robot pair once; '
    'pad coactivity does not imply opposition, a common object or grasp success')

INSTANCE_SEMANTICS = BIMANUAL_SEMANTICS.replace('left-right', 'cross-instance')


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _region(link):
    for finger, pad in FINGER_LINKS.items():
        if link == pad:
            return finger + '_pad'
        if link.startswith(pad.removesuffix('elastomer')):
            return finger + '_shell'
    if re.fullmatch(r'fr3_link[0-8]', link):
        return 'arm'
    if link in ('right_hand_flange', 'right_hand_wrist', 'right_hand_C_MC'):
        return 'palm_mount'
    return None


def configure_contact_history(metadata, robot_id):
    """Bind every actual robot link to a sensor; private paths never serialize."""
    if robot_id in ('dual_fr3_sharpa_w01', 'fr3_sharpa_instances'):
        return _configure_bimanual(metadata, robot_id)
    tactile = tactile_configuration(metadata, robot_id)
    config = {'protocol': PROTOCOL, 'available': False, 'tactile': tactile,
              'regions_by_path': {}, 'physics_dt_s': tactile['native_physics_dt_s'],
              'control_period_s': tactile['control_period_s']}
    if not tactile['available']:
        return config
    links = metadata.get('scene_info', {}).get('robot_link_paths')
    sensors = metadata.get('contact_evidence', {}).get('sensor_paths', [])
    bodies = metadata.get('body_names', [])
    objects = metadata.get('object_names', [])
    if (not isinstance(links, dict) or not links or not isinstance(objects, list)
            or any(not isinstance(name, str) for name in objects)
            or set(bodies) != set(links) | set(objects)):
        return config
    mapping = {}
    for link, path in links.items():
        if not isinstance(link, str) or not isinstance(path, str):
            return config
        region = _region(link)
        if (region is None or not path.startswith('/World/Robot/')
                or path.rsplit('/', 1)[-1] != link or sensors.count(path) != 1
                or bodies.count(link) != 1 or path in mapping):
            return config
        mapping[path] = region
    # Configured coverage is every declared rigid robot link. Bodies not in this
    # declared map cannot be inferred from name fragments in live observations.
    config.update(available=True, regions_by_path=mapping)
    return config


def _bimanual_region(link, namespaces=SIDES):
    """Match the declared W01 link vocabulary, never a side substring guess."""
    for side in namespaces:
        if re.fullmatch('fr3_' + re.escape(side) + r'_link[0-8]', link):
            return side, 'arm'
        if link in {side + '_hand_' + suffix for suffix in ('flange', 'wrist', 'C_MC')}:
            return side, 'palm_mount'
        for finger, native in (('thumb', 'thumb'), ('index', 'index'), ('middle', 'middle'),
                               ('ring', 'ring'), ('little', 'pinky')):
            prefix = side + '_' + native + '_'
            if link == prefix + 'elastomer':
                return side, finger + '_pad'
            shells = {'MCP_VL', 'PP', 'DP', 'fingertip'}
            shells |= {'CMC_VL', 'MC'} if finger == 'thumb' else {'MP'}
            if finger == 'little':
                shells.add('MC')
            if link in {prefix + suffix for suffix in shells}:
                return side, finger + '_shell'
    return None


def _configure_bimanual(metadata, robot_id):
    namespaces = SIDES if robot_id == 'dual_fr3_sharpa_w01' else tuple(sorted(metadata.get('hand_spec', {}).get('arms', {})))
    if (not 2 <= len(namespaces) <= 8 or any(not isinstance(n, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,31}', n) for n in namespaces)):
        raise ValueError('Native multi-instance metadata needs 2–8 valid namespaces')
    tactile = {side: tactile_configuration(metadata, robot_id, side) for side in namespaces}
    config = {'protocol': BIMANUAL_PROTOCOL if robot_id == 'dual_fr3_sharpa_w01' else INSTANCE_PROTOCOL, 'available': False, 'tactile': tactile,
              'regions_by_path': {}, 'sides_by_path': {},
              'physics_dt_s': tactile[namespaces[0]]['native_physics_dt_s'], 'control_period_s': .02}
    if robot_id != 'dual_fr3_sharpa_w01':
        config['instance_names'] = list(namespaces)
    if not all(value['available'] for value in tactile.values()):
        return config
    links = metadata.get('scene_info', {}).get('robot_link_paths')
    sensors = metadata.get('contact_evidence', {}).get('sensor_paths', [])
    bodies, objects = metadata.get('body_names'), metadata.get('object_names')
    if (not isinstance(links, dict) or not links or not isinstance(objects, list)
            or not isinstance(bodies, list) or not isinstance(sensors, list)
            or any(not isinstance(name, str) for name in objects + bodies + sensors)
            or len(set(bodies)) != len(bodies) or len(set(objects)) != len(objects)
            or set(links) & set(objects) or set(bodies) != set(links) | set(objects)):
        return config
    mapping, sides = {}, {}
    for link, path in links.items():
        if not isinstance(link, str) or not isinstance(path, str):
            return config
        classified = _bimanual_region(link, namespaces)
        if classified is None:
            return config
        side, region = classified
        scope = side.title() if robot_id == 'dual_fr3_sharpa_w01' else side
        if (not path.startswith('/World/Robot/' + scope + '/')
                or path.rsplit('/', 1)[-1] != link or any(part in ('.', '..', '') for part in path.split('/')[1:])
                or sensors.count(path) != 1 or path in mapping):
            return config
        mapping[path], sides[path] = side + ':' + region, side
    filters = metadata.get('contact_evidence', {}).get('filter_paths')
    if (set(path for path in sensors if path.startswith('/World/Robot/')) != set(mapping)
            or not isinstance(filters, list) or len(filters) != len(sensors)):
        return config
    for path in mapping:
        row = filters[sensors.index(path)]
        if (not isinstance(row, list) or any(not isinstance(value, str) for value in row)
                or len(set(row)) != len(row) or not set(sensors) <= set(row)):
            return config
    config.update(available=True, regions_by_path=mapping, sides_by_path=sides)
    return config


def _native_sampling_window(raw, start, now, dt, period, previous):
    """Verify actual acquisitions, including the zero-contact substeps."""
    sampling = raw.get('contacts', {}).get('sampling')
    if not isinstance(sampling, dict):
        return None
    clocks = ('physics_dt_s', 'start_sim_time_s', 'end_sim_time_s',
              'native_start_time_s', 'native_end_time_s')
    counts = ('sampled_substeps', 'native_start_step_index', 'native_end_step_index')
    if (sampling.get('protocol') != 'native_rigid_contact_sampling_v1'
            or sampling.get('complete') is not True
            or any(not _finite(sampling.get(key)) or sampling[key] < 0 for key in clocks)
            or any(type(sampling.get(key)) is not int or sampling[key] < 0 for key in counts)):
        return None
    count = round(period / dt)
    expected = ((sampling['physics_dt_s'], dt), (sampling['start_sim_time_s'], start),
                (sampling['end_sim_time_s'], now),
                (sampling['native_end_time_s'] - sampling['native_start_time_s'], period))
    if (any(not math.isclose(a, b, rel_tol=0., abs_tol=1e-8) for a, b in expected)
            or sampling['sampled_substeps'] != count
            or sampling['native_end_step_index'] - sampling['native_start_step_index'] != count
            or (previous is not None and (sampling['native_start_step_index'] != previous[0]
                or not math.isclose(sampling['native_start_time_s'], previous[1], rel_tol=0., abs_tol=1e-8)))):
        return None
    return sampling['native_end_step_index'], sampling['native_end_time_s']


def _counter():
    return {'contact_substeps': 0, 'current_contact_substeps': 0,
            'longest_contact_substeps': 0, 'normal_impulse_sum_ns': 0.}


class ContactHistory:
    """Constant-memory, fail-closed summary; reads never reset or advance it."""

    def __init__(self, configuration):
        self.config = deepcopy(configuration)
        self.bimanual = self.config['protocol'] in (BIMANUAL_PROTOCOL, INSTANCE_PROTOCOL)
        self.sides = tuple(self.config.get('instance_names', SIDES))
        self.region_names = tuple(side + ':' + region for side in self.sides for region in REGIONS) if self.bimanual else REGIONS
        self.pair_names = tuple(side + ':' + pair for side in self.sides for pair in PAIRS) if self.bimanual else PAIRS
        self.begin(None, 0.)

    def begin(self, action_id, sim_time_s):
        if (action_id is not None and (not isinstance(action_id, str)
                or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,160}', action_id))):
            raise ValueError('Expected a public action identifier')
        if not _finite(sim_time_s) or sim_time_s < 0:
            raise ValueError('Expected a finite action start time')
        self.action_id = action_id
        self.start = self.time = float(sim_time_s)
        self.windows = self.substeps = 0
        self.status = 'not_sampled' if self.config['available'] else 'unavailable'
        self.reason = None
        self.regions = {name: _counter() for name in self.region_names}
        self.pairs = {name: _counter() for name in self.pair_names}
        self.self_contact = _counter()
        self.side_self_contact = {side: _counter() for side in self.sides}
        self.cross_side_contact = _counter()
        self.native_sample_end = None

    def _invalid(self, reason, now=None):
        self.status, self.reason = 'invalid', reason
        if _finite(now) and now >= self.time:
            self.time = float(now)

    @staticmethod
    def _accumulate(counter, flags, impulses=None):
        for active in flags:
            counter['contact_substeps'] += int(active)
            counter['current_contact_substeps'] = counter['current_contact_substeps'] + 1 if active else 0
            counter['longest_contact_substeps'] = max(counter['longest_contact_substeps'], counter['current_contact_substeps'])
        if impulses is not None:
            counter['normal_impulse_sum_ns'] = math.fsum([counter['normal_impulse_sum_ns'], *impulses])
            if not math.isfinite(counter['normal_impulse_sum_ns']):
                raise ValueError('Nonfinite contact aggregate')

    def observe(self, raw):
        """Call once after each acknowledged 20ms step, including terminal steps."""
        if self.status in ('invalid', 'unavailable'):
            # Keep the timestamp current while preserving the unknown status;
            # later valid samples cannot erase a gap earlier in this action.
            now = raw.get('sim_time_s') if isinstance(raw, dict) else None
            if _finite(now) and now >= self.time:
                self.time = float(now)
            return
        if not isinstance(raw, dict):
            self._invalid('malformed_observation')
            return
        now = raw.get('sim_time_s')
        period, dt = self.config['control_period_s'], self.config['physics_dt_s']
        if (not _finite(now) or not _finite(raw.get('last_duration_s'))
                or abs(raw['last_duration_s'] - period) > 1e-9
                or abs(now - self.time - period) > 1e-8):
            self._invalid('noncontiguous_acknowledged_window', now)
            return
        try:
            tactile_configs = self.config['tactile'].values() if self.bimanual else [self.config['tactile']]
            valid = all(project_tactile(raw, config, period)['status'] == 'available' for config in tactile_configs)
        except (ValueError, TypeError, KeyError, OverflowError):
            self._invalid('invalid_rigid_contact_window', now)
            return
        if not valid:
            self._invalid('invalid_rigid_contact_window', now)
            return
        count = round(period / dt)
        native_end = _native_sampling_window(raw, self.time, now, dt, period, self.native_sample_end) if self.bimanual else None
        if self.bimanual and native_end is None:
            self._invalid('invalid_rigid_contact_window', now)
            return
        flags = {name: [False] * count for name in self.region_names}
        impulses = {name: [] for name in self.region_names}
        self_flags, self_impulses = [False] * count, []
        side_flags = {side: [False] * count for side in self.sides}
        side_impulses = {side: [] for side in self.sides}
        cross_flags, cross_impulses = [False] * count, []
        mapping = self.config['regions_by_path']
        for row in raw['contacts']['rigid']:
            actors = (row['actor0'], row['actor1'])
            if any(actor.startswith('/World/Robot/') and actor not in mapping for actor in actors):
                self._invalid('unmapped_robot_contact', now)
                return
            magnitude = row['impulse_magnitude_sum_ns']
            if self.bimanual and (magnitude > 0) != (row['nonzero_impulse_normal_count'] > 0):
                self._invalid('invalid_rigid_contact_window', now)
                return
            if magnitude <= 0:
                continue
            tick = round((row['time_s'] - (now - period)) / dt) - 1
            # A unique pair contributes once to each involved region, even when
            # both robot links belong to that region. No wrench is inferred.
            for region in {mapping[actor] for actor in actors if actor in mapping}:
                flags[region][tick] = True
                impulses[region].append(magnitude)
            if all(actor in mapping for actor in actors):
                self_flags[tick] = True
                self_impulses.append(magnitude)
                if self.bimanual:
                    sides = {self.config['sides_by_path'][actor] for actor in actors}
                    if len(sides) == 1:
                        side = next(iter(sides))
                        side_flags[side][tick] = True
                        side_impulses[side].append(magnitude)
                    else:
                        cross_flags[tick] = True
                        cross_impulses.append(magnitude)
        # Commit only complete windows. Malformed evidence never leaves a valid
        # partial history; once invalid, the whole action remains unknown.
        try:
            for region in self.region_names:
                self._accumulate(self.regions[region], flags[region], impulses[region])
            for key in self.pair_names:
                side, pair = key.split(':') if self.bimanual else ('', key)
                prefix = side + ':' if side else ''
                a, b = pair.split('+')
                self._accumulate(self.pairs[key], [x and y for x, y in zip(flags[prefix+a+'_pad'], flags[prefix+b+'_pad'])])
            self._accumulate(self.self_contact, self_flags, self_impulses)
            if self.bimanual:
                for side in self.sides:
                    self._accumulate(self.side_self_contact[side], side_flags[side], side_impulses[side])
                self._accumulate(self.cross_side_contact, cross_flags, cross_impulses)
        except (ValueError, OverflowError):
            self._invalid('nonfinite_contact_aggregate', now)
            return
        self.windows += 1
        self.substeps += count
        self.time = float(now)
        self.status = 'available'
        self.native_sample_end = native_end

    def snapshot(self):
        available = self.status == 'available'
        timing = ('contact_substeps', 'current_contact_substeps', 'longest_contact_substeps')
        result = {'protocol': self.config['protocol'], 'status': self.status, 'reason': self.reason,
                'action_id': self.action_id, 'start_sim_time_s': self.start,
                'sim_time_s': self.time, 'duration_s': self.time - self.start,
                'physics_dt_s': self.config['physics_dt_s'],
                'observed_control_windows': self.windows,
                'sampled_substeps': self.substeps,
                'coverage': {'all_declared_robot_rigid_links': self.config['available'],
                             'cloth': False, 'contact_directions': False, 'taxels': False,
                             'force_calibrated': False,
                             'completeness': 'assumed_from_native_step_contract'},
                'regions': deepcopy(self.regions) if available else None,
                'pad_coactivity': {pair: {key: row[key] for key in timing}
                                   for pair, row in self.pairs.items()} if available else None,
                'self_contact': deepcopy(self.self_contact) if available else None,
                'semantics': SEMANTICS}
        if self.bimanual:
            result.pop('regions')
            result.pop('pad_coactivity')
            result['coverage']['completeness'] = 'verified_native_sampling_clock'
            result['hands'] = {side: {
                'regions': {name: deepcopy(self.regions[side+':'+name]) for name in REGIONS},
                'pad_coactivity': {pair: {key: self.pairs[side+':'+pair][key] for key in timing} for pair in PAIRS},
                'self_contact': deepcopy(self.side_self_contact[side])} for side in self.sides} if available else None
            result['cross_side_contact'] = deepcopy(self.cross_side_contact) if available else None
            result['semantics'] = BIMANUAL_SEMANTICS
            if self.config['protocol'] == INSTANCE_PROTOCOL:
                result['instance_names'] = list(self.sides)
                result['semantics'] = INSTANCE_SEMANTICS
        return result
