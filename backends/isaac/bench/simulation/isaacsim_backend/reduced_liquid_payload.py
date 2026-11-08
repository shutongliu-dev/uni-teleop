"""Opt-in finite-volume liquid payloads on actual native rigid bodies.

This is a carried-mass approximation, not PBD/SPH. A frozen provider supplies
conserved volumes, never simulator handles. Only native mass/COM/inertia are
changed; robot controls, poses, velocities and external forces are untouched.
Liquid occupies an equivalent body-fixed cylinder: slosh, pressure, transfer
momentum and jet reaction are not resolved. Each update applies after the
measured tick and before the next solver step, including the reset baseline.
"""
from copy import deepcopy
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

SCHEMA = 'reduced-liquid-payload-v1'
LIMITATIONS = [
    'Equivalent body-fixed liquid cylinder; no resolved free-surface slosh or pressure.',
    'Absorbed/emitted liquid changes carried mass and inertia without resolving transfer momentum or jet reaction.',
    'In-flight and spilled liquid remain in the conserved volume ledger but do not load a rigid body.',
    'Mass properties sampled after a native tick apply to the following solver step; no pose or velocity correction.',
]


def parameterized_dry_masses(spec, reduced):
    """Bind canonical payload dry masses to the release's sampled body masses.

    Parameterized releases scale task-body mass and record each body's exact
    canonical ('before') and sampled mass under reduced_dynamics. A payload
    binding that still holds the canonical mass takes the recorded sampled
    mass; the native dry-mass check then still requires the actual body to
    match it. Unscaled bodies (seed 0) keep their frozen value, and any other
    mismatch is left unchanged for that check to reject.
    Returns the spec to use and the list of applied rebindings.
    """
    configuration = (reduced or {}).get('configuration') if isinstance(reduced, Mapping) else None
    record = ((configuration or {}).get('reset_case', {}).get('parameterized_physics', {})
              .get('applied', {}).get('body_physics')) if isinstance(configuration, Mapping) else None
    if not isinstance(spec, Mapping) or not isinstance(record, Mapping):
        return spec, []
    result, rebound = deepcopy(spec), []
    for binding in result.get('bindings', []) if isinstance(result.get('bindings'), list) else []:
        body = record.get(binding.get('body_id')) if isinstance(binding, Mapping) else None
        if not isinstance(body, Mapping) or not isinstance(body.get('before'), Mapping):
            continue
        canonical, sampled, frozen = body['before'].get('mass_kg'), body.get('mass_kg'), binding.get('dry_mass_kg')
        if (all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (canonical, sampled, frozen))
                and math.isclose(frozen, canonical, rel_tol=1.e-6, abs_tol=1.e-9)
                and not math.isclose(sampled, canonical, rel_tol=1.e-6, abs_tol=1.e-9)):
            binding['dry_mass_kg'] = float(sampled)
            rebound.append({'body_id': binding['body_id'], 'frozen_payload_dry_mass_kg': frozen,
                            'recorded_canonical_mass_kg': canonical, 'recorded_sampled_mass_kg': sampled})
    return result, rebound


def numpy_copy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value).copy()


def number(value, lo, hi):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not lo <= value <= hi):
        raise ValueError('Liquid payload requires finite numbers within declared bounds')
    return float(value)


def state_path(value):
    if not isinstance(value, list) or not value or any(type(k) is not str or not k for k in value):
        raise ValueError('Liquid payload requires explicit nonempty state paths')
    return tuple(value)


def at(state, keys):
    for key in keys:
        if not isinstance(state, Mapping) or key not in state:
            raise ValueError('Missing liquid payload state path: ' + '.'.join(keys))
        state = state[key]
    return state


def parallel_axis(mass, offset):
    return mass * (float(offset @ offset)*np.eye(3)-np.outer(offset, offset))


def composite_properties(dry, volume_ml, density_kg_m3, binding):
    """Combine about the new COM, with both tensors in the body prim frame."""
    water_mass = volume_ml*1.e-6*density_kg_m3
    if water_mass == 0.:
        return {'mass_kg': dry['mass_kg'], 'com_m': np.asarray(dry['com_m']).copy(),
                'inertia_kg_m2': np.asarray(dry['inertia_kg_m2']).copy()}
    radius = binding['radius_m']
    height = volume_ml*1.e-6/(math.pi*radius**2)
    axis = np.asarray(binding['axis_body'])
    water_com = np.asarray(binding['floor_center_m']) + axis*(height/2.)
    along = water_mass*radius**2/2.
    across = water_mass*(3.*radius**2+height**2)/12.
    water_inertia = across*np.eye(3)+(along-across)*np.outer(axis, axis)
    mass = dry['mass_kg']+water_mass
    dry_com = np.asarray(dry['com_m'])
    com = (dry['mass_kg']*dry_com+water_mass*water_com)/mass
    inertia = (np.asarray(dry['inertia_kg_m2']) + parallel_axis(dry['mass_kg'], dry_com-com)
               + water_inertia + parallel_axis(water_mass, water_com-com))
    return {'mass_kg': mass, 'com_m': com, 'inertia_kg_m2': inertia}


class NativeLiquidPayload:
    """Restricted generic native adapter; construction captures dry inputs once."""
    def __init__(self, spec, views, tensor, *, physical_particle_sets=False):
        self.spec = deepcopy(spec)
        required = {'schema', 'state_schema', 'density_kg_m3', 'initial_total_ml',
                    'unbound_volume_paths', 'bindings'}
        if not isinstance(spec, dict) or set(spec) != required or spec['schema'] != SCHEMA:
            raise ValueError('Expected explicit reduced-liquid-payload-v1 specification')
        if physical_particle_sets:
            raise ValueError('Liquid payload cannot duplicate native particle mass')
        if type(spec['state_schema']) is not str or not spec['state_schema']:
            raise ValueError('Liquid payload requires a frozen provider state schema')
        self.density = number(spec['density_kg_m3'], 1., 1.e5)
        self.total = number(spec['initial_total_ml'], 1.e-9, 1.e6)
        if not isinstance(spec['unbound_volume_paths'], list):
            raise ValueError('Expected explicit unbound volume partition')
        self.unbound = [state_path(path) for path in spec['unbound_volume_paths']]
        if not isinstance(spec['bindings'], list) or not spec['bindings']:
            raise ValueError('Expected explicit native rigid body bindings')
        self.bindings = []
        seen_bodies, seen_paths = set(), set(self.unbound)
        if len(seen_paths) != len(self.unbound):
            raise ValueError('Liquid inventory paths must be unique')
        self.views, self.tensor, self.dry, self.devices = {}, tensor, {}, {}
        fields = {'body_id', 'volume_path', 'dry_mass_kg', 'radius_m', 'floor_center_m',
                  'axis_body', 'maximum_volume_ml'}
        for original in spec['bindings']:
            if not isinstance(original, dict) or set(original) != fields:
                raise ValueError('Invalid native liquid payload binding fields')
            row = deepcopy(original)
            name = row['body_id']
            if type(name) is not str or name not in views or name in seen_bodies:
                raise ValueError('Liquid payload bodies must be distinct task rigid objects')
            path = state_path(row['volume_path'])
            if path in seen_paths:
                raise ValueError('Liquid inventory paths must be unique')
            row['dry_mass_kg'] = number(row['dry_mass_kg'], 1.e-8, 1.e4)
            row['radius_m'] = number(row['radius_m'], 1.e-4, 1.)
            row['maximum_volume_ml'] = number(row['maximum_volume_ml'], 1.e-9, 1.e6)
            if row['maximum_volume_ml']*1.e-6/(math.pi*row['radius_m']**2) > 2.:
                raise ValueError('Liquid equivalent cylinder exceeds supported height')
            for key in ('floor_center_m', 'axis_body'):
                v = np.asarray(row[key], dtype=float)
                if v.shape != (3,) or not np.isfinite(v).all():
                    raise ValueError('Liquid cylinder needs finite 3D coordinates')
                row[key] = v.tolist()
            if np.max(np.abs(row['floor_center_m'])) > 10. or not np.isclose(np.linalg.norm(row['axis_body']), 1., atol=1.e-7, rtol=0.):
                raise ValueError('Liquid cylinder requires a local unit axis and bounded center')
            view = views[name]
            if view.count != 1:
                raise ValueError('Liquid payload requires one native body per binding')
            self.views[name] = view
            read = self._read(name)
            if not np.isclose(read['mass_kg'], row['dry_mass_kg'], atol=1.e-8, rtol=1.e-5):
                raise ValueError('Initial native dry mass differs from frozen payload input')
            eig = np.linalg.eigvalsh(read['inertia_kg_m2'])
            if eig.min() <= 0. or eig.max() > eig.sum()-eig.max()+1.e-9:
                raise ValueError('Native dry inertia is not a physical positive tensor')
            self.dry[name] = read
            self.devices[name] = getattr(view.get_masses(), 'device', None)
            self.bindings.append(row)
            seen_bodies.add(name)
            seen_paths.add(path)
        self.body_ids = [row['body_id'] for row in self.bindings]
        self.last = None
        self.last_step = None
        self.last_time = None
        self.applied_count = 0

    def _read(self, name):
        view = self.views[name]
        mass = numpy_copy(view.get_masses()).reshape(-1)
        com = numpy_copy(view.get_coms()).reshape(-1)
        inertia = numpy_copy(view.get_inertias()).reshape(-1)
        if mass.shape != (1,) or com.shape != (7,) or inertia.shape != (9,):
            raise RuntimeError('Incomplete native payload mass-property readback')
        if not all(np.isfinite(x).all() for x in (mass, com, inertia)) or mass[0] <= 0.:
            raise RuntimeError('Nonfinite or nonpositive native mass properties')
        if abs(np.linalg.norm(com[3:])-1.) > 1.e-4:
            raise RuntimeError('Native principal-axes frame is not a unit quaternion')
        # Native tensors use column-major, at COM, expressed in body prim frame.
        inertia = inertia.reshape(3, 3, order='F').astype(float)
        if not np.allclose(inertia, inertia.T, atol=1.e-10, rtol=1.e-5):
            raise RuntimeError('Native inertia is not symmetric')
        return {'mass_kg': float(mass[0]), 'com_m': com[:3].astype(float),
                'com_pose_xyzw': com.astype(float), 'inertia_kg_m2': inertia}

    def begin_episode(self):
        """Keep original dry properties when resetting the same native bodies."""
        self.last_step, self.last_time, self.last = None, None, None
        self.applied_count = 0

    def sync(self, state, *, native_step, native_time_s, baseline=False):
        if (not isinstance(state, Mapping) or type(native_step) is not int or native_step < 0 or type(baseline) is not bool
                or type(state.get('native_step')) is not int or state['native_step'] != native_step
                or state.get('schema') != self.spec['state_schema']):
            raise ValueError('Liquid payload needs matching schema and native tick')
        now = number(native_time_s, 0., 1.e9)
        if abs(number(state.get('native_time_s'), 0., 1.e9)-now) > 1.e-9:
            raise ValueError('Liquid payload provider clock is stale')
        if state.get('errors'):
            raise ValueError('Liquid payload provider reported invalid dynamics')
        if 'density_kg_m3' in state and number(state['density_kg_m3'], 1., 1.e5) != self.density:
            raise ValueError('Liquid payload density differs from its frozen provider')
        if self.last_step is None:
            if not baseline:
                raise ValueError('Liquid payload requires a native reset baseline')
        elif baseline or native_step != self.last_step+1 or now <= self.last_time:
            raise ValueError('Liquid payload requires consecutive native ticks')
        volumes = [number(at(state, row['volume_path']), 0., row['maximum_volume_ml']) for row in self.bindings]
        unbound = [number(at(state, path), 0., self.total) for path in self.unbound]
        if abs(math.fsum(volumes+unbound)-self.total) > 1.e-7:
            raise ValueError('Liquid payload volume inventory is not conserved')
        planned = [composite_properties(self.dry[row['body_id']], volume, self.density, row)
                   for row, volume in zip(self.bindings, volumes)]
        actual = []
        # Validate the full volume partition before the first native mutation.
        for row, properties in zip(self.bindings, planned):
            name = row['body_id']
            view, device = self.views[name], self.devices[name]
            current = self._read(name)
            changed = not (np.isclose(current['mass_kg'], properties['mass_kg'], rtol=1.e-7, atol=1.e-9)
                and np.allclose(current['com_m'], properties['com_m'], rtol=0., atol=1.e-9)
                and np.allclose(current['inertia_kg_m2'], properties['inertia_kg_m2'], rtol=1.e-6, atol=1.e-12))
            if changed:
                # COM rotation is a principal-axis convention. The following full
                # inertia setter computes it; compare physical tensors, not its sign.
                indices = self.tensor(np.array([0], dtype=np.int32), device)
                view.set_masses(self.tensor(np.array([[properties['mass_kg']]], dtype=np.float32), device), indices)
                com = np.array([[*properties['com_m'], 0., 0., 0., 1.]], dtype=np.float32)
                view.set_coms(self.tensor(com, device), indices)
                values = properties['inertia_kg_m2'].reshape(1, 9, order='F').astype(np.float32)
                view.set_inertias(self.tensor(values, device), indices)
                current = self._read(name)
                self.applied_count += 1
            if not (np.isclose(current['mass_kg'], properties['mass_kg'], rtol=1.e-5, atol=1.e-8)
                    and np.allclose(current['com_m'], properties['com_m'], rtol=0., atol=1.e-6)
                    and np.allclose(current['inertia_kg_m2'], properties['inertia_kg_m2'], rtol=2.e-4, atol=1.e-10)):
                raise RuntimeError('Native liquid payload mass/COM/inertia readback mismatch')
            actual.append(current)
        self.last_step, self.last_time = native_step, now
        self.last = {'volumes_ml': volumes, 'unbound_ml': unbound, 'actual': actual,
                     'native_step': native_step, 'native_time_s': now}

    def capture(self):
        if self.last is None:
            raise RuntimeError('Liquid payload has no applied native baseline')
        actual = [self._read(name) for name in self.body_ids]
        return {'liquid_payload_volume_ml': np.asarray(self.last['volumes_ml']),
            'liquid_payload_unbound_volume_ml': np.asarray(self.last['unbound_ml']),
            'liquid_payload_mass_kg': np.asarray([row['mass_kg'] for row in actual]),
            'liquid_payload_com_pose_xyzw': np.asarray([row['com_pose_xyzw'] for row in actual]),
            'liquid_payload_inertia_kg_m2': np.asarray([row['inertia_kg_m2'] for row in actual]),
            'liquid_payload_applied_after_native_step': np.asarray(self.last_step),
            'liquid_payload_effective_from_native_step': np.asarray(self.last_step+1),
            'liquid_payload_native_time_s': np.asarray(self.last_time)}

    def metadata(self):
        def serial(row):
            return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row.items()}
        return {'schema': SCHEMA, 'spec': deepcopy(self.spec), 'body_ids': list(self.body_ids),
            'dry_native_properties': {name: serial(row) for name, row in self.dry.items()},
            'current_readback': {name: serial(self._read(name)) for name in self.body_ids},
            'current_volumes_ml': dict(zip(self.body_ids, self.last['volumes_ml'])) if self.last else None,
            'applied_after_native_step': self.last_step, 'native_time_s': self.last_time,
            'native_property_updates': self.applied_count, 'limitations': list(LIMITATIONS),
            'pose_or_velocity_setters_used': False, 'external_forces_applied': False,
            'inertia_frame': 'At COM, expressed in rigid-body-prim local coordinates; full symmetric tensor.',
            'dry_baseline_scope': 'Native dry properties captured once before the first liquid update; preserved across resets.'}


def authored_object_masses(metadata):
    """Read dry input masses, including referenced USD bodies without inline mass.

    The native bundle's pinned USD is inspected read-only. Native wet readbacks
    are never a substitute for authored dry mass, including unbound task bodies.
    """
    rows = metadata['scene_info']['objects']
    if set(rows) != set(metadata['object_names']):
        raise ValueError('Authored object mass inventory differs from recorded bodies')
    missing = {name for name, row in rows.items() if 'mass_kg' not in row}
    stage = None
    if missing:
        from pxr import Usd, UsdPhysics
        bundle = metadata['scene_spec']['native_task_bundle']
        path = Path(bundle['usd_path'])
        with path.open('rb') as stream:
            actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual_hash != bundle['usd_sha256']:
            raise ValueError('Authored dry mass source USD hash mismatch')
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise ValueError('Authored dry mass source USD cannot be opened')
        # This fallback accepts self-contained frozen bundles only. Do not trust
        # an outer USD hash when an unpinned sublayer/reference supplies mass.
        if any(not layer.anonymous and Path(layer.realPath).resolve() != path.resolve()
               for layer in stage.GetUsedLayers(includeClipLayers=True)):
            raise ValueError('Authored dry mass source has unpinned USD dependencies')
    result = {}
    for name, row in rows.items():
        if name in missing:
            if (row.get('kind') != 'referenced_native_task_body'
                    or row.get('source_usd_sha256') != actual_hash
                    or bundle['rigid_bodies'].get(name) != row['path']):
                raise ValueError('Authored dry mass source body identity mismatch')
            prim = stage.GetPrimAtPath(row['path'])
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Authored dry mass source is not a rigid body')
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        else:
            mass = row['mass_kg']
        if mass is None or not np.isfinite(mass) or mass <= 0:
            raise ValueError('Authored dry object mass is missing or nonpositive')
        result[name] = float(mass)
    return result


def audit_native_payload(metadata, states, *, authored_masses=None):
    """Check archived native mass properties; never reinterpret wet mass as dry.

    This validates carried-load accounting and actual tensor readbacks only.
    Dynamic response requires a separate native force/contact experiment.
    """
    scene = metadata['scene_spec']
    spec, rebinding = parameterized_dry_masses(scene['reduced_liquid_payload'],
                                               scene.get('native_task_bundle', {}).get('reduced_dynamics'))
    if metadata.get('reduced_liquid_payload_dry_mass_rebinding', []) != rebinding:
        raise ValueError('Native liquid dry-mass rebinding differs from the frozen parameterized record')
    if authored_masses is None:
        authored_masses = authored_object_masses(metadata)
    payload = metadata.get('reduced_liquid_payload')
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA or payload.get('spec') != spec:
        raise ValueError('Missing or mismatched native liquid payload metadata')
    names = [row['body_id'] for row in spec['bindings']]
    if (payload['body_ids'] != names or len(set(names)) != len(names)
            or set(payload['dry_native_properties']) != set(names)
            or set(payload['current_readback']) != set(names)
            or set(payload['current_volumes_ml']) != set(names)):
        raise ValueError('Native liquid payload body inventory mismatch')
    n = len(states['native_physics_step_index'])
    fields = {'volume_ml': (n, len(names)), 'unbound_volume_ml': (n, len(spec['unbound_volume_paths'])),
              'mass_kg': (n, len(names)), 'com_pose_xyzw': (n, len(names), 7),
              'inertia_kg_m2': (n, len(names), 3, 3), 'applied_after_native_step': (n,),
              'effective_from_native_step': (n,), 'native_time_s': (n,)}
    arrays = {}
    for key, shape in fields.items():
        array = np.asarray(states['liquid_payload_'+key])
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('Incomplete native payload capture: '+key)
        arrays[key] = array
    if (n < 1 or not np.array_equal(arrays['applied_after_native_step'], states['native_physics_step_index'])
            or not np.array_equal(arrays['effective_from_native_step'], arrays['applied_after_native_step']+1)
            or not np.allclose(arrays['native_time_s'], states['native_world_time_s'], atol=1.e-9, rtol=0)):
        raise ValueError('Native payload archive clock mismatch')
    volumes, unbound = arrays['volume_ml'], arrays['unbound_volume_ml']
    if np.any(volumes < 0.) or np.any(unbound < 0.) or not np.allclose(
            volumes.sum(1)+unbound.sum(1), spec['initial_total_ml'], atol=1.e-7, rtol=0):
        raise ValueError('Archived payload volume inventory is not conserved')
    indices = np.flatnonzero(arrays['applied_after_native_step'] == payload['applied_after_native_step'])
    if len(indices) != 1:
        raise ValueError('Metadata payload tick has no unique native capture')
    captured_index = int(indices[0])
    expected_masses = {}
    for j, row in enumerate(spec['bindings']):
        name = row['body_id']
        dry = payload['dry_native_properties'][name]
        authored = authored_masses[name]
        if (not np.isclose(dry['mass_kg'], authored, rtol=1.e-5, atol=1.e-8)
                or not np.isclose(dry['mass_kg'], row['dry_mass_kg'], rtol=1.e-5, atol=1.e-8)
                or np.any(volumes[:, j] > row['maximum_volume_ml'])):
            raise ValueError('Archived payload dry mass or capacity differs from frozen input')
        for i in range(n):
            expected = composite_properties(dry, float(volumes[i, j]), spec['density_kg_m3'], row)
            if (not np.isclose(arrays['mass_kg'][i, j], expected['mass_kg'], rtol=1.e-5, atol=1.e-8)
                    or not np.allclose(arrays['com_pose_xyzw'][i, j, :3], expected['com_m'], atol=1.e-6, rtol=0)
                    or not np.allclose(arrays['inertia_kg_m2'][i, j], expected['inertia_kg_m2'], rtol=2.e-4, atol=1.e-10)
                    or abs(np.linalg.norm(arrays['com_pose_xyzw'][i, j, 3:])-1.) > 1.e-4):
                raise ValueError('Archived native liquid mass/COM/inertia mismatch')
        i = captured_index
        current = payload['current_readback'][name]
        if (payload['current_volumes_ml'][name] != volumes[i, j]
                or not np.isclose(current['mass_kg'], arrays['mass_kg'][i, j], rtol=1.e-5, atol=1.e-8)
                or not np.allclose(current['com_m'], arrays['com_pose_xyzw'][i, j, :3], atol=1.e-6, rtol=0)
                or not np.allclose(current['inertia_kg_m2'], arrays['inertia_kg_m2'][i, j], rtol=2.e-4, atol=1.e-10)
                or not np.allclose(metadata['native_object_com_pose_xyzw'][name][:3], current['com_m'], atol=1.e-6, rtol=0)
                or not np.allclose(metadata['native_object_inertias_kg_m2'][name], current['inertia_kg_m2'], rtol=2.e-4, atol=1.e-10)):
            raise ValueError('Payload metadata differs from actual native capture')
        expected_masses[name] = dry['mass_kg']+volumes[i, j]*1.e-6*spec['density_kg_m3']
    return expected_masses, {'schema': SCHEMA, 'body_ids': names, 'captured_frames': n,
        'metadata_frame_index': captured_index, 'conserved_inventory_ml': spec['initial_total_ml'],
        'scope': 'Native tensor mass/COM/inertia readbacks match frozen carried-volume accounting. '
                 'This check does not validate momentum transfer, sloshing or dynamic response.'}
