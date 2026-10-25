"""Hash-pinned USD mechanisms, optional reduced dynamics and frozen observers.

No task names, controller choices or state setters occur in live sampling. The
observer and optional reduced model receive values only; simulator objects
remain inside the backend. Reduced state is explicitly labelled and does not
modify native bodies, particles, controls or solver settings.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np


SCHEMA = 'native-usd-task-bundle-v1'


def task_observer_enabled(scene):
    """Explicit unscored rigid practice; never skip constitutive or visual models."""
    enabled = scene.get('native_task_observer_enabled', True)
    if type(enabled) is not bool:
        raise ValueError('native_task_observer_enabled must be boolean')
    bundle = scene.get('native_task_bundle') or {}
    if not enabled and any(bundle.get(key) for key in (
            'particle_sets', 'reduced_dynamics', 'visual_state_bindings', 'volume_visual_bindings')):
        raise ValueError('Task observation is required for particles, reduced dynamics or task-driven visuals')
    return enabled


def checked(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Native task dependency hash differs: ' + str(path))
    return path


def readonly(value):
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.flags.writeable = False
        return result
    if isinstance(value, Mapping):
        return types.MappingProxyType({key: readonly(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(readonly(item) for item in value)
    return value


def load_observer(spec):
    """Load a separate package identified by all frozen helper hashes."""
    path = checked(spec['path'], spec['sha256'])
    files = dict(spec.get('files', {}))
    if files.get(path.name) != spec['sha256']:
        raise ValueError('Observer inventory must include the entrypoint hash')
    for relative, sha in files.items():
        item = (path.parent / relative).resolve()
        if not item.is_relative_to(path.parent) or item.suffix != '.py':
            raise ValueError('Observer dependencies must be Python files within its package')
        checked(item, sha)
    actual = {str(p.relative_to(path.parent)) for p in path.parent.rglob('*.py')}
    if actual != set(files):
        raise ValueError('Observer package has undeclared or missing Python dependencies')
    identity = hashlib.sha256(json.dumps({'root': str(path.parent), 'files': files}, sort_keys=True).encode()).hexdigest()
    package = '_native_task_observer_' + identity
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(path.parent)]
        sys.modules[package] = module
    name = package + '.' + path.stem
    if name not in sys.modules:
        source = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(source)
        sys.modules[name] = module
        source.loader.exec_module(module)
    factory = getattr(sys.modules[name], spec.get('class_name', 'NativeTaskObserver'))
    observer = factory(deepcopy(spec['configuration']))
    if not callable(getattr(observer, 'on_frame', None)) or not callable(getattr(observer, 'on_observe', None)):
        raise ValueError('Frozen task observer must implement on_frame and on_observe')
    return observer


def add_bundle(stage, spec, *, physics_scene_path):
    """Reference a complete task assembly; preserve its rigid/joint materials."""
    if spec is None:
        return None
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    if spec.get('schema') != SCHEMA:
        raise ValueError('Unknown native task bundle schema')
    source = checked(spec['usd_path'], spec['usd_sha256'])
    root = spec['root_prim_path']
    if not isinstance(root, str) or not root.startswith('/World/') or root.count('/') != 2:
        raise ValueError('Task root must be a direct child of /World')
    if stage.GetPrimAtPath(root):
        raise ValueError('Task bundle cannot replace an existing stage subtree')
    source_stage = Usd.Stage.Open(str(source))
    if not source_stage or not source_stage.GetPrimAtPath(root):
        raise ValueError('Task USD lacks its declared root')
    destination = stage.DefinePrim(root, 'Xform')
    destination.GetReferences().AddReference(str(source), root)
    excluded = spec.get('excluded_static_roots', [])
    for path in excluded:
        prim = stage.GetPrimAtPath(path)
        if not prim or prim.GetParent() != destination:
            raise ValueError('Excluded static root must be a direct task-bundle child')
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Joint)
               or p.HasAPI(UsdPhysics.ArticulationRootAPI)
               or any('Particle' in schema for schema in p.GetAppliedSchemas())
               for p in Usd.PrimRange(prim)):
            raise ValueError('Excluded static root cannot contain dynamics')
        prim.SetActive(False)
    for path in spec.get('hidden_display_prims', []):
        prim = stage.GetPrimAtPath(path)
        if not prim or not str(prim.GetPath()).startswith(root + '/'):
            raise ValueError('Hidden display must belong to the task bundle')
        if any(p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI)
               or p.IsA(UsdPhysics.Joint) for p in Usd.PrimRange(prim)):
            raise ValueError('Hidden display cannot contain physical geometry')
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            raise ValueError('Hidden display must be imageable')
        imageable.MakeInvisible()
    cache = UsdGeom.XformCache()
    if not np.allclose(np.asarray(cache.GetLocalToWorldTransform(destination)), np.eye(4), atol=1e-10):
        raise ValueError('Native task bundle root must preserve the world frame')
    bodies = dict(spec.get('rigid_bodies', {}))
    if not bodies or len(set(bodies.values())) != len(bodies):
        raise ValueError('Native task rigid bodies need distinct declared identifiers')
    actual = {str(p.GetPath()) for p in Usd.PrimRange(destination)
              if p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False}
    if actual != set(bodies.values()):
        raise ValueError('Native task rigid-body inventory differs from its declaration')
    objects = {}
    for name, path in bodies.items():
        prim = stage.GetPrimAtPath(path)
        if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True:
            raise ValueError('Declared task objects must be dynamic rigid bodies')
        tf = cache.GetLocalToWorldTransform(prim)
        quat = tf.ExtractRotationQuat()
        collision_paths = [str(p.GetPath()) for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
        prim.AddAppliedSchema('PhysxContactReportAPI')
        prim.CreateAttribute('physxContactReport:threshold', Sdf.ValueTypeNames.Float).Set(0.)
        objects[name] = {'path': path, 'collision_paths': collision_paths,
            'position_m': list(tf.ExtractTranslation()),
            'quaternion_wxyz': [quat.GetReal(), *quat.GetImaginary()],
            'kind': 'referenced_native_task_body', 'source_usd_sha256': spec['usd_sha256']}
    fixed = []
    for prim in Usd.PrimRange(destination):
        path = str(prim.GetPath())
        if prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError('Task bundles cannot supply a PhysicsScene or a robot articulation')
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not any(path == p or path.startswith(p + '/') for p in actual):
            fixed.append(path)
    joints = {}
    path_ids = {path: name for name, path in bodies.items()}
    joint_paths = dict(spec.get('joints', {}))
    actual_joints = {str(p.GetPath()) for p in Usd.PrimRange(destination) if p.IsA(UsdPhysics.Joint)}
    if actual_joints != set(joint_paths.values()) or len(set(joint_paths.values())) != len(joint_paths):
        raise ValueError('Native task joint inventory differs from its declaration')
    for name, path in joint_paths.items():
        prim = stage.GetPrimAtPath(path)
        if prim.IsA(UsdPhysics.PrismaticJoint):
            kind = 'prismatic'
        elif prim.IsA(UsdPhysics.SphericalJoint):
            kind = 'spherical'
        elif prim.IsA(UsdPhysics.FixedJoint):
            kind = 'fixed'
        else:
            raise ValueError('Task joint readback supports prismatic, passive spherical and passive fixed joints')
        joint = UsdPhysics.Joint(prim)
        targets = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
        if any(len(row) != 1 or str(row[0]) not in path_ids for row in targets):
            raise ValueError('Task joints must connect two declared dynamic bodies')
        row = {'body0': path_ids[str(targets[0][0])], 'body1': path_ids[str(targets[1][0])],
            'anchor0_m': list(joint.GetLocalPos0Attr().Get()), 'anchor1_m': list(joint.GetLocalPos1Attr().Get())}
        if kind == 'prismatic':
            # Preserve the original metadata and scalar readback for button and
            # piston releases, including their source-authored physical drives.
            local_rotation = joint.GetLocalRot0Attr().Get()
            axis = {'X': Gf.Vec3d(1, 0, 0), 'Y': Gf.Vec3d(0, 1, 0),
                    'Z': Gf.Vec3d(0, 0, 1)}[UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get()]
            row['axis0'] = list(Gf.Rotation(Gf.Quatd(local_rotation)).TransformDir(axis))
        else:
            if (row['body0'] == row['body1'] or joint.GetJointEnabledAttr().Get() is False
                    or any(schema.startswith('PhysicsDriveAPI:') for schema in prim.GetAppliedSchemas())):
                raise ValueError('Passive task links require distinct bodies, an enabled joint and no drives')
            row['kind'] = kind
            for index, attr in enumerate((joint.GetLocalRot0Attr(), joint.GetLocalRot1Attr())):
                quat = attr.Get()
                value = [quat.GetReal(), *quat.GetImaginary()]
                _finite_vector(row[f'anchor{index}_m'], 3, 'Passive joint local anchor')
                _unit_rotation(value)
                row[f'local_rotation{index}_wxyz'] = value
        joints[name] = row
    particles = deepcopy(spec.get('particle_sets', {}))
    for name, row in particles.items():
        prim = stage.GetPrimAtPath(row['prim_path'])
        system = stage.GetPrimAtPath(row['system_path'])
        if not prim or not system or row.get('position_attribute') != 'points':
            raise ValueError('Particle sets require an explicit unsmoothed Points source')
        if not prim.IsA(UsdGeom.Points) or system.GetAttribute('physxParticleSmoothing:particleSmoothingEnabled').Get() is not False:
            raise ValueError('Particle readback forbids smoothed or inferred particle positions')
        if not np.allclose(np.asarray(cache.GetLocalToWorldTransform(prim)), np.eye(4), atol=1e-10):
            raise ValueError('Raw particle Points must be authored in the world frame')
        points = np.asarray(prim.GetAttribute('points').Get())
        ids = np.asarray(prim.GetAttribute('ids').Get())
        if points.shape != (row['particle_count'], 3) or ids.shape != (row['particle_count'],) or len(set(ids.tolist())) != len(ids):
            raise ValueError('Frozen particle inventory differs from its declaration')
        if not np.isclose(UsdPhysics.MassAPI(prim).GetMassAttr().Get(), row['total_mass_kg'], atol=1e-8, rtol=0):
            raise ValueError('Frozen particle mass differs from its declaration')
        # The bench owns the PhysicsScene; this is reference composition, not
        # a change to the original asset or any particle material parameter.
        system.GetRelationship('simulationOwner').SetTargets([Sdf.Path(physics_scene_path)])
    return {'schema': SCHEMA, 'root_prim_path': root, 'objects': objects, 'joints': joints,
            'excluded_static_roots': list(excluded),
            'particle_sets': particles, 'fixtures': ({'native_bundle_fixed': {'collision_paths': fixed}} if fixed else {}),
            'usd_path': str(source), 'usd_sha256': spec['usd_sha256']}


def restore_bundle_before_world_reset(stage, info):
    """Restore the pinned assembly only while the caller has stopped World."""
    from pxr import Usd, UsdGeom
    source = Usd.Stage.Open(str(checked(info['usd_path'], info['usd_sha256'])))
    for row in info['objects'].values():
        target = stage.GetPrimAtPath(row['path'])
        original = source.GetPrimAtPath(row['path'])
        for op in UsdGeom.Xformable(original).GetOrderedXformOps():
            target.GetAttribute(op.GetOpName()).Set(op.GetAttr().Get())
        target.GetAttribute('xformOpOrder').Set(original.GetAttribute('xformOpOrder').Get())
        for name in ('physics:velocity', 'physics:angularVelocity'):
            attr = target.GetAttribute(name)
            if attr:
                attr.Set(original.GetAttribute(name).Get())
    for row in info['particle_sets'].values():
        target, original = (s.GetPrimAtPath(row['prim_path']) for s in (stage, source))
        for name in ('points', 'velocities', 'ids'):
            target.GetAttribute(name).Set(original.GetAttribute(name).Get())


def rotation(q):
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _finite_vector(value, size, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(label + ' must be a finite vector')
    return array


def _unit_rotation(value):
    quat = _finite_vector(value, 4, 'Passive joint orientation')
    length = np.linalg.norm(quat)
    if abs(length - 1.) > 1.e-4:
        raise ValueError('Passive joint orientation must be a unit wxyz quaternion')
    return rotation(quat / length)


def joint_positions(objects, joints):
    """Read joint-frame geometry; passive links expose no inferred force or score."""
    result = {}
    for name, row in joints.items():
        a, b = objects[row['body0']], objects[row['body1']]
        kind = row.get('kind', 'prismatic')
        if kind in ('spherical', 'fixed'):
            ra, rb = _unit_rotation(a['quaternion_wxyz']), _unit_rotation(b['quaternion_wxyz'])
            pa = (_finite_vector(a['position_m'], 3, 'Passive body position')
                  + ra @ _finite_vector(row['anchor0_m'], 3, 'Passive joint local anchor'))
            pb = (_finite_vector(b['position_m'], 3, 'Passive body position')
                  + rb @ _finite_vector(row['anchor1_m'], 3, 'Passive joint local anchor'))
            joint_a = ra @ _unit_rotation(row['local_rotation0_wxyz'])
            joint_b = rb @ _unit_rotation(row['local_rotation1_wxyz'])
            delta = _finite_vector(pb-pa, 3, 'Passive joint anchor delta')
            separation = float(np.linalg.norm(delta))
            if not np.isfinite(separation):
                raise ValueError('Passive joint anchor separation must be finite')
            result[name] = {'kind': kind, 'body0': row['body0'], 'body1': row['body1'],
                            'anchor_delta_world_m': delta.tolist(),
                            'anchor_separation_m': separation,
                            'relative_rotation_matrix': (joint_a.T @ joint_b).tolist()}
            continue
        if kind != 'prismatic':
            raise ValueError('Unsupported task joint readback kind')
        ra, rb = rotation(a['quaternion_wxyz']), rotation(b['quaternion_wxyz'])
        pa = np.asarray(a['position_m']) + ra @ row['anchor0_m']
        pb = np.asarray(b['position_m']) + rb @ row['anchor1_m']
        result[name] = {'position_m': float(np.dot(pb-pa, ra @ row['axis0'])),
                        'body0': row['body0'], 'body1': row['body1']}
    return result


class NativeTaskReader:
    def __init__(self, stage, info, spec, *, world=None):
        self.stage, self.info, self.spec = stage, info, spec
        self.world = world
        if info['particle_sets'] and world is None:
            raise ValueError('Physical particle readback requires the actual native World clock')
        self.observer = load_observer(spec['observer'])
        reduced = spec.get('reduced_dynamics')
        self.reduced_dynamics = load_observer(reduced) if reduced is not None else None
        self._last_observed_step = None
        self._state = None
        self._particles = {}
        from .task_visual_state import BooleanMaterialBindings
        self._visual_bindings = BooleanMaterialBindings(stage, spec.get('visual_state_bindings'))
        from .volume_visual_state import CylinderVolumeBindings
        self._volume_visual_bindings = CylinderVolumeBindings(stage, spec.get('volume_visual_bindings'))
        from pxr import Usd
        from .task_readback import validate_particle_arrays
        self.initial_particles = {}
        if info['particle_sets']:
            source = Usd.Stage.Open(info['usd_path'])
            for name, row in info['particle_sets'].items():
                prim = source.GetPrimAtPath(row['prim_path'])
                self.initial_particles[name] = validate_particle_arrays(
                    *(prim.GetAttribute(key).Get() for key in ('points', 'velocities', 'ids')),
                    row['particle_count'])
        self._particle_ids = {name: arrays[2].copy() for name, arrays in self.initial_particles.items()}

    def sample(self, *, objects, contacts, time_s, native_time_s, native_step, physics_dt_s, baseline=False):
        particles = {}
        if self.info['particle_sets']:
            from .task_readback import synchronize_particle_output
            if (int(self.world.current_time_step_index) != int(native_step)
                    or float(self.world.current_time) != float(native_time_s)):
                raise RuntimeError('Particle observer frame does not match the actual native tick')
            synchronize_particle_output(self.stage, self.world, {
                row['path']: [*objects[name]['position_m'], *objects[name]['quaternion_wxyz'][1:],
                              objects[name]['quaternion_wxyz'][0]]
                for name, row in self.info['objects'].items()})
        from .task_readback import validate_particle_arrays
        for name, row in self.info['particle_sets'].items():
            prim = self.stage.GetPrimAtPath(row['prim_path'])
            points, velocities, ids = validate_particle_arrays(
                *(prim.GetAttribute(key).Get() for key in ('points', 'velocities', 'ids')),
                row['particle_count'], self._particle_ids[name])
            particles[name] = {'positions_m': points, 'velocities_m_s': velocities, 'ids': ids,
                               'total_mass_kg': row['total_mass_kg']}
        frame = {'baseline': baseline, 'time_s': float(time_s), 'native_time_s': float(native_time_s),
                 'native_step': int(native_step), 'physics_dt_s': float(physics_dt_s),
                 'objects': objects, 'joints': joint_positions(objects, self.info['joints']),
                 'particle_sets': particles, 'contacts': contacts}
        if self.reduced_dynamics is not None:
            # A separately frozen constitutive/finite-volume model receives
            # the same native tick as the referee, never a simulator handle.
            self.reduced_dynamics.on_frame(readonly(frame))
            reduced_state = json.loads(json.dumps(self.reduced_dynamics.on_observe(), allow_nan=False))
            if not isinstance(reduced_state, dict):
                raise ValueError('Reduced dynamics must return a finite JSON state mapping')
            frame['reduced_dynamics'] = reduced_state
        self.observer.on_frame(readonly(frame))
        self._particles = particles
        self._step = native_step

    def capture_particles(self):
        return {key: value.copy() for name, row in self._particles.items()
                for key, value in ((f'task_particle_{name}_q', row['positions_m']),
                                   (f'task_particle_{name}_qd', row['velocities_m_s']),
                                   (f'task_particle_{name}_ids', row['ids']))}

    def observe(self):
        if self._last_observed_step != self._step:
            result = self.observer.on_observe()
            # Reject simulator handles, arbitrary objects and nonfinite values;
            # only the compact task-owned JSON state crosses worker IPC.
            self._state = json.loads(json.dumps(result, allow_nan=False))
            self._last_observed_step = self._step
        return deepcopy(self._state)

    def sync_visual_state(self):
        """Map sanitized observer state to declared render materials, without stepping."""
        if not self._visual_bindings.bindings and not self._volume_visual_bindings.bindings:
            return None
        state = self.observe()
        material = (self._visual_bindings.sync(state, native_step=self._step)
                    if self._visual_bindings.bindings else None)
        volume = (self._volume_visual_bindings.sync(state, native_step=self._step)
                  if self._volume_visual_bindings.bindings else None)
        if material is None:
            return volume
        if volume is None:
            return material  # Preserve the existing material-only receipt.
        return {'source_native_step': self._step, 'material_bindings': material, 'volume_bindings': volume}
