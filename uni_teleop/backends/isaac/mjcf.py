"""Isaac PhysX joint interface for a MuJoCo-source articulated model (Isaac 6.1).

Imports source MJCF through NVIDIA's converter; it does not run MuJoCo dynamics.
PD torque is evaluated every physics step, matching the shared SONIC contract.
"""
from pathlib import Path
import numpy as np


class IsaacMjcfRobot:
    def __init__(self, xml_path, *, output, headless=True, timestep=.005):
        from isaacsim import SimulationApp
        self.app = SimulationApp({'headless':headless, 'multi_gpu':False, 'width':960, 'height':720})
        self.timestep = timestep
        self.world = None
        try:
            self._initialize(xml_path, output)
        except BaseException:
            self.close()
            raise

    def _initialize(self, xml_path, output):
        import omni.kit.app, omni.usd
        from pxr import UsdPhysics, PhysxSchema
        manager = omni.kit.app.get_app().get_extension_manager()
        for extension in ('isaacsim.asset.importer.mjcf', 'isaacsim.core.api'):
            manager.set_extension_enabled_immediate(extension, True)
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
        from isaacsim.core.api import World
        importer = MJCFImporter()
        importer.config = MJCFImporterConfig(mjcf_path=str(Path(xml_path).resolve()), usd_path=str(Path(output).resolve()),
                                             run_asset_transformer=False, allow_self_collision=False)
        usd = importer.import_mjcf()
        if not omni.usd.get_context().open_stage(usd):
            raise RuntimeError('Failed to open converted MJCF stage')
        self.stage = omni.usd.get_context().get_stage()
        roots = [p for p in self.stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            raise ValueError(f'Expected one articulation, found {len(roots)}')
        # PhysX cannot use triangle meshes on dynamic bodies as-is.
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr('convexHull')
        # The importer also emits fixed joints for static camera rigs. PhysX
        # rejects joints between static bodies; remove only those inert joints.
        def has_rigid_parent(path):
            prim = self.stage.GetPrimAtPath(path)
            while prim and not prim.IsPseudoRoot():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    api = UsdPhysics.RigidBodyAPI(prim)
                    return api.GetRigidBodyEnabledAttr().Get() is not False and not api.GetKinematicEnabledAttr().Get()
                prim = prim.GetParent()
            return False
        for prim in list(self.stage.Traverse()):
            if prim.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.FixedJoint(prim)
                bodies = joint.GetBody0Rel().GetTargets()+joint.GetBody1Rel().GetTargets()
                if bodies and not any(has_rigid_parent(p) for p in bodies):
                    self.stage.RemovePrim(prim.GetPath())
        self.world = World(stage_units_in_meters=1., physics_dt=self.timestep, rendering_dt=.02,
                           backend='numpy', device='cpu')
        context = self.world.get_physics_context()
        context.enable_gpu_dynamics(False)
        context.set_broadphase_type('MBP')
        self.world.reset()
        view = self.world.physics_sim_view
        view.set_subspace_roots('/')
        self.view = view.create_articulation_view(str(roots[0].GetPath()))
        if self.view.count != 1: raise RuntimeError('PhysX articulation initialization failed')
        self.names = tuple(self.view.shared_metatype.dof_names)
        self.one = np.array([0], dtype=np.int32)
        zero = np.zeros((1, len(self.names)), dtype=np.float32)
        self.view.set_dof_stiffnesses(zero, self.one)
        self.view.set_dof_dampings(zero, self.one)
        self._initial_q = self.view.get_dof_positions().copy()
        self._initial_root = self.view.get_root_transforms().copy()
        self._efforts = zero.copy()
        self.limits = self.view.get_dof_limits()[0].copy()

    @property
    def time(self): return self.world.current_time

    def indices(self, names):
        missing = set(names)-set(self.names)
        if missing: raise ValueError(f'Model missing joints: {sorted(missing)}')
        return np.array([self.names.index(n) for n in names], dtype=np.int32)

    def reset(self):
        self.view.set_dof_positions(self._initial_q, self.one)
        self.view.set_dof_velocities(np.zeros_like(self._initial_q), self.one)
        self.view.set_root_transforms(self._initial_root, self.one)
        self.view.set_root_velocities(np.zeros((1,6), dtype=np.float32), self.one)
        self._efforts.fill(0)

    def step_targets(self, positions, velocities, kp, kd, effort_limit, feedforward=None, steps=1):
        vectors = [np.asarray(v, dtype=np.float32) for v in (positions, velocities, kp, kd, effort_limit)]
        if any(v.shape != (len(self.names),) or not np.isfinite(v).all() for v in vectors):
            raise ValueError('Joint command must match all model DOFs and be finite')
        pos, vel, kp, kd, limit = vectors
        ff = np.zeros_like(pos) if feedforward is None else np.asarray(feedforward, dtype=np.float32)
        if ff.shape != pos.shape or not np.isfinite(ff).all() or steps < 1:
            raise ValueError('Invalid feedforward or step count')
        for _ in range(steps):
            q, v = self.view.get_dof_positions()[0], self.view.get_dof_velocities()[0]
            self._efforts[0] = np.clip(kp*(pos-q)+kd*(vel-v)+ff, -limit, limit)
            self.view.set_dof_actuation_forces(self._efforts, self.one)
            self.world.step(render=False)

    def render(self): self.world.render()

    def close(self):
        if self.world is not None:
            self.world.stop()
            self.world.clear_instance()
        self.app.close()
