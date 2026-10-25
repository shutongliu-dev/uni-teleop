"""Explicit solver comparison for private engineering probes only."""


def diagnostic_solver_env(base, solver, report):
    if solver is not None and solver not in ('TGS', 'PGS'):
        raise ValueError('Explicit diagnostic solver must be TGS or PGS')
    class ProbeEnv(base):
        def _initialize(self):
            if solver is None:
                return super()._initialize()
            from bench.simulation.isaacsim_backend import scene as module
            from pxr import PhysxSchema
            original = module.build_scene
            def author_solver(stage, scene_spec, robot_usd):
                info = original(stage, scene_spec, robot_usd)
                context = self.world.get_physics_context()
                api = PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(context.prim_path))
                before = api.GetSolverTypeAttr().Get()
                api.CreateSolverTypeAttr(solver)
                report['diagnostic_solver'] = {'before': before, 'requested': solver,
                    'usd_readback': api.GetSolverTypeAttr().Get(),
                    'gpu_dynamics_enabled': api.GetEnableGPUDynamicsAttr().Get(),
                    'authored_before_native_initialization': True,
                    'native_solver_identity_readback_claimed': False}
                if report['diagnostic_solver']['usd_readback'] != solver or not report['diagnostic_solver']['gpu_dynamics_enabled']:
                    raise ValueError('Diagnostic GPU solver authoring failed')
                return info
            module.build_scene = author_solver
            try:
                super()._initialize()
            finally:
                module.build_scene = original
    return ProbeEnv
