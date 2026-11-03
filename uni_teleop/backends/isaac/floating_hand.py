"""Standalone SHARPA in native PhysX, using the common hand model and targets."""
import numpy as np

from uni_teleop.backends.sharpa import PALM, INITIAL_POSITION, INITIAL_QUATERNION
from uni_teleop.control.floating_hand import wrist_wrench


class FloatingHand:
    control_dt = .02

    def __init__(self, hand, output, *, headless=True, bench_scene=None):
        self.hand = hand
        self.bench_scene = bench_scene
        from isaacsim import SimulationApp
        self.app = SimulationApp({'headless': headless, 'multi_gpu': False, 'width': 1280, 'height': 960})
        self.world = None
        self.desktop_views = None
        try:
            self._initialize(output, headless)
        except BaseException as error:
            import json, traceback
            (output/'initialization_error.json').write_text(json.dumps({'error': str(error)}))
            traceback.print_exc()
            self.close(exit_code=1)
            raise

    def _initialize(self, output, headless):
        hand = self.hand
        import omni.kit.app, omni.usd
        from pxr import PhysxSchema, UsdPhysics, UsdLux, UsdGeom
        manager = omni.kit.app.get_app().get_extension_manager()
        for extension in ('isaacsim.asset.importer.urdf', 'isaacsim.core.api', 'isaacsim.sensors.camera'):
            manager.set_extension_enabled_immediate(extension, True)
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
        from isaacsim.core.utils.viewports import set_camera_view
        config = URDFImporterConfig(urdf_path=str(hand.urdf), usd_path=str(output/'isaac_asset'),
            fix_base=False, merge_fixed_joints=False, merge_mesh=False, collision_from_visuals=False,
            collision_type='Convex Hull', allow_self_collision=False,
            joint_drive_type='force', joint_target_type='position',
            override_joint_stiffness=3., override_joint_damping=.08,
            run_asset_transformer=True, run_multi_physics_conversion=True)
        usd = URDFImporter(config).import_urdf()
        if not omni.usd.get_context().open_stage(usd): raise RuntimeError('Failed to open SHARPA USD')
        self.stage = omni.usd.get_context().get_stage()
        self.stage.GetDefaultPrim().GetVariantSet('Physics').SetVariantSelection('physx')
        roots = [p for p in self.stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1: raise ValueError(f'Expected one SHARPA articulation, got {len(roots)}')
        # Author before creating tensor views: schema changes can invalidate them.
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        self.timestep = .002
        self.world = World(stage_units_in_meters=1., physics_dt=self.timestep, rendering_dt=.02,
                           backend='numpy', device='cpu')
        context = self.world.get_physics_context()
        context.enable_gpu_dynamics(False)
        context.set_broadphase_type('MBP')
        if self.bench_scene is not None:
            from .hand_scene import mount
            self.task_assets = mount(self.stage, self.bench_scene)
            import json
            (output/'task_assets.json').write_text(json.dumps(self.task_assets, indent=2))
        else:
            self.world.scene.add_default_ground_plane()
            self.world.scene.add(FixedCuboid('/World/table', position=[.15, 0., .72], scale=[1.1, .9, .06],
                                         color=np.array([.6, .48, .32])))
            self.block = self.world.scene.add(DynamicCuboid('/World/block', position=[.16, 0., .785],
            scale=[.06, .06, .07], mass=.08, color=np.array([.12, .65, .48])))
            light = UsdLux.DomeLight.Define(self.stage, '/World/Light')
            light.CreateIntensityAttr(700.)
        self.world.reset()
        simulation = self.world.physics_sim_view
        simulation.set_subspace_roots('/')
        self.view = simulation.create_articulation_view(str(roots[0].GetPath()))
        if self.view.count != 1: raise RuntimeError('SHARPA articulation did not initialize')
        self.one = np.array([0], dtype=np.int32)
        self.names = tuple(self.view.shared_metatype.dof_names)
        root = self.view.get_root_transforms().copy()
        start_position = [.27, 0., 1.12] if self.bench_scene is not None else INITIAL_POSITION
        root[0] = np.r_[start_position, INITIAL_QUATERNION]
        self.view.set_root_transforms(root, self.one)
        self.view.set_root_velocities(np.zeros((1, 6), dtype=np.float32), self.one)
        self.order = np.array([self.names.index(n) for n in hand.names], dtype=np.int32)
        q = self.view.get_dof_positions().copy()
        q[0, self.order] = hand.initial
        self.view.set_dof_positions(q, self.one)
        self.view.set_dof_position_targets(q, self.one)
        kp, kd, limits = (np.zeros_like(q) for _ in range(3))
        kp[0, self.order], kd[0, self.order], limits[0, self.order] = 3., .08, hand.effort
        self.view.set_dof_stiffnesses(kp, self.one)
        self.view.set_dof_dampings(kd, self.one)
        self.view.set_dof_max_forces(limits, self.one)
        self.root_index = tuple(self.view.shared_metatype.link_names).index(PALM)
        self.root_com = self.view.get_coms()[0, self.root_index, :3].copy()
        self.forces = np.zeros((1, self.view.max_links, 3), dtype=np.float32)
        self.torques = self.forces.copy()
        self.events = []
        self.camera_path = '/OmniverseKit_Persp'
        self.camera_eye = [-.35, -.85, 1.5] if self.bench_scene is not None else [.8, -.8, 1.35]
        self.camera_target = [.42, 0., .87] if self.bench_scene is not None else [.08, 0., .94]
        set_camera_view(eye=self.camera_eye, target=self.camera_target, camera_prim_path=self.camera_path)
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(self.camera_path))
        camera.CreateFocalLengthAttr(26.)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(27.)
        if not headless:
            import omni.ui as ui
            import carb.input
            import omni.appwindow
            from omni.kit.hotkeys.core import KeyCombination, get_hotkey_registry
            registry = get_hotkey_registry()
            play_key = registry.get_hotkey('omni.kit.widget.toolbar', KeyCombination(carb.input.KeyboardInput.SPACE))
            if play_key is not None:
                registry.deregister_hotkey(play_key)
            self.input = carb.input.acquire_input_interface()
            self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self.key_subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self._key)
            self.window = ui.Window('PICO + MANUS | SHARPA hand', width=480, height=150)
            with self.window.frame:
                with ui.VStack():
                    ui.Label('Space: start/pause. R: recenter. Hold right grip: move wrist.')
                    with ui.HStack():
                        ui.Button('Start / Pause', clicked_fn=lambda: self.events.append('toggle'))
                        ui.Button('Recenter', clicked_fn=lambda: self.events.append('recenter'))
                    self.label = ui.Label('Paused', word_wrap=True)
            if self.bench_scene is not None:
                from uni_teleop.video.floating_hand_views import HandDesktopViews
                self.desktop_views = HandDesktopViews(self.stage, self.bench_scene, self.pose())
                self.window.position_x, self.window.position_y = 920, 610
        self.camera = None

    def _key(self, event):
        import carb.input
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            action = {'SPACE': 'toggle', 'R': 'recenter'}.get(event.input.name)
            if action:
                self.events.append(action)
        return True

    def physics_running(self):
        return self.world.is_playing()

    def close(self, exit_code=None):
        import sys
        if exit_code is None: exit_code = 1 if sys.exc_info()[0] is not None else 0
        if hasattr(self, 'key_subscription'):
            self.input.unsubscribe_to_keyboard_events(self.keyboard, self.key_subscription)
        if self.desktop_views is not None:
            self.desktop_views.close()
        if self.world is not None:
            self.world.stop()
            self.world.clear_instance()
        self.app.close(exit_code=exit_code)

    def pose(self):
        root = self.view.get_root_transforms()[0]
        return root[:3].copy(), root[3:7].copy()

    def view_forward(self):
        from pxr import Gf, Usd, UsdGeom
        transform = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.camera_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array(transform.TransformDir(Gf.Vec3d(0, 0, -1)))

    def positions(self):
        return self.view.get_dof_positions()[0, self.order].copy()

    def step(self, position, quaternion, fingers):
        from scipy.spatial.transform import Rotation
        # PhysX force submissions accumulate until the next simulated step.
        # An editor timeline pause must never accumulate controller wrenches.
        if not self.physics_running():
            return
        q = self.view.get_dof_position_targets().copy()
        q[0, self.order] = fingers
        self.view.set_dof_position_targets(q, self.one)
        for _ in range(round(self.control_dt/self.timestep)):
            if not self.physics_running():
                break
            p, r = self.pose()
            velocity = self.view.get_root_velocities()[0].copy()
            # PhysX linear velocity is at COM; the controller tracks palm origin.
            velocity[:3] -= np.cross(velocity[3:], Rotation.from_quat(r).apply(self.root_com))
            force, torque = wrist_wrench(p, r, velocity, position, quaternion)
            self.forces.fill(0)
            self.torques.fill(0)
            self.forces[0, self.root_index] = force
            self.torques[0, self.root_index] = torque
            self.view.apply_forces_and_torques_at_position(self.forces, self.torques, None, self.one, True)
            self.world.step(render=False)
        if not np.isfinite(self.view.get_root_transforms()).all():
            raise RuntimeError('Non-finite PhysX hand state')

    def render(self, status=''):
        if hasattr(self, 'label'): self.label.text = status
        if self.desktop_views is not None:
            self.desktop_views.update(self.pose())
        self.world.render()

    def running(self):
        return self.app.is_running()

    def screenshot(self, path):
        from PIL import Image
        from pxr import UsdGeom
        from isaacsim.sensors.camera import Camera
        from isaacsim.core.utils.viewports import set_camera_view
        if self.camera is None:
            self.camera = Camera('/World/HandPreview', resolution=(1280, 960))
            self.camera.initialize()
            self.camera.set_clipping_range(.02, 20.)
            set_camera_view(eye=self.camera_eye, target=self.camera_target, camera_prim_path='/World/HandPreview')
            camera = UsdGeom.Camera(self.stage.GetPrimAtPath('/World/HandPreview'))
            camera.CreateFocalLengthAttr(26.)
            camera.CreateHorizontalApertureAttr(36.)
            camera.CreateVerticalApertureAttr(27.)
        for _ in range(8): self.world.render()
        rgba = self.camera.get_rgba()
        if rgba is None or rgba.size == 0: raise RuntimeError('Isaac camera has no image')
        Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(path)
