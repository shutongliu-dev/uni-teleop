import unittest

import mujoco
import numpy as np

from sonic_mujoco.tactile import TactileRecorder, TaxelLayout

SUPPORTED_GEOMETRY_XML = """
<mujoco>
  <asset>
    <mesh name="tetrahedron"
          vertex="0.00 0.00 0.08  -0.07 -0.05 -0.05
                  0.07 -0.05 -0.05  0.00 0.08 -0.05"
          face="0 2 1  0 3 2  0 1 3  1 2 3"/>
  </asset>
  <worldbody>
    <body name="robot">
      <geom name="skin_box" type="box" pos="0.0 0.0 0.0"
            size="0.10 0.08 0.06" contype="0" conaffinity="0"/>
      <geom name="skin_sphere" type="sphere" pos="0.4 0.0 0.0"
            size="0.08" contype="0" conaffinity="0"/>
      <geom name="skin_capsule" type="capsule" pos="0.8 0.0 0.0"
            size="0.05 0.12" contype="0" conaffinity="0"/>
      <geom name="skin_cylinder" type="cylinder" pos="1.2 0.0 0.0"
            size="0.06 0.11" contype="0" conaffinity="0"/>
      <geom name="skin_ellipsoid" type="ellipsoid" pos="1.6 0.0 0.0"
            size="0.09 0.05 0.07" contype="0" conaffinity="0"/>
      <geom name="skin_mesh" type="mesh" pos="2.0 0.0 0.0"
            mesh="tetrahedron" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def geom_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def contact_model(*, robot_first: bool = True) -> tuple[mujoco.MjModel, int, int]:
    robot = """
    <body name="robot">
      <geom name="robot_skin" type="box" size="0.10 0.10 0.10"
            friction="1.0 0.01 0.001"/>
    </body>
    """
    object_body = """
    <body name="object" pos="0 0 0.19">
      <freejoint/>
      <geom name="object_geom" type="sphere" size="0.10" mass="1.0"
            friction="1.0 0.01 0.001"/>
    </body>
    """
    bodies = robot + object_body if robot_first else object_body + robot
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <option gravity="0 0 0" timestep="0.002"/>
          <worldbody>{bodies}</worldbody>
        </mujoco>
        """
    )
    return model, geom_id(model, "robot_skin"), geom_id(model, "object_geom")


def uncovered_shell_model() -> tuple[mujoco.MjModel, int, int]:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0" timestep="0.002"/>
          <worldbody>
            <body name="robot">
              <geom name="covered_shell" type="box" size="0.1 0.1 0.1"/>
              <geom name="uncovered_shell" type="box" pos="1 0 0"
                    size="0.1 0.1 0.1"/>
            </body>
            <body name="object" pos="1 0 0.19">
              <freejoint/>
              <geom name="object_geom" type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    return model, geom_id(model, "covered_shell"), geom_id(model, "uncovered_shell")


def self_contact_model() -> tuple[mujoco.MjModel, int, int]:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0" timestep="0.002"/>
          <worldbody>
            <body name="robot">
              <body name="vest_body">
                <joint name="vest_slide" type="slide" axis="1 0 0"/>
                <geom name="vest_shell" type="box" size="0.1 0.1 0.1"/>
              </body>
              <body name="arm_body" pos="0 0 0.19">
                <joint name="arm_slide" type="slide" axis="1 0 0"/>
                <geom name="arm_shell" type="sphere" size="0.1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    return model, geom_id(model, "vest_shell"), geom_id(model, "arm_shell")


def robot_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geom: int,
) -> np.ndarray:
    total = np.zeros(3)
    result = np.zeros(6)
    body_id = int(model.geom_bodyid[robot_geom])
    body_rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if robot_geom not in contact.geom:
            continue
        side = 0 if int(contact.geom[0]) == robot_geom else 1
        mujoco.mj_contactForce(model, data, contact_id, result)
        force_world = np.asarray(contact.frame).reshape(3, 3).T @ result[:3]
        if side == 0:
            force_world *= -1.0
        total += body_rotation.T @ force_world
    return total


class TaxelLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(SUPPORTED_GEOMETRY_XML)
        self.geom_ids = tuple(range(self.model.ngeom))

    def test_supported_surfaces_have_fixed_stable_indices(self) -> None:
        geom_count = self.model.ngeom
        first = TactileRecorder(
            self.model,
            robot_root="robot",
            geom_ids=reversed(self.geom_ids),
            spacing=0.035,
            max_taxels_per_geom=128,
        )
        second = TactileRecorder(
            self.model,
            robot_root="robot",
            geom_ids=self.geom_ids,
            spacing=0.035,
            max_taxels_per_geom=128,
        )

        self.assertEqual(self.model.ngeom, geom_count)
        self.assertGreater(len(first.layout), 0)
        np.testing.assert_array_equal(first.layout.geom_id, second.layout.geom_id)
        np.testing.assert_allclose(
            first.layout.local_center, second.layout.local_center
        )
        np.testing.assert_allclose(
            first.layout.local_normal, second.layout.local_normal
        )
        np.testing.assert_allclose(
            np.linalg.norm(first.layout.local_normal, axis=1), 1.0, atol=1e-6
        )
        for taxel_geom in self.geom_ids:
            count = np.count_nonzero(first.layout.geom_id == taxel_geom)
            self.assertGreater(count, 0)
            self.assertLessEqual(count, 128)
        mesh_geom = geom_id(self.model, "skin_mesh")
        mesh_id = int(self.model.geom_dataid[mesh_geom])
        self.assertGreater(
            np.count_nonzero(first.layout.geom_id == mesh_geom),
            int(self.model.mesh_facenum[mesh_id]),
        )

        def geom_points(name: str) -> np.ndarray:
            taxel_geom = geom_id(self.model, name)
            return (
                first.layout.local_center[first.layout.geom_id == taxel_geom]
                - self.model.geom_pos[taxel_geom]
            )

        box = geom_points("skin_box")
        np.testing.assert_allclose(
            np.max(np.abs(box) / (0.10, 0.08, 0.06), axis=1), 1.0
        )
        sphere = geom_points("skin_sphere")
        np.testing.assert_allclose(np.linalg.norm(sphere, axis=1), 0.08)
        ellipsoid = geom_points("skin_ellipsoid")
        np.testing.assert_allclose(
            np.sum(np.square(ellipsoid / (0.09, 0.05, 0.07)), axis=1), 1.0
        )
        cylinder = geom_points("skin_cylinder")
        on_cylinder = np.isclose(np.linalg.norm(cylinder[:, :2], axis=1), 0.06)
        on_cap = np.isclose(np.abs(cylinder[:, 2]), 0.11)
        self.assertTrue(np.all(on_cylinder | on_cap))
        capsule = geom_points("skin_capsule")
        axis_point = capsule.copy()
        axis_point[:, :2] = 0.0
        axis_point[:, 2] = np.clip(axis_point[:, 2], -0.12, 0.12)
        np.testing.assert_allclose(np.linalg.norm(capsule - axis_point, axis=1), 0.05)
        with self.assertRaises(ValueError):
            first.layout.local_center[0, 0] = 1.0
        np.testing.assert_array_equal(
            first.region_mask(body_names=("robot",)),
            np.ones(len(first.layout), dtype=bool),
        )
        np.testing.assert_array_equal(
            first.region_mask(body_prefixes=("missing_",)),
            np.zeros(len(first.layout), dtype=bool),
        )
        with self.assertRaisesRegex(ValueError, "region mask"):
            first.region_mask()

    def test_frame_is_fixed_size_and_empty_before_updates(self) -> None:
        recorder = TactileRecorder(
            self.model,
            robot_root="robot",
            geom_ids=self.geom_ids,
            spacing=0.05,
        )
        frame = recorder.last_frame
        count = len(recorder.layout)

        self.assertEqual(frame.force.shape, (count, 3))
        self.assertEqual(frame.normal_force.shape, (count,))
        self.assertEqual(frame.tangent_force.shape, (count, 3))
        self.assertEqual(frame.contact.shape, (count,))
        self.assertFalse(np.any(frame.contact))
        self.assertTrue(np.all(frame.other_body_id == -1))
        self.assertEqual(frame.duration, 0.0)

    def test_external_layout_preserves_region_and_channel_metadata(self) -> None:
        generated = TactileRecorder(
            self.model,
            robot_root="robot",
            geom_ids=self.geom_ids,
            spacing=0.05,
        ).layout
        count = len(generated)
        region_id = np.arange(count, dtype=np.int16) % 3
        channel_id = np.arange(count, dtype=np.int32) % 256
        external = TaxelLayout(
            body_id=generated.body_id,
            geom_id=generated.geom_id,
            local_center=generated.local_center,
            local_normal=2.0 * generated.local_normal,
            region_id=region_id,
            channel_id=channel_id,
            region_names=("vest", "left_arm", "right_arm"),
        )
        recorder = TactileRecorder(
            self.model,
            robot_root="robot",
            layout=external,
            spacing=0.05,
        )

        np.testing.assert_array_equal(recorder.layout.region_id, region_id)
        np.testing.assert_array_equal(recorder.layout.channel_id, channel_id)
        np.testing.assert_allclose(
            np.linalg.norm(recorder.layout.local_normal, axis=1), 1.0
        )
        np.testing.assert_array_equal(
            recorder.region_mask(regions=("left_arm",)), region_id == 1
        )
        for taxel_geom in self.geom_ids:
            np.testing.assert_array_equal(
                recorder.geom_indices(taxel_geom),
                np.flatnonzero(generated.geom_id == taxel_geom),
            )
        self.assertEqual(recorder.geom_indices(-1).shape, (0,))

        region_id.fill(-1)
        self.assertTrue(np.all(recorder.layout.region_id >= 0))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            TactileRecorder(
                self.model,
                robot_root="robot",
                geom_ids=self.geom_ids,
                layout=external,
            )

    def test_rejects_invalid_layout_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "spacing"):
            TactileRecorder(
                self.model,
                robot_root="robot",
                geom_ids=self.geom_ids,
                spacing=0.0,
            )
        with self.assertRaisesRegex(ValueError, "neighbors"):
            TactileRecorder(
                self.model,
                robot_root="robot",
                geom_ids=self.geom_ids,
                neighbors=0,
            )

    def test_external_hardware_channels_are_zero_based_and_unique(self) -> None:
        generated = TactileRecorder(
            self.model,
            robot_root="robot",
            geom_ids=self.geom_ids,
            spacing=0.05,
        ).layout

        def external(region_id: np.ndarray, channel_id: np.ndarray) -> TaxelLayout:
            count = len(region_id)
            return TaxelLayout(
                body_id=generated.body_id[:count],
                geom_id=generated.geom_id[:count],
                local_center=generated.local_center[:count],
                local_normal=generated.local_normal[:count],
                region_id=region_id,
                channel_id=channel_id,
                region_names=("vest",),
            )

        with self.assertRaisesRegex(ValueError, "pairs must be unique"):
            TactileRecorder(
                self.model,
                robot_root="robot",
                layout=external(
                    np.array((0, 0), dtype=np.int16),
                    np.array((0, 0), dtype=np.int32),
                ),
            )
        with self.assertRaisesRegex(ValueError, "raw slot 0..255"):
            TactileRecorder(
                self.model,
                robot_root="robot",
                layout=external(
                    np.array((0,), dtype=np.int16),
                    np.array((256,), dtype=np.int32),
                ),
            )
        with self.assertRaisesRegex(ValueError, "mapped together"):
            TactileRecorder(
                self.model,
                robot_root="robot",
                layout=external(
                    np.array((0,), dtype=np.int16),
                    np.array((-1,), dtype=np.int32),
                ),
            )


class TactileForceTest(unittest.TestCase):
    def _one_step(
        self,
        *,
        robot_first: bool,
        slide_velocity: float = 0.0,
    ) -> tuple[
        mujoco.MjModel,
        mujoco.MjData,
        TactileRecorder,
        int,
        int,
        np.ndarray,
    ]:
        model, robot_geom, object_geom = contact_model(robot_first=robot_first)
        data = mujoco.MjData(model)
        data.qvel[0] = slide_velocity
        recorder = TactileRecorder(
            model,
            robot_root="robot",
            geom_ids=(robot_geom,),
            spacing=0.10,
            neighbors=4,
        )
        recorder.begin()
        mujoco.mj_step(model, data)
        expected = robot_force(model, data, robot_geom)
        self.assertGreater(data.ncon, 0)
        self.assertGreater(np.linalg.norm(expected), 0.0)
        recorder.update(data)
        return (
            model,
            data,
            recorder,
            robot_geom,
            object_geom,
            expected,
        )

    def test_boundary_contact_splits_without_losing_force(self) -> None:
        model, _, recorder, robot_geom, object_geom, expected = self._one_step(
            robot_first=True
        )
        frame = recorder.finish()
        active = np.flatnonzero(frame.contact)

        self.assertEqual(len(active), 4)
        np.testing.assert_allclose(
            frame.layout.local_normal[active],
            np.tile((0.0, 0.0, 1.0), (4, 1)),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            frame.normal_force[active],
            np.full(4, frame.normal_force[active].sum() / 4.0),
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(frame.force.sum(axis=0), expected, rtol=1e-7)
        np.testing.assert_allclose(
            frame.impulse.sum(axis=0),
            expected * model.opt.timestep,
            rtol=1e-7,
        )
        self.assertLess(expected[2], 0.0)
        self.assertAlmostEqual(frame.normal_force.sum(), -expected[2], places=7)
        np.testing.assert_array_equal(
            frame.other_geom_id[active], np.full(4, object_geom)
        )
        np.testing.assert_array_equal(
            frame.layout.geom_id[active], np.full(4, robot_geom)
        )
        np.testing.assert_array_equal(frame.sample_count[active], np.ones(4))
        self.assertEqual(frame.mapped_contact_count, 1)
        self.assertEqual(frame.unmapped_contact_count, 0)

        recorder.reset()
        self.assertFalse(np.any(recorder.last_frame.contact))
        self.assertEqual(recorder.last_frame.duration, 0.0)
        self.assertEqual(recorder.last_frame.mapped_contact_count, 0)

    def test_force_direction_is_correct_for_either_contact_side(self) -> None:
        for robot_first in (True, False):
            with self.subTest(robot_first=robot_first):
                _, _, recorder, _, _, expected = self._one_step(robot_first=robot_first)
                frame = recorder.finish()
                np.testing.assert_allclose(
                    frame.force.sum(axis=0), expected, rtol=1e-7, atol=1e-9
                )
                self.assertLess(frame.force[:, 2].sum(), 0.0)

    def test_tangent_force_and_vector_reconstruction_are_conservative(self) -> None:
        model, _, recorder, _, _, expected = self._one_step(
            robot_first=True, slide_velocity=0.8
        )
        frame = recorder.finish()
        active = frame.contact
        normals = frame.layout.local_normal
        reconstructed = -frame.normal_force[:, None] * normals + frame.tangent_force

        self.assertGreater(np.linalg.norm(expected[:2]), 0.0)
        np.testing.assert_allclose(frame.force, reconstructed, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(frame.force.sum(axis=0), expected, rtol=1e-7)
        np.testing.assert_allclose(
            np.sum(frame.tangent_force[active] * normals[active], axis=1),
            0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            frame.tangent_impulse,
            frame.tangent_force * frame.duration,
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            frame.normal_impulse,
            frame.normal_force * frame.duration,
            rtol=1e-7,
        )
        self.assertAlmostEqual(frame.duration, model.opt.timestep)

    def test_multiple_physics_samples_preserve_integrated_impulse(self) -> None:
        model, robot_geom, _ = contact_model()
        data = mujoco.MjData(model)
        recorder = TactileRecorder(
            model,
            robot_root="robot",
            geom_ids=(robot_geom,),
            spacing=0.10,
        )
        expected_impulse = np.zeros(3)
        step_normal_force = []
        recorder.begin()
        for _ in range(3):
            mujoco.mj_step(model, data)
            expected_impulse += (
                robot_force(model, data, robot_geom) * model.opt.timestep
            )
            step_normal_force.append(recorder.update(data))
        frame = recorder.finish()

        np.testing.assert_allclose(
            frame.impulse.sum(axis=0), expected_impulse, rtol=1e-7, atol=1e-9
        )
        np.testing.assert_allclose(
            frame.force.sum(axis=0),
            expected_impulse / frame.duration,
            rtol=1e-7,
            atol=1e-9,
        )
        self.assertAlmostEqual(frame.duration, 3.0 * model.opt.timestep)
        self.assertTrue(np.all(frame.sample_count[frame.contact] == 3))
        np.testing.assert_allclose(
            frame.normal_force,
            np.mean(step_normal_force, axis=0),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            recorder.step_normal_force,
            step_normal_force[-1],
        )

    def test_unselected_robot_contact_is_reported_as_unmapped(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <option gravity="0 0 0" timestep="0.002"/>
              <worldbody>
                <body name="robot">
                  <geom name="selected_skin" type="box" pos="-1 0 0"
                        size="0.1 0.1 0.1"/>
                  <geom name="unselected_skin" type="box"
                        size="0.1 0.1 0.1"/>
                </body>
                <body name="object" pos="0 0 0.19">
                  <freejoint/>
                  <geom type="sphere" size="0.1" mass="1"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        recorder = TactileRecorder(
            model,
            robot_root="robot",
            geom_ids=(geom_id(model, "selected_skin"),),
        )
        recorder.begin()
        mujoco.mj_step(model, data)
        recorder.update(data)
        frame = recorder.finish()

        self.assertFalse(np.any(frame.contact))
        self.assertEqual(frame.mapped_contact_count, 0)
        self.assertEqual(frame.unmapped_contact_count, 1)

    def test_same_body_shell_outside_skin_radius_stays_unmapped(self) -> None:
        model, covered, uncovered = uncovered_shell_model()
        data = mujoco.MjData(model)
        body_id = int(model.geom_bodyid[covered])
        layout = TaxelLayout(
            body_id=np.array((body_id,), dtype=np.int32),
            geom_id=np.array((covered,), dtype=np.int32),
            local_center=np.array(((0.0, 0.0, 0.1),)),
            local_normal=np.array(((0.0, 0.0, 1.0),)),
            region_id=np.array((0,), dtype=np.int16),
            channel_id=np.array((0,), dtype=np.int32),
            region_names=("vest",),
        )
        recorder = TactileRecorder(
            model,
            robot_root="robot",
            layout=layout,
            mapping_radius=0.04,
        )

        recorder.begin()
        mujoco.mj_step(model, data)
        self.assertGreater(data.ncon, 0)
        self.assertGreater(len(recorder.geom_indices(uncovered)), 0)
        recorder.update(data)
        frame = recorder.finish()

        self.assertFalse(np.any(frame.contact))
        self.assertEqual(frame.mapped_contact_count, 0)
        self.assertEqual(frame.unmapped_contact_count, 1)

    def test_robot_arm_on_vest_contact_activates_both_skin_sides(self) -> None:
        model, vest_geom, arm_geom = self_contact_model()
        data = mujoco.MjData(model)
        vest_body = int(model.geom_bodyid[vest_geom])
        arm_body = int(model.geom_bodyid[arm_geom])
        layout = TaxelLayout(
            body_id=np.array((vest_body, arm_body), dtype=np.int32),
            geom_id=np.array((vest_geom, arm_geom), dtype=np.int32),
            local_center=np.array(((0.0, 0.0, 0.1), (0.0, 0.0, -0.1))),
            local_normal=np.array(((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            region_id=np.array((0, 1), dtype=np.int16),
            channel_id=np.array((0, 0), dtype=np.int32),
            region_names=("vest", "left_arm"),
        )
        recorder = TactileRecorder(
            model,
            robot_root="robot",
            layout=layout,
            mapping_radius=0.03,
        )

        recorder.begin()
        mujoco.mj_step(model, data)
        self.assertGreater(data.ncon, 0)
        recorder.update(data)
        frame = recorder.finish()

        np.testing.assert_array_equal(frame.contact, (True, True))
        self.assertEqual(frame.mapped_contact_count, 2)
        self.assertEqual(frame.unmapped_contact_count, 0)
        self.assertTrue(np.all(frame.normal_force > 0.0))
        self.assertAlmostEqual(frame.normal_force[0], frame.normal_force[1])
        np.testing.assert_array_equal(
            frame.other_body_id,
            (arm_body, vest_body),
        )


if __name__ == "__main__":
    unittest.main()
