import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import mujoco

from sonic_mujoco.envs.mujoco.g1 import MujocoG1LabScanEnv


class LabScanEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.collision_path = (
            Path(__file__).parents[1] / "tests/fixtures/lab_scan_collision.json"
        )
        cls.env = MujocoG1LabScanEnv(
            collision_geometry="boxes",
            collision_path=cls.collision_path,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_loads_scan_derived_collision_boxes(self) -> None:
        names = {
            mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, index)
            for index in range(self.env.model.ngeom)
        }

        collisions = {
            name for name in names if name and name.startswith("scan_collision_")
        }
        expected = len(json.loads(self.collision_path.read_text())["boxes"])
        self.assertEqual(len(collisions), expected)

    def test_rejects_unknown_collision_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "collision_geometry"):
            MujocoG1LabScanEnv(collision_geometry="triangles")

    def test_strict_mesh_mode_rejects_a_missing_mesh_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_manifest(Path(directory), meshes=None)

            with self.assertRaisesRegex(ValueError, "does not define meshes"):
                MujocoG1LabScanEnv(
                    collision_geometry="mesh",
                    collision_path=manifest,
                )

    def test_loads_valid_v2_convex_meshes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_manifest(Path(directory))
            env = MujocoG1LabScanEnv(
                collision_geometry="mesh",
                collision_path=manifest,
            )
            self.addCleanup(env.close)

            geom_id = mujoco.mj_name2id(
                env.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "scan_collision_hull_000",
            )
            self.assertGreaterEqual(geom_id, 0)
            self.assertEqual(
                env.model.geom_type[geom_id],
                mujoco.mjtGeom.mjGEOM_MESH,
            )
            self.assertEqual(env.collision_geometry, "mesh")
            self.assertIsNone(env.mesh_manifest_sha256)

    def test_auto_warns_and_uses_fresh_box_spec_for_bad_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_manifest(
                Path(directory),
                mesh_contents="this is not an OBJ\n",
                default_collision_geometry="mesh",
            )
            with self.assertWarnsRegex(RuntimeWarning, "falling back to boxes"):
                env = MujocoG1LabScanEnv(
                    collision_geometry="auto",
                    collision_path=manifest,
                )
            self.addCleanup(env.close)

            self.assertEqual(env.collision_geometry, "boxes")
            self.assertEqual(
                mujoco.mj_name2id(
                    env.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    "scan_collision_hull_000",
                ),
                -1,
            )

    def test_strict_mesh_mode_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_manifest(
                Path(directory),
                sha256="0" * 64,
            )

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                MujocoG1LabScanEnv(
                    collision_geometry="mesh",
                    collision_path=manifest,
                )

    def test_auto_keeps_v1_box_manifest_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_manifest(
                Path(directory),
                version=1,
                meshes=None,
            )
            with warnings.catch_warnings(record=True) as caught:
                env = MujocoG1LabScanEnv(
                    collision_geometry="auto",
                    collision_path=manifest,
                )
            self.addCleanup(env.close)

            self.assertEqual(caught, [])
            self.assertEqual(env.collision_geometry, "boxes")

    def test_auto_safely_defaults_v2_inline_manifest_to_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_manifest(Path(directory))
            with warnings.catch_warnings(record=True) as caught:
                env = MujocoG1LabScanEnv(
                    collision_geometry="auto",
                    collision_path=manifest,
                )
            self.addCleanup(env.close)

            self.assertEqual(caught, [])
            self.assertEqual(env.collision_geometry, "boxes")
            self.assertIsNone(env.mesh_manifest_sha256)

    def test_auto_loads_valid_relative_mesh_manifest_when_defaulted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, sidecar = self._write_sidecar_manifest(Path(directory))
            env = MujocoG1LabScanEnv(
                collision_geometry="auto",
                collision_path=manifest,
            )
            self.addCleanup(env.close)

            geom_id = mujoco.mj_name2id(
                env.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "scan_collision_hull_000",
            )
            self.assertGreaterEqual(geom_id, 0)
            self.assertEqual(
                env.model.geom_type[geom_id],
                mujoco.mjtGeom.mjGEOM_MESH,
            )
            self.assertEqual(env.collision_geometry, "mesh")
            self.assertEqual(
                env.mesh_manifest_sha256,
                hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )

    def test_auto_boxes_does_not_read_mesh_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, sidecar = self._write_sidecar_manifest(
                Path(directory),
                default_collision_geometry="boxes",
            )
            sidecar.unlink()
            with warnings.catch_warnings(record=True) as caught:
                env = MujocoG1LabScanEnv(
                    collision_geometry="auto",
                    collision_path=manifest,
                )
            self.addCleanup(env.close)

            self.assertEqual(caught, [])
            self.assertEqual(env.collision_geometry, "boxes")
            self.assertIsNone(env.mesh_manifest_sha256)

    def test_explicit_mesh_overrides_primary_box_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, sidecar = self._write_sidecar_manifest(
                Path(directory),
                default_collision_geometry="boxes",
            )
            env = MujocoG1LabScanEnv(
                collision_geometry="mesh",
                collision_path=manifest,
            )
            self.addCleanup(env.close)

            self.assertEqual(env.collision_geometry, "mesh")
            self.assertEqual(
                env.mesh_manifest_sha256,
                hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )

    def test_auto_bad_sidecar_warns_and_falls_back_to_primary_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, sidecar = self._write_sidecar_manifest(Path(directory))
            mesh = json.loads(sidecar.read_text())
            mesh["validation"]["passed"] = False
            sidecar.write_text(json.dumps(mesh))

            with self.assertWarnsRegex(RuntimeWarning, "falling back to boxes"):
                env = MujocoG1LabScanEnv(
                    collision_geometry="auto",
                    collision_path=manifest,
                )
            self.addCleanup(env.close)

            self.assertEqual(env.collision_geometry, "boxes")
            self.assertIsNone(env.mesh_manifest_sha256)
            self.assertEqual(
                mujoco.mj_name2id(
                    env.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    "scan_collision_hull_000",
                ),
                -1,
            )

    def test_strict_sidecar_rejects_incompatible_metadata(self) -> None:
        cases = (
            (
                "schema",
                "schema",
                lambda mesh: mesh.__setitem__("schema", "unknown"),
            ),
            (
                "version",
                "version",
                lambda mesh: mesh.__setitem__("version", 3),
            ),
            (
                "coordinate frame",
                "coordinate_frame.mesh_axes",
                lambda mesh: mesh["coordinate_frame"].__setitem__(
                    "mesh_axes", "floor_axes"
                ),
            ),
            (
                "scale",
                "scan_units_per_meter",
                lambda mesh: mesh.__setitem__("scan_units_per_meter", 0.25),
            ),
            (
                "scale status",
                "scale calibration status",
                lambda mesh: (
                    mesh["scale_calibration"].__setitem__("status", "metric"),
                    mesh["coordinate_frame"].__setitem__("scale_status", "metric"),
                    mesh["coordinate_frame"].__setitem__("occupancy_units", "meters"),
                ),
            ),
            (
                "anchor",
                "anchor_camera_to_world",
                lambda mesh: mesh["anchor_camera_to_world"][0].__setitem__(3, 10.0),
            ),
            (
                "up",
                "scan_up",
                lambda mesh: mesh.__setitem__("scan_up", [0.0, 1.0, 0.0]),
            ),
            (
                "floor",
                "floor_plane",
                lambda mesh: mesh["floor_plane"].__setitem__(3, 10.0),
            ),
            (
                "validation",
                "validation.passed",
                lambda mesh: mesh["validation"].__setitem__("passed", False),
            ),
            (
                "combined hash",
                "combined hash mismatch",
                lambda mesh: mesh["validation"].__setitem__(
                    "combined_mesh_sha256", "0" * 64
                ),
            ),
        )
        for label, message, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                manifest, sidecar = self._write_sidecar_manifest(Path(directory))
                mesh = json.loads(sidecar.read_text())
                mutate(mesh)
                sidecar.write_text(json.dumps(mesh))

                with self.assertRaisesRegex((ValueError, TypeError), message):
                    MujocoG1LabScanEnv(
                        collision_geometry="mesh",
                        collision_path=manifest,
                    )

    def test_strict_sidecar_rejects_paths_outside_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._write_sidecar_manifest(root)
            collision = json.loads(manifest.read_text())
            collision["mesh_manifest"] = str(root / "absolute.json")
            manifest.write_text(json.dumps(collision))

            with self.assertRaisesRegex(ValueError, "relative path"):
                MujocoG1LabScanEnv(
                    collision_geometry="mesh",
                    collision_path=manifest,
                )

    def test_strict_sidecar_rejects_absolute_mesh_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, sidecar = self._write_sidecar_manifest(Path(directory))
            mesh = json.loads(sidecar.read_text())
            mesh["meshes"][0]["file"] = str(sidecar.parent / mesh["meshes"][0]["file"])
            sidecar.write_text(json.dumps(mesh))

            with self.assertRaisesRegex(ValueError, "relative path"):
                MujocoG1LabScanEnv(
                    collision_geometry="mesh",
                    collision_path=manifest,
                )

    def test_exposes_versioned_reconstruction_scale(self) -> None:
        metadata = self.env.reconstruction_metadata

        self.assertEqual(metadata["version"], 2)
        self.assertEqual(
            self.env.scan_units_per_meter,
            metadata["scan_units_per_meter"],
        )
        self.assertEqual(metadata["scale_calibration"]["status"], "estimated")
        self.assertFalse(metadata["geometry_provenance"]["training_depth_is_metric"])
        self.assertEqual(len(self.env.collision_manifest_sha256), 64)
        int(self.env.collision_manifest_sha256, 16)
        self.assertIsNone(self.env.mesh_manifest_sha256)

    def test_reset_places_robot_outside_scan_obstacles(self) -> None:
        self.env.reset()
        obstacle_contacts = []
        for index in range(self.env.data.ncon):
            contact = self.env.data.contact[index]
            names = (
                mujoco.mj_id2name(
                    self.env.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    contact.geom1,
                ),
                mujoco.mj_id2name(
                    self.env.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    contact.geom2,
                ),
            )
            if any(name and name.startswith("scan_collision_") for name in names):
                obstacle_contacts.append(names)

        self.assertEqual(obstacle_contacts, [])

    def _write_manifest(
        self,
        directory: Path,
        *,
        version: int = 2,
        meshes: object = True,
        default_collision_geometry: str | None = None,
        mesh_contents: str = (
            "v 0 0 0\n"
            "v 0.1 0 0\n"
            "v 0 0.1 0\n"
            "v 0 0 0.1\n"
            "f 1 3 2\n"
            "f 1 2 4\n"
            "f 1 4 3\n"
            "f 2 3 4\n"
        ),
        sha256: str | None = None,
    ) -> Path:
        collision = json.loads(self.collision_path.read_text())
        collision["version"] = version
        collision.pop("mesh_manifest", None)
        if default_collision_geometry is None:
            collision.pop("default_collision_geometry", None)
        else:
            collision["default_collision_geometry"] = default_collision_geometry
        if meshes is None:
            collision.pop("meshes", None)
        else:
            collision["meshes"] = [
                self._write_mesh(directory, mesh_contents, sha256=sha256)
            ]
        manifest = directory / "collision.json"
        manifest.write_text(json.dumps(collision))
        return manifest

    def _write_sidecar_manifest(
        self,
        directory: Path,
        *,
        default_collision_geometry: str = "mesh",
    ) -> tuple[Path, Path]:
        primary = json.loads(self.collision_path.read_text())
        primary["default_collision_geometry"] = default_collision_geometry
        primary["mesh_manifest"] = "collision_mesh.json"
        manifest = directory / "collision.json"
        manifest.write_text(json.dumps(primary))

        scale_status = primary["scale_calibration"]["status"]
        mesh = self._write_mesh(directory, self._mesh_contents())
        combined_hash = hashlib.sha256(
            json.dumps(
                [(mesh["file"], mesh["sha256"])],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        sidecar = {
            "schema": "sonic_mujoco.scan_collision_convex.v1",
            "version": 2,
            "scan_units_per_meter": primary["scan_units_per_meter"],
            "scale_calibration": {
                "status": scale_status,
                "method": "test",
                "has_measured_reference": scale_status == "metric",
            },
            "anchor_camera_to_world": primary["anchor_camera_to_world"],
            "scan_up": primary["scan_up"],
            "floor_plane": primary["floor_plane"],
            "coordinate_frame": {
                "mesh_vertex_units": "meters",
                "mesh_vertices_are_local": True,
                "mesh_axes": "scan_axes",
                "mesh_local_origin": "center_scan",
                "center_units": "scan_units",
                "scan_units_per_meter": primary["scan_units_per_meter"],
                "scale_status": scale_status,
                "occupancy_units": (
                    "meters" if scale_status == "metric" else "pseudo_metric_meters"
                ),
            },
            "validation": {
                "passed": True,
                "combined_mesh_sha256": combined_hash,
            },
            "meshes": [mesh],
        }
        sidecar_path = directory / "collision_mesh.json"
        sidecar_path.write_text(json.dumps(sidecar))
        return manifest, sidecar_path

    @staticmethod
    def _mesh_contents() -> str:
        return (
            "v 0 0 0\n"
            "v 0.1 0 0\n"
            "v 0 0.1 0\n"
            "v 0 0 0.1\n"
            "f 1 3 2\n"
            "f 1 2 4\n"
            "f 1 4 3\n"
            "f 2 3 4\n"
        )

    @staticmethod
    def _write_mesh(
        directory: Path,
        contents: str,
        *,
        sha256: str | None = None,
    ) -> dict:
        mesh_directory = directory / "meshes"
        mesh_directory.mkdir(exist_ok=True)
        mesh_path = mesh_directory / "hull.obj"
        mesh_path.write_text(contents)
        return {
            "name": "scan_collision_hull_000",
            "file": "meshes/hull.obj",
            "center_scan": [100.0, 100.0, 100.0],
            "vertex_count": 4,
            "face_count": 4,
            "aabb_size_m": [0.1, 0.1, 0.1],
            "sha256": sha256 or hashlib.sha256(contents.encode()).hexdigest(),
        }


if __name__ == "__main__":
    unittest.main()
