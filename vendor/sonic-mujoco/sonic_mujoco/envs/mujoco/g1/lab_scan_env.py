"""G1 physics in collision geometry derived from the lab scan."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ....scan import RelativeCameraMapper, mujoco_camera_pose
from .g1_env import MujocoG1Env

CollisionGeometry = Literal["auto", "mesh", "boxes"]

_MESH_MANIFEST_SCHEMA = "sonic_mujoco.scan_collision_convex.v1"
_MESH_MANIFEST_VERSION = 2
_MESH_FIELDS = {
    "name",
    "file",
    "center_scan",
    "vertex_count",
    "face_count",
    "aabb_size_m",
    "sha256",
}
_MESH_COORDINATE_FRAME = {
    "mesh_vertex_units": "meters",
    "mesh_vertices_are_local": True,
    "mesh_axes": "scan_axes",
    "mesh_local_origin": "center_scan",
    "center_units": "scan_units",
}
_COLLISION_PATH_ENV = "SONIC_RECONSTRUCTION_COLLISION"


def _default_collision_path(package_root: Path) -> Path:
    configured = os.environ.get(_COLLISION_PATH_ENV)
    path = (
        Path(configured).expanduser()
        if configured
        else package_root / "assets" / "reconstruction" / "lab_scan_collision.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "lab reconstruction collision manifest not found; pass collision_path "
            f"or set {_COLLISION_PATH_ENV}"
        )
    return path


def _vector(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite numbers")
    if positive and np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain positive numbers")
    return vector


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"collision manifest must contain an object: {path}")
    return manifest


def _relative_asset_path(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{name} must be a relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the reconstruction directory") from error
    return path


def _finite_array(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values with shape {shape}")
    return array


def _matching_scale(primary: dict[str, object], mesh: dict[str, object]) -> float:
    primary_scale = float(primary["scan_units_per_meter"])
    mesh_scale = float(mesh["scan_units_per_meter"])
    if (
        not np.isfinite(primary_scale)
        or not np.isfinite(mesh_scale)
        or primary_scale <= 0.0
        or mesh_scale <= 0.0
        or not np.isclose(primary_scale, mesh_scale, rtol=1e-9, atol=1e-12)
    ):
        raise ValueError(
            "mesh manifest scan_units_per_meter does not match the primary manifest"
        )
    return mesh_scale


def _validate_matching_alignment(
    primary: dict[str, object],
    mesh: dict[str, object],
) -> None:
    primary_anchor = _finite_array(
        primary.get("anchor_camera_to_world"),
        shape=(4, 4),
        name="primary anchor_camera_to_world",
    )
    mesh_anchor = _finite_array(
        mesh.get("anchor_camera_to_world"),
        shape=(4, 4),
        name="mesh anchor_camera_to_world",
    )
    if not np.allclose(primary_anchor, mesh_anchor, rtol=1e-9, atol=1e-9):
        raise ValueError(
            "mesh manifest anchor_camera_to_world does not match the primary manifest"
        )

    primary_up = _vector(primary.get("scan_up"), name="primary scan_up")
    mesh_up = _vector(mesh.get("scan_up"), name="mesh scan_up")
    primary_up_norm = np.linalg.norm(primary_up)
    mesh_up_norm = np.linalg.norm(mesh_up)
    if primary_up_norm < 1e-12 or mesh_up_norm < 1e-12:
        raise ValueError("scan_up vectors must be non-zero")
    primary_up /= primary_up_norm
    mesh_up /= mesh_up_norm
    if not np.allclose(primary_up, mesh_up, rtol=1e-9, atol=1e-9):
        raise ValueError("mesh manifest scan_up does not match the primary manifest")

    primary_plane = _finite_array(
        primary.get("floor_plane"),
        shape=(4,),
        name="primary floor_plane",
    )
    mesh_plane = _finite_array(
        mesh.get("floor_plane"),
        shape=(4,),
        name="mesh floor_plane",
    )
    primary_plane_norm = np.linalg.norm(primary_plane[:3])
    mesh_plane_norm = np.linalg.norm(mesh_plane[:3])
    if primary_plane_norm < 1e-12 or mesh_plane_norm < 1e-12:
        raise ValueError("floor_plane normals must be non-zero")
    primary_plane /= primary_plane_norm
    mesh_plane /= mesh_plane_norm
    if np.dot(primary_plane[:3], mesh_plane[:3]) < 0.0:
        mesh_plane *= -1.0
    if not np.allclose(primary_plane, mesh_plane, rtol=1e-9, atol=1e-9):
        raise ValueError(
            "mesh manifest floor_plane does not match the primary manifest"
        )


def _validate_mesh_manifest(
    primary: dict[str, object],
    mesh: dict[str, object],
) -> None:
    if mesh.get("schema") != _MESH_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported collision mesh schema: {mesh.get('schema')!r}")
    version = mesh.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != _MESH_MANIFEST_VERSION
    ):
        raise ValueError(f"unsupported collision mesh manifest version: {version!r}")

    coordinate_frame = mesh.get("coordinate_frame")
    if not isinstance(coordinate_frame, dict):
        raise TypeError("mesh manifest coordinate_frame must be an object")
    for field, expected in _MESH_COORDINATE_FRAME.items():
        if coordinate_frame.get(field) != expected:
            raise ValueError(
                f"mesh manifest coordinate_frame.{field} must be {expected!r}"
            )
    mesh_scale = _matching_scale(primary, mesh)
    coordinate_scale = float(coordinate_frame.get("scan_units_per_meter", np.nan))
    if not np.isclose(coordinate_scale, mesh_scale, rtol=1e-9, atol=1e-12):
        raise ValueError(
            "mesh manifest coordinate_frame.scan_units_per_meter is inconsistent"
        )

    scale_calibration = mesh.get("scale_calibration")
    if not isinstance(scale_calibration, dict):
        raise TypeError("mesh manifest scale_calibration must be an object")
    scale_status = scale_calibration.get("status")
    if scale_status not in {"estimated", "metric"}:
        raise ValueError("mesh manifest scale calibration status is invalid")
    primary_scale_calibration = primary.get("scale_calibration")
    if (
        not isinstance(primary_scale_calibration, dict)
        or primary_scale_calibration.get("status") != scale_status
    ):
        raise ValueError(
            "mesh manifest scale calibration status does not match the primary manifest"
        )
    if coordinate_frame.get("scale_status") != scale_status:
        raise ValueError("mesh manifest coordinate_frame.scale_status is inconsistent")
    expected_occupancy_units = (
        "meters" if scale_status == "metric" else "pseudo_metric_meters"
    )
    if coordinate_frame.get("occupancy_units") != expected_occupancy_units:
        raise ValueError(
            "mesh manifest coordinate_frame.occupancy_units is inconsistent"
        )

    _validate_matching_alignment(primary, mesh)
    validation = mesh.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("mesh manifest validation.passed must be true")
    combined_hash = validation.get("combined_mesh_sha256")
    if (
        not isinstance(combined_hash, str)
        or len(combined_hash) != 64
        or any(character not in "0123456789abcdef" for character in combined_hash)
    ):
        raise ValueError(
            "mesh manifest validation.combined_mesh_sha256 must be a SHA-256 digest"
        )


def _mesh_collision_manifest(
    primary: dict[str, object],
    primary_path: Path,
) -> tuple[dict[str, object], Path, str | None]:
    reference = primary.get("mesh_manifest")
    if reference is None:
        return primary, primary_path, None
    path = _relative_asset_path(
        primary_path.parent.resolve(),
        reference,
        name="mesh_manifest",
    )
    if not path.is_file():
        raise FileNotFoundError(f"collision mesh manifest is missing: {path}")
    mesh = _read_manifest(path)
    _validate_mesh_manifest(primary, mesh)
    return mesh, path, _file_sha256(path)


def _validated_meshes(
    collision: dict[str, object],
    collision_path: Path,
) -> list[tuple[dict[str, object], Path]]:
    version = collision.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 2:
        raise ValueError("collision meshes require a version 2 manifest")

    meshes = collision.get("meshes")
    if not isinstance(meshes, list) or not meshes:
        raise ValueError("version 2 collision manifest does not define meshes")

    root = collision_path.parent.resolve()
    names: set[str] = set()
    validated = []
    combined_records = []
    for index, mesh in enumerate(meshes):
        label = f"meshes[{index}]"
        if not isinstance(mesh, dict):
            raise TypeError(f"{label} must be an object")
        missing = _MESH_FIELDS.difference(mesh)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"{label} is missing required fields: {fields}")

        name = mesh["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}.name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate collision mesh name: {name}")
        names.add(name)

        mesh_path = _relative_asset_path(
            root,
            mesh["file"],
            name=f"{label}.file",
        )
        if not mesh_path.is_file():
            raise FileNotFoundError(f"collision mesh asset is missing: {mesh_path}")

        _vector(mesh["center_scan"], name=f"{label}.center_scan")
        _vector(mesh["aabb_size_m"], name=f"{label}.aabb_size_m", positive=True)
        for field in ("vertex_count", "face_count"):
            count = mesh[field]
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(f"{label}.{field} must be a positive integer")

        expected_hash = mesh["sha256"]
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 digest")
        actual_hash = _file_sha256(mesh_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"collision mesh hash mismatch for {mesh_path.name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        validated.append((mesh, mesh_path))
        combined_records.append((mesh["file"], actual_hash))
    validation = collision.get("validation")
    if isinstance(validation, dict) and "combined_mesh_sha256" in validation:
        actual_combined_hash = hashlib.sha256(
            json.dumps(combined_records, separators=(",", ":")).encode()
        ).hexdigest()
        if actual_combined_hash != validation["combined_mesh_sha256"]:
            raise ValueError("collision mesh combined hash mismatch")
    return validated


def _camera_mapper(
    spec: mujoco.MjSpec,
    collision: dict[str, object],
    scan_units_per_meter: float,
) -> RelativeCameraMapper:
    anchor_model = spec.compile()
    anchor_data = mujoco.MjData(anchor_model)
    mujoco.mj_forward(anchor_model, anchor_data)
    return RelativeCameraMapper(
        collision["anchor_camera_to_world"],
        mujoco_camera_pose(anchor_model, anchor_data, "head_camera_scan"),
        scan_units_per_meter=scan_units_per_meter,
        scan_up=collision["scan_up"],
    )


def _add_box_collisions(
    spec: mujoco.MjSpec,
    collision: dict[str, object],
    mapper: RelativeCameraMapper,
    scan_units_per_meter: float,
) -> None:
    boxes = collision.get("boxes")
    if not isinstance(boxes, list):
        raise TypeError("collision manifest does not define boxes")
    for box in boxes:
        rotation = mapper.scan_rotation_to_mujoco(box["rotation"])
        quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
        quaternion_wxyz = np.roll(quaternion_xyzw, 1)
        spec.worldbody.add_geom(
            name=box["name"],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=mapper.scan_points_to_mujoco(box["center"]),
            quat=quaternion_wxyz,
            size=np.asarray(box["size"], dtype=np.float64)
            / (2.0 * scan_units_per_meter),
            contype=1,
            conaffinity=1,
            friction=(0.8, 0.02, 0.002),
            rgba=(0.15, 0.45, 0.9, 0.0),
            group=3,
        )


def _add_mesh_collisions(
    spec: mujoco.MjSpec,
    meshes: list[tuple[dict[str, object], Path]],
    mapper: RelativeCameraMapper,
) -> None:
    rotation = mapper.scan_rotation_to_mujoco(np.eye(3, dtype=np.float64))
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    quaternion_wxyz = np.roll(quaternion_xyzw, 1)
    for mesh, mesh_path in meshes:
        name = str(mesh["name"])
        asset_name = f"{name}_asset"
        spec.add_mesh(name=asset_name, file=str(mesh_path), scale=(1.0, 1.0, 1.0))
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            pos=mapper.scan_points_to_mujoco(mesh["center_scan"]),
            quat=quaternion_wxyz,
            meshname=asset_name,
            contype=1,
            conaffinity=1,
            friction=(0.8, 0.02, 0.002),
            rgba=(0.15, 0.45, 0.9, 0.0),
            group=3,
        )


def _compile_collision_model(
    scene: Path,
    primary: dict[str, object],
    collision_geometry: Literal["mesh", "boxes"],
    scan_units_per_meter: float,
    *,
    mesh_collision: dict[str, object] | None = None,
    mesh_collision_path: Path | None = None,
) -> mujoco.MjModel:
    meshes = None
    if collision_geometry == "mesh":
        if mesh_collision is None or mesh_collision_path is None:
            raise ValueError("mesh collision manifest was not loaded")
        meshes = _validated_meshes(mesh_collision, mesh_collision_path)

    spec = mujoco.MjSpec.from_file(str(scene))
    mapper = _camera_mapper(spec, primary, scan_units_per_meter)
    if collision_geometry == "mesh":
        assert meshes is not None
        _add_mesh_collisions(spec, meshes, mapper)
    else:
        _add_box_collisions(spec, primary, mapper, scan_units_per_meter)
    return spec.compile()


class MujocoG1LabScanEnv(MujocoG1Env):
    """G1 with static collisions extracted from the lab 3DGS."""

    def __init__(
        self,
        timestep: float = 0.005,
        collision_geometry: CollisionGeometry = "auto",
        *,
        collision_path: str | Path | None = None,
    ) -> None:
        if collision_geometry not in {"auto", "mesh", "boxes"}:
            raise ValueError("collision_geometry must be 'auto', 'mesh', or 'boxes'")

        package_root = Path(__file__).resolve().parents[3]
        scene = package_root / "assets" / "mujoco" / "scenes" / "g1" / "empty.xml"
        if collision_path is None:
            collision_path = _default_collision_path(package_root)
        collision_path = Path(collision_path).resolve()
        collision = _read_manifest(collision_path)
        self.reconstruction_metadata = collision
        self.collision_manifest_sha256 = _file_sha256(collision_path)
        self.scan_units_per_meter = float(collision["scan_units_per_meter"])
        if not np.isfinite(self.scan_units_per_meter) or self.scan_units_per_meter <= 0:
            raise ValueError("scan_units_per_meter must be positive and finite")

        default_geometry = collision.get("default_collision_geometry", "boxes")
        if not isinstance(default_geometry, str) or default_geometry not in {
            "mesh",
            "boxes",
        }:
            raise ValueError("default_collision_geometry must be 'mesh' or 'boxes'")
        selected_geometry = (
            default_geometry if collision_geometry == "auto" else collision_geometry
        )
        mesh_manifest_sha256 = None

        try:
            mesh_collision = None
            mesh_collision_path = None
            candidate_mesh_hash = None
            if selected_geometry == "mesh":
                (
                    mesh_collision,
                    mesh_collision_path,
                    candidate_mesh_hash,
                ) = _mesh_collision_manifest(collision, collision_path)
            model = _compile_collision_model(
                scene,
                collision,
                selected_geometry,
                self.scan_units_per_meter,
                mesh_collision=mesh_collision,
                mesh_collision_path=mesh_collision_path,
            )
            if selected_geometry == "mesh":
                mesh_manifest_sha256 = candidate_mesh_hash
        except Exception as error:
            if collision_geometry != "auto" or selected_geometry != "mesh":
                raise
            warnings.warn(
                "scan collision meshes are unavailable or invalid; "
                f"falling back to boxes ({error})",
                RuntimeWarning,
                stacklevel=2,
            )
            selected_geometry = "boxes"
            model = _compile_collision_model(
                scene,
                collision,
                selected_geometry,
                self.scan_units_per_meter,
            )

        self.collision_geometry = selected_geometry
        self.mesh_manifest_sha256 = mesh_manifest_sha256
        super().__init__(scene, timestep, model=model)
