"""Logical surface taxels derived from MuJoCo contact forces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

_SUPPORTED_GEOMS = frozenset(
    {
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_MESH),
    }
)


@dataclass(frozen=True, slots=True)
class TaxelLayout:
    """Fixed taxel geometry expressed in each taxel's robot-body frame.

    ``channel_id`` is the zero-based raw-packet slot (0 through 255), not a
    physical grid position. ``(region_id, channel_id)`` uniquely identifies a
    mapped hardware taxel. Dynamically generated simulation-only taxels use
    ``-1`` for both fields.
    """

    body_id: np.ndarray
    geom_id: np.ndarray
    local_center: np.ndarray
    local_normal: np.ndarray
    region_id: np.ndarray
    channel_id: np.ndarray
    region_names: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.geom_id)


@dataclass(frozen=True, slots=True)
class TactileFrame:
    """Robot-skin forces accumulated over one control interval.

    Vector quantities are expressed in the local frame of ``layout.body_id``.
    The outward taxel normal is ``layout.local_normal``. Therefore a purely
    compressive force reconstructs as ``-normal_force * local_normal``.
    ``other_*_id`` identifies the object contributing the largest normal
    impulse to a taxel during the interval.
    """

    layout: TaxelLayout
    force: np.ndarray
    normal_force: np.ndarray
    tangent_force: np.ndarray
    impulse: np.ndarray
    normal_impulse: np.ndarray
    tangent_impulse: np.ndarray
    contact: np.ndarray
    other_body_id: np.ndarray
    other_geom_id: np.ndarray
    sample_count: np.ndarray
    duration: float
    mapped_contact_count: int
    unmapped_contact_count: int


@dataclass(frozen=True, slots=True)
class _TaxelRoute:
    indices: np.ndarray
    centers: np.ndarray
    normals: np.ndarray


@dataclass(frozen=True, slots=True)
class _ContactProjection:
    taxels: np.ndarray
    weights: np.ndarray
    force: np.ndarray
    other_body_id: int
    other_geom_id: int


class TactileRecorder:
    """Project existing MuJoCo contacts onto a fixed logical taxel layout.

    No geoms or constraints are added to the model. Analytic primitives are
    tiled directly; mesh geoms use deterministic surface sampling followed by
    voxel consolidation. Contact forces are distributed to nearby taxels with
    weights that sum to one, preserving force and impulse. An external
    body-local layout is routed from every collidable geom owned by a covered
    body, so multi-part robot shells do not create tactile blind spots.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        robot_root: str = "pelvis",
        *,
        spacing: float = 0.02,
        geom_ids: Sequence[int] | None = None,
        layout: TaxelLayout | None = None,
        neighbors: int = 4,
        max_taxels_per_geom: int = 2048,
        mapping_radius: float | np.ndarray | None = None,
    ) -> None:
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("taxel spacing must be finite and positive")
        if neighbors < 1:
            raise ValueError("taxel neighbors must be positive")
        if max_taxels_per_geom < 1:
            raise ValueError("max_taxels_per_geom must be positive")
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root)
        if root < 0:
            raise ValueError(f"missing robot root body: {robot_root}")

        self.model = model
        self.spacing = float(spacing)
        self.neighbors = int(neighbors)
        self.max_taxels_per_geom = int(max_taxels_per_geom)
        self.robot_body_ids = frozenset(
            body_id
            for body_id in range(model.nbody)
            if self._is_descendant(body_id, root)
        )
        self.body_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or f"body_{body_id}"
            for body_id in range(model.nbody)
        )
        self.geom_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            or f"geom_{geom_id}"
            for geom_id in range(model.ngeom)
        )
        if layout is not None and geom_ids is not None:
            raise ValueError("layout and geom_ids are mutually exclusive")
        if layout is None:
            selected = self._selected_geoms(geom_ids)
            self.layout, self._geom_taxels = self._build_layout(selected)
        else:
            self.layout, self._geom_taxels = self._use_layout(layout)
        if not len(self.layout):
            raise ValueError("robot has no supported collision geoms for taxels")
        self.mapping_radius = self._validate_mapping_radius(mapping_radius)

        self._flex_body_ids = tuple(
            self._flex_root_body(flex_id) for flex_id in range(model.nflex)
        )
        self._impulse = np.zeros((len(self.layout), 3), dtype=np.float64)
        self._normal_impulse = np.zeros(len(self.layout), dtype=np.float64)
        self._tangent_impulse = np.zeros((len(self.layout), 3), dtype=np.float64)
        self._step_normal_force = np.zeros(len(self.layout), dtype=np.float64)
        self._sample_count = np.zeros(len(self.layout), dtype=np.int32)
        self._contact = np.zeros(len(self.layout), dtype=bool)
        self._object_impulse: dict[tuple[int, int, int], float] = {}
        self._steps = 0
        self._mapped_contact_count = 0
        self._unmapped_contact_count = 0
        self.last_frame = self._empty_frame()

    def begin(self) -> None:
        """Start a new control interval."""

        self._impulse.fill(0.0)
        self._normal_impulse.fill(0.0)
        self._tangent_impulse.fill(0.0)
        self._step_normal_force.fill(0.0)
        self._sample_count.fill(0)
        self._contact.fill(False)
        self._object_impulse.clear()
        self._steps = 0
        self._mapped_contact_count = 0
        self._unmapped_contact_count = 0

    def reset(self) -> None:
        """Discard accumulated and previously finished tactile data."""

        self.begin()
        self.last_frame = self._empty_frame()

    def region_mask(
        self,
        *,
        regions: Sequence[str] = (),
        body_names: Sequence[str] = (),
        body_prefixes: Sequence[str] = (),
    ) -> np.ndarray:
        """Return a stable taxel mask for named robot-body regions."""

        requested_regions = frozenset(regions)
        names = frozenset(body_names)
        prefixes = tuple(body_prefixes)
        if not requested_regions and not names and not prefixes:
            raise ValueError("region mask needs regions, body_names, or body_prefixes")
        selected_bodies = np.asarray(
            [name in names or name.startswith(prefixes) for name in self.body_names],
            dtype=bool,
        )
        mask = selected_bodies[self.layout.body_id]
        if requested_regions:
            region_ids = {
                region_id
                for region_id, name in enumerate(self.layout.region_names)
                if name in requested_regions
            }
            mask |= np.isin(self.layout.region_id, tuple(region_ids))
        return mask

    def geom_indices(self, geom_id: int) -> np.ndarray:
        """Return layout indices routed from one MuJoCo contact geom."""

        entry = self._geom_taxels.get(int(geom_id))
        if entry is None:
            return np.empty(0, dtype=np.int32)
        return entry.indices.copy()

    def _validate_mapping_radius(
        self,
        value: float | np.ndarray | None,
    ) -> np.ndarray | None:
        if value is None:
            return None
        radius = np.asarray(value, dtype=np.float64)
        if radius.ndim == 0:
            radius = np.full(len(self.layout), float(radius), dtype=np.float64)
        if radius.shape != (len(self.layout),):
            raise ValueError("mapping_radius must be scalar or have shape (N,)")
        if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
            raise ValueError("mapping_radius must be finite and positive")
        radius = radius.copy()
        radius.setflags(write=False)
        return radius

    @property
    def step_normal_force(self) -> np.ndarray:
        """Normal force observed by the most recent physics-step update."""

        return self._step_normal_force.copy()

    def update(self, data: mujoco.MjData) -> np.ndarray:
        """Accumulate and return one physics step of normal contact force."""

        timestep = float(self.model.opt.timestep)
        self._step_normal_force.fill(0.0)
        contact_force = np.zeros(6, dtype=np.float64)
        touched: set[int] = set()
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            bodies = (self._contact_body(contact, 0), self._contact_body(contact, 1))
            robot_sides = self._robot_contact_sides(contact, bodies)
            if not robot_sides:
                continue

            mujoco.mj_contactForce(self.model, data, contact_id, contact_force)
            contact_frame = np.asarray(contact.frame).reshape(3, 3)
            for robot_side in robot_sides:
                projection = self._project_contact(
                    data,
                    contact,
                    bodies,
                    robot_side,
                    contact_force,
                    contact_frame,
                )
                if projection is None:
                    self._unmapped_contact_count += 1
                    continue
                self._mapped_contact_count += 1
                self._accumulate_projection(projection, timestep, touched)

        if touched:
            indices = np.fromiter(touched, dtype=np.int32)
            self._sample_count[indices] += 1
        self._steps += 1
        return self.step_normal_force

    def _robot_contact_sides(
        self,
        contact: mujoco.MjContact,
        bodies: tuple[int, int],
    ) -> tuple[int, ...]:
        sides = tuple(
            side
            for side, body_id in enumerate(bodies)
            if body_id in self.robot_body_ids
        )
        if len(sides) != 2:
            return sides
        return tuple(
            side for side in sides if int(contact.geom[side]) in self._geom_taxels
        )

    def _project_contact(
        self,
        data: mujoco.MjData,
        contact: mujoco.MjContact,
        bodies: tuple[int, int],
        robot_side: int,
        contact_force: np.ndarray,
        contact_frame: np.ndarray,
    ) -> _ContactProjection | None:
        robot_geom = int(contact.geom[robot_side])
        if robot_geom not in self._geom_taxels:
            return None

        force_world = contact_frame.T @ contact_force[:3]
        outward_world = contact_frame[0].copy()
        if robot_side == 0:
            force_world *= -1.0
        else:
            outward_world *= -1.0

        geom_rotation = np.asarray(data.geom_xmat[robot_geom]).reshape(3, 3)
        local_position = geom_rotation.T @ (
            np.asarray(contact.pos) - data.geom_xpos[robot_geom]
        )
        taxels, weights = self._nearby_taxels(
            robot_geom,
            local_position,
            geom_rotation.T @ outward_world,
        )
        if not len(taxels):
            return None

        robot_body = int(self.model.geom_bodyid[robot_geom])
        body_rotation = np.asarray(data.xmat[robot_body]).reshape(3, 3)
        other_side = 1 - robot_side
        return _ContactProjection(
            taxels=taxels,
            weights=weights,
            force=body_rotation.T @ force_world,
            other_body_id=bodies[other_side],
            other_geom_id=int(contact.geom[other_side]),
        )

    def _accumulate_projection(
        self,
        projection: _ContactProjection,
        timestep: float,
        touched: set[int],
    ) -> None:
        for taxel, weight in zip(projection.taxels, projection.weights, strict=True):
            index = int(taxel)
            weighted_force = float(weight) * projection.force
            normal = self.layout.local_normal[index]
            normal_force = max(0.0, -float(np.dot(weighted_force, normal)))
            tangent_force = weighted_force + normal_force * normal
            normal_impulse = normal_force * timestep

            self._impulse[index] += weighted_force * timestep
            self._normal_impulse[index] += normal_impulse
            self._tangent_impulse[index] += tangent_force * timestep
            self._step_normal_force[index] += normal_force
            self._contact[index] = True
            touched.add(index)
            key = index, projection.other_body_id, projection.other_geom_id
            self._object_impulse[key] = (
                self._object_impulse.get(key, 0.0) + normal_impulse
            )

    def finish(self) -> TactileFrame:
        """Finish the interval and return mean forces plus integrated impulses."""

        duration = self._steps * float(self.model.opt.timestep)
        if duration > 0.0:
            force = self._impulse / duration
            normal_force = self._normal_impulse / duration
            tangent_force = self._tangent_impulse / duration
        else:
            force = np.zeros_like(self._impulse)
            normal_force = np.zeros_like(self._normal_impulse)
            tangent_force = np.zeros_like(self._tangent_impulse)

        other_body_id = np.full(len(self.layout), -1, dtype=np.int32)
        other_geom_id = np.full(len(self.layout), -1, dtype=np.int32)
        dominant: dict[int, tuple[float, int, int]] = {}
        for (taxel, body_id, geom_id), impulse in self._object_impulse.items():
            candidate = impulse, -body_id, -geom_id
            if taxel not in dominant or candidate > dominant[taxel]:
                dominant[taxel] = candidate
                other_body_id[taxel] = body_id
                other_geom_id[taxel] = geom_id

        frame = TactileFrame(
            layout=self.layout,
            force=force.copy(),
            normal_force=normal_force.copy(),
            tangent_force=tangent_force.copy(),
            impulse=self._impulse.copy(),
            normal_impulse=self._normal_impulse.copy(),
            tangent_impulse=self._tangent_impulse.copy(),
            contact=self._contact.copy(),
            other_body_id=other_body_id,
            other_geom_id=other_geom_id,
            sample_count=self._sample_count.copy(),
            duration=duration,
            mapped_contact_count=self._mapped_contact_count,
            unmapped_contact_count=self._unmapped_contact_count,
        )
        self.last_frame = frame
        return frame

    def _selected_geoms(self, geom_ids: Sequence[int] | None) -> tuple[int, ...]:
        explicit = geom_ids is not None
        if geom_ids is None:
            geom_ids = (
                geom_id
                for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) in self.robot_body_ids
                and (
                    int(self.model.geom_contype[geom_id]) != 0
                    or int(self.model.geom_conaffinity[geom_id]) != 0
                )
            )

        selected = []
        for geom_id in sorted({int(value) for value in geom_ids}):
            if not 0 <= geom_id < self.model.ngeom:
                raise ValueError(f"invalid taxel geom id: {geom_id}")
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id not in self.robot_body_ids:
                raise ValueError(f"taxel geom {geom_id} is outside {self._root_name()}")
            geom_type = int(self.model.geom_type[geom_id])
            if geom_type not in _SUPPORTED_GEOMS:
                if explicit:
                    name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    ) or str(geom_id)
                    raise ValueError(f"unsupported taxel geom type for {name}")
                continue
            selected.append(geom_id)
        return tuple(selected)

    def _build_layout(
        self, geom_ids: Sequence[int]
    ) -> tuple[TaxelLayout, dict[int, _TaxelRoute]]:
        body_ids = []
        layout_geom_ids = []
        body_centers = []
        body_normals = []
        geom_taxels = {}
        offset = 0
        for geom_id in geom_ids:
            centers, normals = self._geom_surface(geom_id)
            if not len(centers):
                continue
            geom_rotation = self._quaternion_matrix(self.model.geom_quat[geom_id])
            centers_in_body = centers @ geom_rotation.T + self.model.geom_pos[geom_id]
            normals_in_body = normals @ geom_rotation.T
            count = len(centers)
            indices = np.arange(offset, offset + count, dtype=np.int32)
            geom_taxels[geom_id] = _TaxelRoute(indices, centers, normals)
            body_id = int(self.model.geom_bodyid[geom_id])
            body_ids.append(np.full(count, body_id, dtype=np.int32))
            layout_geom_ids.append(np.full(count, geom_id, dtype=np.int32))
            body_centers.append(centers_in_body)
            body_normals.append(normals_in_body)
            offset += count

        layout = TaxelLayout(
            body_id=self._readonly_concatenate(body_ids, np.int32),
            geom_id=self._readonly_concatenate(layout_geom_ids, np.int32),
            local_center=self._readonly_concatenate(
                body_centers, np.float64, shape=(0, 3)
            ),
            local_normal=self._readonly_concatenate(
                body_normals, np.float64, shape=(0, 3)
            ),
            region_id=self._readonly_full(offset, -1, np.int16),
            channel_id=self._readonly_full(offset, -1, np.int32),
        )
        return layout, geom_taxels

    def _use_layout(
        self, layout: TaxelLayout
    ) -> tuple[TaxelLayout, dict[int, _TaxelRoute]]:
        validated = self._validated_layout(layout)
        return validated, self._layout_routes(validated)

    def _validated_layout(self, layout: TaxelLayout) -> TaxelLayout:
        body_id = np.asarray(layout.body_id, dtype=np.int32)
        geom_id = np.asarray(layout.geom_id, dtype=np.int32)
        local_center = np.asarray(layout.local_center, dtype=np.float64)
        local_normal = np.asarray(layout.local_normal, dtype=np.float64)
        region_id = np.asarray(layout.region_id, dtype=np.int16)
        channel_id = np.asarray(layout.channel_id, dtype=np.int32)
        count = len(geom_id)
        arrays = {
            "body_id": (body_id, (count,)),
            "geom_id": (geom_id, (count,)),
            "local_center": (local_center, (count, 3)),
            "local_normal": (local_normal, (count, 3)),
            "region_id": (region_id, (count,)),
            "channel_id": (channel_id, (count,)),
        }
        for name, (array, expected) in arrays.items():
            if array.shape != expected:
                shape = "(N, 3)" if len(expected) == 2 else "(N,)"
                raise ValueError(f"layout {name} must have shape {shape}")
        if not np.all(np.isfinite(local_center)):
            raise ValueError("layout local_center must be finite")
        normal_length = np.linalg.norm(local_normal, axis=1)
        if np.any(~np.isfinite(normal_length)) or np.any(normal_length < 1e-12):
            raise ValueError("layout local_normal must be finite and nonzero")
        local_normal = local_normal / normal_length[:, None]
        self._validate_channel_mapping(layout, region_id, channel_id)

        return TaxelLayout(
            body_id=self._readonly_copy(body_id),
            geom_id=self._readonly_copy(geom_id),
            local_center=self._readonly_copy(local_center),
            local_normal=self._readonly_copy(local_normal),
            region_id=self._readonly_copy(region_id),
            channel_id=self._readonly_copy(channel_id),
            region_names=tuple(layout.region_names),
        )

    @staticmethod
    def _validate_channel_mapping(
        layout: TaxelLayout,
        region_id: np.ndarray,
        channel_id: np.ndarray,
    ) -> None:
        if np.any(region_id < -1) or np.any(region_id >= len(layout.region_names)):
            raise ValueError("layout region_id is outside region_names")
        if len(set(layout.region_names)) != len(layout.region_names):
            raise ValueError("layout region_names must be unique")
        if np.any(channel_id < -1) or np.any(channel_id > 255):
            raise ValueError("layout channel_id must be -1 or a raw slot 0..255")
        mapped_region = region_id >= 0
        mapped_channel = channel_id >= 0
        if np.any(mapped_region != mapped_channel):
            raise ValueError("layout region_id and channel_id must be mapped together")
        mapped_pairs = np.column_stack(
            (region_id[mapped_region], channel_id[mapped_region])
        )
        if len(mapped_pairs) != len(np.unique(mapped_pairs, axis=0)):
            raise ValueError("layout (region_id, channel_id) pairs must be unique")

    def _layout_routes(self, layout: TaxelLayout) -> dict[int, _TaxelRoute]:
        routes = {}
        for geom_id in sorted({int(value) for value in layout.geom_id}):
            if not 0 <= geom_id < self.model.ngeom:
                raise ValueError(f"invalid layout geom id: {geom_id}")
            body_id = int(self.model.geom_bodyid[geom_id])
            indices = np.flatnonzero(layout.geom_id == geom_id).astype(np.int32)
            if np.any(layout.body_id[indices] != body_id):
                raise ValueError(f"layout body_id does not own geom {geom_id}")
            if body_id not in self.robot_body_ids:
                raise ValueError(
                    f"layout geom {geom_id} is outside {self._root_name()}"
                )
            routes[geom_id] = self._taxel_route(layout, geom_id, indices)

        covered_bodies = {
            int(body_id): np.flatnonzero(layout.body_id == body_id).astype(np.int32)
            for body_id in np.unique(layout.body_id)
        }
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            indices = covered_bodies.get(body_id)
            if (
                geom_id not in routes
                and indices is not None
                and self._is_supported_collision_geom(geom_id)
            ):
                routes[geom_id] = self._taxel_route(layout, geom_id, indices)
        return routes

    def _taxel_route(
        self,
        layout: TaxelLayout,
        geom_id: int,
        indices: np.ndarray,
    ) -> _TaxelRoute:
        rotation = self._quaternion_matrix(self.model.geom_quat[geom_id])
        centers = (
            layout.local_center[indices] - self.model.geom_pos[geom_id]
        ) @ rotation
        normals = layout.local_normal[indices] @ rotation
        return _TaxelRoute(indices, centers, normals)

    def _is_supported_collision_geom(self, geom_id: int) -> bool:
        return (
            int(self.model.geom_contype[geom_id]) != 0
            or int(self.model.geom_conaffinity[geom_id]) != 0
        ) and int(self.model.geom_type[geom_id]) in _SUPPORTED_GEOMS

    def _geom_surface(self, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
        geom_type = int(self.model.geom_type[geom_id])
        size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            centers, normals = self._box_taxels(size)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            centers, normals = self._ellipsoid_taxels(np.repeat(size[0], 3))
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            centers, normals = self._ellipsoid_taxels(size)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            centers, normals = self._cylinder_taxels(size[0], size[1])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            centers, normals = self._capsule_taxels(size[0], size[1])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            centers, normals = self._mesh_taxels(int(self.model.geom_dataid[geom_id]))
        else:  # Guarded by _selected_geoms.
            raise AssertionError(f"unsupported geom type: {geom_type}")
        if len(centers) > self.max_taxels_per_geom:
            centers, normals = self._voxelize(
                centers, normals, self.spacing, self.max_taxels_per_geom
            )
        return centers, normals

    def _box_taxels(self, half_size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centers = []
        normals = []
        for axis in range(3):
            tangents = [value for value in range(3) if value != axis]
            u = self._linear_centers(half_size[tangents[0]])
            v = self._linear_centers(half_size[tangents[1]])
            uu, vv = np.meshgrid(u, v, indexing="ij")
            for sign in (-1.0, 1.0):
                points = np.zeros((uu.size, 3), dtype=np.float64)
                points[:, axis] = sign * half_size[axis]
                points[:, tangents[0]] = uu.ravel()
                points[:, tangents[1]] = vv.ravel()
                normal = np.zeros(3)
                normal[axis] = sign
                centers.append(points)
                normals.append(np.tile(normal, (len(points), 1)))
        return np.concatenate(centers), np.concatenate(normals)

    def _ellipsoid_taxels(self, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        maximum = float(np.max(radii))
        latitude_count = max(2, int(np.ceil(np.pi * maximum / self.spacing)))
        centers = []
        normals = []
        for latitude in range(latitude_count):
            polar = np.pi * (latitude + 0.5) / latitude_count
            ring_radius = maximum * np.sin(polar)
            longitude_count = max(
                1, int(np.ceil(2.0 * np.pi * ring_radius / self.spacing))
            )
            longitude = 2.0 * np.pi * np.arange(longitude_count) / longitude_count
            directions = np.column_stack(
                (
                    np.sin(polar) * np.cos(longitude),
                    np.sin(polar) * np.sin(longitude),
                    np.full(longitude_count, np.cos(polar)),
                )
            )
            points = directions * radii
            surface_normals = points / np.square(radii)
            surface_normals /= np.linalg.norm(surface_normals, axis=1, keepdims=True)
            centers.append(points)
            normals.append(surface_normals)
        return np.concatenate(centers), np.concatenate(normals)

    def _cylinder_taxels(
        self, radius: float, half_length: float
    ) -> tuple[np.ndarray, np.ndarray]:
        angle_count = max(3, int(np.ceil(2.0 * np.pi * radius / self.spacing)))
        height = self._linear_centers(half_length)
        angle = 2.0 * np.pi * np.arange(angle_count) / angle_count
        aa, zz = np.meshgrid(angle, height, indexing="ij")
        side = np.column_stack(
            (radius * np.cos(aa.ravel()), radius * np.sin(aa.ravel()), zz.ravel())
        )
        side_normals = np.column_stack(
            (np.cos(aa.ravel()), np.sin(aa.ravel()), np.zeros(aa.size))
        )
        centers = [side]
        normals = [side_normals]
        disk = self._disk_centers(radius)
        for sign in (-1.0, 1.0):
            cap = np.column_stack((disk, np.full(len(disk), sign * half_length)))
            normal = np.tile((0.0, 0.0, sign), (len(disk), 1))
            centers.append(cap)
            normals.append(normal)
        return np.concatenate(centers), np.concatenate(normals)

    def _capsule_taxels(
        self, radius: float, half_length: float
    ) -> tuple[np.ndarray, np.ndarray]:
        angle_count = max(3, int(np.ceil(2.0 * np.pi * radius / self.spacing)))
        angle = 2.0 * np.pi * np.arange(angle_count) / angle_count
        height = self._linear_centers(half_length)
        aa, zz = np.meshgrid(angle, height, indexing="ij")
        centers = [
            np.column_stack(
                (
                    radius * np.cos(aa.ravel()),
                    radius * np.sin(aa.ravel()),
                    zz.ravel(),
                )
            )
        ]
        normals = [
            np.column_stack((np.cos(aa.ravel()), np.sin(aa.ravel()), np.zeros(aa.size)))
        ]

        latitude_count = max(1, int(np.ceil(0.5 * np.pi * radius / self.spacing)))
        for end_sign in (-1.0, 1.0):
            for latitude in range(latitude_count):
                polar = 0.5 * np.pi * (latitude + 0.5) / latitude_count
                radial = radius * np.sin(polar)
                z_normal = end_sign * np.cos(polar)
                ring_count = max(1, int(np.ceil(2.0 * np.pi * radial / self.spacing)))
                ring_angle = 2.0 * np.pi * np.arange(ring_count) / ring_count
                ring_normals = np.column_stack(
                    (
                        np.sin(polar) * np.cos(ring_angle),
                        np.sin(polar) * np.sin(ring_angle),
                        np.full(ring_count, z_normal),
                    )
                )
                ring = radius * ring_normals
                ring[:, 2] += end_sign * half_length
                centers.append(ring)
                normals.append(ring_normals)
        return np.concatenate(centers), np.concatenate(normals)

    def _mesh_taxels(self, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
        vertex_address = int(self.model.mesh_vertadr[mesh_id])
        vertex_count = int(self.model.mesh_vertnum[mesh_id])
        vertices = np.asarray(
            self.model.mesh_vert[vertex_address : vertex_address + vertex_count],
            dtype=np.float64,
        )
        face_address = int(self.model.mesh_faceadr[mesh_id])
        face_count = int(self.model.mesh_facenum[mesh_id])
        faces = np.asarray(
            self.model.mesh_face[face_address : face_address + face_count],
            dtype=np.int32,
        )
        if not len(faces):
            return self._point_cloud_taxels(vertices)

        triangles = vertices[faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        norm = np.linalg.norm(face_normals, axis=1)
        valid = norm > 1e-12
        triangles = triangles[valid]
        face_normals = face_normals[valid] / norm[valid, None]
        if not len(triangles):
            return self._point_cloud_taxels(vertices)
        mesh_center = np.mean(vertices, axis=0)
        face_centers = np.mean(triangles, axis=1)
        inward = np.sum(face_normals * (face_centers - mesh_center), axis=1) < 0.0
        face_normals[inward] *= -1.0

        centers = []
        normals = []
        for triangle, normal in zip(triangles, face_normals):
            edge_length = max(
                np.linalg.norm(triangle[1] - triangle[0]),
                np.linalg.norm(triangle[2] - triangle[1]),
                np.linalg.norm(triangle[0] - triangle[2]),
            )
            divisions = max(1, int(np.ceil(edge_length / self.spacing)))
            barycentric = np.asarray(
                [
                    (row / divisions, column / divisions)
                    for row in range(divisions + 1)
                    for column in range(divisions + 1 - row)
                ],
                dtype=np.float64,
            )
            weight0 = 1.0 - barycentric[:, 0] - barycentric[:, 1]
            samples = (
                weight0[:, None] * triangle[0]
                + barycentric[:, :1] * triangle[1]
                + barycentric[:, 1:] * triangle[2]
            )
            centers.append(samples)
            normals.append(np.tile(normal, (len(samples), 1)))
        return self._voxelize(
            np.concatenate(centers),
            np.concatenate(normals),
            self.spacing,
            self.max_taxels_per_geom,
        )

    def _point_cloud_taxels(
        self,
        vertices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        normals = vertices - np.mean(vertices, axis=0)
        valid = np.linalg.norm(normals, axis=1) > 1e-12
        vertices = vertices[valid]
        normals = normals[valid]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return self._voxelize(
            vertices,
            normals,
            self.spacing,
            self.max_taxels_per_geom,
        )

    def _voxelize(
        self,
        centers: np.ndarray,
        normals: np.ndarray,
        spacing: float,
        maximum: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(centers):
            return centers.reshape(0, 3), normals.reshape(0, 3)
        current_spacing = spacing
        for iteration in range(32):
            origin = np.min(centers, axis=0)
            spatial_key = np.floor((centers - origin) / current_spacing).astype(
                np.int64
            )
            if iteration < 24:
                normal_key = np.floor(2.0 * (np.clip(normals, -1.0, 1.0) + 1.0))
                normal_key = np.clip(normal_key, 0, 3).astype(np.int64)
                keys = np.column_stack((spatial_key, normal_key))
            else:
                keys = spatial_key
            _, first, count = np.unique(
                keys,
                axis=0,
                return_index=True,
                return_counts=True,
            )
            if len(count) <= maximum:
                break
            current_spacing *= max(1.05, np.sqrt(len(count) / maximum))
        else:  # Defensive fallback for an unexpectedly pathological mesh.
            raise RuntimeError("could not bound mesh taxel sampling")

        # Keep an actual surface sample rather than a voxel mean, which could
        # move a curved or thin two-sided skin patch inside the geometry.
        sampled_centers = centers[first].copy()
        sampled_normals = normals[first].copy()
        sampled_normals /= np.linalg.norm(sampled_normals, axis=1, keepdims=True)
        return sampled_centers, sampled_normals

    def _nearby_taxels(
        self,
        geom_id: int,
        local_position: np.ndarray,
        local_outward: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        route = self._geom_taxels[geom_id]
        alignment = route.normals @ local_outward
        candidates = np.flatnonzero(alignment > 0.2)
        if not len(candidates):
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        distance_squared = np.sum(
            np.square(route.centers[candidates] - local_position), axis=1
        )
        if self.mapping_radius is not None:
            within = distance_squared <= np.square(
                self.mapping_radius[route.indices[candidates]]
            )
            candidates = candidates[within]
            distance_squared = distance_squared[within]
            if not len(candidates):
                return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        order = np.lexsort((route.indices[candidates], distance_squared))
        selected = candidates[order[: self.neighbors]]
        selected_distance = distance_squared[order[: self.neighbors]]
        scale_squared = max(self.spacing * self.spacing, 1e-12)
        log_weight = -0.5 * selected_distance / scale_squared
        log_weight -= np.max(log_weight)
        weights = np.exp(log_weight)
        weights /= np.sum(weights)
        return route.indices[selected], weights

    def _contact_body(self, contact: mujoco.MjContact, side: int) -> int:
        geom_id = int(contact.geom[side])
        if geom_id >= 0:
            return int(self.model.geom_bodyid[geom_id])
        flex_id = int(contact.flex[side])
        return self._flex_body_ids[flex_id] if flex_id >= 0 else 0

    def _flex_root_body(self, flex_id: int) -> int:
        address = int(self.model.flex_nodeadr[flex_id])
        count = int(self.model.flex_nodenum[flex_id])
        body_ids = self.model.flex_nodebodyid[address : address + count]
        if not len(body_ids):
            address = int(self.model.flex_vertadr[flex_id])
            count = int(self.model.flex_vertnum[flex_id])
            body_ids = self.model.flex_vertbodyid[address : address + count]
        body_ids = [int(body_id) for body_id in body_ids if body_id > 0]
        if not body_ids:
            return 0

        root = body_ids[0]
        for body_id in body_ids[1:]:
            ancestors = set()
            while body_id > 0:
                ancestors.add(body_id)
                body_id = int(self.model.body_parentid[body_id])
            while root not in ancestors and root > 0:
                root = int(self.model.body_parentid[root])
        return root

    def _is_descendant(self, body_id: int, root: int) -> bool:
        while body_id > 0:
            if body_id == root:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return False

    def _root_name(self) -> str:
        root = min(self.robot_body_ids)
        return (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, root)
            or f"body {root}"
        )

    def _linear_centers(self, half_extent: float) -> np.ndarray:
        count = max(1, int(np.ceil(2.0 * half_extent / self.spacing)))
        return (np.arange(count) + 0.5) * (2.0 * half_extent / count) - half_extent

    def _disk_centers(self, radius: float) -> np.ndarray:
        coordinates = self._linear_centers(radius)
        xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel()))
        points = points[np.sum(np.square(points), axis=1) <= radius * radius]
        if not len(points):
            return np.zeros((1, 2), dtype=np.float64)
        return points

    @staticmethod
    def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
        matrix = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
        return matrix.reshape(3, 3)

    @staticmethod
    def _readonly_concatenate(
        values: list[np.ndarray],
        dtype: np.dtype,
        *,
        shape: tuple[int, ...] = (0,),
    ) -> np.ndarray:
        result = (
            np.concatenate(values).astype(dtype, copy=False)
            if values
            else np.empty(shape, dtype=dtype)
        )
        result.setflags(write=False)
        return result

    @staticmethod
    def _readonly_copy(value: np.ndarray) -> np.ndarray:
        result = value.copy()
        result.setflags(write=False)
        return result

    @staticmethod
    def _readonly_full(count: int, value: int, dtype: np.dtype) -> np.ndarray:
        result = np.full(count, value, dtype=dtype)
        result.setflags(write=False)
        return result

    def _empty_frame(self) -> TactileFrame:
        count = len(self.layout)
        return TactileFrame(
            layout=self.layout,
            force=np.zeros((count, 3), dtype=np.float64),
            normal_force=np.zeros(count, dtype=np.float64),
            tangent_force=np.zeros((count, 3), dtype=np.float64),
            impulse=np.zeros((count, 3), dtype=np.float64),
            normal_impulse=np.zeros(count, dtype=np.float64),
            tangent_impulse=np.zeros((count, 3), dtype=np.float64),
            contact=np.zeros(count, dtype=bool),
            other_body_id=np.full(count, -1, dtype=np.int32),
            other_geom_id=np.full(count, -1, dtype=np.int32),
            sample_count=np.zeros(count, dtype=np.int32),
            duration=0.0,
            mapped_contact_count=0,
            unmapped_contact_count=0,
        )
