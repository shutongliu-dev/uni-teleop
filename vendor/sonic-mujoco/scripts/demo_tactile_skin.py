#!/usr/bin/env python3
"""Demonstrate a logical tactile-skin grid on the G1 torso.

The taxels in this demo are visual ``site`` elements only.  MuJoCo resolves
contact against the existing G1 torso collision mesh and a small box probe.
Each contact's normal force is then distributed conservatively to the four
nearest logical taxels.

Run headlessly with::

    MUJOCO_GL=egl .venv/bin/python scripts/demo_tactile_skin.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from sonic_mujoco.tactile import TactileRecorder, TaxelLayout

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised only in minimal envs
    raise SystemExit(
        "Pillow is required for the demo dashboard; install the gr00t extra"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "sonic_mujoco/assets/mujoco/scenes/g1/empty.xml"
RESULTS_DIR = ROOT / "results"

TARGET_BODY = "torso_link"
PROBE_BODY = "tactile_probe"
PROBE_GEOM = "tactile_probe_geom"
PROBE_HALF_LENGTH = 0.040
PROBE_HALF_FACE = 0.0075

TAXEL_ROWS = 7
TAXEL_COLS = 7
TAXEL_Y = np.linspace(-0.075, 0.075, TAXEL_COLS)
TAXEL_Z = np.linspace(0.080, 0.230, TAXEL_ROWS)
TAXEL_PITCH = float(TAXEL_Y[1] - TAXEL_Y[0])
TAXEL_PREFIX = "torso_taxel"

INACTIVE_RGBA = np.array([0.08, 0.28, 0.55, 0.22], dtype=np.float32)
TIMESTEP = 0.005


@dataclass(frozen=True, slots=True)
class SurfaceSample:
    geom_id: int
    position: np.ndarray
    normal: np.ndarray
    quaternion: np.ndarray


@dataclass(frozen=True, slots=True)
class ContactReadout:
    total_normal_force: float
    taxel_force: np.ndarray
    contact_positions_local: np.ndarray
    contact_normal_force: np.ndarray
    raw_impulse: np.ndarray
    taxel_impulse: np.ndarray

    @property
    def raw_normal_impulse(self) -> float:
        return self.total_normal_force * TIMESTEP

    @property
    def raw_impulse_norm(self) -> float:
        return float(np.linalg.norm(self.raw_impulse))

    @property
    def taxel_impulse_norm(self) -> float:
        return float(np.linalg.norm(self.taxel_impulse))

    @property
    def impulse_relative_error(self) -> float:
        denominator = max(self.raw_impulse_norm, 1e-12)
        return (
            float(np.linalg.norm(self.taxel_impulse - self.raw_impulse)) / denominator
        )


@dataclass(frozen=True, slots=True)
class DemoModel:
    model: mujoco.MjModel
    data: mujoco.MjData
    target_body_id: int
    target_geom_ids: tuple[int, ...]
    probe_body_id: int
    probe_geom_id: int
    probe_mocap_id: int
    taxel_site_ids: np.ndarray
    tactile: TactileRecorder
    base_collision_geom_count: int


def _surface_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    geom_ids: tuple[int, ...],
    y: float,
    z: float,
) -> SurfaceSample:
    body_rotation = data.xmat[body_id].reshape(3, 3)
    body_position = data.xpos[body_id]
    ray_start = body_position + body_rotation @ np.array([0.30, y, z])
    ray_direction = body_rotation @ np.array([-1.0, 0.0, 0.0])

    nearest_distance = np.inf
    nearest_geom_id = -1
    nearest_normal = np.zeros(3, dtype=np.float64)
    for geom_id in geom_ids:
        normal = np.zeros(3, dtype=np.float64)
        distance = mujoco.mj_rayMesh(
            model,
            data,
            geom_id,
            ray_start,
            ray_direction,
            normal,
        )
        if 0.0 <= distance < nearest_distance:
            nearest_distance = distance
            nearest_geom_id = geom_id
            nearest_normal = normal
    if not np.isfinite(nearest_distance):
        raise RuntimeError(f"torso surface ray missed at local y={y:.3f}, z={z:.3f}")

    world_position = ray_start + nearest_distance * ray_direction
    world_normal = nearest_normal / np.linalg.norm(nearest_normal)
    local_position = body_rotation.T @ (world_position - body_position)
    local_normal = body_rotation.T @ world_normal

    # Build a local site frame whose thin x-axis follows the surface normal.
    local_z_hint = np.array([0.0, 0.0, 1.0])
    local_y_axis = np.cross(local_z_hint, local_normal)
    if np.linalg.norm(local_y_axis) < 1e-8:
        local_z_hint = np.array([0.0, 1.0, 0.0])
        local_y_axis = np.cross(local_z_hint, local_normal)
    local_y_axis /= np.linalg.norm(local_y_axis)
    local_z_axis = np.cross(local_normal, local_y_axis)
    local_rotation = np.stack(
        (local_normal, local_y_axis, local_z_axis),
        axis=1,
    )
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, local_rotation.ravel())
    return SurfaceSample(
        geom_id=nearest_geom_id,
        position=local_position,
        normal=local_normal,
        quaternion=quaternion,
    )


def _collidable_geom_count(model: mujoco.MjModel) -> int:
    collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    return int(np.count_nonzero(collidable))


def _target_geometry_ids(
    model: mujoco.MjModel,
    body_id: int,
) -> tuple[int, ...]:
    return tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] == body_id
        and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
    )


def build_demo_model() -> DemoModel:
    spec = mujoco.MjSpec.from_file(str(SCENE_PATH))

    fixture = spec.add_equality()
    fixture.name = "tactile_bench_fixture"
    fixture.type = mujoco.mjtEq.mjEQ_WELD
    fixture.objtype = mujoco.mjtObj.mjOBJ_BODY
    fixture.name1 = TARGET_BODY

    base_model = spec.compile()
    base_data = mujoco.MjData(base_model)
    base_model.opt.gravity[:] = 0.0
    mujoco.mj_forward(base_model, base_data)
    target_body_id = mujoco.mj_name2id(
        base_model,
        mujoco.mjtObj.mjOBJ_BODY,
        TARGET_BODY,
    )
    target_geom_ids = _target_geometry_ids(base_model, target_body_id)
    if not target_geom_ids:
        raise RuntimeError(f"{TARGET_BODY} has no collidable mesh geometry")
    base_collision_geom_count = _collidable_geom_count(base_model)

    torso = spec.body(TARGET_BODY)
    layout_geom_ids = []
    layout_centers = []
    layout_normals = []
    for row, z in enumerate(TAXEL_Z):
        for col, y in enumerate(TAXEL_Y):
            sample = _surface_sample(
                base_model,
                base_data,
                target_body_id,
                target_geom_ids,
                float(y),
                float(z),
            )
            layout_geom_ids.append(sample.geom_id)
            layout_centers.append(sample.position)
            layout_normals.append(sample.normal)
            site = torso.add_site(name=f"{TAXEL_PREFIX}_{row}_{col}")
            site.type = mujoco.mjtGeom.mjGEOM_BOX
            site.pos = sample.position + 0.0015 * sample.normal
            site.quat = sample.quaternion
            site.size = [0.0012, 0.0105, 0.0105]
            site.rgba = INACTIVE_RGBA
            site.group = 2

    probe_body = spec.worldbody.add_body(name=PROBE_BODY)
    probe_body.mocap = True
    probe_geom = probe_body.add_geom(name=PROBE_GEOM)
    probe_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    probe_geom.size = [PROBE_HALF_LENGTH, PROBE_HALF_FACE, PROBE_HALF_FACE]
    probe_geom.density = 500.0
    probe_geom.friction = [0.8, 0.01, 0.001]
    probe_geom.rgba = [0.88, 0.38, 0.08, 0.72]

    model = spec.compile()
    model.opt.gravity[:] = 0.0
    model.opt.timestep = TIMESTEP
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    target_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        TARGET_BODY,
    )
    target_geom_ids = _target_geometry_ids(model, target_body_id)
    probe_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        PROBE_BODY,
    )
    probe_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        PROBE_GEOM,
    )
    taxel_site_ids = np.array(
        [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                f"{TAXEL_PREFIX}_{row}_{col}",
            )
            for row in range(TAXEL_ROWS)
            for col in range(TAXEL_COLS)
        ],
        dtype=np.int32,
    )
    taxel_count = TAXEL_ROWS * TAXEL_COLS
    layout = TaxelLayout(
        body_id=np.full(taxel_count, target_body_id, dtype=np.int32),
        geom_id=np.asarray(layout_geom_ids, dtype=np.int32),
        local_center=np.asarray(layout_centers, dtype=np.float64),
        local_normal=np.asarray(layout_normals, dtype=np.float64),
        region_id=np.zeros(taxel_count, dtype=np.int16),
        channel_id=np.arange(taxel_count, dtype=np.int32),
        region_names=("torso_front",),
    )
    tactile = TactileRecorder(model, layout=layout, neighbors=4)
    return DemoModel(
        model=model,
        data=data,
        target_body_id=target_body_id,
        target_geom_ids=target_geom_ids,
        probe_body_id=probe_body_id,
        probe_geom_id=probe_geom_id,
        probe_mocap_id=int(model.body_mocapid[probe_body_id]),
        taxel_site_ids=taxel_site_ids,
        tactile=tactile,
        base_collision_geom_count=base_collision_geom_count,
    )


def _world_probe_pose(
    demo: DemoModel,
    y: float,
    z: float,
    normal_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample = _surface_sample(
        demo.model,
        demo.data,
        demo.target_body_id,
        demo.target_geom_ids,
        y,
        z,
    )
    body_rotation = demo.data.xmat[demo.target_body_id].reshape(3, 3)
    body_position = demo.data.xpos[demo.target_body_id]
    world_position = body_position + body_rotation @ (
        sample.position + normal_offset * sample.normal
    )
    world_rotation = body_rotation @ _quaternion_matrix(sample.quaternion)
    world_quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(world_quaternion, world_rotation.ravel())
    return world_position, world_quaternion


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def read_tactile_skin(demo: DemoModel) -> ContactReadout:
    contact_force = np.zeros(6, dtype=np.float64)
    positions: list[np.ndarray] = []
    normal_forces: list[float] = []
    raw_impulse = np.zeros(3, dtype=np.float64)
    body_rotation = demo.data.xmat[demo.target_body_id].reshape(3, 3)
    body_position = demo.data.xpos[demo.target_body_id]

    for contact_id in range(demo.data.ncon):
        contact = demo.data.contact[contact_id]
        geom1, geom2 = (int(contact.geom[0]), int(contact.geom[1]))
        if demo.probe_geom_id not in (geom1, geom2):
            continue
        other_geom = geom2 if geom1 == demo.probe_geom_id else geom1
        if other_geom not in demo.target_geom_ids:
            continue

        mujoco.mj_contactForce(
            demo.model,
            demo.data,
            contact_id,
            contact_force,
        )
        normal_force = abs(float(contact_force[0]))
        if normal_force <= 0.0:
            continue
        local_position = body_rotation.T @ (np.asarray(contact.pos) - body_position)
        positions.append(local_position)
        normal_forces.append(normal_force)
        robot_side = 0 if geom1 == other_geom else 1
        contact_frame = np.asarray(contact.frame).reshape(3, 3)
        force_world = contact_frame.T @ contact_force[:3]
        if robot_side == 0:
            force_world *= -1.0
        raw_impulse += body_rotation.T @ force_world * TIMESTEP

    demo.tactile.begin()
    demo.tactile.update(demo.data)
    tactile_frame = demo.tactile.finish()

    return ContactReadout(
        total_normal_force=float(sum(normal_forces)),
        taxel_force=tactile_frame.normal_force.reshape(TAXEL_ROWS, TAXEL_COLS),
        contact_positions_local=np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        contact_normal_force=np.asarray(normal_forces, dtype=np.float64),
        raw_impulse=raw_impulse,
        taxel_impulse=tactile_frame.impulse.sum(axis=0),
    )


def set_probe(
    demo: DemoModel,
    y: float,
    z: float,
    normal_offset: float,
) -> ContactReadout:
    position, quaternion = _world_probe_pose(demo, y, z, normal_offset)
    demo.data.mocap_pos[demo.probe_mocap_id] = position
    demo.data.mocap_quat[demo.probe_mocap_id] = quaternion
    demo.data.qvel[:] = 0.0
    demo.data.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(demo.model, demo.data)
    return read_tactile_skin(demo)


def calibrated_offset(
    demo: DemoModel,
    y: float,
    z: float,
    target_force: float,
) -> float:
    inward = PROBE_HALF_LENGTH - 0.006
    outward = PROBE_HALF_LENGTH + 0.010
    inward_force = set_probe(demo, y, z, inward).total_normal_force
    outward_force = set_probe(demo, y, z, outward).total_normal_force
    if inward_force < target_force or outward_force > target_force:
        raise RuntimeError(
            "failed to bracket requested probe force: "
            f"inward={inward_force:.2f} N, outward={outward_force:.2f} N"
        )
    for _ in range(28):
        middle = 0.5 * (inward + outward)
        middle_force = set_probe(demo, y, z, middle).total_normal_force
        if middle_force > target_force:
            inward = middle
        else:
            outward = middle
    return 0.5 * (inward + outward)


def _force_color(force: float, scale: float) -> np.ndarray:
    value = float(np.clip(force / max(scale, 1e-9), 0.0, 1.0))
    if value <= 0.0:
        return INACTIVE_RGBA.copy()
    if value < 0.5:
        fraction = value / 0.5
        rgb = (1.0 - fraction) * np.array([0.05, 0.45, 1.0]) + fraction * np.array(
            [0.05, 1.0, 0.55]
        )
    else:
        fraction = (value - 0.5) / 0.5
        rgb = (1.0 - fraction) * np.array([0.05, 1.0, 0.55]) + fraction * np.array(
            [1.0, 0.12, 0.02]
        )
    return np.array([*rgb, 0.96], dtype=np.float32)


def update_taxel_colors(demo: DemoModel, taxel_force: np.ndarray) -> None:
    scale = max(float(taxel_force.max()), 1.0)
    for site_id, force in zip(demo.taxel_site_ids, taxel_force.ravel()):
        demo.model.site_rgba[site_id] = _force_color(float(force), scale)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _draw_force_curve(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    force_history: list[float],
    target_force: float,
) -> None:
    left, top, right, bottom = bounds
    draw.rectangle(bounds, fill=(15, 25, 36), outline=(78, 101, 122), width=1)
    if len(force_history) < 2:
        return
    maximum = max(target_force * 1.25, max(force_history), 1.0)
    points = []
    for index, force in enumerate(force_history):
        x = left + 8 + index * (right - left - 16) / (len(force_history) - 1)
        y = bottom - 8 - force * (bottom - top - 16) / maximum
        points.append((x, y))
    target_y = bottom - 8 - target_force * (bottom - top - 16) / maximum
    draw.line((left + 8, target_y, right - 8, target_y), fill=(130, 143, 154), width=1)
    draw.line(points, fill=(255, 150, 48), width=3)


def make_dashboard(
    scene_rgb: np.ndarray,
    readout: ContactReadout,
    force_history: list[float],
    target_force: float,
    collision_geom_delta: int,
) -> Image.Image:
    scene = Image.fromarray(scene_rgb).resize((800, 600), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1200, 600), (10, 18, 27))
    canvas.paste(scene, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(23)
    body_font = _font(16)
    small_font = _font(13)
    draw.text(
        (824, 18), "G1 logical tactile skin", font=title_font, fill=(236, 242, 248)
    )
    draw.text(
        (824, 50),
        "MuJoCo mesh contact -> 7 x 7 taxels",
        font=small_font,
        fill=(143, 166, 187),
    )

    cell = 38
    grid_left = 862
    grid_top = 82
    scale = max(float(readout.taxel_force.max()), 1.0)
    for display_row in range(TAXEL_ROWS):
        source_row = TAXEL_ROWS - 1 - display_row
        for col in range(TAXEL_COLS):
            force = float(readout.taxel_force[source_row, col])
            rgba = _force_color(force, scale)
            rgb = tuple(int(255 * component) for component in rgba[:3])
            x0 = grid_left + col * cell
            y0 = grid_top + display_row * cell
            draw.rounded_rectangle(
                (x0 + 2, y0 + 2, x0 + cell - 3, y0 + cell - 3),
                radius=3,
                fill=rgb if force > 1e-5 else (26, 48, 69),
                outline=(80, 105, 127),
                width=1,
            )
    draw.text(
        (824, 356), "taxel normal force [N]", font=small_font, fill=(143, 166, 187)
    )

    draw.text(
        (824, 382),
        f"raw contact force     {readout.total_normal_force:7.3f} N",
        font=body_font,
        fill=(236, 242, 248),
    )
    draw.text(
        (824, 406),
        f"raw impulse norm      {readout.raw_impulse_norm:7.5f} N s",
        font=body_font,
        fill=(236, 242, 248),
    )
    draw.text(
        (824, 430),
        f"taxel vector sum      {readout.taxel_impulse_norm:7.5f} N s",
        font=body_font,
        fill=(236, 242, 248),
    )
    draw.text(
        (824, 454),
        f"relative error        {readout.impulse_relative_error:7.2e}",
        font=body_font,
        fill=(115, 230, 155),
    )
    draw.text(
        (824, 478),
        f"collision geom delta  {collision_geom_delta:+d} (taxels only)",
        font=body_font,
        fill=(115, 230, 155) if collision_geom_delta == 0 else (255, 105, 90),
    )
    _draw_force_curve(draw, (824, 512, 1178, 586), force_history, target_force)
    draw.text((832, 516), "force over scan path", font=small_font, fill=(180, 194, 207))
    return canvas


def _render_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 1.0]
    camera.distance = 0.70
    camera.azimuth = 164.0
    camera.elevation = -4.0
    return camera


def run_demo(target_force: float, output_dir: Path) -> None:
    demo = build_demo_model()
    output_dir.mkdir(parents=True, exist_ok=True)

    final_collision_count = _collidable_geom_count(demo.model)
    probe_collision_count = int(
        demo.model.geom_contype[demo.probe_geom_id] != 0
        or demo.model.geom_conaffinity[demo.probe_geom_id] != 0
    )
    collision_geom_delta = (
        final_collision_count - demo.base_collision_geom_count - probe_collision_count
    )

    # The path moves a loaded box across four taxel pitches on the chest.
    path_y = np.linspace(-0.050, 0.040, 21)
    path_z = np.linspace(0.130, 0.190, 21)
    loaded_offsets = [
        calibrated_offset(demo, float(y), float(z), target_force)
        for y, z in zip(path_y, path_z)
    ]

    # Add a short unloaded approach and retraction around the loaded traverse.
    poses: list[tuple[float, float, float]] = []
    first_clear = PROBE_HALF_LENGTH + 0.018
    for fraction in np.linspace(0.0, 1.0, 5, endpoint=False):
        offset = (1.0 - fraction) * first_clear + fraction * loaded_offsets[0]
        poses.append((float(path_y[0]), float(path_z[0]), float(offset)))
    poses.extend(
        (float(y), float(z), float(offset))
        for y, z, offset in zip(path_y, path_z, loaded_offsets)
    )
    last_clear = PROBE_HALF_LENGTH + 0.018
    for fraction in np.linspace(0.0, 1.0, 5)[1:]:
        offset = (1.0 - fraction) * loaded_offsets[-1] + fraction * last_clear
        poses.append((float(path_y[-1]), float(path_z[-1]), float(offset)))

    renderer = mujoco.Renderer(demo.model, height=480, width=640)
    camera = _render_camera()
    frames: list[Image.Image] = []
    readouts: list[ContactReadout] = []
    force_history: list[float] = []
    try:
        for y, z, offset in poses:
            readout = set_probe(demo, y, z, offset)
            readouts.append(readout)
            force_history.append(readout.total_normal_force)
            update_taxel_colors(demo, readout.taxel_force)
            renderer.update_scene(demo.data, camera=camera)
            scene_rgb = renderer.render()
            frames.append(
                make_dashboard(
                    scene_rgb,
                    readout,
                    force_history,
                    target_force,
                    collision_geom_delta,
                )
            )
    finally:
        renderer.close()

    loaded_start = 5
    middle_frame = loaded_start + len(path_y) // 2
    png_path = output_dir / "tactile_skin_mujoco_demo.png"
    gif_path = output_dir / "tactile_skin_mujoco_demo.gif"
    frames[middle_frame].save(png_path)
    animation_frames = frames + [frames[-1]] * 4
    animation_frames[0].save(
        gif_path,
        save_all=True,
        append_images=animation_frames[1:],
        duration=110,
        loop=0,
        optimize=False,
    )

    loaded_readouts = readouts[loaded_start : loaded_start + len(path_y)]
    raw_impulse = np.sum(
        [readout.raw_impulse for readout in loaded_readouts],
        axis=0,
    )
    taxel_impulse = np.sum(
        [readout.taxel_impulse for readout in loaded_readouts],
        axis=0,
    )
    relative_error = float(np.linalg.norm(taxel_impulse - raw_impulse)) / max(
        float(np.linalg.norm(raw_impulse)),
        1e-12,
    )
    peak_indices = [
        np.unravel_index(np.argmax(readout.taxel_force), readout.taxel_force.shape)
        for readout in loaded_readouts
    ]
    peak_travel = np.linalg.norm(
        np.array(
            [
                TAXEL_Y[peak_indices[-1][1]] - TAXEL_Y[peak_indices[0][1]],
                TAXEL_Z[peak_indices[-1][0]] - TAXEL_Z[peak_indices[0][0]],
            ]
        )
    )
    localization_error = []
    for readout, peak_index in zip(loaded_readouts, peak_indices):
        contact_center = np.average(
            readout.contact_positions_local[:, 1:3],
            axis=0,
            weights=readout.contact_normal_force,
        )
        peak_center = np.array(
            [TAXEL_Y[peak_index[1]], TAXEL_Z[peak_index[0]]],
        )
        localization_error.append(float(np.linalg.norm(contact_center - peak_center)))
    maximum_localization_error = max(localization_error)
    no_contact_force = readouts[0].total_normal_force
    loaded_force = np.array([readout.total_normal_force for readout in loaded_readouts])
    maximum_force_error = float(np.max(np.abs(loaded_force - target_force)))

    print(f"taxels: {TAXEL_ROWS * TAXEL_COLS} visual sites")
    print(f"base collidable geoms: {demo.base_collision_geom_count}")
    print(f"final collidable geoms: {final_collision_count}")
    print(f"probe collidable geoms: {probe_collision_count}")
    print(f"collision geom delta from taxels: {collision_geom_delta}")
    print(f"loaded frames: {len(loaded_readouts)} at {TIMESTEP:.3f} s")
    print(f"raw MuJoCo impulse: {np.array2string(raw_impulse, precision=9)} N s")
    print(
        f"taxel impulse vector sum: {np.array2string(taxel_impulse, precision=9)} N s"
    )
    print(f"impulse relative error: {relative_error:.3e}")
    print(f"taxel peak travel: {1000.0 * peak_travel:.1f} mm")
    print(
        f"maximum peak localization error: {1000.0 * maximum_localization_error:.1f} mm"
    )
    print(f"unloaded force: {no_contact_force:.3e} N")
    print(f"maximum loaded force error: {maximum_force_error:.3e} N")
    print(f"PNG: {png_path}")
    print(f"GIF: {gif_path}")

    if collision_geom_delta != 0:
        raise AssertionError("logical taxels unexpectedly added collision geometry")
    if relative_error > 1e-9:
        raise AssertionError("taxel mapping did not conserve vector impulse")
    if peak_travel < 0.075:
        raise AssertionError("taxel hotspot did not follow the moving probe")
    if maximum_localization_error > TAXEL_PITCH:
        raise AssertionError("taxel hotspot was more than one pitch from contact")
    if no_contact_force > 1e-6:
        raise AssertionError("unloaded probe produced a false tactile contact")
    if maximum_force_error > 0.05:
        raise AssertionError("calibrated contact force left its acceptance band")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-force",
        type=float,
        default=8.0,
        help="normal load maintained while the box traverses the torso (N)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="directory for PNG and GIF outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1.0 <= args.target_force <= 20.0:
        raise SystemExit("--target-force must be in [1, 20] N")
    run_demo(args.target_force, args.output_dir.resolve())


if __name__ == "__main__":
    main()
