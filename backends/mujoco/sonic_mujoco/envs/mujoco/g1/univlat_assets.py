"""Procedural assets for the Uni-VLaT task approximations (SI units)."""

from pathlib import Path

import mujoco
import numpy as np

SCENES = Path(__file__).resolve().parents[3] / "assets/mujoco/scenes/g1"
SKIN = (0.69, 0.46, 0.33, 1.0)
SHIRT = (0.66, 0.27, 0.31, 1.0)
WOOD = (0.49, 0.32, 0.20, 1.0)


def geom(body, name, kind, size, pos=(0, 0, 0), rgba=(0.7, 0.7, 0.7, 1), **kwargs):
    return body.add_geom(
        name=name,
        type=getattr(mujoco.mjtGeom, f"mjGEOM_{kind.upper()}"),
        size=size,
        pos=pos,
        rgba=rgba,
        **kwargs,
    )


def table(spec, name, xy, height, half_size):
    body = spec.worldbody.add_body(name=name, pos=(*xy, height - 0.025))
    geom(
        body,
        f"{name}_top",
        "box",
        (*half_size, 0.025),
        rgba=WOOD,
        friction=(0.5, 0.005, 0.001),
    )
    length = height - 0.05
    for index, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        geom(
            body,
            f"{name}_leg_{index}",
            "box",
            (0.025, 0.025, length / 2),
            (x * (half_size[0] - 0.05), y * (half_size[1] - 0.05), -0.025 - length / 2),
            rgba=(0.16, 0.18, 0.19, 1),
        )
    return body


def container(spec, name, position, half_size, height, *, mass=None, rgba):
    """Five independent convex walls leave the container interior open."""
    body = spec.worldbody.add_body(name=name, pos=position)
    if mass is not None:
        body.add_freejoint(name=f"{name}_joint")
    x, y = half_size
    thickness = 0.008
    pieces = (
        ("bottom", (x, y, thickness), (0, 0, 0)),
        ("front", (thickness, y, height / 2), (-x, 0, height / 2)),
        ("back", (thickness, y, height / 2), (x, 0, height / 2)),
        ("left", (x, thickness, height / 2), (0, y, height / 2)),
        ("right", (x, thickness, height / 2), (0, -y, height / 2)),
    )
    for suffix, size, pos in pieces:
        properties = {} if mass is None else {"mass": mass / len(pieces)}
        geom(
            body,
            f"{name}_{suffix}",
            "box",
            size,
            pos,
            rgba,
            friction=(0.8, 0.01, 0.001),
            **properties,
        )
    body.add_site(
        name=f"{name}_interior",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0, 0, height / 2),
        size=(x - thickness, y - thickness, height / 2 - thickness),
        rgba=(0, 0, 0, 0),
    )
    return body


def human(spec, name="participant", position=(1.0, 0, 0)):
    """Kinematic clothed contact proxy; not a biomechanical human model."""
    body = spec.worldbody.add_body(name=name, pos=position, mocap=True)
    geom(
        body,
        f"{name}_torso",
        "ellipsoid",
        (0.13, 0.23, 0.31),
        (0, 0, 1.16),
        SHIRT,
        solref=(0.035, 1),
    )
    geom(body, f"{name}_neck", "capsule", (0.055, 0.07, 0), (0, 0, 1.49), SKIN)
    geom(body, f"{name}_head", "sphere", (0.105, 0, 0), (0, 0, 1.61), SKIN)
    geom(
        body,
        f"{name}_hips",
        "ellipsoid",
        (0.12, 0.19, 0.12),
        (0, 0, 0.84),
        (0.12, 0.17, 0.23, 1),
    )
    for side, sign in (("left", 1), ("right", -1)):
        geom(
            body,
            f"{name}_{side}_leg",
            "capsule",
            (0.065, 0.31, 0),
            (0, sign * 0.1, 0.44),
            (0.12, 0.17, 0.23, 1),
        )
        geom(
            body,
            f"{name}_{side}_foot",
            "box",
            (0.12, 0.065, 0.045),
            (-0.05, sign * 0.1, 0.045),
            (0.06, 0.06, 0.065, 1),
        )
        geom(
            body,
            f"{name}_{side}_arm",
            "capsule",
            (0.045, 0.24, 0),
            (-0.04, sign * 0.23, 1.11),
            SHIRT,
            solref=(0.035, 1),
        )
    return body


def _room(spec):
    spec.geom("floor").rgba = (0.48, 0.49, 0.50, 1)
    spec.geom("floor").material = ""
    spec.site("com_marker").rgba = (0, 0, 0, 0)
    geom(
        spec.worldbody,
        "task_curtain",
        "box",
        (0.025, 3.5, 1.2),
        (3.5, 0, 1.2),
        (0.035, 0.037, 0.045, 1),
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_light(
        pos=(0, -2, 3), dir=(0.3, 0.4, -1), diffuse=(0.65, 0.65, 0.65)
    )


def build_task(task):
    if task == "table_sweeping":
        spec = mujoco.MjSpec.from_file(str(SCENES / "sweep.xml"))
        spec.delete(spec.joint("table_joint"))
        return spec
    spec = mujoco.MjSpec.from_file(str(SCENES / "empty.xml"))
    _room(spec)
    if task == "back_tap_walking":
        for sign in (-1, 1):
            geom(
                spec.worldbody,
                f"walk_lane_{sign}",
                "box",
                (1.6, 0.01, 0.001),
                (1.0, sign * 0.6, 0.002),
                (0.03, 0.48, 0.85, 1),
                contype=0,
                conaffinity=0,
            )
        human(spec, "participant", (-0.65, -0.55, 0))
        palm = spec.worldbody.add_body(name="back_palm", mocap=True, pos=(-0.4, 0, 1.1))
        geom(
            palm,
            "back_palm_contact",
            "ellipsoid",
            (0.035, 0.07, 0.065),
            rgba=SKIN,
            solref=(0.04, 1),
            solimp=(0.8, 0.95, 0.01, 0.5, 2),
        )
    elif task == "human_robot_hugging":
        human(spec)
        geom(
            spec.worldbody,
            "hug_start_mark",
            "box",
            (0.1, 0.4, 0.001),
            (0.05, 0, 0.002),
            (0.03, 0.48, 0.85, 1),
            contype=0,
            conaffinity=0,
        )
    elif task == "basket_loading":
        table(spec, "handover_table", (0.7, 0), 0.74, (0.38, 0.55))
        container(
            spec,
            "basket",
            (0.66, 0, 0.757),
            (0.22, 0.29),
            0.18,
            mass=1.0,
            rgba=(0.62, 0.77, 0.81, 0.7),
        )
        human(spec, position=(1.65, -0.5, 0))
        table(spec, "loading_table", (0.8, -1.1), 0.74, (0.58, 0.22))
        support = spec.worldbody.add_body(
            name="handover_support", mocap=True, pos=(0.66, 0, 0.735)
        )
        geom(
            support,
            "handover_support_geom",
            "box",
            (0.16, 0.20, 0.015),
            rgba=(*SKIN[:3], 0.6),
        )
        # Each bowl uses a bottom and eight convex rim pieces, not a solid bowl hull.
        for i, mass in enumerate((0.06, 0.08, 0.10, 1.2, 0.35, 0.6)):
            height = 0.832 if i == 4 else 0.777 if i == 3 else 0.748
            body = spec.worldbody.add_body(
                name=f"load_{i}", pos=(0.35 + i * 0.18, -1.1, height)
            )
            body.add_freejoint(name=f"load_{i}_joint")
            if i in (0, 1, 2, 5):
                color = (0.95, 0.78, 0.18, 1) if i < 3 else (0.9, 0.88, 0.85, 1)
                geom(
                    body,
                    f"load_{i}_bottom",
                    "cylinder",
                    (0.045, 0.006, 0),
                    rgba=color,
                    mass=mass * 0.4,
                )
                for k in range(8):
                    a = k * np.pi / 4
                    geom(
                        body,
                        f"load_{i}_rim_{k}",
                        "box",
                        (0.008, 0.021, 0.022),
                        (0.045 * np.cos(a), 0.045 * np.sin(a), 0.022),
                        color,
                        quat=(np.cos(a / 2), 0, 0, np.sin(a / 2)),
                        mass=mass * 0.6 / 8,
                    )
            else:
                geom(
                    body,
                    f"load_{i}_geom",
                    "cylinder" if i == 4 else "box",
                    (0.032, 0.09, 0) if i == 4 else (0.035, 0.045, 0.035),
                    rgba=(0.2, 0.55, 0.65, 1) if i == 4 else (0.17, 0.17, 0.19, 1),
                    mass=mass,
                )
    elif task == "composed_cleanup":
        table(spec, "high_table", (0.82, -0.25), 0.90, (0.36, 0.55))
        table(spec, "low_table", (0.82, 0.64), 0.62, (0.36, 0.34))
        container(
            spec,
            "cleanup_box",
            (0.82, 0.62, 0.636),
            (0.27, 0.29),
            0.22,
            mass=0.65,
            rgba=(0.64, 0.76, 0.80, 0.65),
        )
        container(
            spec,
            "waste_bin",
            (0.35, 2.2, 0.025),
            (0.39, 0.42),
            0.74,
            rgba=(0.045, 0.047, 0.05, 1),
        )
        for i, xy in enumerate(
            ((0.64, -0.52), (0.96, -0.48), (0.68, -0.19), (0.97, -0.10))
        ):
            body = spec.worldbody.add_body(name=f"rubbish_{i}", pos=(*xy, 0.93))
            body.add_freejoint(name=f"rubbish_{i}_joint")
            geom(
                body,
                f"rubbish_{i}_geom",
                "box",
                (0.035, 0.04, 0.018),
                rgba=((0.86, 0.82, 0.72, 1) if i % 2 else (0.24, 0.6, 0.72, 1)),
                mass=0.025 + 0.015 * i,
                friction=(0.5, 0.005, 0.001),
            )
    else:
        raise ValueError(f"unknown Uni-VLaT task: {task}")
    return spec
