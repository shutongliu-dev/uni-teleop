"""Render task setups and robot sensor views without starting PICO services."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from uni_teleop.video.cameras import CameraRig
from sonic_mujoco.envs.mujoco.g1.registry import G1_ENVIRONMENTS, UNIVLAT_SCENES


def labelled(image, label):
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 26), fill=(18, 25, 35))
    draw.text((10, 7), label, fill="white")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/univlat"))
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gallery = Image.new("RGB", (1200, 960), (18, 25, 35))
    reports = {}
    for index, name in enumerate(UNIVLAT_SCENES):
        env = G1_ENVIRONMENTS[name]()
        try:
            env.reset(seed=args.seed)
            renderer = mujoco.Renderer(env.model, height=320, width=600)
            try:
                observer = mujoco.MjvCamera()
                observer.lookat[:] = (
                    0.5,
                    0.7 if name == "composed_cleanup" else 0,
                    0.8,
                )
                observer.distance = 4.7 if name == "composed_cleanup" else 3.4
                observer.azimuth, observer.elevation = 40, -23
                if name == "back_tap":
                    observer.azimuth = 135
                renderer.update_scene(env.data, camera=observer)
                overview = labelled(renderer.render(), name + " | initial scene")
                overview.save(args.output / f"{name}.png")
                gallery.paste(overview, ((index % 2) * 600, (index // 2) * 320))
            finally:
                renderer.close()
            rig = CameraRig(env.model, env.data, width=320, height=240)
            try:
                frames = rig.capture()
                panel = Image.new("RGB", (960, 960))
                for row, (view, frame) in enumerate(frames.items()):
                    depth = np.clip(frame.depth_m / 3.0, 0, 1)
                    depth_rgb = np.uint8(
                        np.stack((depth, np.sqrt(depth), 1 - depth), -1) * 255
                    )
                    depth_rgb[~frame.valid_depth] = 0
                    normal_rgb = np.uint8(
                        np.clip(frame.normals_camera * 0.5 + 0.5, 0, 1) * 255
                    )
                    normal_rgb[~frame.valid_normals] = 0
                    for col, (image, label) in enumerate(
                        (
                            (frame.rgb, view),
                            (depth_rgb, "depth | 0-3 estimated/metric m"),
                            (normal_rgb, "normal | optical axes"),
                        )
                    ):
                        panel.paste(labelled(image, label), (col * 320, row * 240))
                panel.save(args.output / f"{name}_sensors.png")
                reports[name] = {
                    "task": env.task_metadata,
                    "cameras": rig.metadata,
                    "initial_state": env.task_observation(),
                    "validation_scope": "initial scene render, not autonomous task rollout",
                }
            finally:
                rig.close()
        finally:
            env.close()
    draw = ImageDraw.Draw(gallery)
    draw.text((625, 690), "Uni-VLaT task approximations", fill="white")
    draw.text((625, 715), "G1 + DEXHAND + 2DoF neck", fill="white")
    draw.text((625, 740), "Robot stereo / wrist RGB / depth / normals", fill="white")
    draw.text(
        (625, 780), "Geometry is estimated; human is a contact proxy.", fill="white"
    )
    draw.text(
        (625, 805), "These are task setups, not policy success results.", fill="white"
    )
    gallery.save(args.output / "task_gallery.png")
    (args.output / "manifest.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Task previews: {args.output.resolve() / 'task_gallery.png'}")


if __name__ == "__main__":
    main()
