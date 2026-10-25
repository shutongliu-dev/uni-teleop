"""Inspect one recorded row and audit episode timing without starting simulation."""

import argparse
import json
from pathlib import Path

import numpy as np

from sonic_mujoco.episode_reader import EpisodeReader, EpisodeReadError


def contact_sheet(sample, path, depth_max=3.0, depth_units="meters"):
    from PIL import Image, ImageDraw, ImageOps

    width, height = 320, 240
    panel = Image.new(
        "RGB", (width * 3, height * max(1, len(sample.cameras)) + 36), (18, 25, 35)
    )
    draw = ImageDraw.Draw(panel)
    draw.text(
        (10, 10),
        f"Episode row {sample.index} | sim time {sample.sim_time:.3f} s",
        fill="white",
    )
    if not sample.cameras:
        draw.text((10, 60), "This recording has no RGB camera streams.", fill="white")
    for row, (name, frame) in enumerate(sample.cameras.items()):
        images = [(frame.rgb, name + " | RGB")]
        if frame.depth_m is not None:
            depth = np.clip(frame.depth_m / depth_max, 0, 1)
            rgb = np.uint8(np.stack((depth, np.sqrt(depth), 1 - depth), axis=-1) * 255)
            rgb[~frame.valid_depth] = 0
            normals = np.uint8(np.clip(frame.normals_camera * 0.5 + 0.5, 0, 1) * 255)
            normals[~frame.valid_normals] = 0
            images.extend(
                (
                    (rgb, f"Depth | 0-{depth_max:g} {depth_units}"),
                    (normals, "Normal | optical camera axes"),
                )
            )
        else:
            empty = np.zeros_like(frame.rgb)
            images.extend(
                (
                    (empty, "Depth unavailable (RGB-only)"),
                    (empty, "Normals unavailable (RGB-only)"),
                )
            )
        for col, (array, title) in enumerate(images):
            tile = ImageOps.pad(
                Image.fromarray(array), (width, height), color=(18, 25, 35)
            )
            label = ImageDraw.Draw(tile)
            label.rectangle((0, 0, width, 25), fill=(18, 25, 35))
            label.text((8, 7), title, fill="white")
            if col == 0 and frame.timestamp is not None:
                label.rectangle((0, height - 24, width, height), fill=(18, 25, 35))
                label.text(
                    (8, height - 18),
                    f"sample {frame.sample_index} | age {frame.age_s:.3f} s",
                    fill="white",
                )
            panel.paste(tile, (col * width, 36 + row * height))
    panel.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session", type=Path, help="Recording session containing meta/info.json"
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/episode_inspection")
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=3.0,
        help="Depth preview upper limit in recorded meters",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.depth_max) or args.depth_max <= 0:
        parser.error("--depth-max must be finite and positive")
    output, session = args.output.resolve(), args.session.resolve()
    if output.is_relative_to(session):
        parser.error(
            "--output must be outside the input dataset; inspection is read-only"
        )
    if any(
        (output / name).is_symlink() for name in ("contact_sheet.png", "audit.json")
    ):
        parser.error("inspection outputs must not be symbolic links")
    try:
        reader = EpisodeReader(session, args.episode)
        sample = reader.read(args.row)
        report = reader.audit(sample)
        output.mkdir(parents=True, exist_ok=True)
        units = (reader.camera_metadata or {}).get("depth_units", "unspecified units")
        contact_sheet(sample, output / "contact_sheet.png", args.depth_max, units)
        (output / "audit.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    except (EpisodeReadError, OSError) as exc:
        parser.exit(2, f"Inspection failed: {exc}\n")
    print(f"Contact sheet: {output / 'contact_sheet.png'}")
    print(f"Timing audit: {output / 'audit.json'}")
    if not report["temporal"]["ok"]:
        parser.exit(
            1,
            f"Found {report['temporal']['error_count']} temporal inconsistencies; see audit.json.\n",
        )


if __name__ == "__main__":
    main()
