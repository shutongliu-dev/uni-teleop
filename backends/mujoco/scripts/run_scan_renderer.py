"""Serve RGB images from a trained Nerfstudio Gaussian-splat scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--listen", default="tcp://127.0.0.1:8765")
    parser.add_argument(
        "--anchor-image",
        default="frame_00138.jpg",
        help="captured image whose camera pose becomes the initial robot viewpoint",
    )
    parser.add_argument(
        "--geometry-summary",
        type=Path,
        help="optional scan geometry JSON containing the measured floor normal",
    )
    return parser.parse_args()


def find_anchor_camera(pipeline, image_name: str):
    for dataset in (
        pipeline.datamanager.train_dataset,
        pipeline.datamanager.eval_dataset,
    ):
        if dataset is None:
            continue
        for index, path in enumerate(dataset.image_filenames):
            if path.name == image_name:
                return dataset.cameras[index : index + 1]
    raise ValueError(f"anchor image is not part of the reconstruction: {image_name}")


def main() -> None:
    import numpy as np
    import torch
    import zmq
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.utils.eval_utils import eval_setup

    args = parse_args()
    if not args.config.is_file():
        raise SystemExit(f"Nerfstudio config not found: {args.config}")

    _, pipeline, checkpoint, step = eval_setup(
        args.config,
        test_mode="inference",
    )
    anchor = find_anchor_camera(pipeline, args.anchor_image)
    anchor_pose = np.eye(4, dtype=np.float32)
    anchor_pose[:3] = anchor.camera_to_worlds[0].detach().cpu().numpy()
    scan_up = [0.0, 0.0, 1.0]
    if args.geometry_summary is not None:
        geometry = json.loads(args.geometry_summary.read_text())
        scan_up = geometry["floor"]["normal"]
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.listen)
    print(
        f"Scan renderer ready on {args.listen}; "
        f"anchor={args.anchor_image}, checkpoint={checkpoint.name}, step={step}"
    )

    try:
        while True:
            request = socket.recv_json()
            try:
                operation = request.get("operation")
                if operation == "info":
                    response = {
                        "status": "ok",
                        "server_pid": os.getpid(),
                        "source_config": str(args.config.resolve()),
                        "anchor_image": args.anchor_image,
                        "anchor_camera_to_world": anchor_pose.tolist(),
                        "scan_up": scan_up,
                        "checkpoint": str(checkpoint),
                        "step": step,
                    }
                    socket.send_multipart([json.dumps(response).encode(), b""])
                    continue
                if operation != "render":
                    raise ValueError(f"unknown operation: {operation}")

                width = int(request["width"])
                height = int(request["height"])
                if width < 1 or height < 1 or width * height > 2_073_600:
                    raise ValueError("requested image dimensions are invalid")
                camera_to_world = torch.tensor(
                    request["camera_to_world"],
                    dtype=torch.float32,
                    device=pipeline.device,
                )
                if camera_to_world.shape != (3, 4):
                    raise ValueError("camera_to_world must have shape (3, 4)")
                camera = Cameras(
                    camera_to_worlds=camera_to_world[None],
                    fx=float(request["fx"]),
                    fy=float(request["fy"]),
                    cx=float(request["cx"]),
                    cy=float(request["cy"]),
                    width=width,
                    height=height,
                    camera_type=CameraType.PERSPECTIVE,
                )
                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera(camera)
                rgb = (
                    outputs["rgb"]
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .to(torch.uint8)
                    .detach()
                    .cpu()
                    .numpy()
                )
                response = {
                    "status": "ok",
                    "width": width,
                    "height": height,
                    "channels": 3,
                    "dtype": "uint8",
                    "depth_dtype": "float32",
                    "depth_units": "scan_units",
                }
                socket.send_multipart(
                    [
                        json.dumps(response).encode(),
                        np.ascontiguousarray(rgb).tobytes(),
                        np.ascontiguousarray(
                            outputs["depth"]
                            .squeeze(-1)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        ).tobytes(),
                    ]
                )
            except Exception as error:  # noqa: BLE001 - keep the render service alive
                response = {"status": "error", "error": str(error)}
                socket.send_multipart([json.dumps(response).encode(), b""])
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
