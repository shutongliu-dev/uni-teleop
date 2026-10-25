"""Render catalog-pinned task resets as individual native RTX PNGs.

This is a presentation tool: it never starts the physics timeline or executes
an Agent. Original assets, frozen releases and task cameras remain unchanged.
Run from a complete source snapshot on one explicitly selected GPU UUID.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from bench.paths import DATA_ROOT, data_root

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from bench.isaacsim.release import digest, dump, task_inputs
from scenes.workrooms.catalog import list_tasks, load_task
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.media.render_workroom import now


def packaged_bimanual(assets_root, version):
    """Read the released dual-arm USDs without regenerating portable URDFs.

    This static renderer does not need a URDF import. Packaged URDF paths were
    made relative during publishing and intentionally differ from a fresh build.
    Validate the published assembly tree before referencing its existing USDs.
    """
    root = Path(assets_root) / 'robots/assemblies/dual_fr3_sharpa_w01' / version
    prefix = root.relative_to(data_root()).as_posix() + '/'
    files = json.loads((data_root() / 'MANIFEST.json').read_text())['files']
    selected = {name: record for name, record in files.items() if name.startswith(prefix)}
    if not selected:
        raise ValueError('Dual-arm assembly is absent from the data manifest')
    for name, record in selected.items():
        if digest(data_root() / name) != record['sha256']:
            raise ValueError('Published dual-arm asset changed: ' + name)
    result = {side: root / f'usd_{side}/fr3_sharpa_{side}/fr3_sharpa_{side}.usda'
              for side in ('left', 'right')}
    for path in result.values():
        if path.relative_to(data_root()).as_posix() not in selected:
            raise ValueError('Dual-arm USD is absent from the data manifest')
    return result


def capture_image(stage, app, name, definition, output, args, record):
    """Use one explicit render product; record its actual camera projection."""
    import numpy as np
    import omni.hydratexture
    import omni.replicator.core as rep
    import omni.usd
    from carb.eventdispatcher import get_eventdispatcher
    from PIL import Image
    from pxr import Gf, Sdf, UsdGeom
    from bench.simulation.isaacsim_backend.render_sync import camera_calibration
    from scripts.media.render_workroom import create_camera

    path = create_camera(stage, name, definition)
    camera = UsdGeom.Camera(stage.GetPrimAtPath(path))
    camera.CreateHorizontalApertureAttr(float(definition["horizontal_aperture_mm"]))
    camera.CreateVerticalApertureAttr(float(definition["vertical_aperture_mm"]))
    camera.CreateClippingRangeAttr(Gf.Vec2f(*definition.get("clipping_m", [.02, 8.])))
    authored = {
        **definition,
        "horizontal_aperture_mm": float(camera.GetHorizontalApertureAttr().Get()),
        "vertical_aperture_mm": float(camera.GetVerticalApertureAttr().Get()),
        "focal_length_mm": float(camera.GetFocalLengthAttr().Get()),
        "clipping_m": list(camera.GetClippingRangeAttr().Get()),
    }
    product = rep.create.render_product(path, (args.width, args.height))
    prim = stage.GetPrimAtPath(product.path)
    for key, kind, value in (
        ("samplesPerPixel", Sdf.ValueTypeNames.UInt, args.samples),
        ("samplesPerIteration", Sdf.ValueTypeNames.Int, min(args.samples, 32)),
        ("clampSpp", Sdf.ValueTypeNames.Int, args.samples),
        ("limits:maxBounces", Sdf.ValueTypeNames.UInt, 12),
        ("denoising:enabled", Sdf.ValueTypeNames.Bool, True),
        ("adaptiveSampling:enabled", Sdf.ValueTypeNames.Bool, False),
    ):
        prim.CreateAttribute("omni:rtx:pt:" + key, kind).Set(value)
    rgb_source = rep.AnnotatorRegistry.get_annotator("rgb")
    parameters = rep.AnnotatorRegistry.get_annotator("camera_params")
    rgb_source.attach([product])
    parameters.attach([product])
    info = {}

    def on_frame(event):
        frame = product.hydra_texture.get_frame_info(event["result_handle"])
        info.update({key: frame[key] for key in
                     ("progression", "frame_number", "status") if key in frame})

    subscription = get_eventdispatcher().observe_event(
        observer_name=name, event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,
        on_event=on_frame, filter=product.hydra_texture.get_event_key(),
    )
    started = time.monotonic()
    try:
        for update in range(args.render_updates):
            app.update()
            pending = int(omni.usd.get_context().get_stage_loading_status()[2])
            if update % 4 == 0:
                print(json.dumps({"event": "render", "task": record["task_id"],
                    "updates": update + 1, "spp": info.get("progression"),
                    "pending": pending, "elapsed_s": time.monotonic() - started}), flush=True)
            if update >= 15 and pending == 0 and int(info.get("progression", 0)) >= args.samples:
                break
        rgb = np.asarray(rgb_source.get_data())
        spp = int(info.get("progression", 0))
        if (pending or spp < args.samples
                or rgb.shape not in ((args.height, args.width, 3), (args.height, args.width, 4))):
            raise RuntimeError(f"Incomplete native frame: spp={spp}, pending={pending}, shape={rgb.shape}")
        rgb = rgb[:, :, :3].astype(np.uint8)
        if rgb.std() < 2:
            raise RuntimeError("Nearly blank native frame")
        calibration = camera_calibration(parameters.get_data(), width=args.width, height=args.height)
        image_path = output / (name + ".png")
        Image.fromarray(rgb).save(image_path)
        record["images"] = {name: {
            "path": str(image_path), "sha256": digest(image_path),
            "size_px": [args.width, args.height], "native_rtx": True,
            "actual_samples_per_pixel": spp, "pending_stage_loads": pending,
            "camera": authored, "camera_calibration": calibration,
            "elapsed_s": time.monotonic() - started, "updates": update + 1,
        }}
    finally:
        rgb_source.detach([product])
        parameters.detach([product])
        subscription = None
        product.destroy()


def pose_robot(stage, scene, initial, physical_links):
    """Place all original robot links using the public robot-only FK model."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    from bench.control.factory import public_robot_geometry, robot_configuration

    geometry = public_robot_geometry(robot_configuration(scene))
    members = geometry.sides.values() if hasattr(geometry, "sides") else [geometry]
    world = {}
    for model in members:
        joints = model.checked_joints(
            {name: initial["joint_positions_rad"][name] for name in model.names},
            enforce_limits=True,
        )
        poses = model.arm.forward(joints, model.arm_base)
        offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
        poses.update(model.hand.forward(joints, poses[model.arm_tip] @ offset))
        if set(world) & set(poses):
            raise ValueError("Robot FK link names must be unique across instances")
        world.update(poses)
    targets = {
        str(prim.GetPath()): np.asarray(world[prim.GetName()])
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot"))
        if prim.GetName() in world and prim.IsA(UsdGeom.Xform)
    }
    if not set(physical_links.values()) <= set(targets):
        raise ValueError("Static FK is missing physical robot links")
    cache = UsdGeom.XformCache()
    for path in sorted(targets, key=lambda value: (value.count("/"), value)):
        prim = stage.GetPrimAtPath(path)
        parent = cache.GetLocalToWorldTransform(prim.GetParent())
        transform = UsdGeom.Xformable(prim)
        transform.ClearXformOpOrder()
        transform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "galleryPose").Set(
            Gf.Matrix4d(targets[path].T.tolist()) * parent.GetInverse()
        )
        cache.Clear()
    maximum_error = max(
        float(np.max(np.abs(
            np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T - expected
        ))) for path, expected in targets.items()
    )
    if maximum_error > 1e-7:
        raise ValueError(f"Static robot pose differs from source FK: {maximum_error}")
    return {"posed_links": len(targets), "physical_links": len(physical_links),
            "maximum_world_matrix_error": maximum_error}


def render_task(args, app, index, definition, robot_cache):
    import omni.timeline
    import omni.usd
    from pxr import UsdPhysics
    from bench.simulation.isaacsim_backend.robot_asset import import_robot
    from bench.simulation.isaacsim_backend.scene import build_scene

    family, task_id = definition["family"], definition["task_id"]
    loaded = load_task(family, task_id)
    release = Path(definition["source_release"]["path"])
    _, scene, task, initial = task_inputs(release, task_id, seed=0)
    # The normalized loader resolves frozen observer paths and seed-owned public
    # markers. It must retain the exact parent selected by the current catalog.
    if scene["workroom"] != loaded["scene"]["workroom"]:
        raise ValueError("Normalized scene changed the frozen parent")
    if args.workbench_only:
        from scenes.workrooms.teleop_workbench import apply
        scene = apply(scene, definition)
    filename = f"{index:02d}_{family}__{task_id}"
    camera_name = "task_" + filename
    image_path = args.output / (filename + ".png")
    if image_path.exists():
        raise FileExistsError(f"Preserve earlier images: {image_path}")
    record_dir = args.run_output / filename
    record_dir.mkdir()
    record = {
        "family": family, "task_id": task_id, "title": definition["title"],
        "release": str(release),
        "release_manifest_sha256": digest(release / "release_manifest.json"),
        "data_manifest_sha256": digest(data_root() / 'MANIFEST.json'),
        "environment": scene["workroom"],
        "state": "canonical authored reset with seed 0; no settling or actions",
        "physics_steps": 0, "agent_calls": 0, "source_root": str(PROJECT),
        "scope": "Static native RTX presentation; no physical task success claim",
        "scene_variant": scene.get("teleop_scene_variant"),
    }
    robot = scene["robot"]
    key = json.dumps({k: robot.get(k) for k in
                      ("assets_root", "hand_id", "assembly_version", "instances")},
                     sort_keys=True)
    if key not in robot_cache:
        assets = robot.get("assets_root", f"{DATA_ROOT}/assets")
        if robot["hand_id"] == 'dual_fr3_sharpa_w01':
            robot_cache[key] = packaged_bimanual(assets, robot['assembly_version'])
        else:
            robot_cache[key] = import_robot(assets, hand_id=robot["hand_id"],
                assembly_version=robot.get("assembly_version"), instances=robot.get("instances"))[0]
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    # Task bundles and particle layers require a declared owner, even when the
    # timeline stays stopped. This scene is never advanced by the renderer.
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    info = build_scene(stage, scene, robot_cache[key])
    record["robot_pose"] = pose_robot(stage, scene, initial, info["robot_link_paths"])
    record["object_count"] = len(info["objects"])
    record["fixture_count"] = len(info["fixtures"])
    camera = deepcopy(scene["cameras"][args.camera])
    if camera.get("kind", "world") != "world":
        raise ValueError("Gallery requires a world camera")
    camera["resolution"] = [args.width, args.height]
    # Preserve authored vertical framing when the requested aspect changes.
    vertical = float(camera.get("vertical_aperture_mm", 27.0))
    camera["vertical_aperture_mm"] = vertical
    camera["horizontal_aperture_mm"] = vertical * args.width / args.height
    settings = SimpleNamespace(
        task=task_id, cameras=[camera_name], width=args.width, height=args.height,
        samples=args.samples, render_updates=args.render_updates,
    )
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError("Presentation must not start physics")
    before = timeline.get_current_time()
    capture_image(stage, app, camera_name, camera, record_dir, settings, record)
    if timeline.is_playing() or timeline.get_current_time() != before:
        raise RuntimeError("Rendering advanced the physics timeline")
    captured = record_dir / (camera_name + ".png")
    captured.rename(image_path)
    record["images"][camera_name]["path"] = str(image_path)
    record["status"] = "rendered"
    dump(record_dir / "provenance.json", record)
    print(json.dumps({"event": "task_complete", "task": task_id,
                      "image": str(image_path)}, ensure_ascii=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--family")
    parser.add_argument("--task", action="append",
                        help="Canonical task ID; repeat to render a selected batch")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2880)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--render-updates", type=int, default=240)
    parser.add_argument("--camera", default="task")
    parser.add_argument("--workbench-only", action="store_true",
                        help="Render the same workbench-only variant used for PICO practice")
    args = parser.parse_args()
    if min(args.width, args.height, args.samples, args.render_updates, args.shard_count) <= 0:
        parser.error("Dimensions, sample counts and shard count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.startswith("GPU-") or "," in gpu:
        parser.error("Select one physical GPU UUID through CUDA_VISIBLE_DEVICES")
    source = verify_if_snapshot(PROJECT, required_modules=("scenes.workrooms.catalog",),
                                entrypoint_file=Path(__file__))
    if source is None:
        parser.error("Run from a complete frozen source snapshot")
    catalog = list_tasks()
    unknown = set(args.task or ()) - {task["task_id"] for task in catalog}
    if unknown:
        parser.error(f"Unknown canonical task IDs: {sorted(unknown)}")
    selected = [(i, task) for i, task in enumerate(catalog, 1)
                if (not args.family or task["family"] == args.family)
                and (not args.task or task["task_id"] in args.task)
                and (i - 1) % args.shard_count == args.shard_index]
    if not selected:
        parser.error("No catalog task matches the requested selection")
    args.output.mkdir(parents=True, exist_ok=True)
    args.run_output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "catalog_task_png_gallery.v1", "status": "initializing",
        "created_utc": now(), "source_snapshot": source, "gpu_uuid": gpu,
        "runtime": "Isaac Sim 6.1.0 native RTX PathTracing",
        "physics_steps": 0, "agent_calls": 0,
        "width": args.width, "height": args.height, "samples": args.samples,
        "workbench_only": args.workbench_only,
        "selected_tasks": [task["task_id"] for _, task in selected],
        "tasks": [], "errors": [],
    }
    app = None
    exit_code = 1
    try:
        from isaacsim import SimulationApp
        from bench.simulation.isaacsim_backend.render_sync import startup_arguments, verify_settings
        extra = startup_arguments()
        override = os.environ.get("ISAAC_ALLOW_UNVALIDATED_DRIVER") == "1"
        if override:
            extra.append("--/rtx/verifyDriverVersion/enabled=false")
        report["isaac_allow_unvalidated_driver"] = override
        app = SimulationApp({
            "headless": True, "width": 640, "height": 480,
            "disable_viewport_updates": True,
            "multi_gpu": False, "active_cuda_gpus": [0], "max_gpu_count": 1,
            "physics_gpu": 0, "renderer": "PathTracing",
            "fast_shutdown": True, "extra_args": extra,
        }, experience=str(PROJECT / "bench/apps/workroom.headless.kit"))
        import carb
        report["render_settings"] = verify_settings(carb.settings.get_settings())
        report["render_settings"]["sync_operation"] = (
            "SimulationApp.update; stopped timeline, no World.render or physics steps"
        )
        report["default_viewport"] = {"size_px": [640, 480], "updates_enabled": False}
        robot_cache = {}
        for index, definition in selected:
            try:
                report["tasks"].append(render_task(args, app, index, definition, robot_cache))
            except Exception:
                report["errors"].append({"family": definition["family"],
                    "task_id": definition["task_id"], "traceback": traceback.format_exc()})
                traceback.print_exc()
            dump(args.run_output / "report.json", report)
        exit_code = int(bool(report["errors"]))
        report["status"] = "completed" if exit_code == 0 else "partial_failure"
    except BaseException:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report["completed_utc"] = now()
        dump(args.run_output / "report.json", report)
        if app is not None:
            app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
