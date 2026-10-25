"""Native object-object scoop/loading/carry/pour probe, without hand control.

The canonical scoop is dynamic. This diagnostic alone prescribes smooth
kinematic poses on that one tool. All 24 granules and the weighing cup are
passive and never receive pose/velocity/force writes after initialization.
The cup is separately placed on its real platform before reset; transfer of the
cup by a robot is not claimed. No force-balance accuracy is inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from bench.paths import OUTPUT_ROOT, load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot

PROJECT = Path(__file__).resolve().parents[2]
DT = .0005


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(value):
    return value.detach().cpu().numpy().copy() if hasattr(value, "detach") else np.asarray(value).copy()


def rotation(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


# Time, scoop origin, pitch (degrees). Minimum jerk segments are finite and
# continuous; no grain target or force is derived from their measured state.
KEYFRAMES = (
    (0., (.535, .488, .9032), 0.),
    (.6, (.535, .488, .9032), 0.),
    (2.6, (.642, .488, .9032), 0.),
    (3.2, (.642, .488, .9532), -20.),
    (3.8, (.642, .488, 1.025), -20.),
    (9.8, (.660, 2.075, 1.005), -20.),
    (11.5, (.660, 2.075, 1.005), 75.),
    (13.0, (.660, 2.075, 1.005), 75.),
    (14.0, (.600, 2.075, 1.105), 75.),
    (15.0, (.600, 2.075, 1.105), 0.),
    (16.0, (.600, 2.075, 1.105), 0.),
)


def target_at(time):
    for a, b in zip(KEYFRAMES, KEYFRAMES[1:]):
        if time <= b[0]+1e-10:
            u = np.clip((time-a[0])/(b[0]-a[0]), 0., 1.)
            s = u**3*(10-15*u+6*u*u)
            xyz = (1-s)*np.asarray(a[1])+s*np.asarray(b[1])
            angle = math.radians((1-s)*a[2]+s*b[2])
            return np.r_[xyz, 0., math.sin(angle/2), 0., math.cos(angle/2)]
    return target_at(KEYFRAMES[-1][0])


def classify(poses, names, meta):
    by_id = dict(zip(names, poses))
    radius = meta["task_parameters"]["pellet_radius_m"]
    grain_names = meta["task_parameters"]["pellet_ids"]
    result = {"cup": [], "scoop": [], "source": [], "outside": []}
    cup = by_id["weighing_cup"]
    scoop = by_id["scoop"]
    for name in grain_names:
        xyz = by_id[name][:3]
        cup_local = rotation(cup[3:]).T@(xyz-cup[:3])
        scoop_local = rotation(scoop[3:]).T@(xyz-scoop[:3])
        source_local = xyz-np.asarray(meta["task_parameters"]["source_tray"]["initial_position_m"])
        # Closed sphere extent must be within the declared cavity. Bottom
        # tolerance accommodates the same explicit contact skin as the asset.
        if np.all(cup_local >= np.array([-.0345+radius, -.0295+radius, .003+radius])-.0005) and np.all(
                cup_local <= np.array([.0345-radius, .0295-radius, .035-radius])+.0005):
            result["cup"].append(name)
        elif np.all(scoop_local >= np.array([-.027+radius, -.019+radius, .002+radius])-.0006) and np.all(
                scoop_local <= np.array([.035, .019-radius, .040])+.0006):
            result["scoop"].append(name)
        elif np.all(source_local >= np.array([-.075+radius, -.052+radius, .003+radius])-.0005) and np.all(
                source_local <= np.array([.072-radius, .052-radius, .029])+.0005):
            result["source"].append(name)
        else:
            result["outside"].append(name)
    return result


def run(args):
    source = verify_if_snapshot(PROJECT, required_modules=(
        "scripts.validation.laboratory_solid_interactions",), entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve() != PROJECT:
        raise ValueError("Run from a complete frozen source root")
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError("Evidence belongs in shared datasets")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "laboratory_solid_interactions.v1", "passed": False,
        "source": source, "source_manifest_sha256": digest(PROJECT/"source_snapshot_manifest.json"),
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "physics_dt_s": DT,
        "model_calls": 0, "hand_object_interaction_tested": False,
        "robot_task_success_claimed": False, "quantitative_force_balance_claimed": False,
        "scope": __doc__, "boundary": {"body": "scoop", "canonical_mode": "dynamic",
            "diagnostic_mode": "kinematic", "api": "RigidBodyView.set_transforms",
            "kinematic_targets_used": False,
            "boundary_limitations": "The pinned GPU backend does not implement kinematic targets; "
                "the tool receives smooth substep poses. Contact conveys displacement but no "
                "claim of quantitatively calibrated moving-wall friction is made.",
            "keyframes": KEYFRAMES, "only_tool_driven": True,
            "cup_reset": [.71, 2.075, .9255], "granule_reset": "canonical supply positions",
            "granule_state_writes_after_reset": False}}
    app = world = None
    records, poses_all, velocities_all, contact_all, support_all = [], [], [], [], []
    exit_code = 1
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "width": 640, "height": 480,
            "multi_gpu": False, "active_cuda_gpus": [0], "max_gpu_count": 1, "physics_gpu": 0,
            "renderer": "RayTracedLighting", "fast_shutdown": True,
            "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]
            if os.environ.get("ISAAC_ALLOW_UNVALIDATED_DRIVER") == "1" else []},
            experience=str(PROJECT/"bench/apps/workroom.headless.kit"))
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
        from scenes.workrooms.laboratory.tasks.solid_weighing import assets
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1., physics_dt=DT, rendering_dt=1/30,
            backend="torch", device="cuda:0", physics_prim_path="/World/PhysicsScene")
        physics = world.get_physics_context()
        physics.enable_gpu_dynamics(True)
        physics.set_broadphase_type("GPU")
        physics.set_solver_type("TGS")
        meta = assets.author_task(stage)
        dump(output/"canonical_metadata.json", meta)
        stage.Export(str(output/"canonical_task.usda"))
        for label, position, size in (
                ("island_support", (.65, .35, .875), (1., .6, .05)),
                ("rear_support", (.71, 2.075, .875), (.65, .3, .05))):
            box = UsdGeom.Cube.Define(stage, "/World/"+label)
            box.CreateSizeAttr(1.)
            box.AddTranslateOp().Set(Gf.Vec3d(*position))
            box.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        scoop_prim = stage.GetPrimAtPath(meta["actors"]["scoop"])
        UsdPhysics.RigidBodyAPI(scoop_prim).CreateKinematicEnabledAttr(True)
        scoop_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*KEYFRAMES[0][1]))
        cup_prim = stage.GetPrimAtPath(meta["actors"]["weighing_cup"])
        cup_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(.71, 2.075, .9255))
        for path in meta["actors"].values():
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
        stage.Export(str(output/"diagnostic_initial.usda"))
        world.reset()
        sim = world.physics_sim_view
        bodies = sim.create_rigid_body_view(meta["root_path"]+"/Actors/*")
        names = [path.rsplit("/", 1)[-1] for path in bodies.prim_paths]
        if set(names) != set(meta["actors"]):
            raise RuntimeError("Native rigid-body inventory differs")
        tool = sim.create_rigid_body_view(meta["actors"]["scoop"])
        device = tool.get_transforms().device
        indices = torch.tensor([0], dtype=torch.int32, device=device)
        contact = sim.create_rigid_contact_view(meta["root_path"]+"/Actors/pellet_*",
            filter_patterns=[meta["actors"]["scoop"], meta["actors"]["weighing_cup"]],
            max_contact_data_count=4096)
        support = sim.create_rigid_contact_view(meta["actors"]["weighing_cup"],
            filter_patterns=[meta["fixtures"]["balance_platform"]+"/Collision/pan"],
            max_contact_data_count=256)
        report["native_body_order"] = names
        report["native_masses_kg"] = array(bodies.get_masses()).reshape(-1).tolist()
        report["contact_mapping"] = {"sensors": list(contact.sensor_paths),
            "filters": list(contact.filter_paths)}
        report["asset_source_sha256"] = digest(PROJECT/"scenes/workrooms/laboratory/tasks/solid_weighing/assets.py")
        report["runtime_version"] = (Path(os.environ["ISAAC_PATH"])/"VERSION").read_text().strip()
        for tick in range(1, round(KEYFRAMES[-1][0]/DT)+1):
            time = tick*DT
            tool.set_transforms(torch.tensor(target_at(time)[None],
                dtype=torch.float32, device=device), indices)
            world.step(render=False)
            positions = array(bodies.get_transforms())
            velocities = array(bodies.get_velocities())
            forces = array(contact.get_contact_force_matrix(DT))
            support_force = array(support.get_contact_force_matrix(DT))
            if not all(np.isfinite(x).all() for x in (positions, velocities, forces)):
                raise RuntimeError("Nonfinite native state")
            poses_all.append(positions)
            velocities_all.append(velocities)
            contact_all.append(forces)
            support_all.append(support_force)
            if tick % 40 == 0:
                state = classify(positions, names, meta)
                records.append({"tick": tick, "time_s": time, "contents": state})
            if tick % 2000 == 0:
                print(json.dumps({"time_s": time, "counts": {k: len(v) for k, v in state.items()},
                    "scoop_position_m": positions[names.index("scoop"), :3].tolist()}), flush=True)
        world.pause()
        pose_array = np.asarray(poses_all)
        velocity_array = np.asarray(velocities_all)
        contact_array = np.asarray(contact_all)
        np.savez_compressed(output/"native_samples.npz", poses_xyzw=pose_array,
            velocities_linear_angular=velocity_array, grain_contact_forces_n=contact_array,
            cup_platform_contact_forces_n=np.asarray(support_all),
            physics_dt_s=DT, body_names=np.array(names))
        dump(output/"contents.json", records)
        final = records[-1]["contents"]
        mid = [r for r in records if 4.0 <= r["time_s"] <= 9.6]
        tail = velocity_array[-round(.5/DT):]
        cup_index = names.index("weighing_cup")
        final_mass = len(final["cup"])*assets.PELLET_MASS_KG
        report["final_contents"] = final
        report["mass_ledger_kg"] = {"initial": .012, "cup": final_mass,
            "source": len(final["source"])*assets.PELLET_MASS_KG,
            "scoop": len(final["scoop"])*assets.PELLET_MASS_KG,
            "outside": len(final["outside"])*assets.PELLET_MASS_KG}
        report["cup_final_speed_m_s"] = float(np.linalg.norm(tail[:, cup_index, :3], axis=-1).max())
        report["cup_final_angular_speed_rad_s"] = float(np.linalg.norm(tail[:, cup_index, 3:], axis=-1).max())
        grain_indices = [names.index(name) for name in final["cup"]]
        report["cup_granules_final_max_speed_m_s"] = float(
            np.linalg.norm(tail[:, grain_indices, :3], axis=-1).max()) if grain_indices else None
        report["cup_platform_contact_final_min_z_n"] = float(np.asarray(support_all)[-round(.5/DT):, ..., 2].min())
        report["minimum_carried_granules"] = min(len(r["contents"]["scoop"]) for r in mid)
        force_norm = np.linalg.norm(contact_array, axis=-1)
        report["native_contact_ticks"] = {"grain_scoop": int((force_norm[:, :, 0].max(-1) > 1e-5).sum()),
            "grain_cup": int((force_norm[:, :, 1].max(-1) > 1e-5).sum())}
        report["checks"] = {
            "passive_loading_and_carry": report["minimum_carried_granules"] >= 10,
            "target_mass": abs(final_mass-.006) <= .0006+1e-12,
            "spill_within_budget": len(final["outside"]) <= 2,
            "empty_scoop": len(final["scoop"]) == 0,
            "real_scoop_contact": report["native_contact_ticks"]["grain_scoop"] > 100,
            "real_cup_contact": report["native_contact_ticks"]["grain_cup"] > 100,
            "cup_released_stable": report["cup_final_speed_m_s"] < .02 and report["cup_final_angular_speed_rad_s"] < .2,
            "cup_supported_on_its_platform": report["cup_platform_contact_final_min_z_n"] > .01
                and abs(float(pose_array[-1, cup_index, 2])-.925) < .001,
            "cup_granules_settled": report["cup_granules_final_max_speed_m_s"] is not None
                and report["cup_granules_final_max_speed_m_s"] < .02,
            "inventory_conserved": sum(len(v) for v in final.values()) == 24}
        report["native_steps"] = len(poses_all)
        report["native_duration_s"] = float(world.current_time)
        report["passed"] = all(report["checks"].values())
        report["source_final"] = verify_if_snapshot(PROJECT, required_modules=(
            "scripts.validation.laboratory_solid_interactions",
            "scenes.workrooms.laboratory.tasks.solid_weighing.assets"), entrypoint_file=Path(__file__))
        exit_code = 0 if report["passed"] else 1
    except BaseException as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        report["requires_external_exit_zero"] = True
        dump(output/"qualification.json", report)
        print(json.dumps({"passed": report["passed"], "output": str(output)}), flush=True)
        if app is not None:
            app.close(exit_code=exit_code)
    return exit_code


def review_native(args):
    """Reassess the complete recorded trace for nominal-mass readout.

    Slow horizontal rolling of spheres does not change a supported mass
    inventory. Preserve the original strict all-axis-speed result verbatim.
    This replay uses the agreed 20 ms vertical-motion bounds and 300 ms
    same-ID dwell; it never changes physics, original evidence or asset files.
    """
    source = verify_if_snapshot(PROJECT, required_modules=(
        "scripts.validation.laboratory_solid_interactions",), entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve() != PROJECT:
        raise ValueError("Review requires a complete frozen source root")
    native = args.review_native.resolve()
    output = args.output.resolve()
    if not native.is_relative_to(OUTPUT_ROOT) or not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError("Recorded evidence and review belong in datasets")
    output.mkdir(parents=True, exist_ok=False)
    original = load_json(native/"qualification.json")
    metadata = load_json(native/"canonical_metadata.json")
    data = np.load(native/"native_samples.npz")
    original_source = Path(original["source"]["snapshot_root"])
    if digest(original_source/"source_snapshot_manifest.json") != original["source_manifest_sha256"]:
        raise ValueError("Original source manifest hash changed")
    from scripts.maintenance.freeze_project_sources import verify
    original_source_verified = verify(original_source)
    from pxr import Usd
    from scenes.workrooms.laboratory.tasks.solid_weighing.assets import author_task
    if metadata != author_task(Usd.Stage.CreateInMemory()):
        raise ValueError("Canonical asset metadata differs from the frozen author")
    if digest(PROJECT/"scenes/workrooms/laboratory/tasks/solid_weighing/assets.py") != original["asset_source_sha256"]:
        raise ValueError("Asset author differs from the measured native source")
    poses, velocities, forces = (data[name] for name in (
        "poses_xyzw", "velocities_linear_angular", "grain_contact_forces_n"))
    dt = float(data["physics_dt_s"])
    names = data["body_names"].tolist()
    if dt != DT or names != original["native_body_order"] or len(poses) != original["native_steps"]:
        raise ValueError("Raw trace shape, identity or timing differs")
    if not all(np.isfinite(a).all() for a in (poses, velocities, forces)):
        raise ValueError("Nonfinite recorded native sample")
    dwell = round(.3/dt)
    window = round(.02/dt)
    states = [classify(row, names, metadata) for row in poses[-dwell:]]
    cup_ids = states[-1]["cup"]
    if not cup_ids:
        raise ValueError("No cup inventory to assess")
    grain_indices = [names.index(name) for name in cup_ids]
    sensors = [path.rsplit("/", 1)[-1] for path in original["contact_mapping"]["sensors"]]
    sensor_indices = [sensors.index(name) for name in cup_ids]
    z = poses[-dwell-window:, grain_indices, 2]
    vertical_rate = np.abs(z[window:]-z[:-window])/.02
    vertical_envelope = np.ptp(np.lib.stride_tricks.sliding_window_view(z, window+1, axis=0), axis=-1)
    grain_support = forces[-dwell-window+1:, sensor_indices, 1, 2]
    support_means = np.lib.stride_tricks.sliding_window_view(grain_support, window, axis=0).mean(-1)
    cup_index, scoop_index = names.index("weighing_cup"), names.index("scoop")
    cup_velocity = velocities[-dwell:, cup_index]
    cup_platform_z = data["cup_platform_contact_forces_n"][-dwell:, ..., 2]
    objects = {item["id"]: item for item in metadata["objects"]}
    corners = {name: np.array([(x, y, z) for x in (bounds[0][0], bounds[1][0])
        for y in (bounds[0][1], bounds[1][1]) for z in (bounds[0][2], bounds[1][2])])
        for name in ("weighing_cup", "scoop")
        for bounds in (objects[name]["local_bounds_m"],)}
    clearances = []
    for row in poses[-dwell:]:
        boxes = []
        for name, index in (("weighing_cup", cup_index), ("scoop", scoop_index)):
            points = corners[name]@rotation(row[index, 3:]).T+row[index, :3]
            boxes.append((points.min(0), points.max(0)))
        low = np.maximum(np.maximum(boxes[0][0]-boxes[1][1], boxes[1][0]-boxes[0][1]), 0.)
        clearances.append(float(np.linalg.norm(low)))
    final_velocity = velocities[-dwell:, grain_indices, :3]
    net_mass = sum(objects[name]["mass_kg"] for name in cup_ids)
    strict_failed = [name for name, passed in original["checks"].items() if not passed]
    checks = {
        "original_native_run_complete": original.get("failure") is None and original["native_steps"] == 32000,
        "original_contact_loading_and_transfer_checks": all(value for key, value in original["checks"].items()
            if key != "cup_granules_settled"),
        "same_full_cup_inventory_for_300ms": all(state["cup"] == cup_ids for state in states),
        "target_nominal_mass": abs(net_mass-.006) <= .0006+1e-12,
        "native_cup_support_for_every_granule_in_each_20ms_window": bool(np.all(support_means > 1e-5)),
        "vertical_20ms_rate_at_most_003_m_s": bool(np.all(vertical_rate <= .03)),
        "vertical_20ms_envelope_at_most_004_m": bool(np.all(vertical_envelope <= .004)),
        "cup_3d_linear_stability": bool(np.linalg.norm(cup_velocity[:, :3], axis=-1).max() < .02),
        "cup_3d_angular_stability": bool(np.linalg.norm(cup_velocity[:, 3:], axis=-1).max() < .2),
        "cup_on_real_platform": bool(np.all(cup_platform_z > .01)),
        "tool_fully_clear_of_cup": min(clearances) > .01,
        "inventory_conserved": all(sum(len(v) for v in state.values()) == 24 for state in states),
        "terminal_spill_within_budget": len(states[-1]["outside"])*.0005 <= .001,
    }
    inputs = [native/name for name in (
        "qualification.json", "canonical_metadata.json", "native_samples.npz", "contents.json")]
    report = {"schema": "laboratory_solid_nominal_mass_review.v1", "passed": all(checks.values()),
        "source": source, "review_source_manifest_sha256": digest(PROJECT/"source_snapshot_manifest.json"),
        "original_source_verified": original_source_verified,
        "native_evidence_directory": str(native), "input_sha256": {str(p): digest(p) for p in inputs},
        "original_native_passed": original["passed"], "original_strict_failed_checks": strict_failed,
        "original_evidence_modified": False, "new_native_simulation_performed": False,
        "robot_task_success_claimed": False, "force_calibration_claimed": False,
        "scope": "Nominal mass readout and object interaction reassessment from the original full native trace. "
            "Spheres may roll slowly horizontally while supported; cup stays still and unloaded. "
            "This is not a new native run or an override of its historical strict result.",
        "criteria": {"same_id_dwell_s": .3, "native_substeps_reviewed": dwell,
            "motion_window_s": .02, "maximum_vertical_rate_m_s": .03,
            "maximum_vertical_envelope_m": .004, "horizontal_sphere_rolling_permitted": True},
        "cup_ids": cup_ids, "mass_ledger_kg": original["mass_ledger_kg"],
        "measurements": {"maximum_granule_total_speed_m_s": float(np.linalg.norm(final_velocity, axis=-1).max()),
            "maximum_granule_vertical_speed_m_s": float(np.abs(final_velocity[:, :, 2]).max()),
            "maximum_vertical_20ms_rate_m_s": float(vertical_rate.max()),
            "maximum_vertical_20ms_envelope_m": float(vertical_envelope.max()),
            "minimum_each_granule_20ms_mean_cup_support_n": float(support_means.min()),
            "minimum_cup_platform_support_n": float(cup_platform_z.min()),
            "minimum_tool_cup_aabb_clearance_m": min(clearances),
            "maximum_cup_linear_speed_m_s": float(np.linalg.norm(cup_velocity[:, :3], axis=-1).max()),
            "maximum_cup_angular_speed_rad_s": float(np.linalg.norm(cup_velocity[:, 3:], axis=-1).max())},
        "checks": checks,
        "source_final": verify_if_snapshot(PROJECT, required_modules=(
            "scripts.validation.laboratory_solid_interactions",
            "scenes.workrooms.laboratory.tasks.solid_weighing.assets"), entrypoint_file=Path(__file__))}
    dump(output/"qualification.json", report)
    print(json.dumps({"passed": report["passed"], "output": str(output)}))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-native", type=Path,
        help="Read-only reassessment of an existing native trace for nominal-mass readout")
    args = parser.parse_args()
    raise SystemExit(review_native(args) if args.review_native else run(args))


if __name__ == "__main__":
    main()
