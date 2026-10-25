"""Offline, evidence-scoped physical checks for a closed NativeWorker recording.

This module never imports Isaac, Newton, MuJoCo, or starts a simulator. ``passed``
means the required checks below passed, not independent certification of every
rigid/robot collision mesh, a calibrated material, or continuous-time clearance.
The caller must keep this detailed report private: it includes teacher state.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.isaacsim import audit as frozen_audit
from bench.evaluation.cloth.audit_cloth_surface import cylinder_sdf, summarize_surface
from bench.geometry import quaternion_matrix
from bench.convex_geometry import convex_triangle_audit
from bench.evaluation.rigid_safety import RIGID_CONTACT_DIAGNOSTIC_M, exceeds_rigid_penetration
from bench.paths import load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset, load_obj_mesh


PROJECT = Path(__file__).resolve().parents[2]
# These values are the existing independent native audit's declared numerical
# gates. They are not textile parameters or measured solver accuracy.
MAX_EDGE_RATIO = 1.25
CLOTH_PENETRATION_M = 1e-4
CYLINDER_SUBDIVISIONS = 12


def physics_profile() -> dict:
    """Versioned AgentBench gates; this does not modify an old frozen release."""
    return {"id": "agent_bench_physics_cloth_single_acquisition_v2", "max_cloth_edge_ratio": MAX_EDGE_RATIO,
            "cloth_degenerate_double_area_m2": 1e-12,
            "cloth_penetration_tolerance_m": CLOTH_PENETRATION_M,
            "max_reported_rigid_penetration_m": RIGID_CONTACT_DIAGNOSTIC_M,
            "rigid_penetration_scope": "all reported rigid pairs",
            "analytic_cylinder_subdivisions": CYLINDER_SUBDIVISIONS,
            "threshold_provenance": "Cloth gates reuse native audit defaults; the 2 mm contact diagnostic is explicitly extended to all reported rigid pairs in this new profile.",
            "material_or_solver_accuracy_calibrated": False,
            "cloth_substep_acceptance": {
                "protocol": "cloth_agent_substep_physics_observer_v1", "native_dt_s": .0005,
                "control_dt_s": .02, "required_for_cloth_enabled": True,
                "required": ["complete_native_cooked_inventory", "enabled_robot_self_collision",
                             "every_existing_substep_raw_state_and_rigid_contacts", "exact_20ms_ACK_mapping",
                             "readonly_observer_and_normal_worker_seal", "offline_geometry_recomputation",
                             "single_contact_acquisition_and_delivery_each_substep"],
                "scope": "Recorded native substeps, conservative cooked envelopes and reported contacts; not continuous CCD or calibrated material accuracy."}}


def physics_profile_sha256() -> str:
    return hashlib.sha256(json.dumps(physics_profile(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def audit_source_paths() -> tuple[Path, ...]:
    """All project implementation/threshold dependencies to include in snapshots."""
    return tuple(PROJECT / name for name in (
        "bench/evaluation/physics.py", "bench/evaluation/rigid_safety.py", "bench/isaacsim/audit.py",
        "bench/evaluation/cloth/audit_cloth_surface.py", "bench/evaluation/cloth/audit_cloth_self_intersection.py",
        "bench/evaluation/cloth/cloth_self_intersection_precomputed.py",
        "scenes/cloth/layer_identity.py",
        "bench/convex_geometry.py", "bench/geometry.py",
        "scenes/cloth/__init__.py", "scenes/cloth/geometry_guard.py",
        "scenes/cloth/worker_observer.py", "scenes/cloth/contact_delivery.py",
        "bench/simulation/isaacsim_backend/camera_calibration.py",
        "bench/simulation/isaacsim_backend/contact_sample.py", "bench/evaluation/cloth/cloth_robot_collision.py",
        "bench/simulation/isaacsim_backend/contact_precision.py",
        "bench/simulation/isaacsim_backend/robot_asset.py",
        "bench/simulation/isaacsim_backend/asset_catalog.py",
        "bench/simulation/isaacsim_backend/usd_asset.py",
        "bench/simulation/assets/task_parts.py", "bench/simulation/assets/__init__.py"))


def _need(condition, message):
    if not condition:
        raise frozen_audit.MissingEvidence(message)


def _require(condition, message):
    frozen_audit.require(condition, message)


def _check(report, name, function, *, required=True, applicable=True):
    if not applicable:
        row = {"status": "unchecked", "reason": "Not applicable to this recorded scene"}
    else:
        try:
            row = dict(function())
            status = row.pop("status", "passed")
            row["status"] = {"pass": "passed", "fail": "failed", "unverified": "unchecked"}.get(status, status)
            _require(row["status"] in {"passed", "failed", "unchecked"}, "Invalid audit status")
        except (frozen_audit.MissingEvidence, FileNotFoundError, ImportError, KeyError) as error:
            row = {"status": "unchecked", "reason": str(error), "error_type": type(error).__name__}
        except Exception as error:
            row = {"status": "failed", "reason": str(error), "error_type": type(error).__name__}
    row.update(required=required and applicable, applicable=applicable)
    report["checks"][name] = row


def _finish(report):
    active = [c for c in report["checks"].values() if c["required"] and c["applicable"]]
    counts = Counter(c["status"] for c in active)
    report["required_check_counts"] = dict(counts)
    report["status"] = "failed" if counts["failed"] else "unchecked" if not active or counts["unchecked"] else "passed"
    report["physics_valid"] = report["status"] == "passed"
    return report


def _context(root):
    for partial in ("unacknowledged_step.json", "unacknowledged_step_state.npz"):
        _require(not (root / partial).exists(), f"Unacknowledged native step evidence: {partial}")
    states = frozen_audit.read_states(root / "states.npz")
    _need(set(frozen_audit.STATE_KEYS) <= set(states), "Required physical state arrays are absent")
    metadata, scene = (frozen_audit.read_json(root / (name + ".json")) for name in ("metadata", "scene"))
    path = root / "physics.jsonl"
    _need(path.is_file(), "Missing physics.jsonl reset/step evidence")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    _require(bool(records) and records[0].get("kind") == "reset", "Recording must start with reset")
    physical = []
    for index, record in enumerate(records):
        kind = record.get("kind")
        _require(kind in {"reset", "step", "cancel_hold_targets"}, "Unknown physical record kind")
        _require(kind != "reset" or index == 0, "Unexpected mid-episode reset")
        if kind in {"reset", "step"}:
            physical.append(record)
    return {"states": states, "metadata": metadata, "scene": scene,
            "observations": [r["observation"] for r in physical],
            "actions": [r["command"] for r in physical if r["kind"] == "step"],
            "result": {"simulation_duration_s": float(states["time_s"][-1])}}


def _finite(context):
    states, meta, obs = (context[k] for k in ("states", "metadata", "observations"))
    count = len(states["time_s"])
    _require(count >= 1 and len(obs) == count and len(context["actions"]) == count - 1,
             "State/observation/action counts disagree")
    for name, array in states.items():
        _require(array.ndim >= 1 and len(array) == count and np.isfinite(array).all(),
                 f"Nonfinite or frame-misaligned state: {name}")
    for name in ("time_s", "native_world_time_s", "native_physics_step_index"):
        _require(states[name].shape == (count,), f"Expected one scalar per frame: {name}")
    _require(np.issubdtype(states["native_physics_step_index"].dtype, np.integer), "Native step counters must be integers")
    names = meta["joint_names"]
    _require(bool(names) and len(set(names)) == len(names), "Physical joint inventory empty or duplicated")
    for name in ("joint_q", "joint_qd", "joint_target_q"):
        _require(states[name].shape == (count, len(names)), f"Physical joint inventory mismatch: {name}")
    _need(meta.get("capture_state_quaternion") == "xyzw" and meta.get("observation_quaternion") == "wxyz",
          "Explicit recorded quaternion conventions are required")
    for prefix in ("body", "object"):
        labels = meta[prefix + "_names"]
        _require(len(set(labels)) == len(labels), f"Duplicated {prefix} names")
        q = states[prefix + "_q"]
        _require(q.shape == (count, len(labels), 7), f"Invalid {prefix} pose shape")
        _require(states[prefix + "_qd"].shape == (count, len(labels), 6), f"Invalid {prefix} velocity shape")
        _require(np.all(np.abs(np.linalg.norm(q[..., 3:], axis=-1) - 1) < 1e-3), f"Nonunit {prefix} quaternion")
    particles = states["particle_q"]
    _require(particles.ndim == 3 and particles.shape[-1] == 3 and particles.shape == states["particle_qd"].shape,
             "Cloth array shape mismatch")
    cloth = context["scene"]["cloth"]
    _require(isinstance(cloth["enabled"], bool), "Cloth enabled flag must be explicit boolean")
    _require(cloth["enabled"] == bool(particles.shape[1]), "Cloth enabled flag and actual particles disagree")
    for i, observation in enumerate(obs):
        _need("state_finite" in observation.get("metrics", {}), "Native state_finite flag absent")
        _require(observation["metrics"]["state_finite"] is True, "Native observation reports nonfinite state")
        for field, state in (("joint_positions_rad", "joint_q"), ("joint_velocities_rad_s", "joint_qd")):
            _require(set(observation[field]) == set(names), "Observation physical joint inventory mismatch")
            _require(np.allclose([observation[field][n] for n in names], states[state][i], atol=1e-7, rtol=0),
                     f"Observation differs from recorded physical arrays: {field}")
        _require(set(observation["objects"]) == set(meta["object_names"]), "Observation object inventory mismatch")
        for j, name in enumerate(meta["object_names"]):
            pose, q = observation["objects"][name], states["object_q"][i, j]
            _require(np.allclose(pose["position_m"], q[:3], atol=1e-7, rtol=0), "Object positions disagree")
            _require(np.allclose(pose["quaternion_wxyz"], q[[6, 3, 4, 5]], atol=1e-7, rtol=0), "Object orientations disagree")
        if particles.shape[1]:
            vertices = np.asarray(observation["cloth"]["vertices_m"])
            _require(vertices.shape == particles[i].shape and np.allclose(vertices, particles[i], atol=1e-7, rtol=0),
                     "Observed cloth differs from recorded physical arrays")
    _need("scene_spec" in meta, "Native loaded scene specification is absent")
    _require(meta["scene_spec"] == context["scene"], "Loaded native scene differs from saved audit geometry")
    return {"frames": count, "physical_joints": len(names), "cloth_vertices": particles.shape[1],
            "passive_and_mimic_joints_included": True}


def _clock(context):
    result = frozen_audit.check_clock(context)
    t = context["states"]["time_s"]
    _require(np.allclose([o["sim_time_s"] for o in context["observations"]], t, atol=1e-8, rtol=0),
             "Public/native simulation clocks disagree")
    _require(all(not isinstance(a["duration_s"], bool) and abs(a["duration_s"] - .02) < 1e-10
                 for a in context["actions"]), "Recorded step duration differs from native control clock")
    return {**result, "reset_only": len(t) == 1, "dynamic_capability_claimed": False}


def _contacts(context):
    checked = frozen_audit.check_contacts(context)
    # Its unverified status means no nonzero *capability* evidence, whereas this
    # check is only stream integrity. A well-formed empty stream is meaningful.
    checked["status"] = "passed"
    checked["contact_capability_established"] = bool(checked["nonzero_impulse_reports_by_pair_type"])
    checked.pop("examples", None)
    return checked


def _rigid_separations(context):
    minimum, measured, missing = None, 0, 0
    failures = []
    for i, observation in enumerate(context["observations"]):
        _need("contacts" in observation and "rigid" in observation["contacts"], "Missing native rigid contact stream")
        for pair in observation["contacts"]["rigid"]:
            value = pair.get("min_separation_m")
            if value is None:
                missing += 1
                continue
            _require(np.isfinite(value), "Nonfinite rigid separation")
            measured += 1
            minimum = float(value) if minimum is None else min(minimum, float(value))
            if not exceeds_rigid_penetration(value):
                continue
            row = {"frame": i, "actor0": pair.get("actor0"), "actor1": pair.get("actor1"), "min_separation_m": float(value)}
            failures.append(row)
    return {"status": "failed" if failures else "unchecked" if missing else "passed",
            "minimum_reported_separation_m": minimum, "measured_pair_reports": measured,
            "reports_without_separation": missing, "violations": failures,
            "diagnostic_threshold_m": RIGID_CONTACT_DIAGNOSTIC_M,
            "scope": "All observed solver-reported rigid pairs under the new agent_bench_physics_v1 numerical profile",
            "independent_rigid_geometry_certified": False}


def _vector(value, name, *, positive=False):
    result = np.asarray(value, float)
    _require(result.shape == (3,) and np.isfinite(result).all(), f"Invalid {name}")
    if positive:
        _require(np.all(result > 0), f"Nonpositive {name}")
    return result


def _rotation(value):
    q = np.asarray(value, float)
    _require(q.shape == (4,) and np.isfinite(q).all() and abs(np.linalg.norm(q) - 1) < 1e-3, "Invalid geometry quaternion")
    return quaternion_matrix(q)


def _task_pieces(context):
    """Keep individual convex pieces: never close a bore with a global hull."""
    from scipy.spatial import ConvexHull

    scene, meta = context["scene"], context["metadata"]
    pieces, inputs, omitted = [], {}, []
    table = scene["table"]
    specs = [("table", {"geometry": "box", "half_extents_m": table["half_extents_m"]},
              table["position_m"], [1, 0, 0, 0], None)]
    for dynamic, category in ((False, "fixtures"), (True, "objects")):
        for item in scene.get(category, []):
            asset = load_task_asset(item["asset_id"], item.get("asset_root"), verify_hashes=True)
            captured = meta["scene_info"][category][item["id"]]
            _require(asset["manifest_sha256"] == captured["manifest_sha256"], "Task catalog differs from native capture")
            _require(asset["asset_id"] == captured["asset_id"], "Captured task asset identity differs")
            _require(len(asset["collision_parts"]) == captured["collision_parts"], "Captured collision piece count differs")
            inputs[asset["manifest_path"]] = asset["manifest_sha256"]
            for index, part in enumerate(asset["collision_parts"]):
                specs.append((f"{item['id']}/{index}", part, item["position_m"], item.get("quaternion_wxyz", [1, 0, 0, 0]),
                              meta["object_names"].index(item["id"]) if dynamic else None))
    for name, source, position, quaternion, object_index in specs:
        row = {"name": name, "position": _vector(position, "piece position"), "rotation": _rotation(quaternion),
               "offset": _vector(source.get("position_m", [0, 0, 0]), "piece offset"),
               "local_rotation": _rotation(np.asarray(source.get("quaternion_xyzw", [0, 0, 0, 1]))[[3, 0, 1, 2]]),
               "object_index": object_index}
        kind = source.get("geometry", "convex_mesh")
        if kind == "box":
            half = _vector(source["half_extents_m"], "box extents", positive=True)
            row.update(planes=np.column_stack((np.concatenate((np.eye(3), -np.eye(3))), -np.tile(half, 2))),
                       bounds=np.stack((-half, half)))
        elif kind in {"mesh", "convex_mesh"}:
            path = Path(source["mesh_path"])
            digest = frozen_audit.digest(path)
            _need(source.get("sha256"), "Collision mesh hash is absent")
            _require(digest == source["sha256"], "Collision mesh hash differs from task asset")
            inputs[str(path)] = digest
            if path.suffix.lower() == ".obj":
                vertices = load_obj_mesh(path)[0]
            else:
                import trimesh
                vertices = np.asarray(trimesh.load(path, force="mesh", process=False).vertices)
            vertices = vertices * _vector(source.get("scale", [1, 1, 1]), "mesh scale", positive=True)
            _require(np.isfinite(vertices).all(), "Nonfinite collision mesh")
            planes = ConvexHull(vertices).equations
            planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
            row.update(planes=np.unique(np.round(planes, 12), axis=0), bounds=np.stack((vertices.min(0), vertices.max(0))))
        elif kind == "cylinder":
            radius, half = float(source["radius_m"]), float(source["half_height_m"])
            _require(np.isfinite([radius, half]).all() and radius > 0 and half > 0, "Invalid cylinder dimensions")
            row.update(radius=radius, half_length=half)
        else:
            omitted.append({"piece": name, "reason": f"Unsupported collision geometry: {kind}"})
            continue
        pieces.append(row)
    return pieces, inputs, omitted


def _penetration(root, context):
    triangles = np.load(root / "triangles.npy", allow_pickle=False)
    pieces, inputs, omitted = _task_pieces(context)
    kernel, rows, unresolved = convex_triangle_audit, [], 0
    for frame, vertices in enumerate(context["states"]["particle_q"]):
        for part in pieces:
            position, rotation = part["position"], part["rotation"]
            if part["object_index"] is not None:
                pose = context["states"]["object_q"][frame, part["object_index"]]
                position, rotation = pose[:3], _rotation(pose[[6, 3, 4, 5]])
            local = ((vertices - position) @ rotation - part["offset"]) @ part["local_rotation"]
            if "planes" in part:
                result = kernel(local, triangles, part["planes"], part["bounds"], CLOTH_PENETRATION_M)
                status = "passed" if result["all_triangles_certified_clear_within_tolerance"] else "failed"
            else:
                result = summarize_surface(local, triangles, lambda p: cylinder_sdf(
                    p, np.zeros(3), np.array([0., 0., 1.]), part["radius"], part["half_length"]), CYLINDER_SUBDIVISIONS)
                status = ("failed" if result["minimum_sampled_surface_sdf_m"] < -CLOTH_PENETRATION_M else
                          "passed" if result["minimum_certified_surface_sdf_lower_bound_m"] >= -CLOTH_PENETRATION_M else "unchecked")
            unresolved += status == "unchecked"
            if status != "passed":
                rows.append({"frame": frame, "piece": part["name"], "status": status, **result})
    # A separate analytic-cylinder field belongs to an older demonstration. It
    # is not silently ignored or replaced with that demonstration's defaults.
    if context["scene"].get("cloth_support_cylinder"):
        omitted.append({"piece": "cloth_support_cylinder", "reason": "Legacy analytic support is not a captured native task asset"})
    return {"status": "failed" if any(r["status"] == "failed" for r in rows) else "unchecked" if omitted or unresolved else "passed",
            "frames_checked": len(context["states"]["particle_q"]), "collision_pieces_checked": len(pieces),
            "piece_names": [p["name"] for p in pieces], "tolerance_m": CLOTH_PENETRATION_M,
            "omitted_pieces": omitted, "unresolved_piece_frames": unresolved, "violations_or_unresolved": rows,
            "asset_input_sha256": inputs, "method": "Full-triangle clipping per convex task piece; analytic cylinders use conservative SDF bounds",
            "scope": "Table and authored task fixture/object collision pieces in recorded moving frames; robot envelopes are not included"}


def audit_native_episode(private_dir: str | Path) -> dict:
    """Audit worker.close() artifacts; return a private JSON-serializable report.

    Missing evidence yields ``unchecked`` and ``physics_valid=False``. No files
    are written, no scene state is changed, and no frames are skipped.
    """
    root = Path(private_dir).resolve()
    report = {"schema": "native_agent_physics_audit.v1", "checks": {},
              "physics_profile": physics_profile(), "physics_profile_sha256": physics_profile_sha256(),
              "scope": "Recorded native state/clock, cloth surface vs task solids, and observed rigid solver diagnostics",
              "threshold_source": "bench/isaacsim/audit.py:audit_episode,audit_contact_pair",
              "limitations": ["No independent full rigid/robot mesh collision certification.",
                              "No continuous clearance between native substeps; cooked envelopes are conservative, not a calibrated collision certificate.",
                              "No calibrated cloth material, cloth contact force, or manipulation success is established.",
                              "Rigid contacts cover observed solver-reported pairs only, not every pair in the scene."]}
    context = None

    def load():
        nonlocal context
        context = _context(root)
        return {"physical_records": len(context["observations"])}

    _check(report, "recording_readable", load)
    if context is None:
        return _finish(report)
    _check(report, "finite_states_and_alignment", lambda: _finite(context))
    _check(report, "native_clock", lambda: _clock(context))
    _check(report, "native_contact_stream", lambda: _contacts(context))
    _check(report, "reported_rigid_separations", lambda: _rigid_separations(context))
    has_cloth = context["scene"].get("cloth", {}).get("enabled") is not False
    safe = report["checks"]["finite_states_and_alignment"]["status"] == "passed"
    deferred = lambda: {"status": "unchecked", "reason": "Invalid or missing finite-state evidence prevents meaningful geometry audit"}
    _check(report, "cloth_topology_strain_self_intersection",
           (lambda: frozen_audit.cloth_shape_audit(root, context, MAX_EDGE_RATIO)) if safe else deferred,
           applicable=has_cloth)
    topology = report["checks"]["cloth_topology_strain_self_intersection"]
    geometry_ready = safe and "frames_checked" in topology
    _check(report, "cloth_task_solid_penetration", (lambda: _penetration(root, context)) if geometry_ready else deferred,
           applicable=has_cloth)
    _check(report, "cloth_every_substep", (lambda: _cloth_every_substep(root, context)) if safe else deferred,
           applicable=has_cloth)
    _check(report, "independent_full_rigid_mesh", lambda: {"status": "unchecked",
           "reason": "No separate full rigid/robot mesh geometry certificate is present"}, required=False)
    return _finish(report)


def _canonical(value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    return json.dumps(value, default=convert, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _strict_json(path):
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return load_json(Path(path), parse_constant=nonfinite)


def _exact_int(value, expected, name):
    _require(type(value) is int and value == expected, f"Invalid {name}")


def _cloth_rigid_sample(rows, robot_paths, mount):
    """Recheck saved getter aggregates; individual raw point forces are not saved."""
    from bench.simulation.isaacsim_backend.contact_precision import NORMAL_UNIT_LENGTH_ATOL
    _require(isinstance(rows, list), "Rigid sample is not a list")
    seen = set()
    for row in rows:
        a, b = row["actor0"], row["actor1"]
        _require(isinstance(a, str) and a.startswith("/") and isinstance(b, str) and b.startswith("/") and a != b,
                 "Invalid rigid actor paths")
        pair = tuple(sorted((a, b)))
        _require(pair not in seen, "Duplicated rigid pair in one native sample")
        seen.add(pair)
        count, nonzero = row["point_count"], row["nonzero_impulse_normal_count"]
        _require(type(count) is int and count > 0 and type(nonzero) is int and 0 <= nonzero <= count,
                 "Invalid rigid contact point counts")
        fields = ["min_separation_m", "impulse_magnitude_sum_ns", "raw_scalar_abs_impulse_sum_ns",
                  "normal_length_max_abs_error"]
        for field in fields:
            _require(type(row[field]) in (int, float) and np.isfinite(row[field]), f"Invalid rigid {field}")
        separation, magnitude, raw, normal_error = (row[field] for field in fields)
        raw_vector = row["impulse_vector_ns"]
        _require(isinstance(raw_vector, list) and len(raw_vector) == 3
                 and all(type(x) in (int, float) for x in raw_vector), "Rigid impulse vector must contain three numbers")
        vector = np.asarray(raw_vector, float)
        _require(vector.shape == (3,) and np.isfinite(vector).all(), "Invalid rigid impulse vector")
        _require(min(magnitude, raw, normal_error) >= 0 and normal_error <= NORMAL_UNIT_LENGTH_ATOL,
                 "Invalid rigid normal or negative impulse magnitude")
        tolerance = max(1e-12, raw * NORMAL_UNIT_LENGTH_ATOL)
        _require(abs(magnitude - raw) <= tolerance and np.linalg.norm(vector) <= magnitude + tolerance,
                 "Inconsistent rigid impulse aggregates")
        _require((nonzero > 0) == (raw > 0), "Rigid nonzero point count and impulse disagree")
        _require(separation >= -RIGID_CONTACT_DIAGNOSTIC_M, "Substep rigid separation exceeds original 2 mm gate")
        robot = a if a in robot_paths else b if b in robot_paths else None
        other = b if robot == a else a
        _require(not (robot is not None and robot != mount and other == "/World/Table" and magnitude > 1e-8),
                 "Substep moving robot/table solved contact")
    return len(rows)


def _cloth_every_substep(root, context):
    """Recompute all samples; a passed observer string alone is never evidence.

    Uses the same pinned cooked-geometry algorithm as the observer, independently
    rerun on saved arrays. Native getter aggregates can be checked for integrity,
    but are not a second measurement of unsaved per-point normals/forces. Stage
    invariance and enabled collision are recorded native configuration evidence,
    not a reconstruction of the running USD stage from this archive.
    """
    from scenes.cloth.geometry_guard import GeometryModel
    from bench.evaluation.cloth.cloth_robot_collision import reanalyse_export, state_digest
    from bench.simulation.isaacsim_backend.robot_asset import collision_configuration

    base = root / "cloth_physics"
    names = ["setup.json", "colliders.json", "task_geometry.json", "substeps/report.json",
             "substeps/trace.json", "substeps/states.npz", "substeps/rigid_contacts_per_substep.json"]
    paths = [base / name for name in names] + [root / name for name in (
        "states.npz", "triangles.npy", "metadata.json", "scene.json", "physics.jsonl", "worker_archive.json")]
    hashes = {str(p): frozen_audit.digest(p) for p in paths}
    setup, export, task, seal, trace, rigid = (_strict_json(base / name) for name in (
        "setup.json", "colliders.json", "task_geometry.json", "substeps/report.json",
        "substeps/trace.json", "substeps/rigid_contacts_per_substep.json"))
    archive = _strict_json(root / "worker_archive.json")
    ack, meta, scene = context["states"], context["metadata"], context["scene"]
    n_ack = len(ack["time_s"])
    _require(archive.get("reason") == "finish_request" and archive.get("source") == "captured_native_states"
             and archive.get("state_reconstruction") is False, "Worker did not normally seal captured state")
    _exact_int(archive["state_count"], n_ack, "worker state count")
    _require(setup.get("schema") == "cloth_agent_physics_observer_setup_v1", "Unknown cloth setup protocol")
    _require(setup.get("camera_setup_preserved_physics") is True, "Camera setup changed or did not verify physics")
    for key in ("controller_reference_consumed", "state_preload_or_reset_applied", "agent_feedback_contains_guard_evidence"):
        _require(setup.get(key) is False, f"Unexpected cloth setup flag: {key}")
    _exact_int(setup.get("physical_steps_added"), 0, "setup added steps")
    for name in ("setup_failure.json", "setup_seal_error.json", "substeps/unacknowledged_state.npz"):
        _require(not (base / name).exists(), f"Failed or unacknowledged cloth evidence: {name}")
    collision = meta["scene_info"]["robot_self_collision"]
    _require(_canonical(setup["robot_self_collision"]) == _canonical(collision), "Cloth setup self-collision metadata differs")
    expected_collision = collision_configuration(scene)
    _require(all(_canonical(collision.get(k)) == _canonical(v) for k, v in expected_collision.items()),
             "Captured self-collision configuration differs from loaded scene")
    _require(collision.get("enabled") is True and collision.get("cross_finger_exclusions") == [],
             "Robot self collision must be enabled without cross-finger exclusions")
    robot = meta["scene_info"]["robot_link_paths"]
    _require(robot and len(set(robot.values())) == len(robot), "Invalid robot body path inventory")
    expected_filtered = [[robot[a], robot[b]] for a, b in expected_collision["excluded_link_pairs"]]
    _require(_canonical(collision.get("authored_filtered_body_paths")) == _canonical(expected_filtered),
             "Actual authored structural collision exclusions differ")
    _require(seal.get("schema") == "cloth_agent_substep_physics_observer_v1" and seal.get("status") == "passed",
             "Cloth observer did not seal successfully")
    for key in ("all_observer_reads_preserved", "world_step_restored", "stage_layers_equal_start_end",
                "all_sampled_geometry_passed", "native_clock_matches_last_sample"):
        _require(seal.get(key) is True, f"Incomplete cloth observer evidence: {key}")
    for key in ("unacknowledged_state_saved", "diagnostic_trajectory_tracking_checked", "task_success_claimed"):
        _require(seal.get(key) is False, f"Unexpected cloth observer flag: {key}")
    _require(seal.get("observer_failure") is None and seal.get("trigger") is None, "Cloth observer reported a failure")
    states = frozen_audit.read_states(base / "substeps/states.npz")
    from scenes.cloth.contact_delivery import validate_delivery
    contact_delivery = validate_delivery(seal, trace, states, meta["physics_dt_s"])
    _require(set(states) == set(ack), "Substep/ACK state fields differ")
    samples = 40 * (n_ack - 1) + 1
    _require(isinstance(trace, list) and isinstance(rigid, list) and len(trace) == len(rigid) == samples,
             "Substep trace/contact count does not cover every ACK interval")
    for key, array in states.items():
        _require(array.shape == (samples, *ack[key].shape[1:]) and array.dtype == ack[key].dtype
                 and np.isfinite(array).all(), f"Invalid substep state: {key}")
    _require(np.issubdtype(states["native_physics_step_index"].dtype, np.integer), "Substep index must be integer")
    for kind in ("body", "object"):
        _require(np.all(abs(np.linalg.norm(states[kind + "_q"][..., 3:], axis=-1) - 1) < 1e-3),
                 f"Invalid substep {kind} quaternion")
    dt = .0005
    _require(type(meta["physics_dt_s"]) in (int, float) and abs(meta["physics_dt_s"] - dt) < 1e-10
             and abs(meta["control_dt_s"] - .02) < 1e-10, "Unsupported cloth native/control clock")
    _require(np.array_equal(states["native_physics_step_index"],
                            int(ack["native_physics_step_index"][0]) + np.arange(samples)), "Missing or reordered native substep")
    _require(np.all(abs(np.diff(states["native_world_time_s"]) - dt) <= 1e-10), "Invalid native substep intervals")
    # Env increments its logical clock AFTER world.step returns; callback time
    # therefore lags each integrated sample by exactly one logical dt.
    expected_callback = float(ack["time_s"][0]) + np.maximum(np.arange(samples) - 1, 0) * dt
    _require(np.allclose(states["time_s"], expected_callback, atol=1e-8, rtol=0), "Invalid callback logical clock")
    for key, array in states.items():
        if key != "time_s":
            _require(np.array_equal(array[::40], ack[key]), f"Raw substep/20ms ACK mapping differs: {key}")
    _require(np.allclose(ack["time_s"], float(ack["time_s"][0]) + np.arange(n_ack) * .02, atol=1e-8, rtol=0),
             "Invalid ACK logical clock")
    for i in range(1, n_ack):
        _require(np.all(states["joint_target_q"][(i-1)*40+1:i*40+1] == ack["joint_target_q"][i]),
                 "Applied target changed within one ACK interval")
    for key, expected in (("existing_steps_observed", samples-1), ("samples_including_initial", samples),
                          ("physical_steps_added", 0), ("native_substep_remainder_modulo40", 0)):
        _exact_int(seal.get(key), expected, key)
    coverage = seal["coverage"]
    _require(coverage.get("geometry_every_existing_native_step") is True, "Substep coverage not complete")
    _require(type(coverage.get("geometry_requested_interval_s")) in (int, float)
             and abs(coverage["geometry_requested_interval_s"]-dt) < 1e-10, "Geometry sampling interval differs")
    _exact_int(coverage.get("raw_rigid_contact_samples_completed"), samples-1, "rigid getter sample count")
    _exact_int(coverage.get("layer_interval_native_steps"), 40, "layer verification interval")
    initial = {k: v[0] for k, v in states.items()}
    initial_hash = state_digest(initial)
    _require(initial_hash == setup["initial_state_sha256"] == export["state_hash_before"] == export["state_hash_after"],
             "Geometry setup not bound to initial captured state")
    _require(export.get("schema") == "cloth_robot_cooked_colliders_v1" and export.get("status") == "complete"
             and export.get("visual_mesh_fallback") is False and export.get("issues") == [], "Invalid native cooked inventory")
    layers = export["layer_hashes_before"]
    _require(isinstance(layers, dict) and layers and layers == export["layer_hashes_after"]
             and all(isinstance(v, str) and len(v) == 64 for v in layers.values()), "Missing readonly native layer evidence")
    queries, colliders = export["queries"], export["colliders"]
    _require(len(queries) == len(robot) and {q["body"] for q in queries} == set(robot), "Native body query inventory differs")
    query_paths = []
    for query in queries:
        _require(query["path"] == robot[query["body"]] and query.get("finished") is True
                 and query.get("body_results") == ["PhysxPropertyQueryResult.VALID"], "Incomplete native body query")
        query_paths.extend(x["path"] for x in query["colliders"])
    collider_paths = [c["path"] for c in colliders]
    _require(len(query_paths) == len(set(query_paths)) and len(collider_paths) == len(set(collider_paths))
             and set(query_paths) == set(collider_paths), "Incomplete or duplicate cooked collider inventory")
    _require(all(c["body_path"] == robot[c["body_name"]] for c in colliders), "Cooked collider body mapping differs")
    checked_export = reanalyse_export(export)
    _require(checked_export["status"] == "complete" and checked_export["issues"] == [], "Cooked native geometry reanalysis failed")
    pieces, assets, omitted = _task_pieces(context)
    _require(not omitted and all("planes" in p for p in pieces), "Unsupported cloth task geometry")
    _require(_canonical(task) == _canonical({"pieces": pieces, "asset_input_sha256": assets}), "Saved task geometry differs from verified assets")
    hashes.update(assets)
    triangles = np.load(root / "triangles.npy", allow_pickle=False)
    model = GeometryModel(checked_export, meta, triangles, initial["particle_q"], pieces)
    rigid_pairs = 0
    for i in range(samples):
        state = {k: v[i] for k, v in states.items()}
        row = trace[i]
        for key, value in (("sample_index", i), ("observed_steps", i),
                           ("native_step_index", int(state["native_physics_step_index"]))):
            _exact_int(row[key], value, key)
        _require(type(row["native_time_s"]) in (int, float) and row["native_time_s"] == float(state["native_world_time_s"])
                 and type(row["public_time_s_at_callback"]) in (int, float)
                 and row["public_time_s_at_callback"] == float(state["time_s"]), "Trace clock differs from raw state")
        digest = state_digest(state)
        _require(row["state_sha256_before"] == row["state_sha256_after"] == digest, "Trace readonly state hashes differ")
        _require(row["readonly_preserved"] is True and row["layer_check"] is (i % 40 == 0)
                 and row["rigid_contact_getter_completed"] is (i > 0), "Incomplete per-sample readonly/getter evidence")
        _require(i > 0 or rigid[i] == [], "Initial sample cannot claim solved rigid contact measurement")
        rigid_pairs += _cloth_rigid_sample(rigid[i], set(robot.values()), robot["fr3_link0"])
        verdict = model.check_physics(state)
        _require(verdict.get("kind") is None, f"Recomputed substep {i} geometry failed: {verdict.get('kind')}")
        _require(_canonical(verdict) == _canonical(row["geometry"]), f"Stored/recomputed geometry differs at sample {i}")
        _require(state_digest(state) == digest, "Offline geometry checker mutated raw state")
    _require(all(frozen_audit.digest(path) == digest for path, digest in hashes.items()), "Inputs changed during cloth audit")
    return {"contact_delivery": contact_delivery,
            "samples_checked_including_initial": samples, "existing_native_steps_checked": samples-1,
            "complete_ACK_intervals": n_ack-1, "rigid_pair_reports_checked": rigid_pairs,
            "cooked_colliders_checked": len(colliders), "task_collision_pieces_checked": len(pieces),
            "input_sha256": hashes, "read_only": True, "native_steps_added": 0,
            "scope": "Every saved existing native substep; raw state/ACK/hash/configuration checks and offline rerun of the pinned cooked geometry algorithm.",
            "limitations": ["Same geometry algorithm, not an independent engine or continuous CCD certificate.",
                            "Rigid input is saved getter pair aggregates, not independently sampled per-point normals/forces.",
                            "Read-only stage/self-collision settings are recorded native evidence; the live USD stage is not reconstructed.",
                            "No task success, calibrated material/force accuracy or full robot self-mesh certificate."]}
