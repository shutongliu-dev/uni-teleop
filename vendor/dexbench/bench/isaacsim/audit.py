"""Independent, read-only audits of recorded native Isaac Sim episodes/suites.

Run with the analysis/Newton Python environment; no Isaac application is started.
Missing measurements are unverified, never successful. Geometry checks inspect
recorded frames and the actual task collision pieces, not the passive cylinder
demo's hard-coded dimensions. These checks do not require a successful full task
policy and cannot certify contacts between recorded timestamps.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from bench.paths import load_json


PROJECT = Path(__file__).resolve().parents[2]
AUDITOR_SOURCE = Path(__file__).read_bytes()
AUDITOR_SHA256 = hashlib.sha256(AUDITOR_SOURCE).hexdigest()
TASK_IDS = ("01_ordered_kitting", "02_loose_gear_assembly", "03_lined_tray_loading", "04_cover_and_secure")
STATE_KEYS = ("time_s", "joint_q", "joint_qd", "joint_target_q", "body_q", "body_qd",
              "object_q", "object_qd", "particle_q", "particle_qd", "native_world_time_s", "native_physics_step_index")
REQUIRED_PINS = ("states.npz", "observations.jsonl", "actions.json", "scores.json", "scene.json", "task.json",
                 "metadata.json", "source_hashes.json", "reset_override.json", "reset_evidence.json", "joint_response.json",
                 "triangles.npy", "reset_replay_states.npz", "result.json", "release_reference.json", "condition_inspection.json",
                 "evaluator/sequence_evaluator.py", "evaluator/geometry.py", "environment_hashes.json")


class MissingEvidence(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    path = Path(path)
    if not path.is_file():
        raise MissingEvidence(f"Missing file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise MissingEvidence(f"Missing file: {path}")
    return load_json(path)


def read_states(path):
    if not Path(path).is_file():
        raise MissingEvidence(f"Missing state archive: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def local_path(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, f"Artifact path escapes episode: {name}")
    result = root / relative
    require(result.resolve().is_relative_to(root.resolve()), f"Artifact symlink escapes episode: {name}")
    return result


def add_check(report, name, function, *, required=True, applicable=True):
    if not applicable:
        report["checks"][name] = {"status": "unverified", "applicable": False, "required": False,
                                  "reason": "Not applicable to this recorded scene/phase"}
        return
    try:
        evidence = function() or {}
        status = evidence.pop("status", "pass")
        require(status in ("pass", "fail", "unverified"), "Invalid audit status")
        report["checks"][name] = {"status": status, "applicable": True, "required": required, **evidence}
    except (MissingEvidence, ImportError) as error:
        report["checks"][name] = {"status": "unverified", "applicable": True, "required": required,
                                  "reason": str(error), "error_type": type(error).__name__}
    except Exception as error:
        report["checks"][name] = {"status": "fail", "applicable": True, "required": required,
                                  "reason": str(error), "error_type": type(error).__name__}


def finish_report(report):
    active = [row for row in report["checks"].values() if row["applicable"] and row["required"]]
    report["required_check_counts"] = dict(Counter(row["status"] for row in active))
    report["status"] = "fail" if any(r["status"] == "fail" for r in active) else (
        "unverified" if not active or any(r["status"] == "unverified" for r in active) else "pass")
    report["audit_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def verify_hashes(root):
    pins = read_json(root / "artifact_hashes.json")
    require(isinstance(pins, dict) and bool(pins), "Artifact hash map is empty or malformed")
    mismatches = []
    for name, expected in pins.items():
        if digest(local_path(root, name)) != expected:
            mismatches.append(name)
    require(not mismatches, f"Artifact SHA256 mismatch: {mismatches}")
    missing = [name for name in REQUIRED_PINS if name not in pins]
    if (root / "recording.json").exists() and "recording.json" not in pins:
        missing.append("recording.json")
    if (root / "diagnostic.json").exists():
        missing.extend(name for name in ("diagnostic.json", "contact_trace.json", "contact_result.json", "submitted_actions.json") if name not in pins)
    if missing:
        raise MissingEvidence(f"Existing manifest does not pin required audit inputs: {missing}")
    return {"pinned_artifacts_verified": len(pins), "artifact_manifest_sha256": digest(root / "artifact_hashes.json")}


def _source_layout(root):
    """Select one recorded source root and backend layout from declared pins."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate source pin: {key}")
            result[key] = value
        return result

    path = root / "source_hashes.json"
    if not path.is_file():
        raise MissingEvidence(f"Missing file: {path}")
    pins = load_json(path, object_pairs_hook=unique_keys)
    require(isinstance(pins, dict) and bool(pins), "Source pin map is empty")
    for original, expected in pins.items():
        source = Path(original)
        require(source.is_absolute() and str(source) == original and ".." not in source.parts,
                f"Source pin path is not canonical and absolute: {original}")
        require(isinstance(expected, str) and len(expected) == 64
                and all(character in "0123456789abcdef" for character in expected),
                f"Malformed source SHA256: {original}")
    launchers = [Path(name) for name in pins
                 if Path(name).parts[-3:] == ("bench", "isaacsim", "run.py")]
    if not launchers:
        raise MissingEvidence("Required implementation not captured: bench/isaacsim/run.py")
    require(len(launchers) == 1, "Ambiguous recorded rollout source root")
    project = launchers[0].parents[2]
    sources = {}
    for original, expected in pins.items():
        source = Path(original)
        require(source.is_relative_to(project), f"Source pin is outside recorded project: {original}")
        relative = source.relative_to(project).as_posix()
        require(relative not in sources, f"Duplicate captured source: {relative}")
        sources[relative] = expected
    layouts = [prefix for prefix in ("simulation", "bench/simulation")
               if any(name.startswith(prefix + "/") for name in sources)]
    if not layouts:
        raise MissingEvidence("Required simulator implementation not captured")
    require(len(layouts) == 1, "Mixed simulator source layouts in source_hashes.json")
    return project, layouts[0], sources


def _captured_source(root, relative, sources):
    if relative not in sources:
        raise MissingEvidence(f"Required implementation not captured: {relative}")
    snapshot = local_path(root, Path("source_snapshot") / relative)
    require(digest(snapshot) == sources[relative], f"Captured source differs from launch pin: {snapshot}")
    return snapshot


def verify_sources(root):
    project, backend, snapshots = _source_layout(root)
    changed_worktree = []
    for relative in snapshots:
        _captured_source(root, relative, snapshots)
        original = project / relative
        if not original.is_file() or digest(original) != snapshots[relative]:
            changed_worktree.append(str(original))
    for required in ("bench/isaacsim/run.py", backend + "/isaacsim_backend/env.py",
                     backend + "/isaacsim_backend/scene.py"):
        if required not in snapshots:
            raise MissingEvidence(f"Required implementation not captured: {required}")
    result = read_json(root / "result.json")
    require(result.get("source_files_changed_during_run") == [], "Sources changed during rollout, or end-of-run source check is missing")
    return {"source_snapshots_verified": len(snapshots), "current_worktree_differences": changed_worktree,
            "recorded_project_root": str(project), "simulator_source_layout": backend,
            "interpretation": "Captured implementation is checked; later worktree changes are listed separately"}


def verify_alignment(root, context):
    ref = read_json(root / "release_reference.json")
    release = Path(ref["path"])
    require(digest(release / "release_manifest.json") == ref["manifest_sha256"], "Release reference SHA256 mismatch")
    manifest = read_json(release / "release_manifest.json")
    ids = [entry["id"] for entry in manifest["tasks"]]
    require(tuple(ids) == TASK_IDS, f"Release does not contain the four original tasks in order: {ids}")
    require(digest(release / "source_release_manifest.json") == manifest["source_release_sha256"], "Source release copy is not pinned")
    original = read_json(release / "source_release_manifest.json")
    inventory = original["inputs_and_artifacts"]
    for name, row in manifest["files"].items():
        require(digest(local_path(release, name)) == row["sha256"], f"Release file mismatch: {name}")
    for entry in manifest["tasks"]:
        source = next(t for t in original["tasks"] if t["id"] == entry["id"])
        for kind in ("task", "scene"):
            require(entry[kind + "_sha256"] == inventory[source[kind]]["sha256"], f"Original task definition changed: {entry['id']} {kind}")
    for name, row in read_json(release / "asset_file_hashes.json").items():
        require(digest(name) == row["sha256"], f"Frozen shared task asset changed: {name}")
    profile_path = release / manifest["physics_profile"]
    require(digest(profile_path) == manifest["physics_profile_sha256"], "Physics profile hash mismatch")
    result, scene, task = context["result"], context["scene"], context["task"]
    entry = next(t for t in manifest["tasks"] if t["id"] == result["task_id"])
    require(ref["task_entry"] == entry, "Episode task entry differs from pinned release")
    require(digest(root / "task.json") == entry["task_sha256"], "Episode changed frozen task or evaluator predicates")
    expected_scene = read_json(release / entry["scene"])
    expected_scene["isaacsim"] = read_json(profile_path)["isaacsim_options"]
    expected_initial = {}
    phase = result["phase"]
    if phase == "qualification_probe":
        probe = task.get("qualification_probe", {})
        if "cloth_origin_m" in probe:
            expected_scene["cloth"]["origin_m"] = probe["cloth_origin_m"]
        names = probe.get("objects_at_goal", True)
        names = list(task["goal_objects"]) if names is True else names
        if names:
            expected_initial["objects"] = {n: {"position_m": task["goal_objects"][n], "quaternion_wxyz": [1, 0, 0, 0]} for n in names}
    elif phase == "contact_probe":
        diagnostic = read_json(root / "diagnostic.json")
        require(diagnostic["task_id"] == task["id"] == TASK_IDS[0], "Push diagnostic is declared only on unchanged task 01")
        expected_initial = diagnostic["reset_override"]
        require(set(expected_initial) == {"joint_positions_rad"}, "Contact diagnostic must not reset objects or cloth")
        require(set(expected_initial["joint_positions_rad"]) == set(context["metadata"]["joint_names"]), "Diagnostic reset must specify exactly 29 joints")
        require(digest(root / "submitted_actions.json") == diagnostic["planned_actions_sha256"], "Contact trajectory differs from declared plan")
        require(diagnostic.get("scene_and_objects_unchanged") is True and diagnostic.get("object_reset_override") is False and diagnostic.get("cloth_reset_override") is False, "Contact diagnostic crosses its declared initialization boundary")
    elif phase != "nominal":
        raise MissingEvidence(f"Diagnostic phase {phase!r} needs explicit separately audited overrides; it is not a nominal task episode")
    require(scene == expected_scene, "Scene differs from frozen scene + declared PhysX profile + allowed qualification reset")
    require(read_json(root / "reset_override.json") == expected_initial, "Reset override differs from phase definition")
    require(context["metadata"]["scene_spec"] == scene, "Loaded engine scene differs from recorded scene")
    for filename in ("sequence_evaluator.py", "geometry.py"):
        require(digest(root / "evaluator" / filename) == manifest["files"][f"evaluator/{filename}"]["sha256"], "Episode evaluator differs from frozen evaluator")
    return {"release": str(release), "task_id": task["id"], "phase": phase,
            "four_tasks": ids, "reset_override": expected_initial, "material_equivalence_claimed": False}


def load_context(root):
    state = read_states(root / "states.npz")
    missing = sorted(set(STATE_KEYS) - set(state))
    if missing:
        raise MissingEvidence(f"Missing physical state fields: {missing}")
    observations_path = root / "observations.jsonl"
    if not observations_path.exists():
        raise MissingEvidence(str(observations_path))
    observations = [json.loads(line) for line in observations_path.read_text().splitlines() if line.strip()]
    return {"states": state, "observations": observations,
            **{name: read_json(root / f"{name}.json") for name in ("scene", "task", "result", "metadata", "actions")}}


def check_finite(context):
    state, observations, metadata, actions = (context[x] for x in ("states", "observations", "metadata", "actions"))
    count = len(state["time_s"])
    require(context["result"].get("status") == "physical_rollout_completed", "Episode did not complete with a fixed implementation")
    require(count >= 2 and len(observations) == count and len(actions) == count - 1, "State/observation/action counts disagree or rollout is empty")
    require(len(metadata["joint_names"]) == len(set(metadata["joint_names"])) == 29, "Joint inventory must have 29 distinct names")
    for name, array in state.items():
        require(array.ndim >= 1 and len(array) == count and np.isfinite(array).all(), f"State field is nonfinite or has inconsistent frame count: {name}")
    for name in ("joint_q", "joint_qd", "joint_target_q"):
        require(state[name].shape == (count, 29), f"Expected [T,29] for {name}")
    for name in ("object_q", "body_q"):
        require(state[name].ndim == 3 and state[name].shape[-1] == 7, f"Invalid pose shape: {name}")
        norms = np.linalg.norm(state[name][..., 3:7], axis=-1)
        require(np.max(np.abs(norms - 1)) < 1e-3, f"Non-unit orientation: {name}")
    require(state["particle_q"].ndim == 3 and state["particle_q"].shape[-1] == 3, "Invalid cloth array shape")
    require(state["particle_q"].shape == state["particle_qd"].shape, "Cloth positions/velocities do not align")
    require(len(metadata["object_names"]) == state["object_q"].shape[1], "Object metadata count mismatch")
    require(len(metadata["body_names"]) == state["body_q"].shape[1], "Body metadata count mismatch")
    require(all(o.get("metrics", {}).get("state_finite") is True for o in observations), "Observation reports a nonfinite/missing state flag")
    for index, observation in enumerate(observations):
        require(np.allclose([observation["joint_positions_rad"][n] for n in metadata["joint_names"]], state["joint_q"][index], atol=1e-7, rtol=0), "Observed joints differ from the captured physical arrays")
        for object_index, name in enumerate(metadata["object_names"]):
            pose = observation["objects"][name]
            recorded = state["object_q"][index, object_index]
            require(np.allclose(pose["position_m"], recorded[:3], atol=1e-7, rtol=0), f"Observed object position differs: {name}")
            require(np.allclose(pose["quaternion_wxyz"], recorded[[6, 3, 4, 5]], atol=1e-7, rtol=0), f"Observed object orientation differs: {name}")
        if state["particle_q"].shape[1]:
            require(np.allclose(observation["cloth"]["vertices_m"], state["particle_q"][index], atol=1e-7, rtol=0), "Observed cloth differs from physical state")
    return {"frames": count, "joint_count": 29, "object_count": state["object_q"].shape[1], "cloth_vertex_count": state["particle_q"].shape[1]}


def check_clock(context):
    state, meta = context["states"], context["metadata"]
    t = state["time_s"].reshape(-1)
    native = state["native_world_time_s"].reshape(-1)
    steps = state["native_physics_step_index"].reshape(-1)
    dt, control = float(meta["physics_dt_s"]), float(meta["control_dt_s"])
    require(dt > 0 and abs(control - .02) < 1e-10, "Invalid physics/control timestep")
    substeps = round(control / dt)
    require(substeps > 0 and abs(substeps * dt - control) < 1e-10, "Control interval does not divide into physics steps")
    require(abs(t[0]) <= 1e-8 and np.allclose(np.diff(t), control, atol=1e-8, rtol=0), "Recorded clock does not advance exactly once per action")
    require(np.allclose(np.diff(native), control, atol=1e-8, rtol=0), "Native world clock did not advance with physical actions")
    require(np.all(np.diff(steps) == substeps), "Native integration step increments are missing or incorrect")
    require(meta.get("native_isaacsim_app") is True and meta.get("standalone_newton_runtime_used") is False, "Engine metadata does not identify native Isaac execution")
    require(np.allclose([o["time_s"] for o in context["observations"]], t, atol=1e-8, rtol=0), "Observation clock differs from physical state clock")
    require(abs(context["result"]["simulation_duration_s"] - t[-1]) < 1e-8, "Result duration differs from actual state clock")
    return {"physical_duration_s": float(t[-1] - t[0]), "native_world_duration_s": float(native[-1] - native[0]),
            "native_steps": int(steps[-1] - steps[0]), "substeps_per_action": substeps, "physics_dt_s": dt,
            "native_initial_time_s": float(native[0]), "native_initial_step": int(steps[0])}


def check_actions(root, context):
    names, state = context["metadata"]["joint_names"], context["states"]
    limits = context["metadata"]["action_spec"]["joint_limits_rad"]
    targets = []
    for index, action in enumerate(context["actions"]):
        require(isinstance(action, dict) and set(action) == {"joint_positions_rad", "duration_s"}, f"Action {index} contains non-joint state controls")
        require(action["duration_s"] == .02 and not isinstance(action["duration_s"], bool), "Wrong action duration")
        q = action["joint_positions_rad"]
        require(set(q) == set(names), f"Action {index} must contain exactly all 29 joints")
        for n in names:
            require(isinstance(q[n], (int, float)) and not isinstance(q[n], bool) and np.isfinite(q[n]) and limits[n][0] <= q[n] <= limits[n][1], f"Invalid/out-of-limit target: {n}")
        targets.append([q[n] for n in names])
    require(np.allclose(targets, state["joint_target_q"][1:], atol=1e-6, rtol=0), "Commands differ from physical drive targets")
    _, backend, sources = _source_layout(root)
    source = _captured_source(root, backend + "/isaacsim_backend/env.py", sources)
    methods = [node for node in ast.walk(ast.parse(source.read_text())) if isinstance(node, ast.FunctionDef) and node.name == "step"]
    require(len(methods) == 1, "Cannot uniquely inspect the captured step method")
    setters = sorted({node.func.attr for node in ast.walk(methods[0]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("set_")})
    require(set(setters) == {"set_dof_position_targets"}, f"Unexpected direct state setters in captured step: {setters}")
    return {"actions_checked": len(targets), "joints_per_action": len(names), "captured_step_setters": setters,
            "limitation": "Checks recorded commands/targets and captured source; no independent write-hook instrumentation is available"}


def check_response(root, context):
    state = context["states"]
    q, targets = state["joint_q"].astype(float), state["joint_target_q"].astype(float)
    require(len(q) >= 3, "Too few states for joint response measurement")
    command = np.max(np.abs(targets - targets[0]), axis=0)
    actual = np.max(np.abs(q - q[0]), axis=0)
    rows = {}
    for index, name in enumerate(context["metadata"]["joint_names"]):
        x, y = targets[:, index] - targets[0, index], q[:, index] - q[0, index]
        peak = int(np.argmax(np.abs(x)))
        signed_peak_response = float(y[peak] * np.sign(x[peak]))
        gain = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 1e-12 else None
        correlations = []
        for lag in range(min(5, max(0, len(x) // 20)) + 1):
            xx, yy = (x[:-lag], y[lag:]) if lag else (x, y)
            if np.std(xx) > 1e-8 and np.std(yy) > 1e-8:
                correlations.append(float(np.corrcoef(xx, yy)[0, 1]))
        correlation = max(correlations) if correlations else None
        rows[name] = {"command_excursion_rad": float(command[index]), "actual_excursion_rad": float(actual[index]),
                      "signed_response_at_command_peak_rad": signed_peak_response, "tracking_gain": gain,
                      "maximum_command_response_correlation_lag_0_to_100ms": correlation,
                      "responded": bool(command[index] > 1e-5 and actual[index] > 1e-4 and gain is not None and gain > .02 and correlation is not None and correlation > .25)}
    commanded, responding = int(np.sum(command > 1e-5)), sum(r["responded"] for r in rows.values())
    saved = read_json(root / "joint_response.json")
    require(saved["joint_count"] == 29 and saved["commanded_joints"] == commanded, "Saved joint-response command counts disagree")
    for name, row in rows.items():
        require(abs(saved["per_joint"][name]["actual_excursion_rad"] - row["actual_excursion_rad"]) < 1e-5, f"Saved response disagrees for {name}")
    return {"status": "unverified" if commanded < 29 else "pass" if responding == 29 else "fail",
            "commanded_joints": commanded, "responding_joints": responding, "per_joint": rows,
            "thresholds": {"minimum_command_rad": 1e-5, "minimum_response_rad": 1e-4, "minimum_positive_tracking_gain": .02,
                           "minimum_demeaned_tracking_correlation": .25, "maximum_lag_s": .10},
            "scope": "Recorded 29-axis drive response; not successful manipulation or system identification"}


def check_reset(root, context):
    replay = read_states(root / "reset_replay_states.npz")
    evidence = read_json(root / "reset_evidence.json")
    original = context["states"]
    differences, replay_max = {}, {}
    for name in STATE_KEYS:
        if name not in replay:
            raise MissingEvidence(f"Reset capture missing {name}")
        value = replay[name]
        require(value.shape[1:] == original[name].shape[1:] and len(value) >= 2, f"Reset replay shape mismatch: {name}")
        require(np.isfinite(value).all(), f"Nonfinite reset/replay state: {name}")
        differences[name] = float(np.max(np.abs(value[0] - original[name][0]))) if value[0].size else 0.
        replay_max[name] = float(np.max(np.abs(value - original[name][:len(value)]))) if value.size else 0.
    require(all(v <= 1e-6 for v in differences.values()), f"Reset failed to restore observed physical arrays: {differences}")
    require(evidence["reset_time_s"] == 0 and abs(float(replay["time_s"][0])) <= 1e-8, "Reset physical time was not zero")
    require(len(replay["time_s"]) - 1 == evidence["replayed_action_count"] == min(5, len(context["actions"])), "Reset replay action count mismatch")
    require(evidence["same_reset_override"] == read_json(root / "reset_override.json"), "Reset replay used a different initialization")
    require(np.allclose(np.diff(replay["time_s"]), .02, atol=1e-8, rtol=0), "Reset replay did not advance physical time")
    require(np.allclose(np.diff(replay["native_world_time_s"]), .02, atol=1e-8, rtol=0), "Reset replay native world time did not advance")
    require(np.all(np.diff(replay["native_physics_step_index"]) == round(.02 / context["metadata"]["physics_dt_s"])), "Reset replay native step counter did not advance")
    require(evidence["initial_restored_within_1e_6"] is True and all(abs(evidence["reset_max_absolute_error"][k] - v) <= 1e-8 for k, v in differences.items()), "Saved reset result disagrees with physical captures")
    return {"reset_max_absolute_errors": differences, "replay_max_absolute_errors": replay_max,
            "replayed_actions": len(replay["time_s"]) - 1,
            "scope": "Observable state reset and short finite replay; complete solver checkpoint determinism is not claimed"}


# Leaf link-name prefixes of hand bodies, as in the evaluator's hand rule
# (scenes.manipulation.evaluator SimpleTaskEvaluator._hand_actor). Until 2026-10-15
# only '/right_' anywhere in the path counted as hand, so left SHARPA and Allegro /
# LEAP / Shadow hand contacts were reported as arm contacts in the audit records.
HAND_LINK_PREFIXES = ("right_", "left_", "allegro_", "leap_", "rh_")


def pair_labels(pair):
    labels = [(str(pair.get(f"actor{i}", "")), str(pair.get(f"collider{i}", ""))) for i in (0, 1)]
    robot = [any("/Robot/" in path for path in label) for label in labels]
    hand = [r and any(path.rsplit("/", 1)[-1].startswith(HAND_LINK_PREFIXES) for path in label)
            for r, label in zip(robot, labels)]
    objects = [any("/Objects/" in path for path in label) for label in labels]
    return labels, robot, hand, objects


def check_contacts(context):
    groups = Counter()
    impulses = Counter()
    examples = {}
    minimum_separation = None
    reports = 0
    sources = set()
    components = set()
    for observation in context["observations"]:
        contacts = observation.get("contacts")
        if not contacts or contacts.get("source") not in {
                "PhysX solver contact report callbacks",
                "PhysX stage-bound rigid contact tensor; normal impulses"}:
            raise MissingEvidence("Native PhysX contact provenance is absent")
        source = contacts["source"]
        sources.add(source)
        tensor = source == "PhysX stage-bound rigid contact tensor; normal impulses"
        provenance = context.get("metadata", {}).get("contact_evidence", {})
        if tensor:
            require(provenance.get("source") == "stage-bound PhysX RigidContactView.get_contact_data(dt=1)", "Tensor contact impulse units/API provenance missing")
            require(provenance.get("impulse_components") == "normal_only", "Tensor normal-only impulse scope is missing")
            require(provenance.get("capacity_points", 0) > 0, "Tensor contact capacity is missing")
            sensors, filters = provenance.get("sensor_paths", []), provenance.get("filter_paths", [])
            require(len(sensors) > 0 and len(filters) == len(sensors), "Tensor sensor/filter provenance is incomplete")
            allowed_pairs = {(sensor, target) for sensor, targets in zip(sensors, filters) for target in targets}
        require(contacts.get("errors") == [], f"Contact API reported errors: {contacts.get('errors')}")
        point_sum, impulse_sum = 0, 0.
        seen = set()
        if "rigid" not in contacts:
            raise MissingEvidence("Raw per-pair solver reports were not recorded")
        for pair in contacts["rigid"]:
            count, impulse = int(pair["point_count"]), float(pair["impulse_magnitude_sum_ns"])
            require(count > 0 and np.isfinite(impulse) and impulse >= 0, "Malformed contact point/impulse report")
            require(observation["time_s"] - .02 - 1e-8 <= pair["time_s"] <= observation["time_s"] + 1e-8, "Contact report timestamp is outside the recorded control interval")
            if tensor:
                require(pair.get("impulse_components") == "normal_only", "Tensor pair lacks normal-only scope")
                require((pair["actor0"], pair["actor1"]) in allowed_pairs, "Reported pair outside recorded tensor sensors/filters")
                require(pair.get("collider0") is None and pair.get("collider1") is None and pair.get("collider_identity_available") is False, "Tensor pair falsely claims collider identity")
                key = (pair["time_s"], *sorted((pair["actor0"], pair["actor1"])))
                require(key not in seen, "Tensor pair double counted within the same native substep")
                seen.add(key)
                vector = np.asarray(pair.get("impulse_vector_ns"), dtype=float)
                require(vector.shape == (3,) and np.isfinite(vector).all() and np.linalg.norm(vector) <= impulse + 1e-8, "Normal impulse vector exceeds sum of point magnitudes")
            components.add("normal_only" if tensor else "callback_reported")
            point_sum += count
            impulse_sum += impulse
            reports += 1
            _, robot, hand, objects = pair_labels(pair)
            if any(robot) and any(objects):
                group = "hand_object" if any(hand[i] and objects[1 - i] for i in (0, 1)) else "arm_object"
            elif all(robot):
                group = "robot_self"
            elif any(robot):
                group = "robot_support"
            elif all(objects):
                group = "object_object"
            else:
                group = "object_support_or_other"
            if impulse > 1e-8:
                groups[group] += 1
                impulses[group] += impulse
                examples.setdefault(group, pair)
            separation = pair.get("min_separation_m")
            if separation is not None:
                require(np.isfinite(separation), "Nonfinite contact separation")
                minimum_separation = float(separation) if minimum_separation is None else min(minimum_separation, float(separation))
        require(point_sum == contacts.get("rigid_contact_point_count"), "Contact point aggregate disagrees with raw reports")
        require(np.isclose(impulse_sum, contacts.get("rigid_impulse_magnitude_sum_ns"), atol=1e-10, rtol=1e-7), "Impulse aggregate disagrees with raw reports")
    return {"status": "pass" if sum(groups.values()) else "unverified", "raw_reports": reports,
            "sources": sorted(sources), "impulse_components": sorted(components),
            "nonzero_impulse_reports_by_pair_type": dict(groups), "reported_impulse_sum_ns_by_pair_type": dict(impulses),
            "examples": examples, "minimum_reported_separation_m": minimum_separation,
            "impulse_threshold_ns": 1e-8, "candidate_contacts_counted_as_solved": False,
            "limits": ["Pair records may repeat over native substeps; report counts are not unique contact counts.",
                       "Rigid impulses do not establish deformable contact forces.",
                       "Tensor reports contain normal impulses only; tangential/friction forces and collider identity are unavailable.",
                       "Absolute force/impulse calibration is unverified: a separate stationary-support check observed approximately twice m*g*T for gear/washer on the pickup nest, versus approximately once for spool/table. Raw reported values are preserved without rescaling."]}


def check_render(root, context):
    recording = read_json(root / "recording.json")
    frames = recording.get("frames", [])
    if not frames:
        raise MissingEvidence("No captured RTX frame map")
    times = context["states"]["time_s"]
    fps = recording["fps"]
    require(fps > 0 and 50 % fps == 0, "Invalid recording fps")
    expected_indices = list(range(0, len(times) - 1, 50 // fps))
    require([r["state_index"] for r in frames] == expected_indices, "Frame map does not cover the declared constant-speed recording")
    from PIL import Image
    for index, frame in enumerate(frames):
        require(frame["frame"] == index and abs(frame["physical_time_s"] - times[frame["state_index"]]) <= 1e-8, "Frame time/index disagrees with physics state")
        for path_key, hash_key in (("raw_rgb", "raw_sha256"), ("overview_rgb", "overview_sha256"), ("video_frame", "video_frame_sha256")):
            path = Path(frame[path_key])
            require(path.resolve().is_relative_to(root.resolve()), f"Frame leaves captured episode: {path}")
            require(digest(path) == frame[hash_key], f"Frame hash mismatch: {path}")
        delta = frame.get("render_state_max_delta", {})
        if not set(STATE_KEYS) <= set(delta):
            raise MissingEvidence("Render isolation did not capture all physical arrays/native clocks")
        require(all(v is not None and np.isfinite(v) and 0 <= v <= 1e-7 for v in delta.values()), "Rendering changed physical state/native time")
        if index in (0, len(frames) - 1):
            with Image.open(frame["raw_rgb"]) as image:
                rgb = np.asarray(image)
                require(rgb.ndim == 3 and rgb.shape[2] in (3, 4) and np.std(rgb[..., :3]) > 1, "Raw captured image is blank or malformed")
    video = Path(recording["video"])
    require(digest(video) == recording["video_sha256"], "Encoded demo hash mismatch")
    require(recording.get("playback_speed") == 1 and recording.get("physics_interpolation") is False, "Recording playback/interpolation assumptions changed")
    completed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,r_frame_rate:format=duration", "-of", "json", str(video)], capture_output=True, text=True)
    require(completed.returncode == 0, f"Independent ffprobe failed: {completed.stderr[-1000:]}")
    info = json.loads(completed.stdout)
    require(int(info["streams"][0]["nb_frames"]) == len(frames), "Video frame count differs from pinned captured frames")
    require(abs(float(info["format"]["duration"]) - len(frames) / fps) <= .02, "Video duration differs from normal-speed frame map")
    return {"captured_frames": len(frames), "frames_all_sha256_verified": True, "video_sha256": recording["video_sha256"],
            "independent_ffprobe": info, "render_state_tolerance": 1e-7,
            "limitation": "Physical before/after arrays are represented by captured deltas; PNG hashes establish artifact identity, not an external camera measurement"}


def decoded_frame_hashes(path):
    """Compare decoded pixels; remuxing may change packet headers without changing a frame."""
    from fractions import Fraction
    completed = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an",
        "-vsync", "0", "-f", "framehash", "-hash", "sha256", "-"], capture_output=True, text=True)
    require(completed.returncode == 0, f"Cannot independently decode video frames: {path}: {completed.stderr[-500:]}")
    timebase, frames = None, []
    for line in completed.stdout.splitlines():
        if line.startswith("#tb 0:"):
            timebase = Fraction(line.split(":", 1)[1].strip())
        elif line and not line.startswith("#"):
            fields = [part.strip() for part in line.split(",")]
            require(timebase is not None and len(fields) == 6 and len(fields[5]) == 64, f"Malformed decoded frame hash: {path}")
            frames.append((fields[5], float(int(fields[3]) * timebase)))
    require(bool(frames) and all(duration > 0 for _, duration in frames), f"Video has missing frame hashes/durations: {path}")
    return frames


def cloth_shape_audit(root, context, max_edge_ratio):
    from bench.evaluation.cloth.audit_cloth_surface import topology_report
    from bench.evaluation.cloth.audit_cloth_self_intersection import self_intersections
    state, spec = context["states"], context["scene"]["cloth"]
    vertices = state["particle_q"].astype(float)
    triangles_path = root / "triangles.npy"
    if not triangles_path.is_file():
        raise MissingEvidence("Missing cloth triangles")
    triangles = np.load(triangles_path, allow_pickle=False)
    nx, ny = spec["cells_xy"]
    expected = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            expected.extend(((a, a + 1, a + nx + 1), (a + 1, a + nx + 2, a + nx + 1)))
    require(vertices.shape[1:] == ((nx + 1) * (ny + 1), 3), "Cloth nodal count differs from frozen task topology")
    require(np.issubdtype(triangles.dtype, np.integer) and np.array_equal(triangles, expected), "Cloth face indices differ from frozen task topology")
    require(all(np.array_equal(o["cloth"]["triangles"], triangles) for o in context["observations"]), "Recorded cloth topology changed across observations")
    topology, edges = topology_report(triangles, vertices.shape[1])
    require(topology["connected_component_sizes"] == [vertices.shape[1]] and topology["nonmanifold_edges"] == 0, "Cloth mesh has disconnected or nonmanifold pieces")
    sx, sy = spec["size_m"]
    rest = np.array([[i * sx / nx, j * sy / ny, 0.] for j in range(ny + 1) for i in range(nx + 1)])
    lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    rest_tri = rest[triangles]
    areas = np.linalg.norm(np.cross(rest_tri[:, 1] - rest_tri[:, 0], rest_tri[:, 2] - rest_tri[:, 0]), axis=1)
    rows = []
    for index, q in enumerate(vertices):
        tri = q[triangles]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        ratios = np.linalg.norm(q[edges[:, 1]] - q[edges[:, 0]], axis=1) / lengths
        hits, candidates = self_intersections(q, triangles)
        rows.append({"frame": index, "time_s": float(state["time_s"][index]), "edge_ratio_max": float(ratios.max()),
                     "edge_ratio_min": float(ratios.min()), "area_ratio_min": float((area / areas).min()),
                     "degenerate_faces": int(np.sum(area < 1e-12)), "self_intersection_pairs": hits,
                     "broadphase_candidates": candidates})
    maximum = max(r["edge_ratio_max"] for r in rows)
    intersections = sum(len(r["self_intersection_pairs"]) for r in rows)
    degenerate = sum(r["degenerate_faces"] for r in rows)
    return {"status": "pass" if maximum <= max_edge_ratio and not intersections and not degenerate else "fail",
            "topology": topology, "frames_checked": len(rows), "maximum_edge_ratio": maximum,
            "edge_ratio_limit": max_edge_ratio, "limit_interpretation": "Declared numerical stability threshold; not a calibrated textile material limit",
            "self_intersection_pairs_total": intersections, "degenerate_faces_total": degenerate,
            "maximum_vertex_displacement_from_reset_m": float(np.linalg.norm(vertices - vertices[0], axis=-1).max()),
            "rows": rows, "limitations": ["Only recorded frames; no continuous-between-frame guarantee.",
                "Authored connectivity and nodal count are checked; no separate solver-internal topology stream exists.",
                "Self-contact thickness is not certified by triangle intersection checks."]}


def cloth_dynamic_response(context):
    vertices = context["states"]["particle_q"].astype(float)
    velocities = context["states"]["particle_qd"].astype(float)
    authored = context["metadata"].get("scene_info", {}).get("cloth")
    if not authored or "solver" not in authored or "fixed_vertices" not in authored:
        raise MissingEvidence("Native deformable body metadata is absent")
    require(authored["fixed_vertices"] == 0, "Cloth contains undeclared fixed attachments")
    movement = float(np.linalg.norm(vertices - vertices[0], axis=-1).max())
    speed = float(np.linalg.norm(velocities, axis=-1).max())
    reference = vertices[0] - vertices[0].mean(0)
    residuals = []
    for q in vertices:
        centered = q - q.mean(0)
        u, _, vt = np.linalg.svd(reference.T @ centered)
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        residuals.append(float(np.sqrt(np.mean(np.sum((centered - reference @ (u @ vt)) ** 2, axis=1)))))
    return {"status": "pass" if movement > 1e-5 and speed > 1e-5 else "unverified",
            "native_solver": authored["solver"], "fixed_vertices": authored["fixed_vertices"],
            "maximum_nodal_displacement_m": movement, "maximum_nodal_speed_m_s": speed,
            "maximum_nonrigid_rms_residual_m": max(residuals),
            "method": "Per-frame nodal change and best rigid fit residual; pure translation/rotation does not count as cloth deformation",
            "limitation": "Passive cloth dynamics are distinguished from active robot-cloth manipulation"}


def collision_envelopes(context):
    """Build individual recorded-task/URDF pieces, preserving all hollow bores."""
    from scipy.spatial import ConvexHull
    from bench.geometry import quaternion_matrix, rpy_matrix
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset, load_obj_mesh
    import trimesh
    scene, metadata = context["scene"], context["metadata"]
    pieces, omissions, inputs = [], [], {}

    def piece(name, part, pose, *, body_index=None, object_index=None):
        geometry = part.get("geometry", "convex_mesh")
        row = {"name": name, "position": np.asarray(pose[0], float), "rotation": np.asarray(pose[1], float),
               "part_position": np.asarray(part.get("position_m", [0, 0, 0]), float),
               "part_rotation": quaternion_matrix(np.asarray(part.get("quaternion_xyzw", [0, 0, 0, 1]))[[3, 0, 1, 2]]),
               "body_index": body_index, "object_index": object_index}
        if geometry == "box":
            half = np.asarray(part["half_extents_m"], float)
            row["planes"] = np.column_stack([np.concatenate([np.eye(3), -np.eye(3)]), -np.tile(half, 2)])
            row["bounds"] = np.stack([-half, half])
        elif geometry in ("mesh", "convex_mesh"):
            path = Path(part["mesh_path"])
            actual = digest(path)
            if part.get("sha256"):
                require(actual == part["sha256"], f"Collision mesh input changed: {path}")
            inputs[str(path)] = actual
            v = load_obj_mesh(path)[0] if path.suffix.lower() == ".obj" else np.asarray(trimesh.load(path, force="mesh", process=False).vertices)
            v = v * np.asarray(part.get("scale", [1, 1, 1]))
            planes = ConvexHull(v).equations
            planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
            row["planes"] = np.unique(np.round(planes, 12), axis=0)
            row["bounds"] = np.stack([v.min(0), v.max(0)])
        else:
            omissions.append({"piece": name, "reason": f"Analytic geometry {geometry!r} has not been independently certified"})
            return
        pieces.append(row)

    table = scene["table"]
    piece("table", {"geometry": "box", "half_extents_m": table["half_extents_m"]}, (table["position_m"], np.eye(3)))
    for dynamic, specifications in ((False, scene.get("fixtures", [])), (True, scene.get("objects", []))):
        for spec in specifications:
            asset = load_task_asset(spec["asset_id"], spec.get("asset_root"), verify_hashes=True)
            inputs[asset["manifest_path"]] = asset["manifest_sha256"]
            for index, part in enumerate(asset["collision_parts"]):
                piece(f"{spec['id']}/{index}", part, (spec["position_m"], quaternion_matrix(spec.get("quaternion_wxyz", [1, 0, 0, 0]))),
                      object_index=metadata["object_names"].index(spec["id"]) if dynamic else None)
    robot = metadata.get("robot_asset", {})
    urdf_path = robot.get("urdf_path")
    if not urdf_path:
        omissions.append({"piece": "robot", "reason": "No captured assembled URDF provenance"})
    else:
        require(digest(urdf_path) == robot.get("urdf_sha256"), "Assembled robot URDF hash differs from recording")
        inputs[urdf_path] = robot["urdf_sha256"]
        for link in ET.parse(urdf_path).getroot().findall("link"):
            for index, collision in enumerate(link.findall("collision")):
                name = link.get("name")
                if name not in metadata["body_names"]:
                    omissions.append({"piece": name, "reason": "Collision link missing from recorded body transforms"})
                    continue
                origin = collision.find("origin")
                position = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                mesh = collision.find("geometry/mesh")
                box = collision.find("geometry/box")
                if mesh is not None:
                    path = Path(mesh.get("filename"))
                    path = path if path.is_absolute() else Path(urdf_path).parent / path
                    part = {"geometry": "convex_mesh", "mesh_path": str(path), "scale": np.fromstring(mesh.get("scale", "1 1 1"), sep=" "),
                            "sha256": robot.get("mesh_inputs", {}).get(str(path))}
                elif box is not None:
                    part = {"geometry": "box", "half_extents_m": np.fromstring(box.get("size"), sep=" ") / 2}
                else:
                    omissions.append({"piece": name, "reason": "Unsupported URDF collision primitive"})
                    continue
                previous = len(pieces)
                piece(f"robot/{name}/{index}", part, ([0, 0, 0], np.eye(3)), body_index=metadata["body_names"].index(name))
                if len(pieces) > previous:
                    pieces[-1]["part_position"], pieces[-1]["part_rotation"] = position, rpy_matrix(rpy)
    return pieces, omissions, inputs


def cloth_penetration_audit(root, context, tolerance):
    from bench.geometry import quaternion_matrix
    from bench.convex_geometry import convex_triangle_audit
    triangles = np.load(root / "triangles.npy", allow_pickle=False)
    pieces, omissions, inputs = collision_envelopes(context)
    require(bool(pieces), "No collision geometry audited")
    rows = []
    states = context["states"]
    for index, vertices in enumerate(states["particle_q"]):
        collisions = []
        for part in pieces:
            if part["body_index"] is not None:
                pose = states["body_q"][index, part["body_index"]]
                position, rotation = pose[:3], quaternion_matrix(pose[[6, 3, 4, 5]])
            elif part["object_index"] is not None:
                pose = states["object_q"][index, part["object_index"]]
                position, rotation = pose[:3], quaternion_matrix(pose[[6, 3, 4, 5]])
            else:
                position, rotation = part["position"], part["rotation"]
            local = ((vertices - position) @ rotation - part["part_position"]) @ part["part_rotation"]
            result = convex_triangle_audit(local, triangles, part["planes"], part["bounds"], tolerance)
            if not result["all_triangles_certified_clear_within_tolerance"]:
                collisions.append({"piece": part["name"], **result})
        rows.append({"frame": index, "time_s": float(states["time_s"][index]), "penetrations": collisions})
    bad = sum(bool(row["penetrations"]) for row in rows)
    return {"status": "fail" if bad else "unverified" if omissions else "pass", "frames_checked": len(rows),
            "collision_pieces_checked": len(pieces), "frames_with_penetration": bad, "omitted_pieces": omissions,
            "maximum_confirmed_penetration_m": max((p["maximum_confirmed_piece_penetration_m"] for r in rows for p in r["penetrations"]), default=0.),
            "tolerance_m": tolerance, "mesh_input_sha256": inputs, "rows": rows,
            "method": "Full recorded triangles clipped against each individual task convex piece and URDF collision envelope in its recorded moving body frame",
            "limitations": ["Only recorded frames, not every native substep.",
                "Source URDF convex envelopes may conservatively differ from cooked PhysX hulls; a robot-envelope hit needs inspection.",
                "Clearance is geometric evidence, not measured cloth contact force.",
                "Decorative noncolliding workcell meshes are excluded."]}


def check_boundary(root, context):
    result, task = context["result"], context["task"]
    phase = result["phase"]
    require(result.get("controller_task_success_claimed") is False, "Basic-platform recording incorrectly claims task-policy success")
    if phase == "qualification_probe":
        require(result.get("qualification_excluded_from_nominal_success") is True, "Qualification episode was not excluded from nominal success")
    elif phase == "nominal":
        require(read_json(root / "reset_override.json") == {}, "Nominal episode has diagnostic goal initialization")
    elif phase == "contact_probe":
        require(result.get("diagnostic_excluded_from_nominal_success") is True, "Robot-reset contact diagnostic was not excluded from nominal success")
        require(set(read_json(root / "reset_override.json")) == {"joint_positions_rad"}, "Contact diagnostic changes more than robot reset")
    else:
        raise MissingEvidence("Unknown episode phase must be reviewed separately")
    require(result.get("material_calibrated") is False, "Material calibration is not evidenced by this recording")
    # Import the captured, hash-verified evaluator rather than the live package.
    ref = read_json(root / "release_reference.json")
    require(digest(Path(ref["path"]) / "release_manifest.json") == ref["manifest_sha256"], "Cannot execute an evaluator without a verified release reference")
    manifest = read_json(Path(ref["path"]) / "release_manifest.json")
    for filename in ("sequence_evaluator.py", "geometry.py"):
        require(digest(root / "evaluator" / filename) == manifest["files"][f"evaluator/{filename}"]["sha256"], "Refusing to execute modified captured evaluator")
    import importlib.util
    import sys
    import types
    name = "_independent_isaac_audit_" + digest(root / "evaluator/sequence_evaluator.py")[:16]
    package = types.ModuleType(name)
    package.__path__ = [str(root / "evaluator")]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".sequence_evaluator", root / "evaluator/sequence_evaluator.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    evaluator = module.SequenceEvaluator(task)
    scores = [evaluator.update(o) for o in context["observations"]]
    require(scores == read_json(root / "scores.json"), "Independent frozen-evaluator replay differs from recorded scores")
    require(scores[-1] == result["evaluator_final"], "Result final score differs from evaluator replay")
    return {"phase": phase, "ordered_evaluator_recomputed": True, "evaluator_final": scores[-1],
            "controller_policy_success_required_for_basic_platform": False,
            "qualification_can_count_as_nominal_success": False}


def check_robot_asset(context):
    metadata = context["metadata"]
    robot = metadata.get("robot_asset", {})
    for path_key, hash_key in (("urdf_path", "urdf_sha256"),):
        if not robot.get(path_key) or not robot.get(hash_key):
            raise MissingEvidence("Robot model is missing acquisition-time URDF provenance")
        require(digest(robot[path_key]) == robot[hash_key], "Loaded robot URDF differs from captured hash")
    total = 1
    for key in ("sources", "mesh_inputs", "license_sha256"):
        if not robot.get(key):
            raise MissingEvidence(f"Robot {key} is not pinned")
        for path, expected in robot[key].items():
            require(digest(path) == expected, f"Robot asset input changed: {path}")
            total += 1
    if not metadata.get("robot_usd_sha256") or not metadata.get("robot_usd_dependency_sha256"):
        raise MissingEvidence("Native imported USD and its dependencies lack acquisition-time hashes")
    require(digest(metadata["robot_usd"]) == metadata["robot_usd_sha256"], "Native imported USD hash changed")
    for path, expected in metadata["robot_usd_dependency_sha256"].items():
        require(digest(path) == expected, f"Native USD dependency changed: {path}")
        total += 1
    import_record = metadata.get("robot_usd_import_manifest")
    if not import_record or "snapshot" not in import_record:
        raise MissingEvidence("Imported USD manifest was not captured")
    require(import_record["snapshot"]["usd_sha256"] == metadata["robot_usd_sha256"], "USD importer manifest points to another model")
    return {"robot_asset_files_verified": total, "robot_usd_sha256": metadata["robot_usd_sha256"],
            "mount_calibrated": metadata.get("mount_calibrated"), "license_files_verified": len(robot["license_sha256"])}


def check_runtime_provenance(root, context):
    environment = read_json(root / "environment_hashes.json")
    metadata = context["metadata"]
    if environment.get("missing_files") or not environment.get("files"):
        raise MissingEvidence("Compatibility wrapper/library provenance was not captured completely")
    for path, expected in environment["files"].items():
        require(digest(path) == expected, f"Compatibility runtime input changed: {path}")
    experience = metadata.get("experience")
    if not experience or "driver_verification_disabled" not in metadata:
        raise MissingEvidence("Kit experience/driver-override provenance is missing")
    require(digest(experience["path"]) == experience["sha256"], "Kit experience changed from captured launch")
    disabled = metadata["driver_verification_disabled"]
    require(disabled == (environment.get("ISAAC_ALLOW_UNVALIDATED_DRIVER") == "1"), "Driver bypass metadata disagrees with invocation environment")
    return {"experience": experience, "compatibility_files_verified": len(environment["files"]),
            "driver_verification_disabled": disabled, "driver_support_status": metadata.get("driver_support_status"),
            "vendor_supported_runtime_claimed": False,
            "interpretation": "A driver-verification bypass is a declared compatibility experiment, even if empirical physics/rendering checks pass"}


def check_native_masses(context):
    metadata = context["metadata"]
    robot = metadata.get("native_robot_link_masses_kg")
    objects = metadata.get("native_object_masses_kg")
    if not robot or not objects:
        raise MissingEvidence("Native PhysX link/object mass readback is missing")
    require(set(objects) == set(metadata["object_names"]), "Native mass readback omits task bodies")
    require(all(np.isfinite(v) and v >= 0 for v in robot.values()) and sum(robot.values()) > 0, "Native robot masses are nonfinite, negative, or all zero")
    object_errors = {}
    for name, mass in objects.items():
        expected = metadata["scene_info"]["objects"][name]["mass_kg"]
        require(np.isfinite(mass) and mass > 0 and np.isclose(mass, expected, atol=1e-7, rtol=1e-5), f"Native object mass differs from authored task asset: {name}")
        object_errors[name] = float(mass - expected)
    return {"native_robot_total_mass_kg": float(sum(robot.values())), "native_robot_link_count": len(robot),
            "native_object_masses_kg": objects, "object_mass_errors_kg": object_errors,
            "limitation": "Robot readback verifies finite physical masses; this check does not calibrate the simulation-only hand mount"}


def robot_contact_capability(context, groups):
    state = context["states"]
    displacement = {name: float(np.linalg.norm(state["object_q"][:, i, :3] - state["object_q"][0, i, :3], axis=1).max()) for i, name in enumerate(context["metadata"]["object_names"])}
    return {"status": "pass" if groups.get("hand_object", 0) else "unverified",
            "nonzero_robot_object_impulse_reports": groups.get("hand_object", 0), "object_displacements_m": displacement,
            "reason": "SHARPA hand/object solver impulse is required; arm contact, gravity motion or object/table support alone does not establish dexterous manipulation"}


def robot_self_collision_evidence(context, contact):
    authored = context["metadata"].get("scene_info", {}).get("robot_self_collision", {})
    events = contact.get("nonzero_impulse_reports_by_pair_type", {}).get("robot_self", 0)
    return {"status": "fail" if authored.get("enabled") is False else "pass" if events else "unverified",
            "authored_profile": authored, "actual_nonzero_self_impulse_reports": events,
            "effective_native_setting_read_back": False,
            "reason": "An authored USD/profile boolean alone does not establish the effective native collision setting; absent self contacts do not prove enabled or disabled behavior",
            "scope": "Actual nonzero self-contact events establish observed behavior for those pairs only; the full collision mask remains separately scoped"}


def audit_episode(root, *, max_edge_ratio=1.25, penetration_tolerance=1e-4, require_video=True):
    root = Path(root).resolve()
    report = {"schema": "independent_isaac_episode_audit.v1", "episode": str(root), "checks": {},
              "auditor_sha256": AUDITOR_SHA256, "simulation_started_by_auditor": False,
              "acceptance_scope": "Basic native platform and observable capabilities; no full-policy task-success requirement"}
    add_check(report, "captured_artifact_hashes", lambda: verify_hashes(root))
    add_check(report, "captured_source_hashes", lambda: verify_sources(root))
    try:
        context = load_context(root)
    except Exception as error:
        add_check(report, "readable_physical_recording", lambda: (_ for _ in ()).throw(error))
        return finish_report(report)
    report.update(task_id=context["result"].get("task_id"), phase=context["result"].get("phase"))
    nominal = report["phase"] == "nominal"
    has_cloth = bool(context["scene"].get("cloth", {}).get("enabled"))
    add_check(report, "four_task_input_alignment", lambda: verify_alignment(root, context))
    add_check(report, "robot_usd_and_asset_provenance", lambda: check_robot_asset(context))
    add_check(report, "runtime_and_driver_override_provenance", lambda: check_runtime_provenance(root, context))
    add_check(report, "native_physical_mass_readback", lambda: check_native_masses(context))
    add_check(report, "finite_states_and_shapes", lambda: check_finite(context))
    add_check(report, "native_physics_clock", lambda: check_clock(context))
    add_check(report, "joint_only_actions", lambda: check_actions(root, context))
    add_check(report, "all_29_joint_response", lambda: check_response(root, context), required=nominal)
    add_check(report, "reset_and_finite_replay", lambda: check_reset(root, context))
    add_check(report, "native_rigid_contact_impulses", lambda: check_contacts(context))
    contact = report["checks"]["native_rigid_contact_impulses"]
    groups = contact.get("nonzero_impulse_reports_by_pair_type", {})
    add_check(report, "robot_self_collision_effective_evidence", lambda: robot_self_collision_evidence(context, contact), required=False)
    add_check(report, "robot_object_contact_capability", lambda: robot_contact_capability(context, groups), required=False)
    add_check(report, "captured_rtx_render_and_video", lambda: check_render(root, context), required=require_video or (root / "recording.json").is_file())
    add_check(report, "nominal_qualification_and_evaluator_boundary", lambda: check_boundary(root, context))
    safe_shape = report["checks"]["finite_states_and_shapes"]["status"] == "pass"
    if safe_shape:
        add_check(report, "cloth_native_dynamic_response", lambda: cloth_dynamic_response(context), applicable=has_cloth)
        add_check(report, "cloth_topology_strain_self_intersection", lambda: cloth_shape_audit(root, context, max_edge_ratio), applicable=has_cloth)
        add_check(report, "cloth_full_triangle_penetration", lambda: cloth_penetration_audit(root, context, penetration_tolerance), applicable=has_cloth)
    else:
        for name in ("cloth_native_dynamic_response", "cloth_topology_strain_self_intersection", "cloth_full_triangle_penetration"):
            add_check(report, name, lambda: {"status": "unverified", "reason": "Invalid state arrays prevent meaningful cloth geometry audit"}, applicable=has_cloth)
    add_check(report, "cloth_solver_contact_forces", lambda: {"status": "unverified", "reason": "Native rigid callbacks do not measure deformable contact; no calibrated cloth contact-force stream was captured"}, required=False, applicable=has_cloth)
    report["limitations"] = ["No continuous collision certificate between recorded frames.",
        "Hash manifests establish consistency with the recorded inputs; they are not an external signed attestation.",
        "Successful drive response and passive support do not by themselves validate active cloth or rigid manipulation."]
    return finish_report(report)


def contact_motion(context):
    names, states = context["metadata"]["object_names"], context["states"]
    require("spool" in names, "Contact diagnostic must contain the original spool")
    index = names.index("spool")
    positions, velocities = states["object_q"][:, index, :3], states["object_qd"][:, index, :3]
    time = states["time_s"]
    points, impulses, hand_support_depth, self_depth = [], [], 0., 0.
    for observation in context["observations"]:
        count, impulse = 0, 0.
        for pair in observation["contacts"]["rigid"]:
            labels, robot, hand, _ = pair_labels(pair)
            spool = [any(path == "/World/Objects/spool" or path.startswith("/World/Objects/spool/") for path in label) for label in labels]
            if any(hand[i] and spool[1 - i] for i in (0, 1)):
                count += pair["point_count"]
                impulse += pair["impulse_magnitude_sum_ns"]
            if all(robot):
                self_depth = max(self_depth, max(0., -float(pair.get("min_separation_m") or 0.)))
            for i in (0, 1):
                if hand[i] and any("/Table" in a or "/Ground" in a or "/Fixtures/" in a for a in labels[1 - i]):
                    hand_support_depth = max(hand_support_depth, max(0., -float(pair.get("min_separation_m") or 0.)))
        points.append(count); impulses.append(impulse)
    solved = (np.asarray(points) > 0) & (np.asarray(impulses) > 1e-8)
    contacts = np.flatnonzero(solved)
    xy = np.linalg.norm(positions[:, :2] - positions[0, :2], axis=1)
    tail = time >= time[-1] - 1.
    return {"final_xy_displacement_m": float(xy[-1]), "maximum_xy_displacement_m": float(xy.max()),
            "initial_position_m": positions[0].tolist(), "final_position_m": positions[-1].tolist(),
            "first_hand_contact_s": float(time[contacts[0]]) if len(contacts) else None,
            "precontact_max_xy_drift_m": float(xy[:contacts[0]].max()) if len(contacts) and contacts[0] > 0 else None,
            "hand_contact_frames": int(solved.sum()), "total_hand_spool_impulse_ns": float(sum(impulses)),
            "final_1s_has_hand_contacts": bool(np.any(np.asarray(points)[tail] > 0)),
            "final_1s_max_speed_m_s": float(np.linalg.norm(velocities[tail], axis=1).max()),
            "max_reported_hand_support_penetration_m": hand_support_depth,
            "max_reported_robot_self_penetration_m": self_depth,
            "duration_s": float(time[-1]), "points_by_frame": points, "impulses_by_frame_ns": impulses}


def audit_contact_pair(push, hover, *, max_edge_ratio=1.25, penetration_tolerance=1e-4):
    report = {"schema": "independent_isaac_contact_pair_audit.v1", "checks": {},
              "push": str(push), "hover": str(hover), "full_policy_success_required": False}
    contexts = {}
    for mode, root in (("push", Path(push)), ("hover", Path(hover))):
        episode = audit_episode(root, max_edge_ratio=max_edge_ratio, penetration_tolerance=penetration_tolerance, require_video=False)
        report[mode + "_episode"] = episode
        report["checks"][mode + "_physical_recording"] = {"status": episode["status"], "required": True, "applicable": True}
        try:
            contexts[mode] = load_context(root)
        except Exception as error:
            add_check(report, mode + "_recording_readable", lambda: (_ for _ in ()).throw(error))
    if len(contexts) != 2:
        return finish_report(report)

    def matched():
        first, second = contexts["push"], contexts["hover"]
        require(first["scene"] == second["scene"] and first["task"] == second["task"], "Push/hover use different task geometry or physics")
        require(first["metadata"]["physics_dt_s"] == second["metadata"]["physics_dt_s"], "Push/hover physics steps differ")
        require(len(first["actions"]) == len(second["actions"]), "Push/hover durations differ")
        require(np.allclose(first["states"]["object_q"][0], second["states"]["object_q"][0], atol=1e-7, rtol=0), "Push/hover object initial states differ")
        definitions = [read_json(Path(root) / "diagnostic.json") for root in (push, hover)]
        for mode, d in zip(("push", "hover"), definitions):
            require(d["mode"] == mode and d["phase"] == "contact_probe", "Diagnostic mode/phase mismatch")
            require(d.get("state_writes_after_reset") is False and d.get("attachments") is False, "Diagnostic state write/attachment boundary is missing")
        for key in ("distance_m", "tip_z_m", "integral_gain_s_inv", "kinematics_sources"):
            require(definitions[0][key] == definitions[1][key], f"Push/hover mismatch beyond declared clearance: {key}")
        return {"object_initial_states_matched": True, "declared_hover_clearance_m": definitions[1]["hover_z_offset_m"]}

    add_check(report, "matched_contact_control", matched)
    motions = {mode: contact_motion(context) for mode, context in contexts.items()}
    report["motion"] = motions
    first, second = motions["push"], motions["hover"]
    conditions = {
        "hand_spool_nonzero_solver_impulse": first["hand_contact_frames"] > 0,
        "push_final_displacement_at_least_5cm": first["final_xy_displacement_m"] >= .05,
        "push_precontact_drift_below_1cm": first["precontact_max_xy_drift_m"] is not None and first["precontact_max_xy_drift_m"] < .01,
        "hover_has_no_hand_spool_contacts": second["hand_contact_frames"] == 0,
        "hover_maximum_drift_below_1cm": second["maximum_xy_displacement_m"] < .01,
        "push_minus_hover_final_displacement_at_least_4cm": first["final_xy_displacement_m"] - second["final_xy_displacement_m"] >= .04,
        "push_final_second_released": not first["final_1s_has_hand_contacts"],
        "push_final_second_speed_below_1cm_s": first["final_1s_max_speed_m_s"] < .01,
        "push_hand_support_reported_penetration_below_2mm": first["max_reported_hand_support_penetration_m"] <= .002,
        "push_reported_robot_self_penetration_below_2mm": first["max_reported_robot_self_penetration_m"] <= .002,
        "push_object_remains_at_table_height": abs(first["final_position_m"][2] - first["initial_position_m"][2]) < .01,
    }
    add_check(report, "contact_push_response", lambda: {"status": "pass" if all(conditions.values()) else "fail", "conditions": conditions,
        "scope": "Joint-only hand push capability with matched hover control; does not qualify grasping or sequential task success"})
    report["independent_rigid_contact_geometry"] = {"status": "unverified", "reason": "Rigid penetration here comes from solver reports; a separate full rigid geometry audit is not yet present"}
    return finish_report(report)


def audit_self_collision_probe(path, episode_roots):
    """Verify a separate four-setting behavioral control, not an invented getter."""
    path = Path(path)
    proof = read_json(path)
    provenance_path = path.with_name("self_collision_schema_probe_provenance.json")
    provenance = read_json(provenance_path)
    require(provenance.get("result_sha256") == digest(path) and provenance.get("process_exit_code") == 0, "Self-collision result is not pinned to a successful process")
    runtime = provenance.get("runtime_provenance_postrun")
    if not runtime:
        raise MissingEvidence("Self-collision probe has no pinned runtime/SDK provenance")
    for name, expected in runtime["sdk_files_sha256"].items():
        require(digest(name) == expected, f"Self-probe runtime/SDK hash changed: {name}")
    kit_reference = runtime["kit_version_reference"]
    require(digest(kit_reference["source"]) == kit_reference["source_sha256"], "Contemporaneous Kit version reference changed")
    require(digest(runtime["probe_log"]) == runtime["probe_log_sha256"], "Self-probe process log changed")
    episodes = [(Path(root), read_json(Path(root) / "metadata.json"), _source_layout(Path(root)))
                for root in episode_roots]
    require(bool(episodes), "Self-collision probe has no episode to bind to")
    checked = {}
    for name, expected in provenance["files_sha256"].items():
        original = Path(name)
        candidates = [original]
        for root, _, (project, _, sources) in episodes:
            if original.is_relative_to(project):
                relative = original.relative_to(project).as_posix()
                if sources.get(relative) == expected:
                    candidates.append(local_path(root, Path("source_snapshot") / relative))
        require(any(candidate.is_file() and digest(candidate) == expected for candidate in candidates), f"Self-collision probe source hash unavailable/mismatched: {name}")
        checked[name] = expected
    require(digest(provenance["robot_usd"]) == provenance["robot_usd_sha256"], "Self-probe robot USD changed")
    for name, expected in provenance["robot_dependencies_sha256"].items():
        require(digest(name) == expected, f"Self-probe robot USD dependency changed: {name}")
    expected_flags = {"legacy_physx": True, "shared_newton": False}
    require(proof.get("original_authored_values") == expected_flags, "Self-probe starting flags differ from the evaluated backend configuration")
    scene_paths = [name for name in checked if Path(name).name == "01_ordered_kitting_scene.json"]
    require(len(scene_paths) == 1, "Self-probe authored scene is not pinned")
    probe_scene = read_json(scene_paths[0])
    require(probe_scene.get("isaacsim", {}).get("robot_self_collisions") is True, "Self-probe source scene does not author self collision enabled")
    expected_exclusions = [["right_hand_C_MC", "right_thumb_MC"]] if probe_scene["isaacsim"].get("filter_thumb_cmc_structure") else []
    q = proof["deliberately_closed_hand_reset_q"]
    limits = provenance["joint_limits"]
    require(len(q) == 29 and set(q) == set(limits), "Probe did not apply the same full 29-joint reset")
    require(all(np.isfinite(value) and limits[name]["lower"] <= value <= limits[name]["upper"] for name, value in q.items()), "Self-probe reset exceeds joint limits")
    for root, metadata, (project, backend, sources) in episodes:
        require(metadata["robot_usd_sha256"] == provenance["robot_usd_sha256"], "Self-probe and episode use different robot USD")
        require(metadata["robot_usd_dependency_sha256"] == provenance["robot_dependencies_sha256"], "Self-probe and episode use different robot USD dependencies")
        require(metadata["scene_info"]["robot_self_collision"]["enabled"] is True, "Episode did not author self-collision enabled")
        require(metadata["scene_info"]["robot_self_collision"].get("excluded_link_pairs") == expected_exclusions and metadata["scene_info"]["robot_self_collision"].get("cross_finger_exclusions") == [], "Episode collision exclusions differ from independent self-probe")
        require(metadata["experience"]["sha256"] == checked[metadata["experience"]["path"]], "Self-probe uses a different Kit experience")
        for name in ("env.py", "scene.py", "robot_asset.py"):
            relative = backend + "/isaacsim_backend/" + name
            if relative not in sources:
                raise MissingEvidence(f"Required implementation not captured: {relative}")
            key = str(project / relative)
            require(key in checked and sources[relative] == checked[key],
                    f"Self-probe and episode use different backend {name}")
        require(metadata["physics_dt_s"] == .0005 and metadata["native_isaacsim_app"] is True, "Probe native-step configuration differs")
        version = metadata["isaacsim_version"]
        require(runtime["isaac_version_file_value"] == f"{version[0]}-{version[1]}+{version[-1]}", "Self-probe and episode Isaac versions differ")
        require(metadata["kit_version"] == kit_reference["value"], "Self-probe and episode Kit versions differ")
    variants, summaries = {}, []
    for mode in proof["modes"]:
        key = (mode["legacy_physx"], mode["shared_newton"])
        require(key not in variants, "Repeated self-collision probe variant")
        require(mode["authored_after_reset"] == {"legacy_physx": key[0], "shared_newton": key[1]}, "Probe reset overwrote the requested flags")
        count, impulse, maximum_penetration, cross_finger = 0, 0., 0., 0
        for pair, values in mode["self_pairs"].items():
            names = pair.split(" | ")
            require(len(names) == 2 and names[0] != names[1] and all("/Robot/" in n for n in names), "Self-probe pair is not between two robot links")
            require(values["point_count"] > 0 and np.isfinite(values["normal_impulse_ns"]) and values["normal_impulse_ns"] >= 0, "Invalid self-probe contact values")
            count += values["point_count"]
            impulse += values["normal_impulse_ns"]
            maximum_penetration = max(maximum_penetration, values["max_penetration_m"])
            fingers = []
            for name in names:
                leaf = name.rsplit("/", 1)[-1]
                fingers.append(next((f for f in ("thumb", "index", "middle", "ring", "pinky") if leaf.startswith("right_" + f)), None))
            if all(fingers) and fingers[0] != fingers[1] and values["normal_impulse_ns"] > 1e-8:
                cross_finger += 1
        require(count == mode["total_self_points"] and np.isclose(impulse, mode["total_self_normal_impulse_ns"], atol=1e-8), "Self-probe pair aggregates disagree")
        require(np.isfinite(mode["max_joint_speed_rad_s"]), "Self-probe state was not finite")
        variants[key] = (count, impulse, cross_finger)
        summaries.append({"legacy_physx": key[0], "shared_newton": key[1], "reported_contact_points": count,
                          "reported_normal_impulse_ns": impulse, "positive_cross_finger_pairs": cross_finger,
                          "maximum_deliberately_interfering_reset_penetration_m": maximum_penetration})
    require(set(variants) == {(a, b) for a in (False, True) for b in (False, True)}, "Self-probe lacks a full two-by-two schema comparison")
    for shared in (False, True):
        require(variants[(True, shared)][0] > 0 and variants[(True, shared)][1] > 1e-8 and variants[(True, shared)][2] > 0, "Enabled self-collision lacks actual cross-finger solver response")
        require(variants[(False, shared)][:2] == (0, 0.), "Disabled control still reports self contacts")
    require(variants[(True, False)] == variants[(True, True)], "Shared Newton field changes observed enabled PhysX behavior")
    return {"status": "pass", "probe_result": str(path), "probe_result_sha256": digest(path),
            "probe_provenance": str(provenance_path), "probe_provenance_sha256": digest(provenance_path),
            "matched_episodes": len(episodes), "verified_source_files": len(checked), "variants": summaries,
            "verified_runtime_sdk_files": len(runtime["sdk_files_sha256"]), "runtime_provenance_scope": runtime["collection_scope"],
            "effective_native_setting_read_back": False,
            "conclusion": "For this pinned backend and robot, legacy PhysX=true produces real cross-finger self-contact response while both legacy=false controls do not; the shared Newton flag does not override this behavior",
            "runtime_versions_in_matched_episodes": [{"isaacsim": metadata["isaacsim_version"], "kit": metadata["kit_version"]} for _, metadata, _ in episodes],
            "limitations": ["Separate diagnostic deliberately initializes an interfering closed hand; its penetrations and large impulses are not accepted nominal-operation geometry.",
                "Probe stores per-pair aggregates rather than raw per-substep arrays or absolute native clocks; frozen env.step asserts the native 40-step/.02s contract.",
                "An experimental behavior comparison is used because the low-level runtime exposes no effective self-collision getter.",
                "Reported absolute impulse magnitude is uncalibrated; only controlled presence/absence supports this functionality result."]}


def audit_suite(path, *, max_edge_ratio=1.25, penetration_tolerance=1e-4, contact_push=None, contact_hover=None, self_collision_probe=None):
    path = Path(path).resolve()
    path = path / "suite_result.json" if path.is_dir() else path
    suite = read_json(path)
    report = {"schema": "independent_isaac_suite_audit.v1", "suite": str(path), "suite_sha256": digest(path),
              "auditor_sha256": AUDITOR_SHA256, "checks": {}, "episodes": []}
    rows = suite.get("episodes", [])
    add_check(report, "suite_run_completion", lambda: {"status": "pass" if suite.get("status") == "completed" else "unverified" if suite.get("status") == "running" else "fail",
        "recorded_suite_status": suite.get("status")})

    def coverage():
        found = Counter((row["task"], row["phase"]) for row in rows)
        wanted = {(task, phase) for task in TASK_IDS for phase in ("nominal", "qualification_probe")}
        missing = sorted(wanted - set(found))
        require(not any(count != 1 for count in found.values()), "Repeated task/phase episodes cannot be pooled")
        if missing:
            raise MissingEvidence(f"Four-task nominal/qualification coverage missing: {missing}")
        require(suite.get("qualification_results_excluded_from_nominal") is True, "Suite does not separate qualification episodes")
        require(suite.get("task_policy_success_claimed") is False, "Suite claims unsupported full-policy success")
        return {"task_phase_pairs": len(found), "required_pairs": 8, "task_policy_success_required": False}

    add_check(report, "four_task_phase_coverage", coverage)
    for row in rows:
        episode = audit_episode(row["output"], max_edge_ratio=max_edge_ratio, penetration_tolerance=penetration_tolerance)
        result_path = Path(row["output"]) / "result.json"
        status = "pass"
        reasons = []
        if row.get("returncode") != 0:
            status = "fail"; reasons.append("Episode subprocess did not exit successfully")
        if not result_path.is_file() or not row.get("result_sha256"):
            status = "unverified" if status == "pass" else status; reasons.append("Suite lacks an episode result hash")
        elif digest(result_path) != row["result_sha256"]:
            status = "fail"; reasons.append("Suite/episode result hash mismatch")
        report["checks"][f"episode_{row['task']}_{row['phase']}"] = {
            "status": "fail" if status == "fail" or episode["status"] == "fail" else "unverified" if status == "unverified" or episode["status"] == "unverified" else "pass",
            "required": True, "applicable": True, "reasons": reasons, "episode_audit_status": episode["status"]}
        report["episodes"].append(episode)

    def video_coverage():
        videos = suite.get("task_videos", [])
        if [v["task"] for v in videos] != list(TASK_IDS):
            raise MissingEvidence("Four per-task demo videos are not all present")
        packet_cache = {}
        def packets(path):
            path = str(Path(path).resolve())
            if path not in packet_cache:
                packet_cache[path] = decoded_frame_hashes(path)
            return packet_cache[path]
        for video in videos:
            require(digest(video["video"]) == video["sha256"], f"Task demo hash mismatch: {video['task']}")
            expected_clips = []
            for phase in ("nominal", "qualification_probe"):
                row = next(r for r in rows if r["task"] == video["task"] and r["phase"] == phase)
                expected_clips.append(read_json(Path(row["output"]) / "recording.json")["video"])
            require([str(Path(p).resolve()) for p in video.get("source_clips", [])] == [str(Path(p).resolve()) for p in expected_clips], f"Task demo source clips are not its nominal and qualification episodes: {video['task']}")
            require(packets(video["video"]) == [packet for clip in expected_clips for packet in packets(clip)], f"Task demo decoded frames/order/durations differ from original episode clips: {video['task']}")
        montage = suite.get("montage")
        if not montage:
            raise MissingEvidence("Four-task combined demo is missing")
        require(digest(montage["path"]) == montage["sha256"], "Combined demo hash mismatch")
        require(packets(montage["path"]) == [packet for video in videos for packet in packets(video["video"])], "Combined demo decoded frames/order/durations differ from four task demos")
        return {"task_videos": 4, "montage": montage["path"], "decoded_frame_pixel_hashes_and_durations_match_sources": True}

    add_check(report, "four_task_demo_artifacts", video_coverage)
    nonrigid = max((e["checks"].get("cloth_native_dynamic_response", {}).get("maximum_nonrigid_rms_residual_m", 0.) for e in report["episodes"]), default=0.)
    add_check(report, "visible_nonrigid_cloth_response", lambda: {"status": "pass" if nonrigid >= .0005 else "unverified",
        "maximum_nonrigid_rms_residual_m": nonrigid, "demonstration_threshold_m": .0005,
        "reason": "At least one cloth demonstration must exhibit measured deformation after removing whole-body translation and rotation"})
    active_contacts = sum(e["checks"].get("robot_object_contact_capability", {}).get("nonzero_robot_object_impulse_reports", 0) for e in report["episodes"])
    if contact_push is not None and contact_hover is not None:
        pair = audit_contact_pair(contact_push, contact_hover, max_edge_ratio=max_edge_ratio, penetration_tolerance=penetration_tolerance)
        report["contact_pair"] = pair
        contact_status = pair["status"]
    else:
        contact_status = "pass" if active_contacts else "unverified"
    add_check(report, "robot_object_contact_function", lambda: {"status": contact_status,
        "nonzero_impulse_reports": active_contacts,
        "reason": "Dedicated separately recorded contact diagnostic can supplement the suite; small joint motions alone cannot qualify this capability"})
    self_events = sum(e["checks"].get("robot_self_collision_effective_evidence", {}).get("actual_nonzero_self_impulse_reports", 0) for e in report["episodes"])
    if self_collision_probe is not None:
        add_check(report, "robot_self_collision_effective_evidence", lambda: audit_self_collision_probe(self_collision_probe,
            [row["output"] for row in rows] + ([contact_push, contact_hover] if contact_push is not None else [])))
    else:
        add_check(report, "robot_self_collision_effective_evidence", lambda: {"status": "pass" if self_events else "unverified",
            "actual_nonzero_self_impulse_reports": self_events,
            "reason": "USD/profile flags alone are not effective native readback; absent contacts need an independent paired native collision probe"})
    report["controller_policy_success_required"] = False
    report["cloth_contact_force_measurement_status"] = "unverified"
    return finish_report(report)


def video_png_fidelity(video, frames):
    """Compare every decoded lossy frame to its pinned presentation PNG."""
    from PIL import Image
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0", "-an",
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    minimum_psnr, maximum_mae = float("inf"), 0.
    width, height = 1600, 1140
    try:
        for row in frames:
            raw = process.stdout.read(width * height * 3)
            require(len(raw) == width * height * 3, "Encoded presentation ended before its mapped frame count")
            decoded = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
            with Image.open(row["presentation_frame"]) as source:
                expected = np.asarray(source.convert("RGB"))
            require(expected.shape == decoded.shape, "Presentation PNG dimensions differ from encoded video")
            error = decoded.astype(np.float32) - expected
            mse, mae = float(np.mean(error * error)), float(np.mean(np.abs(error)))
            psnr = float(10 * np.log10(255 ** 2 / max(mse, 1e-12)))
            minimum_psnr, maximum_mae = min(minimum_psnr, psnr), max(maximum_mae, mae)
        require(not process.stdout.read(1), "Encoded presentation contains unmapped extra frames")
        process.wait(timeout=30)
        require(process.returncode == 0, f"Presentation decode failed: {process.stderr.read().decode(errors='replace')[-500:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=30)
        process.stdout.close()
        process.stderr.close()
    require(minimum_psnr >= 28. and maximum_mae <= 3., f"Encoded presentation differs materially from PNG mapping: PSNR={minimum_psnr}, MAE={maximum_mae}")
    return {"decoded_frames_compared": len(frames), "minimum_psnr_db": minimum_psnr,
            "maximum_mean_absolute_rgb_error": maximum_mae, "thresholds": {"minimum_psnr_db": 28., "maximum_mean_absolute_rgb_error": 3.},
            "scope": "Lossy encoding fidelity check; small visually similar frame substitutions are not a cryptographic identity guarantee"}


def audit_presentation(path):
    """Audit final display artifacts without importing or trusting the composer."""
    from PIL import Image, ImageOps
    path = Path(path)
    manifest = read_json(path)
    report = {"schema": "independent_isaac_presentation_audit.v1", "manifest": str(path),
              "manifest_sha256": digest(path), "auditor_sha256": AUDITOR_SHA256, "checks": {}}
    if "derivations" in manifest:
        def index_inputs():
            require(digest(manifest["suite"]) == manifest["suite_sha256"], "Presentation suite reference changed")
            require(manifest.get("source_episodes_modified") is False and manifest.get("policy_success_claimed") is False, "Presentation index claims unsupported source changes/policy success")
            for source, expected in manifest["derivations"].items():
                require(digest(source) == expected, f"Presentation manifest changed: {source}")
            suite = read_json(manifest["suite"])
            require(suite.get("status") == "completed", "Final presentation does not reference a completed suite")
            source_pairs = {(row["task"], row["phase"]): str(Path(row["output"]).resolve()) for row in suite["episodes"]}
            clips, combined, controls = {}, [], []
            for source in manifest["derivations"]:
                item = read_json(source)
                if item.get("type") == "rgb_presentation_concatenation":
                    combined.append(item)
                    continue
                rows = item["sources"]
                if all(row["phase"].startswith("contact_probe_") for row in rows):
                    controls.extend(row["phase"] for row in rows)
                    continue
                require(len(rows) == 2 and [r["phase"] for r in rows] == ["nominal", "qualification_probe"] and rows[0]["task"] == rows[1]["task"], "Per-task presentation does not retain the two ordered phases")
                require(rows[0]["task"] not in clips, "Repeated task presentation")
                clips[rows[0]["task"]] = str(Path(source).resolve())
                for row in rows:
                    require(str(Path(row["episode"]).resolve()) == source_pairs.get((row["task"], row["phase"])), "Final presentation source episode is not from the pinned suite")
            require(set(clips) == set(TASK_IDS) and len(combined) == 1, "Final presentation lacks four distinct tasks and one combined demo")
            require([str(Path(row["manifest"]).resolve()) for row in combined[0]["source_presentations"]] == [clips[task] for task in TASK_IDS], "Final combined demo does not use the four task presentations in order")
            require(not controls or sorted(controls) == ["contact_probe_hover", "contact_probe_push"], "Contact presentation supplement must preserve both push and hover labels")
            return {"derived_manifests_verified": len(manifest["derivations"]), "task_presentations": 4,
                    "combined_presentations": 1, "contact_presentations": controls}
        add_check(report, "index_sources", index_inputs)
        report["presentations"] = []
        for index, source in enumerate(manifest["derivations"]):
            child = audit_presentation(source)
            report["presentations"].append(child)
            add_check(report, f"presentation_{index}", lambda child=child: {"status": child["status"], "manifest": child["manifest"]})
        return finish_report(report)

    def common():
        require(manifest.get("type") in {"rgb_presentation_derivation", "rgb_presentation_concatenation"}, "Unknown presentation derivation type")
        require(manifest["playback_speed"] == 1 and manifest["still_holds"] == [] and manifest["slow_motion_segments"] == [], "Presentation declares speed changes or inserted holds")
        require(manifest["physics_interpolation"] is False and manifest["generated_or_replayed_physics"] is False and manifest["source_episodes_modified"] is False and manifest["policy_success_claimed"] is False, "Presentation declared unsupported generated states/source changes/policy success")
        require(digest(manifest["video"]) == manifest["video_sha256"], "Presentation video hash mismatch")
        changed_composer = []
        for source, expected in manifest["composer_sources"].items():
            snapshot = manifest.get("composer_source_snapshots", {}).get(source)
            require(digest(snapshot or source) == expected, f"Presentation composer source/snapshot changed: {source}")
            if snapshot and (not Path(source).is_file() or digest(source) != expected):
                changed_composer.append(source)
        if manifest.get("font"):
            require(digest(manifest["font"]["path"]) == manifest["font"]["sha256"], "Presentation font changed")
        require(manifest["fps"] > 0 and bool(manifest["frames"]), "Empty/invalid presentation frame map")
        for index, row in enumerate(manifest["frames"]):
            require(row["frame"] == index and abs(row["video_time_s"] - index / manifest["fps"]) < 1e-9, "Presentation frame timing/order is not declared normal speed")
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,r_frame_rate:format=duration", "-of", "json", manifest["video"]], capture_output=True, text=True)
        require(probe.returncode == 0, "Presentation ffprobe failed")
        info = json.loads(probe.stdout)
        require(int(info["streams"][0]["nb_frames"]) == len(manifest["frames"]) and abs(float(info["format"]["duration"]) - len(manifest["frames"]) / manifest["fps"]) <= .02, "Presentation duration/frame count differs from map")
        require((info["streams"][0]["width"], info["streams"][0]["height"]) == (1600, 1140), "Presentation dimensions differ from audited uncropped layout")
        return {"frames": len(manifest["frames"]), "fps": manifest["fps"], "independent_ffprobe": info,
                "current_composer_differences": changed_composer,
                "composer_source_snapshots": manifest.get("composer_source_snapshots", {})}
    add_check(report, "presentation_hashes_timing_and_provenance", common)

    if manifest.get("type") == "rgb_presentation_concatenation":
        def concatenation():
            expected_frames, expected_sources, expected_pixels = [], [], []
            for source in manifest["source_presentations"]:
                require(digest(source["manifest"]) == source["manifest_sha256"] and digest(source["video"]) == source["video_sha256"], "Concatenated source manifest/video changed")
                item = read_json(source["manifest"])
                require(item["fps"] == manifest["fps"] and item["video"] == source["video"] and item["video_sha256"] == source["video_sha256"], "Concatenated source timing/video identity differs")
                offset = len(expected_frames)
                expected_sources.extend({**row, "start_frame": row["start_frame"] + offset} for row in item["sources"])
                for row in item["frames"]:
                    index = len(expected_frames)
                    expected_frames.append({**row, "frame": index, "video_time_s": index / manifest["fps"]})
                expected_pixels.extend(decoded_frame_hashes(source["video"]))
            require(manifest["sources"] == expected_sources and manifest["frames"] == expected_frames, "Combined presentation changed source/physical frame mapping")
            require(decoded_frame_hashes(manifest["video"]) == expected_pixels, "Combined presentation decoded pixel hashes/order/durations differ from source clips")
            return {"source_presentations": len(manifest["source_presentations"]), "decoded_pixel_hashes_and_durations_match_sources": True}
        add_check(report, "source_presentation_concatenation", concatenation)
        return finish_report(report)

    def raw_mapping():
        count = 0
        for source in manifest["sources"]:
            episode = Path(source["episode"])
            require(digest(episode / "artifact_hashes.json") == source["artifact_hashes_sha256"], "Original episode artifact manifest changed")
            pins = read_json(episode / "artifact_hashes.json")
            for filename in ("recording.json", "states.npz", "task.json", "scores.json", "result.json"):
                require(digest(episode / filename) == pins.get(filename), f"Original episode input no longer matches capture: {filename}")
            require(source["recording_sha256"] == pins["recording.json"] and source["states_sha256"] == pins["states.npz"], "Presentation source hash differs from acquisition")
            recording, result, task = (read_json(episode / name) for name in ("recording.json", "result.json", "task.json"))
            phase = result["phase"]
            if phase == "contact_probe":
                require(digest(episode / "diagnostic.json") == pins.get("diagnostic.json"), "Contact presentation diagnostic phase is unpinned")
                phase += "_" + read_json(episode / "diagnostic.json")["mode"]
            require(source["task"] == task["id"] and source["phase"] == phase, "Presentation relabeled the source task/phase")
            require(source["start_frame"] == count and source["frame_count"] == len(recording["frames"]) and recording["fps"] == manifest["fps"], "Presentation omitted/duplicated frames or changed source FPS")
            require(source["original_video"] == recording["video"] and source["original_video_sha256"] == recording["video_sha256"] and digest(recording["video"]) == recording["video_sha256"], "Original acquisition video changed")
            for acquisition in recording["frames"]:
                row = manifest["frames"][count]
                require(row["episode"] == str(episode) and row["source_frame"] == acquisition["frame"] and row["state_index"] == acquisition["state_index"] and row["physical_time_s"] == acquisition["physical_time_s"], "Presentation physical frame mapping differs from recording")
                require(row["main_camera"] == ("overview" if phase == "nominal" else "task"), "Presentation camera label differs from expected phase")
                for image_key, hash_key in (("raw_rgb", "raw_sha256"), ("overview_rgb", "overview_sha256")):
                    image_path = Path(row[image_key])
                    require(image_path.resolve().is_relative_to(episode.resolve()), "Presentation raw source leaves captured episode")
                    require(row[image_key] == acquisition[image_key] and row[hash_key] == acquisition[hash_key] == pins.get(str(image_path.relative_to(episode))) == digest(image_path), "Presentation raw RGB hash differs from acquisition")
                require(digest(row["presentation_frame"]) == row["presentation_sha256"], "Presentation PNG hash mismatch")
                with Image.open(row["raw_rgb"]) as rgb, Image.open(row["overview_rgb"]) as overview, Image.open(row["presentation_frame"]) as presentation:
                    actual = np.asarray(presentation.convert("RGB"))
                    require(actual.shape == (1140, 1600, 3), "Presentation PNG dimensions differ from uncropped layout")
                    primary, secondary = (overview, rgb) if phase == "nominal" else (rgb, overview)
                    main = ImageOps.contain(primary.convert("RGB"), (1600, 900), Image.Resampling.LANCZOS)
                    x, y = (1600 - main.width) // 2, 110 + (900 - main.height) // 2
                    yy, xx = np.ogrid[y:y + main.height, x:x + main.width]
                    mask = ~(((xx <= 320) & (yy <= 165)) | ((xx >= 1176) & (xx <= 1576) & (yy >= 130) & (yy <= 390)))
                    require(np.array_equal(actual[y:y + main.height, x:x + main.width][mask], np.asarray(main)[mask]), "Visible primary camera pixels do not match an uncropped aspect fit of recorded RGB")
                    inset = ImageOps.contain(secondary.convert("RGB"), (400, 225), Image.Resampling.LANCZOS)
                    ix, iy = 1176 + (400 - inset.width) // 2, 165 + (225 - inset.height) // 2
                    require(np.array_equal(actual[iy:iy + inset.height, ix:ix + inset.width], np.asarray(inset)), "Inset pixels do not match uncropped recorded secondary camera")
                count += 1
        require(count == len(manifest["frames"]), "Presentation has unmapped extra PNG frames")
        return {"frames_all_raw_and_presentation_hashes_verified": count, "visible_primary_and_all_inset_pixels_exact": True,
                "scope": "Camera content outside explicit label/inset overlays matches raw capture exactly after the declared aspect-preserving resize. Caption text semantics are not OCR-audited."}
    add_check(report, "raw_rgb_mapping_and_uncropped_pixels", raw_mapping)
    add_check(report, "encoded_video_corresponds_to_png_sequence", lambda: video_png_fidelity(manifest["video"], manifest["frames"]))
    return finish_report(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", type=Path)
    source.add_argument("--suite", type=Path)
    source.add_argument("--presentation", type=Path, help="Final derivation manifest or derivation index JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-edge-ratio", type=float, default=1.25)
    parser.add_argument("--penetration-tolerance", type=float, default=1e-4)
    parser.add_argument("--contact-push", type=Path, help="Supplement suite with a separately recorded contact-probe push")
    parser.add_argument("--contact-hover", type=Path, help="Matched hover control for --contact-push")
    parser.add_argument("--self-collision-probe", type=Path, help="Separate native four-variant self-collision result JSON")
    args = parser.parse_args()
    require(args.max_edge_ratio >= 1 and args.penetration_tolerance > 0, "Invalid audit thresholds")
    require((args.contact_push is None) == (args.contact_hover is None), "Provide both push and hover diagnostic episodes")
    result = (audit_presentation(args.presentation) if args.presentation else audit_episode(args.episode, max_edge_ratio=args.max_edge_ratio, penetration_tolerance=args.penetration_tolerance)
              if args.episode else audit_suite(args.suite, max_edge_ratio=args.max_edge_ratio, penetration_tolerance=args.penetration_tolerance,
                                               contact_push=args.contact_push, contact_hover=args.contact_hover,
                                               self_collision_probe=args.self_collision_probe))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.with_suffix(".auditor.py")
    snapshot.write_bytes(AUDITOR_SOURCE)
    result["auditor_source_snapshot"] = str(snapshot)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"], "required_check_counts": result["required_check_counts"]}, ensure_ascii=False))
    return 1 if result["status"] == "fail" else 2 if result["status"] == "unverified" else 0


if __name__ == "__main__":
    raise SystemExit(main())
