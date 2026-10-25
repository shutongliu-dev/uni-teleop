"""Physical-failure regressions; CPU geometry only, no simulation runtime."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bench.evaluation.physics import (
    audit_native_episode, audit_source_paths, physics_profile, physics_profile_sha256,
)


class Recording:
    def __init__(self, root, *, cloth=True, frames=2):
        self.root = root
        self.scene = {"cloth": {"enabled": cloth, "cells_xy": [1, 1], "size_m": [.1, .1]},
                      "table": {"position_m": [0, 0, -.1], "half_extents_m": [1, 1, .1]},
                      "fixtures": [], "objects": []}
        # Passive coordinates remain in physical captures although ingress has
        # only two independently actuated joints.
        self.meta = {"joint_names": ["arm", "active_finger", "passive_mimic"],
                     "action_spec": {"joint_names": ["arm", "active_finger"]},
                     "body_names": ["palm"], "object_names": [],
                     "capture_state_quaternion": "xyzw", "observation_quaternion": "wxyz",
                     "native_isaacsim_app": True, "standalone_newton_runtime_used": False,
                     "physics_dt_s": .0005, "control_dt_s": .02,
                     "scene_info": {"fixtures": {}, "objects": {}}}
        vertices = [[0, 0, 1], [.1, 0, 1], [0, .1, 1], [.1, .1, 1]] if cloth else np.zeros((0, 3))
        particle_q = np.repeat(np.asarray(vertices, float)[None, ...], frames, axis=0)
        self.states = {"time_s": np.arange(frames) * .02,
                       "native_world_time_s": 12 + np.arange(frames) * .02,
                       "native_physics_step_index": 24000 + np.arange(frames) * 40,
                       "joint_q": np.zeros((frames, 3)), "joint_qd": np.zeros((frames, 3)),
                       "joint_target_q": np.zeros((frames, 3)),
                       "body_q": np.tile([[[0, 0, 0, 0, 0, 0, 1]]], (frames, 1, 1)),
                       "body_qd": np.zeros((frames, 1, 6)),
                       "object_q": np.zeros((frames, 0, 7)), "object_qd": np.zeros((frames, 0, 6)),
                       "particle_q": particle_q, "particle_qd": np.zeros_like(particle_q)}
        self.triangles = np.asarray([[0, 1, 2], [1, 3, 2]], int) if cloth else np.zeros((0, 3), int)
        self.observations = []

    def write(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta["scene_spec"] = deepcopy(self.scene)
        self.observations = []
        for i, t in enumerate(self.states["time_s"]):
            observation = {"time_s": float(t), "sim_time_s": float(t), "metrics": {"state_finite": True},
                           "joint_positions_rad": dict(zip(self.meta["joint_names"], self.states["joint_q"][i].tolist())),
                           "joint_velocities_rad_s": dict(zip(self.meta["joint_names"], self.states["joint_qd"][i].tolist())),
                           "objects": {}, "cloth": {"vertices_m": self.states["particle_q"][i].tolist(),
                                                     "triangles": self.triangles.tolist()},
                           "contacts": {"source": "PhysX solver contact report callbacks", "errors": [], "rigid": [],
                                        "rigid_contact_point_count": 0, "rigid_impulse_magnitude_sum_ns": 0.}}
            for j, name in enumerate(self.meta["object_names"]):
                q = self.states["object_q"][i, j]
                observation["objects"][name] = {"position_m": q[:3].tolist(), "quaternion_wxyz": q[[6, 3, 4, 5]].tolist()}
            self.observations.append(observation)
        np.savez(self.root / "states.npz", **self.states)
        np.save(self.root / "triangles.npy", self.triangles)
        for name, value in (("scene", self.scene), ("metadata", self.meta)):
            (self.root / (name + ".json")).write_text(json.dumps(value))
        self.write_observations()
        return self.root

    def write_observations(self):
        rows = []
        for i, obs in enumerate(self.observations):
            row = {"kind": "reset" if i == 0 else "step", "observation": obs}
            if i:
                row["command"] = {"duration_s": .02, "joint_positions_rad": {"arm": 0., "active_finger": 0.}}
            rows.append(row)
        (self.root / "physics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def contact(self, separation):
        pair = {"time_s": float(self.states["time_s"][-1]), "actor0": "/World/Objects/block", "actor1": "/World/Table",
                "collider0": None, "collider1": None, "point_count": 1,
                "impulse_magnitude_sum_ns": .01, "min_separation_m": separation}
        self.observations[-1]["contacts"].update(rigid=[pair], rigid_contact_point_count=1, rigid_impulse_magnitude_sum_ns=.01)
        self.write_observations()


def test_cloth_frames_without_substeps_do_not_certify_physics(tmp_path):
    r = audit_native_episode(Recording(tmp_path).write())
    assert r["status"] == "unchecked", r
    assert r["physics_valid"] is False
    assert r["checks"]["finite_states_and_alignment"]["physical_joints"] == 3
    assert r["checks"]["cloth_topology_strain_self_intersection"]["status"] == "passed"
    assert r["checks"]["cloth_task_solid_penetration"]["status"] == "passed"
    assert r["checks"]["cloth_every_substep"]["status"] == "unchecked"
    assert r["checks"]["cloth_every_substep"]["required"] is True
    assert r["checks"]["cloth_every_substep"]["error_type"] == "MissingEvidence"
    assert r["checks"]["independent_full_rigid_mesh"]["status"] == "unchecked"
    assert r["checks"]["independent_full_rigid_mesh"]["required"] is False
    assert r["physics_profile_sha256"] == physics_profile_sha256()
    json.dumps(r, allow_nan=False)


def test_no_cloth_and_reset_only_has_no_dynamic_success_claim(tmp_path):
    r = audit_native_episode(Recording(tmp_path, cloth=False, frames=1).write())
    assert r["status"] == "passed", r
    assert r["checks"]["native_clock"]["reset_only"] is True
    assert r["checks"]["native_clock"]["dynamic_capability_claimed"] is False
    assert r["checks"]["cloth_topology_strain_self_intersection"]["applicable"] is False


@pytest.mark.parametrize("filename", ["states.npz", "metadata.json", "scene.json", "physics.jsonl", "triangles.npy"])
def test_missing_evidence_never_defaults_to_pass(tmp_path, filename):
    Recording(tmp_path).write()
    (tmp_path / filename).unlink()
    report = audit_native_episode(tmp_path)
    assert report["status"] == "unchecked", report
    assert report["physics_valid"] is False


@pytest.mark.parametrize("mutation", ["nan", "clock", "native_steps", "quaternion", "observation_joints", "loaded_scene"])
def test_recording_disagreement_is_physics_failure(tmp_path, mutation):
    recording = Recording(tmp_path)
    if mutation == "nan":
        recording.states["joint_q"][1, 0] = np.nan
    elif mutation == "clock":
        recording.states["native_world_time_s"][1] += .02
    elif mutation == "native_steps":
        recording.states["native_physics_step_index"][1] -= 1
    elif mutation == "quaternion":
        recording.states["body_q"][1, 0, 6] = 0
    recording.write()
    if mutation == "observation_joints":
        recording.observations[1]["joint_positions_rad"]["passive_mimic"] = .5
        recording.write_observations()
    elif mutation == "loaded_scene":
        recording.meta["scene_spec"]["table"]["position_m"][2] = 99
        (tmp_path / "metadata.json").write_text(json.dumps(recording.meta))
    report = audit_native_episode(tmp_path)
    assert report["status"] == "failed", report


@pytest.mark.parametrize("mutation,metric", [
    ("stretch", "maximum_edge_ratio"), ("collapse", "degenerate_faces_total"),
    ("self_intersection", "self_intersection_pairs_total"),
])
def test_finite_but_broken_cloth_is_rejected(tmp_path, mutation, metric):
    recording = Recording(tmp_path)
    if mutation == "stretch":
        recording.states["particle_q"][1, :, 0] *= 1.3
    elif mutation == "collapse":
        recording.states["particle_q"][1, 3] = recording.states["particle_q"][1, 1]
    else:
        recording.states["particle_q"][1, 3] = recording.states["particle_q"][1, 0]
    report = audit_native_episode(recording.write())
    assert report["checks"]["finite_states_and_alignment"]["status"] == "passed"
    assert report["checks"]["cloth_topology_strain_self_intersection"]["status"] == "failed", report
    assert report["checks"]["cloth_topology_strain_self_intersection"][metric] > 0
    assert report["physics_valid"] is False


@pytest.mark.parametrize("triangles", [[], [[0, 1, 3], [1, 3, 2]], [[0, 1, 2], [1, 9, 2]]])
def test_missing_or_changed_topology_rejected(tmp_path, triangles):
    recording = Recording(tmp_path)
    recording.triangles = np.asarray(triangles, int).reshape(-1, 3)
    report = audit_native_episode(recording.write())
    assert report["checks"]["cloth_topology_strain_self_intersection"]["status"] == "failed", report


def test_triangle_interior_penetration_detected_when_all_vertices_outside_table(tmp_path):
    recording = Recording(tmp_path)
    recording.scene["cloth"]["size_m"] = [4, 4]
    recording.scene["table"] = {"position_m": [0, 0, 0], "half_extents_m": [.1, .1, .1]}
    recording.states["particle_q"][:] = [[-2, -2, 0], [2, -2, 0], [-2, 2, 0], [2, 2, 0]]
    report = audit_native_episode(recording.write())
    assert report["checks"]["cloth_topology_strain_self_intersection"]["status"] == "passed"
    check = report["checks"]["cloth_task_solid_penetration"]
    assert check["status"] == "failed", check
    assert any(v["maximum_confirmed_piece_penetration_m"] > .001 for v in check["violations_or_unresolved"])


@pytest.mark.parametrize("separation,expected", [(-.00201, "failed"), (-.00199, "passed"), (None, "unchecked")])
def test_new_profile_gates_all_reported_rigid_pairs(tmp_path, separation, expected):
    recording = Recording(tmp_path, cloth=False)
    recording.write()
    recording.contact(separation)
    report = audit_native_episode(tmp_path)
    assert report["checks"]["reported_rigid_separations"]["status"] == expected, report
    assert report["status"] == expected


def test_missing_contacts_not_a_silent_empty_stream(tmp_path):
    recording = Recording(tmp_path, cloth=False)
    recording.write()
    del recording.observations[1]["contacts"]
    recording.write_observations()
    report = audit_native_episode(tmp_path)
    assert report["status"] == "unchecked", report


def test_contact_aggregation_errors_rejected(tmp_path):
    recording = Recording(tmp_path, cloth=False)
    recording.write()
    recording.contact(0.)
    recording.observations[1]["contacts"]["rigid_contact_point_count"] = 9
    recording.write_observations()
    report = audit_native_episode(tmp_path)
    assert report["checks"]["native_contact_stream"]["status"] == "failed"


@pytest.mark.parametrize("cloth,expected_status", [(False, "passed"), (True, "unchecked")])
def test_hold_marker_does_not_create_fake_physics_step(tmp_path, cloth, expected_status):
    recording = Recording(tmp_path, cloth=cloth)
    recording.write()
    with (tmp_path / "physics.jsonl").open("a") as file:
        file.write(json.dumps({"kind": "cancel_hold_targets", "sim_time_s": .02}) + "\n")
    report = audit_native_episode(tmp_path)
    assert report["status"] == expected_status
    assert report["checks"]["recording_readable"]["physical_records"] == 2
    assert report["checks"]["native_clock"]["native_steps"] == 40


def test_profile_and_implementation_dependencies_are_hashable_and_immutable():
    original = physics_profile_sha256()
    profile = physics_profile()
    profile["max_reported_rigid_penetration_m"] = 9
    assert physics_profile_sha256() == original
    assert original == hashlib.sha256(json.dumps(physics_profile(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    paths = audit_source_paths()
    assert all(path.is_absolute() and path.is_file() for path in paths)
    assert len(paths) == len(set(paths))
    assert {"physics.py", "convex_geometry.py", "audit.py", "task_parts.py",
            "camera_calibration.py"} <= {p.name for p in paths}


def test_geometry_audit_does_not_import_any_simulator(tmp_path):
    Recording(tmp_path).write()
    code = "from bench.evaluation.physics import audit_native_episode; import sys; " \
           "report = audit_native_episode(sys.argv[1]); " \
           "assert report['status'] == 'unchecked' and report['physics_valid'] is False; " \
           "assert report['checks']['cloth_every_substep']['error_type'] == 'MissingEvidence'; " \
           "assert not any(x == p or x.startswith(p + '.') for x in sys.modules for p in " \
           "('newton', 'warp', 'mujoco', 'isaacsim', 'omni', 'bench.simulation.newton_backend'))"
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def _add_cylinder(recording, asset_root, *, dynamic=False, geometry="cylinder"):
    """Actual shared-catalog reader with a tiny, hash-pinned test-only asset."""
    asset_root.mkdir()
    visual = asset_root / "visual.obj"
    visual.write_text("v 0 0 0\n")
    visual_sha = hashlib.sha256(visual.read_bytes()).hexdigest()
    manifest = {"assets": {"cylinder": {"mesh_path": visual.name, "sha256": visual_sha, "visual_parts": [],
                "collision_parts": [{"geometry": geometry, "radius_m": .1, "half_height_m": .1}]}}}
    manifest_path = asset_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    category = "objects" if dynamic else "fixtures"
    recording.scene[category].append({"id": "cylinder", "asset_id": "cylinder", "asset_root": str(asset_root),
                                      "position_m": [.05, .05, 1.]})
    recording.meta["scene_info"][category]["cylinder"] = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "asset_id": "cylinder", "collision_parts": 1}
    if dynamic:
        recording.meta["object_names"] = ["cylinder"]
        frames = len(recording.states["time_s"])
        recording.states["object_q"] = np.tile([[[.05, .05, 1., 0., 0., 0., 1.]]], (frames, 1, 1))
        recording.states["object_qd"] = np.zeros((frames, 1, 6))
    return manifest_path


@pytest.mark.parametrize("z,expected", [(1., "failed"), (1.10005, "unchecked"), (1.2, "passed")])
def test_analytic_cylinder_distinguishes_penetration_unresolved_and_certified_clear(tmp_path, z, expected):
    recording = Recording(tmp_path / "record")
    _add_cylinder(recording, tmp_path / "asset")
    recording.states["particle_q"][:, :, 2] = z
    report = audit_native_episode(recording.write())
    check = report["checks"]["cloth_task_solid_penetration"]
    assert check["status"] == expected, check
    assert "cylinder/0" in check["piece_names"]
    if expected == "unchecked":
        rows = check["violations_or_unresolved"]
        assert all(row["minimum_sampled_surface_sdf_m"] > 0 for row in rows)
        assert all(row["minimum_certified_surface_sdf_lower_bound_m"] < 0 for row in rows)


def test_dynamic_cylinder_uses_recorded_pose_instead_of_initial_scene_pose(tmp_path):
    recording = Recording(tmp_path / "record")
    _add_cylinder(recording, tmp_path / "asset", dynamic=True)
    recording.scene["objects"][0]["position_m"] = [10, 10, 10]
    recording.states["object_q"][0, 0, :3] = [10, 10, 10]
    report = audit_native_episode(recording.write())
    check = report["checks"]["cloth_task_solid_penetration"]
    assert check["status"] == "failed", check
    assert {row["frame"] for row in check["violations_or_unresolved"]} == {1}


def test_unsupported_task_geometry_withholds_certificate(tmp_path):
    recording = Recording(tmp_path / "record")
    _add_cylinder(recording, tmp_path / "asset", geometry="unknown_primitive")
    report = audit_native_episode(recording.write())
    assert report["status"] == "unchecked", report
    assert report["checks"]["cloth_task_solid_penetration"]["omitted_pieces"]


def test_catalog_changed_after_native_capture_rejected(tmp_path):
    recording = Recording(tmp_path / "record")
    manifest = _add_cylinder(recording, tmp_path / "asset")
    recording.write()
    manifest.write_text(manifest.read_text() + "\n")
    report = audit_native_episode(recording.root)
    assert report["checks"]["cloth_task_solid_penetration"]["status"] == "failed", report
