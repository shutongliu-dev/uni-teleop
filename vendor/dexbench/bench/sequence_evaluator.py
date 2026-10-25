"""Read-only, ordered workbench task evaluation over physical observations.

``SequenceEvaluator(task_spec).update(env.observation())`` returns a JSON-ready
progress dictionary. No simulator or reference trajectory is accepted or set.
The first observation establishes a baseline; each stage requires a fresh
geometric entry after activation followed by uninterrupted, sampled dwell.

Schema (distances in metres, quaternion order wxyz)::

    {"id": "kit", "max_observation_gap_s": 0.1,
     "vertex_groups": {"near": [0, 1, 2]},
     "steps": [{"id": "place", "hold_s": 0.3,
                "complete": [{"kind": "rigid", "object": "part",
                              "position_m": [0.5, 0, 0.8], "radius_m": 0.02,
                              "max_speed_m_s": 0.04}],
                "maintain": [...]}]}

``maintain`` defaults to ``complete`` and can use more permissive geometry to
allow cloth deformation in later stages. Breaking any completed stage rolls
back to that stage. Even terminal success is revoked when its maintenance
conditions no longer hold. A future stage already achieved out of order must
leave and re-enter its geometric goal after activation.

Supported predicates:
* rigid: position_m + radius_m or xy_radius_m/z_tolerance_m; optional target
  quaternion_wxyz/orientation_tolerance_rad, or axis_local/axis_world/
  axis_tolerance_rad for rotationally symmetric parts; speed limits optional.
* cloth_region: group (a vertex_groups name, default "all") or indices;
  bounds_m (two xyz corners), min_fraction; optional centroid_m and
  centroid_radius_m, min_span_m, max_speed_m_s.
* cloth_samples: labelled group or indices; points_m sampled desired surface,
  xy_radius_m and z_tolerance_m, min_fraction; optional max_speed_m_s.

All predicates may use frame_object to express positions, axes, and cloth
geometry in that object's local frame. Cloth coverage counts target samples
with at least one nearby labelled vertex at the required height. For a wrap,
combine surface samples with separate low near/far edge regions; a centroid
alone cannot distinguish a drape from translation or a crumpled cloth.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from .geometry import quaternion_matrix, rotation_distance


VERSION = "ordered_workbench_sequence_v1"
_KINDS = {"rigid", "cloth_region", "cloth_samples"}
_EPS = 1e-10


def _array(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _number(value, label, *, positive=False):
    result = float(value)
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


class SequenceEvaluator:
    """An independent streaming scorer; input state and task spec are never mutated."""

    def __init__(self, task_spec):
        self.task_spec = deepcopy(task_spec)
        self.steps = self.task_spec.get("steps", [])
        if not isinstance(self.steps, list) or not self.steps:
            raise ValueError("task_spec.steps must be a nonempty list")
        self.max_gap = _number(self.task_spec.get("max_observation_gap_s", .1),
                               "max_observation_gap_s", positive=True)
        self.groups = self.task_spec.get("vertex_groups", {})
        ids = []
        for step in self.steps:
            ids.append(step["id"])
            _number(step.get("hold_s", .3), "hold_s", positive=True)
            for key in ("complete", "maintain"):
                if key not in step and key == "maintain":
                    continue
                predicates = step.get(key)
                if not isinstance(predicates, list) or not predicates:
                    raise ValueError(f"step {step['id']} {key} must be a nonempty list")
                for predicate in predicates:
                    self._validate_predicate(predicate)
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Step ids must be unique nonempty strings")
        self.reset()

    def reset(self):
        """Start a new episode; the next observation records its initial baseline."""
        self._completed = 0
        self._last_time = None
        self._hold_start = None
        self._armed = False
        self._initial_observation = None
        self._completion_times = []

    @property
    def initial_observation(self):
        """Return a copy so callers cannot alter the recorded baseline."""
        return deepcopy(self._initial_observation)

    def _validate_predicate(self, p):
        if not isinstance(p, Mapping) or p.get("kind") not in _KINDS:
            raise ValueError(f"Unknown predicate kind: {p.get('kind') if isinstance(p, Mapping) else p}")
        for key in ("radius_m", "xy_radius_m", "z_tolerance_m", "centroid_radius_m",
                    "orientation_tolerance_rad", "axis_tolerance_rad", "max_speed_m_s",
                    "max_angular_speed_rad_s"):
            if key in p:
                _number(p[key], key)
        if p["kind"] == "rigid":
            if not isinstance(p.get("object"), str):
                raise ValueError("rigid predicate requires an object name")
            _array(p["position_m"], (3,), "position_m")
            if "radius_m" not in p and not {"xy_radius_m", "z_tolerance_m"} <= p.keys():
                raise ValueError("rigid requires radius_m or xy_radius_m and z_tolerance_m")
            if "quaternion_wxyz" in p:
                quaternion_matrix(p["quaternion_wxyz"])
                if "orientation_tolerance_rad" not in p:
                    raise ValueError("quaternion goal requires orientation_tolerance_rad")
            if any(key in p for key in ("axis_local", "axis_world", "axis_tolerance_rad")):
                for key in ("axis_local", "axis_world"):
                    axis = _array(p[key], (3,), key)
                    if np.linalg.norm(axis) < _EPS:
                        raise ValueError(f"{key} must be nonzero")
                if "axis_tolerance_rad" not in p:
                    raise ValueError("axis goal requires axis_tolerance_rad")
            return
        fraction = float(p.get("min_fraction", 1.))
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("min_fraction must lie in (0, 1]")
        if "indices" in p and "group" in p:
            raise ValueError("Use either indices or a named vertex group")
        group = p.get("group", "all")
        if "indices" not in p and group != "all" and group not in self.groups:
            raise ValueError(f"Unknown cloth vertex group: {group}")
        if p["kind"] == "cloth_region":
            bounds = _array(p["bounds_m"], (2, 3), "bounds_m")
            if np.any(bounds[0] > bounds[1]):
                raise ValueError("bounds_m lower corner exceeds upper corner")
            if "centroid_m" in p:
                _array(p["centroid_m"], (3,), "centroid_m")
                if "centroid_radius_m" not in p:
                    raise ValueError("centroid goal requires centroid_radius_m")
            if "min_span_m" in p:
                span = _array(p["min_span_m"], (3,), "min_span_m")
                if np.any(span < 0):
                    raise ValueError("min_span_m must be nonnegative")
        else:
            points = np.asarray(p["points_m"], dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
                raise ValueError("points_m must be a nonempty finite Nx3 array")
            if not {"xy_radius_m", "z_tolerance_m"} <= p.keys():
                raise ValueError("cloth_samples requires xy_radius_m and z_tolerance_m")

    @staticmethod
    def _pose(observation, name):
        obj = observation["objects"][name]
        position = _array(obj["position_m"], (3,), f"{name} position_m")
        rotation = quaternion_matrix(obj["quaternion_wxyz"])
        return obj, position, rotation

    def _frame(self, observation, predicate):
        if "frame_object" not in predicate:
            return np.zeros(3), np.eye(3)
        _, position, rotation = self._pose(observation, predicate["frame_object"])
        return position, rotation

    def _vertices(self, observation, p):
        cloth = observation["cloth"]
        vertices = np.asarray(cloth["vertices_m"], dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or not np.isfinite(vertices).all():
            raise ValueError("cloth.vertices_m must be nonempty finite Nx3")
        selected = p.get("indices", self.groups.get(p.get("group", "all")))
        if selected is None:
            indices = np.arange(len(vertices))
        else:
            indices = np.asarray(selected)
            if (indices.ndim != 1 or len(indices) == 0 or not np.issubdtype(indices.dtype, np.integer)
                    or np.min(indices) < 0 or np.max(indices) >= len(vertices)
                    or len(np.unique(indices)) != len(indices)):
                raise ValueError("Cloth group must contain unique in-range integer vertex indices")
        position, rotation = self._frame(observation, p)
        return cloth, indices, (vertices[indices] - position) @ rotation

    def _predicate(self, observation, p, *, geometry_only=False):
        if p["kind"] == "rigid":
            obj, position, rotation = self._pose(observation, p["object"])
            frame_p, frame_r = self._frame(observation, p)
            position = frame_r.T @ (position - frame_p)
            rotation = frame_r.T @ rotation
            delta = position - np.asarray(p["position_m"])
            if "radius_m" in p and np.linalg.norm(delta) > p["radius_m"] + _EPS:
                return False, "position"
            if "xy_radius_m" in p and np.linalg.norm(delta[:2]) > p["xy_radius_m"] + _EPS:
                return False, "xy_position"
            if "z_tolerance_m" in p and abs(delta[2]) > p["z_tolerance_m"] + _EPS:
                return False, "height"
            if "quaternion_wxyz" in p:
                error = rotation_distance(rotation, quaternion_matrix(p["quaternion_wxyz"]))
                if error > p["orientation_tolerance_rad"] + _EPS:
                    return False, "orientation"
            if "axis_local" in p:
                actual = rotation @ np.asarray(p["axis_local"], dtype=float)
                desired = np.asarray(p["axis_world"], dtype=float)
                cosine = np.dot(actual, desired) / (np.linalg.norm(actual) * np.linalg.norm(desired))
                if np.arccos(np.clip(cosine, -1, 1)) > p["axis_tolerance_rad"] + _EPS:
                    return False, "axis_orientation"
            if not geometry_only:
                for field, limit, reason in (("linear_velocity_m_s", "max_speed_m_s", "speed"),
                                             ("angular_velocity_rad_s", "max_angular_speed_rad_s", "angular_speed")):
                    if limit in p and np.linalg.norm(_array(obj[field], (3,), field)) > p[limit] + _EPS:
                        return False, reason
            return True, "satisfied"

        cloth, indices, vertices = self._vertices(observation, p)
        if p["kind"] == "cloth_region":
            bounds = np.asarray(p["bounds_m"], dtype=float)
            covered = np.all((vertices >= bounds[0] - _EPS) & (vertices <= bounds[1] + _EPS), axis=1)
            if np.mean(covered) + _EPS < p.get("min_fraction", 1.):
                return False, "cloth_region_coverage"
            if "centroid_m" in p and np.linalg.norm(vertices.mean(axis=0) - p["centroid_m"]) > p["centroid_radius_m"] + _EPS:
                return False, "cloth_centroid"
            if "min_span_m" in p and np.any(np.ptp(vertices, axis=0) + _EPS < p["min_span_m"]):
                return False, "cloth_span"
        else:
            # Chunking bounds memory for fine cloth meshes / dense target grids.
            points = np.asarray(p["points_m"], dtype=float)
            covered_count = 0
            for start in range(0, len(points), 128):
                delta = vertices[None, :, :] - points[start:start + 128, None, :]
                covered = ((np.linalg.norm(delta[..., :2], axis=-1) <= p["xy_radius_m"] + _EPS)
                           & (np.abs(delta[..., 2]) <= p["z_tolerance_m"] + _EPS)).any(axis=1)
                covered_count += int(covered.sum())
            if covered_count / len(points) + _EPS < p.get("min_fraction", 1.):
                return False, "cloth_surface_coverage"
        if not geometry_only and "max_speed_m_s" in p:
            velocities = np.asarray(cloth["velocities_m_s"], dtype=float)
            if velocities.shape != np.asarray(cloth["vertices_m"]).shape or not np.isfinite(velocities).all():
                raise ValueError("cloth.velocities_m_s must match vertices_m and be finite")
            if np.linalg.norm(velocities[indices], axis=1).max() > p["max_speed_m_s"] + _EPS:
                return False, "cloth_speed"
        return True, "satisfied"

    def _conditions(self, observation, predicates, *, geometry_only=False):
        for index, predicate in enumerate(predicates):
            good, reason = self._predicate(observation, predicate, geometry_only=geometry_only)
            if not good:
                return False, f"{index}:{reason}"
        return True, "satisfied"

    def inspect_conditions(self, observation):
        """Inspect every predicate without recording baseline or advancing order.

        This is useful for reset-at-goal qualification. Satisfied conditions
        here never imply completion of the ordered task.
        """
        result = []
        for step in self.steps:
            row = {"id": step["id"], "hold_required_s": step.get("hold_s", .3)}
            for name in ("complete", "maintain"):
                records = []
                for index, predicate in enumerate(step.get(name, step["complete"])):
                    record = {"index": index, "kind": predicate["kind"],
                              "object": predicate.get("object"), "group": predicate.get("group", "all")}
                    try:
                        record["satisfied"], record["reason"] = self._predicate(observation, predicate)
                        record["geometry_satisfied"], record["geometry_reason"] = self._predicate(
                            observation, predicate, geometry_only=True)
                    except (KeyError, ValueError, TypeError, IndexError) as exc:
                        record.update(satisfied=False, geometry_satisfied=False,
                                      reason=f"invalid_observation:{exc}", geometry_reason=f"invalid_observation:{exc}")
                    records.append(record)
                row[name] = records
                row[name + "_satisfied"] = all(r["satisfied"] for r in records)
            result.append(row)
        return result

    def _result(self, reason, *, time=None):
        active = self.steps[self._completed] if self._completed < len(self.steps) else None
        dwell = max(0., time - self._hold_start) if time is not None and self._hold_start is not None else 0.
        return {"evaluator": VERSION, "task_id": self.task_spec.get("id"),
                "progress": self._completed / len(self.steps),
                "success": self._completed == len(self.steps), "reason": reason,
                "completed_steps": [s["id"] for s in self.steps[:self._completed]],
                "active_step": active["id"] if active else None,
                "active_step_armed": self._armed if active else False,
                "hold_elapsed_s": dwell, "hold_required_s": active.get("hold_s", .3) if active else 0.,
                "completion_times_s": list(self._completion_times)}

    def update(self, observation):
        """Score one physical state; timestamps must be strictly increasing.

        Missing/nonfinite scoring data yields invalid_observation and revokes
        progress. A bad timestamp is an API error and raises ValueError. Extra
        fields (including hidden source references) are ignored.
        """
        time = float(observation["time_s"])
        if not np.isfinite(time) or (self._last_time is not None and time <= self._last_time):
            raise ValueError("Observation time_s must be finite and strictly increasing")
        gap = self._last_time is not None and time - self._last_time > self.max_gap + _EPS
        first = self._last_time is None
        self._last_time = time
        if first:
            self._initial_observation = deepcopy(observation)
        try:
            if not observation.get("metrics", {}).get("state_finite", True):
                raise ValueError("Physics state reports nonfinite data")
            # Check all task predicates so missing or malformed future state
            # cannot go unnoticed until that stage has become active.
            complete = [self._conditions(observation, s["complete"]) for s in self.steps]
            geometry = [self._conditions(observation, s["complete"], geometry_only=True)[0] for s in self.steps]
            maintain = [self._conditions(observation, s.get("maintain", s["complete"])) for s in self.steps]
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            self._completed, self._armed, self._hold_start = 0, False, None
            self._completion_times.clear()
            return self._result(f"invalid_observation:{exc}", time=time)

        for index in range(self._completed):
            if not maintain[index][0]:
                self._completed = index
                self._completion_times = self._completion_times[:index]
                self._hold_start = None
                self._armed = not geometry[index]
                return self._result(f"previous_step_lost:{self.steps[index]['id']}:{maintain[index][1]}", time=time)
        if self._completed == len(self.steps):
            return self._result("success", time=time)

        index = self._completed
        if not geometry[index]:
            self._armed = True
        if first:
            return self._result("baseline_recorded", time=time)
        if gap:
            self._hold_start = None
            return self._result("observation_gap:dwell_reset", time=time)
        if not self._armed:
            self._hold_start = None
            return self._result("waiting_for_fresh_entry", time=time)
        good, reason = complete[index]
        if not good:
            self._hold_start = None
            return self._result(f"waiting_for_condition:{self.steps[index]['id']}:{reason}", time=time)
        if self._hold_start is None:
            self._hold_start = time
        if time - self._hold_start + _EPS < self.steps[index].get("hold_s", .3):
            return self._result(f"holding:{self.steps[index]['id']}", time=time)

        self._completed += 1
        self._completion_times.append(time)
        self._hold_start = None
        self._armed = self._completed < len(self.steps) and not geometry[self._completed]
        return self._result("success" if self._completed == len(self.steps)
                            else f"step_completed:{self.steps[index]['id']}", time=time)
