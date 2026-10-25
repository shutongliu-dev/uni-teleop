"""Offline physical support review, independent of task goal scoring."""
import numpy as np


def support_evidence(runtime, records, task, dwell_s=.3):
    metadata = task["asset_metadata"]
    actors = metadata["actors"]
    states = [r["observation"] for r in records if r["kind"] == "step"]
    if not states:
        return {"passed": False, "reason": "no_native_steps"}
    end = states[-1]["time_s"]
    terminal = [r for r in states if r["time_s"] >= end-dwell_s-1.e-8]
    loaded = {name: 0 for name in actors}
    robot_contacts = {name: 0 for name in actors}
    for obs in terminal:
        for row in obs["contacts"]["rigid"]:
            pair = row["actor0"], row["actor1"]
            for name, path in actors.items():
                indices = [i for i, item in enumerate(pair) if item == path or item.startswith(path+"/")]
                if not indices:
                    continue
                index = indices[0]
                other = pair[1-index]
                if other == "/World/Robot" or other.startswith("/World/Robot/"):
                    robot_contacts[name] += 1
                # Signed normal impulse only; no force calibration claim.
                upward = row["impulse_vector_ns"][2]*(1 if index == 0 else -1)
                if upward > 1.e-10 and not any(
                        other == p or other.startswith(p+"/") for p in actors.values()):
                    loaded[name] += 1
    groups = metadata["task_parameters"].get("flexible_groups", [])
    membership = {name: group for group in groups for name in group["actor_ids"]}
    results = {}
    for name in actors:
        positions = np.array([r["objects"][name]["position_m"] for r in terminal])
        velocities = np.array([r["objects"][name]["linear_velocity_m_s"] for r in terminal])
        excursion = float(np.max(np.linalg.norm(positions-positions[-1], axis=1)))
        speed = float(np.max(np.linalg.norm(velocities, axis=1)))
        group = membership.get(name)
        # An intact passive cable may transmit support through its own links.
        # This is declared constraint support, never invented contact force.
        connected = group["actor_ids"] if group else [name]
        group_loaded = sum(loaded[n] for n in connected)
        finite = bool(np.all(np.isfinite(positions)) and np.all(np.isfinite(velocities)))
        results[name] = {
            "passed": bool(finite and group_loaded > 0 and not robot_contacts[name]
                           and excursion <= .006 and speed <= .05),
            "loaded_external_contact_rows": loaded[name],
            "connected_group_loaded_rows": group_loaded,
            "passive_group_support": bool(group),
            "robot_contact_rows": robot_contacts[name],
            "terminal_excursion_m": excursion, "maximum_terminal_linear_speed_m_s": speed,
        }
    duration = terminal[-1]["time_s"]-terminal[0]["time_s"]
    return {"passed": bool(duration >= dwell_s-.020001 and all(r["passed"] for r in results.values())),
            "actors": results, "terminal_window_s": duration,
            "scope": "Recorded native reset support and rest; no quantitative force or robot manipulation acceptance."}
