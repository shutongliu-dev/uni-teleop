"""Explicit observation projection. Never send a raw simulator observation."""
from __future__ import annotations

from .schemas import AgentObservation, CameraObservation, EpisodeSpec, Pose, Proprioception


def project_observation(raw: dict, spec: EpisodeSpec, sequence: int, *,
                        cameras: list[CameraObservation] | None = None,
                        robot_geometry: dict | None = None,
                        tactile: dict | None = None,
                        hand_tactile: dict | None = None,
                        last_action_id: str | None = None,
                        last_action_status: str = "idle") -> AgentObservation:
    """Copy named permitted fields; unknown fields and evaluator data are dropped.

    Images must be produced by the trusted sensor adapter. Dataset/host file paths
    should be mapped to opaque episode-scoped resource URLs by the HTTP server.
    This function neither reads arbitrary image paths nor turns object names into
    oracle poses. Geometry supplied by an agent is its own inferred target.
    """
    result = dict(
        episode_id=spec.episode_id, sequence=sequence,
        sim_time_s=float(raw["sim_time_s"]), instruction=spec.instruction,
        profile=spec.observation_profile,
        proprioception=Proprioception(
            joint_positions_rad=raw["joint_positions_rad"],
            joint_velocities_rad_s=raw["joint_velocities_rad_s"],
            commanded_joint_positions_rad=raw.get("commanded_joint_positions_rad"),
            # A bimanual robot publishes one FK per hand: {"hand_geometries": {...}}.
            geometry=None if robot_geometry is not None and "hand_geometries" in robot_geometry else robot_geometry,
            hand_geometries=(robot_geometry or {}).get("hand_geometries")),
        cameras=cameras or [], tactile=tactile, last_action_id=last_action_id,
        **({} if hand_tactile is None else {'hand_tactile': hand_tactile}),
        last_action_status=last_action_status)
    if spec.observation_profile == "privileged_state":
        result["privileged_object_poses"] = {
            name: Pose(frame_id="world", position_m=pose["position_m"],
                       quaternion_wxyz=pose["quaternion_wxyz"])
            for name, pose in raw.get("objects", {}).items()}
        result["privileged_cloth_vertices_m"] = raw.get("cloth", {}).get("vertices_m", [])
    return AgentObservation(**result)


def project_policy_observation(raw: dict, spec: EpisodeSpec, sequence: int, *,
                               cameras: list[CameraObservation] | None = None) -> AgentObservation:
    """Separate trusted System0 view. Never return this object through Agent HTTP.

    A fixed teacher may use ground truth while the agent remains vision-only.
    Its visibility is independently pinned in the descriptor and report cohort.
    """
    if spec.system0 is None:
        raise ValueError("no System0 policy is pinned")
    policy_spec = spec.model_copy(update={"observation_profile": spec.system0.observation_profile})
    return project_observation(raw, policy_spec, sequence, cameras=cameras)
