import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from sonic_mujoco.controllers.sonic import SonicController
from sonic_mujoco.envs.mujoco.g1 import MujocoG1SweepEnv
from sonic_mujoco.envs.mujoco.g1.sweep_env import OBJECT_NAMES
from sonic_mujoco.gr00t import (
    INITIAL_MOTION_TOKEN,
    PROMPT,
    Gr00tClient,
    StereoCamera,
    VideoWriter,
    build_observation,
    projected_gravity,
    save_image,
)
from sonic_mujoco.tactile_calibration import load_tactile_calibration_profile

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GR00T on the MuJoCo sweep task"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument(
        "--decoder", type=Path, default=ROOT / "models/model_decoder.onnx"
    )
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--action-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--tactile-profile",
        type=Path,
        help="versioned tactile Sim2Real profile applied to policy packets",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/gr00t_sweep")
    return parser.parse_args()


def has_fallen(state) -> bool:
    return bool(
        state.base_position[2] < 0.45
        or projected_gravity(state.base_quaternion)[2] > -0.5
    )


def main() -> None:
    args = parse_args()
    tactile_profile = (
        load_tactile_calibration_profile(args.tactile_profile)
        if args.tactile_profile is not None
        else None
    )
    run_dir = args.output_dir / datetime.now().astimezone().strftime(
        "%Y-%m-%d-%H-%M-%S"
    )
    run_dir.mkdir(parents=True)

    env = MujocoG1SweepEnv()
    if tactile_profile is not None:
        env.configure_tactile_profile(tactile_profile, seed=args.seed)
    controller = SonicController.from_onnx(args.decoder)
    camera = StereoCamera(env.model, env.data)
    policy = Gr00tClient(args.host, args.port)
    video = None
    if not args.no_video:
        video = VideoWriter(run_dir / "rollout.mp4", 1280, 480, args.video_fps)
    env.reset(seed=args.seed)
    controller.reset()

    initial = env.get_scene_state().object_position.copy()
    action_chunks = 0
    token_min = np.inf
    token_max = -np.inf
    fell = False
    fall_stage = None
    max_object_displacement = np.zeros(len(OBJECT_NAMES))
    video_period = max(1, round(50 / args.video_fps))

    def record_video_frame() -> None:
        if video is None:
            return
        ego = camera.render()["ego_view_left"]
        video.write(np.concatenate((camera.render_observer(), ego), axis=1))

    try:
        if not policy.ping():
            raise RuntimeError(
                f"PolicyServer is unavailable at {args.host}:{args.port}"
            )
        policy.reset()
        initial_images = camera.render()
        save_image(run_dir / "initial_left.png", initial_images["ego_view_left"])

        warmup_steps = round(args.warmup_seconds * 50)
        for step in range(warmup_steps):
            state = env.get_robot_state()
            command = controller.act(state, INITIAL_MOTION_TOKEN)
            env.step(command, controller.steps_per_action)
            fell = has_fallen(env.get_robot_state())
            if fell:
                fall_stage = "warmup"
                break
            if step % video_period == 0:
                record_video_frame()

        ready_images = camera.render()
        save_image(run_dir / "ready_left.png", ready_images["ego_view_left"])
        ready_robot = env.get_robot_state()
        max_joint_delta = np.zeros(29)
        total_steps = round(args.seconds * 50)
        completed = 0
        while completed < total_steps and not env.is_success() and not fell:
            state = env.get_robot_state()
            observation = build_observation(
                state,
                camera.render(),
                args.prompt,
                env.tactile_suit,
            )
            tokens = policy.get_action(observation)["motion_token"][0]
            token_min = min(token_min, float(tokens.min()))
            token_max = max(token_max, float(tokens.max()))
            action_chunks += 1
            for token in tokens[: args.action_steps]:
                state = env.get_robot_state()
                env.step(controller.act(state, token), controller.steps_per_action)
                completed += 1
                state = env.get_robot_state()
                max_joint_delta = np.maximum(
                    max_joint_delta,
                    np.abs(state.joint_position - ready_robot.joint_position),
                )
                object_position = env.get_scene_state().object_position
                max_object_displacement = np.maximum(
                    max_object_displacement,
                    np.linalg.norm(object_position - initial, axis=1),
                )
                fell = has_fallen(state)
                if fell:
                    fall_stage = "policy"
                if completed % video_period == 0:
                    record_video_frame()
                if completed >= total_steps or env.is_success() or fell:
                    break

        final_images = camera.render()
        save_image(run_dir / "final_left.png", final_images["ego_view_left"])
        final = env.get_scene_state().object_position.copy()
        robot = env.get_robot_state()
        result = {
            "success": env.is_success(),
            "fell": fell,
            "fall_stage": fall_stage,
            "prompt": args.prompt,
            "seed": args.seed,
            "control_steps": completed,
            "sim_seconds": completed / 50,
            "policy_requests": action_chunks,
            "tactile": env.tactile_adapter.metadata,
            "motion_token_range": ([token_min, token_max] if action_chunks else None),
            "final_base_position": robot.base_position.tolist(),
            "final_projected_gravity": projected_gravity(
                robot.base_quaternion
            ).tolist(),
            "base_displacement": float(
                np.linalg.norm(robot.base_position - ready_robot.base_position)
            ),
            "max_left_arm_joint_delta": max_joint_delta[15:22].tolist(),
            "max_right_arm_joint_delta": max_joint_delta[22:29].tolist(),
            "objects": {
                name: {
                    "initial": start.tolist(),
                    "final": end.tolist(),
                    "max_displacement": float(displacement),
                }
                for name, start, end, displacement in zip(
                    OBJECT_NAMES, initial, final, max_object_displacement
                )
            },
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"Results saved to {run_dir}")
    finally:
        if video is not None:
            video.close()
        policy.close()
        camera.close()
        env.close()


if __name__ == "__main__":
    main()
