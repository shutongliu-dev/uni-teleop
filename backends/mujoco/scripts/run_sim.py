import argparse
import time

import numpy as np

from sonic_mujoco.envs.mujoco.g1 import (
    RobotCommand,
)
from sonic_mujoco.envs.mujoco.g1.registry import G1_ENVIRONMENTS
from sonic_mujoco.envs.mujoco.h2 import MujocoH2EmptyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a supported MuJoCo robot")
    parser.add_argument("--robot", choices=("g1", "h2"), default="g1")
    parser.add_argument(
        "--scene",
        choices=tuple(G1_ENVIRONMENTS),
        default="empty",
    )
    parser.add_argument(
        "--headless", action="store_true", help="do not open the viewer"
    )
    parser.add_argument(
        "--steps", type=int, default=0, help="stop after this many steps"
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="inspect the initial scene without advancing physics",
    )
    parser.add_argument(
        "--start-task",
        action="store_true",
        help="start scripted human interactions (pose hold is not a task policy)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.robot == "h2":
        if args.scene != "empty":
            raise SystemExit("H2 currently supports --scene empty only")
        env = MujocoH2EmptyEnv()
    else:
        env = G1_ENVIRONMENTS[args.scene]()
    env.reset(seed=args.seed)
    if args.start_task and hasattr(env, "start_task"):
        env.start_task()
    state = env.get_robot_state()
    if args.robot == "h2":
        command = env.home_command()
    else:
        zeros = np.zeros_like(state.joint_position)
        command = RobotCommand(
            joint_position=state.joint_position,
            joint_velocity=zeros,
            feedforward_torque=zeros,
            kp=np.full_like(state.joint_position, 20.0),
            kd=np.full_like(state.joint_position, 1.0),
        )

    print(command)

    try:
        if args.headless:
            if not args.paused:
                env.step(command, args.steps or 1)
            print(f"{args.robot.upper()} simulation OK: time={env.time:.3f}s")
            return

        env.render()
        completed = 0
        while env.viewer_running and (args.steps == 0 or completed < args.steps):
            started = time.monotonic()
            if not args.paused:
                env.step(command)
            env.render()
            completed += 1
            time.sleep(max(0.0, env.timestep - (time.monotonic() - started)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
