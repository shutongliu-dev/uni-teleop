"""Experimental live SIMPLE teleoperation with HSSD rendering in Isaac Sim."""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
import simple.envs  # noqa: F401
import tyro
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from simple.agents.pico_decoupled_agent import PicoDecoupledAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_id")
    parser.add_argument("--target", default="graspnet1b:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)
    parser.add_argument("--webrtc", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max-episode-steps", type=int, default=30_000)
    parser.add_argument("--render-hz", type=int, default=30)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--dr-level", type=int, default=0)
    parser.add_argument("--success-criteria", type=float, default=2.0)
    parser.add_argument("--save-dir")
    return parser.parse_args()


def load_sonic_config() -> dict:
    config = tyro.cli(
        SimLoopConfig,
        config=(tyro.conf.ConsolidateSubcommandArgs,),
        args=[],
    )
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


def configure_editor_view(simple_env) -> None:
    """Point Isaac's editor camera at the live task workspace."""
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(
        eye=np.array([4.5, -4.5, 3.2]),
        target=np.array([0.0, 0.5, 0.9]),
    )
    simple_env.simulation_app.update()


def main() -> None:
    args = parse_args()
    sonic_config = load_sonic_config()
    print("Starting experimental live HSSD rendering; initial load may take minutes")
    env = gym.make(
        args.env_id,
        sim_mode="mujoco_isaac",
        render_hz=args.render_hz,
        physics_dt=sonic_config["SIMULATE_DT"],
        headless=args.headless,
        webrtc=args.webrtc,
        max_episode_steps=args.max_episode_steps,
        sonic_config=sonic_config,
        target=args.target,
        dr_level=args.dr_level,
        success_criteria=args.success_criteria,
    )
    simple_env = env.unwrapped
    agent = PicoDecoupledAgent(simple_env.task.robot)
    agent.num_episodes = args.num_episodes
    observation, privileged_info = env.reset()
    configure_editor_view(simple_env)
    print("[RichTeleop] HSSD room is loaded and the editor viewport is ready")
    control_dt = 1.0 / args.render_hz

    try:
        while simple_env.simulation_app.is_running():
            started = time.monotonic()
            action = agent.get_action(
                observation,
                instruction=simple_env.task.instruction,
                privileged_info=privileged_info,
            )
            observation, _, _, _, privileged_info = env.step(action)
            if agent.reset_requested:
                observation, privileged_info = env.reset()
                print("[RichTeleop] Environment reset complete")
            agent.update_render_caches(observation)
            delay = control_dt - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("Rich-scene simulator interrupted by user")
    finally:
        agent.close()
        env.close()


if __name__ == "__main__":
    main()
