"""Project SuperDex runtime for PICO-driven DG5F hands."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..hand import DEXHAND_DOF, dexhand_controller_targets
from .mapping import DG5F_JOINT_NAMES, dexhand_to_dg5f
from .protocol import DEFAULT_HAND_ENDPOINT, HandCommandSubscriber

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SuperDexHandsConfig:
    assets_root: Path
    endpoint: str = DEFAULT_HAND_ENDPOINT
    timestep: float = 0.005
    sides: tuple[str, ...] = ("left", "right")
    kp: float = 3.0
    kd: float = 0.2
    effort_limit: float = 2.0

    def __post_init__(self) -> None:
        assets_root = self.assets_root.expanduser().resolve()
        if not assets_root.is_dir():
            raise ValueError(f"SuperDex assets not found: {assets_root}")
        if not self.sides or any(side not in {"left", "right"} for side in self.sides):
            raise ValueError("sides must contain 'left', 'right', or both")
        if self.timestep <= 0.0:
            raise ValueError("timestep must be positive")
        if min(self.kp, self.kd, self.effort_limit) < 0.0:
            raise ValueError("controller gains and effort limit must be non-negative")
        object.__setattr__(self, "assets_root", assets_root)


@dataclass(slots=True)
class _ControlledHand:
    side: str
    bot: Any
    actor: Any
    controller: Any
    dof_indices: NDArray[np.int32]


class SuperDexHandsRuntime:
    """Run one or two fixed DG5F hands in the SuperDex contact engine."""

    def __init__(self, config: SuperDexHandsConfig) -> None:
        os.environ["SUPERDEX_ASSETS_PATH"] = str(config.assets_root)
        try:
            from superdex import physics, robotics
        except ImportError as error:
            raise RuntimeError(
                "SuperDex is unavailable; run scripts/setup_superdex.py first"
            ) from error

        self.config = config
        self.physics = physics
        self.robotics = robotics
        self.real_dtype = np.float64 if physics.uses_double_precision() else np.float32
        physics.initialize(num_worker_threads=-1)
        self.scene = physics.create_scene("SONIC PICO DG5F Teleoperation")
        self.scene.set_gravity([0.0, 0.0, -9.81])
        self.robotics_context = robotics.create_context()
        self.hands = [self._create_hand(side) for side in config.sides]
        self._closed = False

        physics.get_debug_server().set_coordinate_space(
            physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
        )

    def _create_hand(self, side: str) -> _ControlledHand:
        asset = (
            self.config.assets_root
            / "bots"
            / "hands"
            / "dg5f_long"
            / side
            / f"dg5f_long_{side}.superdex_bot"
        )
        if not asset.is_file():
            raise RuntimeError(f"DG5F {side} asset not found: {asset}")
        prefab = self.robotics.load_bot_prefab_from_file(str(asset))
        prefab.joints[0].type = self.physics.ArticulatedJointType.HARD
        lateral = 0.16 if side == "left" else -0.16
        prefab.world_from_root = self.physics.TransformRT(
            translation=[0.0, lateral, 0.22]
        )
        prefab.default_pose = dexhand_to_dg5f(np.zeros(20), side).astype(
            self.real_dtype
        )
        for index in range(len(prefab.links)):
            prefab.links[index].has_gravity = False

        moving_names = tuple(
            joint.name
            for joint in prefab.joints
            if joint.type == self.physics.ArticulatedJointType.REVOLUTE
        )
        if moving_names != DG5F_JOINT_NAMES:
            raise RuntimeError(f"unexpected DG5F {side} joint order: {moving_names}")

        bot = self.robotics.create_bot(
            self.scene,
            prefab,
            self.robotics_context,
        )
        actor = bot.get_articulated_actor()
        if actor.get_num_dofs() != len(DG5F_JOINT_NAMES):
            raise RuntimeError(
                f"DG5F {side} has {actor.get_num_dofs()} DoFs, expected 20"
            )
        controller = bot.create_controller("BASIC_JSC_PD")
        params = self.robotics.ControllerBasicJscPdParams()
        params.kp = np.full(20, self.config.kp, dtype=np.float32)
        params.kd = np.full(20, self.config.kd, dtype=np.float32)
        params.saturation = np.full(20, self.config.effort_limit, dtype=np.float32)
        params.deadband = np.zeros(20, dtype=np.float32)
        controller.set_params(params)
        return _ControlledHand(
            side,
            bot,
            actor,
            controller,
            np.arange(20, dtype=np.int32),
        )

    def step(self, target: Array) -> None:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != (DEXHAND_DOF,) or not np.isfinite(value).all():
            raise ValueError(
                f"target must be a finite vector of shape ({DEXHAND_DOF},)"
            )
        for hand in self.hands:
            offset = 0 if hand.side == "left" else 20
            target_pose = dexhand_to_dg5f(
                value[offset : offset + 20], hand.side
            ).astype(self.real_dtype)
            observation = hand.controller.get_current_observations_from_mochi()
            observation.dt = self.config.timestep
            torque = np.asarray(
                hand.controller.compute_output(
                    observation,
                    self.robotics.ControllerBasicJscPdTarget(target_pose=target_pose),
                ),
                dtype=np.float32,
            )
            hand.actor.set_external_forces_on_dofs(
                dof_indices=hand.dof_indices,
                force_values=torque,
            )
        self.scene.step(self.config.timestep)

    def attach_debugger(self) -> bool:
        return bool(self.physics.debugger.attach())

    @property
    def debugger_attached(self) -> bool:
        return bool(self.physics.debugger.is_attached())

    def close(self) -> None:
        if self._closed:
            return
        for hand in reversed(self.hands):
            self.robotics.destroy_bot(self.scene, hand.bot)
        self.physics.shutdown()
        self._closed = True


def run_superdex_hands(
    config: SuperDexHandsConfig,
    *,
    headless: bool = False,
    steps: int = 0,
    demo: bool = False,
) -> int:
    """Run the DG5F bridge until the debugger closes or ``steps`` expires."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    runtime = SuperDexHandsRuntime(config)
    subscriber = HandCommandSubscriber(config.endpoint)
    target = np.zeros(DEXHAND_DOF, dtype=np.float64)
    received = False
    completed = 0
    try:
        if not headless and not runtime.attach_debugger():
            raise RuntimeError("unable to attach the SuperDex Physics Debugger")
        while steps == 0 or completed < steps:
            if not headless and not runtime.debugger_attached:
                break
            tick = time.monotonic()
            command = subscriber.receive()
            if command is not None:
                target = command.joint_position
                if not received:
                    print("SuperDex is receiving PICO five-finger targets.")
                    received = True
            elif demo:
                phase = 0.5 - 0.5 * np.cos(2.0 * np.pi * completed / 400.0)
                target = dexhand_controller_targets(
                    phase,
                    phase,
                    phase,
                    phase,
                )
            runtime.step(target)
            completed += 1
            time.sleep(max(0.0, config.timestep - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.close()
        runtime.close()
    return completed
