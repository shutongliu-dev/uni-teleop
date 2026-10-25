from pathlib import Path

from .g1_env import MujocoG1Env


class MujocoG1EmptyEnv(MujocoG1Env):
    """G1 in the shared empty scene."""

    def __init__(self, timestep: float = 0.005) -> None:
        package_root = Path(__file__).resolve().parents[3]
        scene = package_root / "assets" / "mujoco" / "scenes" / "g1" / "empty.xml"
        super().__init__(scene, timestep)
