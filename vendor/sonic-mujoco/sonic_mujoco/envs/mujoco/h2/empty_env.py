from pathlib import Path

from .h2_env import MujocoH2Env


class MujocoH2EmptyEnv(MujocoH2Env):
    """H2 in its nominal home pose in the shared empty MuJoCo scene."""

    def __init__(self, timestep: float = 0.002) -> None:
        package_root = Path(__file__).resolve().parents[3]
        scene = package_root / "assets" / "mujoco" / "scenes" / "h2" / "empty.xml"
        super().__init__(scene, timestep)
