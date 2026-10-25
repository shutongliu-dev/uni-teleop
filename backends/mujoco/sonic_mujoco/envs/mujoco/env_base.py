import time
from pathlib import Path

import mujoco
import mujoco.viewer


class MujocoEnvBase:
    """Own a MuJoCo model, data, and optional passive viewer."""

    def __init__(
        self,
        xml_path: str | Path,
        timestep: float = 0.005,
        *,
        model: mujoco.MjModel | None = None,
    ) -> None:
        self.xml_path = Path(xml_path).resolve()
        self.model = model or mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self._viewer = None

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def viewer_running(self) -> bool:
        return self._viewer is None or self._viewer.is_running()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, steps: int = 1) -> None:
        if steps < 1:
            raise ValueError("steps must be positive")
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def render(self) -> None:
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            time.sleep(0.1)  # Let GLFW release X11 before process exit.
