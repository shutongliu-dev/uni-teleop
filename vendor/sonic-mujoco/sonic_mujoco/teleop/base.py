from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _array(name: str, value: Array, shape: tuple[int | None, ...]) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        text = ", ".join("N" if size is None else str(size) for size in shape)
        raise ValueError(f"{name} must have shape ({text})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


@dataclass(frozen=True, slots=True)
class TeleopCommand:
    """A batch of reference poses on one monotonically increasing timeline."""

    frame_index: NDArray[np.int64]
    smpl_joints: Array
    root_quaternion: Array
    joint_position: Array
    heading_increment: float = 0.0
    hand_joint_position: Array | None = None
    neck_joint_position: Array | None = None

    def __post_init__(self) -> None:
        frame_index = np.asarray(self.frame_index, dtype=np.int64)
        if frame_index.ndim != 1 or frame_index.size == 0:
            raise ValueError("frame_index must be a non-empty vector")
        if np.any(np.diff(frame_index) <= 0):
            raise ValueError("frame_index must be strictly increasing")

        frames = frame_index.size
        object.__setattr__(self, "frame_index", frame_index.copy())
        object.__setattr__(
            self,
            "smpl_joints",
            _array("smpl_joints", self.smpl_joints, (frames, 24, 3)),
        )
        object.__setattr__(
            self,
            "root_quaternion",
            _array("root_quaternion", self.root_quaternion, (frames, 4)),
        )
        object.__setattr__(
            self,
            "joint_position",
            _array("joint_position", self.joint_position, (frames, 29)),
        )
        if self.hand_joint_position is not None:
            object.__setattr__(
                self,
                "hand_joint_position",
                _array(
                    "hand_joint_position",
                    self.hand_joint_position,
                    (frames, 40),
                ),
            )
        if self.neck_joint_position is not None:
            object.__setattr__(
                self,
                "neck_joint_position",
                _array(
                    "neck_joint_position",
                    self.neck_joint_position,
                    (frames, 2),
                ),
            )
        heading_increment = float(self.heading_increment)
        if not np.isfinite(heading_increment):
            raise ValueError("heading_increment must be finite")
        object.__setattr__(self, "heading_increment", heading_increment)


class TeleopBase(ABC):
    @abstractmethod
    def read(self) -> TeleopCommand | None:
        """Return the next command batch, or None when no message is ready."""

    def close(self) -> None:
        pass
