import numpy as np

TOKEN_DIM = 64
HISTORY_FRAMES = 10
OBSERVATION_DIM = 994
CONTROL_TIMESTEP = 0.02
PHYSICS_STEPS_PER_CONTROL = 4

# Target order in source-order indices. These are the two arrays used by the
# reference C++ deploy, with names that state the direction explicitly.
ISAACLAB_FROM_HARDWARE = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
     17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)
HARDWARE_FROM_ISAACLAB = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21,
     23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)

DEFAULT_ANGLES = np.array(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     0.0, 0.0, 0.0,
     0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
     0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0]
)

_NATURAL_FREQUENCY = 10.0 * 2.0 * 3.1415926535
_ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_MOTOR_TYPES = (
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_14", "5020", "5020",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
)


def _stiffness(motor_type: str) -> float:
    return _ARMATURE[motor_type] * _NATURAL_FREQUENCY**2


ACTION_SCALE = np.array(
    [0.25 * _EFFORT[kind] / _stiffness(kind) for kind in _MOTOR_TYPES]
)
KP = np.array(
    [2.0 * _stiffness(kind) if 4 <= i <= 5 or 10 <= i <= 11 or 13 <= i <= 14
     else _stiffness(kind) for i, kind in enumerate(_MOTOR_TYPES)],
    dtype=np.float32,
).astype(np.float64)
KD = np.array(
    [2.0 * 2.0 * _ARMATURE[kind] * _NATURAL_FREQUENCY *
     (2.0 if 4 <= i <= 5 or 10 <= i <= 11 or 13 <= i <= 14 else 1.0)
     for i, kind in enumerate(_MOTOR_TYPES)],
    dtype=np.float32,
).astype(np.float64)
