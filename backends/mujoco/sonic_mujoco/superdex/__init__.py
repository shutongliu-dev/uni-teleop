"""Optional bridge from SONIC/PICO commands to Project SuperDex."""

from .mapping import DG5F_JOINT_NAMES, dexhand_to_dg5f
from .protocol import (
    DEFAULT_HAND_ENDPOINT,
    HandCommandPublisher,
    HandCommandSubscriber,
    SuperDexHandCommand,
)

__all__ = [
    "DEFAULT_HAND_ENDPOINT",
    "DG5F_JOINT_NAMES",
    "HandCommandPublisher",
    "HandCommandSubscriber",
    "SuperDexHandCommand",
    "dexhand_to_dg5f",
]
