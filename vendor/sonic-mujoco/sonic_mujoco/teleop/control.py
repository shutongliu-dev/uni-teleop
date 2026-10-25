from enum import Enum

from .pico_direct import PicoEvents


class TeleopMode(Enum):
    OFF = "off"
    READY = "ready"
    POSE = "pose"


def next_mode(mode: TeleopMode, events: PicoEvents) -> TeleopMode:
    if events.start_stop:
        return TeleopMode.READY if mode is TeleopMode.OFF else TeleopMode.OFF
    if events.toggle_pose and mode is not TeleopMode.OFF:
        return TeleopMode.READY if mode is TeleopMode.POSE else TeleopMode.POSE
    return mode
