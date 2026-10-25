from collections.abc import Callable, Sequence

import numpy as np

from ..contact import ContactFrame

HAPTIC_PERIOD = 0.05
IMPACT_HAPTIC = (120, 180)
PRESS_HAPTIC = (80, 130)
SLIDE_HAPTIC = (45, 90)


class ContactHaptics:
    """Convert upper-body contacts into rate-limited controller vibration."""

    def __init__(
        self,
        body_names: Sequence[str],
        send: Callable[[float, float, int, int], bool],
    ) -> None:
        self._side = np.zeros(len(body_names), dtype=np.int8)
        for body_id, name in enumerate(body_names):
            if name.startswith(
                ("left_shoulder", "left_elbow", "left_wrist", "left_hand")
            ):
                self._side[body_id] = 1
            elif name.startswith(
                ("right_shoulder", "right_elbow", "right_wrist", "right_hand")
            ):
                self._side[body_id] = 2
            elif name == "pelvis" or name.startswith(("waist_", "torso_")):
                self._side[body_id] = 3
        self._send = send
        self._last_send = -HAPTIC_PERIOD
        self._active = np.zeros(2, dtype=bool)

    def update(self, contacts: ContactFrame, now: float) -> bool:
        impulse = np.zeros(2)
        normal_force = np.zeros(2)
        tangent_force = np.zeros(2)
        for index in range(min(contacts.count, len(contacts.robot_body_id))):
            body_id = contacts.robot_body_id[index]
            if 0 <= body_id < len(self._side) and self._side[body_id]:
                sides = (0, 1) if self._side[body_id] == 3 else (self._side[body_id] - 1,)
                for side in sides:
                    impulse[side] += contacts.normal_impulse[index]
                    normal_force[side] += contacts.normal_force[index]
                    tangent_force[side] += contacts.tangent_force[index]

        amplitude = np.sqrt(np.clip((impulse - 0.01) / 0.30, 0.0, 1.0))
        active = amplitude > 0.0
        impact = active & ~self._active
        sliding = active & (tangent_force > 0.30 * normal_force)
        self._active = active
        if not np.any(active) or now - self._last_send < HAPTIC_PERIOD:
            return True

        amplitude[sliding & ~impact] *= 0.65
        if np.any(impact):
            duration, frequency = IMPACT_HAPTIC
        elif np.any(sliding):
            duration, frequency = SLIDE_HAPTIC
        else:
            duration, frequency = PRESS_HAPTIC
        self._last_send = now
        return self._send(
            float(amplitude[0]),
            float(amplitude[1]),
            duration,
            frequency,
        )
