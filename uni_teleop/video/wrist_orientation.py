"""Display-only roll alignment of a moving wrist camera."""
import math
import time
import numpy as np


class WristDisplayRotation:
    def __init__(self):
        self.angle = None
        self.last_time = None

    def update(self, camera_world, control_world, now=None):
        now = time.monotonic() if now is None else now
        camera = np.asarray(camera_world)[:3, :3]
        control = np.asarray(control_world)[:3, :3]
        right = camera[:, :2].T @ control[:, 0]
        forward = camera[:, :2].T @ -control[:, 2]
        # Best in-plane rotation: projected control right -> screen right,
        # projected control forward -> screen up. No reflection or image warp.
        cosine = right[0] + forward[1]
        sine = forward[0] - right[1]
        strength = math.hypot(cosine, sine)
        dt = 0. if self.last_time is None else max(0., min(.1, now-self.last_time))
        self.last_time = now
        if strength >= .25:
            target = math.atan2(sine, cosine)
            if self.angle is None:
                self.angle = target
            else:
                delta = math.atan2(math.sin(target-self.angle), math.cos(target-self.angle))
                step = delta * (1.-math.exp(-dt/.20))
                self.angle += max(-math.pi/2*dt, min(math.pi/2*dt, step))
        return -90. if self.angle is None else math.degrees(self.angle)

    def render(self, pixels, camera_world, control_world):
        from PIL import Image
        angle = self.update(camera_world, control_world)
        # Expand instead of clipping fingers; panel fitting preserves the image.
        return np.asarray(Image.fromarray(pixels).rotate(
            angle, resample=Image.Resampling.BICUBIC, expand=True,
            fillcolor=(18, 22, 26))).copy()
