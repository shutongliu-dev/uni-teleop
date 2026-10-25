"""Small OpenGL window for reconstructed-scene camera frames."""

from __future__ import annotations

import numpy as np


def fit_viewport(
    image_width: int,
    image_height: int,
    window_width: int,
    window_height: int,
) -> tuple[int, int, int, int]:
    """Fit an image inside a window while preserving its aspect ratio."""
    if min(image_width, image_height, window_width, window_height) < 1:
        raise ValueError("image and window dimensions must be positive")
    scale = min(window_width / image_width, window_height / image_height)
    width = max(1, round(image_width * scale))
    height = max(1, round(image_height * scale))
    return (window_width - width) // 2, (window_height - height) // 2, width, height


class DesktopFrameViewer:
    """Display RGB frames in a resizable desktop window."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        title: str = "Reconstructed Lab Camera",
    ) -> None:
        try:
            import glfw
            from OpenGL import GL
        except ImportError as exc:
            raise RuntimeError(
                "desktop reconstructed-scene video requires PyOpenGL"
            ) from exc
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize the desktop display")
        # MuJoCo's offscreen renderer leaves GLFW's global VISIBLE hint disabled.
        # Reset it before creating the user-facing camera window.
        glfw.default_window_hints()
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)
        glfw.window_hint(glfw.FOCUSED, glfw.TRUE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        self._window = glfw.create_window(width, height, title, None, None)
        if self._window is None:
            raise RuntimeError("GLFW could not create the desktop video window")
        self._glfw = glfw
        self._gl = GL
        self._width = width
        self._height = height
        glfw.set_window_pos(self._window, 240, 120)
        glfw.show_window(self._window)
        glfw.focus_window(self._window)
        glfw.request_window_attention(self._window)
        glfw.poll_events()
        glfw.make_context_current(self._window)
        glfw.swap_interval(1)
        self._texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB,
            width,
            height,
            0,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            None,
        )

    def publish(self, frame: np.ndarray) -> bool:
        if self._window is None:
            return False
        if self._glfw.window_should_close(self._window):
            self.close()
            return False
        image = np.ascontiguousarray(frame, dtype=np.uint8)
        if image.shape != (self._height, self._width, 3):
            raise ValueError(
                f"desktop video frame must have shape "
                f"({self._height}, {self._width}, 3), got {image.shape}"
            )
        glfw, GL = self._glfw, self._gl
        glfw.make_context_current(self._window)
        window_width, window_height = glfw.get_framebuffer_size(self._window)
        viewport = fit_viewport(
            self._width,
            self._height,
            window_width,
            window_height,
        )
        GL.glViewport(*viewport)
        GL.glClearColor(0.03, 0.03, 0.03, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            self._width,
            self._height,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            image,
        )
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 1.0)
        GL.glVertex2f(-1.0, -1.0)
        GL.glTexCoord2f(1.0, 1.0)
        GL.glVertex2f(1.0, -1.0)
        GL.glTexCoord2f(1.0, 0.0)
        GL.glVertex2f(1.0, 1.0)
        GL.glTexCoord2f(0.0, 0.0)
        GL.glVertex2f(-1.0, 1.0)
        GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)
        glfw.swap_buffers(self._window)
        glfw.poll_events()
        return True

    def close(self) -> None:
        if self._window is None:
            return
        self._glfw.make_context_current(self._window)
        self._gl.glDeleteTextures([self._texture])
        self._glfw.destroy_window(self._window)
        self._window = None
