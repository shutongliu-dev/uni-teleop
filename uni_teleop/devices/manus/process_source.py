"""Isolated MANUS SDK producer for robot teleoperation.

The vendor SDK stays in its own Python process. This module only owns the
JSONL boundary and never imports native MANUS libraries into Isaac/Kit.
"""
from __future__ import annotations
from dataclasses import dataclass
import json, os, queue, subprocess, threading, time
from pathlib import Path

@dataclass(frozen=True)
class ManusSourceConfig:
    sdk_python: Path
    python_executable: Path
    glove_id: int
    calibration_id: str
    sdk_version: str = "3.2.1"
    stderr_log: Path | None = None
    mode: str = "integrated"
    duration_s: float = 86400.0
    max_line_bytes: int = 2 * 1024 * 1024

class ManusProcessSource:
    def __init__(self, config: ManusSourceConfig):
        if config.mode not in ("integrated", "remote", "local"):
            raise ValueError("invalid MANUS SDK mode")
        if not 0 <= config.glove_id <= 0xFFFFFFFF:
            raise ValueError("glove_id must be uint32")
        if not config.calibration_id:
            raise ValueError("calibration_id is required")
        self.config = config
        self.process = None
        self._thread = None
        self._stderr = None
        self._stop = threading.Event()
        self._frames = queue.Queue(maxsize=8)
        self.latest = None
        self.error = None
        self.received = 0
        self.dropped = 0

    def start(self):
        if self.process is not None:
            raise RuntimeError("MANUS source already started")
        env = os.environ.copy()
        sdk_root = str(self.config.sdk_python.resolve())
        env["PYTHONPATH"] = sdk_root + os.pathsep + env.get("PYTHONPATH", "")
        env["LD_LIBRARY_PATH"] = str(self.config.sdk_python / "manus_sdk" / "linux" / "amd64") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        self._stderr = self.config.stderr_log.open("a") if self.config.stderr_log else None
        self.process = subprocess.Popen([
            str(self.config.python_executable), "-m", "uni_teleop.devices.manus.sdk_process",
            "--sdk-python", sdk_root, "--sdk-version", self.config.sdk_version, "--mode", self.config.mode,
            "--glove-id", str(self.config.glove_id),
            "--calibration-id", self.config.calibration_id,
            "--duration", str(self.config.duration_s),
            "--parent-pid", str(os.getpid()),
        ], stdout=subprocess.PIPE, stderr=self._stderr or subprocess.DEVNULL, text=True, env=env, bufsize=1)
        self._thread = threading.Thread(target=self._read, name="manus-jsonl", daemon=True)
        self._thread.start()

    def _read(self):
        assert self.process and self.process.stdout
        try:
            for line in self.process.stdout:
                if self._stop.is_set():
                    break
                if len(line.encode()) > self.config.max_line_bytes:
                    self.error = "MANUS JSONL frame exceeds size limit"
                    break
                try:
                    packet = json.loads(line)
                    if packet.get("protocol") != "manus_acquisition_v1":
                        continue
                    if packet.get("device", {}).get("device_id") != str(self.config.glove_id):
                        continue
                    self.latest = packet
                    self.received += 1
                    try: self._frames.put_nowait(packet)
                    except queue.Full:
                        self.dropped += 1
                        self._frames.get_nowait(); self._frames.put_nowait(packet)
                except (ValueError, TypeError) as exc:
                    self.error = f"invalid MANUS frame: {exc}"
        finally:
            if self.process and self.process.poll() not in (None, 0):
                self.error = self.error or f"MANUS producer exited with code {self.process.returncode}"

    def read(self):
        try: return self._frames.get_nowait()
        except queue.Empty: return None

    def health(self):
        return {"running": self.process is not None and self.process.poll() is None,
                "received": self.received, "dropped": self.dropped,
                "latest": self.latest is not None, "error": self.error,
                "latest_age_s": (time.monotonic_ns()-self.latest["capture_time_ns"])/1e9 if self.latest else None}

    def close(self):
        self._stop.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
        if self._thread: self._thread.join(timeout=3)
        if self._stderr: self._stderr.close()
        self.process = None

    def __enter__(self): self.start(); return self
    def __exit__(self, *_): self.close()
