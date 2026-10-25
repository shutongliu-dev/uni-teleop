"""Record guided raw sleeve calibration using the existing JuQiao driver.

Run with the real-robot collection environment (pyserial, numpy, OpenCV).
The supplied driver root must contain jq_tactile_skin/protocol.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PRESS_RELEASE_STAGES = (
    (0.0, 5.0, "release"),
    (5.0, 8.0, "press"),
    (8.0, 13.0, "release"),
    (13.0, 16.0, "press"),
    (16.0, 21.0, "release"),
    (21.0, 24.0, "press"),
    (24.0, 30.0, "release"),
)


class SleeveCapture:
    """A finite, explicitly triggered capture; timestamps are host-side only."""

    def __init__(self, output: Path, seconds: float, metadata: dict) -> None:
        self.output = output
        self.seconds = seconds
        self.duration = seconds
        self.kind = "quiet"
        self.metadata = metadata
        self.stream = None
        self.directory = None
        self.started = 0.0
        self.frames = 0
        self.status = "READY: B = quiet baseline | P = press/release"

    def start(self, now: float, kind: str = "quiet") -> None:
        if self.stream is not None:
            return
        if kind not in ("quiet", "press_release"):
            raise ValueError(f"Unknown capture kind: {kind}")
        self.kind = kind
        self.duration = 30.0 if kind == "press_release" else self.seconds
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.directory = self.output / f"right_sleeve_{kind}_{stamp}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / "samples.jsonl").open("x", encoding="utf-8")
        self.started = now
        self.frames = 0
        self._metadata("recording")
        self.status = "RECORDING"
        print(f"Recording {kind} for {self.duration:g}s -> {self.directory}", flush=True)

    def stage(self, elapsed: float) -> tuple[str, float]:
        if self.kind == "press_release":
            for start, end, name in PRESS_RELEASE_STAGES:
                if start <= elapsed < end:
                    return name, end - elapsed
        return "release", max(0.0, self.duration - elapsed)

    def add(self, sample, received_wall: float, received_mono: float) -> None:
        if self.stream is None or received_mono - self.started >= self.duration:
            return
        row = {
            "frame_index": self.frames,
            "host_receive_unix_s": received_wall,
            "host_receive_monotonic_s": received_mono,
            "elapsed_s": received_mono - self.started,
            "instructed_phase": self.stage(received_mono - self.started)[0],
            "packet1_host_unix_s": sample.packet1_time,
            "packet2_host_unix_s": sample.packet2_time,
            "sensor_type": sample.sensor_type,
            "raw": list(sample.raw),
            "gyro_bytes": list(sample.gyro),
        }
        self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.frames += 1
        if self.frames % 100 == 0:
            self.stream.flush()

    def update(self, now: float) -> None:
        if self.stream is not None and now - self.started >= self.duration:
            self.finish("complete" if self.frames else "no_data")

    def finish(self, status: str) -> None:
        if self.stream is None:
            return
        self.stream.close()
        self.stream = None
        self._metadata(status)
        self.status = f"{status.upper()}: saved {self.frames} frames; B/P = new trial"
        print(f"{self.status} -> {self.directory}", flush=True)

    def _metadata(self, status: str) -> None:
        metadata = {
            **self.metadata,
            "schema_version": 2,
            "status": status,
            "device": "right_arm",
            "capture_kind": self.kind,
            "requested_duration_s": self.duration,
            "frames": self.frames,
            "started_host_monotonic_s": self.started,
            "instruction": (
                "repeat gentle fingerpad press and release at the same central sensing location"
                if self.kind == "press_release" else "leave sleeve untouched and stationary"
            ),
            "instructed_stages": PRESS_RELEASE_STAGES if self.kind == "press_release" else [],
            "phase_note": "scheduled screen prompts, not measured contact onset or force ground truth",
            "quiet_condition_verified": False,
            "mount_condition": self.metadata.get("mount_condition", "not_yet_confirmed"),
            "baseline_subtracted": False,
            "raw_order": "wire channels 1..256",
            "display_order": "wire channels 129..256 then 1..128",
            "source_timestamp_available": False,
            "source_sequence_available": False,
            "timestamp_note": "host read times, not sensor sample times; buffered packets can share a timestamp",
            "force_calibrated": False,
        }
        (self.directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-root", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--mount", default="not_yet_confirmed",
                        choices=("not_yet_confirmed", "flat_on_table", "worn_on_arm", "on_fixture"))
    args = parser.parse_args()
    if not 0 < args.seconds <= 600:
        parser.error("--seconds must be between 0 and 600")
    if not (args.driver_root / "jq_tactile_skin/protocol.py").is_file():
        parser.error("--driver-root must contain jq_tactile_skin/protocol.py")

    import cv2
    import numpy as np
    import serial

    sys.path.insert(0, str(args.driver_root.resolve()))
    from jq_tactile_skin.protocol import FrameParser, SampleAssembler

    capture = SleeveCapture(
        args.output, args.seconds,
        {"port": args.port, "baud": args.baud, "driver_root": str(args.driver_root),
         "mount_condition": args.mount},
    )
    frame_parser, assembler = FrameParser(), SampleAssembler()
    order = np.r_[128:256, 0:128]
    raw = np.zeros(256, dtype=np.uint8)
    last = None
    frames = previous = 0
    rate = 0.0
    report = time.monotonic()
    title = "Right sleeve - RAW calibration"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 850, 760)
    try:
        with serial.Serial(args.port, args.baud, timeout=0.01, exclusive=True) as ser:
            print("Ready. B = quiet baseline; P = 30s guided press/release; Q = stop. No device commands sent.", flush=True)
            while True:
                chunk = ser.read(min(ser.in_waiting or 1, 65536))
                now, wall = time.monotonic(), time.time()
                for packet in frame_parser.feed(chunk):
                    if packet.sensor_type != 0x02:
                        raise ValueError(f"Expected RIGHT_ARM (0x02), received {packet.sensor_type:#x}")
                    sample = assembler.add_packet(packet, wall)
                    if sample is None:
                        continue
                    raw = np.frombuffer(sample.raw, dtype=np.uint8)
                    frames += 1
                    last = now
                    capture.add(sample, wall, now)
                capture.update(now)
                if now - report >= 1.0:
                    rate = (frames - previous) / (now - report)
                    report, previous = now, frames
                canvas = np.full((760, 850, 3), 25, dtype=np.uint8)
                grid = raw[order].reshape(16, 16)
                # Fixed display scale: no frame-dependent amplification or subtraction.
                heat = cv2.applyColorMap(grid, cv2.COLORMAP_INFERNO)
                canvas[150:662, 169:681] = cv2.resize(heat, (512, 512), interpolation=cv2.INTER_NEAREST)
                for i in range(17):
                    cv2.line(canvas, (169 + i * 32, 150), (169 + i * 32, 662), (70, 70, 70))
                    cv2.line(canvas, (169, 150 + i * 32), (681, 150 + i * 32), (70, 70, 70))
                live = last is not None and now - last < 0.5
                lines = [
                    "RIGHT ARM | RAW ADC 0..255 | fixed color scale",
                    f"{'LIVE' if live else 'STALE / NO DATA'} | host arrival {rate:.1f} Hz | min {raw.min()} max {raw.max()}",
                    capture.status,
                    f"Remaining: {max(0, capture.duration - (now - capture.started)):.1f}s | frames {capture.frames}"
                    if capture.stream is not None else "B: quiet baseline | P: press/release | Q/Esc: close",
                ]
                if capture.stream is not None:
                    phase, remaining = capture.stage(now - capture.started)
                    cue = "PRESS gently, same spot" if phase == "press" else "RELEASE - hands off sleeve"
                    lines[2] = f"{cue} | next cue in {remaining:.1f}s"
                for y, line in zip((28, 58, 91, 124), lines):
                    cv2.putText(canvas, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (230, 230, 230), 1, cv2.LINE_AA)
                cv2.putText(canvas, "Top: channels 129..256 | Bottom: 1..128", (130, 704), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
                if capture.stream is not None:
                    color = (0, 180, 255) if phase == "press" else (80, 210, 100)
                    cv2.rectangle(canvas, (3, 3), (846, 756), color, 5)
                cv2.imshow(title, canvas)
                key = cv2.waitKey(1) & 255
                if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("b"), ord("B"), ord("p"), ord("P")) and live and capture.stream is None:
                    # Discard incomplete pairing and queued pre-trigger bytes.
                    ser.reset_input_buffer()
                    frame_parser, assembler = FrameParser(), SampleAssembler()
                    kind = "press_release" if key in (ord("p"), ord("P")) else "quiet"
                    capture.start(time.monotonic(), kind)
    finally:
        capture.finish("interrupted")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
