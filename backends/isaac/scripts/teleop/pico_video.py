"""XRoboToolkit video wire protocol, adapted from sonic-mujoco (no simulator dependency)."""
import argparse
import mmap
import socket
import struct
import threading
from dataclasses import dataclass

FRAME_HEADER = struct.Struct("<4sIIII")


@dataclass(frozen=True, slots=True)
class CameraConfig:
    width: int
    height: int
    fps: int
    bitrate: int
    hevc: bool
    render_mode: int
    port: int
    camera: str
    ip: str


def parse_command(packet: bytes) -> tuple[str, bytes]:
    if len(packet) < 8:
        raise ValueError("camera command is too short")
    command_size = struct.unpack_from("<i", packet)[0]
    data_offset = 4 + command_size
    if command_size < 0 or data_offset + 4 > len(packet):
        raise ValueError("invalid camera command length")
    command = packet[4:data_offset].rstrip(b"\0").decode("ascii")
    data_size = struct.unpack_from("<i", packet, data_offset)[0]
    data = packet[data_offset + 4 :]
    if data_size < 0 or len(data) != data_size:
        raise ValueError("invalid camera data length")
    return command, data


def parse_camera_config(data: bytes) -> CameraConfig:
    if len(data) < 33 or data[:3] != b"\xca\xfe\x01":
        raise ValueError("invalid camera configuration")
    values = struct.unpack_from("<7i", data, 3)
    offset = 31

    def read_string() -> str:
        nonlocal offset
        if offset >= len(data):
            raise ValueError("missing camera string")
        size = data[offset]
        offset += 1
        value = data[offset : offset + size]
        if len(value) != size:
            raise ValueError("truncated camera string")
        offset += size
        return value.decode("utf-8")

    camera, ip = read_string(), read_string()
    width, height, fps, bitrate, hevc, render_mode, port = values
    if not (64 <= width <= 8192 and 64 <= height <= 4320 and 1 <= fps <= 120):
        raise ValueError("unsupported camera dimensions or frame rate")
    if not (1 <= port <= 65535) or not ip:
        raise ValueError("invalid PICO video destination")
    return CameraConfig(
        width, height, fps, bitrate, bool(hevc), render_mode, port, camera, ip
    )


def receive_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def connect_video(ip: str, port: int) -> socket.socket:
    connection = socket.create_connection((ip, port), timeout=3)
    # Bound shutdown if a connected headset stops consuming encoded frames.
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024)
    connection.settimeout(2.0)
    return connection


class VideoBridge:
    def __init__(self, frame_path: str, listen: str) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        Gst.init(None)
        self.GLib, self.Gst = GLib, Gst
        # mmap remains valid only while its backing file stays open.
        self._frame_file = open(frame_path, "rb")  # noqa: SIM115
        self._frames = mmap.mmap(self._frame_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._listen = listen
        self._pipeline = None
        self._timer = None
        self._video_socket = None
        self._lock = threading.Lock()
        self._last_sequence = None
        self._retry_timer = None
        self._requested_config = None

    def run(self) -> None:
        host, port = self._listen.rsplit(":", 1)
        self._server = socket.create_server((host, int(port)), reuse_port=False)
        print(f"PICO video control listening on {host}:{port}", flush=True)
        threading.Thread(target=self._serve, daemon=True).start()
        self.GLib.MainLoop().run()

    def _serve(self) -> None:
        with self._server as server:
            while True:
                connection, _ = server.accept()
                with connection:
                    while True:
                        try:
                            header = receive_exact(connection, 4)
                            if header is None:
                                break
                            size = struct.unpack(">I", header)[0]
                            if size > 1_048_576:
                                break
                            packet = receive_exact(connection, size)
                            if packet is None:
                                break
                        except OSError:
                            break
                        try:
                            command, data = parse_command(packet)
                            if command == "OPEN_CAMERA":
                                config = parse_camera_config(data)
                                self.GLib.idle_add(self._open, config)
                            elif command == "CLOSE_CAMERA":
                                self.GLib.idle_add(self._close)
                        except ValueError as error:
                            print(f"Ignored invalid PICO command: {error}", flush=True)
                self.GLib.idle_add(self._close)

    def _open(self, config: CameraConfig) -> bool:
        self._close()
        self._requested_config = config
        self._last_sequence = None
        header = FRAME_HEADER.unpack(self._frames[: FRAME_HEADER.size])
        _, _, source_width, source_height, _ = header
        codec = "265" if config.hevc else "264"
        bitrate = min(6000, max(100, config.bitrate // 1000))
        codec_options = "vbv-buf-capacity=100 " if codec == "264" else ""
        pipeline_text = (
            f"appsrc name=source is-live=true format=time do-timestamp=true "
            f"max-buffers=1 max-bytes=0 max-time=0 leaky-type=downstream block=false "
            f"caps=video/x-raw,format=RGB,width={source_width},height={source_height},"
            f"framerate={config.fps}/1 ! videoconvert ! videoscale add-borders=true ! "
            f"video/x-raw,format=I420,width={config.width},height={config.height} ! "
            f"x{codec}enc tune=zerolatency speed-preset=ultrafast key-int-max=15 "
            f"bitrate={bitrate} {codec_options}! h{codec}parse config-interval=1 ! "
            f"video/x-h{codec},stream-format=byte-stream,alignment=au ! "
            "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )
        try:
            pipeline = self.Gst.parse_launch(pipeline_text)
            video_socket = connect_video(config.ip, config.port)
        except (OSError, self.GLib.Error) as error:
            print(f"Unable to open PICO video: {error}", flush=True)
            self._schedule_retry()
            return False
        with self._lock:
            self._video_socket = video_socket
        self._pipeline = pipeline
        pipeline.get_by_name("sink").connect("new-sample", self._send_sample, pipeline)
        pipeline.set_state(self.Gst.State.PLAYING)
        self._timer = self.GLib.timeout_add(max(1, 1000 // config.fps), self._push_frame)
        print(
            f"Streaming {config.width}x{config.height}@{config.fps} to "
            f"{config.ip}:{config.port}; {bitrate} kbps, latest-only raw buffer, encode only new source frames",
            flush=True,
        )
        return False

    def _push_frame(self) -> bool:
        if self._pipeline is None:
            return False
        first = FRAME_HEADER.unpack(self._frames[: FRAME_HEADER.size])
        magic, sequence, _, _, size = first
        if magic != b"SMVF" or sequence % 2 or sequence == self._last_sequence:
            return True
        data = self._frames[FRAME_HEADER.size : FRAME_HEADER.size + size]
        if FRAME_HEADER.unpack(self._frames[: FRAME_HEADER.size])[1] != sequence:
            return True
        self._last_sequence = sequence
        buffer = self.Gst.Buffer.new_allocate(None, size, None)
        buffer.fill(0, data)
        self._pipeline.get_by_name("source").emit("push-buffer", buffer)
        return True

    def _send_sample(self, sink, pipeline):
        if pipeline is not self._pipeline:
            return self.Gst.FlowReturn.FLUSHING
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        success, mapping = buffer.map(self.Gst.MapFlags.READ)
        if not success:
            return self.Gst.FlowReturn.ERROR
        try:
            payload = bytes(mapping.data)
            with self._lock:
                if self._video_socket is not None:
                    self._video_socket.sendall(
                        struct.pack(">I", len(payload)) + payload
                    )
        except OSError as error:
            print(f"PICO video connection closed: {error}", flush=True)
            self.GLib.idle_add(self._reconnect, pipeline)
        finally:
            buffer.unmap(mapping)
        return self.Gst.FlowReturn.OK

    def _schedule_retry(self):
        if self._requested_config is not None and self._retry_timer is None:
            self._retry_timer = self.GLib.timeout_add(1000, self._retry)

    def _retry(self):
        self._retry_timer = None
        config = self._requested_config
        if config is not None:
            self._open(config)  # New encoder emits fresh headers and IDR; no old backlog.
        return False

    def _reconnect(self, pipeline):
        if pipeline is self._pipeline and self._requested_config is not None:
            self._stop_stream()
            self._schedule_retry()
        return False

    def _close(self) -> bool:
        self._requested_config = None
        if self._retry_timer is not None:
            self.GLib.source_remove(self._retry_timer)
            self._retry_timer = None
        self._stop_stream()
        return False

    def _stop_stream(self):
        if self._timer is not None:
            self.GLib.source_remove(self._timer)
            self._timer = None
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            pipeline.set_state(self.Gst.State.NULL)
        with self._lock:
            if self._video_socket is not None:
                self._video_socket.close()
                self._video_socket = None
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Send Isaac stereo frames to XRRobotKit")
    parser.add_argument("--frames", required=True)
    parser.add_argument("--listen", default="0.0.0.0:13579")
    args = parser.parse_args()
    VideoBridge(args.frames, args.listen).run()


if __name__ == "__main__":
    main()
