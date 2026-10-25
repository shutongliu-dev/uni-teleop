"""Public HTTP-only bench access, with no simulator or private-data imports."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import math
import re
import time
from urllib.parse import urlencode, urlsplit

import httpx
from PIL import Image

from bench.sdk.depth import MAX_DEPTH_PIXELS, MAX_NPY_HEADER_BYTES, decode_metric_depth_npy
from .actions import ActionValidationError, public_observation
from .time_limits import minimum_limit, optional_positive


class BenchHTTPError(RuntimeError):
    """Sanitized error: never includes response bodies, headers, or credentials."""
    def __init__(self, message: str, *, status_code: int | None = None,
                 category: str | None = None):
        self.status_code = status_code
        self.category = category or ("http_status" if status_code is not None else "unknown")
        super().__init__(message if status_code is None else f"{message} (HTTP {status_code})")

    def public_metadata(self):
        return {"error_category": self.category, "http_status": self.status_code}


def _token(value: str) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}", value)
            or value in (".", "..")):
        raise BenchHTTPError("invalid public endpoint identifier")
    return value


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _task_query(task_id):
    if task_id is None:
        return ''
    # Task IDs are public Name values, not path components. Encode separators
    # as query data so namespaced tasks retain their complete identity.
    if not isinstance(task_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,160}', task_id):
        raise BenchHTTPError('invalid public task identifier')
    return '?' + urlencode({'task_id': task_id})


class BenchHTTPClient:
    def __init__(self, base_url: str, *, client: httpx.Client | None = None,
                 timeout_s: float | None = 30.0, get_retries: int = 2):
        try:
            url = urlsplit(base_url)
            valid = (url.scheme in ("http", "https") and url.hostname and url.port != 0
                and url.username is None and url.password is None and not url.query
                and not url.fragment and url.path in ("", "/"))
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError("bench base URL must be an HTTP(S) origin without credentials or path")
        self.base_url = f"{url.scheme}://{url.netloc}"
        self.timeout_s = optional_positive(timeout_s, "timeout_s")
        if isinstance(get_retries, bool) or not isinstance(get_retries, int) or not 0 <= get_retries <= 5:
            raise ValueError("get_retries must be an integer between 0 and 5")
        self.get_retries = get_retries
        self.deadline = None  # Optional absolute host deadline for one trusted operation.
        self.client = client if client is not None else httpx.Client(trust_env=False, follow_redirects=False)
        self._owns_client = client is None

    def close(self):
        if self._owns_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _request_headers(self, path):
        """Per-request transport metadata; subclasses may bind player credentials."""
        return {}

    def _json(self, method, path, *, payload=None, allow_409=False, deadline=None):
        local_deadline = (None if self.timeout_s is None else time.monotonic()+self.timeout_s)
        deadline = minimum_limit(deadline if deadline is not None else local_deadline,
                                 self.deadline)
        attempts = self.get_retries + 1 if method == "GET" else 1
        for attempt in range(attempts):
            remaining = minimum_limit(self.timeout_s,
                                      None if deadline is None else deadline-time.monotonic())
            if remaining is not None and remaining <= 0:
                raise BenchHTTPError("public request deadline expired", category="request_budget_exhausted")
            try:
                reply = self.client.request(method, self.base_url + path, json=payload,
                                            headers=self._request_headers(path),
                                            timeout=remaining, follow_redirects=False)
            except httpx.TransportError as error:
                if attempt + 1 < attempts:
                    continue
                raise BenchHTTPError("public bench transport failed", category=(
                    "transport_timeout" if isinstance(error, httpx.TimeoutException) else "transport_error")) from None
            if allow_409 and reply.status_code == 409:
                return None
            if reply.status_code in (502, 503, 504) and attempt + 1 < attempts:
                continue
            if not 200 <= reply.status_code < 300:
                raise BenchHTTPError("public bench request failed", status_code=reply.status_code)
            try:
                result = reply.json()
            except (ValueError, UnicodeError):
                raise BenchHTTPError("public bench returned invalid JSON", status_code=reply.status_code, category="invalid_response") from None
            if not isinstance(result, (dict, list)):
                raise BenchHTTPError("public bench returned an invalid response shape", status_code=reply.status_code, category="invalid_response")
            return result
        raise AssertionError("unreachable request loop")

    def list_tasks(self):
        return self._json("GET", "/tasks")

    def list_robots(self):
        return self._json("GET", "/robots")

    def list_policies(self, task_id: str | None = None):
        return self._json("GET", "/policies" + _task_query(task_id))

    def list_interfaces(self, task_id: str | None = None):
        return self._json("GET", "/interfaces" + _task_query(task_id))

    def create(self, payload: dict):
        return self._json("POST", "/episodes", payload=payload)

    def get_observation(self, episode_id: str):
        return self._json("GET", f"/episodes/{_token(episode_id)}/observation")

    def get_result(self, episode_id: str):
        return self._json("GET", f"/episodes/{_token(episode_id)}/result", allow_409=True)

    def submit(self, episode_id: str, payload: dict):
        return self._json("POST", f"/episodes/{_token(episode_id)}/actions", payload=payload)

    @staticmethod
    def _action_status(result, action_id):
        if (not isinstance(result, dict) or result.get("action_id") != action_id
                or result.get("status") not in ("accepted", "running", "completed", "cancelled", "failed")):
            raise BenchHTTPError("public action status is invalid", category="invalid_response")
        return result

    def get_action(self, episode_id: str, action_id: str, *, deadline=None):
        """Read one action status; the bounded GET transport policy still applies."""
        path = f"/episodes/{_token(episode_id)}/actions/{_token(action_id)}"
        return self._action_status(self._json("GET", path, deadline=deadline), action_id)

    def cancel_action(self, episode_id: str, action_id: str):
        """Request cancellation once. Its acknowledgement is not proof of stopping."""
        path = f"/episodes/{_token(episode_id)}/actions/{_token(action_id)}/cancel"
        return self._action_status(self._json("POST", path), action_id)

    def finish(self, episode_id: str):
        return self._json("POST", f"/episodes/{_token(episode_id)}/finish")

    def wait_action(self, episode_id: str, action_id: str, *, timeout_s: float | None,
                    poll_interval_s: float = 0.25):
        """Wait until a terminal action state; never submit/retry an action."""
        duration = optional_positive(timeout_s, "timeout_s")
        deadline = None if duration is None else time.monotonic()+duration
        interval = _positive(poll_interval_s, "poll_interval_s")
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise BenchHTTPError("action polling deadline expired", category="request_budget_exhausted")
            result = self.get_action(episode_id, action_id, deadline=deadline)
            if result["status"] in ("completed", "cancelled", "failed"):
                return result
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise BenchHTTPError("action polling deadline expired", category="request_budget_exhausted")
            time.sleep(minimum_limit(interval, remaining))

    def _png(self, uri, *, expected_sha, width, height, max_bytes=16 * 1024 * 1024):
        try:
            with self.client.stream("GET", self.base_url + uri, timeout=self._remaining_timeout(),
                                    headers=self._request_headers(uri), follow_redirects=False) as response:
                if response.status_code != 200:
                    raise BenchHTTPError("public camera request failed", status_code=response.status_code)
                if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "image/png":
                    raise BenchHTTPError("public camera response must be PNG")
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    self._remaining_timeout()
                    size += len(chunk)
                    if size > max_bytes:
                        raise BenchHTTPError("public camera PNG exceeds the size limit")
                    chunks.append(chunk)
                value = b"".join(chunks)
        except httpx.TransportError as error:
            raise BenchHTTPError("public camera transport failed", category=(
                "transport_timeout" if isinstance(error, httpx.TimeoutException) else "transport_error")) from None
        if not value.startswith(b"\x89PNG\r\n\x1a\n"):
            raise BenchHTTPError("public camera response must be PNG")
        if hashlib.sha256(value).hexdigest() != expected_sha:
            raise BenchHTTPError("public camera digest does not match its observation")
        try:
            with Image.open(BytesIO(value)) as frame:
                if frame.format != "PNG" or frame.size != (width, height) or width * height > 16_777_216:
                    raise BenchHTTPError("public camera dimensions or format do not match")
                frame.verify()
        except BenchHTTPError:
            raise
        except Exception:
            raise BenchHTTPError("public camera PNG is invalid") from None
        return value

    def download_cameras(self, episode_id: str, observation: dict) -> list[dict]:
        """Download only SHA-verified frames named by this public observation.

        Every result has metadata and png_bytes. RGB-D cameras additionally
        return depth_npy_bytes and the verified, unmodified depth_m array.
        Absolute URLs, redirects, other episodes and encoded paths are rejected
        before following any camera-supplied location.
        """
        episode_id = _token(episode_id)
        try:
            parsed = public_observation(observation)
        except ActionValidationError:
            raise BenchHTTPError("invalid public camera observation") from None
        if parsed.episode_id != episode_id:
            raise BenchHTTPError("camera observation belongs to another episode")
        paths = []
        seen = set()
        revisions = set()
        for camera in parsed.cameras:
            camera_id = _token(camera.camera_id)
            uri = camera.rgb_uri
            # Do not decode, normalize or join untrusted paths. Exact matching
            # excludes traversal, //host, percent escapes, query and fragments.
            pattern = rf"/episodes/{re.escape(episode_id)}/images/(0|[1-9][0-9]*)/{re.escape(camera_id)}"
            match = re.fullmatch(pattern, uri)
            if not match:
                raise BenchHTTPError("camera URI is outside this episode's public image endpoint")
            revision = match.group(1)
            revisions.add(revision)
            if camera.width * camera.height > MAX_DEPTH_PIXELS:
                raise BenchHTTPError("public camera dimensions exceed the pixel limit")
            if camera.depth_uri is not None:
                expected_depth_uri = f'/episodes/{episode_id}/depth/{revision}/{camera_id}'
                if camera.depth_uri != expected_depth_uri:
                    raise BenchHTTPError("depth URI does not match the RGB episode, revision and camera")
                if (camera.depth_encoding != 'float32_npy' or camera.depth_unit != 'm'
                        or camera.depth_semantics != 'camera_z' or camera.depth_invalid_value != 0.
                        or camera.depth_rgb_aligned is not True
                        or abs(camera.depth_sim_time_s - camera.sim_time_s) > 1e-9):
                    raise BenchHTTPError("depth metadata does not describe aligned metric camera-z data")
            if camera_id in seen or abs(camera.sim_time_s - parsed.sim_time_s) > 1e-8:
                raise BenchHTTPError("duplicate or stale public camera frame")
            seen.add(camera_id)
            paths.append(camera)
        if len(revisions) > 1:
            raise BenchHTTPError("public camera frames have different revisions")
        # Validate every URI first: a bad second camera must not cause even a
        # partial first-camera download before rejecting the observation.
        output = []
        for camera in paths:
            value = self._png(camera.rgb_uri, expected_sha=camera.rgb_sha256,
                              width=camera.width, height=camera.height)
            row = {"metadata": deepcopy(camera.model_dump(mode="json", exclude_none=True)),
                   "png_bytes": value}
            if camera.depth_uri is not None:
                raw, depth = self._depth_npy(camera.depth_uri, expected_sha=camera.depth_sha256,
                                            width=camera.width, height=camera.height)
                row.update(depth_npy_bytes=raw, depth_m=depth)
            output.append(row)
        return output

    def _depth_npy(self, uri, *, expected_sha, width, height):
        max_bytes = width * height * 4 + MAX_NPY_HEADER_BYTES + 12
        try:
            with self.client.stream('GET', self.base_url + uri, timeout=self._remaining_timeout(),
                                    headers=self._request_headers(uri), follow_redirects=False) as response:
                if response.status_code != 200:
                    raise BenchHTTPError('public depth request failed', status_code=response.status_code)
                mime = response.headers.get('content-type', '').split(';', 1)[0].strip().lower()
                if mime != 'application/octet-stream':
                    raise BenchHTTPError('public depth response must be a binary NPY file')
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    self._remaining_timeout()
                    size += len(chunk)
                    if size > max_bytes:
                        raise BenchHTTPError('public depth NPY exceeds the size limit')
                    chunks.append(chunk)
                raw = b''.join(chunks)
        except httpx.TransportError as error:
            raise BenchHTTPError('public depth transport failed', category=(
                'transport_timeout' if isinstance(error, httpx.TimeoutException) else 'transport_error')) from None
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise BenchHTTPError('public depth digest does not match its observation')
        try:
            depth = decode_metric_depth_npy(raw, width=width, height=height)
        except Exception:
            raise BenchHTTPError('public depth NPY has invalid shape, dtype, contents or encoding') from None
        return raw, depth

    def _remaining_timeout(self):
        remaining = minimum_limit(self.timeout_s,
                                  None if self.deadline is None else self.deadline-time.monotonic())
        if remaining is not None and remaining <= 0:
            raise BenchHTTPError('Public request deadline expired', category='request_budget_exhausted')
        return remaining
