"""HTTP isolation and bounded polling with a public-only mock transport."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json

import httpx
import numpy as np
from PIL import Image
import pytest

from agent.model_client.bench_http import BenchHTTPClient, BenchHTTPError


@pytest.fixture
def camera_observation():
    stream = BytesIO()
    Image.new("RGB", (8, 6), (20, 60, 100)).save(stream, format="PNG")
    png = stream.getvalue()
    obs = dict(episode_id="ep1", sequence=2, sim_time_s=0.4, instruction="Move it.",
        profile="vision_proprio", proprioception={"joint_positions_rad": {"joint": 0.0},
            "joint_velocities_rad_s": {"joint": 0.0}},
        cameras=[dict(camera_id="overview", sim_time_s=0.4,
            rgb_uri="/episodes/ep1/images/3/overview", rgb_sha256=hashlib.sha256(png).hexdigest(),
            width=8, height=6)])
    return obs, png


def client_for(handler, **options):
    transport = httpx.MockTransport(handler)
    return BenchHTTPClient("http://bench.test:8765", client=httpx.Client(transport=transport), **options)


def test_retries_share_one_deadline(monkeypatch):
    import agent.model_client.bench_http as module
    clock=[0.]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    requests=[]
    def handler(request):
        requests.append(request);clock[0]=2.
        return httpx.Response(503,json={'error':'busy'})
    client=client_for(handler,timeout_s=1.,get_retries=2)
    with pytest.raises(BenchHTTPError,match='deadline'):
        client.list_tasks()
    assert len(requests)==1


def test_expired_host_deadline_blocks_image_request(monkeypatch,camera_observation):
    import agent.model_client.bench_http as module
    monkeypatch.setattr(module.time,'monotonic',lambda:10.)
    def handler(_):pytest.fail('expired request must not reach transport')
    client=client_for(handler);client.deadline=9.
    with pytest.raises(BenchHTTPError,match='deadline'):
        client.download_cameras('ep1',camera_observation[0])


def test_public_endpoint_inventory_and_post_payloads():
    seen = []
    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.url.path.endswith("/result"):
            return httpx.Response(409, json={"private": "must not escape"})
        return httpx.Response(200, json={"ok": True})
    client = client_for(handler)
    client.list_tasks(); client.list_robots(); client.list_policies()
    client.create({"task_id": "demo"}); client.get_observation("ep1")
    assert client.get_result("ep1") is None
    client.submit("ep1", {"action_id": "a1"}); client.finish("ep1")
    assert [(method, path) for method, path, _ in seen] == [
        ("GET", "/tasks"), ("GET", "/robots"), ("GET", "/policies"),
        ("POST", "/episodes"), ("GET", "/episodes/ep1/observation"),
        ("GET", "/episodes/ep1/result"), ("POST", "/episodes/ep1/actions"),
        ("POST", "/episodes/ep1/finish")]
    assert seen[3][2] == {"task_id": "demo"}
    assert seen[6][2] == {"action_id": "a1"}


def test_image_bytes_match_digest_and_only_current_episode_is_requested(camera_observation):
    obs, png = camera_observation
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=png, headers={"content-type": "image/png"})
    images = client_for(handler).download_cameras("ep1", obs)
    assert images[0]["png_bytes"] == png
    assert images[0]["metadata"]["rgb_sha256"] == hashlib.sha256(png).hexdigest()
    assert str(requests[0].url) == "http://bench.test:8765/episodes/ep1/images/3/overview"


@pytest.mark.parametrize("uri", [
    "https://attacker.test/episodes/ep1/images/3/overview",
    "http://bench.test:8765/episodes/ep1/images/3/overview",
    "//attacker.test/episodes/ep1/images/3/overview",
    "/episodes/ep2/images/3/overview", "/episodes/ep1/private/physics.jsonl",
    "/episodes/ep1/images/3/../../private", "/episodes/ep1/images/3/%2e%2e",
    "/episodes/ep1/images/3/overview?url=https://attacker.test", "/episodes/ep1/images/3/overview#fragment",
    "/episodes/ep1/images/3/task", "/episodes/ep1/images/03/overview",
    "/episodes/ep1/images/3\\overview", "episodes/ep1/images/3/overview",
])
def test_image_path_escape_rejected_before_network(camera_observation, uri):
    obs, _ = camera_observation
    obs["cameras"][0]["rgb_uri"] = uri
    def handler(_):
        pytest.fail("unsafe camera URI reached the network")
    with pytest.raises(BenchHTTPError):
        client_for(handler).download_cameras("ep1", obs)


def test_all_camera_uris_checked_before_any_fetch(camera_observation):
    obs, _ = camera_observation
    bad = deepcopy(obs["cameras"][0])
    bad.update(camera_id="task", rgb_uri="https://attacker.test/frame")
    obs["cameras"].append(bad)
    def handler(_):
        pytest.fail("no camera should be downloaded before all URI checks pass")
    with pytest.raises(BenchHTTPError):
        client_for(handler).download_cameras("ep1", obs)


@pytest.mark.parametrize("change", [
    lambda obs: obs.update(episode_id="ep2"),
    lambda obs: obs.update(privileged_object_poses={}),
    lambda obs: obs["cameras"][0].update(sim_time_s=0.2),
    lambda obs: obs["cameras"].append(deepcopy(obs["cameras"][0])),
])
def test_stale_duplicate_or_other_episode_observation_rejected(camera_observation, change):
    obs, _ = camera_observation
    change(obs)
    def handler(_):
        pytest.fail("invalid observation reached the network")
    with pytest.raises(BenchHTTPError):
        client_for(handler).download_cameras("ep1", obs)


@pytest.mark.parametrize("kind", ["sha", "mime", "magic", "dimensions", "broken"])
def test_nonpng_corrupt_or_mismatched_frame_rejected(camera_observation, kind):
    obs, png = camera_observation
    mime = "image/png"
    if kind == "sha": obs["cameras"][0]["rgb_sha256"] = "f" * 64
    if kind == "mime": mime = "image/jpeg"
    if kind == "magic": png = b"not a PNG"
    if kind == "dimensions": obs["cameras"][0]["width"] = 9
    if kind == "broken":
        png = b"\x89PNG\r\n\x1a\ncorrupt"
        obs["cameras"][0]["rgb_sha256"] = hashlib.sha256(png).hexdigest()
    with pytest.raises(BenchHTTPError):
        client_for(lambda _: httpx.Response(200, content=png, headers={"content-type": mime})).download_cameras("ep1", obs)


def test_redirect_never_sends_credentials_to_another_origin(camera_observation):
    obs, _ = camera_observation
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"location": "https://attacker.test/steal"})
    injected = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True,
                           headers={"Authorization": "Bearer fake-test-only"})
    client = BenchHTTPClient("http://bench.test", client=injected)
    with pytest.raises(BenchHTTPError) as error:
        client.download_cameras("ep1", obs)
    assert error.value.status_code == 302
    assert len(seen) == 1 and seen[0].url.host == "bench.test"


def test_get_has_bounded_retry_but_post_is_never_retried():
    seen = []
    def handler(request):
        seen.append(request.method)
        raise httpx.ReadTimeout("SECRET upstream token", request=request)
    client = client_for(handler, get_retries=2)
    with pytest.raises(BenchHTTPError) as error:
        client.list_tasks()
    assert "SECRET" not in str(error.value)
    assert seen == ["GET", "GET", "GET"]
    with pytest.raises(BenchHTTPError):
        client.submit("ep1", {"action_id": "once"})
    assert seen == ["GET", "GET", "GET", "POST"]


def test_response_error_is_sanitized_and_status_preserved():
    client = client_for(lambda _: httpx.Response(422, json={"detail": "SECRET token and private path"}))
    with pytest.raises(BenchHTTPError) as error:
        client.create({})
    assert error.value.status_code == 422
    assert "SECRET" not in str(error.value) and "private" not in str(error.value)


def test_wait_action_reaches_terminal_state_without_posts(monkeypatch):
    statuses = iter(["accepted", "running", "completed"])
    seen = []
    def handler(request):
        seen.append(request.method)
        return httpx.Response(200, json={"action_id": "a1", "status": next(statuses)})
    monkeypatch.setattr("agent.model_client.bench_http.time.sleep", lambda _: None)
    assert client_for(handler).wait_action("ep1", "a1", timeout_s=1)["status"] == "completed"
    assert seen == ["GET"] * 3


def test_wait_action_deadline_expires_deterministically(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("agent.model_client.bench_http.time.monotonic", lambda: now[0])
    monkeypatch.setattr("agent.model_client.bench_http.time.sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    client = client_for(lambda _: httpx.Response(200, json={"action_id": "a1", "status": "running"}))
    with pytest.raises(BenchHTTPError, match="deadline"):
        client.wait_action("ep1", "a1", timeout_s=0.5, poll_interval_s=0.2)
    assert now[0] == 0.5


@pytest.mark.parametrize("url", ["file:///etc/passwd", "https://user:secret@bench.test", "https://bench.test/api",
                                  "https://bench.test?secret=token", "https://bench.test/#frag", "//bench.test"])
def test_unsafe_base_url_rejected(url):
    with pytest.raises(ValueError):
        BenchHTTPClient(url)


@pytest.mark.parametrize("episode", ["../private", "ep1/../ep2", "ep1%2fprivate", "https://attacker.test"])
def test_episode_identifier_cannot_escape_endpoint(episode):
    with pytest.raises(BenchHTTPError):
        client_for(lambda _: pytest.fail("unsafe identifier reached network")).get_result(episode)


@pytest.fixture
def rgbd_observation(camera_observation):
    obs, png = camera_observation
    depth = np.linspace(.1, 5., 48, dtype=np.float32).reshape(6, 8)
    depth[0, 0] = 0.
    stream = BytesIO()
    np.save(stream, depth, allow_pickle=False)
    raw = stream.getvalue()
    obs['cameras'][0].update(
        depth_uri='/episodes/ep1/depth/3/overview', depth_sha256=hashlib.sha256(raw).hexdigest(),
        depth_encoding='float32_npy', depth_semantics='camera_z', depth_unit='m',
        depth_invalid_value=0.0, depth_sim_time_s=.4, depth_rgb_aligned=True,
        intrinsics_3x3=[10., 0., 4., 0., 10., 3., 0., 0., 1.],
        world_from_camera={'frame_id':'world', 'position_m':[0., 0., 0.],
                           'quaternion_wxyz':[1., 0., 0., 0.]})
    return obs, png, raw, depth


def test_rgbd_download_preserves_metric_array_bytes_and_same_revision(rgbd_observation):
    obs, png, raw, depth = rgbd_observation
    paths = []
    def handler(request):
        paths.append(request.url.path)
        is_depth = '/depth/' in request.url.path
        return httpx.Response(200, content=raw if is_depth else png,
                              headers={'content-type':'application/octet-stream' if is_depth else 'image/png'})
    row, = client_for(handler).download_cameras('ep1', obs)
    assert row['png_bytes'] == png and row['depth_npy_bytes'] == raw
    np.testing.assert_array_equal(row['depth_m'], depth)
    assert paths == ['/episodes/ep1/images/3/overview', '/episodes/ep1/depth/3/overview']


@pytest.mark.parametrize('uri', [
    'https://attacker.test/file.npy', '//attacker.test/file.npy',
    '/episodes/ep2/depth/3/overview', '/episodes/ep1/depth/4/overview',
    '/episodes/ep1/depth/3/task', '/episodes/ep1/depth/03/overview',
    '/episodes/ep1/depth/3/overview?x=1', '/episodes/ep1/depth/3/%2e%2e',
    '/episodes/ep1/private/physics.jsonl'])
def test_depth_uri_rejected_before_rgb_or_depth_network(rgbd_observation, uri):
    obs, _, _, _ = rgbd_observation
    obs['cameras'][0]['depth_uri'] = uri
    with pytest.raises(BenchHTTPError):
        client_for(lambda _: pytest.fail('invalid depth URI caused a network request')).download_cameras('ep1', obs)


@pytest.mark.parametrize('updates', [
    {'depth_sim_time_s': .3}, {'depth_rgb_aligned': False}, {'depth_semantics':'ray_range'},
    {'depth_encoding':'uint16_png'}, {'depth_unit':'mm'}, {'depth_invalid_value':-1.},
    {'intrinsics_3x3':None}, {'depth_sha256':None}, {'height':2**30},
])
def test_bad_depth_contract_rejected_before_network(rgbd_observation, updates):
    obs, _, _, _ = rgbd_observation
    obs['cameras'][0].update(updates)
    with pytest.raises(BenchHTTPError):
        client_for(lambda _: pytest.fail('invalid depth metadata caused a network request')).download_cameras('ep1', obs)


@pytest.mark.parametrize('kind', ['hash', 'mime', 'not_npy', 'shape', 'dtype', 'nan',
                                 'negative', 'all_invalid', 'size_limit', 'redirect'])
def test_invalid_downloaded_depth_is_rejected(rgbd_observation, kind):
    obs, png, raw, depth = rgbd_observation
    mime = 'application/octet-stream'
    if kind == 'hash': obs['cameras'][0]['depth_sha256'] = 'f' * 64
    if kind == 'mime': mime = 'application/json'
    if kind == 'not_npy': raw = b'not NPY'
    if kind in ('shape', 'dtype', 'nan', 'negative', 'all_invalid'):
        if kind == 'shape': depth = depth.reshape(8, 6)
        if kind == 'dtype': depth = depth.astype(np.float64)
        if kind == 'nan': depth[0, 1] = np.nan
        if kind == 'negative': depth[0, 1] = -.1
        if kind == 'all_invalid': depth[:] = 0.
        stream = BytesIO()
        np.save(stream, depth, allow_pickle=False)
        raw = stream.getvalue()
    if kind == 'size_limit': raw += b'0' * 5000
    if kind != 'hash': obs['cameras'][0]['depth_sha256'] = hashlib.sha256(raw).hexdigest()
    seen = []
    def handler(request):
        seen.append(request.url.host)
        if '/images/' in request.url.path:
            return httpx.Response(200, content=png, headers={'content-type':'image/png'})
        if kind == 'redirect':
            return httpx.Response(302, headers={'location':'https://attacker.test/steal'})
        return httpx.Response(200, content=raw, headers={'content-type':mime})
    with pytest.raises(BenchHTTPError):
        client_for(handler).download_cameras('ep1', obs)
    assert set(seen) == {'bench.test'}


def test_different_camera_revisions_rejected_before_any_fetch(camera_observation):
    obs, _ = camera_observation
    other = deepcopy(obs['cameras'][0])
    other.update(camera_id='task', rgb_uri='/episodes/ep1/images/4/task')
    obs['cameras'].append(other)
    with pytest.raises(BenchHTTPError, match='revisions'):
        client_for(lambda _: pytest.fail('mixed revisions reached network')).download_cameras('ep1', obs)
