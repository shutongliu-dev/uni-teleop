"""A model-selected pixel cannot be rebound by a later read-only tool refresh."""
from copy import deepcopy
import hashlib
import json

import pytest

from agent.model_client import runner
from bench.paths import load_json
from test_multimode_runner import loop


@pytest.mark.parametrize('tool', ['point_at', 'observe'])
def test_model_read_rejects_refresh_race_and_recovers(monkeypatch, tmp_path, tool):
    call = {'tool': tool, 'arguments': (
        {'camera_id': 'task', 'u': 8, 'v': 8} if tool == 'point_at' else {})}
    args, bench, model = loop(monkeypatch, tmp_path, [
        call, {'tool': 'stop', 'arguments': {}}], mode='tools', rgbd=True)
    original_download = bench.download_cameras
    frames = {}

    def publish_camera_fixture():
        rows = original_download()
        sequence = bench.obs['sequence']
        for row in rows:
            name = row['metadata']['camera_id']
            row['metadata'].update(
                sim_time_s=bench.obs['sim_time_s'], width=32, height=24,
                rgb_uri=f'/episodes/episode/images/{sequence}/{name}',
                depth_uri=f'/episodes/episode/depth/{sequence}/{name}',
                depth_sha256=hashlib.sha256(row['depth_npy_bytes']).hexdigest(),
                depth_unit='m', depth_encoding='float32_npy', depth_semantics='camera_z',
                depth_invalid_value=0., depth_sim_time_s=bench.obs['sim_time_s'],
                depth_rgb_aligned=True, intrinsics_3x3=[20.,0.,16.,0.,20.,12.,0.,0.,1.],
                world_from_camera={'frame_id':'world', 'position_m':[0.,0.,0.],
                                   'quaternion_wxyz':[1.,0.,0.,0.]})
        frames[sequence] = deepcopy(rows)
        bench.obs['cameras'] = [deepcopy(row['metadata']) for row in rows]

    publish_camera_fixture()
    monkeypatch.setattr(bench, 'download_cameras',
        lambda episode, observation: deepcopy(frames[observation['sequence']]))
    get_observation = bench.get_observation
    reads = 0

    def advance_at_final_tool_refresh(episode):
        nonlocal reads
        if len(model.requests) == 1:
            reads += 1
            # Runner precheck, tool precheck, then context's final refresh.
            if reads == 3:
                bench.obs['sequence'] += 1
                publish_camera_fixture()
        return get_observation(episode)

    monkeypatch.setattr(bench, 'get_observation', advance_at_final_tool_refresh)
    report = runner.run(args)
    assert reads >= 3 and report['error_type'] is None
    assert report['termination'] == 'model_requested_stop'
    assert report['primitive_attempts'] == report['submitted_actions'] == 0
    assert not bench.commands and len(model.requests) == 2
    first = args.output / 'decisions/0000'
    execution = load_json(first / 'execution.json')
    assert execution['status'] == 'rejected_by_client'
    assert 'stale' in execution['reason'].lower()
    assert not (first / 'tools/0000/view').exists()
    assert not list(first.rglob('primitive_request.json'))
    second_input = json.loads(model.requests[1][1][-1]['content'][0]['text'])
    assert second_input['observation']['sequence'] == 1
    assert second_input['previous_action_feedback']['status'] == 'rejected_by_client'

