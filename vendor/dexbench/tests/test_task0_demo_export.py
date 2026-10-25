"""Synthetic recorded images exercise integrity and exact presentation timing."""
import json
import shutil

from PIL import Image
import pytest

from bench.paths import load_json
from scripts.media.export_task0_demo import (canonical, digest, export_demo, load_live_prefix,
                                             load_sources, outcome_label, verify_live_prefix)


COMMIT = 'b671df1d984461a4a082d5dcbbea9cd68504429b'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n')


def seal(episode):
    dump(episode / 'artifact_hashes.json', {str(path.relative_to(episode)): digest(path)
        for path in episode.rglob('*') if path.is_file() and path.name != 'artifact_hashes.json'})


@pytest.fixture(params=('public_function_agent.v1', 'public_function_agent.v2', 'public_function_agent.v3'))
def recorded(tmp_path, request):
    episode, client, snapshot = (tmp_path / name for name in ('test_episode', 'client', 'snapshot'))
    verification = {'verified': True, 'runtime_imports_verified': True, 'snapshot_root': str(snapshot)}
    config = {'condition': '公开函数基线', 'protocol': request.param,
              'task_id': 'task0', 'mode': 'single_agent', 'source_verification_sha256': canonical(verification),
              'models': {'single': {'model': 'gpt-6-astra', 'requested_reasoning_effort': 'max'}}}
    spec = {'episode_id': 'test_episode', 'task_id': 'task0', 'agent_id': 'public_function_agent',
            'agent_config_sha256': canonical(config)}
    result = {'spec': spec, 'elapsed_sim_s': .14, 'outcome': 'cancelled', 'physics_valid': True}
    report = {'schema': request.param, 'episode_id': 'test_episode', 'board_game': False,
              'result': result, 'run_status': 'interrupted', 'termination': 'interrupted'}
    dump(episode / 'agent/result.json', result)
    dump(episode / 'agent/spec.json', spec)
    dump(client / 'agent_config.json', config)
    dump(client / 'run_report.json', report)
    dump(client / 'episode_ready.json', {'episode_id': 'test_episode', 'spec': spec})
    dump(client / 'source_verification.json', verification)
    dump(snapshot / 'source_snapshot_manifest.json', {'synthetic_test_source': True})
    root = episode / 'private/demo_recording'
    frames = []
    for index, step in enumerate((0, 5, 7)):
        cameras = {}
        for camera in ('task', 'overview'):
            relative = f'rgb/{camera}/{index:08d}.png'
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (64, 48), (30 + index * 80, 50, 90)).save(path)
            cameras[camera] = {'path': relative, 'sha256': digest(path), 'width': 64, 'height': 48}
        frames.append({'schema': 'native_episode_demo_frame.v1', 'frame_index': index,
            'state_index': step, 'sim_time_s': step * .02,
            'capture_kind': ('reset', 'periodic', 'terminal')[index],
            'interval_from_previous_s': (None, .1, .04)[index],
            'capture_state_unchanged': True, 'rgbd_timestamp_verified': True,
            'score': {'placed_count': index, 'required_count': 3, 'success': False}, 'cameras': cameras})
    frame_path = root / 'frames.jsonl'
    frame_path.write_text(''.join(json.dumps(row) + '\n' for row in frames))
    recording = {'schema': 'native_episode_demo_recording.v1', 'enabled': True, 'completed': True,
        'reason': 'finish_request', 'error': None, 'capture_source': 'original_native_episode_rtx',
        'sampling_rate_hz': 10, 'control_period_s': .02, 'stride': 5, 'camera_ids': ['task', 'overview'],
        'last_control_step': 7, 'sim_duration_s': .14, 'frame_count': 3,
        'frames_file': 'frames.jsonl', 'frames_sha256': digest(frame_path)}
    dump(root / 'recording.json', recording)
    seal(episode)
    return episode, client, root, recording, frames


def test_formally_interrupted_episode_is_a_complete_recording(recorded):
    episode, client, *_ = recorded
    source = load_sources(episode, client, COMMIT)
    assert source['report']['run_status'] == 'interrupted'
    assert source['duration_s'] == .14 and len(source['frames']) == 3
    assert source['frames'][-1]['score']['placed_count'] == 2
    assert outcome_label(source) == '取消（未完成）'


@pytest.mark.parametrize('fault', ['incomplete', 'capture_error', 'missing_frame', 'frame_hash',
                                  'missing_hash', 'timestamp', 'interval', 'terminal_step', 'result_duration',
                                  'camera_missing', 'unsafe_path', 'progress', 'unverified_capture'])
def test_rejects_unfinished_missing_corrupt_or_inconsistent_recordings(recorded, fault):
    episode, client, root, recording, frames = recorded
    if fault == 'incomplete':
        recording['completed'] = False
    elif fault == 'capture_error':
        recording['error'] = {'stage': 'render', 'type': 'RuntimeError', 'message': 'failed'}
    elif fault == 'missing_frame':
        frames.pop(1)
        recording['frame_count'] = 2
    elif fault == 'frame_hash':
        frames[0]['cameras']['task']['sha256'] = '0' * 64
    elif fault == 'missing_hash':
        frames[0]['cameras']['task'].pop('sha256')
    elif fault == 'timestamp':
        frames[1]['sim_time_s'] = .11
    elif fault == 'interval':
        frames[-1]['interval_from_previous_s'] = .1
    elif fault == 'terminal_step':
        frames[-1]['state_index'] = 8
    elif fault == 'result_duration':
        result = load_json(episode / 'agent/result.json')
        result['elapsed_sim_s'] = .16
        dump(episode / 'agent/result.json', result)
        report = load_json(client / 'run_report.json')
        report['result'] = result
        dump(client / 'run_report.json', report)
    elif fault == 'camera_missing':
        frames[-1]['cameras'].pop('overview')
    elif fault == 'unsafe_path':
        frames[0]['cameras']['task']['path'] = '../outside.png'
    elif fault == 'progress':
        frames[0]['score']['placed_count'] = 4
    elif fault == 'unverified_capture':
        frames[0]['capture_state_unchanged'] = False
    path = root / 'frames.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in frames))
    recording['frames_sha256'] = digest(path)
    dump(root / 'recording.json', recording)
    seal(episode)
    with pytest.raises(ValueError):
        load_sources(episode, client, COMMIT)


def test_rejects_unsealed_or_changed_server_result(recorded):
    episode, client, *_ = recorded
    result = episode / 'agent/result.json'
    result.write_text(result.read_text() + ' ')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_sources(episode, client, COMMIT)


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg is required')
def test_real_encoder_has_exact_two_times_duration_and_no_terminal_hold(recorded, tmp_path):
    episode, client, *_ = recorded
    source = load_sources(episode, client, COMMIT)
    output = tmp_path / 'demo/task0_2x.mp4'
    report = export_demo(source, output)
    assert output.is_file() and output.with_suffix('.terminal.png').is_file()
    assert load_json(output.with_suffix('.json')) == report
    assert report['capture_sampling_rate_hz'] == 10
    assert report['encoded_frames_per_second'] == 100
    assert report['encoded_frame_count'] == 7
    assert report['encoded_duration_s'] == .07
    assert report['terminal_frame_padding_s'] == 0
    assert report['frame_map'] == [
        {'source_frame_index': 0, 'sim_time_s': 0., 'output_start_frame': 0, 'output_end_frame_exclusive': 5},
        {'source_frame_index': 1, 'sim_time_s': .1, 'output_start_frame': 5, 'output_end_frame_exclusive': 7}]
    assert report['source_result_outcome'] == 'cancelled' and report['source_run_status'] == 'interrupted'
    assert report['original_execution_video'] and not report['state_replay'] and not report['physics_interpolation']
    with pytest.raises(ValueError, match='overwritten'):
        export_demo(source, output)


@pytest.fixture
def live(recorded):
    episode, client, root, recording, frames = recorded
    (episode / 'agent/result.json').unlink()
    (episode / 'artifact_hashes.json').unlink()
    (client / 'run_report.json').unlink()
    prefix = ''.join(json.dumps(row) + '\n' for row in frames[:2]).encode()
    (root / 'frames.jsonl').write_bytes(prefix)
    recording.update(completed=False, status='recording', reason='recording', frame_count=2,
                     last_control_step=5, sim_duration_s=.1, frames_sha256=digest(root / 'frames.jsonl'))
    dump(root / 'recording.json', recording)
    # This unfinished future row has not been committed by recording.json.
    with (root / 'frames.jsonl').open('ab') as stream:
        stream.write(b'{"incomplete_next_frame":')
    return episode, client, root, recording, frames[:2], prefix


def test_live_prefix_uses_committed_rows_without_reading_partial_tail(live):
    episode, client, root, _, _, prefix = live
    source = load_live_prefix(episode, client, COMMIT, 5)
    assert source['prefix_bytes'] == prefix
    assert source['result'] is None and source['report'] is None
    assert source['frames'][-1]['score']['placed_count'] == 1
    assert source['preview_prefix']['through_sim_time_s'] == .1
    # Appending progress is allowed, whereas changing a selected old row is not.
    with (root / 'frames.jsonl').open('ab') as stream:
        stream.write(b'123}\n')
    verify_live_prefix(source)
    data = (root / 'frames.jsonl').read_bytes()
    (root / 'frames.jsonl').write_bytes(data.replace(b'"placed_count": 0', b'"placed_count": 2', 1))
    with pytest.raises(ValueError, match='Selected live prefix changed'):
        verify_live_prefix(source)
    # The sealed-mode loader remains strict despite the new preview option.
    with pytest.raises(FileNotFoundError):
        load_sources(episode, client, COMMIT)


@pytest.mark.parametrize('boundary', [0, 1, 7, 10])
def test_live_boundary_must_be_a_committed_grid_sample(live, boundary):
    episode, client, *_ = live
    with pytest.raises(ValueError):
        load_live_prefix(episode, client, COMMIT, boundary)


@pytest.mark.parametrize('fault', ['incomplete_row', 'index_hash', 'timestamp', 'guard', 'image_hash',
                                  'frame_order', 'identity', 'verification', 'capture_error'])
def test_live_prefix_rejects_bad_evidence_without_seal(live, fault):
    episode, client, root, recording, frames, _ = live
    if fault == 'timestamp':
        frames[-1]['sim_time_s'] = .08
    elif fault == 'guard':
        frames[-1]['capture_state_unchanged'] = False
    elif fault == 'image_hash':
        frames[-1]['cameras']['task']['sha256'] = '0' * 64
    elif fault == 'frame_order':
        frames[-1]['state_index'] = 4
    elif fault == 'identity':
        spec = load_json(episode / 'agent/spec.json')
        spec['episode_id'] = 'another_episode'
        dump(episode / 'agent/spec.json', spec)
    elif fault == 'verification':
        verification = load_json(client / 'source_verification.json')
        verification['runtime_imports_verified'] = False
        dump(client / 'source_verification.json', verification)
    elif fault == 'capture_error':
        recording['error'] = {'stage': 'capture', 'type': 'RuntimeError'}
    data = ''.join(json.dumps(row) + '\n' for row in frames).encode()
    (root / 'frames.jsonl').write_bytes(data)
    recording['frames_sha256'] = digest(root / 'frames.jsonl')
    if fault == 'incomplete_row':
        (root / 'frames.jsonl').write_bytes(data[:-1])
    elif fault == 'index_hash':
        recording['frames_sha256'] = '0' * 64
    dump(root / 'recording.json', recording)
    with pytest.raises(ValueError):
        load_live_prefix(episode, client, COMMIT, 5)


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg is required')
def test_live_video_has_prefix_evidence_and_no_fabricated_terminal_result(live, tmp_path):
    episode, client, root, _, _, prefix = live
    source = load_live_prefix(episode, client, COMMIT, 5)
    # Recorder progress must not invalidate an already selected prefix.
    dump(root / 'recording.json', {'newer_progress': True})
    output = tmp_path / 'demo/preview_2x.mp4'
    report = export_demo(source, output)
    assert report['is_partial_preview'] and report['original_execution_video']
    assert report['episode_final_result'] is None and not report['episode_completeness_claimed']
    assert report['source_episode_result_path'] is None
    assert report['source_physics_valid'] is None and report['source_result_outcome'] is None
    assert report['final_placed_count'] is None and report['current_placed_count'] == 1
    assert report['encoded_duration_s'] == .05 and report['encoded_frame_count'] == 5
    assert 'terminal_image' not in report and output.with_suffix('.boundary.png').is_file()
    assert output.with_suffix('.frames.prefix.jsonl').read_bytes() == prefix
    assert digest(output.with_suffix('.recording.snapshot.json')) == report['preview_prefix']['recording_metadata_snapshot_sha256']
    assert load_json(output.with_suffix('.json')) == report
