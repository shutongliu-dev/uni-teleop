"""Export original, regularly sampled task0 RTX frames; never run a simulator.

The source worker records 10 Hz. Encoding on the 20 ms control-time grid keeps
an irregular terminal interval exact. Repeated encoded frames only display a
real sample until the next sample; they are not newly captured/interpolated data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from bench.isaacsim.video import FONT, probe
from bench.paths import OUTPUT_ROOT, load_json

CAMERAS = ('task', 'overview')
CLIENT_PROTOCOLS = ('public_function_agent.v1', 'public_function_agent.v2', 'public_function_agent.v3')
WIDTH, HEIGHT = 1920, 840
DEMO_ROOT = Path(f'{OUTPUT_ROOT}/media/agent_demos')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return load_json(Path(path))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return type(a) in (int, float) and math.isfinite(a) and abs(a - b) <= 1.e-6


def hash_value(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def inside(root, relative):
    relative = Path(relative)
    check(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe recording path')
    path = root / relative
    check(path.resolve().is_relative_to(root), 'Recording path escapes its source')
    return path


def pinned(root, manifest, relative, expected=None):
    path = inside(root, relative)
    value = manifest.get(str(relative))
    check(hash_value(value), 'Missing source artifact pin: ' + str(relative))
    check(expected is None or expected == value, 'Source pin disagreement: ' + str(relative))
    check(digest(path) == value, 'Source artifact hash mismatch: ' + str(relative))
    return path


def validate_frames(frames, expected, recording_dir, *, episode=None, manifest=None):
    """The same timestamp, cadence, camera and native guard checks for both modes."""
    check(len(frames) == len(expected), 'Missing or extra recording frames')
    dimensions = {}
    for index, (frame, step) in enumerate(zip(frames, expected)):
        check(frame.get('schema') == 'native_episode_demo_frame.v1'
              and type(frame.get('frame_index')) is int and frame['frame_index'] == index
              and type(frame.get('state_index')) is int and frame['state_index'] == step,
              'Frame order or fixed sampling cadence differs')
        check(close(frame.get('sim_time_s'), step * .02), 'Frame timestamp is off the control grid')
        check(frame.get('capture_state_unchanged') is True and frame.get('rgbd_timestamp_verified') is True,
              'Frame lacks native state/timestamp verification')
        kind = 'reset' if index == 0 else 'terminal' if step % 5 else 'periodic'
        check(frame.get('capture_kind') == kind, 'Unexpected capture kind at the terminal boundary')
        interval = None if index == 0 else (step - expected[index - 1]) * .02
        check(frame.get('interval_from_previous_s') is None if interval is None
              else close(frame.get('interval_from_previous_s'), interval), 'Frame interval is inconsistent')
        score = frame.get('score', {})
        check(type(score.get('placed_count')) is int and 0 <= score['placed_count'] <= 3
              and score.get('required_count') == 3 and type(score.get('success')) is bool,
              'Placement progress is invalid')
        check(set(frame.get('cameras', {})) == set(CAMERAS), 'A frame is missing one camera')
        for camera in CAMERAS:
            row = frame['cameras'][camera]
            check(hash_value(row.get('sha256')), 'Missing camera frame hash')
            path = inside(recording_dir, row['path'])
            check(digest(path) == row['sha256'], 'Camera frame hash mismatch')
            if manifest is not None:
                relative = Path('private/demo_recording') / row['path']
                pinned(episode, manifest, str(relative), row['sha256'])
            check(type(row.get('width')) is int and type(row.get('height')) is int
                  and row['width'] > 0 and row['height'] > 0, 'Invalid source camera dimensions')
            with Image.open(path) as image:
                check(image.format == 'PNG' and image.size == (row['width'], row['height']), 'PNG dimensions differ')
                image.verify()
            dimensions.setdefault(camera, (row['width'], row['height']))
            check(dimensions[camera] == (row['width'], row['height']), 'Source camera dimensions changed')


def load_sources(episode, client, source_commit):
    """Require a sealed episode, complete camera timeline and matching result."""
    episode, client = Path(episode).resolve(), Path(client).resolve()
    check(re.fullmatch(r'[0-9a-f]{40}', source_commit) is not None, 'Use the full source commit SHA')
    manifest_path = episode / 'artifact_hashes.json'
    manifest = read(manifest_path)
    result_path = pinned(episode, manifest, 'agent/result.json')
    result = read(result_path)
    report_path, config_path = client / 'run_report.json', client / 'agent_config.json'
    report, config = read(report_path), read(config_path)
    ready_path, verification_path = client / 'episode_ready.json', client / 'source_verification.json'
    ready, verification = read(ready_path), read(verification_path)
    check(report.get('schema') in CLIENT_PROTOCOLS and report.get('schema') == config.get('protocol')
          and report.get('board_game') is False,
          'A finalized single-agent public-function run is required')
    check(report.get('result') == result and ready.get('spec') == result.get('spec'),
          'Client and sealed server result/spec differ')
    spec = result['spec']
    check(spec.get('task_id') == 'task0' and spec.get('agent_id') == 'public_function_agent',
          'The source must be a task0 public-function episode')
    check(ready.get('episode_id') == report.get('episode_id') == spec.get('episode_id'), 'Episode identities differ')
    check(spec.get('agent_config_sha256') == canonical(config), 'Agent configuration hash differs')
    model = config.get('models', {}).get('single', {})
    check(model.get('model') == 'gpt-6-astra' and model.get('requested_reasoning_effort') == 'max',
          'Video labels require requested gpt-6-astra / max')
    check(verification.get('verified') is True, 'Source import verification is missing')
    snapshot_manifest = Path(verification['snapshot_root']) / 'source_snapshot_manifest.json'
    check(snapshot_manifest.is_file(), 'Source snapshot manifest is missing')
    recording_dir = episode / 'private/demo_recording'
    recording_path = pinned(episode, manifest, 'private/demo_recording/recording.json')
    recording = read(recording_path)
    check(recording.get('schema') == 'native_episode_demo_recording.v1'
          and recording.get('capture_source') == 'original_native_episode_rtx', 'Original native RTX recording required')
    check(recording.get('enabled') is True and recording.get('completed') is True
          and recording.get('error') is None, 'Recording is unfinished or has capture errors')
    check(recording.get('sampling_rate_hz') == 10 and recording.get('stride') == 5
          and close(recording.get('control_period_s'), .02), 'Expected 10 Hz samples on the 20 ms control grid')
    check(set(recording.get('camera_ids', [])) == set(CAMERAS), 'Task and overview cameras are required')
    steps = recording.get('last_control_step')
    check(type(steps) is int and steps > 0, 'A positive recorded physical duration is required')
    duration = steps * .02
    check(close(recording.get('sim_duration_s'), duration) and close(result.get('elapsed_sim_s'), duration),
          'Recording terminal boundary differs from the episode result')
    check(recording.get('frames_file') == 'frames.jsonl', 'Unexpected frame index path')
    check(hash_value(recording.get('frames_sha256')), 'Missing frame index hash')
    frames_path = pinned(episode, manifest, 'private/demo_recording/frames.jsonl', recording.get('frames_sha256'))
    frames = [json.loads(line) for line in frames_path.read_text().splitlines() if line]
    expected = list(range(0, steps + 1, 5))
    if expected[-1] != steps:
        expected.append(steps)
    check(recording.get('frame_count') == len(frames) == len(expected), 'Missing or extra recording frames')
    validate_frames(frames, expected, recording_dir, episode=episode, manifest=manifest)
    source_files = [manifest_path, result_path, recording_path, frames_path, report_path, config_path,
                    ready_path, verification_path, snapshot_manifest]
    return {'episode': episode, 'client': client, 'recording_dir': recording_dir, 'recording': recording,
            'frames': frames, 'result': result, 'report': report, 'duration_s': duration,
            'control_steps': steps, 'source_commit': source_commit,
            'sources': {str(path): digest(path) for path in source_files}}


def load_live_prefix(episode, client, source_commit, through_state_index):
    """Freeze a committed index prefix, without constructing a final result.

    Metadata is atomically published after complete JSONL rows. Read exactly
    its declared row count and verify that prefix hash; later appends or a
    partial next row are outside this snapshot and are never decoded.
    """
    episode, client = Path(episode).resolve(), Path(client).resolve()
    check(re.fullmatch(r'[0-9a-f]{40}', source_commit) is not None, 'Use the full source commit SHA')
    check(type(through_state_index) is int and through_state_index > 0 and through_state_index % 5 == 0,
          'Choose a positive committed 10 Hz boundary (state index divisible by 5)')
    config_path, ready_path = client / 'agent_config.json', client / 'episode_ready.json'
    verification_path, spec_path = client / 'source_verification.json', episode / 'agent/spec.json'
    config, ready, verification, spec = (read(path) for path in
                                       (config_path, ready_path, verification_path, spec_path))
    check(config.get('protocol') in CLIENT_PROTOCOLS and config.get('task_id') == 'task0'
          and config.get('mode') == 'single_agent', 'A task0 public-function client is required')
    check(ready.get('spec') == spec and ready.get('episode_id') == spec.get('episode_id') == episode.name,
          'Live client and server episode identities differ')
    check(spec.get('task_id') == 'task0' and spec.get('agent_id') == 'public_function_agent'
          and spec.get('agent_config_sha256') == canonical(config), 'Live agent configuration identity differs')
    model = config.get('models', {}).get('single', {})
    check(model.get('model') == 'gpt-6-astra' and model.get('requested_reasoning_effort') == 'max',
          'Video labels require requested gpt-6-astra / max')
    check(verification.get('verified') is True and verification.get('runtime_imports_verified') is True
          and config.get('source_verification_sha256') == canonical(verification),
          'Live source import verification is missing or mismatched')
    snapshot_manifest = Path(verification['snapshot_root']) / 'source_snapshot_manifest.json'
    check(snapshot_manifest.is_file(), 'Source snapshot manifest is missing')
    recording_dir = episode / 'private/demo_recording'
    recording_path, frames_path = recording_dir / 'recording.json', recording_dir / 'frames.jsonl'
    metadata_raw = recording_path.read_bytes()
    recording = json.loads(metadata_raw)
    check(recording.get('schema') == 'native_episode_demo_recording.v1'
          and recording.get('capture_source') == 'original_native_episode_rtx'
          and recording.get('enabled') is True, 'Original native RTX recording required')
    check(recording.get('completed') is False and recording.get('status') == 'recording'
          and recording.get('error') is None, 'Live preview requires an active, error-free recording snapshot')
    check(recording.get('sampling_rate_hz') == 10 and recording.get('stride') == 5
          and close(recording.get('control_period_s'), .02)
          and set(recording.get('camera_ids', [])) == set(CAMERAS), 'Expected task/overview at 10 Hz')
    check(recording.get('frames_file') == 'frames.jsonl' and hash_value(recording.get('frames_sha256')),
          'Live frame index is not identified')
    committed_count, committed_step = recording.get('frame_count'), recording.get('last_control_step')
    check(type(committed_count) is int and committed_count > 1 and type(committed_step) is int
          and committed_step >= through_state_index and committed_step % 5 == 0
          and committed_count == committed_step // 5 + 1
          and close(recording.get('sim_duration_s'), committed_step * .02), 'Requested boundary is not committed')
    rows = []
    with frames_path.open('rb') as stream:
        for _ in range(committed_count):
            row = stream.readline()
            check(row.endswith(b'\n'), 'Committed frame index row is incomplete')
            rows.append(row)
    committed_bytes = b''.join(rows)
    check(hashlib.sha256(committed_bytes).hexdigest() == recording['frames_sha256'],
          'Committed frame index hash differs from its metadata snapshot')
    prefix_bytes = b''.join(rows[:through_state_index // 5 + 1])
    frames = [json.loads(row) for row in rows[:through_state_index // 5 + 1]]
    validate_frames(frames, list(range(0, through_state_index + 1, 5)), recording_dir)
    immutable = (config_path, ready_path, verification_path, spec_path, snapshot_manifest)
    return {'episode': episode, 'client': client, 'recording_dir': recording_dir, 'recording': recording,
            'frames': frames, 'spec': spec, 'result': None, 'report': None,
            'duration_s': through_state_index * .02, 'control_steps': through_state_index,
            'source_commit': source_commit, 'is_partial_preview': True,
            'sources': {str(path): digest(path) for path in immutable},
            'prefix_bytes': prefix_bytes, 'recording_snapshot_bytes': metadata_raw,
            'preview_prefix': {'through_state_index': through_state_index,
                'through_sim_time_s': through_state_index * .02, 'included_frame_count': len(frames),
                'frames_file': str(frames_path), 'prefix_byte_count': len(prefix_bytes),
                'prefix_sha256': hashlib.sha256(prefix_bytes).hexdigest(),
                'recording_metadata_path': str(recording_path),
                'recording_metadata_snapshot_sha256': hashlib.sha256(metadata_raw).hexdigest(),
                'metadata_committed_frame_count': committed_count, 'metadata_committed_state_index': committed_step,
                'metadata_committed_index_sha256': recording['frames_sha256'],
                'metadata_committed_index_hash_verified': True,
                'source_append_after_boundary_allowed': True,
                'score_at_boundary': frames[-1]['score']}}


def verify_live_prefix(source):
    prefix = source['preview_prefix']
    with Path(prefix['frames_file']).open('rb') as stream:
        raw = stream.read(prefix['prefix_byte_count'])
    check(len(raw) == prefix['prefix_byte_count']
          and hashlib.sha256(raw).hexdigest() == prefix['prefix_sha256'], 'Selected live prefix changed while exporting')


@lru_cache(maxsize=8)
def font(size):
    return ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default()


def outcome_label(source):
    result = source['result']
    if result.get('outcome') == 'success' and result.get('physics_valid') is True:
        label = '成功'
        if source['report'].get('run_status') != 'completed':
            label += '（运行异常）'
        return label
    labels = {'cancelled': '取消（未完成）', 'task_failure': '任务失败',
              'sim_timeout': '仿真预算耗尽', 'wall_timeout': '时间预算耗尽',
              'action_budget': '动作预算耗尽', 'model_budget': '模型预算耗尽',
              'infrastructure_error': '基础设施异常', 'evaluator_invalid': '评分证据无效',
              'invalid_action': '动作无效', 'policy_error': '控制器异常', 'safety_stop': '安全停止'}
    label = labels.get(result.get('outcome'), '未完成')
    if result.get('physics_valid') is not True:
        label += ' · 物理证据无效'
    return label


def frame_card(source, frame, speed):
    image = Image.new('RGB', (WIDTH, HEIGHT), '#142333')
    drawing = ImageDraw.Draw(image)
    drawing.text((20, 8), 'Task0 · 公开函数基线 · gpt-6-astra / 请求 max', font=font(27), fill='white')
    partial = source.get('is_partial_preview', False)
    outcome = None if partial else outcome_label(source)
    final_count = source['frames'][-1]['score']['placed_count']
    label = (f"进行中片段 · 截至 {source['duration_s']:.2f} 秒 · 当前 {final_count}/3" if partial
             else f'最终：{outcome} · 最终放置 {final_count}/3')
    drawing.text((1050, 10), label, font=font(25), fill='#f3cb76')
    drawing.text((20, 44), f'{speed}× 仿真时间 · 原回合 RTX 10 Hz 采样 · 已省略模型等待', font=font(22), fill='#a9e3df')
    for index, camera in enumerate(CAMERAS):
        row = frame['cameras'][camera]
        path = inside(source['recording_dir'], row['path'])
        raw = path.read_bytes()
        check(hashlib.sha256(raw).hexdigest() == row['sha256'], 'Camera source changed while exporting')
        with Image.open(BytesIO(raw)) as picture:
            fitted = ImageOps.contain(picture.convert('RGB'), (960, 720), Image.Resampling.LANCZOS)
            image.paste(fitted, (index * 960 + (960 - fitted.width) // 2, 80 + (720 - fitted.height) // 2))
        drawing.rectangle((index * 960 + 8, 88, index * 960 + 144, 120), fill='#142333')
        drawing.text((index * 960 + 16, 90), '任务近景' if camera == 'task' else '整机环境', font=font(22), fill='white')
    status = ('进行中片段 · 任务仍继续，尚无终局结果' if partial
              else f"最终：{outcome} · {source['report'].get('run_status', 'unknown')}")
    drawing.text((20, 805), f"t={frame['sim_time_s']:.2f} / {source['duration_s']:.2f} s   放置 {frame['score']['placed_count']}/3   {status}",
                 font=font(22), fill='#d9e9f0')
    return image


def export_demo(source, output, *, speed=2):
    check(type(speed) is int and speed in (1, 2), 'Playback speed must be 1 or 2')
    output = Path(output).resolve()
    check(output.suffix == '.mp4', 'Use an MP4 output path')
    partial = source.get('is_partial_preview', False)
    sidecar = output.with_suffix('.json')
    terminal = output.with_suffix('.boundary.png' if partial else '.terminal.png')
    evidence = ([output.with_suffix('.frames.prefix.jsonl'), output.with_suffix('.recording.snapshot.json')]
                if partial else [])
    check(not any(path.exists() for path in (output, sidecar, terminal, *evidence)), 'Existing demo artifacts cannot be overwritten')
    check(not output.is_relative_to(source['episode']) and not output.is_relative_to(source['client']),
          'Output must remain outside the source episode/client')
    output.parent.mkdir(parents=True, exist_ok=True)
    if partial:
        verify_live_prefix(source)
    # 100 fps at 2x is an encoding time grid, NOT 100 Hz capture. It represents
    # every final 20 ms source interval exactly without extending the episode.
    fps = 50 * speed
    mappings = []
    with tempfile.TemporaryDirectory(prefix='.task0-export-', dir=output.parent) as temporary:
        temporary = Path(temporary)
        video = temporary / 'video.mp4'
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-threads', '2',
                   '-filter_threads', '1', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                   '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate', str(fps), '-i', 'pipe:0',
                   '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-threads', '2',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
        with (temporary / 'ffmpeg.log').open('w+') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                for frame, following in zip(source['frames'], source['frames'][1:]):
                    start, end = frame['state_index'], following['state_index']
                    raw = frame_card(source, frame, speed).tobytes()
                    for _ in range(end - start):
                        process.stdin.write(raw)
                    mappings.append({'source_frame_index': frame['frame_index'], 'sim_time_s': frame['sim_time_s'],
                                     'output_start_frame': start, 'output_end_frame_exclusive': end})
                process.stdin.close()
                code = process.wait()
                if code:
                    log.seek(0)
                    raise RuntimeError('ffmpeg encoding failed: ' + log.read()[-2000:])
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                raise
        information = probe(video)
        stream = information['streams'][0]
        duration = source['duration_s'] / speed
        check(int(stream['nb_frames']) == source['control_steps']
              and Fraction(stream['avg_frame_rate']) == fps
              and abs(float(information['format']['duration']) - duration) <= .001,
              'Encoded video frame count, rate or terminal duration differs')
        terminal_file = temporary / 'terminal.png'
        frame_card(source, source['frames'][-1], speed).save(terminal_file)
        for path, expected in source['sources'].items():
            check(digest(path) == expected, 'Sealed source changed while exporting: ' + path)
        if partial:
            verify_live_prefix(source)
        result, run = source['result'], source['report']
        spec = source['spec'] if partial else result['spec']
        report = {'schema': 'task0_original_episode_demo.v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'video': str(output), 'video_sha256': digest(video), 'ffprobe': information,
                  'source_episode_id': spec['episode_id'],
                  'source_episode_result_path': None if partial else str(source['episode'] / 'agent/result.json'),
                  'source_commit': source['source_commit'], 'source_commit_evidence': 'operator_declared; frozen source manifest and import-verification hashes recorded separately',
                  'source_files_sha256': source['sources'], 'exporter_sha256': digest(__file__),
                  'source_result_outcome': None if partial else result['outcome'],
                  'source_physics_valid': None if partial else result['physics_valid'],
                  'source_run_status': None if partial else run.get('run_status'),
                  'source_termination': None if partial else run.get('termination'),
                  'final_placed_count': None if partial else source['frames'][-1]['score']['placed_count'],
                  'model': 'gpt-6-astra', 'requested_reasoning_effort': 'max', 'actual_effort_verified': False,
                  'agent_condition': 'task0_public_function_baseline', 'original_execution_video': True,
                  'capture_source': 'original_native_episode_rtx', 'capture_sampling_rate_hz': 10,
                  'control_period_s': .02, 'playback_speed': speed, 'speed_timebase': 'simulation_time',
                  'physical_duration_s': source['duration_s'], 'encoded_duration_s': duration,
                  'encoded_frames_per_second': fps, 'encoded_frame_count': source['control_steps'],
                  'captured_frame_count_per_camera': len(source['frames']), 'camera_ids': list(CAMERAS),
                  'wall_wait_omitted': True, 'state_replay': False, 'physics_interpolation': False,
                  'temporal_image_interpolation': False, 'spatial_resize': 'aspect-preserving Lanczos, letterboxed',
                  'outcome_based_frame_selection': False,
                  'timing_semantics': 'Each original 10 Hz sample is displayed until the next sample on the 20 ms control grid. Encoded duplicates are sample display, not extra captures. Every recorded physical interval is included, including failed actions/recovery. The terminal sample is a separate PNG; no extra terminal hold is encoded.',
                  'terminal_frame_padding_s': 0, 'terminal_image': str(terminal), 'terminal_image_sha256': digest(terminal_file),
                  'frame_map': mappings}
        staged_evidence = []
        if partial:
            report.update(is_partial_preview=True, episode_final_result=None,
                          current_placed_count=source['frames'][-1]['score']['placed_count'],
                          preview_prefix=source['preview_prefix'],
                          preview_boundary_image=str(terminal), preview_boundary_image_sha256=digest(terminal_file),
                          preview_prefix_index=str(evidence[0]), recording_metadata_snapshot=str(evidence[1]),
                          episode_completeness_claimed=False,
                          preview_scope='Committed recording prefix only; no sealed episode or final result is claimed.',
                          timing_semantics='All original sampled intervals from reset through the explicit preview boundary are retained. Each 10 Hz sample is displayed until the next sample on the 20 ms control grid. The boundary still is separate; no extra display hold or terminal result is added.')
            report.pop('terminal_image')
            report.pop('terminal_image_sha256')
            for index, raw in enumerate((source['prefix_bytes'], source['recording_snapshot_bytes'])):
                path = temporary / f'evidence_{index}'
                path.write_bytes(raw)
                staged_evidence.append((path, evidence[index]))
        metadata = temporary / 'video.json'
        metadata.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        # Hard links publish without replacing an existing artifact in a race.
        for staged, destination in ((video, output), (terminal_file, terminal), *staged_evidence, (metadata, sidecar)):
            os.link(staged, destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--playback-speed', type=int, choices=(1, 2), default=2)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--live-prefix', action='store_true', help='Export an explicitly labelled ongoing prefix, without a final result')
    parser.add_argument('--through-state-index', type=int, help='Last committed 10 Hz boundary for a live prefix')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(DEMO_ROOT):
        parser.error('Output must be under shared demo/interface/agent_demos')
    if args.live_prefix != (args.through_state_index is not None):
        parser.error('--live-prefix and --through-state-index must be supplied together')
    source = (load_live_prefix(args.episode, args.client, args.source_commit, args.through_state_index)
              if args.live_prefix else load_sources(args.episode, args.client, args.source_commit))
    if args.verify_only:
        print(json.dumps({'verified': True, 'source_frame_count': len(source['frames']),
                          'sim_duration_s': source['duration_s'], 'native_started': False}))
        return
    report = export_demo(source, args.output, speed=args.playback_speed)
    print(json.dumps({key: report[key] for key in ('video', 'video_sha256', 'encoded_duration_s',
                                                'source_result_outcome', 'source_run_status')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
