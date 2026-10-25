"""Make an uncropped presentation from pinned, actual RGB episode recordings.

This changes titles/layout only. It does not launch a simulator, interpolate
states/images, change frame timing, or modify acquisition artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from bench.paths import DATA_ROOT, load_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.isaacsim.release import digest, dump
from bench.isaacsim.video import FONT, encode_frames, concat_videos
from bench.isaacsim import video as video_module

DEFAULT_FRAMES = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/presentations/baseline_v1_20261010')


def snapshot_composer(base):
    hashes, snapshots = {}, {}
    for source in (Path(__file__).resolve(), Path(video_module.__file__).resolve()):
        sha = digest(source)
        destination = Path(base)/'composer_source_snapshot'/sha/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
        if digest(destination) != sha:
            raise ValueError(f'Composer snapshot hash mismatch: {destination}')
        hashes[str(source)] = sha
        snapshots[str(source)] = str(destination.resolve())
    return hashes, snapshots


def verify_composer_unchanged(hashes):
    if any(digest(path) != sha for path, sha in hashes.items()):
        raise ValueError('Composer changed while generating the presentation')


def compose(rgb, overview, task, score, time_s, phase):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    is_overview = phase == 'nominal'
    primary, secondary = (overview, rgb) if is_overview else (rgb, overview)
    image = Image.new('RGB', (1600, 1140), '#152332')
    main = ImageOps.contain(primary.convert('RGB'), (1600, 900), Image.Resampling.LANCZOS)
    image.paste(main, ((1600-main.width)//2, 110+(900-main.height)//2))
    drawing = ImageDraw.Draw(image)

    def text(x, y, value, size=24, fill='#f2f6fa'):
        font = ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default()
        drawing.text((x, y), value, font=font, fill=fill)

    mode = {'nominal': '任务初态 · 关节控制检查', 'qualification_probe': '重置后的接触检查 · 独立实验',
            'contact_probe_push': '食指接触推移 · 独立实验', 'contact_probe_hover': '悬空运动负对照 · 独立实验'}[phase]
    text(28, 14, task['title_zh'], 36)
    text(28, 64, 'Isaac Sim  /  FR3 + SHARPA  /  ' + mode, 23)
    text(1130, 27, f't = {time_s:5.2f} s   |   1× 正常速度', 24, '#8ce4e4')
    text(1130, 66, f"任务步骤 {len(score['completed_steps'])} / {len(task['steps'])}", 23)
    text(24, 122, '整机环境' if is_overview else '任务近景', 22, '#ffffff')
    inset = ImageOps.contain(secondary.convert('RGB'), (400, 225), Image.Resampling.LANCZOS)
    drawing.rectangle((1176, 130, 1576, 390), fill='#152332')
    image.paste(inset, (1176+(400-inset.width)//2, 165+(225-inset.height)//2))
    text(1188, 133, '任务近景' if is_overview else '整机环境', 20)
    for index, step in enumerate(task['steps']):
        done = step['id'] in score['completed_steps']
        text(28+(index % 2)*800, 1024+(index//2)*44,
             f"{index+1}. {step['instruction_zh']}"+('  ✓' if done else ''),
             24, '#7bebba' if done else '#e0e9f1')
    return image


def present(episodes, output, frames_root=None):
    from PIL import Image
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a fresh presentation directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = Path(frames_root or DEFAULT_FRAMES/output.name).resolve()
    if frame_dir.exists() and any(frame_dir.iterdir()):
        raise FileExistsError(f'Choose a fresh presentation frame directory: {frame_dir}')
    frame_dir.mkdir(parents=True, exist_ok=True)
    composer_hashes, composer_snapshots = snapshot_composer(frame_dir.parent)
    sources, frames = [], []
    fps = None
    for episode in episodes:
        episode = Path(episode).resolve()
        pins = load_json(episode/'artifact_hashes.json')
        for name in ('recording.json', 'scores.json', 'states.npz', 'task.json', 'result.json', 'metadata.json'):
            if pins.get(name) != digest(episode/name):
                raise ValueError(f'Acquisition hash does not verify: {episode/name}')
        recording = load_json(episode/'recording.json')
        if recording['playback_speed'] != 1. or recording['still_holds'] or recording['slow_motion_segments'] or recording['physics_interpolation']:
            raise ValueError('Only original real-time, uninterpolated recordings are accepted')
        if fps is not None and recording['fps'] != fps:
            raise ValueError('Use separate presentations for recordings with different frame rates')
        fps = recording['fps']
        task = load_json(episode/'task.json')
        scores = load_json(episode/'scores.json')
        result = load_json(episode/'result.json')
        phase = result['phase']
        if phase == 'contact_probe':
            phase += '_'+load_json(episode/'diagnostic.json')['mode']
        start = len(frames)
        for row in recording['frames']:
            paths = [Path(row['raw_rgb']).resolve(), Path(row['overview_rgb']).resolve()]
            expected = [row['raw_sha256'], row['overview_sha256']]
            for path, sha in zip(paths, expected):
                if not path.is_relative_to(episode) or digest(path) != sha or pins.get(str(path.relative_to(episode))) != sha:
                    raise ValueError(f'Raw RGB acquisition hash does not verify: {path}')
            index = len(frames)
            path = frame_dir/f'{index:06d}.png'
            with Image.open(paths[0]) as rgb, Image.open(paths[1]) as overview:
                compose(rgb, overview, task, scores[row['state_index']], row['physical_time_s'], phase).save(path)
            frames.append({'frame': index, 'video_time_s': index/fps, 'episode': str(episode),
                           'source_frame': row['frame'], 'state_index': row['state_index'],
                           'physical_time_s': row['physical_time_s'], 'raw_rgb': str(paths[0]), 'raw_sha256': expected[0],
                           'overview_rgb': str(paths[1]), 'overview_sha256': expected[1],
                           'main_camera': 'overview' if phase == 'nominal' else 'task',
                           'presentation_frame': str(path), 'presentation_sha256': digest(path)})
        sources.append({'episode': str(episode), 'task': task['id'], 'phase': phase,
                        'artifact_hashes_sha256': digest(episode/'artifact_hashes.json'),
                        'recording_sha256': digest(episode/'recording.json'),
                        'states_sha256': pins['states.npz'], 'start_frame': start, 'frame_count': len(frames)-start,
                        'original_video': recording['video'], 'original_video_sha256': recording['video_sha256']})
    if not frames:
        raise ValueError('No recorded RGB frames')
    video = output/'video.mp4'
    info = encode_frames(frame_dir, video, fps)
    verify_composer_unchanged(composer_hashes)
    dump(output/'derivation_manifest.json', {'type': 'rgb_presentation_derivation', 'sources': sources, 'frames': frames,
         'fps': fps, 'playback_speed': 1., 'still_holds': [], 'slow_motion_segments': [], 'physics_interpolation': False,
         'generated_or_replayed_physics': False, 'image_operation': 'Aspect-preserving fit without cropping; titles and picture-in-picture only.',
         'source_episodes_modified': False, 'policy_success_claimed': False,
         'frame_directory': str(frame_dir), 'poster': str(frame_dir/'000000.png'),
         'composer_sources': composer_hashes, 'composer_source_snapshots': composer_snapshots,
         'font': {'path': str(FONT), 'sha256': digest(FONT)} if FONT.exists() else None,
         'video': str(video), 'video_sha256': digest(video), 'ffprobe': info})
    print(json.dumps({'output': str(output), 'frames': len(frames), 'fps': fps, 'video': str(video)}, indent=2))


def combine(manifests, output):
    """Reuse exported task videos/PNG paths without creating a second frame set."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a fresh combined presentation directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    inputs, sources, frames, videos, fps = [], [], [], [], None
    for path in manifests:
        path = Path(path).resolve()
        item = load_json(path)
        if fps is not None and item['fps'] != fps:
            raise ValueError('Combined presentation inputs have different frame rates')
        fps = item['fps']
        if digest(item['video']) != item['video_sha256']:
            raise ValueError(f'Presentation video hash changed: {item["video"]}')
        offset = len(frames)
        sources.extend({**row, 'start_frame': row['start_frame']+offset} for row in item['sources'])
        for row in item['frames']:
            index = len(frames)
            frames.append({**row, 'frame': index, 'video_time_s': index/fps})
        inputs.append({'manifest': str(path), 'manifest_sha256': digest(path),
                       'video': item['video'], 'video_sha256': item['video_sha256']})
        videos.append(item['video'])
    if not frames:
        raise ValueError('No presentation frames to combine')
    composer_hashes, composer_snapshots = snapshot_composer(Path(frames[0]['presentation_frame']).parent.parent)
    video = output/'video.mp4'
    info = concat_videos(videos, video)
    video.with_suffix('.concat.txt').unlink()
    verify_composer_unchanged(composer_hashes)
    dump(output/'derivation_manifest.json', {
        'type': 'rgb_presentation_concatenation', 'source_presentations': inputs,
        'sources': sources, 'frames': frames, 'fps': fps, 'playback_speed': 1.,
        'still_holds': [], 'slow_motion_segments': [], 'physics_interpolation': False,
        'generated_or_replayed_physics': False, 'source_episodes_modified': False,
        'additional_pngs_created': False, 'policy_success_claimed': False,
        'composer_sources': composer_hashes, 'composer_source_snapshots': composer_snapshots,
        'poster': frames[0]['presentation_frame'], 'video': str(video), 'video_sha256': digest(video), 'ffprobe': info})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--episode', type=Path, action='append')
    selection.add_argument('--suite', type=Path)
    parser.add_argument('--contact-push', type=Path, help='Optional separate recorded active-push presentation')
    parser.add_argument('--contact-hover', type=Path, help='Optional separate recorded hover presentation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames-root', type=Path, default=DEFAULT_FRAMES,
                        help='Persistent presentation PNGs belong in the shared datasets directory')
    args = parser.parse_args()
    if args.episode:
        if args.contact_push or args.contact_hover:
            parser.error('Contact presentations are separate exports selected with --suite')
        present(args.episode, args.output, args.frames_root/args.output.name)
        return
    suite = load_json(args.suite)
    if suite['status'] != 'completed' or any(not x['episode_ok'] for x in suite['episodes']):
        raise ValueError('Only a completed suite with all episodes recorded is accepted')
    grouped = {}
    for row in suite['episodes']:
        grouped.setdefault(row['task'], []).append(Path(row['output']))
    if len(grouped) != 4:
        raise ValueError('The final suite presentation requires all four distinct tasks')
    exports = []
    for task, episodes in grouped.items():
        present(episodes, args.output/task, args.frames_root/task)
        exports.append(args.output/task/'derivation_manifest.json')
    combine(exports, args.output/'four_tasks')
    exports.append(args.output/'four_tasks/derivation_manifest.json')
    for name, episode in [('active_push', args.contact_push), ('hover_control', args.contact_hover)]:
        if episode:
            present([episode], args.output/name, args.frames_root/name)
            exports.append(args.output/name/'derivation_manifest.json')
    dump(args.output/'derivation_index.json', {
        'suite': str(args.suite.resolve()), 'suite_sha256': digest(args.suite),
        'derivations': {str(p.resolve()): digest(p) for p in exports},
        'source_episodes_modified': False, 'policy_success_claimed': False,
        'playback_speed': 1., 'still_holds': [], 'slow_motion_segments': []})


if __name__ == '__main__':
    main()
