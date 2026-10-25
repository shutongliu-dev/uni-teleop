"""Run each scene in a separate Isaac Sim process and package four actual videos."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .release import DEFAULT, PROJECT, digest, dump, validate
from bench.paths import OUTPUT_ROOT, load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--demo', type=Path, default=Path(f'{OUTPUT_ROOT}/media/isaacsim_workcell_v1'))
    parser.add_argument('--launcher', type=Path, default=PROJECT / 'scripts/isaac_python.sh', help='Python-compatible launch script providing the isolated glibc runtime')
    parser.add_argument('--phases', nargs='+', choices=['nominal', 'qualification_probe'], default=['nominal', 'qualification_probe'])
    parser.add_argument('--tasks', nargs='+', help='Task IDs or 01..04; omitted means all four')
    parser.add_argument('--baseline', choices=['hold', 'joint-response'], default='joint-response')
    parser.add_argument('--duration', type=float, default=3.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    manifest = validate(args.release)
    entries = manifest['tasks']
    if args.tasks:
        entries = [entry for entry in entries if any(entry['id'] == value or entry['id'][:2] == value.zfill(2) for value in args.tasks)]
        if len(entries) != len(set(args.tasks)):
            raise ValueError('Task filter contains unknown or duplicate task IDs')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose a new suite run directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    args.demo.mkdir(parents=True, exist_ok=True)
    rows, task_videos = [], []
    for entry in entries:
        phase_videos = []
        for phase in args.phases:
            output, demo = args.output / entry['id'] / phase, args.demo / entry['id']
            command = [str(args.launcher.resolve()) if args.launcher else sys.executable, '-m', 'bench.isaacsim.run',
                       '--release', str(args.release.resolve()), '--task', entry['id'], '--phase', phase,
                       '--output', str(output.resolve()), '--demo', str(demo.resolve()),
                       '--baseline', args.baseline, '--duration', str(args.duration), '--device', args.device,
                       '--fps', str(args.fps), '--width', str(args.width), '--height', str(args.height)]
            if args.no_video:
                command.append('--no-video')
            log = args.output / f"{entry['id']}_{phase}.log"
            print(f"Starting {entry['id']} / {phase}; log: {log}", flush=True)
            with log.open('w') as stream:
                process = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
            row = {'task': entry['id'], 'phase': phase, 'command': command, 'returncode': process.returncode, 'log': str(log), 'output': str(output)}
            result = output / 'result.json'
            if result.exists():
                row['result'] = load_json(result)
                row['result_sha256'] = digest(result)
            recording = output / 'recording.json'
            row['artifact_complete'] = result.exists() and (args.no_video or recording.exists())
            row['episode_ok'] = process.returncode == 0 and row['artifact_complete']
            rows.append(row)
            dump(args.output / 'suite_result.json', {'status': 'running', 'episodes': rows})
            clip = demo / f'{phase}.mp4'
            if not args.no_video and row['episode_ok'] and clip.exists() and load_json(recording)['video_sha256'] == digest(clip):
                phase_videos.append(clip)
        if phase_videos:
            from .video import concat_videos
            video = args.demo / entry['id'] / 'demo.mp4'
            info = concat_videos(phase_videos, video)
            task_videos.append({'task': entry['id'], 'video': str(video), 'sha256': digest(video),
                                'source_clips': [str(x) for x in phase_videos], 'ffprobe': info})
    montage = None
    if task_videos:
        from .video import concat_videos
        path = args.demo / 'four_tasks.mp4'
        info = concat_videos([item['video'] for item in task_videos], path)
        montage = {'path': str(path), 'sha256': digest(path), 'ffprobe': info, 'playback_speed': 1., 'still_holds': [], 'slow_motion_segments': []}
    final = {'status': 'completed' if all(r['episode_ok'] for r in rows) else 'completed_with_failures',
             'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'release': str(args.release.resolve()),
             'release_sha256': digest(args.release / 'release_manifest.json'), 'episodes': rows,
             'task_videos': task_videos, 'montage': montage, 'task_policy_success_claimed': False,
             'qualification_results_excluded_from_nominal': True,
             'notes': 'Hold and small joint-response baseline checks. Videos show actual solver states and RTX frames. This report does not qualify a trained task policy.'}
    dump(args.output / 'suite_result.json', final)
    dump(args.demo / 'manifest.json', final)
    print(json.dumps({'status': final['status'], 'episodes': len(rows), 'task_videos': len(task_videos), 'montage': montage}, ensure_ascii=False, indent=2))
    if any(not r['episode_ok'] for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
