"""Human teleoperation review ledger; never a benchmark qualification or scorer."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from bench.paths import output_root, portable


def append_event(root, event):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), **event},
                                ensure_ascii=False, allow_nan=False) + '\n')


def initialize(root):
    from scenes.workrooms.catalog import list_tasks
    tasks = [{key: portable(task[key]) for key in
              ('family', 'task_id', 'title', 'source_release', 'objects', 'robot_requirements')}
             | {'instruction': task['rules'].get('intent', ''), 'status': 'pending'}
             for task in list_tasks()]
    root.mkdir(parents=True, exist_ok=True)
    # An existing campaign is immutable: new releases need a new campaign directory.
    with (root / 'catalog.json').open('x') as stream:
        json.dump({'kind': 'human_asset_review', 'formal_evaluation': False, 'tasks': tasks},
                  stream, ensure_ascii=False, indent=2)
    return tasks


def note(root, session, *, target, check, outcome, detail):
    report = json.loads((session / 'session.json').read_text())
    catalog = json.loads((root / 'catalog.json').read_text())
    task = next(t for t in catalog['tasks']
                if (t['family'], t['task_id']) == (report['family'], report['task']))
    scene = json.loads((session / 'scene.json').read_text())
    # Compare the release identity frozen into the session, not today's source tree.
    if (portable(report['release']) != task['source_release']['path'] or
            report.get('review_release_sha256') != task['source_release']['manifest_sha256']):
        raise ValueError('Session release differs from this review campaign')
    append_event(root, {'event': 'operator_observation', 'session': str(session.resolve()),
                       'family': report['family'], 'task': report['task'],
                       'scene_kind': report['scene_kind'], 'scene_variant': scene.get('teleop_scene_variant'),
                       'target': target, 'check': check, 'outcome': outcome, 'detail': detail,
                       'evidence': 'operator_report', 'formal_evaluation': False})


def check_release(root, family, task_id, release):
    catalog = json.loads((root / 'catalog.json').read_text())
    task = next(t for t in catalog['tasks'] if (t['family'], t['task_id']) == (family, task_id))
    digest = hashlib.sha256((release / 'release_manifest.json').read_bytes()).hexdigest()
    if (portable(str(release)) != task['source_release']['path'] or
            digest != task['source_release']['manifest_sha256']):
        raise ValueError('Release changed; start a new review campaign')
    return digest


class Recorder:
    """Reuse existing readback and render; bounded asynchronous 1 Hz keyframes.

    State rows describe rigid objects and contact reports, not a complete soft-body
    replay. Human marks and task completion remain separate from these traces.
    """
    def __init__(self, output):
        self.output = output
        (output / 'review_images').mkdir()
        self.stream = (output / 'review_states.jsonl').open('x', buffering=1)
        self.images = (output / 'review_images.jsonl').open('x', buffering=1)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='review-images')
        self.pending = None
        self.next_image = 0
        self.count = 0

    def state(self, observation, episode, *, event='step'):
        self.stream.write(json.dumps({'event': event, 'episode': episode,
            'host_ns': time.monotonic_ns(), **{key: observation[key] for key in
            ('sim_time_s', 'robot', 'objects', 'contacts', 'terminated', 'truncated')},
            'task_state': observation.get('task_state')}, allow_nan=False) + '\n')

    def image(self, pixels, observation, episode):
        now = time.monotonic()
        if now < self.next_image or (self.pending and not self.pending.done()):
            return
        if self.pending:
            self.pending.result()  # Surface disk/encoding failures rather than silently losing evidence.
        from PIL import Image
        name = f'review_images/{self.count:06d}.jpg'
        frame = Image.fromarray(pixels.copy())
        self.pending = self.pool.submit(frame.save, self.output / name, quality=80)
        self.images.write(json.dumps({'file': name, 'episode': episode,
            'host_ns': time.monotonic_ns(), 'sim_time_s': observation['sim_time_s']}) + '\n')
        self.count += 1
        self.next_image = now + 1.

    def close(self):
        self.pool.shutdown(wait=True)
        self.stream.close()
        self.images.close()
        if self.pending:
            self.pending.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=output_root() / 'teleop_review')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('init')
    commands.add_parser('status')
    record = commands.add_parser('note')
    record.add_argument('--session', required=True, type=Path)
    record.add_argument('--target', required=True, help='Object name, fixture, or whole_task')
    record.add_argument('--check', choices=('grasp_place', 'mechanism', 'full_task', 'room', 'control'), required=True)
    record.add_argument('--outcome', choices=('pass', 'needs_fix', 'retest'), required=True)
    record.add_argument('--detail', required=True, help='What happened; include trial count and approximate time')
    args = parser.parse_args()
    if args.command == 'init':
        print(f'Created {len(initialize(args.root))} pending tasks in {args.root}')
    elif args.command == 'note':
        note(args.root, args.session, target=args.target, check=args.check,
             outcome=args.outcome, detail=args.detail)
    else:
        catalog = json.loads((args.root / 'catalog.json').read_text())
        events_path = args.root / 'events.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        for task in catalog['tasks']:
            observations = [e for e in events if e.get('event') == 'operator_observation'
                            and (e['family'], e['task']) == (task['family'], task['task_id'])]
            # Counts preserve failures and partial passes; no automatic whole-task promotion.
            counts = {value: sum(e['outcome'] == value for e in observations)
                      for value in ('pass', 'needs_fix', 'retest')}
            print(f"{task['family']}/{task['task_id']}: {counts if observations else 'pending'}")


if __name__ == '__main__':
    main()
