"""Native diagnostic rollout of the scripted joint-only controller.

This development runner preserves failed attempts. The parent Agent API run
provides separate end-to-end acceptance; this script does not claim that path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

from .controller import SimpleTaskController
from bench.paths import load_json


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=native_scalar) + '\n')


def native_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported recorded value: {type(value)}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--variant', choices=['scripted','hold','hover'], default='scripted')
    p.add_argument('--max-duration', type=float, default=40.)
    p.add_argument('--render', action='store_true')
    p.add_argument('--pregrasp-reset', action='store_true',
                   help='Development only: reset robot above cube; never a nominal task result.')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    scene, task = load_json(a.scene), load_json(a.task)
    controller = SimpleTaskController(scene, task, variant=a.variant)
    from .evaluator import SimpleTaskEvaluator
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    evaluator = SimpleTaskEvaluator(task)
    for src, name in ((a.scene, 'scene.json'), (a.task, 'task.json'),
                      (Path(__file__), 'probe_source.py'), (Path(__file__).with_name('controller.py'), 'controller_source.py')):
        shutil.copy2(src, a.output / name)
    dump(a.output/'source_hashes.json', {name: hashlib.sha256((a.output/name).read_bytes()).hexdigest()
        for name in ('scene.json','task.json','probe_source.py','controller_source.py')})
    started = time.monotonic()
    env = None
    states, scores = [], []
    failure = None
    try:
        env = FR3DexterousIsaacEnv(scene, device=a.device, render_width=960, render_height=720)
        obs = env.reset(seed=0)
        reset_override = {}
        if a.pregrasp_reset:
            controller._build_phases(obs)
            _, _, point, hand = controller.phases[0]
            reset_override = {'joint_positions_rad': {**controller._arm_at(point), **hand}}
            obs = env.reset(seed=0, initial_state=reset_override)
            controller = SimpleTaskController(scene, task, variant=a.variant)
            controller.next_action(obs)
            first = controller.phases[0]
            controller.phases[0] = (first[0], .6, first[2], first[3])
        dump(a.output/'reset_override.json', reset_override)
        dump(a.output/'validation_scope.json', {'nominal_robot_reset':not a.pregrasp_reset,
            'diagnostic_robot_reset':a.pregrasp_reset, 'ranked_episode':False,
            'physics_or_object_overrides':False, 'stop_file':str(a.output/'STOP')})
        dump(a.output/'metadata.json', env.metadata())
        phase = None
        with (a.output/'observations.jsonl').open('w') as of, (a.output/'actions.jsonl').open('w') as af:
            for index in range(round(a.max_duration/.02) + 1):
                of.write(json.dumps(obs, allow_nan=False)+'\n')
                score = evaluator.update(obs)
                scores.append(score)
                states.append(env.capture_state())
                if phase != controller.phase or index%100 == 0:
                    print(json.dumps({'step': index, 'sim_s': obs['time_s'], 'phase':controller.phase,
                        'cube':obs['objects']['cube']['position_m'], 'score':score}, allow_nan=False,
                        default=native_scalar), flush=True)
                    if a.render and phase != controller.phase:
                        from PIL import Image
                        Image.fromarray(env.render('task')).save(a.output/f'{index:05d}_{controller.phase}.png')
                    phase = controller.phase
                    dump(a.output/'progress.json', {'score': score, 'controller': controller.diagnostic()})
                if (a.output/'STOP').exists():
                    failure = 'operator_stop_file_interrupted_not_scored'
                    break
                if controller.done or index == round(a.max_duration/.02):
                    break
                action = controller.next_action(obs)
                af.write(json.dumps(action, allow_nan=False)+'\n')
                obs = env.step(action)
                if not obs['metrics']['state_finite']:
                    raise RuntimeError('Nonfinite native state')
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        try:
            if states:
                np.savez_compressed(a.output/'states.npz', **{k:np.stack([s[k] for s in states]) for k in states[0]})
            dump(a.output/'scores.json', scores)
            dump(a.output/'result.json', {'last_score': scores[-1] if scores else None,
                'controller':controller.diagnostic(), 'error': failure, 'wall_seconds': time.monotonic()-started,
                'validation_scope': 'Native controller development, not Agent API or ranked evaluation'})
        finally:
            if env:
                env.close()


if __name__ == '__main__':
    main()
