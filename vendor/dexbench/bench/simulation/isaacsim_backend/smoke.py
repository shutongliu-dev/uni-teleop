"""Run a native Isaac Sim four-task scene and record auditable solver states."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np

from bench.paths import load_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--task', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--duration', type=float, default=.2)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--joint-delta', type=float, default=0.)
    p.add_argument('--render', action='store_true')
    p.add_argument('--reset-state', type=Path)
    a = p.parse_args()
    from .env import FR3DexterousIsaacEnv
    from bench.sequence_evaluator import SequenceEvaluator
    output = a.output
    output.mkdir(parents=True, exist_ok=False)
    scene = load_json(a.scene)
    (output / 'scene.json').write_text(json.dumps(scene, indent=2) + '\n')
    task = load_json(a.task) if a.task else None
    evaluator = SequenceEvaluator(task) if task else None
    env = FR3DexterousIsaacEnv(scene, device=a.device)
    observations, actions, states, scores = [], [], [], []
    start = time.perf_counter()
    try:
        initial = load_json(a.reset_state) if a.reset_state else None
        obs = env.reset(initial_state=initial)
        (output / 'metadata.json').write_text(json.dumps(env.metadata(), indent=2) + '\n')
        if a.render:
            from PIL import Image
            Image.fromarray(env.render('overview')).save(output / 'initial.png')
        for i in range(round(a.duration / .02) + 1):
            if i:
                targets = dict(env.home)
                targets['fr3_joint1'] += a.joint_delta
                action = {'joint_positions_rad': targets, 'duration_s': .02}
                actions.append(action)
                obs = env.step(action)
            observations.append(obs)
            states.append(env.capture_state())
            if evaluator:
                scores.append(evaluator.update(obs))
            if not obs['metrics']['state_finite']:
                raise RuntimeError('Native PhysX state became nonfinite')
            if i % 10 == 0:
                print(json.dumps({'frame': i, 'time_s': obs['time_s'], 'contact_points': obs['contacts']['rigid_contact_point_count']}), flush=True)
        np.savez_compressed(output / 'states.npz', **{k: np.stack([s[k] for s in states]) for k in states[0]})
        np.save(output / 'triangles.npy', env.triangles)
        for name, values in [('observations', observations), ('actions', actions), ('scores', scores)]:
            (output / f'{name}.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in values))
        if a.render:
            Image.fromarray(env.render('task')).save(output / 'final.png')
        result = {'native_isaacsim': True, 'simulation_duration_s': env.time_s,
                  'frames': len(states), 'wall_s': time.perf_counter()-start,
                  'joint1_actual_delta_rad': float(states[-1]['joint_q'][env.joint_names.index('fr3_joint1')] - states[0]['joint_q'][env.joint_names.index('fr3_joint1')]),
                  'cloth_max_node_displacement_m': float(np.linalg.norm(states[-1]['particle_q']-states[0]['particle_q'], axis=1).max()) if env.has_cloth else None,
                  'rigid_contact_report_count': sum(len(o['contacts']['rigid']) for o in observations),
                  'contact_errors': [e for o in observations for e in o['contacts']['errors']],
                  'final_score': scores[-1] if scores else None,
                  'object_state_writes_after_reset': 0}
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        env.stage.GetRootLayer().Export(str(output / 'scene.usda'))
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
