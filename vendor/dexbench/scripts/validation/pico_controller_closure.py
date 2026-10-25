"""Native open/pinch/close/reopen cycle; robot-only, no grasp qualification.

Use the ordinary PICO launcher environment and a frozen source snapshot. Run
headless with --no-video --profile; --review-dir also records contact/keyframe
inspection evidence. No hardware input or task-object commands are used.
"""
import json
import time

import numpy as np

from bench.adapters.pico.protocol import Frame
from bench.control.factory import public_robot_geometry, robot_configuration
from scripts.teleop.pico_isaac import load_inputs, parser, run


class TriggerCycle:
    rejected = 0

    def __init__(self):
        self.sequence = 0

    def read(self):
        self.sequence += 1
        amount = .5 if self.sequence <= 200 else 1. if self.sequence <= 500 else 0.
        tracking = {'head': [0., 0., 0., 0., 0., 0., 1.], 'hands': {},
                    'controllers': {'right': {'pose': [0., 0., 0., 0., 0., 0., 1.],
                                              'grip': 1., 'trigger': amount}}}
        return Frame('closure-check', self.sequence, time.monotonic_ns(), str(self.sequence),
                     'SYNTHETIC_NO_HEADSET', tracking)

    def health(self):
        return {'source': 'synthetic_trigger_cycle'}

    def close(self):
        pass


def check(output, model):
    rows = [json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
    samples = {}
    for amount in (.5, 1., 0.):
        row = [r for r in rows if r['input']['tracking']['controllers']['right']['trigger'] == amount][-1]
        measured = row['measured_joints']
        tips = model.public_finger_geometry(measured)['fingertip_positions_m']
        samples[str(amount)] = {'thumb_index_reference_distance_m': float(np.linalg.norm(
            np.array(tips['thumb'])-tips['index'])),
            'max_joint_target_error_rad': max(abs(measured[n]-row['action']['joint_positions_rad'][n])
                                              for n in model.hand.names)}
    complete = all(r['contact_sampling']['sampled_substeps'] == 40
                   and r['contact_sampling']['complete'] for r in rows)
    result = {'source_kind': 'synthetic_no_hardware', 'samples': samples,
              'contact_sampling_complete': complete,
              'safety_triggered': any(r['safety_trigger'] for r in rows),
              'steps': len(rows), 'task_success': None, 'formal_evaluation': False}
    result['passed'] = bool(complete and not result['safety_triggered']
        and samples['1.0']['thumb_index_reference_distance_m'] < .025
        and samples['0.5']['thumb_index_reference_distance_m'] > .055
        and samples['0.0']['thumb_index_reference_distance_m'] > .07
        and all(s['max_joint_target_error_rad'] < .08 for s in samples.values()))
    (output/'closure_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise RuntimeError('Controller closure cycle failed; inspect measured motion and contacts')


if __name__ == '__main__':
    args = parser().parse_args()
    args.mode, args.start_active, args.profile = 'controllers', True, True
    args.steps = args.steps or 420
    scene, _, _ = load_inputs(args)
    model = public_robot_geometry(robot_configuration(scene))
    if hasattr(model, 'sides'):
        raise ValueError('This native cycle checks a single right hand')
    run(args, receiver=TriggerCycle(), source_kind='synthetic_no_hardware',
        on_finish=lambda output: check(output, model))
