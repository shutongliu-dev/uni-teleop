"""Native controller rotation-pivot check; synthetic input, no grasp claims."""
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from bench.adapters.pico.controller_hand import ControllerHand
from bench.adapters.pico.protocol import Frame
from bench.adapters.pico.retarget import wrist_matrix
from bench.control.factory import public_robot_geometry, robot_configuration
from scripts.teleop.pico_isaac import load_inputs, parser, run


class RotationInput:
    rejected = 0

    def __init__(self):
        self.sequence = 0

    def read(self):
        self.sequence += 1
        angle = min(max(self.sequence-20, 0)*.001, .06)
        pose = [0., 0., 0.] + Rotation.from_euler('y', angle).as_quat().tolist()
        tracking = {'head': [0., 0., 0., 0., 0., 0., 1.], 'hands': {},
                    'controllers': {'right': {'pose': pose, 'grip': 1., 'trigger': 0.}}}
        return Frame('pivot-check', self.sequence, time.monotonic_ns(), str(self.sequence),
                     'SYNTHETIC_NO_HEADSET', tracking)

    def health(self):
        return {'source': 'synthetic_rotation'}

    def close(self):
        pass


def check(output, model, locked):
    rows = [json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
    reference = rows[0]['measured_joints']
    tips = model.public_geometry(reference | ControllerHand(model).prepose)['fingertip_positions_m']
    offset = (np.array(tips['thumb']) + tips['index']) / 2
    poses = [wrist_matrix(model.public_geometry(row['measured_joints'])) for row in rows]
    centers = np.array([p[:3, 3] + p[:3, :3] @ offset for p in poses])
    drift = float(np.max(np.linalg.norm(centers-centers[0], axis=1)))
    angle = float(Rotation.from_matrix(poses[-1][:3, :3] @ poses[0][:3, :3].T).magnitude())
    complete = all(r['contact_sampling']['complete'] and r['contact_sampling']['sampled_substeps'] == 40 for r in rows)
    result = {'source_kind': 'synthetic_no_hardware', 'lock_orientation': locked,
              'max_measured_pinch_center_drift_m': drift, 'final_rotation_rad': angle,
              'contact_sampling_complete': complete, 'steps': len(rows),
              'safety_triggered': any(r['safety_trigger'] for r in rows),
              'task_success': None, 'formal_evaluation': False}
    result['passed'] = bool(len(rows) == 120 and complete and not result['safety_triggered']
                            and drift < .008 and (angle < .01 if locked else .04 < angle < .08))
    (output/'control_point_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise RuntimeError('Control point native check failed; inspect recorded motion')


if __name__ == '__main__':
    args = parser().parse_args()
    args.mode, args.control_point = 'controllers', 'pinch'
    args.start_active, args.profile, args.steps = True, True, 120
    scene, _, _ = load_inputs(args)
    model = public_robot_geometry(robot_configuration(scene))
    if hasattr(model, 'sides'):
        raise ValueError('This native check uses a single right hand')
    run(args, receiver=RotationInput(), source_kind='synthetic_no_hardware',
        on_finish=lambda output: check(output, model, args.lock_orientation))
