"""Declared robot-reset push diagnostic and matched hover control on Isaac task 01."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controllers import push_plan
from .release import DEFAULT, PROJECT, digest, dump, task_inputs
from .run import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=DEFAULT)
    parser.add_argument('--mode', choices=['push', 'hover'], default='push')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--demo', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tip-z', type=float, default=.778)
    parser.add_argument('--distance', type=float, default=.10)
    parser.add_argument('--integral-gain', type=float, default=4.)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    _, scene, task, _ = task_inputs(args.release, '01', 'nominal')
    initial, actions, trace, residuals = push_plan(scene, mode=args.mode, tip_z=args.tip_z, distance=args.distance)
    plan_dir = args.output.with_name(args.output.name + '_plan')
    if plan_dir.exists() and any(plan_dir.iterdir()):
        raise FileExistsError(f'Use a fresh diagnostic plan directory: {plan_dir}')
    plan_dir.mkdir(parents=True, exist_ok=True)
    dump(plan_dir / 'planned_actions.json', actions)
    diagnostic = {'id': 'isaac_task01_index_spool_' + args.mode, 'mode': args.mode, 'task_id': task['id'],
                  'phase': 'contact_probe', 'reset_override': {'joint_positions_rad': initial},
                  'robot_reset': 'Near-spool fingertip IK pose; explicitly differs from nominal robot home.',
                  'object_reset_override': False, 'cloth_reset_override': False, 'scene_and_objects_unchanged': True,
                  'integral_gain_s_inv': args.integral_gain, 'max_integral_offset_rad': .10,
                  'max_joint_target_velocity_rad_s': .65, 'max_joint_target_acceleration_rad_s2': 2.,
                  'feedback_observation': 'Actual robot joint positions only; arm error integral as position-target offset.',
                  'planned_trace': trace, 'ik_residuals': residuals, 'tip_z_m': args.tip_z, 'distance_m': args.distance,
                  'hover_z_offset_m': .08 if args.mode == 'hover' else 0.,
                  'state_writes_after_reset': False, 'attachments': False, 'sequential_task_success_claimed': False,
                  'contact_gate': 'Require actual hand-spool solver pair/impulse, displacement after contact, stable released object; table contact alone is insufficient.',
                  'planned_actions_sha256': digest(plan_dir / 'planned_actions.json'),
                  'kinematics_sources': {str(path): digest(path) for path in [PROJECT / 'bench/control/__init__.py', PROJECT / 'bench/control/kinematics.py', PROJECT / 'bench/geometry.py', Path(__file__).with_name('controllers.py')]},
                  'design_reference': {'path': str(Path(__file__).with_name('controllers.py')), 'sha256': digest(Path(__file__).with_name('controllers.py')),
                                       'use': 'Current pure URDF fingertip planner and fixed diagnostic durations; no recorded object motion is used.'}}
    dump(plan_dir / 'diagnostic.json', diagnostic)
    print(json.dumps({'plan': str(plan_dir), 'action_count': len(actions), 'duration_s': len(actions) * .02, 'ik': residuals}, indent=2), flush=True)
    if args.plan_only:
        return
    args.task = '01'; args.phase = 'contact_probe'; args.baseline = 'hold'; args.seed = 0
    args.actions = plan_dir / 'planned_actions.json'; args.duration = len(actions) * .02
    args.diagnostic = diagnostic
    rollout(args)


if __name__ == '__main__':
    main()
