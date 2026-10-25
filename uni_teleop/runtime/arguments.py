"""Shared Franka teleoperation options; importing this module never loads a simulator."""
import argparse
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--family', default='kitchen')
    result.add_argument('--task', default='breakfast_serving')
    result.add_argument('--sandbox', action='store_true', help='Use a table-edge robot and two objects for diagnostics; no benchmark task or score')
    result.add_argument('--workbench-only', action='store_true',
                        help='Load the versioned task workbench without unrelated room scenery')
    result.add_argument('--no-task-scoring', action='store_true',
                        help='Unscored rigid practice; keep physical contact safety checks')
    result.add_argument('--hide-task-displays', action='store_true',
                        help='Hide task-driven displays for unscored practice without task observation')
    result.add_argument('--keep-task-observer', action='store_true',
                        help='Keep task-driven visuals and observation during unscored practice')
    result.add_argument('--arm-base-yaw', type=float, metavar='DEG',
                        help='Absolute base heading in degrees for unscored practice')
    result.add_argument('--arm-home', type=float, nargs=7, metavar='RAD',
                        help='Explicit single-arm initial posture for unscored practice')
    result.add_argument('--practice-contact-warnings', action='store_true',
                        help='Unscored practice: report penetrations without pausing')
    result.add_argument('--physics-profile', choices=('scene', 'cpu_rigid_v1'), default='scene',
                        help='Explicit operator-only physics override; scene preserves the release configuration')
    result.add_argument('--mode', choices=('hands', 'controllers'), default='hands')
    result.add_argument('--bind', action='append', metavar='ROBOT=PICO', help='E.g. right=right or front=right')
    result.add_argument('--port', type=int, default=5557)
    result.add_argument('--scale', type=float, default=1.)
    result.add_argument('--control-point', choices=('wrist', 'pinch'), default='wrist',
                        help='Controller rotation pivot: wrist or fixed pinch-prepose center')
    result.add_argument('--lock-orientation', action='store_true',
                        help='Controllers translate without rotating the calibrated hand orientation')
    result.add_argument('--headless', action='store_true')
    result.add_argument('--start-active', action='store_true', help='Arm at the first fresh frame; GUI defaults to paused')
    result.add_argument('--steps', type=int, default=0, help='Stop after this many native steps (0: interactive)')
    result.add_argument('--output', type=Path)
    result.add_argument('--camera', default='overview', help='Main stereo camera (default: fixed overview)')
    result.add_argument('--side-view', action='store_true', help='Fixed side overview for comparing leader and simulated arm posture')
    result.add_argument('--inset-camera', default='wrist', help='Mono picture-in-picture camera; use none to disable')
    result.add_argument('--rear-camera', action='store_true', help='Additional fixed desktop view from behind the arm toward the workbench')
    result.add_argument('--head-tracked', action='store_true', help='Move a world-camera view with the headset; incompatible with wrist cameras')
    result.add_argument('--width', type=int, default=640)
    result.add_argument('--height', type=int, default=480)
    result.add_argument('--video-fps', type=float, default=30.)
    result.add_argument('--video-listen', default='0.0.0.0:13579')
    result.add_argument('--no-video', action='store_true')
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--profile', action='store_true', help='Record render, mapping and native-step wall times in session evidence')
    result.add_argument('--review-dir', type=Path, help='Existing human review campaign; record rigid states, contacts and 1 Hz keyframes')
    return result
