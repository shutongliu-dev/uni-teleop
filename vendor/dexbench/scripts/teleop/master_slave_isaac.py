"""Read FACTR leader joints and command a simulated Franka/SHARPA only."""
from functools import partial
from pathlib import Path
import hashlib
import json

from bench.adapters.master_slave.calibration import ArmCalibration
from bench.adapters.master_slave.input import LeaderReceiver
from bench.adapters.master_slave.retarget import JointRetargeter
from scripts.teleop.pico_isaac import parser, run


def main():
    cli = parser()
    cli.add_argument('--arm-source', choices=['factr','pico'], default='factr')
    cli.add_argument('--leader-port')
    cli.add_argument('--leader-baud', type=int, default=4000000)
    cli.add_argument('--leader-practice-follow',action='store_true',
                     help='Unscored simulation: saturate only limited joints and slew toward fresh targets without latching displacement holds')
    cli.add_argument('--unrestricted-arm',action='store_true',
                     help='Practice simulation only: direct arm targets and unlimited native arm joints')
    cli.add_argument('--fixed-reference',action='store_true',help='Keep the saved seven-joint zero across Start/Pause')
    cli.add_argument('--align-from-leader',action='store_true',help='Initialize the simulated arm from fresh calibrated leader joints; stay paused')
    cli.add_argument('--leader-feedback',action='store_true',help='Explicit low-current joint resistance; GUI enables output separately')
    cli.add_argument('--leader-gravity-urdf',type=Path,help='Leader gravity model; requires measured physical joint calibration')
    cli.add_argument('--leader-gravity-calibration', type=Path,
                     help='Independent physical leader calibration for gravity; leaves follower mapping unchanged')
    cli.add_argument('--leader-gravity-current-scale',type=int,choices=[1,2],default=1,
                     help='Gravity-only current cap stage: 1=30/60 mA, 2=60/120 mA (XC/XM)')
    cli.add_argument('--arm-calibration', type=Path, help='Saved, measured fixed arm calibration; required unless hand-only')
    cli.add_argument('--allow-provisional-calibration', action='store_true',
                     help='Explicit simulation practice with approximate reference and unverified arm directions')
    cli.add_argument('--trigger-sign', type=int, choices=[-1, 1], default=1)
    cli.add_argument('--grip-range', type=float, default=.8)
    cli.add_argument('--manus-retarget-root', type=Path, help='External official SHARPA retargeting_alg_release_V4.0')
    cli.add_argument('--manus-retarget-python', type=Path, help='Isolated Python 3.10 for the official optimizer')
    cli.add_argument('--manus-hand-only', action='store_true', help='Test glove fingers with the arm held; no leader serial access')
    cli.add_argument('--manus-sdk-python', type=Path)
    cli.add_argument('--manus-python', type=Path)
    cli.add_argument('--manus-sdk-version', choices=['3.2.0', '3.2.1'], default='3.2.1')
    cli.add_argument('--manus-glove-id', type=int)
    cli.add_argument('--manus-calibration', type=Path)
    cli.add_argument('--manus-finger-speed', type=float, default=4.,
                     help='Finger target rad/s in simulation time, bounded by original URDF speeds')
    cli.add_argument('--capture-reference-on-start',action='store_true',
                     help='One-time operator-matched zero capture on first GUI Start; requires feedback off')
    args = cli.parse_args()
    pico_arm = args.arm_source == 'pico'
    if args.capture_reference_on_start and (pico_arm or args.manus_hand_only or args.leader_feedback or
            args.headless or args.start_active or not (args.sandbox or args.no_task_scoring)):
        cli.error('--capture-reference-on-start requires paused GUI FACTR practice without physical feedback')
    if not pico_arm and not args.manus_hand_only:
        args.fixed_reference=True
    if args.leader_practice_follow and (pico_arm or args.manus_hand_only or not (args.sandbox or args.no_task_scoring)):
        cli.error('--leader-practice-follow requires unscored FACTR simulation')
    if args.unrestricted_arm and not args.leader_practice_follow:
        cli.error('--unrestricted-arm requires --leader-practice-follow')
    if args.align_from_leader and (pico_arm or args.manus_hand_only or not args.fixed_reference or
                                  args.start_active or not (args.sandbox or args.no_task_scoring)):
        cli.error('--align-from-leader requires paused, unscored FACTR practice with --fixed-reference')
    if args.leader_feedback and (pico_arm or args.manus_hand_only or not args.fixed_reference):
        cli.error('--leader-feedback requires FACTR arm and --fixed-reference')
    if not pico_arm:
        args.mode = 'controllers'
    if pico_arm and (args.manus_hand_only or args.manus_sdk_python is None):
        cli.error('PICO+MANUS requires MANUS configuration and an enabled PICO arm')
    if args.head_tracked and not pico_arm:
        cli.error('Master/slave mode uses fixed overview and wrist cameras')
    if args.manus_hand_only and args.manus_sdk_python is None:
        cli.error('--manus-hand-only requires the MANUS SDK configuration')
    if not pico_arm and not args.manus_hand_only and not args.leader_port:
        cli.error('--leader-port is required for master/slave control')
    arm_calibration = None
    gravity=None
    gravity_calibration=None
    if args.leader_gravity_calibration and not args.leader_gravity_urdf:
        cli.error('--leader-gravity-calibration requires --leader-gravity-urdf')
    if args.leader_gravity_current_scale!=1 and not args.leader_gravity_urdf:
        cli.error('Higher current stage requires --leader-gravity-urdf')
    if not pico_arm and not args.manus_hand_only:
        if args.arm_calibration is None:
            cli.error('--arm-calibration is required; Start never creates arm calibration')
        arm_calibration = ArmCalibration.load(args.arm_calibration)
        if arm_calibration.status == 'provisional' and not args.allow_provisional_calibration:
            cli.error('Approximate calibration requires --allow-provisional-calibration')
        if (arm_calibration.source_kind != 'hardware' or
                arm_calibration.leader_id != Path(args.leader_port).name):
            cli.error('Require hardware calibration matching the selected leader serial identity')
        if Path(args.leader_port).parent != Path('/dev/serial/by-id'):
            cli.error('Use the stable /dev/serial/by-id leader port saved in calibration')
    if args.leader_gravity_urdf:
        gravity_calibration = (ArmCalibration.load(args.leader_gravity_calibration)
                               if args.leader_gravity_calibration else arm_calibration)
        if (gravity_calibration is None or gravity_calibration.status!='measured'
                or gravity_calibration.source_kind!='hardware'
                or gravity_calibration.leader_id!=Path(args.leader_port).name or not args.leader_feedback):
            cli.error('Leader gravity requires measured physical joint calibration and --leader-feedback')
        from bench.adapters.master_slave.gravity import LeaderGravity
        gravity=LeaderGravity(args.leader_gravity_urdf)
    if pico_arm:
        from bench.adapters.pico.protocol import Receiver
        receiver = Receiver(args.port)
    else:
        from bench.adapters.master_slave.haptics import HapticReceiver
        from bench.adapters.master_slave.haptic_process import GravityProcessReceiver
        receiver_class=GravityProcessReceiver if gravity else HapticReceiver if args.leader_feedback else LeaderReceiver
        receiver = None if args.manus_hand_only else receiver_class(args.leader_port, baud=args.leader_baud,
            **({'gravity':gravity,'gravity_current_scale':args.leader_gravity_current_scale} if args.leader_feedback else {}),
            **({'gravity_calibration':gravity_calibration} if gravity else {}))
    feedback=receiver if args.leader_feedback else None
    calibration_output=args.output.parent if args.output else None
    if args.manus_sdk_python:
        from bench.adapters.manus.process_source import ManusProcessSource, ManusSourceConfig
        from bench.adapters.master_slave.manus import CombinedReceiver, ManusJointRetargeter
        glove=optimizer=None
        try:
            if args.manus_python is None or args.manus_glove_id is None or args.manus_calibration is None:
                cli.error('MANUS requires Python, glove ID and saved official calibration file')
            calibration = args.manus_calibration.read_bytes()
            text = calibration.decode()
            profile = json.loads(text[text.index('{'):])['gloveProfiles'][f'0x{args.manus_glove_id:08X}']
            if profile['side'] != 'right' or not profile['calibrationProfile']:
                raise ValueError('Require a saved calibration for the selected right glove')
            calibration_id = 'official/'+hashlib.sha256(calibration).hexdigest()
            if args.output is None:
                cli.error('MANUS requires an explicit --output session directory')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # run() creates the session directory; keep SDK diagnostics beside it.
            sdk_log = args.output.parent/'manus_sdk.log'
            glove = ManusProcessSource(ManusSourceConfig(args.manus_sdk_python,
                args.manus_python, args.manus_glove_id, calibration_id,
                sdk_version=args.manus_sdk_version, stderr_log=sdk_log))
            glove.start()
            if args.manus_retarget_root:
                import time
                from bench.adapters.manus.sharpa import SharpaOptimizerSource
                if args.manus_retarget_python is None:
                    cli.error('Official optimizer requires --manus-retarget-python')
                optimizer=SharpaOptimizerSource(glove,args.manus_retarget_python,args.manus_retarget_root,
                                               args.output.parent/'sharpa_optimizer.log')
                optimizer.start()
                deadline=time.monotonic()+15
                while optimizer.metadata is None and optimizer.error is None and time.monotonic()<deadline:
                    time.sleep(.02)
                if optimizer.metadata is None:
                    raise RuntimeError(optimizer.error or 'Official SHARPA optimizer startup timed out')
        except BaseException:
            if optimizer:optimizer.close()
            if glove:glove.close()
            if receiver:
                receiver.close()
            raise
        if pico_arm:
            from bench.adapters.manus.pico_teleop import PicoManusReceiver, PicoManusRetargeter
            run(args, receiver=PicoManusReceiver(receiver, glove, optimizer), source_kind='pico_manus_hardware',
                mapper_factory=partial(PicoManusRetargeter, glove_id=args.manus_glove_id,
                    calibration_id=calibration_id, finger_speed=args.manus_finger_speed,
                    optimizer_metadata=optimizer.metadata if optimizer else None))
            return
        run(args, receiver=CombinedReceiver(receiver, glove, optimizer), source_kind='manus_hand_only_hardware' if args.manus_hand_only else 'factr_manus_hardware',
            feedback=feedback,
            mapper_factory=partial(ManusJointRetargeter, arm_calibration=arm_calibration,
                fixed_reference=args.fixed_reference,calibration_output=calibration_output,follow_on_start=not args.manus_hand_only,
                practice_follow=args.leader_practice_follow,unrestricted_arm=args.unrestricted_arm,
                glove_id=args.manus_glove_id, calibration_id=calibration_id, finger_speed=args.manus_finger_speed, hand_only=args.manus_hand_only,
                optimizer_metadata=optimizer.metadata if optimizer else None))
        return
    run(args, receiver=receiver, source_kind='factr_current_feedback_hardware' if feedback else 'factr_readonly_leader',
        feedback=feedback,
        mapper_factory=partial(JointRetargeter, trigger_sign=args.trigger_sign, arm_calibration=arm_calibration,
            fixed_reference=args.fixed_reference,calibration_output=calibration_output,follow_on_start=True,
            practice_follow=args.leader_practice_follow,unrestricted_arm=args.unrestricted_arm,grip_range=args.grip_range))


if __name__ == '__main__':
    main()
