"""One PICO full-body/SONIC loop, selectable MuJoCo or Isaac physics."""
import argparse
from dataclasses import replace
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['isaac','mujoco'], required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, required=True)
    parser.add_argument('--endpoint', help='External PICO ZMQ source instead of local SDK')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('runs/sonic'))
    args = parser.parse_args()
    from uni_teleop.mapping.sonic import SonicController, SonicEncoder
    from uni_teleop.devices.pico.xr import PicoTeleop, PicoEvents
    from uni_teleop.devices.pico.zmq import PicoZmqTeleop
    from uni_teleop.control.sonic import TeleopMode, next_mode
    from contextlib import ExitStack
    with ExitStack() as stack:
        if args.backend == 'isaac':
            from uni_teleop.backends.isaac.g1 import IsaacG1Env
            env = IsaacG1Env(output=args.output, headless=args.headless)
        else:
            from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv
            env = MujocoG1EmptyEnv()
        stack.callback(env.close)
        encoder = SonicEncoder.from_onnx(args.encoder)
        controller = SonicController.from_onnx(args.decoder)
        teleop = PicoZmqTeleop(args.endpoint) if args.endpoint else PicoTeleop()
        stack.callback(teleop.close)
        direct = not args.endpoint
        mode = TeleopMode.OFF if direct else TeleopMode.POSE
        completed = 0
        last_packet = 0.
        print('SONIC: A+B+X+Y to arm; A+X to enter POSE. Missing input freezes simulation after 150 ms.', flush=True)
        while not args.steps or completed < args.steps:
            tick = time.monotonic()
            packet = teleop.read()
            if packet is not None: last_packet = tick
            events = teleop.pop_events() if direct else PicoEvents()
            updated = next_mode(mode, events)
            if events.reset_scene:
                env.reset(); encoder.reset(); controller.reset(); last_packet = 0.
            if updated != mode:
                encoder.reset(); controller.reset()
                mode = updated
            if mode == TeleopMode.POSE and tick-last_packet <= .15:
                state = env.get_robot_state()
                try:
                    token = encoder.encode(state, packet)
                except RuntimeError:
                    token = None  # Encoder has not received its first complete reference batch.
                if token is not None:
                    command = controller.act(state, token)
                    if packet is not None:
                        command = replace(command,
                            hand_joint_position=None if packet.hand_joint_position is None else packet.hand_joint_position[-1],
                            neck_joint_position=None if packet.neck_joint_position is None else packet.neck_joint_position[-1])
                    env.step(command, steps=controller.steps_per_action)
                    completed += 1
            if not args.headless: env.render()
            time.sleep(max(0., .02-(time.monotonic()-tick)))


if __name__ == '__main__': main()
