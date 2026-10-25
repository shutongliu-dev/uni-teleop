import argparse
import time

from sonic_mujoco.superdex import DEFAULT_HAND_ENDPOINT, HandCommandPublisher
from sonic_mujoco.teleop import PicoTeleop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish PICO hand tracking to a SuperDex simulation"
    )
    parser.add_argument("--endpoint", default=DEFAULT_HAND_ENDPOINT)
    parser.add_argument("--steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teleop = PicoTeleop(enable_dexhand=True)
    publisher = HandCommandPublisher(args.endpoint)
    completed = 0
    started = False
    print(f"Publishing PICO hand targets on {args.endpoint} ...")
    try:
        while args.steps == 0 or completed < args.steps:
            tick = time.monotonic()
            command = teleop.read()
            if command is not None and command.hand_joint_position is not None:
                publisher.publish(command.hand_joint_position[-1])
                completed += 1
                if not started:
                    print("PICO five-finger tracking is streaming to SuperDex.")
                    started = True
            time.sleep(max(0.0, 0.02 - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        publisher.close()
        teleop.close()
    if started:
        print(f"Stopped after {completed} PICO hand frames.")


if __name__ == "__main__":
    main()
