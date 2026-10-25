"""Own the optional PICO acquisition process and one MuJoCo teleop session."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['pico', 'master', 'pico-manus'], required=True)
    args, rest = parser.parse_known_args()
    if rest[:1] == ['--']: rest = rest[1:]
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument('--output', type=Path)
    output_parser.add_argument('--port', default='5557')
    options, _ = output_parser.parse_known_args(rest)
    output = options.output or Path(os.environ['UNI_TELEOP_ROOT'])/'runs/mujoco'/f'{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    output.mkdir(parents=True, exist_ok=True)
    if options.output is None: rest += ['--output', str(output)]
    module = 'uni_teleop.runtime.mujoco_franka' if args.source == 'pico' else 'uni_teleop.runtime.master'
    command = [sys.executable, '-B', '-m', module]
    if args.source != 'pico': command += ['--backend', 'mujoco']
    processes = []
    def stop(signum, frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop)
    try:
        with (output/'pico_sdk.log').open('w') as log:
            if args.source.startswith('pico'):
                processes.append(subprocess.Popen([sys.executable, '-B', '-m', 'uni_teleop.devices.pico.source',
                    '--library', os.environ['PICO_SDK_LIBRARY'], '--port', options.port], stdout=log, stderr=subprocess.STDOUT))
            native = subprocess.Popen(command+rest)
            processes.append(native)
            raise SystemExit(native.wait())
    finally:
        for process in reversed(processes):
            if process.poll() is not None: continue
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


if __name__ == '__main__': main()
