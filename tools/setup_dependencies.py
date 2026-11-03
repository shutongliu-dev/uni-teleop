"""Prepare isolated local dependencies and write launcher settings; no device access."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def run(*command):
    subprocess.run([str(part) for part in command], check=True)


def environment(interpreter, directory, *, version=None):
    executable = shutil.which(str(interpreter))
    if executable is None:
        raise ValueError(f'Python not found: {interpreter}')
    if version:
        actual = subprocess.check_output([executable, '-c',
            'import sys; print("%s.%s" % sys.version_info[:2])'], text=True).strip()
        if actual != version:
            raise ValueError(f'This optimizer requires CPython {version}, got {actual}')
    python = directory / 'bin/python'
    if not python.exists():
        run(executable, '-m', 'venv', directory)
    return python


def local_package(path, relative, marker):
    path = Path(path).expanduser().resolve()
    if (path / relative / marker).exists():
        path /= relative
    if not (path / marker).exists():
        raise ValueError(f'Expected {marker} inside {path}')
    return path


def download_pico(destination):
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise ValueError('The pinned PICO download is for Linux x86_64 only')
    spec = json.loads((ROOT / 'dependencies/sources.json').read_text())['pico']
    destination.mkdir(parents=True, exist_ok=True)
    for item in spec['files']:
        path = destination / Path(item['path']).name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']:
            continue
        url = spec['raw_base_url'] + item['path']
        with urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Official download checksum mismatch: {item["path"]}')
        path.write_bytes(data)
    return destination / 'libPXREARobotSDK.so'


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--mujoco', action='store_true', help='Create a dedicated MuJoCo environment')
    cli.add_argument('--cpu', action='store_true', help='Create the CPU launcher environment for Franka/Isaac')
    cli.add_argument('--manus-sdk', type=Path, help='Authorized extracted MANUS 3.2.0 SDK or its Python folder')
    cli.add_argument('--sharpa-retarget', type=Path, help='Authorized SHARPA repository or V4 folder')
    cli.add_argument('--pico', action='store_true', help='Download pinned official PICO C SDK into local/')
    cli.add_argument('--pico-library', type=Path, help='Use an existing PICO C SDK library instead')
    cli.add_argument('--data', type=Path, help='Existing DexBench data directory')
    cli.add_argument('--isaac-python', type=Path, help='Existing Isaac Python launcher')
    cli.add_argument('--python', default='python3.12', help='Python for CPU, MuJoCo and MANUS')
    cli.add_argument('--retarget-python', default='python3.10')
    cli.add_argument('--config', type=Path, default=ROOT / 'local/dependencies.json',
                     help='Local flat launcher configuration; existing fields are retained')
    args = cli.parse_args()
    selected = any((args.mujoco, args.cpu, args.manus_sdk, args.sharpa_retarget,
                    args.pico, args.pico_library, args.data, args.isaac_python))
    if not selected:
        cli.error('Select dependencies; see --help')
    settings = json.loads(args.config.read_text()) if args.config.exists() else {}
    if not isinstance(settings, dict) or 'recipe' in settings:
        cli.error('--config must be a flat host settings file, not a recipe/session file')
    local = ROOT / 'local/dependencies'
    local.mkdir(parents=True, exist_ok=True)
    try:
        for name, selected, extra, setting in (
            ('mujoco', args.mujoco, 'mujoco', 'SONIC_PYTHON'),
            ('cpu', args.cpu, 'franka', 'DEXBENCH_PYTHON')):
            if selected:
                python = environment(args.python, local / name)
                run(python, '-m', 'pip', 'install', '-e', f'{ROOT}[{extra}]')
                if name == 'cpu':
                    run(python, '-m', 'pip', 'install', '-e', ROOT / 'backends/isaac')
                settings[setting] = str(python)
        if args.manus_sdk:
            sdk = local_package(args.manus_sdk, 'Python', 'manus_sdk/__init__.py')
            python = environment(args.python, local / 'manus')
            run(python, '-m', 'pip', 'install', '-r', ROOT / 'dependencies/manus.txt')
            settings.update(MANUS_SDK_PYTHON=str(sdk), MANUS_PYTHON=str(python),
                            MANUS_SDK_VERSION='3.2.0')
        if args.sharpa_retarget:
            if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
                raise ValueError('SHARPA V4 binaries require Linux x86_64')
            source = local_package(args.sharpa_retarget, 'retargeting_alg_release_V4.0',
                                   'include/hand_retargeting_optimizer.so')
            python = environment(args.retarget_python, local / 'sharpa', version='3.10')
            run(python, '-m', 'pip', 'install', '-r', ROOT / 'dependencies/sharpa-v4.txt')
            settings.update(SHARPA_RETARGET_ROOT=str(source), SHARPA_RETARGET_PYTHON=str(python))
        if args.pico and args.pico_library:
            raise ValueError('Choose either --pico or --pico-library')
        if args.pico:
            settings['PICO_SDK_LIBRARY'] = str(download_pico(local / 'pico'))
        if args.pico_library:
            library = args.pico_library.expanduser().resolve()
            if not library.is_file():
                raise ValueError(f'PICO library not found: {library}')
            settings['PICO_SDK_LIBRARY'] = str(library)
        if args.data:
            data = local_package(args.data, '.',
                'assets/robots/assemblies/fr3_sharpa/isaacsim_6_1_v2/fr3_sharpa.urdf')
            settings['DEXBENCH_DATA'] = str(data)
        if args.isaac_python:
            path = args.isaac_python.expanduser().resolve()
            if not path.is_file():
                raise ValueError(f'Isaac launcher not found: {path}')
            settings['ISAAC_PYTHON_LAUNCHER'] = str(path)
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(json.dumps(settings, indent=2) + '\n')
        args.config.chmod(0o600)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        cli.exit(1, f'Dependency setup failed: {error}\n')
    print(f'Wrote {args.config}; no SDK initialized and no devices connected.')
    print('Add your glove ID, calibration and GPU settings, then run ./teleop doctor with this config.')


if __name__ == '__main__':
    main()
