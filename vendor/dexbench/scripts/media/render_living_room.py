#!/usr/bin/env python3
"""Photograph a pinned static living room with the existing native Isaac RTX renderer.

Use --gpu-uuid GPU-... to select a physical device, or CUDA_VISIBLE_DEVICES.
All other arguments, including --cameras room workcell tabletop, are passed to
the shared static renderer. The default is the room photograph only.
An optional static robot composition may be photographed. No live articulation,
PhysicsScene, controller or model call is started.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from bench.paths import load_json

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from scripts.media.render_card_room import main as render_static
from scripts.media.render_workroom import digest, dump


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--gpu-uuid', help='Physical GPU UUID; overrides CUDA_VISIBLE_DEVICES')
    parser.add_argument('--_native-worker', action='store_true', help=argparse.SUPPRESS)
    options, forwarded = parser.parse_known_args()
    if options.gpu_uuid:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_uuid
    if '-h' in forwarded or '--help' in forwarded:
        print('Living-room option: --gpu-uuid GPU-... (physical device; default: CUDA_VISIBLE_DEVICES)\n')
    if not any(value == '--cameras' or value.startswith('--cameras=') for value in forwarded):
        forwarded.extend(['--cameras', 'room'])
    # This parser only reads values needed for the additional receipt. The
    # shared renderer remains responsible for validation and native settings.
    receipt = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    receipt.add_argument('--asset', type=Path)
    receipt.add_argument('--output', type=Path)
    receipt.add_argument('--samples', type=int, default=128)
    args, _ = receipt.parse_known_args(forwarded)
    if options._native_worker or '-h' in forwarded or '--help' in forwarded:
        sys.argv = [sys.argv[0], *forwarded]
        return render_static(expected_family='living_room')
    if args.asset is not None and args.output is not None and args.output.resolve().exists():
        raise FileExistsError('Preserve previous attempts; use a new output directory')
    # Isaac fast_shutdown calls os._exit, so the native renderer runs in its
    # own process. The parent survives to attach the light/settings receipt.
    # Reuse the pinned launch script to preserve its compatibility loader.
    result = subprocess.run([str(PROJECT / 'scripts/isaac_python.sh'), '-m',
        'scripts.media.render_living_room', '--_native-worker', *forwarded],
        check=False).returncode
    if args.asset is None or args.output is None:
        # The shared parser has emitted its own required-argument error.
        return result
    # Camera poses, native sample counts, GPU identity and the complete source
    # snapshot are already recorded by render_static. Add the composed light
    # inputs from the hashed asset; do not claim requested settings as readback.
    provenance = args.output.resolve() / 'provenance.json'
    if provenance.is_file():
        report = load_json(provenance)
        description = args.asset / 'scene_description.json'
        scene = load_json(description)
        report['lighting'] = scene['lighting']
        report['authored_lights'] = scene['authored_lights']
        report['lighting_source'] = {'path': str(description.resolve()), 'sha256': digest(description)}
        report['render_settings_requested'] = {
            'renderer': 'PathTracing', 'samples_per_pixel': args.samples,
            'samples_per_iteration': min(args.samples, 32), 'clamp_spp': args.samples,
            'max_bounces': 8, 'denoising_enabled': True, 'adaptive_sampling_enabled': False,
            'scope': 'Existing static renderer configuration; actual sample counts are recorded per image'}
        dump(provenance, report)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
