"""Render archived rigid states without re-executing their actions."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np


def render_archive(env, session, output=None):
    from PIL import Image
    from bench.simulation.isaacsim_backend.visual_replay import restore
    from uni_teleop.video.isaac import picture_in_picture
    session = Path(session)
    root = session/'replay'
    info = json.loads((root/'manifest.json').read_text())
    if info['schema'] != 'rigid_visual_replay.v1':
        raise ValueError('Unsupported replay schema')
    for name, digest in info.get('session_files_sha256', {}).items():
        if hashlib.sha256((session/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen session file changed: '+name)
    output = Path(output) if output else session/'replay_images'
    output.mkdir(exist_ok=False)
    count = 0
    before = (env.world.current_time_step_index, env.world.current_time)
    with (output/'frames.jsonl').open('x') as index, (root/'frames.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            with np.load(root/row['file'], allow_pickle=False) as archive:
                state = dict(archive)
            restore(env, state, info)
            frames = env.render_many(tuple(info['cameras']))
            paths = {}
            for camera, pixels in frames.items():
                name = f'{row["frame"]:06d}_{camera}.png'
                Image.fromarray(pixels).save(output/name)
                paths[camera] = name
            if 'pico_inset' in frames:
                pixels = picture_in_picture(frames['pico_left'], frames['pico_inset'], 'WRIST')
                name = f'{row["frame"]:06d}_review_left.png'
                Image.fromarray(pixels).save(output/name)
                paths['review_left'] = name
            index.write(json.dumps(dict(row, images=paths))+'\n')
            count += 1
    after = (env.world.current_time_step_index, env.world.current_time)
    if before != after:
        raise RuntimeError('Offline rendering unexpectedly advanced physics')
    result = {'frames': count, 'physics_steps': 0, 'status': 'complete',
              'format': 'lossless PNG', 'action_alignment': 'state_before_action'}
    (output/'result.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    scene = json.loads((args.session/'scene.json').read_text())
    initial = json.loads((args.session/'initial_state.json').read_text())
    env = FR3DexterousIsaacEnv(scene, device='cuda:0', headless=True)
    try:
        report = json.loads((args.session/'session.json').read_text())
        env.reset(seed=report.get('settings', {}).get('seed', 0), initial_state=initial)
        env.prepare_cameras()
        print(render_archive(env, args.session, args.output), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
