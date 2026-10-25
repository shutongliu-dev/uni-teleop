"""Native embodiment import, joint response, mimic and contact validation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from bench.paths import DATA_ROOT, load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hand', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--disable-self-collision', action='store_true')
    parser.add_argument('--zero-passive', action='store_true', help='Explicit diagnostic: zero native hand joint friction properties')
    parser.add_argument('--assembly-version', help='Registered assembly (default: the hand\'s v1-era assembly)')
    parser.add_argument('--release-collision-filters', action='store_true',
                        help="Apply the hand's audited release collision filters, as its releases do")
    args = parser.parse_args()
    from .robot_registry import scene_with_hand
    from .env import FR3DexterousIsaacEnv
    release = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/sequential_workcell/v1')
    base = load_json(args.scene or release / 'tasks/01_ordered_kitting/scene.json')
    base['isaacsim'] = load_json(release / 'physics_profile.json')['isaacsim_options']
    scene = scene_with_hand(base, args.hand, assembly_version=args.assembly_version)
    if args.release_collision_filters:
        from .robot_registry import get_hand_spec
        scene['isaacsim'].update(get_hand_spec(args.hand, args.assembly_version).get('release_collision_filters', {}))
    if args.disable_self_collision:
        scene['isaacsim'].update(robot_self_collisions=False, filter_thumb_cmc_structure=False)
    args.output.mkdir(parents=True, exist_ok=False)
    source_snapshot = args.output / 'source_snapshot'
    source_snapshot.mkdir()
    source_hashes = {}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        data = path.read_bytes(); (source_snapshot/path.name).write_bytes(data)
        source_hashes[str(path)] = hashlib.sha256(data).hexdigest()
    (args.output / 'scene.json').write_text(json.dumps(scene, indent=2) + '\n')
    started = time.perf_counter()
    env = FR3DexterousIsaacEnv(scene, device=args.device, render_width=960, render_height=720)
    try:
        if args.zero_passive:
            friction = env._robot_view.get_dof_friction_properties().clone()
            original = friction.cpu().numpy().tolist()
            for i,n in enumerate(env.joint_names):
                if not n.startswith('fr3_joint'): friction[0,i,:] = 0
            env._robot_view.set_dof_friction_properties(friction, env._tensor(np.array([0],dtype=np.int32),friction.device))
            (args.output / 'diagnostic_override.json').write_text(json.dumps({'zero_hand_passive_friction':True, 'original_native_friction':original, 'property_order':['static','dynamic','viscous']},indent=2)+'\n')
        (args.output / 'initial_metadata.json').write_text(json.dumps(env.metadata(), indent=2) + '\n')
        from PIL import Image
        Image.fromarray(env.render('overview')).save(args.output / 'initial.png')
        spec = env.action_spec()
        home = spec['initial_joint_positions_rad']
        states, contacts, actions = [env.capture_state()], [], []
        obs = env.observation()
        # Hold, then ramp all independently controllable joints away from a limit.
        for i in range(60):
            q = dict(home)
            if i >= 10:
                fraction = min((i - 9) / 20, 1.)
                for n in q:
                    lo, hi = spec['joint_limits_rad'][n]
                    direction = 1. if q[n] + .1 < hi else -1.
                    q[n] += direction * (.03 if n.startswith('fr3_joint') else .10) * fraction
            action = {'joint_positions_rad': q, 'duration_s': .02}
            obs = env.step(action)
            actions.append(action); states.append(env.capture_state()); contacts.append(obs['contacts'])
            if not obs['metrics']['state_finite']:
                print('Nonfinite native state: diagnostic failed', flush=True)
                break
            if i % 10 == 0:
                print(json.dumps({'hand': args.hand, 'time': env.time_s, 'contacts': obs['contacts']['rigid_contact_point_count']}), flush=True)
        metadata = env.metadata()
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        from PIL import Image
        before = env.world.current_time_step_index
        Image.fromarray(env.render('overview')).save(args.output / 'overview.png')
        render_isolation = env.world.current_time_step_index == before
        np.savez_compressed(args.output / 'states.npz', **{k: np.stack([s[k] for s in states]) for k in states[0]})
        (args.output / 'actions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in actions))
        (args.output / 'contacts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in contacts))
        active = spec['joint_names']
        responses = {n: float((states[-1]['joint_q'][env.joint_names.index(n)]-states[min(10,len(states)-1)]['joint_q'][env.joint_names.index(n)]) * np.sign(actions[-1]['joint_positions_rad'][n]-home[n])) for n in active}
        mimics = {n: float(max(abs(s['joint_q'][env.joint_names.index(n)] - m['multiplier']*s['joint_q'][env.joint_names.index(m['joint'])] - m['offset']) for s in states)) for n,m in spec['mimic_joints'].items()}
        errors = [e for c in contacts for e in c['errors']]
        self_contacts = [r for c in contacts for r in c['rigid'] if all('Robot/' in r.get(k,'') for k in ('actor0','actor1'))]
        points = sum(c['rigid_contact_point_count'] for c in contacts)
        positive = sum(c['rigid_impulse_magnitude_sum_ns'] for c in contacts)
        finite = all(np.isfinite(v).all() for state in states for v in state.values())
        object_drift = {n: float(np.linalg.norm(states[-1]['object_q'][i,:3]-states[0]['object_q'][i,:3])) for i,n in enumerate(env.object_names)}
        reset = env.reset(); reset_state = env.capture_state()
        reset_matches = all(np.allclose(reset_state[k], states[0][k], atol=1e-6) for k in ('joint_q','object_q','particle_q'))
        result = {'hand_id': args.hand, 'assembly_version': scene['robot'].get('assembly_version'),
                  'self_contact_rows': len(self_contacts),
                  'positive_self_contact_pairs': sorted({tuple(sorted((r['actor0'].split('/')[-1], r['actor1'].split('/')[-1])))
                                                         for r in self_contacts if r.get('impulse_magnitude_sum_ns', 0.) > 0}),
                  'native_isaacsim': True, 'duration_s': float(states[-1]['time_s']), 'physical_joint_count': len(env.joint_names), 'actuated_joint_count': len(active),
                  'all_active_joints_respond': all(v > 1e-4 for v in responses.values()), 'joint_response_rad': responses,
                  'state_finite': finite, 'mimic_max_error_rad': mimics, 'mimic_constraints_pass': all(v < .01 for v in mimics.values()),
                  'reset_matches': reset_matches, 'render_isolation': render_isolation,
                  'native_contact_points': points, 'native_normal_impulse_sum_ns': positive,
                  'contact_errors': errors, 'object_displacement_m': object_drift,
                  'object_stability_pass': all(v < .01 for v in object_drift.values()),
                  'wall_s': time.perf_counter()-started, 'trained_policy': False,
                  'qualification_scope': 'Native model import, independent joint response, mimic tracking, passive object stability, positive support contacts; not a task-solving policy or hardware fidelity certificate.'}
        result['diagnostic_overrides'] = {'self_collision_disabled': args.disable_self_collision, 'zero_passive_friction': args.zero_passive}
        result['sources_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in source_hashes.items())
        result['source_snapshot_sha256'] = source_hashes
        result['passed'] = not args.disable_self_collision and not args.zero_passive and result['sources_unchanged'] and all(result[k] for k in ('all_active_joints_respond','state_finite','mimic_constraints_pass','reset_matches','render_isolation','object_stability_pass')) and not errors and positive > 0
        result['artifact_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.is_file()}
        (args.output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
