"""Offline support/settling review of a canonical laboratory reset."""
import numpy as np


def support_evidence(runtime, records, task, dwell_s=.3):
    meta = task['asset_metadata']
    states = [r['observation'] for r in records if r['kind'] == 'step']
    if not states:
        return {'passed': False, 'reason': 'no_native_steps'}
    end = states[-1]['time_s']
    terminal = [r for r in states if r['time_s'] >= end-dwell_s-1.e-8]
    actors = meta['actors']
    loaded = {k: 0 for k in actors}
    paths = {k: set() for k in actors}
    for obs in states:
        for row in obs['contacts']['rigid']:
            for name, path in actors.items():
                pair = (row['actor0'], row['actor1'])
                match = [p == path or p.startswith(path+'/') for p in pair]
                if any(match):
                    index = 0 if match[0] else 1
                    force_z = row['impulse_vector_ns'][2]*(1 if index == 0 else -1)
                    if force_z > 1.e-10:
                        loaded[name] += 1
                        paths[name].add(pair[1-index])
    results = {}
    for name in actors:
        positions = np.asarray([r['objects'][name]['position_m'] for r in terminal])
        excursion = float(np.max(np.linalg.norm(positions-positions[-1], axis=1)))
        vertical_excursion = float(np.ptp(positions[:, 2]))
        receipts = [r['task_state']['native_support_by_actor'][name] for r in terminal]
        quiet = all(r['released_quiet_native_s'] >= .02 for r in receipts)
        direct_support = loaded[name] > 0 and all(r['supported'] for r in receipts)
        # The piston is mechanically held by a declared native prismatic joint,
        # with intentionally filtered barrel/piston contact. This is not a weld
        # to the world and does not create a fictitious contact-force receipt.
        joint_support = (task['id'] == 'pipette_transfer' and name == 'plunger'
            and bool(meta['joints']) and loaded['pipette'] > 0
            and all(r['task_state']['native_support_by_actor']['pipette']['supported'] for r in terminal))
        in_window = vertical_excursion <= .004 if name.startswith('pellet_') else excursion <= .004
        results[name] = {'passed': bool((direct_support or joint_support) and quiet and in_window),
            'native_loaded_contact_rows': loaded[name], 'support_paths': sorted(paths[name]),
            'native_joint_support': joint_support, 'terminal_pose_excursion_m': excursion,
            'terminal_vertical_excursion_m': vertical_excursion,
            'released_quiet_receipts': quiet}
    return {'passed': bool(terminal and terminal[-1]['time_s']-terminal[0]['time_s'] >= dwell_s-.020001
                          and all(r['passed'] for r in results.values())),
        'actors': results, 'scope': 'Canonical native support and released rest; no hand or task execution qualification.'}
