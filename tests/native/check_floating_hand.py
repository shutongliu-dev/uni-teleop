"""Hardware-free native motion/contact check; run once per simulator Python."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from uni_teleop.backends.sharpa import extract_hand


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--backend', choices=('isaac', 'mujoco'), required=True)
    cli.add_argument('--data', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--scene', choices=('sandbox', 'parts-sorting'), default='sandbox')
    args = cli.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    hand = extract_hand(args.data, args.output)
    if args.backend == 'mujoco':
        from uni_teleop.backends.mujoco.floating_hand import FloatingHand
    else:
        from uni_teleop.backends.isaac.floating_hand import FloatingHand
    options = {}
    if args.scene == 'parts-sorting':
        from uni_teleop.backends.isaac.hand_scene import parts_sorting
        options['bench_scene'] = parts_sorting(args.data)[0]
    env = FloatingHand(hand, args.output, headless=True, **options)
    result = {'backend': args.backend, 'source': 'synthetic', 'task_success': None}
    try:
        p0, q0 = env.pose()
        if args.backend == 'isaac':
            # Reproduce an editor pause with a nonzero held target. Repeated
            # control calls must not bank impulses for the first resumed step.
            target = p0+np.array([.08, 0., .03])
            env.world.pause()
            for _ in range(200): env.step(target, q0, hand.initial)
            np.testing.assert_allclose(env.pose()[0], p0, atol=1e-6)
            env.world.play()
            excursions = []
            for _ in range(100):
                env.step(target, q0, hand.initial)
                excursions.append(float(np.linalg.norm(env.pose()[0]-p0)))
            result['pause_resume_max_excursion_m'] = max(excursions)
            assert max(excursions) < .12, result
            assert np.linalg.norm(env.pose()[0]-target) < .015, result
            for _ in range(100): env.step(p0, q0, hand.initial)
        target = p0+np.array([.07, -.04, .03])
        rotation = (Rotation.from_euler('z', .3)*Rotation.from_quat(q0)).as_quat()
        fingers = np.clip(np.full(22, .4), hand.lower, hand.upper)
        begin = time.monotonic()
        for i in range(120):
            alpha = min(1., (i+1)/60.)
            env.step(p0+(target-p0)*alpha, rotation, fingers)
            if i % 5 == 0: env.render()
        p, q = env.pose()
        result.update(motion_error_m=float(np.linalg.norm(p-target)),
            rotation_error_rad=float((Rotation.from_quat(rotation)*Rotation.from_quat(q).inv()).magnitude()),
            finger_max_error_rad=float(np.max(np.abs(env.positions()-fingers))),
            native_mean_control_s=(time.monotonic()-begin)/120)
        assert result['motion_error_m'] < .015, result
        assert result['rotation_error_rad'] < .05, result
        assert result['finger_max_error_rad'] < .12, result
        # Place the target through the table: physical contacts must prevent
        # teleportation. Reopen fingers first, then descend vertically.
        for _ in range(30): env.step(target, q0, hand.initial)
        down = target.copy(); down[2] = .70
        for i in range(150):
            alpha = min(1., (i+1)/100.)
            env.step(target+(down-target)*alpha, q0, hand.initial)
            if i % 5 == 0: env.render()
        p, _ = env.pose()
        result['blocked_target_gap_m'] = float(p[2]-down[2])
        result['contact_palm_z_m'] = float(p[2])
        if args.backend == 'mujoco':
            result['contact_count'] = env.data.ncon
            result['min_contact_distance_m'] = float(min((c.dist for c in env.data.contact), default=0.))
            assert env.data.ncon > 0, result
            assert result['min_contact_distance_m'] > -.015, result
        assert p[2] > .80 and result['blocked_target_gap_m'] > .10, result
        env.screenshot(args.output/'contact.png')
        # Back out of contact through normal targets and verify recovery.
        for i in range(100): env.step(target, q0, hand.initial)
        result['recovery_error_m'] = float(np.linalg.norm(env.pose()[0]-target))
        assert result['recovery_error_m'] < .015, result
        result['passed'] = True
    except BaseException as error:
        result.update(passed=False, error=str(error))
        raise
    finally:
        (args.output/'check.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        env.close()


if __name__ == '__main__': main()
