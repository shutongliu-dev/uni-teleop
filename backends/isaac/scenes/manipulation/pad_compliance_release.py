"""Derive a release whose SHARPA elastomer pads use a PhysX compliant contact.

v12 box lift (runs/mount_v2_baseline_01/box_lift_r2) hit a bimanual rigid-squeeze
artifact: rigid pads, a rigid carton and two stiff position-controlled arms form
a closed chain, the solver's pad impulses grew to ~7 kN per substep and the
carton was ejected (physics audit invalid). Benchmarks avoid this rigid two-arm
squeeze (handles, support from below, force-driven floating hands); PhysX
recommends compliant contacts for such unstable contacts, and NVIDIA's own SHARPA
example models the elastomer pads as a compliant layer. Replaying the failing
window (runs/dual_contact_replay_04) with a 5e3 N/m pad spring lowered the
per-step impulse from 640.6 to 3.8 N*s and the carton speed from 1.20 to 0.010
m/s. Research note: EXP/research_bimanual_squeeze_20261015.md.

This copies a release and changes only its physics profile:
* robot_material_binding collider_instance_v1, so the pads really get robot_pad
  (the legacy link binding resolved every collider to robot_shell);
* robot_pad_compliance on the robot_pad material (uncalibrated: the SHARPA pad
  stiffness is not published). r2 uses 2e4 N/m, 40 N*s/m per contact point
  (replay: 14.6 N*s per step, carton 0.038 m/s). For scale only: NVIDIA's
  estimated 5e8 N/m^3 pad pressure gradient times an assumed 0.25-1 cm^2
  fingertip contact area gives ~1e4-5e4 N/m per pad.
  r1 (5e3 N/m, 20 N*s/m; runs/pad_compliance_baseline_01) was too soft: the box
  lift held without ejection but the pads sank 4-6 mm (physics audit invalid,
  2 mm gate), and the single-arm top-down pinch stalled with the index pad ~1 mm
  short of the cube (lift, place and grasp failed; push and dual lift passed).
Scenes, tasks, evaluator files, assets and initial states stay byte-identical.
A separate stratum: never pool its scores with the source's.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import digest, dump, validate
from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
EXP = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/distortion_repair_20261014T043445Z')
BINDING = 'collider_instance_v1'
PAD_COMPLIANCE = {'model': 'force_spring_v1', 'stiffness': 20000., 'damping': 40.}
R1_PAD_COMPLIANCE = {'model': 'force_spring_v1', 'stiffness': 5000., 'damping': 20.}   # too soft (see above)
EVIDENCE = ('runs/dual_contact_replay_04/pad_force_k2e4_c40/report.json', 'runs/dual_contact_replay_04/pad_force_k5e3_c20/report.json',
            'runs/dual_contact_replay_04/rebind_control/report.json', 'runs/material_binding_01/bimanual.json',
            'runs/pad_compliance_baseline_01/NOTE.md', 'research_bimanual_squeeze_20261015.md')


def compliant_profile(profile, compliance=PAD_COMPLIANCE):
    from bench.simulation.isaacsim_backend.robot_materials import material_options
    result = deepcopy(profile)
    options = result['isaacsim_options']
    if options.get('robot_contact_materials') is None:
        raise ValueError('Source physics profile has no robot_contact_materials to make compliant')
    if 'robot_material_binding' in options or 'robot_pad_compliance' in options:
        raise ValueError('Source physics profile already sets the robot material opt-ins')
    options.update(robot_material_binding=BINDING, robot_pad_compliance=deepcopy(compliance))
    material_options(options)
    result.setdefault('embodiment_assumptions', {})['robot_pad_compliance'] = (
        f'SHARPA *_elastomer pads: PhysX compliant contact {compliance["model"]}, stiffness {compliance["stiffness"]:g} and '
        f'damping {compliance["damping"]:g} per contact point; uncalibrated (no published pad stiffness). Pads bound to robot_pad '
        'per collider instance prim.')
    return result


def prepare(source, output, compliance=PAD_COMPLIANCE):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new release path; destination exists: ' + str(output))
    old = validate(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.preparing-', dir=output.parent))
    try:
        shutil.rmtree(staging)
        shutil.copytree(source, staging, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
        manifest = load_json(staging / 'release_manifest.json')
        profile_path = staging / old['physics_profile']
        dump(profile_path, compliant_profile(load_json(profile_path), compliance))
        upstream = {'source_release': manifest.get('source_release'), 'source_release_sha256': manifest.get('source_release_sha256')}
        shutil.copy2(source / 'release_manifest.json', staging / 'source_release_manifest.json')
        provenance = {'protocol': 'pad_compliance_release_v1', 'robot_material_binding': BINDING,
                      'robot_pad_compliance': deepcopy(compliance),
                      'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
                      'source_release_upstream': upstream,
                      'changed': ['Physics profile isaacsim_options += robot_material_binding, robot_pad_compliance; '
                                  'embodiment_assumptions += robot_pad_compliance.'],
                      'unchanged': ['Scenes, tasks, evaluator files, asset hashes and initial states (byte-identical).'],
                      'evidence': {str(EXP / p): digest(EXP / p) for p in EVIDENCE if (EXP / p).exists()},
                      'generator_sources': {str(p): digest(p) for p in (Path(__file__),
                          PROJECT / 'bench/simulation/isaacsim_backend/robot_materials.py',
                          PROJECT / 'bench/simulation/isaacsim_backend/scene.py')},
                      'calibrated': False, 'physics_accepted': False, 'task_policy_success_claimed': False}
        dump(staging / 'pad_compliance_provenance.json', provenance)
        manifest['files']['pad_compliance_provenance.json'] = {'sha256': digest(staging / 'pad_compliance_provenance.json')}
        manifest.update(release_id=f"{manifest['release_id']}__pad_compliance_{output.name}",
                        status='prepared_unvalidated', prepared_at_utc=datetime.now(timezone.utc).isoformat(),
                        source_release=str(source), source_release_sha256=digest(source / 'release_manifest.json'),
                        original_release_modified=False, physics_profile_sha256=digest(profile_path),
                        robot_pad_compliance=deepcopy(compliance), physics_accepted=False, task_policy_success_claimed=False,
                        purpose=(manifest.get('purpose', '') + ' Compliant SHARPA elastomer pads (PhysX compliant contact); '
                                 'separate stratum from the rigid-pad source.').strip())
        dump(staging / 'release_manifest.json', manifest)
        validate(staging)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output)
    print(json.dumps({'release': str(args.output), 'release_id': manifest['release_id'],
                      'tasks': [t['id'] for t in manifest['tasks']], 'status': manifest['status']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
