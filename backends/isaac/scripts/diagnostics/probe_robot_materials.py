"""Native pad/shell material readback probe; no task or friction calibration claim.

Run from a complete frozen snapshot. The differentiated mode overrides this
process's scene only, never a release, shared robot asset or default friction.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np

from bench.isaacsim.release import dump
from bench.paths import DATA_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_scenario import digest, seal_report_before_shutdown

BIMANUAL_RELEASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/'
                        'simple_manipulation/v13_pad_compliance_r2/bimanual')
BIMANUAL_MANIFEST_SHA256 = 'be1799c4c3e95c25d468f1622a2a5929b3b3e5c6fb839acd67403c972b1f74eb'
BIMANUAL_TASK = '05_bimanual_box_lift'
BIMANUAL_COMPLIANCE = {'model': 'force_spring_v1', 'stiffness': 20000., 'damping': 40.}


def bimanual_source():
    """Load the original release scene and profile through their hash validator."""
    from bench.isaacsim.release import task_inputs
    manifest = BIMANUAL_RELEASE/'release_manifest.json'
    if digest(manifest) != BIMANUAL_MANIFEST_SHA256:
        raise ValueError('Bimanual release manifest differs from the pinned source')
    entry, spec, _task, initial = task_inputs(BIMANUAL_RELEASE, BIMANUAL_TASK)
    if initial:
        raise ValueError('Bimanual material probe expects the original nominal scene reset')
    return spec, {'scope': 'Original bimanual scene, options and assets validated by task_inputs; no task execution or score.',
                  'release': str(BIMANUAL_RELEASE), 'task_id': BIMANUAL_TASK,
                  'release_manifest_sha256': digest(manifest),
                  'physics_profile_sha256': digest(BIMANUAL_RELEASE/'physics_profile.json'),
                  'scene_sha256': entry['scene_sha256'], 'task_sha256': entry['task_sha256'],
                  'isaacsim_options': deepcopy(spec['isaacsim']), 'scene_modified': False}


def probe_scene(source, mode):
    """Make only a recorded in-memory rigid material override."""
    result = deepcopy(source)
    options = result['isaacsim']
    if mode == 'bimanual_release':
        if (result['robot']['hand_id'] != 'dual_fr3_sharpa_w01'
                or options.get('robot_pad_compliance') != BIMANUAL_COMPLIANCE
                or options.get('robot_material_binding') != 'collider_instance_v1'
                or options.get('robot_contact_materials') != {'pad_friction': .5, 'shell_friction': .5}):
            raise ValueError('Bimanual probe must preserve the pinned release material profile')
        return result
    if options.get('robot_pad_compliance') is not None:
        raise ValueError('Single-arm material probe requires the rigid pad scene')
    if options.get('robot_contact_materials') != {'pad_friction': .5, 'shell_friction': .5}:
        raise ValueError('Probe source must retain the 0.5/0.5 release friction')
    if mode == 'differentiated':
        options['robot_contact_materials'] = {'pad_friction': .8, 'shell_friction': .2}
    elif mode != 'default':
        raise ValueError(f'Unknown material probe mode: {mode}')
    options['robot_material_binding'] = 'collider_instance_v1'
    return result


def compliant_readback(sim_view, material_record, request):
    """Read supported Isaac 6.1 RigidBodyView spring properties for this probe.

    This diagnostic does not add a new getter to normal environment startup.
    RigidBodyView.get_compliant_material_properties returns (values, modes),
    both [body, shape, 2]; the value columns are stiffness and damping.
    """
    def host(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)

    def encoded(value):
        return float(value) if math.isfinite(value) else ('NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity'))

    records, passed = [], True
    for link in material_record['links']:
        view = sim_view.create_rigid_body_view(link['link_path'])
        if view.count != 1 or list(view.prim_paths) != [link['link_path']] or view.max_shapes != link['native_shape_count']:
            raise RuntimeError('Compliant readback native link/shape identity changed')
        values, modes = map(host, view.get_compliant_material_properties())
        if values.shape != (1, view.max_shapes, 2) or modes.shape != (1, view.max_shapes, 2):
            raise RuntimeError('Invalid native compliant material tensor dimensions')
        shapes = []
        for index, (value, mode) in enumerate(zip(values[0], modes[0])):
            finite = bool(np.isfinite(value).all())
            enabled = link['shapes'][index]['material_properties'][2] < 0
            shape_passed = enabled == (link['kind'] == 'robot_pad')
            if enabled:
                # The pinned release uses the schema defaults: average
                # restitution/stiffness and damping combination (native 0).
                shape_passed &= finite and list(mode) == [0, 0] and all(
                    math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7)
                    for actual, expected in zip(value, (request['stiffness'], request['damping'])))
            # Spring parameters are unused on native rigid shapes. Their
            # restitution/friction already passed the independent base check;
            # preserve the getter bytes even if its unused fields are NaN/Inf.
            passed &= shape_passed
            shapes.append({'shape_index': index, 'stiffness': encoded(value[0]), 'damping': encoded(value[1]),
                           'stiffness_combine_mode': int(mode[0]), 'damping_combine_mode': int(mode[1]),
                           'finite_spring_fields': finite,
                           'spring_enabled_by_native_restitution': enabled,
                           'spring_fields_used': enabled, 'passed': bool(shape_passed)})
        records.append({'link_path': link['link_path'], 'link_name': link['link_name'],
                        'kind': link['kind'], 'native_shape_count': view.max_shapes, 'shapes': shapes,
                        'raw_values': {'dtype': values.dtype.str, 'shape': list(values.shape), 'data_hex': values.tobytes().hex()},
                        'raw_combine_modes': {'dtype': modes.dtype.str, 'shape': list(modes.shape), 'data_hex': modes.tobytes().hex()}})
    return {'passed': bool(passed), 'requested_pad_compliance': request,
            'expected_enabled_combine_modes': {'stiffness': 0, 'damping': 0, 'meaning': 'average'},
            'source': 'RigidBodyView.get_compliant_material_properties, one exact link per view', 'links': records}


def enabled_spring_values(readback):
    """Compare reset only over meaningful enabled fields; retain all raw bytes separately."""
    return [(link['link_path'], shape['shape_index'], shape['stiffness'], shape['damping'],
             shape['stiffness_combine_mode'], shape['damping_combine_mode'])
            for link in readback['links'] for shape in link['shapes']
            if shape['spring_enabled_by_native_restitution']]


def record_incomplete_initialization(output, plan):
    """Leave a failed receipt if Kit exits before the constructor can return.

    env.__init__ closes its native app on failure. Fast shutdown can terminate
    Python inside that close, bypassing this probe's outer exception handler.
    A normal run replaces this provisional report after collecting evidence.
    """
    dump(output/'report.json', {
        'schema': 'robot_material_probe_v2', 'mode': plan['mode'],
        'passed': False, 'provisional': True, 'checks': {}, 'error': None,
        'status': 'native_initialization_completion_not_observed',
        'detail': 'Written before native construction; if still present, inspect native log and external exit receipt for the cause.',
        'scope': plan['scope'], 'task_success_claimed': False,
        'friction_calibration_claimed': False, 'compliance_requested': plan.get('compliance_requested', False),
        'model_calls': 0, 'scene_override_only': plan['mode'] != 'bimanual_release',
        'native_shutdown_verified': False, 'native_shutdown_status': 'not_observed_by_report',
        'external_acceptance_required': 'report.passed and observed_process_exit_code == 0',
        'plan_sha256': digest(output/'plan.json'),
        'artifacts_sha256': {str(p.relative_to(output)): digest(p)
                            for p in output.rglob('*') if p.is_file() and p != output/'report.json'}})


def run(args):
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bimanual = args.mode == 'bimanual_release'
    plan = {'schema': 'robot_material_probe_plan_v2', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'family': None if bimanual else args.family, 'mode': args.mode, 'control_steps': 10, 'control_dt_s': .02,
            'material_override': {'pad_friction': .8, 'shell_friction': .2} if args.mode == 'differentiated' else None,
            'binding': 'collider_instance_v1', 'compliance_requested': bimanual,
            'pad_compliance': BIMANUAL_COMPLIANCE if bimanual else None,
            'expected_pad_shapes': 10 if bimanual else 5,
            'repeat_reset_after_hold': bimanual,
            'device': args.device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            'source_root': str(root), 'entrypoint_sha256': digest(__file__),
            'scope': 'Native pad/shell binding and material readback by link and shape index; bimanual mode preserves release compliance.',
            'task_success_claimed': False, 'friction_calibration_claimed': False, 'model_calls': 0}
    dump(output/'plan.json', plan)
    env, error, checks = None, None, {}
    try:
        from scenes.workrooms import scene, validate
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        from bench.simulation.isaacsim_backend.robot_materials import runtime_material_check
        required = ('scripts.diagnostics.probe_robot_materials', 'scenes.workrooms',
                    'bench.simulation.isaacsim_backend.env', 'bench.simulation.isaacsim_backend.robot_materials')
        verified = verify_if_snapshot(root, required_modules=required, entrypoint_file=Path(__file__))
        if verified is None:
            raise ValueError('Material probe requires a complete frozen source snapshot')
        dump(output/'runtime_imports.json', verified)
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if args.device != 'cuda:0' or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1' or not uuid.startswith('GPU-') or ',' in uuid:
            raise ValueError('Use one UUID-selected CUDA device with --device cuda:0')
        query = subprocess.run(['nvidia-smi', '-i', uuid, '--query-gpu=uuid,pci.bus_id,name', '--format=csv,noheader'],
                               check=True, capture_output=True, text=True, timeout=20)
        if len(query.stdout.strip().splitlines()) != 1 or not query.stdout.startswith(uuid+','):
            raise RuntimeError('Selected GPU UUID did not resolve uniquely')
        dump(output/'gpu_identity.json', {'query': query.args, 'result': query.stdout.strip()})
        if bimanual:
            source, preflight = bimanual_source()
            dump(output/'source_asset_preflight.json', preflight)
        else:
            source = scene(args.family, diagnostic=True)
        spec = probe_scene(source, args.mode)
        dump(output/'source_scene.json', source)
        dump(output/'scene.json', spec)
        print(json.dumps({'event': 'material_probe_start', 'mode': args.mode}), flush=True)
        record_incomplete_initialization(output, plan)
        env = FR3DexterousIsaacEnv(spec, device=args.device, render_width=320, render_height=240)
        observation = env.reset(seed=0)
        if not bimanual:
            dump(output/'source_asset_preflight.json', validate(args.family))
        initial = runtime_material_check(env.stage, env._sim_view, env._robot_view, env.scene_info)
        dump(output/'initial_native_materials.json', initial)
        expected_compliant = initial['pad_shapes'] if bimanual else 0
        if (not initial['passed'] or initial['pad_shapes'] != plan['expected_pad_shapes']
                or not initial['shell_shapes'] or initial['compliant_shapes'] != expected_compliant):
            raise RuntimeError('Material probe pad/shell inventory or compliance differs from its plan')
        if bimanual:
            spring = compliant_readback(env._sim_view, initial, source['isaacsim']['robot_pad_compliance'])
            dump(output/'initial_native_compliance.json', spring)
            checks['original_pad_stiffness_damping'] = spring['passed']
        home = env.action_spec()['initial_joint_positions_rad']
        native_steps, native_duration = 0, 0.
        with (output/'physics.jsonl').open('x') as stream:
            def write(index, obs, command):
                state = {name: value.tolist() for name, value in env.capture_state().items()}
                stream.write(json.dumps({'control_index': index, 'command': command,
                                         'observation': obs, 'state': state}, allow_nan=False)+'\n')
                stream.flush()
                return state
            previous = write(0, observation, None)
            for index in range(1, plan['control_steps']+1):
                command = {'joint_positions_rad': home, 'duration_s': plan['control_dt_s']}
                observation = env.step(command)
                current = write(index, observation, command)
                step_delta = int(current['native_physics_step_index'])-int(previous['native_physics_step_index'])
                time_delta = float(current['native_world_time_s'])-float(previous['native_world_time_s'])
                if step_delta != round(.02/env.dt) or abs(time_delta-.02) > 1e-6:
                    raise RuntimeError('Native material probe did not advance the requested physics interval')
                native_steps += step_delta
                native_duration += time_delta
                previous = current
        final = runtime_material_check(env.stage, env._sim_view, env._robot_view, env.scene_info)
        dump(output/'final_native_materials.json', final)
        checks.update({'initial_materials': initial['passed'], 'final_materials': final['passed'],
                  'material_readback_unchanged': initial == final,
                  'pads_and_shells_both_observed': initial['pad_shapes'] > 0 and initial['shell_shapes'] > 0,
                  'requested_pad_model': initial['compliant_shapes'] == final['compliant_shapes'] == expected_compliant})
        if bimanual:
            reset = env.reset(seed=0)
            reset_materials = runtime_material_check(env.stage, env._sim_view, env._robot_view, env.scene_info)
            reset_spring = compliant_readback(env._sim_view, reset_materials, source['isaacsim']['robot_pad_compliance'])
            dump(output/'reset_state.json', {'observation': reset,
                'state': {name: value.tolist() for name, value in env.capture_state().items()}})
            dump(output/'reset_native_materials.json', reset_materials)
            dump(output/'reset_native_compliance.json', reset_spring)
            checks.update(reset_materials_unchanged=reset_materials == initial,
                          reset_stiffness_damping_unchanged=reset_spring['passed'] and
                          enabled_spring_values(reset_spring) == enabled_spring_values(spring))
        dump(output/'metadata.json', env.metadata())
        dump(output/'native_steps.json', {'control_steps': plan['control_steps'], 'physics_substeps': native_steps,
                                         'duration_s': native_duration, 'initialization_reset_ticks_excluded': True})
        dump(output/'runtime_imports_final.json', verify_if_snapshot(root, required_modules=required, entrypoint_file=Path(__file__)))
        print(json.dumps({'event': 'material_probe_complete', 'pad_shapes': initial['pad_shapes'],
                          'shell_shapes': initial['shell_shapes'], 'physics_substeps': native_steps}), flush=True)
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        print(json.dumps({'event': 'material_probe_error', 'error': error['detail']}), flush=True)
    report = {'schema': 'robot_material_probe_v2', 'mode': args.mode,
              'passed': bool(error is None and checks and all(checks.values())), 'provisional': False,
              'checks': checks, 'error': error,
              'scope': plan['scope'], 'task_success_claimed': False, 'friction_calibration_claimed': False,
              'compliance_requested': bimanual, 'model_calls': 0, 'scene_override_only': not bimanual,
              'release_modified': False,
              'plan_sha256': digest(output/'plan.json'),
              'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                                  if p.is_file() and p != output/'report.json'}}
    return seal_report_before_shutdown(output, report, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--family', default='mechanical_assembly')
    parser.add_argument('--mode', choices=('differentiated', 'default', 'bimanual_release'), default='differentiated')
    if not run(parser.parse_args())['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
