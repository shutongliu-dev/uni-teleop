"""Build the zero-model privileged L3 validation contract.

This entrypoint is intentionally CPU-only.  It freezes the task reset,
task-owned tabletop layout, endpoint targets, and recovery phases that a native
privileged runner must execute.  It never calls a model, writes object poses,
or reports manipulation success; native contact evidence is a separate gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from bench.paths import load_json
from scripts.validation import mechanical_robot_phase_timing as phase_timing

from scenes.workrooms.mechanical_assembly.tasks.shaft_spacer_assembly import definition, layout, task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'mechanical_l3_privileged_validation_plan.v1'
NATIVE_INPUT_SCHEMA = 'mechanical_l3_native_probe_input.v1'


def _native_case_contract(robot_phases, scope):
    return {
        'release_contract': {
            'free_dynamic_objects': True,
            'gravity_only_after_reset': True,
            'object_pose_writeback_forbidden': True,
            'external_object_forces_forbidden': True,
        },
        'phase_receipt_contract': {
            'record_raw_contacts': True,
            'task_success_credit': 0,
            'required_phase_ids': [phase['id'] for phase in robot_phases],
            'contact_actor_paths': {
                **{name: '/World/Objects/'+name for name in task.PART_ORDER},
                'shaft_fixture': '/World/Fixtures/shaft_fixture',
                'robot': '/World/Robot',
            },
            'scope': 'Native substep contact receipts only; no manipulation or task score.',
        },
        'scope': scope,
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_plan(*, asset_root=None, stand_root=None):
    """Return a deterministic privileged plan without starting Isaac Sim."""
    kwargs = {}
    if asset_root is not None:
        kwargs['asset_root'] = asset_root
    if stand_root is not None:
        kwargs['stand_root'] = stand_root
    values = task.build_task_inputs(**kwargs)
    scene, frozen_task, initial = values['scene'], values['task'], values['initial_state']
    layout_report = layout.validate_layout(scene, frozen_task)
    contract_report = layout.validate_privileged_contract()
    contract = frozen_task['privileged_validation']
    task_layout = frozen_task['task_layout']
    return {
        'schema': SCHEMA,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'family': 'mechanical_assembly',
        'task_id': 'shaft_spacer_assembly',
        'layout_id': task_layout['layout_id'],
        'model_calls': 0,
        'ordinary_agent_validation': False,
        'native_runtime_started': False,
        'task_success_claimed': False,
        'object_state_writeback': False,
        'external_object_forces': False,
        'fixed_parent_scene': scene['workroom'],
        'scene_inventory': {
            'objects': [row['id'] for row in scene['objects']],
            'fixtures': [row['id'] for row in scene['fixtures']],
            'robot_layout_id': scene['mechanical_task_provenance']['shared_layout_id'],
        },
        'initial_state': initial,
        'layout_contract': task_layout,
        'layout_check': layout_report,
        'reachability_targets': frozen_task['cpu_contract']['reachability_targets'],
        'privileged_contract': contract,
        'privileged_contract_check': contract_report,
        'reset_contract': {
            'kind': frozen_task['reset_conditions']['kind'],
            'supply_positions_m': frozen_task['reset_conditions']['supply_positions_m'],
            'part_order': frozen_task['evaluation']['part_order'],
            'required_support_pairs': frozen_task['evaluation']['required_support_pairs'],
            'initial_robot_joint_count': len(initial['joint_positions_rad']),
            'object_reset_pose_source': 'scene.objects and task.evaluation.initial_objects',
        },
        'required_native_observations': contract['required_native_observations'],
        'forbidden_shortcuts': contract['forbidden_shortcuts'],
        'source_sha256': {
            str(path): digest(path) for path in (
                Path(__file__), Path(phase_timing.__file__), Path(task.__file__), Path(layout.__file__))
        },
        'scope': 'CPU contract for a zero-model privileged native run; no native physics or task manipulation acceptance.',
    }


def _robot_target(scene, initial, position_m):
    """Solve a robot-only wrist endpoint for a native diagnostic phase."""
    from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model, solve_wrist
    model = geometry_model(scene['robot'])
    solved = solve_wrist(model, position_m)
    if not solved.get('converged'):
        raise ValueError('L3 native endpoint IK did not converge: '+repr(position_m))
    target = dict(initial['joint_positions_rad'])
    target.update(solved['joint_positions_rad'])
    model.checked_joints(target, enforce_limits=True)
    return target, {'position_m': list(position_m),
                    'position_error_m': float(solved['position_error_m']),
                    'rotation_error_rad': float(solved['rotation_error_rad'])}


def _robot_phases(scene, initial):
    """Build bounded robot-only target phases for the native observer.

    These phases move the shared robot through the task's approach stations.
    They intentionally do not attach, move, or pose-write any ring; native
    reports must show whether contacts and releases occur on their own.
    """
    positions = [
        ('ready', (.10, 0., 1.40)),
        ('approach_spacer', (.25, -.27, 1.00)),
        ('shaft_approach_spacer', (.40, 0., 1.00)),
        ('approach_wide_ring', (.25, 0., 1.00)),
        ('shaft_approach_wide_ring', (.40, 0., 1.00)),
        ('approach_top_ring', (.25, .27, 1.00)),
        ('shaft_approach_top_ring', (.40, 0., 1.00)),
        ('withdraw', (.10, 0., 1.40)),
    ]
    phases, previous = [], None
    for name, position in positions:
        target, ik = _robot_target(scene, initial, position)
        if previous is not None and target == previous:
            continue
        phases.append({'id': name, 'duration_s': .24,
                       'joint_positions_rad': target,
                       'endpoint_ik': ik,
                       'scope': 'robot endpoint movement only; no object attachment or pose writeback'})
        previous = target
    return phase_timing.time_phases(phases, scene, initial)


def _near_seat_case(robot_phases):
    initial = {
        'spacer': {'position_m': [.40, 0., .791], 'quaternion_wxyz': [1., 0., 0., 0.]},
        'wide_ring': {'position_m': [.40, 0., .824], 'quaternion_wxyz': [1., 0., 0., 0.]},
        'top_ring': {'position_m': [.40, 0., .847], 'quaternion_wxyz': [1., 0., 0., 0.]},
    }
    target = {
        'spacer': [.40, 0., .788], 'wide_ring': [.40, 0., .818], 'top_ring': [.40, 0., .838]}
    heights = {'spacer': .030, 'wide_ring': .020, 'top_ring': .025}
    order = ('spacer', 'wide_ring', 'top_ring')
    assertions = {}
    for index, name in enumerate(order):
        support = '/World/Fixtures/shaft_fixture' if index == 0 else '/World/Objects/'+order[index-1]
        assertions[name] = {
            'position_m': target[name], 'position_tolerance_m': [.00425, .00425, .0005],
            'support_path_prefix': support, 'maximum_tilt_rad': math.radians(5.),
            'minimum_descent_m': .0025 + .003*index,
            'require_guidance_contact': False,
            'guidance_path_prefix': '/World/Fixtures/shaft_fixture',
        }
    value = {
        'id': 'near_seat_with_robot_targets', 'duration_s': sum(p['duration_s'] for p in robot_phases),
        'reset_objects': initial, 'assertions': assertions,
        'expected_functional_success': True, 'robot_phases': robot_phases,
        'support_chain': [['spacer', 'fixture'], ['wide_ring', 'spacer'], ['top_ring', 'wide_ring']],
        'scope': 'Native gravity-guided stack support plus robot endpoint motion; not grasp or task-score evidence.',
    }
    value.update(_native_case_contract(robot_phases, value['scope']))
    return value


def _nominal_supply_case(robot_phases, scene):
    reset = {row['id']: {'position_m': list(row['position_m']),
                         'quaternion_wxyz': list(row['quaternion_wxyz'])}
             for row in scene['objects']}
    assertions = {}
    for row in scene['objects']:
        name = row['id']
        assertions[name] = {
            'position_m': list(row['position_m']), 'position_tolerance_m': [.002, .002, .0005],
            'support_path_prefix': '/World/Fixtures/'+name+'_supply_rest',
            'maximum_tilt_rad': math.radians(5.), 'minimum_descent_m': 0.,
        }
    value = {
        'id': 'nominal_supply_with_robot_targets', 'duration_s': sum(p['duration_s'] for p in robot_phases),
        'reset_objects': reset, 'assertions': assertions,
        'expected_functional_success': True, 'robot_phases': robot_phases,
        'scope': 'Native supply reset/support and robot endpoint motion; not grasp or task-score evidence.',
    }
    value.update(_native_case_contract(robot_phases, value['scope']))
    return value


def _wrong_order_case(robot_phases):
    value = _near_seat_case(robot_phases)
    value.update({
        'id': 'wrong_order_recovery_probe',
        'reset_variant': 'wrong_order_near_seat',
        'reset_objects': {
            'top_ring': {'position_m': [.40, 0., .791], 'quaternion_wxyz': [1., 0., 0., 0.]},
            'wide_ring': {'position_m': [.40, 0., .819], 'quaternion_wxyz': [1., 0., 0., 0.]},
            'spacer': {'position_m': [.40, 0., .842], 'quaternion_wxyz': [1., 0., 0., 0.]},
        },
        'engineering_reset_order': ['top_ring', 'wide_ring', 'spacer'],
        'expected_functional_success': False,
        'scope': 'Wrong-order reset and rejection diagnostic; removal/reassembly and recovery success are not exercised.',
    })
    return value


def _guided_stack_case(robot_phases):
    """Start all free rings above the tapered tip for a real gravity release."""
    reset = {
        'spacer': {'position_m': [.407, 0., .891], 'quaternion_wxyz': [1., 0., 0., 0.]},
        'wide_ring': {'position_m': [.407, 0., .931], 'quaternion_wxyz': [1., 0., 0., 0.]},
        'top_ring': {'position_m': [.407, 0., .961], 'quaternion_wxyz': [1., 0., 0., 0.]},
    }
    target = {'spacer': [.40, 0., .788], 'wide_ring': [.40, 0., .818], 'top_ring': [.40, 0., .838]}
    order = ('spacer', 'wide_ring', 'top_ring')
    assertions = {}
    for index, name in enumerate(order):
        support = '/World/Fixtures/shaft_fixture' if index == 0 else '/World/Objects/'+order[index-1]
        assertions[name] = {
            'position_m': target[name], 'position_tolerance_m': [.00425, .00425, .0005],
            'support_path_prefix': support, 'maximum_tilt_rad': math.radians(5.),
            'minimum_descent_m': .070, 'require_guidance_contact': True,
            'guidance_path_prefix': '/World/Fixtures/shaft_fixture',
        }
    value = {
        'id': 'guided_stack_drop_with_robot_targets',
        'duration_s': sum(p['duration_s'] for p in robot_phases),
        'reset_objects': reset, 'assertions': assertions,
        'expected_functional_success': True, 'robot_phases': robot_phases,
        'support_chain': [['spacer', 'fixture'], ['wide_ring', 'spacer'], ['top_ring', 'wide_ring']],
        'scope': 'Native gravity release and three-layer support diagnostic with robot endpoint motion; not task-score evidence.',
    }
    value.update(_native_case_contract(robot_phases, value['scope']))
    value['release_contract'].update({'lateral_offset_m': .007, 'shaft_tip_z_m': .888})
    return value


def build_native_input(release=None):
    """Freeze an input consumed by the native contact runner.

    This input adds no task score and never changes the release.  It carries
    the exact scene/release hash, object reset cases, and robot-only endpoint
    phases; the native runner is responsible for recording contacts, support,
    stability and any failure.
    """
    from bench.isaacsim.release import task_inputs
    if release is None:
        release = Path(definition.TASK['source_release']['path'])
    release = Path(release).resolve()
    manifest = release/'release_manifest.json'
    if not manifest.is_file():
        raise ValueError('L3 native input requires a frozen release manifest')
    release_sha256 = digest(manifest)
    expected = definition.TASK['source_release']['manifest_sha256']
    if release_sha256 != expected:
        raise ValueError(f'L3 release manifest drift: {release_sha256} != {expected}')
    _, scene, frozen_task, initial = task_inputs(release, 'shaft_spacer_assembly')
    if frozen_task.get('task_id') != 'shaft_spacer_assembly' or scene.get('isaacsim', {}).get('solver_type') != 'PGS':
        raise ValueError('L3 native input requires the pinned PGS shaft-spacer candidate')
    phases = _robot_phases(scene, initial)
    native_sdf_source_meshes = {}
    for name in task.PART_ORDER:
        spec = next(row for row in scene['objects'] if row['id'] == name)
        catalog = load_json(Path(spec['asset_root']) / 'manifest.json')
        asset = catalog['assets'][spec['asset_id']]
        source_meshes = [path for path in asset.get('source_dependencies_sha256', {})
                         if path.endswith('/visual.obj')]
        if len(source_meshes) != 1:
            raise ValueError(f'Expected one preserved source mesh for {name}')
        native_sdf_source_meshes[name] = source_meshes[0]
    cases = [_nominal_supply_case(phases, scene), _guided_stack_case(phases),
             _near_seat_case(phases), _wrong_order_case(phases)]
    source_files = [Path(__file__), Path(phase_timing.__file__), Path(task.__file__), Path(layout.__file__)]
    return {
        'schema': NATIVE_INPUT_SCHEMA, 'family': 'mechanical_assembly',
        'task_id': 'shaft_spacer_assembly', 'release': str(release),
        'release_manifest_sha256': release_sha256, 'scene': scene,
        'task': frozen_task, 'initial_state': initial, 'cases': cases,
        'native_sdf_source_meshes': native_sdf_source_meshes,
        'native_sdf_subgrid_resolution': 0,
        'robot_phase_contract': {
            'phase_count': len(phases), 'duration_s': sum(p['duration_s'] for p in phases),
            'phases': phases, 'object_attachment': False, 'object_state_writeback': False,
            'external_object_forces': False,
            'scope': 'Privileged robot joint targets only; endpoint IK is not grasp qualification.',
        },
        'model_calls': 0, 'ordinary_agent_validation': False,
        'task_success_claimed': False, 'physics_accepted': False,
        'object_state_writeback_after_reset': False, 'external_object_forces': False,
        'source_sha256': {str(path): digest(path) for path in source_files},
        'scope': 'Frozen native reset/support/recovery probe input. Native report may pass only its declared engineering gates; it cannot promote the task release.',
    }


def run_native_input(output, release=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    value = build_native_input(release)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return value


def run(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    plan = build_plan()
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-input', type=Path)
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    plan = run(args.output)
    if args.native_input is not None:
        native = run_native_input(args.native_input, args.release)
    else:
        native = None
    print(json.dumps({
        'passed': plan['layout_check']['passed'] and plan['privileged_contract_check']['passed'],
        'output': str(args.output.resolve()),
        'model_calls': plan['model_calls'],
        'native_runtime_started': plan['native_runtime_started'],
        'phase_count': len(plan['privileged_contract']['phases']),
        'native_input': None if native is None else str(args.native_input.resolve()),
        'native_case_count': None if native is None else len(native['cases']),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()

