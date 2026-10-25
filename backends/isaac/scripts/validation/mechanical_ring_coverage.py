"""Read-only evidence for the original four dense-ring coverage probes."""
from copy import deepcopy
import numpy as np

from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scenes.workrooms.mechanical_assembly.rules import COMMON
from scripts.validation.mechanical_iteration_readback import stage_iteration_settings
from scripts.validation.validate_mechanical_ring_sdf import stack_impulse_comparison


def setup(env, inputs, report):
    expected = inputs['numerical_accuracy_candidate']['expected_body_parameters']
    actual = {}
    originals = {}
    for name, fields in expected.items():
        prim = env.stage.GetPrimAtPath(env.scene_info['objects'][name]['path'])
        actual[name] = {key: prim.GetAttribute('physxRigidBody:'+key).Get() for key in fields}
        source = load_task_asset(**inputs['sdf_original_assets'][name], verify_hashes=True)
        native = report['checks']['native_parameters']['objects'][name]
        originals[name] = {
            'mass': bool(np.isclose(native['mass_kg'], source['mass_kg'], atol=1e-7, rtol=1e-5)),
            'com': bool(np.allclose(native['native_com_principal_pose_xyzw'][:3], source['center_of_mass_m'], atol=1e-7, rtol=0)),
            'inertia': bool(np.allclose(native['native_inertia_body_frame_kg_m2'], np.diag(source['inertia_diag_kg_m2']), atol=1e-9, rtol=2e-5))}
    report['initial_stage_iteration_settings'] = stage_iteration_settings(env)
    scenes = list(report['initial_stage_iteration_settings']['physics_scenes'].values())
    report['checks']['candidate_body_parameters'] = {'passed': actual == expected, 'expected': expected, 'recorded': actual}
    report['checks']['original_native_mass_properties'] = {'passed': all(all(v.values()) for v in originals.values()), 'objects': originals}
    report['checks']['scene_iteration_clamps'] = {'passed': bool(scenes) and all(
        s['minVelocityIterationCount'] <= b['solverVelocityIterationCount'] <= s['maxVelocityIterationCount']
        and s['minPositionIterationCount'] <= b['solverPositionIterationCount'] <= s['maxPositionIterationCount']
        for s in scenes for b in actual.values())}


def reset_receipt(env, case):
    before = env.capture_state()
    objects = {}
    for index, name in enumerate(env.object_names):
        p = np.asarray(before['object_q'][index], dtype=float)
        v = np.asarray(before['object_qd'][index], dtype=float)
        expected = case['reset_objects'][name]
        q = np.asarray(expected['quaternion_wxyz'], dtype=float)[[1, 2, 3, 0]]
        actual_q = p[3:]/np.linalg.norm(p[3:])
        q = q/np.linalg.norm(q)
        if np.dot(q, actual_q) < 0:
            actual_q = -actual_q
        angle = float(4*np.arctan2(np.linalg.norm(q-actual_q), np.linalg.norm(q+actual_q)))
        error = float(np.linalg.norm(p[:3]-expected['position_m']))
        objects[name] = {'requested': deepcopy(expected), 'native_pose_xyzw': p.tolist(),
            'native_velocity': v.tolist(), 'position_error_m': error, 'orientation_error_rad': angle,
            'passed': error <= COMMON['initial_position_tolerance_m']
                and angle <= COMMON['initial_orientation_tolerance_rad'] and bool(np.all(v == 0))}
    after = env.capture_state()
    unchanged = all(np.array_equal(before[k], after[k]) for k in before)
    return {'passed': unchanged and all(v['passed'] for v in objects.values()), 'objects': objects,
        'read_only_state_unchanged': unchanged, 'expected_zero_initial_velocity': True,
        'scope': 'Actual native state immediately after original reset, before observed integration.'}


def support_topology(case):
    if case['id'] == 'nominal_supply':
        return None
    order = ['top_ring', 'wide_ring', 'spacer'] if case['id'] == 'negative' else ['spacer', 'wide_ring', 'top_ring']
    return {name: {'support_path_prefix': ('/World/Fixtures/shaft_fixture/' if i == 0
            else '/World/Objects/'+order[i-1])} for i, name in enumerate(order)}


def result_evidence(env, rows, case, masses):
    topology = support_topology(case)
    result = {'scope': 'Supplementary physical support and entry evidence; canonical task assertions are unchanged.'}
    if topology:
        result['actual_support_chain'] = stack_impulse_comparison(rows, env.scene_info, masses, env.dt, topology)
    if case['id'] != 'nominal_supply':
        tip = .888
        entries = {}
        for name in env.object_names:
            crossing = next((r for r in rows if r['poses_xyzw'][name][2] < tip), None)
            last = rows[-1]['poses_xyzw'][name]
            entries[name] = {'reset_bottom_above_tip': case['reset_objects'][name]['position_m'][2] > tip,
                'first_bottom_below_tip_time_s': crossing['time_s'] if crossing else None,
                'radial_offset_at_crossing_m': float(np.linalg.norm(np.asarray(crossing['poses_xyzw'][name][:2])-[.4, 0])) if crossing else None,
                'final_bottom_below_tip': last[2] < tip,
                'final_radial_axis_offset_m': float(np.linalg.norm(np.asarray(last[:2])-[.4, 0]))}
        result['shaft_tip_entry'] = {'objects': entries, 'shaft_tip_world_z_m': tip,
            'scope': 'Recorded ring bottom-origin crossing of the unchanged shaft tip, with radial offset. Finite source geometry, orientation, contacts and descent are separate gates; this alone is not full bore containment.'}
    return result
