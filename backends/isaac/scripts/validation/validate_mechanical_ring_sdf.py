"""One sealed SDF1024/64-convex ring comparison, geometry gate before gravity.

This privileged reset-only probe does not alter a task or controller. A finite
20 um SDF surface gate must pass before either body receives observed gravity.
Raw normal, net tensor and pair-matrix impulses are retained without rescaling.
"""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import re
import traceback
import numpy as np

from bench.isaacsim.release import dump, digest
from bench.paths import load_json
from scripts.assets.prepare_mechanical_ring_sdf import ASSET_ID, DEFAULT_SOURCE, SOURCE_ID, SOURCE_IDS
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_assembly_contacts import (
    Observer, assess_case, native_parameters, contact_exchange_check,
    stage_inventory, contact_filter_coverage, check_convex_geometry, capture,
    plain, seal_report_before_shutdown)
from scripts.validation.mechanical_ring_sdf import check_ring_sdf

MODULE = 'scripts.validation.validate_mechanical_ring_sdf'


def solver_selection(metadata, expected):
    recorded = metadata.get('physics_solver', {})
    return {'passed': expected in ('PGS', 'TGS') and recorded.get('profile_selection') == expected
            and recorded.get('usd_readback') == expected,
            'expected': expected, 'recorded': deepcopy(recorded),
            'scope': 'Canonical selector and composed PhysicsScene USD readback; not a native tensor solver-enum query.'}


def candidate_sdf_subgrid_resolution(candidate_root):
    """Read the authored SDF storage mode from a frozen candidate catalog.

    Dense and sparse SDF candidates intentionally use different subgrid
    settings.  The native gate must compare the composed value with the
    candidate that was frozen into the input rather than silently assuming the
    sparse setting.
    """
    candidate_root = Path(candidate_root).resolve()
    manifest_path = candidate_root / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError('SDF candidate manifest is required: ' + str(manifest_path))
    manifest = load_json(manifest_path)
    assets = manifest.get('assets')
    if not isinstance(assets, dict) or not assets:
        raise ValueError('SDF candidate manifest has no assets')
    values = []
    for asset_id, asset in sorted(assets.items()):
        usd_path = Path(asset.get('usd_path', ''))
        if not usd_path.is_file():
            usd_path = candidate_root / usd_path
        if not usd_path.is_file():
            raise ValueError(f'SDF candidate USD is missing for {asset_id}: {usd_path}')
        matches = re.findall(
            r'int\s+physxSDFMeshCollision:sdfSubgridResolution\s*=\s*(-?\d+)',
            usd_path.read_text())
        if len(matches) != 1:
            raise ValueError(f'Expected one authored SDF subgrid value for {asset_id}')
        values.append(int(matches[0]))
    if len(set(values)) != 1 or values[0] not in (0, 6):
        raise ValueError('SDF candidate uses inconsistent or unsupported subgrid resolution')
    return values[0]


def build_inputs(candidate_root):
    from scenes.workrooms.mechanical_assembly.tasks.shaft_spacer_assembly.task import build_task_inputs
    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.release import _asset_inventory
    candidate_root = Path(candidate_root).resolve()
    native_sdf_subgrid_resolution = candidate_sdf_subgrid_resolution(candidate_root)
    value = build_task_inputs(solver_type='PGS')
    scene = value['scene']
    original = deepcopy(scene['objects'][0])
    rest = deepcopy(next(x for x in scene['fixtures'] if x['id']=='spacer_supply_rest'))
    scene['objects'] = []
    scene['fixtures'] = [x for x in scene['fixtures'] if x['id']=='robot_stand']
    assertions, reset = {}, {}
    for name,y in [('convex_ring',-.27), ('sdf_ring',.27)]:
        body, support = deepcopy(original), deepcopy(rest)
        body.update(id=name,position_m=[.25,y,.789])
        support.update(id=name+'_rest',position_m=[.25,y,.78])
        if name=='sdf_ring':
            body.pop('collision_mesh_coordinate_scale')
            body.pop('friction')
            body.update(asset_id=ASSET_ID,asset_root=str(candidate_root),
                usd_physics_policy='preserve_authored_v1',
                usd_physics_material_fallback={'static_friction':.4,'dynamic_friction':.4,
                    'restitution':0.,'friction_combine':'average'})
        scene['objects'].append(body)
        scene['fixtures'].append(support)
        reset[name] = {k:deepcopy(body[k]) for k in ('position_m','quaternion_wxyz')}
        assertions[name] = {'position_m':[.25,y,.786], 'position_tolerance_m':[.002,.002,.0003],
            'support_path_prefix':'/World/Fixtures/'+support['id']+'/',
            'maximum_tilt_rad':.03,'minimum_descent_m':.0025}
    scene['description'] = 'Independent single SDF1024 spacer and original 64-cell spacer on identical single-shape rests.'
    scene['engineering_contact_force_crosscheck'] = True
    scene['scenario']['scope'] = 'Reset-only ring representation comparison; not task execution.'
    value['task'] = {'task_id':'mechanical_ring_sdf_probe','task_success_claimed':False,
        'scope':'One SDF1024 surface candidate. Geometry before gravity; original task/catalog unchanged.'}
    value['initial_state']['objects'] = reset
    value['cases'] = [{'id':'support','duration_s':1.,'reset_objects':reset,'assertions':assertions}]
    value['sdf_source_mesh'] = str(DEFAULT_SOURCE/SOURCE_ID/'visual.obj')
    value['native_sdf_subgrid_resolution'] = native_sdf_subgrid_resolution
    value['asset_files_sha256'] = {p:r['sha256'] for p,r in _asset_inventory(scene).items()}
    value['asset_files_sha256'][value['sdf_source_mesh']] = digest(value['sdf_source_mesh'])
    return value


def build_stack_inputs(candidate_root, source_input):
    """Derive an independent candidate from the frozen ordinary PGS near-seat case."""
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.release import _asset_inventory
    source_input, candidate_root = Path(source_input).resolve(), Path(candidate_root).resolve()
    native_sdf_subgrid_resolution = candidate_sdf_subgrid_resolution(candidate_root)
    value = load_json(source_input)
    if (value['scene']['isaacsim']['solver_type'] != 'PGS' or len(value['cases']) != 1
            or value['cases'][0]['id'] != 'near_seat' or value['cases'][0]['duration_s'] != 2.):
        raise ValueError('The original ordinary-PGS two-second near-seat case is required')
    value['sdf_source_meshes'], value['sdf_original_assets'] = {}, {}
    for spec, source_id in zip(value['scene']['objects'], SOURCE_IDS, strict=True):
        if spec['asset_id'] != source_id:
            raise ValueError('Original three-ring order required')
        original = load_task_asset(source_id,spec['asset_root'],verify_hashes=True)
        candidate = load_task_asset(source_id+'_sdf1024',candidate_root,verify_hashes=True)
        for key in ('mass_kg','center_of_mass_m','inertia_diag_kg_m2','material_prior',
                'contact_offset_m','rest_offset_m','geometry_contract','geometry_features'):
            if candidate[key] != original[key]:
                raise ValueError('Candidate changed original physical parameter: '+key)
        name = spec['id']
        value['sdf_source_meshes'][name] = original['mesh_path']
        value['sdf_original_assets'][name] = {'asset_id':source_id,'asset_root':spec['asset_root']}
        spec.pop('collision_mesh_coordinate_scale')
        spec.pop('friction')
        spec.update(asset_id=source_id+'_sdf1024',asset_root=str(candidate_root),
            usd_physics_policy='preserve_authored_v1',
            usd_physics_material_fallback={k:original['material_prior'][k] for k in (
                'static_friction','dynamic_friction','restitution','friction_combine')})
    value['initial_state']['objects'] = deepcopy(value['cases'][0]['reset_objects'])
    value['scene']['engineering_contact_force_crosscheck'] = True
    value['sdf_stack_candidate'] = {'source_input':str(source_input),'source_input_sha256':digest(source_input),
        'original_cases_unchanged':True,'original_fixtures_unchanged':True,
        'original_task_rules_unchanged':True,'source_geometry_modified':False,
        'scope':'Single SDF1024 per complete original ring. Near-seat support only; no catalog promotion or task score.'}
    value['native_sdf_subgrid_resolution'] = native_sdf_subgrid_resolution
    value['asset_files_sha256'] = {p:r['sha256'] for p,r in _asset_inventory(value['scene']).items()}
    value['asset_files_sha256'].update({p:digest(p) for p in value['sdf_source_meshes'].values()})
    # Bind the frozen parent input and all original asset dependencies too.
    value['asset_files_sha256'].update(load_json(source_input)['asset_files_sha256'])
    value['asset_files_sha256'][str(source_input)] = digest(source_input)
    return value


class ReadOnlySelection:
    """Pass selected audit expectations without writing to the live environment."""
    def __init__(self, env):
        self.env = env
        self.scene_info = deepcopy(env.scene_info)
        self.scene_spec = deepcopy(env.scene_spec)

    def __getattr__(self, name):
        return getattr(self.env,name)


def impulse_comparison(rows, scene_info, masses, dt):
    """Keep three independent reporting APIs and the unscaled weight reference."""
    tail = rows[-round(.5/dt):]
    result = {}
    for name,mass in masses.items():
        path = scene_info['objects'][name]['path']
        raw, net, matrix = np.zeros(3), np.zeros(3), np.zeros(3)
        for row in tail:
            for contact in row['raw_contacts']:
                if path not in (contact['actor0'],contact['actor1']):
                    continue
                vector = (np.asarray(contact['normal_impulses_ns'])[:,None]*np.asarray(contact['normals'])).sum(0)
                raw += vector*(1 if path==contact['actor0'] else -1)
            native = row['native_contact_impulse_crosscheck'][name]
            net += np.asarray(native['net_tensor_ns'])
            matrix += np.asarray(native['pair_matrix_sum_ns'])
        reference = mass*9.81*len(tail)*dt
        sums = {'raw_signed_normal':raw,'native_net_tensor':net,'native_pair_matrix':matrix}
        ratios = {k:float(v[2]/reference) for k,v in sums.items()}
        result[name] = {'mass_kg':mass, 'sample_count':len(tail),'gravity_impulse_ns':reference,
            'impulse_vectors_ns':{k:v.tolist() for k,v in sums.items()},
            'vertical_weight_ratios':ratios,
            'all_three_weight_ratios_within_5_percent':all(abs(x-1)<=.05 for x in ratios.values()),
            'weight_ratio_tolerance':.05}
    return {'objects':result,'quantitative_weight_balance_passed':all(
        x['all_three_weight_ratios_within_5_percent'] for x in result.values()),
        'scope':'Last 0.5 s reporting crosscheck on independent horizontal supports. No collider-count normalization; this cannot identify the cause of a reporting discrepancy.'}


def stack_impulse_comparison(rows, scene_info, masses, dt, assertions):
    """Test each body's net weight, total external weight and all three load paths."""
    result = impulse_comparison(rows,scene_info,masses,dt)
    tail = rows[-round(.5/dt):]
    total_reference = sum(masses.values())*9.81*len(tail)*dt
    vectors = {key:np.sum([body['impulse_vectors_ns'][key] for body in result['objects'].values()],axis=0)
        for key in ('raw_signed_normal','native_net_tensor','native_pair_matrix')}
    ratios = {key:float(value[2]/total_reference) for key,value in vectors.items()}
    layers, names = {}, list(assertions)
    for i,name in enumerate(names):
        path, support = scene_info['objects'][name]['path'], assertions[name]['support_path_prefix']
        impulse = 0.
        for row in tail:
            for contact in row['raw_contacts']:
                if path not in (contact['actor0'],contact['actor1']):
                    continue
                other = contact['actor1'] if contact['actor0'] == path else contact['actor0']
                if other.startswith(support):
                    impulse += sum(float(s)*float(n[2])*(1 if contact['actor0']==path else -1)
                        for s,n in zip(contact['normal_impulses_ns'],contact['normals']))
        reference = sum(masses[n] for n in names[i:])*9.81*len(tail)*dt
        layers[name] = {'support_path_prefix':support, 'supported_parts':names[i:],
            'raw_signed_vertical_impulse_ns':impulse, 'gravity_impulse_ns':reference,
            'weight_ratio':impulse/reference, 'passed':abs(impulse/reference-1)<=.05}
    result.update(total={'gravity_impulse_ns':total_reference,
        'impulse_vectors_ns':{k:v.tolist() for k,v in vectors.items()},'vertical_weight_ratios':ratios},
        layers=layers, weight_ratio_tolerance=.05,
        scope='Last 0.5 s; internal ring contact cancels in whole-stack sums. Signed native impulses are never rescaled.')
    result['quantitative_weight_balance_passed'] &= all(abs(v-1)<=.05 for v in ratios.values())
    result['quantitative_weight_balance_passed'] &= all(v['passed'] for v in layers.values())
    return result


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    env = observer = None
    report = {'schema':'mechanical_ring_sdf_comparison.v1','passed':False,
        'task_success_claimed':False,'model_calls':0,'source_geometry_modified':False,
        'object_state_writeback_after_reset':False,
        'scope':'Geometry and passive support diagnostic; quantitative impulse comparison reported separately.'}
    try:
        imports = verify_if_snapshot(root,required_modules=(MODULE,),entrypoint_file=Path(__file__))
        if imports is None:
            raise ValueError('Complete sealed source snapshot required')
        dump(output/'runtime_imports.json',imports)
        if os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE')!='1' or not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'):
            raise ValueError('Select a single GPU by UUID')
        if digest(args.input)!=args.input_sha256:
            raise ValueError('Frozen inputs changed')
        inputs = load_json(args.input)
        stack = inputs.get('sdf_stack_candidate') is not None
        if stack:
            report.update(schema='mechanical_ring_sdf_stack.v1',
                scope='Three original complete ring surfaces; finite geometry gate, near-seat passive support and unscaled weight balance. No task score.')
        sdf_sources = inputs.get('sdf_source_meshes', {'sdf_ring':inputs.get('sdf_source_mesh')})
        if inputs['scene']['isaacsim']['solver_type']!='PGS' or len(inputs['cases'])!=1:
            raise ValueError('This frozen diagnostic requires canonical PGS and one support case')
        dump(output/'inputs.json',inputs)
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        env = FR3DexterousIsaacEnv(inputs['scene'],device='cuda:0',render_width=640,render_height=480)
        env.reset(initial_state=inputs['initial_state'])
        convex_view = ReadOnlySelection(env)
        for name in sdf_sources:
            del convex_view.scene_info['objects'][name]
        # Only expected-material metadata is supplied here; no live USD/spec is modified.
        parameter_view = ReadOnlySelection(env)
        for spec in parameter_view.scene_spec['objects']:
            if 'usd_physics_material_fallback' in spec:
                spec['friction'] = spec['usd_physics_material_fallback']['static_friction']
        metadata = env.metadata()
        iteration_candidate=inputs.get('numerical_accuracy_candidate',inputs.get('solver_iteration_candidate'))
        if iteration_candidate is None and 'native_sdf_subgrid_resolution' not in inputs:
            raise ValueError('Frozen SDF input must declare native_sdf_subgrid_resolution')
        subgrid = inputs.get('native_sdf_subgrid_resolution', 6)
        if iteration_candidate:
            subgrid = iteration_candidate.get('expected_sdf_subgrid_resolution', subgrid)
        report['native_sdf_subgrid_resolution'] = subgrid
        sdf_checks = {}
        for name, mesh in sdf_sources.items():
            directory = output/name if stack else output
            directory.mkdir(exist_ok=True)
            sdf_checks[name] = check_ring_sdf(env,name,mesh,directory,subgrid_resolution=subgrid)
        report['checks'] = {'solver_selection':solver_selection(metadata,inputs['scene']['isaacsim']['solver_type']),
            'stage_inventory':stage_inventory(env),
            'contact_filters':contact_filter_coverage(env),
            'native_convex_geometry':check_convex_geometry(convex_view),
            'native_parameters':native_parameters(parameter_view),
            'native_sdf_geometry':{'passed':all(r['passed'] for r in sdf_checks.values()),'objects':sdf_checks}}
        body_parameters = {}
        for name in env.object_names:
            prim = env.stage.GetPrimAtPath(env.scene_info['objects'][name]['path'])
            body_parameters[name] = {k:prim.GetAttribute('physxRigidBody:'+k).Get() for k in (
                'solverPositionIterationCount','solverVelocityIterationCount','enableSpeculativeCCD',
                'disableGravity','linearDamping','angularDamping','sleepThreshold','maxDepenetrationVelocity')}
        report['checks']['equal_body_parameters'] = {'passed':
            all(value==next(iter(body_parameters.values())) for value in body_parameters.values()), 'recorded':body_parameters,
            'scope':'Read-only composed USD effective fields; mass/material are independently read from native tensors.'}
        if iteration_candidate:
            report['numerical_accuracy_candidate']=iteration_candidate
            from scripts.validation.mechanical_iteration_readback import stage_iteration_settings
            report['checks']['candidate_body_parameters']={'passed':
                body_parameters==iteration_candidate['expected_body_parameters'],
                'expected':iteration_candidate['expected_body_parameters'],'recorded':body_parameters}
            report['initial_stage_iteration_settings']=stage_iteration_settings(env)
            scenes=list(report['initial_stage_iteration_settings']['physics_scenes'].values())
            report['checks']['candidate_scene_iteration_clamps']={'passed':bool(scenes) and all(
                r['minVelocityIterationCount']<=body['solverVelocityIterationCount']<=r['maxVelocityIterationCount']
                and r['minPositionIterationCount']<=body['solverPositionIterationCount']<=r['maxPositionIterationCount']
                for r in scenes for body in body_parameters.values())}
        if stack:
            from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
            originals = {}
            for name,spec in inputs['sdf_original_assets'].items():
                original = load_task_asset(**spec,verify_hashes=True)
                native = report['checks']['native_parameters']['objects'][name]
                checks = {'mass':np.isclose(native['mass_kg'],original['mass_kg'],atol=1e-7,rtol=1e-5),
                    'com':np.allclose(native['native_com_principal_pose_xyzw'][:3],original['center_of_mass_m'],atol=1e-7,rtol=0),
                    'inertia':np.allclose(native['native_inertia_body_frame_kg_m2'],np.diag(original['inertia_diag_kg_m2']),atol=1e-9,rtol=2e-5)}
                originals[name] = {'passed':bool(all(checks.values())),'checks':plain(checks),'original':spec}
            report['checks']['original_native_mass_properties'] = {'passed':all(r['passed'] for r in originals.values()),'objects':originals}
        dump(output/'metadata.json',metadata)
        dump(output/'geometry_and_parameters.json',plain(report['checks']))
        if not all(r['passed'] for r in report['checks'].values()):
            raise RuntimeError('Geometry/parameter gate failed; no observed gravity comparison started')
        case = inputs['cases'][0]
        case_root = output/case['id'];case_root.mkdir()
        env.reset(initial_state=inputs['initial_state'])
        if iteration_candidate:
            from scripts.validation.mechanical_iteration_readback import SleepObserver
            observer=SleepObserver(env,case_root)
        else:
            observer = Observer(env,case_root)
        start = int(env.world.current_time_step_index)
        for i in range(round(case['duration_s']/.02)):
            state = env.step({'joint_positions_rad':inputs['initial_state']['joint_positions_rad'],'duration_s':.02})
            if state['terminated'] or not state['metrics']['state_finite']:
                raise RuntimeError('Probe terminated or nonfinite')
            if i%25==0:
                print(json.dumps({'case':case['id'],'sim_time_s':state['sim_time_s']}),flush=True)
        count = int(env.world.current_time_step_index)-start
        rows = observer.rows
        if iteration_candidate:
            sleep=observer.sleep_rows
            report['native_sleep_observation']={'passed':len(sleep)==len(rows) and all(
                a['native_step']==b['native_step'] and a['time_s']==b['time_s']
                for a,b in zip(sleep,rows,strict=True)),
                'samples':len(sleep),'sleeping_samples':{name:sum(r['sleeping'][name] for r in sleep)
                    for name in env.object_names},'sleep_state_write_performed':False}
        exchange = observer.close();observer=None
        result = assess_case(rows,case,env.scene_info,env.dt)
        result['contact_exchange'] = contact_exchange_check(exchange,count,len(rows))
        result['passed'] &= result['contact_exchange']['passed']
        result['rgbd'] = capture(env,case_root,'final')
        result['passed'] &= result['rgbd']['passed']
        masses = {n:r['mass_kg'] for n,r in report['checks']['native_parameters']['objects'].items()}
        report['impulse_comparison'] = (stack_impulse_comparison(rows,env.scene_info,masses,env.dt,case['assertions'])
            if stack else impulse_comparison(rows,env.scene_info,masses,env.dt))
        if stack:
            tail = rows[-round(.5/env.dt):]
            object_paths = {r['path'] for r in env.scene_info['objects'].values()}
            robot_paths = set(env.scene_info['robot_link_paths'].values())
            result['checks']['released_no_robot_manifold'] = not any(
                object_paths.intersection((c['actor0'],c['actor1'])) and
                robot_paths.intersection((c['actor0'],c['actor1'])) for row in tail for c in row['raw_contacts'])
            result['passed'] &= result['checks']['released_no_robot_manifold']
            result['passed'] &= report['impulse_comparison']['quantitative_weight_balance_passed']
        report['support'] = result
        metadata = env.metadata();dump(output/'metadata_final.json',metadata)
        if iteration_candidate:
            report['final_stage_iteration_settings']=stage_iteration_settings(env)
            result['checks']['stage_iteration_settings_unchanged']=(report['initial_stage_iteration_settings']
                ==report['final_stage_iteration_settings'])
            result['passed'] &= result['checks']['stage_iteration_settings_unchanged']
            result['passed'] &= report['native_sleep_observation']['passed']
        report['final_solver_selection'] = solver_selection(metadata,inputs['scene']['isaacsim']['solver_type'])
        guard = metadata['static_mesh_contact_runtime']
        report['fixed_support_guard'] = guard
        report['passed'] = (result['passed'] and report['final_solver_selection']['passed'] and guard['maximum_translation_m']<=1e-6
                            and guard['maximum_angle_rad']<=1e-5)
        dump(output/'runtime_imports_final.json',verify_if_snapshot(root,
            required_modules=(MODULE,),entrypoint_file=Path(__file__)))
    except BaseException as exc:
        report.update(passed=False,error={'type':type(exc).__name__,'detail':str(exc),'traceback':traceback.format_exc()})
    finally:
        if observer is not None:
            report['interrupted_exchange'] = observer.close()
    report['artifacts_sha256'] = {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    return seal_report_before_shutdown(output,plain(report),env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--input-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args)['passed'] else 1)


if __name__=='__main__':
    main()
