"""Strict provenance for an authored cloth state followed by native warmup.

No simulator is started here. The two internal World.reset steps are setup,
not operation credit; their intermediate states are not observable in this
contract. Authored geometry and the final warmup/baseline are audited separately.
"""
from pathlib import Path

import numpy as np


SCHEMA = 'laundry_author_only_initialization.v1'
MODES = ('tensor-reset', 'author-only')
WARMUP_STEPS = 2  # Pinned Isaac Sim 6.1 SimulationManager.initialize_physics.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def world_clock_tick(dt_s):
    """Pinned Isaac 6.1 physics callbacks supply a float32 step size."""
    tick = float(np.float32(dt_s))
    require(np.isfinite(tick) and tick > 0, 'Invalid native World clock tick')
    return tick


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def initialization_mode(run):
    if 'initialization' not in run:
        return 'tensor-reset'
    value = run['initialization']
    require(isinstance(value, dict) and value.get('schema') == SCHEMA
            and value.get('mode') in MODES, 'Unknown native cloth initialization contract or mode')
    return value['mode']


class CountedSurfaceView:
    """Local engineering receipt for every q/qd setter on this native handle."""
    def __init__(self, view):
        self._view = view
        self.setter_calls = [0, 0]

    def __getattr__(self, name):
        return getattr(self._view, name)

    def set_simulation_nodal_positions(self, *args, **kwargs):
        self.setter_calls[0] += 1
        return self._view.set_simulation_nodal_positions(*args, **kwargs)

    def set_simulation_nodal_velocities(self, *args, **kwargs):
        self.setter_calls[1] += 1
        return self._view.set_simulation_nodal_velocities(*args, **kwargs)


def authored_arrays(stage, specs):
    """Read actual USD mesh points/topology, never substitute spec positions."""
    from pxr import Gf, Usd, UsdGeom

    result = {}
    for name, spec in specs.items():
        mesh = UsdGeom.Mesh.Get(stage, '/World/Cloths/'+name)
        require(bool(mesh), 'Authored USD cloth mesh is missing: '+name)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        require(points.shape == np.asarray(spec['initial_vertices_m']).shape
                and np.isfinite(points).all(), 'Invalid authored USD cloth points: '+name)
        transform = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world = np.asarray([transform.Transform(Gf.Vec3d(*map(float, p))) for p in points], dtype=np.float32)
        expected = np.asarray(spec['initial_vertices_m'], dtype=np.float32)
        require(same_bits(world, expected), 'Authored USD cloth differs bitwise from frozen initial vertices: '+name)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        require(counts.ndim == 1 and len(counts) and np.all(counts == 3)
                and indices.size == 3*len(counts), 'Authored USD cloth is not the frozen triangle mesh: '+name)
        triangles = indices.reshape(-1, 3).astype(np.int32)
        require(np.array_equal(triangles, spec['triangles']), 'Authored USD cloth topology differs: '+name)
        velocity = mesh.GetVelocitiesAttr().Get()
        require(velocity is None or (np.isfinite(np.asarray(velocity)).all()
                and not np.asarray(velocity).any()), 'Authored USD contains nonzero initial cloth velocities: '+name)
        result[name+'_authored_q'], result[name+'_triangles'] = world, triangles
    return result


def anchor_targets(stage, declared):
    from pxr import Usd, UsdGeom

    targets = []
    for row in declared:
        prim = stage.GetPrimAtPath(row['anchor_path'])
        require(bool(prim), 'Missing authored warmup anchor')
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        quat = matrix.ExtractRotationQuat()
        targets.append([*matrix.ExtractTranslation(), quat.GetReal(), *quat.GetImaginary()])
    return np.asarray(targets, dtype=np.float64)


def authored_control_masks(stage, specs, declared):
    """Bind the enabled finite warmup patches to the actual authored USD."""
    masks = {name: np.zeros(len(spec['initial_vertices_m']), dtype=bool) for name, spec in specs.items()}
    actual = {str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == 'OmniPhysicsVtxXformAttachment'}
    require(actual == {row['path'] for row in declared}, 'Authored warmup attachment inventory differs')
    require(bool(declared), 'Author-only formal runs require declared local attachments')
    for row in declared:
        name = row['cloth_id']
        require(name in specs, 'Warmup attachment references unknown cloth')
        prim = stage.GetPrimAtPath(row['path'])
        require(prim.GetAttribute('omniphysics:attachmentEnabled').Get() is True
                and list(map(str, prim.GetRelationship('omniphysics:src0').GetTargets())) == ['/World/Cloths/'+name]
                and list(map(str, prim.GetRelationship('omniphysics:src1').GetTargets())) == [row['anchor_path']]
                and list(prim.GetAttribute('omniphysics:vtxIndicesSrc0').Get()) == row['vertex_ids'],
                'Actual authored warmup attachment differs from its registry')
        ids = row['vertex_ids']
        require(ids and len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < len(masks[name]) for i in ids),
                'Invalid authored warmup attachment nodes')
        require(not masks[name][ids].any(), 'Duplicate warmup node attachments')
        masks[name][ids] = True
    require(all(mask.sum() <= .10*len(mask) for mask in masks.values()), 'Warmup control exceeds local patch limit')
    return masks


def validate_author_only(arrays, run, scene, task, bind):
    """Verify authored USD, native trace and stream zero as separate evidence."""
    from pxr import Usd

    contract = run['initialization']
    require(initialization_mode(run) == 'author-only', 'Expected explicit author-only initialization')
    require(run['engineering_boundary']['mode'] == 'native_plain_xform_attachment',
            'Author-only initialization is restricted to the native attachment path')
    require(contract.get('warmup_policy') == 'isaacsim_6_1_world_reset_two_steps'
            and contract.get('intermediate_warmup_states_recorded') is False
            and contract.get('warmup_operation_credit') is False,
            'Unknown warmup policy or hidden warmup credit')
    authored = bind(contract['authored_stage'])
    trace_path = bind(contract['trace'])
    require(contract['trace'] == run.get('initialization_trace'), 'Initialization trace bindings disagree')
    parent = Path(run['state_archive']['path']).resolve().parent
    require(authored == parent/'initial.usda' and trace_path == parent/'native_reset_trace.npz',
            'Initialization artifacts do not belong to this native recording')
    specs = {row['id']: row for row in task['asset_metadata']['cloths']}
    scene_specs = {row['id']: row for row in scene['cloths']}
    require(set(scene_specs) == set(specs), 'Scene/task authored cloth inventory differs')
    for name, spec in specs.items():
        require(same_bits(np.asarray(spec['initial_vertices_m'], np.float32),
                          np.asarray(scene_specs[name]['initial_vertices_m'], np.float32)),
                'Scene/task initial cloth states differ')
    stage = Usd.Stage.Open(str(authored))
    require(stage is not None, 'Cannot open bound authored USD')
    usd = authored_arrays(stage, specs)
    declared = run['engineering_boundary']['attachments']
    masks = authored_control_masks(stage, specs, declared)
    targets = anchor_targets(stage, declared)
    actual_targets = arrays['anchor_targets']
    require(actual_targets.shape == (len(arrays['native_step']), len(declared), 7)
            and actual_targets.dtype == np.float64 and np.isfinite(actual_targets).all(),
            'Author-only stream lacks complete finite anchor target poses')
    dt = task['evaluation']['physics_dt_s']
    tick = world_clock_tick(dt)
    require(type(contract.get('world_clock_tick_s')) is float and contract['world_clock_tick_s'] == tick,
            'Declared native World clock tick differs from the frozen float32 physics period')
    zero_calls = {name: [0, 0] for name in specs}
    for field in ('reset_position_velocity_calls_by_cloth', 'final_position_velocity_calls_by_cloth'):
        calls = run.get(field)
        require(calls == zero_calls and all(type(i) is int for value in calls.values() for i in value),
                'Author-only initialization contains missing/nonzero native setter receipts: '+field)
    with np.load(trace_path, allow_pickle=False) as trace:
        recorded_tick = trace['world_clock_tick_s']
        require(recorded_tick.shape == () and recorded_tick.dtype == np.float64
                and float(recorded_tick) == tick, 'Native warmup trace clock tick differs')
        for phase, expected in (('before_reset', 0), ('after_warmup', WARMUP_STEPS), ('at_baseline', WARMUP_STEPS)):
            step, time = trace['world_step_'+phase], trace['world_time_'+phase]
            require(step.shape == time.shape == () and step.dtype.kind in 'iu' and time.dtype.kind in 'fiu'
                    and int(step) == expected and np.isfinite(time) and abs(float(time)-expected*tick) <= 1.e-8,
                    'Author-only initialization has hidden or inconsistent warmup steps: '+phase)
        offset = run.get('native_world_reset_offset', {})
        require(type(offset.get('absolute_step')) is int and offset['absolute_step'] == WARMUP_STEPS
                and type(offset.get('absolute_time_s')) in (int, float)
                and abs(offset['absolute_time_s']-float(trace['world_time_at_baseline'])) <= 1.e-12,
                'Native baseline clock offset differs from warmup trace')
        for phase in ('warmup', 'baseline'):
            require(same_bits(trace[phase+'_anchor_targets'], targets), 'Warmup anchor moved before operation baseline')
        require(same_bits(arrays['anchor_targets'][0], targets), 'Frame-zero anchor targets differ from authored warmup')
        for name, spec in specs.items():
            require(same_bits(trace[name+'_authored_q'], usd[name+'_authored_q'])
                    and same_bits(trace[name+'_triangles'], usd[name+'_triangles'])
                    and np.array_equal(arrays[name+'_triangles'], usd[name+'_triangles']),
                    'Trace authored mesh differs from bound USD: '+name)
            require(same_bits(trace[name+'_warmup_control_mask'], masks[name])
                    and same_bits(arrays[name+'_control_mask'][0], masks[name]),
                    'Warmup and baseline masks differ from authored attachments: '+name)
            for suffix in ('_q', '_qd'):
                warmup, baseline = trace[name+'_warmup'+suffix], trace[name+'_baseline'+suffix]
                require(warmup.dtype == np.float32 and warmup.shape == (len(spec['initial_vertices_m']), 3)
                        and np.isfinite(warmup).all() and same_bits(warmup, baseline)
                        and same_bits(baseline, arrays[name+suffix][0]),
                        'Warmup/baseline/frame-zero cloth state differs bitwise: '+name+suffix)
            calls = trace[name+'_setter_calls']
            require(calls.shape == (2,) and calls.dtype.kind in 'iu' and not calls.any(),
                    'Warmup trace contains native cloth setters: '+name)
    for key in ('absolute_world_step', 'absolute_world_time_s'):
        require(key in arrays, 'Author-only stream lacks actual absolute World clock: '+key)
    steps, times = arrays['absolute_world_step'], arrays['absolute_world_time_s']
    require(steps.shape == arrays['native_step'].shape and steps.dtype.kind in 'iu'
            and np.array_equal(steps, arrays['native_step']+WARMUP_STEPS)
            and times.shape == arrays['time_s'].shape and np.isfinite(times).all()
            and np.allclose(times, steps*tick, atol=1.e-8, rtol=0),
            'Author-only complete stream contains missing or hidden absolute World steps')
    return {'mode': 'author-only', 'schema': SCHEMA, 'verified': True,
        'authored_stage': contract['authored_stage'], 'trace': contract['trace'],
        'world_clock_tick_s': tick, 'relative_task_period_s': dt,
        'baseline_absolute_step': WARMUP_STEPS, 'baseline_absolute_time_s': offset['absolute_time_s'],
        'native_setter_calls_by_cloth': zero_calls, 'warmup_operation_credit': False,
        'intermediate_warmup_states_recorded': False,
        'scope': 'Authored geometry and native warmup endpoint are checked; the two internal intermediate warmup states are not recorded or continuously certified.'}
