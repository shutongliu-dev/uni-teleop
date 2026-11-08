"""Exercise the actual phase-entry block with a tiny robot-only clock harness."""
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation import laundry_captured_garment_fold as runner
from scripts.validation.laundry_prepare_timing import (
    BEFORE_PARK, AFTER_PARK, phase_prepare_timing, prepare_at_stage)
from scripts.validation.laundry_phase_motions import validate_phase_motions, CartesianParkTransition
from scripts.validation.laundry_engineering_motion import minimum_jerk


def entry_harness(timing, *, transient_until=None):
    """Use production AST through via_park, before the first capture batch.

    Models and stepping are CPU doubles, so this proves call order and does not
    qualify native motion or contact. The actual branch/assist construction and
    preparation callbacks are taken directly from native(), not reimplemented.
    """
    tree = ast.parse(inspect.getsource(runner.native))
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'phase_index')
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.For)
               and isinstance(n.target, ast.Tuple)
               and any(isinstance(x, ast.Name) and x.id == 'batch_index' for x in n.target.elts))
    prefix = ast.parse('''
def exercise():
    step=1175
    dt=.02
    phase_index=1
    current=[]
    assists=[]
    capture_sides={'right'}
    seed={'arm':0., 'hand':.4}
    park={'arm':1., 'hand':0.}
    events=[]
    report={'phase_plans':[]}
    def cloth():
        return np.array([[step*dt, 0., 0.]])
    def prepare(*args, **kwargs):
        q=args[3]
        events.append(('prepare', step*dt, q.copy()))
        if transient_until is not None and q[0,0]<transient_until-1.e-10:
            raise ValueError('Original planar guard rejected current transient')
        return {'edges':[{'side':'right','allowed_material_vertex_ids':[0,1,2,3],
                         'contact_plan':{}}]}
    def command(values):
        nonlocal seed
        seed=dict(values)
    def run_action(name, seconds, controller):
        nonlocal step
        events.append((name, step*dt, len(current), sorted(capture_sides)))
        if controller:controller(1.)
        step+=round(seconds/dt)
    def construct(*args, **kwargs):
        events.append(('construct_assist',step*dt))
        return SimpleNamespace(active=False)
    CapturedPatchAssist=construct
''').body[0]
    prefix.body.extend(deepcopy(loop.body[:end]))
    prefix.body.extend(ast.parse('return phase, current, assists, report, events, step*.02').body)
    code = ast.fix_missing_locations(ast.Module(body=[prefix], type_ignores=[]))
    model = SimpleNamespace(frame_id='flange',arm_base=np.eye(4),arm_tip='tip',
        wrist_offset=np.eye(4),hand=SimpleNamespace(active=[SimpleNamespace(name='hand')]),
        arm=SimpleNamespace(forward=lambda *a: {'tip':np.eye(4)},
            inverse=lambda *a,**k: {'converged':True,'joint_positions_rad':{'arm':1.}}))
    scope = dict(np=np,json=json,hashlib=hashlib,SimpleNamespace=SimpleNamespace,
        BEFORE_PARK=BEFORE_PARK,AFTER_PARK=AFTER_PARK,prepare_at_stage=prepare_at_stage,
        preparation_timings=[BEFORE_PARK,timing],release_configurations=[{},
            {'release_profile':'fixed_v1','release_hand_profile':'pinch_open_v1',
             'retreat_profile':'base_directed_v1'}],
        planned_openings=[{},{}],capture_configurations=[{},{}],
        park_transitions=[None,{'side':'left','duration_s':3.,'peak_height_m':.12}],
        models={'left':model},env=SimpleNamespace(body_names=['flange'],joint_names=['arm','hand']),
        robot_readback=lambda _: {'robot_body_q':np.array([[0,0,1,0,0,0,1]]),
                                  'joint_q':np.array([0.,.4])},
        actual_ik_frames=lambda *a:{'left':np.eye(4)},
        CartesianParkTransition=CartesianParkTransition,minimum_jerk=minimum_jerk,
        release=None,task=None,scene=None,export=None,out=Path('/unused'),options={},
        spec={'id':'cloth'},dump=lambda *a:None,transient_until=transient_until)
    exec(compile(code,'<actual native phase-entry>','exec'),scope)
    return scope['exercise']()


def test_default_phase_entry_remains_before_park_and_existing_failure_is_preserved():
    phase, current, assists, report, events, end = entry_harness(BEFORE_PARK)
    assert [e[0] for e in events] == ['prepare','construct_assist','cartesian_to_park','via_park']
    assert events[0][1] == 23.5 and end == 28.5
    assert events[2][2:] == (1, []) and events[3][2:] == (1, [])
    assert 'phase_preparation_events' not in report
    assert phase == report['phase_plans'][0] and current == assists
    with pytest.raises(ValueError,match='Original planar guard'):
        entry_harness(BEFORE_PARK,transient_until=28.5)


def test_optin_native_flow_parks_with_no_new_assist_then_observes_actual_later_cloth():
    phase, current, assists, report, events, end = entry_harness(AFTER_PARK,transient_until=28.5)
    assert [e[0] for e in events] == ['cartesian_to_park','via_park','prepare','construct_assist']
    assert events[0][2:] == (0, []) and events[1][2:] == (0, [])
    assert events[2][1] == events[3][1] == end == 28.5
    event=report['phase_preparation_events'][0]
    assert event['observed_native_step']==1425 and event['profile']==AFTER_PARK
    assert event['observed_cloth_q_float64_sha256']==hashlib.sha256(
        np.asarray([[28.5,0,0]],dtype='<f8').tobytes()).hexdigest()
    assert phase == report['phase_plans'][0] and current == assists
    # Planning uncertainty at the real later observation must still abort.
    with pytest.raises(ValueError,match='Original planar guard'):
        entry_harness(AFTER_PARK,transient_until=29.)


@pytest.mark.parametrize('blocker', ['prior_active','current_constructed','capture_sides'])
@pytest.mark.parametrize('stage', [BEFORE_PARK,AFTER_PARK])
def test_deferred_stage_cannot_start_parking_or_construct_assists_while_holding(blocker,stage):
    kw={'previous_assists':[SimpleNamespace(active=blocker=='prior_active')],
        'current':[object()] if blocker=='current_constructed' else [],
        'capture_sides':{'right'} if blocker=='capture_sides' else set()}
    called=[]
    with pytest.raises(RuntimeError):
        prepare_at_stage(AFTER_PARK,stage,lambda:called.append(True),**kw)
    assert called==[]


@pytest.mark.parametrize('bad',[None,True,[],{},'after_park',17])
def test_bad_unused_phase_timing_rejected_before_geometry(bad):
    options={'phases':{'2':{'prepare_timing':bad}}}
    with pytest.raises(ValueError):
        runner.prepare_phase(None,{'id':'trousers_folding'},{},None,0,options)


def test_explicit_legacy_and_new_timing_leave_duration_and_other_phases_unchanged():
    options={'phases':{'1':{'prepare_timing':AFTER_PARK}}};before=deepcopy(options)
    validate_phase_motions(options,'trousers_folding')
    assert phase_prepare_timing(options,0)==BEFORE_PARK
    assert phase_prepare_timing(options,1)==AFTER_PARK
    assert runner.planned_duration(2,options,2.)==runner.planned_duration(2,{},2.)
    assert options==before
    with pytest.raises(ValueError):
        validate_phase_motions({'phases':{'0':{'prepare_timing':AFTER_PARK}}},'trousers_folding')
    assert phase_prepare_timing({'phases':{'1':{'prepare_timing':BEFORE_PARK}}},1)==BEFORE_PARK
