"""Trusted runner contracts with real frozen tasks and a CPU-only worker double.

These checks are protocol/lifecycle evidence, not native physics acceptance.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest
import numpy as np
from PIL import Image

from bench.paths import DATA_ROOT, load_json
from bench.protocol import Budgets, EpisodeClock, PolicyRegistry
from bench.protocol.policies import PolicyStep
from bench.protocol.schemas import PolicyDescriptor, RobotJointCommand
from bench.isaacsim.release import DEFAULT, task_inputs, validate
from bench.runtime import service
from bench.runtime.service import AgentService, CreateEpisodeRequest, ServiceError

PRIVATE_MARKER = 'private_evaluator_canary_190734'


def test_frontend_provenance_binds_relocated_sensor_implementations():
    sources = service.frontend_sources()
    for relative in ('bench/sensors/tactile.py',
                     'bench/simulation/isaacsim_backend/camera_calibration.py'):
        assert sources[relative] == hashlib.sha256((service.PROJECT / relative).read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def frozen_inputs():
    """Verify actual shared release once; tests do not substitute a toy task contract."""
    manifest = validate(DEFAULT)
    inputs = task_inputs(DEFAULT, '01_ordered_kitting')
    return manifest, inputs


class ManualTime:
    def __init__(self):
        self.value = 0.
    def __call__(self):
        return self.value


class FakeWorker:
    hand_counts = {'sharpa_w01_right':22, 'allegro_v5_right':16,
                   'leap_v1_right':16, 'shadow_hand_e_right':20}
    def __init__(self, *, output, hand_id, on_step=None, camera_escape=False,
                 depth=False, depth_escape=False, depth_frame_updates=None,
                 sensor_config_updates=None, acknowledged_targets=False,
                 native_velocity_caps=None, **kwargs):
        self.output = Path(output)
        self.output.mkdir(parents=True)
        self.closed = False
        self.closed_force = False
        self.calls = []
        self.deadlines = []
        self.startup_timeout_s = kwargs.get('startup_timeout_s',240)
        self.on_step = on_step
        self.camera_escape = camera_escape
        self.depth = depth
        self.depth_escape = depth_escape
        self.depth_frame_updates = depth_frame_updates or {}
        self.arm = [f'fr3_joint{i}' for i in range(1,8)]
        self.hand = [f'active_finger_joint_{i}' for i in range(self.hand_counts[hand_id])]
        self.active = self.arm+self.hand
        self.raw = {'sim_time_s': 0., 'joint_positions_rad': dict.fromkeys(self.active+['passive_mimic'],0.),
            'joint_velocities_rad_s': dict.fromkeys(self.active+['passive_mimic'],0.),
            'objects': {PRIVATE_MARKER:{'position_m':[.123, .456, .789], 'quaternion_wxyz':[1.,0.,0.,0.]}},
            'cloth':{'vertices_m':[[.731,.521,.117]]}, 'metrics':{'state_finite':True, 'secret':PRIVATE_MARKER},
            'evaluator_debug':PRIVATE_MARKER, 'terminated':False}
        self.score = {'success':False, 'private_detail':PRIVATE_MARKER}
        self.initial = {'metadata':{'robot_usd_sha256':'a'*64, 'native_isaacsim_app':False,
            'scene_spec':{'isaacsim':{'physics_dt_s':.0005}},
            'action_spec':{'joint_names':self.active,'arm_joint_names':self.arm,'hand_joint_names':self.hand,
                           'joint_limits_rad':{n:[-1.,1.] for n in self.active}}},
            'observation':deepcopy(self.raw),'score':deepcopy(self.score)}
        if acknowledged_targets:
            self.raw['robot']={'joint_targets_rad':dict.fromkeys(self.active,0.)}
            self.initial['observation']=deepcopy(self.raw)
        if native_velocity_caps is not None:
            self.initial['metadata']['native_joint_max_velocities']=dict(native_velocity_caps)
        if depth:
            self.initial['metadata']['agent_sensor_config'] = {
                'protocol':'agent_rgbd_v1','rgb':['overview','task'],'width':8,'height':6,
                'depth':True,'depth_encoding':'float32_npy','depth_unit':'m',
                'depth_semantics':'camera_z','depth_invalid_value':0.0,'depth_rgb_aligned':True,
                'camera_frame':'optical_x_right_y_down_z_forward','capture':'decision_and_policy_steps',
                **(sensor_config_updates or {})}

    def request(self, operation, **payload):
        if self.closed:
            raise RuntimeError('fake worker already closed')
        self.calls.append((operation,deepcopy(payload)))
        if operation == 'step':
            command = payload['command']
            assert set(command['joint_positions_rad']) == set(self.active)
            self.raw['sim_time_s'] = round(self.raw['sim_time_s']+command['duration_s'], 10)
            self.raw['joint_positions_rad'].update(command['joint_positions_rad'])
            if self.on_step:
                self.on_step(self)
            return {'ok':True,'observation':deepcopy(self.raw),'score':deepcopy(self.score)}
        if operation == 'cameras':
            path = self.output/'frame.png'
            if self.camera_escape:
                path = self.output.parent/'outside.png'
            Image.new('RGB',(8,6),color=(50,100,150)).save(path)
            frame = dict(path=str(path),sim_time_s=self.raw['sim_time_s'],
                rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),width=8,height=6,
                intrinsics_3x3=[5.,0.,4.,0.,5.,3.,0.,0.,1.],
                world_from_camera={'frame_id':'world','position_m':[1.,2.,3.], 'quaternion_wxyz':[1.,0.,0.,0.]},
                camera_calibration={'source':'RTX camera_params annotator on the RGB-D render product',
                                    'resolution':[8,6],'private_canary':PRIVATE_MARKER},
                render_synchronization={'version':'fabric_scene_delegate_sync_v2',
                                        'effective_settings':{'/app/useFabricSceneDelegate':True}})
            if self.depth:
                depth_path = self.output/f'depth_{payload["sequence"]}.npy'
                if self.depth_escape:
                    depth_path = self.output.parent/'outside_depth.npy'
                values = np.full((6,8),.75,dtype=np.float32)
                values[0,0] = 0.0
                np.save(depth_path,values,allow_pickle=False)
                frame.update(depth_path=str(depth_path),
                    depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                    depth_encoding='float32_npy',depth_unit='m',depth_semantics='camera_z',
                    depth_invalid_value=0.0,depth_rgb_aligned=True,depth_sim_time_s=self.raw['sim_time_s'],
                    depth_statistics={'valid_pixel_count':47,'invalid_pixel_count':1},
                    depth_source='omni.replicator.core.distance_to_image_plane')
                frame.update(self.depth_frame_updates)
            return {'ok':True,'frames':[dict(frame,camera_id=n) for n in ('overview','task')]}
        if operation == 'hold':
            return {'ok':True}
        raise AssertionError(operation)

    def close(self,force=False):
        self.closed = True
        self.closed_force |= force

    def set_deadline(self,value):
        self.deadlines.append(value)


@pytest.fixture
def runner(tmp_path, monkeypatch, frozen_inputs):
    manifest, inputs = frozen_inputs
    monkeypatch.setattr(service,'validate',lambda release:deepcopy(manifest))
    monkeypatch.setattr(service,'task_inputs',lambda release, task_id:deepcopy(inputs))
    clock = ManualTime()
    monkeypatch.setattr(service,'EpisodeClock',lambda budgets:EpisodeClock(budgets,monotonic=clock))
    created = []
    workers = []
    def make(*, registry=None, allow_privileged=False, evaluation_plan=None, **worker_options):
        def factory(**kwargs):
            worker = FakeWorker(**kwargs, **worker_options)
            workers.append(worker)
            return worker
        value = AgentService(tmp_path/f'run_{len(created)}',registry=registry,
                             worker_factory=factory,allow_privileged=allow_privileged,evaluation_plan=evaluation_plan)
        created.append(value)
        return value
    yield make, clock, workers
    for item in created:
        for session in item.sessions.values():
            if session.policy and hasattr(session.policy,'release'):
                session.policy.release.set()
            if session.result is None:
                session.finish()
            if session.thread:
                session.thread.join(timeout=3)


def request(**changes):
    value = dict(task_id='01_ordered_kitting',agent_id='test_agent',agent_version='v1',agent_config_sha256='b'*64)
    value.update(changes)
    return CreateEpisodeRequest(**value)


def session_for(api,**changes):
    reply = api.create(request(**changes))
    return api.get(reply['episode_id']),reply


@pytest.mark.parametrize('seed,variant,expected', [
    (0, 'original_v1', 'seed must match'),
    (11, 'enriched_v2', 'frozen original_v1'),
])
def test_frozen_scenario_rejects_mislabeled_initialization_before_worker(
        runner, frozen_inputs, monkeypatch, seed, variant, expected):
    make, _, workers = runner
    inputs = deepcopy(frozen_inputs[1])
    inputs[1]['scenario'] = {'seed':11}
    monkeypatch.setattr(service, 'task_inputs', lambda *args:deepcopy(inputs))
    with pytest.raises(ServiceError, match=expected):
        make().create(request(seed=seed, scene_variant=variant))
    assert workers == []


def test_frozen_scenario_accepts_matching_initialization(runner, frozen_inputs, monkeypatch):
    make, _, workers = runner
    inputs = deepcopy(frozen_inputs[1])
    inputs[1]['scenario'] = {'seed':11}
    monkeypatch.setattr(service, 'task_inputs', lambda *args:deepcopy(inputs))
    session, _ = session_for(make(), seed=11)
    assert len(workers) == 1
    assert session.result is None


def action(session, name='move_1', duration=.02, sequence=None):
    return {'kind':'direct_joints','action_id':name,'duration_s':duration,
        'observation_sequence':session.sequence if sequence is None else sequence,
        'arm_joint_positions_rad':dict.fromkeys(session.capabilities.arm_joints, .1),
        'hand_joint_positions_rad':dict.fromkeys(session.capabilities.hand_joints, .2)}


def join_action(session):
    assert session.thread is not None
    session.thread.join(timeout=3)
    assert not session.thread.is_alive(), 'CPU action did not terminate'


@pytest.mark.parametrize('robot_id,hand_dof',[('sharpa_w01_right',22),('allegro_v5_right',16),
    ('leap_v1_right',16),('shadow_hand_e_right',20)])
def test_public_projection_and_dynamic_active_joints(runner,robot_id,hand_dof):
    make,_,workers = runner
    session,reply = session_for(make(),robot_id=robot_id)
    assert len(reply['capabilities']['hand_joints']) == hand_dof
    public = json.dumps(reply)
    assert PRIVATE_MARKER not in public
    assert 'passive_mimic' not in public
    assert str(session.directory) not in public
    obs = session.observe()
    assert len(obs['proprioception']['joint_positions_rad']) == 7+hand_dof
    assert obs['cameras'][0]['rgb_uri'].startswith(f'/episodes/{session.directory.name}/images/')
    session.submit(action(session))
    join_action(session)
    assert session.sequence == 1
    executed=[payload['command'] for operation,payload in workers[-1].calls if operation=='step']
    assert len(executed)==1 and set(executed[0])=={'joint_positions_rad','duration_s'}
    assert PRIVATE_MARKER not in json.dumps(session.observe())


def test_exact_action_retry_is_idempotent_after_execution(runner):
    make,_,workers = runner
    session,_ = session_for(make())
    payload=action(session)
    session.submit(payload);join_action(session)
    attempts=session.clock.action_attempts
    count=sum(op=='step' for op,_ in workers[-1].calls)
    retry=session.submit(payload)
    assert retry['status']=='completed'
    assert session.clock.action_attempts==attempts==1
    assert sum(op=='step' for op,_ in workers[-1].calls)==count==1


def test_conflicting_action_id_counts_as_rejected_attempt(runner):
    make,_,_ = runner
    session,_=session_for(make())
    payload=action(session)
    session.submit(payload);join_action(session)
    changed=deepcopy(payload);changed['duration_s']=.04
    with pytest.raises(ServiceError) as error:
        session.submit(changed)
    assert error.value.status==409
    assert session.clock.action_attempts==2
    assert session.sequence==1


@pytest.mark.parametrize('payload',[{'action_id':[]},{'action_id':{}},[],None,{'action_id':'bad','object_state':[1,2,3]}])
def test_malformed_requests_return_contract_error_and_consume_attempt(runner,payload):
    make,_,_ = runner
    session,_=session_for(make())
    with pytest.raises(ServiceError) as error:
        session.submit(payload)
    assert error.value.status==422
    assert session.clock.action_attempts==1
    assert session.sequence==0


def test_stale_observation_and_invalid_targets_never_step(runner):
    make,_,workers=runner
    session,_=session_for(make())
    session.submit(action(session));join_action(session)
    for payload in (action(session,'stale',sequence=0),action(session,'limits')):
        if payload['action_id']=='limits':
            payload['hand_joint_positions_rad'][session.capabilities.hand_joints[0]]=100.
        with pytest.raises(ServiceError) as error:
            session.submit(payload)
        assert error.value.status==422
    assert session.clock.action_attempts==3
    assert sum(op=='step' for op,_ in workers[-1].calls)==1


@pytest.mark.parametrize('succeed,outcome',[(False,'sim_timeout'),(True,'success')])
def test_final_native_step_on_sim_horizon_is_scored(runner,succeed,outcome):
    make,_,workers=runner
    def score(worker): worker.score['success']=succeed
    session,_=session_for(make(on_step=score),budgets=Budgets(sim_time_s=.02))
    session.submit(action(session));join_action(session)
    assert session.result.outcome==outcome
    assert session.result.elapsed_sim_s==.02
    assert workers[-1].closed
    assert PRIVATE_MARKER not in json.dumps(session.result.model_dump(mode='json'))


def test_oversize_final_step_never_advances_physics(runner):
    make,_,workers=runner
    session,_=session_for(make(),budgets=Budgets(sim_time_s=.03))
    session.submit(action(session,duration=.04));join_action(session)
    assert session.result.outcome=='sim_timeout'
    assert session.result.elapsed_sim_s==.02
    assert sum(op=='step' for op,_ in workers[-1].calls)==1


def test_late_native_success_becomes_wall_timeout(runner):
    make,clock,workers=runner
    def slow_step(worker):
        clock.value=1.01
        worker.score['success']=True
    session,_=session_for(make(on_step=slow_step),budgets=Budgets(wall_time_s=1.))
    session.submit(action(session));join_action(session)
    assert session.result.outcome=='wall_timeout'
    assert session.result.completion_sim_s is None
    assert workers[-1].closed


def test_idle_wall_watchdog_releases_worker(runner):
    make,clock,workers=runner
    session,_=session_for(make(),budgets=Budgets(wall_time_s=1.))
    clock.value=1.
    assert session.done.wait(timeout=1)
    assert session.result.outcome=='wall_timeout'
    assert workers[-1].closed


def test_unlimited_session_disables_watchdog_and_native_deadlines(runner):
    make,clock,workers=runner
    def slow_step(worker):clock.value=1e9
    session,reply=session_for(make(on_step=slow_step),budgets=Budgets(wall_time_s=None,sim_time_s=None))
    assert reply['spec']['budgets']['wall_time_s'] is None
    assert reply['spec']['budgets']['sim_time_s'] is None
    assert session.watchdog is None
    assert workers[-1].startup_timeout_s is None
    session.submit(action(session));join_action(session)
    assert session.result is None
    assert session.status('move_1')['status']=='completed'
    terminal=session.finish()
    assert terminal['outcome']=='cancelled' and terminal['elapsed_wall_s']==1e9
    assert workers[-1].deadlines and all(value is None for value in workers[-1].deadlines)
    assert (session.directory/'artifact_hashes.json').is_file()


def test_final_camera_failure_seals_result_without_falsifying_timestamps(runner):
    make,_,workers=runner
    session,_=session_for(make(),budgets=Budgets(wall_time_s=None,sim_time_s=None))
    session.submit(action(session));join_action(session)
    worker=workers[-1]
    original=worker.request
    def reject_cameras(operation,**payload):
        if operation=='cameras':raise RuntimeError('terminal capture unavailable')
        return original(operation,**payload)
    worker.request=reject_cameras
    terminal=session.finish()
    assert terminal['outcome']=='infrastructure_error' and terminal['physics_valid'] is False
    assert worker.closed and not worker.closed_force
    assert session.observe()['cameras']==[]
    assert session.observe()['sim_time_s']==.02
    assert session.done.is_set() and session.recorder.closed
    for name in ['agent/result.json','agent/artifacts.json','artifact_hashes.json','private/finalize_error.json']:
        assert (session.directory/name).is_file()
    errors=load_json(session.directory/'private/finalize_error.json')['errors']
    assert errors[0]['stage']=='terminal_cameras'


def test_lost_step_reply_preserves_acknowledged_state_and_terminal_archive(runner):
    make,_,workers=runner
    steps=[0]
    def lose_second_reply(worker):
        steps[0]+=1
        if steps[0]==2:
            worker.broken=True
            raise TimeoutError('step reply lost')
    session,_=session_for(make(on_step=lose_second_reply),budgets=Budgets(wall_time_s=None,sim_time_s=None))
    # Both steps are within one action: camera remains at reset while the first
    # step is acknowledged. The second state exists only in the native worker.
    session.submit(action(session,duration=.04));join_action(session)
    assert session.result.outcome=='infrastructure_error'
    assert session.result.physics_valid is False
    assert session.result.elapsed_sim_s==.02
    assert workers[-1].raw['sim_time_s']==.04
    assert session.observe()['sim_time_s']==.02 and session.observe()['cameras']==[]
    assert workers[-1].closed_force and session.done.is_set()
    assert (session.directory/'agent/result.json').is_file()
    hashes=load_json(session.directory/'artifact_hashes.json')
    assert 'agent/result.json' in hashes and 'private/action_error.json' in hashes


def test_nonfinite_terminal_joints_still_seal_invalid_result_and_preserve_public_snapshot(runner):
    make,_,workers=runner
    def invalid_step(worker):
        worker.raw['joint_positions_rad'][worker.arm[0]]=float('nan')
        worker.raw['metrics']['state_finite']=False
    session,_=session_for(make(on_step=invalid_step,depth=True),budgets=Budgets(wall_time_s=None,sim_time_s=None))
    previous=session.observe()
    public_images=dict(session.public_images)
    public_depths=dict(session.public_depths)
    session.submit(action(session));join_action(session)
    assert session.result.outcome=='infrastructure_error' and session.result.physics_valid is False
    assert session.result.failure_detail=='terminal_observation_missing'
    assert session.result.elapsed_sim_s==.02
    assert session.result.completion_sim_s is None and session.result.completion_wall_s is None
    assert np.isnan(session.raw['joint_positions_rad'][workers[-1].arm[0]])
    assert session.observe()==previous
    assert session.public_images==public_images and session.public_depths==public_depths
    assert session.observe()['sim_time_s']==0. and session.sequence==1
    assert workers[-1].closed and session.recorder.closed and session.done.is_set()
    events=[json.loads(line) for line in (session.directory/'agent/events.jsonl').read_text().splitlines()]
    observations=[row for row in events if row['kind']=='agent_observation']
    assert len(observations)==1 and observations[0]['payload']['sim_time_s']==0.
    missing=load_json(session.directory/'private/terminal_observation_missing.json')
    assert missing=={'terminal_observation_missing':True,'attempted_sequence':1,
                    'acknowledged_sim_time_s':.02,'last_public_observation_sequence':0,
                    'last_public_observation_sim_time_s':0.,'last_public_observation_preserved':True,
                    'terminal_observation_appended':False}
    hashes=load_json(session.directory/'artifact_hashes.json')
    assert 'agent/result.json' in hashes and 'private/terminal_observation_missing.json' in hashes
    for name,expected in hashes.items():
        assert hashlib.sha256((session.directory/name).read_bytes()).hexdigest()==expected


def test_final_projection_exception_cannot_seal_success_or_repeat_old_observation(runner,monkeypatch):
    make,_,workers=runner
    session,_=session_for(make(),budgets=Budgets(wall_time_s=None,sim_time_s=None))
    session.submit(action(session));join_action(session)
    previous=session.observe()
    events_path=session.directory/'agent/events.jsonl'
    event_count=len(events_path.read_text().splitlines())
    def failed_projection():raise RuntimeError('FK projection unavailable')
    monkeypatch.setattr(session,'_projection',failed_projection)
    with session.lock:terminal=session._finish('success')
    assert terminal.outcome=='infrastructure_error' and terminal.physics_valid is False
    assert terminal.failure_detail=='terminal_observation_missing'
    assert terminal.completion_sim_s is None and terminal.completion_wall_s is None
    assert len(events_path.read_text().splitlines())==event_count
    assert session.observe()==previous and session.finish()==terminal.model_dump(mode='json')
    assert workers[-1].closed and session.done.is_set()
    errors=load_json(session.directory/'private/finalize_error.json')
    assert errors['terminal_observation_missing'] is True
    assert errors['errors'][0]['stage']=='terminal_projection'


def test_native_exit_archive_contains_only_real_captured_states(tmp_path):
    from bench.runtime.worker import seal_native_recording
    states=[{'time_s':np.array([.0]),'joint_q':np.array([1.,2.])},
            {'time_s':np.array([.02]),'joint_q':np.array([1.1,2.2])}]
    triangles=np.array([[0,1,2]],dtype=np.int32)
    seal_native_recording(tmp_path,states,triangles,{'success':False},reason='worker_exit_without_finish')
    with np.load(tmp_path/'states.npz') as saved:
        assert np.array_equal(saved['joint_q'],np.stack([state['joint_q'] for state in states]))
        assert np.array_equal(saved['time_s'],np.array([[0.],[.02]]))
    assert np.array_equal(np.load(tmp_path/'triangles.npy'),triangles)
    metadata=load_json(tmp_path/'worker_archive.json')
    assert metadata=={'state_count':2,'reason':'worker_exit_without_finish',
                      'source':'captured_native_states','state_reconstruction':False}


def test_invalid_attempt_budget_stops_episode(runner):
    make,_,workers=runner
    session,_=session_for(make(),budgets=Budgets(max_actions=1))
    with pytest.raises(ServiceError):session.submit({'action_id':'invalid'})
    with pytest.raises(ServiceError) as error:session.submit({'action_id':'invalid_2'})
    assert error.value.status==409
    assert session.result.outcome=='action_budget'
    assert session.result.action_attempts==1
    assert workers[-1].closed


class Teacher:
    def __init__(self, terminal='completed',block=False):
        self.descriptor=PolicyDescriptor(policy_id='test_teacher',version='v1',implementation_sha256='c'*64,
            trained=False,observation_profile='privileged_state',skills=['test_skill'])
        self.terminal=terminal
        self.block=block
        self.started=threading.Event();self.release=threading.Event()
        self.observations=[];self.cancel_reasons=[]
    def reset(self,spec,capabilities):self.capabilities=capabilities
    def begin(self,action,observation):self.observations.append(observation)
    def step(self,observation):
        self.observations.append(observation);self.started.set()
        if self.block:assert self.release.wait(timeout=3)
        if self.terminal=='running':
            return PolicyStep(status='running',command=RobotJointCommand(duration_s=.02,
                joint_positions_rad=dict.fromkeys(self.capabilities.joint_limits,0.)),detail=PRIVATE_MARKER)
        return PolicyStep(status=self.terminal,detail=PRIVATE_MARKER)
    def cancel(self,reason):self.cancel_reasons.append(reason)


def teacher_session(make,teacher,**changes):
    registry=PolicyRegistry();registry.register(teacher)
    return session_for(make(registry=registry),track='system0',policy_id='test_teacher',policy_version='v1',**changes)[0]


def skill_action(session):
    return {'kind':'system0','action_id':'skill_1','observation_sequence':session.sequence,
            'skill':'test_skill','timeout_sim_s':.04}


class TimedHoldTeacher(Teacher):
    """CPU test double for time advancement, not contact/physics validation."""
    def __init__(self, premature=False):
        super().__init__()
        self.descriptor.skills=['hold']
        self.descriptor.camera_observation_mode='decision_only'
        self.premature=premature

    def begin(self,action,observation):
        self.start=observation.sim_time_s
        self.duration=action.hold_duration_s
        self.target=dict(observation.proprioception.joint_positions_rad)

    def step(self,observation):
        if self.premature or observation.sim_time_s-self.start >= self.duration-1e-9:
            return PolicyStep(status='completed',detail=PRIVATE_MARKER)
        return PolicyStep(status='running',command=RobotJointCommand(duration_s=.02,
            joint_positions_rad=self.target))


@pytest.mark.parametrize('duration',[.02,.3,1.])
def test_timed_hold_requires_real_worker_steps_to_requested_duration(runner,duration):
    make,_,workers=runner
    policy=TimedHoldTeacher()
    session=teacher_session(make,policy)
    payload=dict(kind='system0',action_id='hold1',observation_sequence=session.sequence,
        skill='hold',hold_duration_s=duration,timeout_sim_s=duration)
    session.submit(payload);join_action(session)
    status=session.status('hold1')
    assert status['status']=='completed'
    assert status['elapsed_sim_s']==pytest.approx(duration)
    assert len([op for op,_ in workers[-1].calls if op=='step'])==round(duration/.02)
    assert all(op!='hold' for op,_ in workers[-1].calls)
    assert session.result is None
    assert PRIVATE_MARKER not in json.dumps(status)


def test_premature_policy_hold_completion_is_rejected(runner):
    make,_,workers=runner
    session=teacher_session(make,TimedHoldTeacher(premature=True))
    session.submit(dict(kind='system0',action_id='hold1',observation_sequence=session.sequence,
        skill='hold',hold_duration_s=.3,timeout_sim_s=.3))
    join_action(session)
    assert session.result.outcome=='policy_error'
    assert not any(op=='step' for op,_ in workers[-1].calls)


class AcknowledgedTeacher(Teacher):
    """A candidate is never a committed reference until the worker replies."""
    def __init__(self,*,block=False,invalid=False):
        super().__init__(terminal='running',block=block)
        self.descriptor.camera_observation_mode='decision_only'
        self.pending=False
        self.commits=self.discards=0
        self.invalid=invalid

    def step(self,observation):
        assert not self.pending
        result=super().step(observation)
        self.pending=True
        if self.invalid:
            result.command.joint_positions_rad[self.capabilities.arm_joints[0]]=100.
        return result

    def commit_pending_step(self):
        assert self.pending
        self.pending=False
        self.commits+=1

    def discard_pending_step(self):
        if self.pending:
            self.discards+=1
            self.pending=False


def test_worker_ack_commits_but_final_check_candidate_is_discarded(runner):
    make,_,workers=runner
    teacher=AcknowledgedTeacher()
    session=teacher_session(make,teacher)
    session.submit(skill_action(session));join_action(session)
    assert teacher.commits==2 and teacher.discards==1 and not teacher.pending
    assert len([op for op,_ in workers[-1].calls if op=='step'])==2
    assert session.status('skill_1')['status']=='failed'
    assert teacher.cancel_reasons==['action_timeout']


def test_cancellation_while_candidate_is_computed_never_commits_it(runner):
    make,_,workers=runner
    teacher=AcknowledgedTeacher(block=True)
    session=teacher_session(make,teacher)
    session.submit(skill_action(session));assert teacher.started.wait(timeout=1)
    session.cancel('skill_1');teacher.release.set();join_action(session)
    assert teacher.commits==0 and teacher.discards==1 and not teacher.pending
    assert not any(op=='step' for op,_ in workers[-1].calls)
    assert session.status('skill_1')['status']=='cancelled'


def test_invalid_candidate_never_advances_policy_reference(runner):
    make,_,workers=runner
    teacher=AcknowledgedTeacher(invalid=True)
    session=teacher_session(make,teacher)
    session.submit(skill_action(session));join_action(session)
    assert teacher.commits==0 and teacher.discards==1 and not teacher.pending
    assert not any(op=='step' for op,_ in workers[-1].calls)
    assert session.result.outcome=='policy_error'


def test_worker_reply_is_committed_before_task_success_finishes_episode(runner):
    make,_,workers=runner
    teacher=AcknowledgedTeacher()
    session=teacher_session(make,teacher)
    workers[-1].on_step=lambda worker:worker.score.update(success=True)
    session.submit(skill_action(session));join_action(session)
    assert teacher.commits==1 and teacher.discards==0 and not teacher.pending
    assert session.result.outcome=='success'


def test_worker_transport_failure_does_not_confirm_unacknowledged_step(runner):
    make,_,workers=runner
    teacher=AcknowledgedTeacher()
    session=teacher_session(make,teacher)
    def fail_reply(_):raise RuntimeError('No step acknowledgement')
    workers[-1].on_step=fail_reply
    session.submit(skill_action(session));join_action(session)
    assert teacher.commits==0 and teacher.discards==1 and not teacher.pending
    assert session.result.outcome=='infrastructure_error'


def test_public_robot_diagnostics_survive_and_private_detail_does_not(runner):
    from bench.protocol.schemas import ActionDiagnostic
    make,_,_=runner
    teacher=Teacher(terminal='failed')
    session=teacher_session(make,teacher)
    joint=session.capabilities.arm_joints[0]
    teacher.step=lambda _:PolicyStep(status='failed',detail=PRIVATE_MARKER,
        diagnostic=ActionDiagnostic(code='trajectory_limit_guard',joint_name=joint,joint_error_rad=.001))
    session.submit(skill_action(session));join_action(session)
    feedback=session.status('skill_1')
    assert feedback['diagnostic']['code']=='trajectory_limit_guard'
    assert feedback['diagnostic']['joint_name']==joint and feedback['diagnostic']['joint_error_rad']==.001
    assert PRIVATE_MARKER not in json.dumps(feedback)
    assert session.life.events[-1].diagnostic.code=='trajectory_limit_guard'
    assert session.life.events[-1].detail==''
    assert session._safe_diagnostic({'code':'trajectory_limit_guard','joint_name':'unadvertised'}).joint_name is None
    assert session._safe_diagnostic({'code':'trajectory_limit_guard','private':PRIVATE_MARKER}) is None


def test_teacher_private_observation_and_detail_do_not_reach_agent(runner):
    make,_,_=runner
    teacher=Teacher()
    session=teacher_session(make,teacher)
    session.submit(skill_action(session));join_action(session)
    assert teacher.observations[0].privileged_object_poses is not None
    assert PRIVATE_MARKER in teacher.observations[0].privileged_object_poses
    assert teacher.observations[0].profile=='privileged_state'
    assert session.observe()['profile']=='vision_proprio'
    assert PRIVATE_MARKER not in json.dumps(session.observe())
    assert PRIVATE_MARKER not in json.dumps(session.status('skill_1'))
    assert all(event.detail=='' for event in session.life.events)
    assert session.result is None  # skill completion is not task success


@pytest.mark.parametrize('terminal',['running','completed','failed'])
def test_cancel_during_policy_call_preserves_target_and_wins_over_terminal_return(runner,terminal):
    make,_,workers=runner
    teacher=Teacher(terminal=terminal,block=True)
    session=teacher_session(make,teacher)
    session.submit(skill_action(session));assert teacher.started.wait(timeout=1)
    assert session.cancel('skill_1')['cancellation_requested']
    teacher.release.set();join_action(session)
    assert session.status('skill_1')['status']=='cancelled'
    assert sum(op=='hold' for op,_ in workers[-1].calls)==0
    assert sum(op=='step' for op,_ in workers[-1].calls)==0
    assert teacher.cancel_reasons


def test_wall_timeout_during_policy_call_does_not_allow_late_command(runner):
    make,clock,workers=runner
    teacher=Teacher(terminal='running',block=True)
    session=teacher_session(make,teacher,budgets=Budgets(wall_time_s=1.))
    session.submit(skill_action(session));assert teacher.started.wait(timeout=1)
    clock.value=1.1
    assert session.done.wait(timeout=1)
    teacher.release.set();join_action(session)
    assert session.result.outcome=='wall_timeout'
    assert sum(op=='step' for op,_ in workers[-1].calls)==0
    assert workers[-1].closed


def test_explicit_episode_finish_cancels_active_action(runner):
    make,_,_=runner
    teacher=Teacher(block=True)
    session=teacher_session(make,teacher)
    session.submit(skill_action(session));assert teacher.started.wait(timeout=1)
    reply=session.finish()
    teacher.release.set();join_action(session)
    assert reply['outcome']=='cancelled'
    assert session.status('skill_1')['status']=='cancelled'


def test_escaped_camera_path_cleans_up_worker(runner):
    make,_,workers=runner
    with pytest.raises(RuntimeError,match='escaped'):
        session_for(make(camera_escape=True))
    assert workers[-1].closed and workers[-1].closed_force


def test_escaped_depth_path_cleans_up_worker(runner):
    make,_,workers=runner
    with pytest.raises(RuntimeError,match='escaped'):
        session_for(make(depth=True,depth_escape=True))
    assert workers[-1].closed and workers[-1].closed_force


@pytest.mark.parametrize('updates',[
    {'depth_sha256':None}, {'depth_sha256':'0'*64},
    {'depth_sim_time_s':.02}, {'depth_encoding':None},
    {'sim_time_s':.02,'depth_sim_time_s':.02},
    {'depth_path':None,'depth_sha256':None,'depth_encoding':None,'depth_semantics':None,
     'depth_invalid_value':None,'depth_rgb_aligned':None,'depth_sim_time_s':None},
    {'width':7}, {'rgb_sha256':'0'*64},
])
def test_depth_setup_fails_closed_for_missing_hash_stale_frame_or_metadata(runner,updates):
    make,_,workers=runner
    with pytest.raises((RuntimeError,ValueError)):
        session_for(make(depth=True,depth_frame_updates=updates))
    assert workers[-1].closed and workers[-1].closed_force


@pytest.mark.parametrize('updates',[{'depth':False},{'depth_semantics':'ray_distance'},
    {'depth_unit':'mm'},{'depth_rgb_aligned':False},{'protocol':'unversioned'},
    {'rgb':['overview']},{'width':True}])
def test_depth_capability_requires_supported_worker_metadata_and_matching_frames(runner,updates):
    make,_,workers=runner
    with pytest.raises((RuntimeError,ValueError)):
        session_for(make(depth=True,sensor_config_updates=updates))
    assert workers[-1].closed and workers[-1].closed_force


def test_rgbd_public_endpoint_serves_aligned_metric_npy_and_pins_sensor_contract(runner):
    from io import BytesIO
    from fastapi.testclient import TestClient
    from bench.runtime.server import create_app
    make,_,_=runner
    rgb_session,rgb_reply=session_for(make())
    assert rgb_reply['capabilities']['depth_available'] is False
    for camera in rgb_reply['observation']['cameras']:
        assert not {'camera_calibration','render_synchronization'} & camera.keys()
    api=make(depth=True)
    with TestClient(create_app(service=api),raise_server_exceptions=False) as client:
        response=client.post('/episodes',json=request().model_dump(mode='json'))
        assert response.status_code==201,response.text
        reply=response.json();episode=reply['episode_id'];session=api.get(episode)
        assert reply['capabilities']['depth_available'] is True
        assert PRIVATE_MARKER not in response.text and str(session.directory) not in response.text
        config=load_json(session.directory/'sensor_config.json')
        assert service.canonical_hash(config)==reply['spec']['sensor_config_sha256']
        assert reply['spec']['sensor_config_sha256']!=rgb_reply['spec']['sensor_config_sha256']
        assert config['protocol']=='agent_rgbd_v1' and config['depth_semantics']=='camera_z'
        assert config['depth_encoding']=='float32_npy' and config['depth_rgb_aligned'] is True
        assert load_json(session.directory/'setup.json')['depth_available'] is True
        for camera in reply['observation']['cameras']:
            assert camera['depth_sim_time_s']==camera['sim_time_s']==reply['observation']['sim_time_s']
            assert camera['depth_unit']=='m' and camera['depth_invalid_value']==0.0
            assert 'depth_statistics' not in camera and 'depth_source' not in camera
            assert not {'camera_calibration','render_synchronization'} & camera.keys()
            depth=client.get(camera['depth_uri'])
            assert depth.status_code==200 and depth.headers['content-type']=='application/octet-stream'
            assert hashlib.sha256(depth.content).hexdigest()==camera['depth_sha256']
            array=np.load(BytesIO(depth.content),allow_pickle=False)
            assert array.dtype==np.float32 and array.shape==(camera['height'],camera['width'])
            assert array[0,0]==0.0 and array[-1,-1]==pytest.approx(.75)
        for route in ('0/metadata.json','-1/task','999999/task','0/..%2F..%2Fprivate%2Fmetadata.json'):
            response=client.get(f'/episodes/{episode}/depth/{route}')
            assert response.status_code in (404,422)
            assert PRIVATE_MARKER not in response.text
        # The old RGB-only session never invents a public depth frame.
        with pytest.raises(ServiceError) as error:
            rgb_session.read_public_depth(0,'task')
        assert error.value.status==404


def test_public_depth_rejects_changed_bytes_and_symlink_escape(runner,tmp_path):
    make,_,_=runner
    session,_=session_for(make(depth=True))
    path,_=session.public_depths[(0,'task')]
    original=path.read_bytes()
    path.write_bytes(b'changed sensor payload')
    with pytest.raises(RuntimeError,match='digest'):
        session.read_public_depth(0,'task')
    outside=tmp_path/'outside_depth.npy'
    outside.write_bytes(original)
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(RuntimeError,match='escaped'):
        session.read_public_depth(0,'task')
    path.unlink()
    path.write_bytes(original)


def test_privileged_agent_profile_requires_explicit_operator_enable(runner):
    make,_,workers=runner
    with pytest.raises(ServiceError) as error:
        session_for(make(),observation_profile='privileged_state')
    assert error.value.status==403
    assert not workers
    session,_=session_for(make(allow_privileged=True),observation_profile='privileged_state')
    assert PRIVATE_MARKER in session.observe()['privileged_object_poses']
    assert 'evaluator_debug' not in session.observe()


def test_manus_adapter_direct_action_roundtrip_into_runner(runner):
    from bench.adapters.manus.agent_adapter import build_command,direct_action
    from bench.adapters.manus.bridge import CaptureBridge,RetargetRegistry
    from bench.adapters.manus.schema import CaptureConfig
    from bench.adapters.manus.sources import mock_packets
    make,_,workers=runner
    session,_=session_for(make(),robot_id='leap_v1_right')
    sample=next(mock_packets(1,clock_id='test_clock',start_ns=1_000_000_000))
    config=CaptureConfig(calibration_id='synthetic_identity',target_clock_id='test_clock',
        glove_calibrated=True,robot_profile_id='leap_v1_right',retarget_version='test_mapper')
    frame=CaptureBridge(config).accept(sample,1_000_000_000).frame
    registry=RetargetRegistry()
    registry.register('leap_v1_right','test_mapper',lambda frame:dict.fromkeys(session.capabilities.joint_limits,.01))
    command,mapping=build_command(frame,registry,session.capabilities,
        previous_targets=dict.fromkeys(session.capabilities.joint_limits,0.),now_ns=1_000_000_000,now_clock_id='test_clock')
    payload=direct_action(command,session.capabilities,action_id='manus_tick_0',observation_sequence=0)
    session.submit(payload.model_dump(mode='json'));join_action(session)
    assert session.sequence==1 and mapping['mode']=='hand_only'
    stepped=[p['command'] for op,p in workers[-1].calls if op=='step'][0]
    assert all(stepped['joint_positions_rad'][n]==0. for n in session.capabilities.arm_joints)
    assert all(stepped['joint_positions_rad'][n]==.01 for n in session.capabilities.hand_joints)


@pytest.mark.parametrize('matrix,expected',[
    ([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],[1.,0.,0.,0.]),
    ([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[0.,1.,0.,0.]),
    ([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]],[0.,0.,1.,0.]),
    ([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]],[0.,0.,0.,1.]),
])
def test_camera_quaternion_conversion_handles_180_degree_branches(matrix,expected):
    from bench.simulation.isaacsim_backend.camera_calibration import rotation_quaternion
    assert rotation_quaternion(matrix)==pytest.approx(expected)


def test_http_projection_camera_routes_and_private_paths(runner):
    from fastapi.testclient import TestClient
    from bench.runtime.server import create_app
    make,_,_=runner
    api=make()
    with TestClient(create_app(service=api),raise_server_exceptions=False) as client:
        assert client.get('/health').json()['backend']=='isaacsim_native_physx'
        response=client.post('/episodes',json=request().model_dump(mode='json'))
        assert response.status_code==201,response.text
        reply=response.json();episode=reply['episode_id']
        assert PRIVATE_MARKER not in response.text
        assert DATA_ROOT not in response.text
        image_url=reply['observation']['cameras'][0]['rgb_uri']
        image=client.get(image_url)
        assert image.status_code==200 and image.headers['content-type']=='image/png'
        assert hashlib.sha256(image.content).hexdigest()==reply['observation']['cameras'][0]['rgb_sha256']
        malicious=[f'/episodes/{episode}/images/0/metadata.json',
                   f'/episodes/{episode}/images/0/..%2F..%2Fprivate%2Fmetadata.json',
                   f'/episodes/{episode}/images/-1/task',
                   f'/episodes/{episode}/images/999999/overview',
                   f'/episodes/{episode}/private/metadata.json',
                   '/episodes/%2E%2E%2Fprivate/observation']
        for url in malicious:
            value=client.get(url)
            assert value.status_code in (404,422), (url,value.status_code,value.text)
            assert PRIVATE_MARKER not in value.text
        assert client.get(f'/episodes/{episode}/result').status_code==409
        session=api.get(episode)
        assert client.post(f'/episodes/{episode}/actions',json=action(session)).status_code==202
        join_action(session)
        observation=client.get(f'/episodes/{episode}/observation')
        assert observation.json()['sequence']==1
        assert PRIVATE_MARKER not in observation.text
        assert client.post(f'/episodes/{episode}/finish').json()['outcome']=='cancelled'
        result=client.get(f'/episodes/{episode}/result')
        assert result.status_code==200 and PRIVATE_MARKER not in result.text


@pytest.mark.parametrize('read_endpoint',['image','result'])
def test_http_blocked_native_read_keeps_health_and_control_requests_schedulable(runner,monkeypatch,read_endpoint):
    """A slow RPC may serialize its episode, but must not block the ASGI loop."""
    import asyncio
    import httpx
    from bench.runtime.server import create_app

    make,_,_=runner
    native_entered=threading.Event()
    release_native=threading.Event()
    forced_release=threading.Event()
    read_entered=threading.Event()
    cancel_entered=threading.Event()
    finish_entered=threading.Event()

    def blocked_step(worker):
        native_entered.set()
        release_native.wait()

    api=make(on_step=blocked_step)
    session,reply=session_for(api,budgets=Budgets(wall_time_s=None,sim_time_s=None))
    episode=reply['episode_id']
    get=api.get
    cancel=session.cancel
    finish=session.finish

    def tracked_get(episode_id):
        read_entered.set()
        return get(episode_id)

    def tracked_cancel(action_id):
        cancel_entered.set()
        return cancel(action_id)

    def tracked_finish():
        finish_entered.set()
        return finish()

    monkeypatch.setattr(api,'get',tracked_get)
    monkeypatch.setattr(session,'cancel',tracked_cancel)
    monkeypatch.setattr(session,'finish',tracked_finish)
    session.submit(action(session))
    assert native_entered.wait(timeout=2)
    url=(reply['observation']['cameras'][0]['rgb_uri'] if read_endpoint=='image'
         else f'/episodes/{episode}/result')

    # asyncio timeouts cannot rescue a synchronously blocked event loop. Use an
    # independent release timer so this regression fails instead of hanging.
    def rescue():
        forced_release.set()
        release_native.set()

    rescue_timer=threading.Timer(3,rescue)
    rescue_timer.start()

    async def check():
        transport=httpx.ASGITransport(app=create_app(service=api),raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport,base_url='http://in-memory') as client:
            pending=[]
            try:
                pending.append(asyncio.create_task(client.get(url)))
                assert await asyncio.to_thread(read_entered.wait,1)
                pending.append(asyncio.create_task(client.post(f'/episodes/{episode}/actions/move_1/cancel')))
                pending.append(asyncio.create_task(client.post(f'/episodes/{episode}/finish')))
                assert await asyncio.to_thread(cancel_entered.wait,1)
                assert await asyncio.to_thread(finish_entered.wait,1)
                health=await asyncio.wait_for(client.get('/health'),timeout=1)
                assert health.status_code==200
                assert health.json()['status']=='ok'
                assert not forced_release.is_set(), 'An episode read blocked the ASGI event loop'
                assert all(not task.done() for task in pending)
                # Scheduling cancel/finish is distinct from interrupting an
                # in-flight physics RPC. They still wait for its session lock.
                assert not session.cancelled.is_set()
            finally:
                release_native.set()
                responses=await asyncio.gather(*pending)
            assert responses[0].status_code in ((200,) if read_endpoint=='image' else (200,409))
            assert responses[1].status_code==responses[2].status_code==200
            assert responses[2].json()['outcome']=='cancelled'
            assert all(PRIVATE_MARKER not in response.text for response in responses[1:])

    try:
        asyncio.run(check())
    finally:
        release_native.set()
        rescue_timer.cancel()
        rescue_timer.join(timeout=1)
        join_action(session)


def test_http_malformed_body_consumes_episode_attempt(runner):
    from fastapi.testclient import TestClient
    from bench.runtime.server import create_app
    make,_,_=runner
    api=make()
    with TestClient(create_app(service=api),raise_server_exceptions=False) as client:
        episode=client.post('/episodes',json=request().model_dump(mode='json')).json()['episode_id']
        response=client.post(f'/episodes/{episode}/actions',json=[])
        assert response.status_code==422
        assert api.get(episode).clock.action_attempts==1


def test_http_worker_diagnostics_are_not_returned(runner):
    from fastapi.testclient import TestClient
    from bench.runtime.server import create_app
    make,_,_=runner
    def explode(worker):raise RuntimeError(PRIVATE_MARKER)
    api=make(on_step=explode)
    with TestClient(create_app(service=api),raise_server_exceptions=False) as client:
        episode=client.post('/episodes',json=request().model_dump(mode='json')).json()['episode_id']
        session=api.get(episode)
        client.post(f'/episodes/{episode}/actions',json=action(session));join_action(session)
        assert session.result.outcome=='infrastructure_error'
        for endpoint in ['observation','result','actions/move_1']:
            response=client.get(f'/episodes/{episode}/{endpoint}')
            assert PRIVATE_MARKER not in response.text
        assert PRIVATE_MARKER in (session.directory/'private/action_error.json').read_text()


def test_native_transport_timeout_is_terminal_for_the_rpc_stream(tmp_path):
    import socket
    from bench.runtime.native import NativeWorker
    from bench.runtime.transport import receive
    client,peer=socket.socketpair();client.settimeout(.02)
    worker=NativeWorker.__new__(NativeWorker)
    worker.connection=client;worker.lock=threading.RLock();worker.closed=False;worker.broken=False
    _native_fatal_fixture(worker,tmp_path)
    try:
        with pytest.raises(TimeoutError):worker.request('step',command={})
        assert receive(peer)['operation']=='step'
        assert worker.broken
        with pytest.raises(RuntimeError,match='transport failed'):worker.request('cameras')
    finally:
        client.close();peer.close()


def test_native_null_deadline_disables_socket_timeout():
    import socket
    from bench.runtime.native import NativeWorker
    client,peer=socket.socketpair()
    worker=NativeWorker.__new__(NativeWorker)
    worker.connection=client;worker.lock=threading.RLock()
    try:
        worker.set_deadline(10.)
        assert client.gettimeout()==10.
        worker.set_deadline(None)
        assert client.gettimeout() is None
        worker.set_deadline(300.)
        assert client.gettimeout()==120.
    finally:
        client.close();peer.close()


@pytest.mark.parametrize('startup_timeout',[None,240.])
def test_native_startup_can_wait_without_a_time_horizon(tmp_path,monkeypatch,startup_timeout):
    from unittest.mock import MagicMock
    from bench.runtime import native
    temporary=tmp_path/'socket_dir';temporary.mkdir()
    clock=[0.];sleeps=[0]
    def wait_for_worker(_):
        clock[0]+=1000.
        sleeps[0]+=1
        if sleeps[0]==2:(temporary/'worker.sock').touch()
    process=MagicMock();process.poll.return_value=None
    connection=MagicMock()
    monkeypatch.setattr(native.tempfile,'mkdtemp',lambda **kwargs:str(temporary))
    monkeypatch.setattr(native.subprocess,'Popen',lambda *args,**kwargs:process)
    monkeypatch.setattr(native.socket,'socket',lambda *args,**kwargs:connection)
    monkeypatch.setattr(native.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(native.time,'sleep',wait_for_worker)
    monkeypatch.setattr(native,'receive',lambda connection, *, wait_for_data=None:{'ok':True})
    monkeypatch.setattr(native,'execution_environment',lambda device:{'capture_status':'recorded'})
    arguments=dict(output=tmp_path/'output',task_id='test',hand_id='test',seed=0,
                   release=tmp_path,device='cuda:0',startup_timeout_s=startup_timeout)
    if startup_timeout is not None:
        with pytest.raises(TimeoutError,match='startup timed out'):native.NativeWorker(**arguments)
        return
    worker=native.NativeWorker(**arguments)
    try:
        assert sleeps[0]==2 and clock[0]==2000.
        assert all(call.args==(None,) for call in connection.settimeout.call_args_list)
        connection.connect.assert_called_once_with(str(temporary/'worker.sock'))
    finally:
        worker.close(force=True)


def test_malformed_native_frame_poisoning_prevents_reusing_rpc_stream(tmp_path):
    import socket,struct
    from bench.runtime.native import NativeWorker
    from bench.runtime.transport import MAX_MESSAGE_BYTES
    client,peer=socket.socketpair();client.settimeout(.02)
    worker=NativeWorker.__new__(NativeWorker)
    worker.connection=client;worker.lock=threading.RLock();worker.closed=False;worker.broken=False
    _native_fatal_fixture(worker,tmp_path)
    try:
        peer.sendall(struct.pack('!I',MAX_MESSAGE_BYTES+1))
        with pytest.raises(ValueError,match='transport limit'):worker.request('cameras')
        assert worker.broken
        with pytest.raises(RuntimeError,match='transport failed'):worker.request('cameras')
    finally:
        client.close();peer.close()


def plan_for(make, *, split='development'):
    from bench.protocol.schemas import EvaluationPlan
    baseline,_=session_for(make())
    spec=baseline.spec.model_copy(deep=True,update={'episode_id':'operator_slot_1','split':split})
    baseline.finish()
    return EvaluationPlan(plan_id='operator_plan',episodes=[spec])


def test_plan_slot_id_rejected_without_registered_operator_plan(runner):
    make,_,workers=runner
    with pytest.raises(ServiceError) as error:
        make().create(request(plan_slot_id='operator_slot_1'))
    assert error.value.status==422
    assert not workers


@pytest.mark.parametrize('split',['development','held_out'])
def test_valid_operator_plan_runs_exact_pinned_spec_and_forbids_rerun(runner,split):
    make,_,_=runner
    plan=plan_for(make,split=split)
    api=make(evaluation_plan=plan)
    session,reply=session_for(api,plan_slot_id='operator_slot_1')
    assert session.spec==plan.episodes[0]
    assert reply['episode_id']=='operator_slot_1'
    session.finish()
    with pytest.raises(ServiceError) as error:
        session_for(api,plan_slot_id='operator_slot_1')
    assert error.value.status==409


def test_plan_rejects_unknown_slot_before_native_worker_creation(runner):
    make,_,workers=runner
    plan=plan_for(make)
    api=make(evaluation_plan=plan)
    before=len(workers)
    with pytest.raises(ServiceError) as error:
        session_for(api,plan_slot_id='not_registered')
    assert error.value.status==422
    assert len(workers)==before


def test_mismatched_plan_request_does_not_poison_correct_retry(runner):
    make,_,_=runner
    plan=plan_for(make)
    api=make(evaluation_plan=plan)
    with pytest.raises(ServiceError) as error:
        session_for(api,plan_slot_id='operator_slot_1',seed=plan.episodes[0].seed+1)
    assert error.value.status==409
    session,_=session_for(api,plan_slot_id='operator_slot_1')
    assert session.spec==plan.episodes[0]


def test_runtime_asset_mismatch_against_plan_cannot_start_episode(runner):
    make,_,workers=runner
    plan=plan_for(make)
    plan.episodes[0].robot_asset_sha256='e'*64
    api=make(evaluation_plan=plan)
    with pytest.raises(ServiceError) as error:
        session_for(api,plan_slot_id='operator_slot_1')
    assert error.value.status==409
    assert not api.sessions
    assert workers[-1].closed_force


def test_registered_plan_is_not_mutated_by_callers_after_creation(runner):
    make,_,_=runner
    plan=plan_for(make)
    pinned=plan.episodes[0].model_copy(deep=True)
    api=make(evaluation_plan=plan)
    plan.episodes[0].seed+=1
    session,_=session_for(api,plan_slot_id='operator_slot_1')
    assert session.spec==pinned


def test_finished_episode_keeps_policy_owned_until_old_execution_thread_exits(runner):
    """A late cancel from an old policy call must never reach a reused new episode."""
    make,clock,workers=runner
    teacher=Teacher(terminal='running',block=True)
    registry=PolicyRegistry();registry.register(teacher)
    api=make(registry=registry)
    session,_=session_for(api,track='system0',policy_id='test_teacher',policy_version='v1',
        budgets=Budgets(wall_time_s=1.))
    session.submit(skill_action(session));assert teacher.started.wait(timeout=1)
    clock.value=1.1
    assert session.done.wait(timeout=1)
    assert session.result.outcome=='wall_timeout'
    assert session.thread.is_alive()
    hashes_path=session.directory/'artifact_hashes.json'
    finalized_hashes=load_json(hashes_path)
    manifest_digest=hashlib.sha256(hashes_path.read_bytes()).hexdigest()
    before=len(workers)
    try:
        with pytest.raises(ServiceError) as error:
            session_for(api,track='system0',policy_id='test_teacher',policy_version='v1')
        assert error.value.status==409
        assert len(workers)==before
    finally:
        teacher.release.set()
        join_action(session)
    assert teacher.cancel_reasons
    assert hashlib.sha256(hashes_path.read_bytes()).hexdigest()==manifest_digest
    for relative,expected in finalized_hashes.items():
        assert hashlib.sha256((session.directory/relative).read_bytes()).hexdigest()==expected, relative
    replacement,_=session_for(api,track='system0',policy_id='test_teacher',policy_version='v1')
    assert replacement is not session
    assert replacement.result is None


def _native_fatal_fixture(worker,tmp_path):
    # __new__ RPC tests need the real constructor's empty per-worker log state.
    # Keep the fatal detector active and every original timeout assertion intact.
    from unittest.mock import MagicMock
    worker.output=tmp_path;(tmp_path/'worker.log').write_bytes(b'')
    worker._fatal_log_offset=0;worker._fatal_log_tail=b''
    worker._native_failure=None;worker._fatal_exception=None
    worker.process=MagicMock();worker.process.poll.return_value=None


def publish_targets(worker):
    # A native worker reports its acknowledged drive targets in robot state.
    worker.raw.setdefault('robot', {})['joint_targets_rad'] = dict(worker.raw['joint_positions_rad'])


def test_direct_target_ramps_from_acknowledged_drive_with_exact_endpoint(runner):
    make,_,workers=runner
    session,reply=session_for(make(on_step=publish_targets,acknowledged_targets=True))
    assert reply['capabilities']['direct_joint_execution']=='min_jerk_velocity_preflight_v2'
    worker=workers[-1]
    publish_targets(worker); session.raw=deepcopy(worker.raw)
    session.submit(action(session,duration=.4));join_action(session)
    steps=[payload['command']['joint_positions_rad'] for op,payload in worker.calls if op=='step'][-20:]
    first=[steps[i]['fr3_joint1'] for i in range(20)]
    assert 0<first[0]<.1 and all(b>=a for a,b in zip(first,first[1:]))
    # One common time scaling: the 0.2 rad hand move at 1 rad/s needs 19 of 20 steps.
    assert first[17]<.1 and all(value==.1 for value in first[18:])
    assert steps[-1]['active_finger_joint_0']==.2


def test_direct_target_without_acknowledged_drive_keeps_step_semantics(runner):
    make,_,workers=runner
    session,reply=session_for(make())
    assert reply['capabilities']['direct_joint_execution']=='step_target_v1'
    session.submit(action(session,duration=.1));join_action(session)
    steps=[payload['command']['joint_positions_rad']['fr3_joint1'] for op,payload in workers[-1].calls if op=='step'][-5:]
    assert steps==[.1]*5


@pytest.mark.parametrize('arm,hand',[(.05,0.),(0.,.1)])
def test_client_limited_moves_at_minimum_duration_respect_speed_caps(runner,arm,hand):
    make,_,workers=runner
    session,_=session_for(make(on_step=publish_targets,acknowledged_targets=True))
    worker=workers[-1]
    publish_targets(worker); session.raw=deepcopy(worker.raw)
    request=action(session,duration=.2)
    request['arm_joint_positions_rad']=dict.fromkeys(session.capabilities.arm_joints,arm)
    request['hand_joint_positions_rad']=dict.fromkeys(session.capabilities.hand_joints,hand)
    session.submit(request);join_action(session)
    steps=[payload['command']['joint_positions_rad'] for op,payload in worker.calls if op=='step'][-10:]
    name,cap=('fr3_joint1',.65) if arm else ('active_finger_joint_0',1.)
    values=[0.]+[row[name] for row in steps]
    assert max(abs(b-a) for a,b in zip(values,values[1:]))<=cap*.02+1e-12
    assert values[-1]==(arm or hand)


def test_evaluator_invalidation_ends_the_episode_as_evaluator_invalid(runner):
    make,_,_=runner
    def invalid(worker):
        worker.score['reason']='invalid_observation:native contact sampling does not cover every physics substep'
    session,_=session_for(make(on_step=invalid))
    session.submit(action(session,duration=.1));join_action(session)
    assert session.result.outcome=='evaluator_invalid' and session.result.physics_valid is False
    assert (session.directory/'private/evaluator_invalid.json').is_file()


def test_violent_object_motion_invalidation_is_a_scored_safety_stop(runner):
    make,_,_=runner
    def violent(worker):
        worker.score['reason']='invalid_observation:object trajectory exceeds frozen motion speed bound'
    session,_=session_for(make(on_step=violent))
    session.submit(action(session,duration=.1));join_action(session)
    assert session.result.outcome=='safety_stop'


def test_public_velocity_is_the_control_period_mean(runner):
    make,_,workers=runner
    def chatter(worker):
        worker.raw['joint_velocities_rad_s']=dict.fromkeys(worker.raw['joint_velocities_rad_s'],13.8)
    session,_=session_for(make(on_step=chatter))
    session.submit(action(session,duration=.02));join_action(session)
    velocities=session.observe()['proprioception']['joint_velocities_rad_s']
    assert velocities['fr3_joint1']==pytest.approx(.1/.02)
    assert velocities['fr3_joint1']!=13.8


def test_irrecoverable_evaluator_state_ends_the_episode_as_a_valid_task_failure(runner):
    make,_,_=runner
    def latched(worker):
        worker.score['irrecoverable_reason']='episode_displacement_limit_exceeded'
    session,_=session_for(make(on_step=latched))
    session.submit(action(session,duration=.1));join_action(session)
    assert session.result.outcome=='task_failure' and session.result.physics_valid is True
