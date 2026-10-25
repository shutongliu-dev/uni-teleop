"""Generic turn HTTP contracts with real controllers and an explicit CPU double.

Synthetic referee replies test transport and ownership only, never task physics.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim import release as release_api
from bench.paths import DATA_ROOT
from bench.protocol.schemas import RobotJointCommand
from bench.runtime import controllers, service
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from bench.simulation.isaacsim_backend.robot_registry import get_robot_spec
from test_agent_service import FakeWorker, PRIVATE_MARKER, frozen_inputs, join_action


class FakeTurnWorker(FakeWorker):
    """The test alone may supply a turn transition; public calls cannot."""
    def __init__(self, *, scene, task_id, initial, hand_id, **kwargs):
        super().__init__(hand_id='sharpa_w01_right', acknowledged_targets=True, depth=True, **kwargs)
        assert hand_id == 'fr3_sharpa_instances'
        model = public_robot_geometry(robot_configuration(scene))
        self.arm, self.active = list(model.arm_names), list(model.names)
        self.hand = [n for n in self.active if n not in self.arm]
        self.raw['joint_positions_rad'] = dict(initial['joint_positions_rad'])
        self.raw['joint_velocities_rad_s'] = dict.fromkeys(self.active, 0.)
        self.raw['robot'] = {'joint_targets_rad': dict(initial['joint_positions_rad'])}
        self.initial['metadata'].update(scene_spec=scene, hand_spec=get_robot_spec(scene['robot']), action_spec={
            'joint_names':self.active, 'arm_joint_names':self.arm, 'hand_joint_names':self.hand,
            'joint_limits_rad':{j.name:[j.lower,j.upper] for j in model.joints}})
        self.state = {'game_id':task_id, 'status':'active', 'turn_index':0, 'current_player_id':'alpha',
            'players':['alpha','beta'], 'winner_player_id':None, 'loser_player_id':None,
            'terminal_reason':None, 'violation_reason':None, 'sim_time_s':0.,
            'public_rules':{'intents':['select']}, 'public_progress':{}}
        self.switch_next = False
        self.corrupt_time = False
        self.peer_drift = 0.
        self.sync_score()
        self.initial.update(observation=deepcopy(self.raw), score=deepcopy(self.score), turn_game=deepcopy(self.state))

    def sync_score(self):
        self.score = {'success':self.state['status']=='completed', **{name:self.state[name] for name in
            ('status','turn_index','winner_player_id','loser_player_id')}, 'private_detail':PRIVATE_MARKER}

    def request(self, operation, **payload):
        if operation == 'turn_game':
            self.calls.append((operation, deepcopy(payload)))
            command = payload['command']
            error = None
            if command['type'] == 'end_episode':
                self.state.update(status='interrupted', current_player_id=None, terminal_reason=command['reason'])
            elif command['name'] == 'select' and set(command['arguments']) == {'slot'}:
                self.state['public_progress'] = {'selected': command['arguments']['slot']}
            else:
                error = {'code':'invalid_intent', 'message':'Intent is not allowed by the referee'}
            self.sync_score()
            return {'turn_game':deepcopy(self.state), 'score':deepcopy(self.score),
                    **({'turn_game_error':error} if error else {})}
        if operation == 'step':
            actor = payload['actor']
            assert actor == {'player_id':self.state['current_player_id'], 'turn_index':self.state['turn_index']}
            owner = 'front' if actor['player_id']=='alpha' else 'rear'
            for name, value in payload['command']['joint_positions_rad'].items():
                if not name.startswith((owner+'_', 'fr3_'+owner+'_')):
                    assert value == self.raw['robot']['joint_targets_rad'][name]
            reply = super().request(operation, **payload)
            self.raw['robot']['joint_targets_rad'] = dict(payload['command']['joint_positions_rad'])
            peer = 'rear' if owner=='front' else 'front'
            self.raw['joint_positions_rad']['fr3_'+peer+'_joint1'] += self.peer_drift
            changed = self.switch_next
            self.switch_next = False
            if changed:
                self.state['turn_index'] += 1
                self.state['current_player_id'] = 'beta' if actor['player_id']=='alpha' else 'alpha'
            self.state['sim_time_s'] = self.raw['sim_time_s']
            self.sync_score()
            state = deepcopy(self.state)
            if self.corrupt_time: state['sim_time_s'] += 1.
            reply.update(observation=deepcopy(self.raw), score=deepcopy(self.score), turn_game=state,
                         turn_game_turn_changed=changed)
            return reply
        return super().request(operation, **payload)


@pytest.fixture
def game(tmp_path, monkeypatch, frozen_inputs):
    manifest, inputs = frozen_inputs
    manifest = deepcopy(manifest)
    manifest['task_kind'] = 'turn_game'
    root = tmp_path/'release'
    root.mkdir()
    tasks = {}
    for index, task_id in enumerate(('first', 'second')):
        entry, scene, task, _ = deepcopy(inputs)
        scene.pop('scenario', None)
        scene['robot'] = {'assets_root':f'{DATA_ROOT}/assets', 'hand_id':'fr3_sharpa_instances',
            'assembly_version':'namespaced_v1', 'base_position_m':[index*.1,0.,0.],
            'base_quaternion_wxyz':[1.,0.,0.,0.], 'instances':{
                side:{'hand_id':'sharpa_w01_right','assembly_version':'isaacsim_6_1_v2',
                      'base_position_m':[.65,y,.78],'base_quaternion_wxyz':[1.,0.,0.,0.]}
                for side,y in [('front',-.76),('rear',.76)]}}
        scene['turn_game_task'] = {'players':[{'player_id':'alpha','robot_id':'front'},
                                               {'player_id':'beta','robot_id':'rear'}]}
        model = public_robot_geometry(robot_configuration(scene))
        q = {j.name:float(np.clip(0.,j.lower,j.upper)) for j in model.joints}
        for member in model.sides.values():
            q.update(zip(member.arm.names,[.2,-.5,0.,-2.3,0.,1.8,.8]))
        initial = {'joint_positions_rad':q}
        entry.update(id=task_id, task_kind='turn_game', task=task_id+'.json', scene=task_id+'_scene.json')
        task.update(task_id=task_id,public_rules={'intents':['select']})
        tasks[task_id] = (entry,scene,task,initial)
        (root/entry['task']).write_text(json.dumps(task))
        (root/entry['scene']).write_text(json.dumps(scene))
    manifest['tasks'] = [row[0] for row in tasks.values()]
    (root/'release_manifest.json').write_text(json.dumps(manifest))
    read = lambda release, task_id, **kwargs: deepcopy(tasks[task_id])
    validate = lambda release, **kwargs: deepcopy(manifest)
    for module in (service, controllers, release_api):
        monkeypatch.setattr(module,'validate',validate)
    for module in (service, controllers):
        monkeypatch.setattr(module,'task_inputs',read)

    @contextmanager
    def start(*, task_id='first', track='system0', mixed=False):
        if mixed: manifest.pop('task_kind',None)
        workers=[]
        def worker(**kwargs):
            _,scene,_,initial = tasks[kwargs['task_id']]
            result=FakeTurnWorker(scene=deepcopy(scene),initial=initial,**kwargs)
            workers.append(result)
            return result
        api=AgentService(tmp_path/('run'+str(len(list(tmp_path.glob('run*'))))), release=root,
                         registry=controllers.default_registry(root),worker_factory=worker)
        payload={'task_id':task_id,'agent_id':'test','agent_version':'v1','agent_config_sha256':'b'*64,'track':track}
        if track=='direct_joints': payload.update(policy_id=None,policy_version=None)
        with TestClient(create_app(service=api)) as client:
            response=client.post('/turn-games',json=payload)
            assert response.status_code==201,response.text
            created=response.json()
            session=api.sessions[created['episode_id']]
            tokens={p['binding']['player_id']:p['session_token'] for p in created['players']}
            yield SimpleNamespace(client=client,api=api,session=session,created=created,tokens=tokens,
                                  worker=workers[-1],workers=workers,payload=payload,tasks=tasks)
    return start


def endpoint(g,player='alpha',kind='turn-games'):
    return f'/{kind}/{g.session.spec.episode_id}/players/{player}'


def headers(g,player='alpha'):
    return {'Authorization':'Bearer '+g.tokens[player]}


def call(g,name,args=None,player='alpha'):
    response=g.client.post(endpoint(g,player)+'/call',headers=headers(g,player),
        json={'function':name,'arguments':args or {},'request_id':'test'})
    assert PRIVATE_MARKER not in response.text
    return response


def steps(g):
    return [payload for operation,payload in g.worker.calls if operation=='step']


def test_authentication_is_player_and_entrypoint_scoped_before_any_accounting(game):
    with game() as g:
        before=g.session.clock.snapshot()
        for suffix in ('state','functions','result'):
            assert g.client.get(endpoint(g)+'/{}'.format(suffix)).status_code==403
            assert g.client.get(endpoint(g,'beta')+'/'+suffix,headers=headers(g)).status_code==403
            assert g.client.get(endpoint(g,kind='board-games')+'/'+suffix,headers=headers(g)).status_code==404
        for suffix in ('call','intent'):
            assert g.client.post(endpoint(g)+'/'+suffix,headers={'Authorization':'Bearer invalid'},json={}).status_code==403
        assert g.client.get('/episodes/'+g.session.spec.episode_id+'/functions').status_code==403
        assert g.client.post('/episodes/'+g.session.spec.episode_id+'/call',json={}).status_code==403
        assert g.session.clock.snapshot()['action_attempts']==before['action_attempts']
        assert not steps(g)
        guide=call(g,'get_robot')
        assert guide.status_code==200,guide.text
        assert 'front_hand_flange' in guide.text and 'rear_hand_flange' in guide.text


def test_selected_task_entry_controls_route_factory_and_geometry(game):
    with game(task_id='second',mixed=True) as g:
        assert g.session.spec.task_id=='second'
        assert g.session.policy.robot_configuration==robot_configuration(g.tasks['second'][1])
        assert g.session.robot_geometry.base[0,3]==.1
        rows=g.client.get('/tasks').json()
        assert all(row['create_endpoint']=='POST /turn-games' for row in rows)
        assert '/turn-games/' in g.client.get('/interfaces',params={'task_id':'second'}).text


def test_wrong_robot_unknown_task_and_unscoped_create_reject_before_worker(game):
    with game(mixed=True) as g:
        response=call(g,'finish_episode')
        assert response.status_code==200,response.text
        for change in ({'robot_id':'dual_fr3_sharpa_w01'}, {'task_id':'missing'}):
            response=g.client.post('/turn-games',json={**g.payload,**change})
            assert response.status_code==422,response.text
        response=g.client.post('/episodes',json={**g.payload,'robot_id':'fr3_sharpa_instances',
            'policy_id':'classical_fingertip_instances','policy_version':'1.2.0-board1'})
        assert response.status_code==403,response.text
        assert len(g.workers)==1


def test_namespaced_ik_and_camera_are_read_only_public_functions(game):
    with game() as g:
        for side in ('front','rear'):
            model=g.session.robot_geometry.sides[side]
            q={name:g.worker.raw['joint_positions_rad'][name] for name in model.names}
            wrist=model.public_geometry(q)['wrist']
            response=call(g,'query_ik',{'observation_sequence':0,'hand_geometries':{
                side:{'wrist':wrist,'fingertip_frame_id':side+'_hand_flange'}}})
            assert response.status_code==200,response.text
            assert response.json()['result']['solution_found']
            assert response.json()['result']['advances_physics'] is False
        response=call(g,'query_ik',{'observation_sequence':0,'hand_geometries':{
            'unknown':{'wrist':wrist,'fingertip_frame_id':'unknown_hand_flange'}}})
        assert response.status_code==422,response.text
        response=call(g,'read_camera',{'observation_sequence':0,'camera_id':'overview'})
        assert response.status_code==200,response.text
        assert not steps(g) and g.session.clock.action_attempts==0


@pytest.mark.parametrize('function,args',[
    ('move_wrist',{'hand':'rear','wrist':{'frame_id':'world','position_m':[.6,0,1.3],'quaternion_wxyz':[1,0,0,0]},'timeout_sim_s':1.}),
    ('set_hand_targets',{'hand':'front','hand_posture_rad':{'rear_thumb_IP':.1},'timeout_sim_s':1.}),
    ('move_geometry',{'hand_geometries':{'rear':{'hand_posture_rad':{'rear_thumb_IP':.1},'fingertip_frame_id':'rear_hand_flange'}},'timeout_sim_s':1.}),
    ('move_joints',{'arm_joint_positions_rad':{'fr3_rear_joint1':.2},'hand_joint_positions_rad':{'front_thumb_IP':.1},'duration_s':.1}),
])
def test_other_instance_commands_are_rejected_before_native_admission(game,function,args):
    with game() as g:
        response=call(g,function,dict(args,action_id='foreign',observation_sequence=0))
        assert response.status_code==403,response.text
        assert not steps(g) and g.session.turn_game.last_targets==g.worker.raw['robot']['joint_targets_rad']


def test_owned_motion_holds_inactive_ack_even_with_measured_peer_drift(game):
    with game() as g:
        before=dict(g.session.turn_game.last_targets)
        g.worker.peer_drift=.01
        model=g.session.robot_geometry.sides['front']
        q={name:g.worker.raw['joint_positions_rad'][name] for name in model.names}
        q[model.arm.names[0]]+=.025
        target=model.public_geometry(q)['wrist']
        response=call(g,'move_wrist',{'hand':'front','wrist':target,'timeout_sim_s':3.,'action_id':'owned','observation_sequence':0})
        assert response.status_code in (200,202),response.text
        join_action(g.session)
        result=call(g,'get_action',{'action_id':'owned'})
        assert result.json()['result']['status']=='completed',result.text
        assert steps(g)
        for step in steps(g):
            assert step['actor']=={'player_id':'alpha','turn_index':0}
            for name in g.session.robot_geometry.sides['rear'].names:
                assert step['command']['joint_positions_rad'][name]==before[name]


def test_owned_partial_direct_targets_expand_from_ack_and_retry_once(game):
    with game(track='direct_joints') as g:
        before=deepcopy(g.session.turn_game.last_targets)
        target=before['fr3_front_joint1']+.002
        payload={'action_id':'partial','observation_sequence':0,'duration_s':.2,
                 'arm_joint_positions_rad':{'fr3_front_joint1':target},
                 'hand_joint_positions_rad':{'front_thumb_IP':before['front_thumb_IP']}}
        response=call(g,'move_joints',payload)
        assert response.status_code in (200,202),response.text
        join_action(g.session)
        assert call(g,'get_action',{'action_id':'partial'}).json()['result']['status']=='completed'
        commands=deepcopy(steps(g))
        assert commands and commands[-1]['command']['joint_positions_rad']['fr3_front_joint1']==target
        for step in commands:
            for name,value in before.items():
                if name!='fr3_front_joint1':
                    assert step['command']['joint_positions_rad'][name]==value
        assert call(g,'move_joints',payload).status_code==200
        assert steps(g)==commands and g.session.clock.action_attempts==1


def test_owned_namespaced_hand_targets_complete_with_original_controller(game):
    with game() as g:
        target={name:value for name,value in g.session.turn_game.last_targets.items() if name.startswith('front_')}
        target['front_thumb_CMC_AA']+=.02
        response=call(g,'set_hand_targets',{'hand':'front','hand_posture_rad':target,
            'timeout_sim_s':1.,'action_id':'hand','observation_sequence':0})
        assert response.status_code in (200,202),response.text
        join_action(g.session)
        result=call(g,'get_action',{'action_id':'hand'})
        assert result.json()['result']['status']=='completed',result.text
        assert steps(g)


def test_native_turn_handoff_stops_old_action_and_allows_only_unchanged_retry(game):
    with game() as g:
        g.worker.switch_next=True
        payload={'action_id':'handoff','observation_sequence':0,'duration_s':.2}
        response=call(g,'hold',payload)
        assert response.status_code in (200,202),response.text
        join_action(g.session)
        assert len(steps(g))==1 and g.session.turn_game.state.current_player=='beta'
        assert call(g,'hold',payload).status_code==200
        assert len(steps(g))==1
        changed={**payload,'duration_s':.4}
        assert call(g,'hold',changed).status_code==403
        assert call(g,'hold',{'action_id':'new','observation_sequence':1,'duration_s':.02}).status_code==403
        assert call(g,'get_action',{'action_id':'handoff'},player='beta').status_code==403
        response=call(g,'hold',{'action_id':'beta','observation_sequence':1,'duration_s':.02},player='beta')
        assert response.status_code in (200,202),response.text
        join_action(g.session)
        assert len(steps(g))==2 and steps(g)[-1]['actor']=={'player_id':'beta','turn_index':1}


def test_explicit_fixed_hold_http_drift_ack_dedup_budget_and_cancel(game):
    with game() as g:
        targets = dict(g.session.turn_game.last_targets)
        # Both owned and inactive measurements differ from the real drive ACK.
        for raw in (g.session.raw,g.worker.raw):
            raw['joint_positions_rad']['fr3_front_joint1'] -= .0001
            raw['joint_positions_rad']['fr3_rear_joint1'] += .0002
        g.worker.peer_drift = .0002
        payload = {'action_id':'fixed','observation_sequence':0,'duration_s':.06,
                   'mode':'acknowledged_drive'}
        response = call(g,'hold',payload)
        assert response.status_code in (200,202),response.text
        join_action(g.session)
        result = call(g,'get_action',{'action_id':'fixed'}).json()['result']
        assert result['status'] == 'completed',result
        assert len(steps(g)) == 3
        assert all(row['command']['joint_positions_rad'] == targets for row in steps(g))
        assert g.session.clock.action_attempts == 1
        assert call(g,'hold',payload).status_code == 200
        assert g.session.clock.action_attempts == 1 and len(steps(g)) == 3
        changed = call(g,'hold',{**payload,'mode':'reference'})
        assert changed.status_code == 409 and changed.json()['error']['code'] == 'action_conflict'
        assert g.session.clock.action_attempts == 2 and len(steps(g)) == 3
        cancelled = call(g,'cancel_action',{'action_id':'fixed'})
        assert cancelled.status_code == 200 and len(steps(g)) == 3


def test_intent_cannot_invoke_private_worker_operations_or_mutate_objects(game):
    with game() as g:
        before=deepcopy(g.worker.raw)
        for payload in ({'type':'end_episode','reason':'won'}, {'type':'step','command':{}},
                        {'name':'select','arguments':{'slot':'x'},'objects':{}},
                        {'name':'select','arguments':{'slot':'x'*5000}}):
            response=g.client.post(endpoint(g)+'/intent',headers=headers(g),json=payload)
            assert response.status_code==422,response.text
        assert not [op for op,_ in g.worker.calls if op=='turn_game']
        payload={'name':'set_object_pose','arguments':{'object_id':'private','position_m':[1,2,3]}}
        response=g.client.post(endpoint(g)+'/intent',headers=headers(g),json=payload)
        assert response.status_code==422,response.text
        assert response.json()['error']['code']=='invalid_arguments'
        assert g.worker.raw==before and not steps(g)
        assert g.session.turn_game.state.turn_index==0
        events=[json.loads(line) for line in (g.session.directory/'private/turn_game_calls.jsonl').read_text().splitlines()]
        assert events[-1]['error']['code']=='invalid_intent'
        assert g.client.post(endpoint(g,'beta')+'/intent',headers=headers(g,'beta'),json={'name':'select'}).status_code==403


@pytest.mark.parametrize('via_command',[False,True])
def test_mismatched_referee_clock_is_rejected_atomically(game,via_command):
    with game() as g:
        facade=g.session.turn_game
        state=facade.state.model_dump(mode='json')
        state['sim_time_s']+=1.
        before=facade.state.model_dump(mode='json'),deepcopy(facade.last_targets),deepcopy(g.session.score)
        reply={'turn_game':state,'score':deepcopy(g.session.score),'turn_game_turn_changed':False,
               'observation':{'sim_time_s':.02}}
        command=RobotJointCommand(duration_s=.02,joint_positions_rad=facade.last_targets) if via_command else None
        with pytest.raises(RuntimeError,match='clock'):
            facade.acknowledge(reply,command)
        assert (facade.state.model_dump(mode='json'),facade.last_targets,g.session.score)==before


@pytest.mark.parametrize('robot_time',[float('nan'),float('inf'),True,'0.0'])
def test_non_numeric_or_non_finite_robot_clock_cannot_acknowledge_turn(game,robot_time):
    with game() as g:
        facade=g.session.turn_game
        before=facade.state.model_dump(mode='json'),deepcopy(facade.last_targets),deepcopy(g.session.score)
        reply={'turn_game':deepcopy(before[0]),'score':deepcopy(g.session.score),
               'turn_game_turn_changed':False,'observation':{'sim_time_s':robot_time}}
        command=RobotJointCommand(duration_s=.02,joint_positions_rad=facade.last_targets)
        with pytest.raises(RuntimeError,match='clock'):
            facade.acknowledge(reply,command)
        assert (facade.state.model_dump(mode='json'),facade.last_targets,g.session.score)==before
