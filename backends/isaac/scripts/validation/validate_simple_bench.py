"""Run complete simple-task diagnostics through the real Agent HTTP boundary.

The scripted controller is an untrained, privileged development baseline. No
native environment handle or evaluator output is supplied to the controller.
All manipulation is through the same named-joint command interface as entrants.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import time

from fastapi.testclient import TestClient

from bench.paths import DATA_ROOT, load_json
from bench.protocol import PolicyRegistry, RobotJointCommand
from bench.protocol.metrics import aggregate_results
from bench.protocol.policies import PolicyStep
from bench.protocol.schemas import PolicyDescriptor, RobotCapabilities
from bench.isaacsim.release import digest, dump, task_inputs, validate
from scenes.manipulation import DEFAULT_TASK_RELEASE
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from uni_teleop.backends.isaac.manus_actions import direct_action

PROJECT = Path(__file__).resolve().parents[2]
ASSETS = Path(f'{DATA_ROOT}/assets')


def controller_observation(observation):
    """Project exactly the supported typed teacher view; never read raw state."""
    if hasattr(observation, 'model_dump'):
        observation = observation.model_dump(mode='json', exclude_none=True)
    return {'time_s': observation['sim_time_s'], 'sim_time_s': observation['sim_time_s'],
            **observation['proprioception'],
            'objects': observation.get('privileged_object_poses', {})}


def implementation_sources(mode='scripted'):
    paths = [Path(__file__)]
    if mode == 'scripted':
        paths += [PROJECT/'scenes/manipulation/controller.py', PROJECT/'bench/isaacsim/controllers.py',
                  PROJECT/'bench/control/__init__.py', PROJECT/'bench/control/kinematics.py', PROJECT/'bench/geometry.py']
    return {str(p.relative_to(PROJECT)): digest(p) for p in paths}


def controller_asset_pins():
    return {str(ASSETS/p):digest(ASSETS/p) for p in (
        'robots/franka/fr3/fr3_no_hand.absolute.urdf',
        'robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf')}


def public_controller_task(task):
    """Keep evaluator configuration outside both controller entry points."""
    goal = task['public_goal']
    return {'id':task['id'], 'object_id':goal['object_id'],
            'target_position_m':list(goal['target_position_m'])}


class ScriptedDiagnostic:
    def __init__(self, scene, task, trace_path, controller_class, source_pins, asset_pins):
        self.scene, self.task, self.trace_path = scene, public_controller_task(task), trace_path
        self.controller_class = controller_class
        sources = {'code':source_pins, 'kinematics_assets':asset_pins,
                   'libraries':{name:importlib.metadata.version(name) for name in ('numpy','scipy')}}
        self.descriptor = PolicyDescriptor(policy_id='simple_scripted_diagnostic', version='v1',
            implementation_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            trained=False, observation_profile='privileged_state', skills=['execute_simple_task'])

    def reset(self, spec, capabilities):
        self.controller = self.controller_class(self.scene, self.task)
        self.capabilities = capabilities
        self.last = None

    def begin(self, action, observation):
        if action.skill != 'execute_simple_task':
            raise ValueError('Unsupported scripted diagnostic skill')

    def step(self, observation):
        raw = controller_observation(observation)
        command = self.controller.next_action(raw)
        self.last = command
        diagnostic = self.controller.diagnostic()
        with self.trace_path.open('a') as stream:
            stream.write(json.dumps({'sim_time_s': observation.sim_time_s,
                'cameras': [c.model_dump(mode='json') for c in observation.cameras],
                'diagnostic': diagnostic}, allow_nan=False)+'\n')
        return PolicyStep(status='running', command=RobotJointCommand(**command))

    def cancel(self, reason):
        pass


def checked(response, expected=200):
    if response.status_code != expected:
        raise RuntimeError(f'HTTP {response.status_code}: {response.text}')
    return response.json()


def check_images(client, observation):
    for camera in observation['cameras']:
        response = client.get(camera['rgb_uri'])
        if response.status_code != 200 or hashlib.sha256(response.content).hexdigest() != camera['rgb_sha256']:
            raise RuntimeError('Public camera bytes do not match advertised digest')


def record_observation(path, observation, diagnostic=None):
    with path.open('a') as stream:
        stream.write(json.dumps({'sim_time_s': observation['sim_time_s'],
            'cameras': observation['cameras'], 'diagnostic': diagnostic or {}}, allow_nan=False)+'\n')


def run_episode(args, task_id, mode, seed):
    output = args.output/f'{task_id}_{mode}_seed{seed}'
    output.mkdir(parents=True, exist_ok=False)
    _, scene, task, _ = task_inputs(args.release, task_id)
    registry = PolicyRegistry()
    trace = output/'camera_timeline.jsonl'
    policy = None
    track = 'direct_joints' if mode == 'idle' else args.track
    source_pins = implementation_sources(mode)
    asset_pins = controller_asset_pins() if mode == 'scripted' else {}
    for relative, expected in source_pins.items():
        destination = output/'controller_source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT/relative, destination)
        if digest(destination) != expected:
            raise RuntimeError('Controller source changed during setup')
    dump(output/'controller_source_hashes.json', source_pins)
    for original,expected in asset_pins.items():
        destination = output/'controller_assets'/Path(original).relative_to(ASSETS)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original,destination)
        assert digest(destination)==expected
    dump(output/'controller_asset_hashes.json',asset_pins)
    controller_class = None
    if mode == 'scripted':
        path = output/'controller_source/scenes/manipulation/controller.py'
        spec = importlib.util.spec_from_file_location('simple_controller_'+digest(path), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        controller_class = module.SimpleTaskController
    if track == 'system0':
        policy = ScriptedDiagnostic(scene, task, trace, controller_class, source_pins, asset_pins)
        registry.register(policy)
    service = AgentService(output/'episodes', release=args.release, device=args.device,
                           registry=registry, allow_privileged=track == 'direct_joints' and mode == 'scripted')
    budget = args.idle_duration if mode == 'idle' else args.duration
    config = {'mode': mode, 'track': track, 'task': task_id, 'seed': seed,
              'controller_sources': source_pins, 'controller_assets':asset_pins,
              'controller_libraries':{name:importlib.metadata.version(name) for name in ('numpy','scipy')} if mode=='scripted' else {},
              'trained': False, 'privileged_diagnostic': mode == 'scripted'}
    with TestClient(create_app(service=service)) as client:
        available = checked(client.get('/tasks'))
        assert task_id in {item['task_id'] for item in available}
        body = dict(task_id=task_id, robot_id='sharpa_w01_right', seed=seed,
            agent_id='simple_'+mode+'_diagnostic', agent_version='v1',
            agent_config_sha256=hashlib.sha256(json.dumps({k:v for k,v in config.items() if k not in ('task','seed')}, sort_keys=True).encode()).hexdigest(),
            track=track, observation_profile='privileged_state' if track == 'direct_joints' and mode == 'scripted' else 'vision_proprio',
            budgets=dict(sim_time_s=budget, wall_time_s=1800., max_actions=6000,
                         max_model_calls=1, max_action_sim_s=budget))
        if policy:
            body.update(policy_id=policy.descriptor.policy_id, policy_version=policy.descriptor.version)
        ready = checked(client.post('/episodes', json=body), 201)
        episode = ready['episode_id']; prefix = f'/episodes/{episode}'
        observation = ready['observation']; caps = RobotCapabilities.model_validate(ready['capabilities'])
        check_images(client, observation)
        record_observation(trace, observation)
        assert client.get(prefix+'/private/metadata.json').status_code == 404
        if policy:
            assert not any(k.startswith('privileged') for k in observation)
        controller = None
        if mode == 'scripted' and track == 'direct_joints':
            controller = controller_class(scene, public_controller_task(task))
        initial_q = {n:max(limit.lower_rad, min(limit.upper_rad, observation['proprioception']['joint_positions_rad'][n]))
                     for n,limit in caps.joint_limits.items()}
        index = 0
        while True:
            action_id = f'action_{index:05d}'
            if policy:
                action = dict(kind='system0', action_id=action_id, observation_sequence=observation['sequence'],
                              skill='execute_simple_task', timeout_sim_s=budget)
            else:
                remaining = round((budget-observation['sim_time_s'])/.02)*.02
                if remaining <= 0:
                    break
                raw_command = ({'joint_positions_rad': initial_q, 'duration_s': min(.5, remaining)}
                               if mode == 'idle' else controller.next_action(controller_observation(observation)))
                action = direct_action(RobotJointCommand(**raw_command), caps, action_id=action_id,
                    observation_sequence=observation['sequence']).model_dump(mode='json')
            checked(client.post(prefix+'/actions', json=action), 202)
            started = time.monotonic(); last_update = started
            while True:
                status = checked(client.get(prefix+'/actions/'+action_id))
                if status['status'] not in ('accepted','running'):
                    break
                if time.monotonic()-last_update > 20:
                    print(json.dumps({'task':task_id,'mode':mode,'sim_s':status['sim_time_s'],'status':status['status']}), flush=True)
                    last_update = time.monotonic()
                if time.monotonic()-started > 1860:
                    raise TimeoutError('HTTP action exceeded server budget and cleanup allowance')
                time.sleep(.02 if track == 'direct_joints' else .1)
            observation = checked(client.get(prefix+'/observation'))
            record_observation(trace, observation, controller.diagnostic() if controller else {})
            index += 1
            result_response = client.get(prefix+'/result')
            if result_response.status_code == 200:
                result = result_response.json()
                break
            if status['status'] == 'failed' or policy:
                result = checked(client.post(prefix+'/finish'))
                break
        else:
            raise AssertionError('Unreachable')
        if 'result' not in locals():
            result = checked(client.post(prefix+'/finish'))
        final_observation = checked(client.get(prefix+'/observation'))
        check_images(client, final_observation)
        directory = output/'episodes'/episode
        hashes = load_json(directory/'artifact_hashes.json')
        assert all(digest(directory/p) == sha for p,sha in hashes.items())
        assert all(digest(output/'controller_source'/p)==expected for p,expected in source_pins.items()), 'Frozen controller source changed'
        assert all(digest(PROJECT/p)==expected for p,expected in source_pins.items() if p!='scenes/manipulation/controller.py'), 'Controller dependency source changed'
        assert all(digest(path)==expected for path,expected in asset_pins.items()), 'Controller kinematics assets changed'
        physics = load_json(directory/'private/physics_audit.json')
        final_score = load_json(directory/'private/evaluator_evidence.json')
        report = {**config, 'episode_id':episode, 'episode_directory':str(directory),
            'result':result, 'physics_audit':physics, 'final_score':final_score,
            'camera_timeline_sha256':digest(trace),
            'public_image_hashes_verified':True, 'episode_artifact_hashes_verified':True,
            'expected_outcome_observed': result['outcome']=='success' if mode=='scripted' else result['outcome']=='sim_timeout' and not final_score['success'],
            'real_agent_or_trained_rl_claimed':False, 'object_state_writeback':False}
        if policy:
            report['controller_final'] = policy.controller.diagnostic()
        elif controller:
            report['controller_final'] = controller.diagnostic()
        dump(output/'validation.json', report)
        print(json.dumps({'task':task_id,'mode':mode,'outcome':result['outcome'],'physics_valid':result['physics_valid'],
                          'sim_s':result['elapsed_sim_s'],'wall_s':result['elapsed_wall_s']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=DEFAULT_TASK_RELEASE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--modes', nargs='+', choices=['scripted','idle'], default=['scripted','idle'])
    parser.add_argument('--track', choices=['system0','direct_joints'], default='system0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--duration', type=float, default=40.)
    parser.add_argument('--idle-duration', type=float, default=6.)
    args = parser.parse_args()
    manifest = validate(args.release)
    tasks = args.tasks or [t['id'] for t in manifest['tasks']]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for task in tasks:
        for mode in args.modes:
            for seed in args.seeds:
                rows.append(run_episode(args, task, mode, seed))
                dump(args.output/'suite_result.json', {'status':'running','episodes':rows})
    results = [row['result'] for row in rows]
    metrics = aggregate_results(results, expected_episode_ids=[r['spec']['episode_id'] for r in results])
    report = {'status':'complete','scope':'Untrained scripted development diagnostics and idle controls; not Agent rankings.',
              'all_expected_outcomes':all(r['expected_outcome_observed'] for r in rows),
              'all_physics_valid':all(r['result']['physics_valid'] for r in rows),
              'episodes':rows,'metrics':metrics}
    dump(args.output/'suite_result.json', report)
    if not report['all_expected_outcomes'] or not report['all_physics_valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
