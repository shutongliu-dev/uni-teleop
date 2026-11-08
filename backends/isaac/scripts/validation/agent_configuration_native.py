"""Exercise the complete Agent auto-choice runner against native Bench HTTP.

The diagnostic model returns deterministic decisions from public observations.
No external model request, object manipulation or task success is claimed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from bench.paths import load_json, OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.public_http_smoke import check, dump


FUNCTION_SEQUENCE = ('get_task', 'query_ik', 'move_geometry', 'move_geometry', 'hold', 'get_task')


def validate_feedback(feedback, function):
    check(feedback.get('ok') is True and feedback.get('function') == function,
          'Rejected or mismatched diagnostic function: '+function)
    result = feedback['result']
    if function == 'query_ik':
        check(result['solution_found'] is True and bool(result['hands'])
              and all(row['solution_found'] is True for row in result['hands'].values())
              and result['advances_physics'] is False and result['mutates_controller'] is False,
              'IK did not produce a read-only solution for every hand')
    elif function in ('move_geometry', 'hold'):
        reason = 'time_hold_completed' if function == 'hold' else 'controller_converged'
        check(result['status'] == 'completed' and result['termination_reason'] == reason
              and result['episode_finished'] is False and result['elapsed_sim_s'] > 0.,
              'Diagnostic action did not complete: '+function)
        if function == 'hold':
            check(abs(result['elapsed_sim_s']-.12) < 1.e-8, 'Hold did not execute the requested native duration')


def audit_execution_records(agent_directory):
    folder = Path(agent_directory)/'single/decisions'
    paths = sorted(folder.glob('*/execution.json'))
    check(len(paths) == len(FUNCTION_SEQUENCE), 'Incomplete diagnostic execution archive')
    receipts = []
    for path, function in zip(paths, FUNCTION_SEQUENCE):
        value = load_json(path)
        validate_feedback(value, function)
        receipts.append({'function': function, 'result': value['result']})
    before = load_json(folder/'0002/observation.json')
    after = load_json(folder/'0003/observation.json')
    check(before == after, 'Read-only IK changed the public observation')
    return {'passed': True, 'functions': list(FUNCTION_SEQUENCE),
            'ik_observation_unchanged': True, 'action_receipts': [r for r in receipts
                if r['function'] in ('move_geometry', 'hold')]}


class PublicDiagnosticModel:
    """Choose an advertised installation, then move every wrist 2 mm and back."""

    def __init__(self, configuration_id, events):
        self.configuration_id = configuration_id
        self.events = events
        self.calls = 0
        self.original = None
        self.initial = self.moved = self.final = None

    def public_metadata(self):
        return {'model': 'deterministic_public_interface_diagnostic',
                'protocol': 'scripted_diagnostic', 'external_model_requests': 0}

    @staticmethod
    def geometries(context):
        proprio = context['state']['observation']['proprioception']
        return proprio.get('hand_geometries') or {'right': proprio['geometry']}

    def arguments(self, targets):
        return ({'hand_geometries': targets} if len(targets) == 2
                else {'geometry': next(iter(targets.values()))})

    def generate(self, prompt, turns, **kwargs):
        context = json.loads(next(row['text'] for row in turns[-1]['content'] if row['type'] == 'text'))
        index = self.calls
        self.calls += 1
        if 2 <= index <= 7:
            validate_feedback(context['last_feedback'], FUNCTION_SEQUENCE[index-2])
        if index == 0:
            check('spec' not in context and 'state' not in context, 'Choice occurred after episode creation')
            check(set(context['robot_configurations']) == {'single_arm', 'dual_same_side', 'dual_opposed'},
                  'Missing public installation choice')
            check(not any(row.get('method') == 'POST' and row.get('path') == '/episodes'
                          for row in self.events), 'Native episode started before model choice')
            decision = {'configuration_id': self.configuration_id,
                        'reason': '诊断逐项检查公开构型的单 Agent 动作接口'}
        elif index == 1:
            decision = {'function': 'get_task', 'arguments': {}}
        elif index == 2:
            self.initial = deepcopy(self.geometries(context))
            check(set(self.initial) == ({'right'} if self.configuration_id == 'single_arm' else {'left', 'right'}),
                  'Selected installation has another hand inventory')
            self.original = {side: {'wrist': deepcopy(geometry['wrist']),
                'fingertip_frame_id': context['robot']['hands'][side]['frame_id'],
                'position_tolerance_m': .0005, 'orientation_tolerance_rad': .05}
                for side, geometry in self.initial.items()}
            decision = {'function': 'query_ik', 'arguments': self.arguments(self.original)}
        elif index == 3:
            targets = deepcopy(self.original)
            for target in targets.values():
                target['wrist']['position_m'][2] += .002
            decision = {'function': 'move_geometry',
                        'arguments': {'timeout_sim_s': 1., **self.arguments(targets)}}
        elif index == 4:
            self.moved = deepcopy(self.geometries(context))
            check(all(self.moved[side]['wrist']['position_m'][2] - geometry['wrist']['position_m'][2] >= .0004
                      for side, geometry in self.initial.items()), 'A selected hand did not move')
            decision = {'function': 'move_geometry',
                        'arguments': {'timeout_sim_s': 1., **self.arguments(self.original)}}
        elif index == 5:
            decision = {'function': 'hold', 'arguments': {'duration_s': .12}}
        elif index == 6:
            self.final = deepcopy(self.geometries(context))
            check(all(math.dist(self.final[side]['wrist']['position_m'], geometry['wrist']['position_m']) < .002
                      for side, geometry in self.initial.items()), 'A selected hand did not return')
            decision = {'function': 'get_task', 'arguments': {}}
        elif index == 7:
            decision = {'stop': True}
        else:
            raise RuntimeError('Unexpected additional model decision')
        self.events.append({'kind': 'model_decision', 'index': index, 'decision': deepcopy(decision)})
        return SimpleNamespace(text=json.dumps(decision),
            public_metadata=lambda: {**self.public_metadata(), 'attempts': 1})


def run(args):
    import httpx
    from agent.model_client.bench_http import BenchHTTPClient
    from agent.model_client.function_runner import RunConfig, run as run_agent
    from agent.model_client.scene_tasks import select_task

    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    check(output.is_relative_to(OUTPUT_ROOT), 'Native evidence belongs under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'native_agent_configuration_runner.v1', 'passed': False,
        'family': args.family, 'task_alias': args.task, 'configuration_id': args.configuration,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'gpu_uuid': args.gpu_uuid,
        'scope': 'Actual function_runner auto choice, native HTTP creation, RGB-D capture, public IK, all-hand wrist micro-motion/return/hold and finalization. Scripted diagnostic decisions; no autonomous task or object manipulation qualification.',
        'external_model_requests': 0, 'robot_manipulation': False, 'physics_accepted': False}
    server = client = None
    events = []
    try:
        source = verify_if_snapshot(root, required_modules=('scripts.validation.agent_configuration_native',),
                                    entrypoint_file=Path(__file__))
        check(source is not None, 'Complete frozen source required')
        report['runtime_imports'] = source
        selection = select_task(args.task, family=args.family)
        selection.require_runnable()
        reference = selection.robot_configurations[args.configuration]['source_release']
        report['selected_release'] = reference
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        environment = os.environ.copy()
        environment.update(BENCH_TASK_RELEASE=selection.source_release['path'],
            BENCH_AGENT_DATA_DIR=str(output/'episodes'), BENCH_DEVICE='cuda:0',
            CUDA_VISIBLE_DEVICES=args.gpu_uuid, ISAAC_BENCH_SINGLE_CUDA_DEVICE='1',
            ISAAC_ALLOW_UNVALIDATED_DRIVER='1',
            ISAAC_BENCH_EXPERIENCE=str(root/'bench/apps/workroom.headless.kit'),
            PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', '-m', 'scripts.validation.public_http_smoke',
                   'serve', '--port', str(port), '--evidence', str(output)]
        report['server_command'] = command
        with (output/'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        report['server_pid'] = server.pid

        def record_request(request):
            events.append({'kind': 'http_request', 'method': request.method,
                           'path': request.url.path, 'query': request.url.query.decode()})

        client = httpx.Client(base_url=url, timeout=600., trust_env=False,
                             event_hooks={'request': [record_request]})
        deadline = time.monotonic()+120.
        while True:
            check(server.poll() is None, 'Native server exited during startup')
            try:
                if client.get('/health', timeout=1.).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            check(time.monotonic() < deadline, 'Native server health deadline')
            time.sleep(.2)
        model = PublicDiagnosticModel(args.configuration, events)
        report['agent_report'] = run_agent(RunConfig(url, args.task, output/'agent',
            family=args.family, configuration='auto', seed=0, sim_budget=None,
            wall_budget=1800., http_timeout=600., model_timeout=60.,
            max_decisions=10, max_actions=4, max_status_queries=100000),
            bench=BenchHTTPClient(url, client=client, timeout_s=600., get_retries=0),
            models={'single': model}, source_verification=source)
        result = report['agent_report']
        check(result['error_type'] is None and result['task_setup_error'] is None
              and result['finalization_error'] is None and not result['action_recovery_pending'],
              'Agent runner failed: '+str(result.get('run_error') or result.get('task_setup_error')))
        check(result['ready_validated'] and result['run_status'] == 'completed'
              and result['termination'] == 'model_requested_stop', 'Agent did not complete normal finalization')
        check(result['pre_episode_model_calls'] == 1 and result['decisions'] == {'single': 8}
              and result['physical_attempts'] == {'single': 3}, 'Full choice/query/move/return/hold path not exercised')
        report['execution_audit'] = audit_execution_records(output/'agent')
        ready = load_json(output/'agent/episode_ready.json')
        request = load_json(output/'agent/create_request.json')
        spec = ready['spec']
        check(spec['configuration_id'] == request['configuration_id'] == args.configuration
              and spec['task_release_sha256'] == reference['manifest_sha256']
              and spec == result['result']['spec'], 'Selected release changed across episode lifecycle')
        check(spec['budgets']['sim_time_s'] is None, 'Client truncated the frozen task time budget')
        dof = len(ready['capabilities']['arm_joints'])+len(ready['capabilities']['hand_joints'])
        check(dof == (29 if args.configuration == 'single_arm' else 58), 'Selected robot has wrong DOF')
        check(result['result']['physics_valid'] and result['result']['outcome'] == 'cancelled',
              'Native episode had invalid physics or unexpected task completion')
        report.update(task_id=selection.task_id, dof=dof, scripted_decisions=model.calls,
            measured_delta_z_m={side: model.moved[side]['wrist']['position_m'][2]-geometry['wrist']['position_m'][2]
                                for side, geometry in model.initial.items()},
            return_position_error_m={side: math.dist(model.final[side]['wrist']['position_m'], geometry['wrist']['position_m'])
                                     for side, geometry in model.initial.items()}, passed=True)
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'detail': str(error), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        dump(output/'public_events.json', events)
        if client is not None:
            client.close()
        if server is not None:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=45.)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10.)
                    report.update(passed=False, forced_server_shutdown=True)
            report['server_exit_code'] = server.returncode
            if server.returncode != 0:
                report['passed'] = False
        shutdown_path = output/'server_shutdown.json'
        if shutdown_path.exists():
            shutdown = load_json(shutdown_path)
            report['server_shutdown'] = shutdown
            if len(shutdown['episodes']) != 1 or any(not row['finished'] or row['worker_exit_code'] != 0
                                                   for row in shutdown['episodes'].values()):
                report['passed'] = False
        else:
            report['passed'] = False
        try:
            report['runtime_imports_final'] = verify_if_snapshot(root,
                required_modules=('scripts.validation.agent_configuration_native',), entrypoint_file=Path(__file__))
            check(report['runtime_imports_final'] is not None, 'Final source snapshot missing')
        except BaseException as error:
            report.update(passed=False, final_verification_error=str(error))
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        dump(output/'qualification.json', report)
    print(json.dumps({'passed': report['passed'], 'output': str(output),
                      'configuration_id': args.configuration}), flush=True)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True, choices=('living_room', 'study', 'laboratory', 'laundry_room'))
    parser.add_argument('--task', required=True)
    parser.add_argument('--configuration', required=True, choices=('single_arm', 'dual_same_side', 'dual_opposed'))
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.gpu_uuid.startswith('GPU-'):
        parser.error('Select an explicit GPU UUID')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
