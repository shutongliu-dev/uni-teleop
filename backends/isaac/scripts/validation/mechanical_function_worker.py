"""Canonical native worker with private, read-only fastener instrumentation.

Only the operator's scripted diagnostic selects this entrypoint. Normal Agent
episodes still use bench.runtime.worker. Joint commands, cameras, task scoring,
ACKs, and state sealing all use that worker's unchanged implementations.
"""
from __future__ import annotations

from pathlib import Path

from bench.isaacsim.release import dump
from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_mechanical_tool_motion import ToolObserver, grip_window
from scripts.validation.validate_mechanical_dynamics import FixtureMonitor, native_task_parameters
from scripts.validation.mechanical_checks import contact_filter_coverage
from scripts.validation.mechanical_tool_checks import (contact_exchange_check,
    preinserted_window, released_phase_window)
from scripts.validation.validate_workrooms import stage_inventory


class FunctionObserver(ToolObserver):
    """Observe existing native steps, keeping a bounded in-memory tail.

    The full raw point stream is retained on disk. No targets, object state,
    material settings, or evaluator results are written by this observer.
    """

    def __init__(self, env, output):
        self.directory = Path(output) / 'fastener_physics'
        self.directory.mkdir(exist_ok=False)
        self.phase_file = Path(output) / 'engineering_phase.json'
        self.original_contact_reader = env._read_rigid_contacts
        self.fixture = FixtureMonitor(env)
        env._read_rigid_contacts = self.fixture.wrap(self.original_contact_reader)
        self.samples = 0
        self.maximum_penetration = 0.
        self.interference_samples = 0
        self.close_result = None
        self.endpoint_targets = None
        self.phase = 'initial'
        try:
            super().__init__(env, self.directory)
            state = env.capture_state()
            self.initial_objects = {name: {
                'position_m': state['object_q'][index, :3].tolist(),
                'quaternion_wxyz': state['object_q'][index, [6, 3, 4, 5]].tolist()}
                for name, index in self.indices.items()}
            self.paths = {name: env.scene_info['objects'][name]['path'] for name in self.indices}
            self.paths['fixture'] = env.scene_info['fixtures']['threaded_fixture']['path']
            root = Path(__file__).resolve().parents[2]
            imports = verify_if_snapshot(root, required_modules=(
                'scripts.validation.mechanical_function_worker',
                'scripts.validation.validate_mechanical_tool_motion', 'bench.runtime.worker'),
                entrypoint_file=Path(__file__))
            dump(self.directory / 'runtime_imports.json', imports)
            checks = {'stage_inventory': stage_inventory(env),
                      'contact_coverage': contact_filter_coverage(env),
                      'task_parameters': native_task_parameters(env)}
            dump(self.directory / 'setup.json', checks)
            if any(not value.get('passed', False) for value in checks.values()):
                raise RuntimeError('Fastener native setup checks failed')
            dump(self.directory / 'scope.json', {
                'privileged_engineering_diagnostic': True, 'model_calls': 0,
                'read_only': True, 'object_state_writes_after_reset': 0,
                'executor': 'canonical EpisodeSession / EpisodeFunctions / native worker',
                'observer': 'Single contact acquisition on every existing native substep',
                'full_task_success_claimed_by_observer': False})
        except BaseException:
            if hasattr(self, 'closed') and not self.closed:
                super().close()
            env._read_rigid_contacts = self.original_contact_reader
            raise

    def step(self, *args, **kwargs):
        if self.samples % round(.02 / self.env.dt) == 0:
            config = load_json(self.phase_file) if self.phase_file.exists() else {'phase': 'settle'}
            self.phase = config['phase']
            self.endpoint_targets = config.get('joint_targets_rad')
        result = super().step(*args, **kwargs)
        self.samples += 1
        row = self.rows[-1]
        self.maximum_penetration = max(self.maximum_penetration,
                                      row['maximum_robot_contact_penetration_m'])
        self.interference_samples += bool(row['robot_interference'])
        if self.samples % round(.02 / self.env.dt) == 0:
            self.stream.flush()
            # The gate requires a complete window from the actual hold phase.
            recent = self.rows[-round(.3 / self.env.dt):]
            current = {'phase': self.phase, 'native_samples': self.samples,
                       'simulation_time_s': row['simulation_time_s'],
                       'grip_last_300ms': grip_window(recent, self.env.dt, .3),
                       'maximum_robot_penetration_m': self.maximum_penetration,
                       'interference_samples': self.interference_samples,
                       'joint_positions_rad': row['joint_positions_rad'],
                       'object_poses_xyzw': row['object_poses_xyzw'],
                       'object_velocities': row['object_velocities']}
            if self.phase == 'settle':
                current['preinserted_reset_stability'] = preinserted_window(
                    self.rows, self.initial_objects, self.paths, self.env.dt)
            if self.phase in ('open', 'retreat', 'final_hold') and self.endpoint_targets:
                current['release_endpoint'] = released_phase_window(self.rows, self.phase,
                    self.endpoint_targets, self.hand, self.tool_path, self.env.dt)
            pending = self.directory / 'progress.pending.json'
            dump(pending, current)
            pending.replace(self.directory / 'progress.json')
            # This bounds memory; every sample remains in native_frames.jsonl.
            self.rows[:] = self.rows[-round(1. / self.env.dt):]
        if row['robot_interference'] or row['maximum_robot_contact_penetration_m'] > .002:
            self.stream.flush()
            raise RuntimeError('Fastener motion caused unexpected robot contact or penetration above 2 mm')
        return result

    def close(self):
        if self.close_result is not None:
            if not self.close_result['passed']:
                raise RuntimeError('Fastener observer failed its native evidence seal')
            return self.close_result
        try:
            exchange = super().close()
            dump(self.directory / 'runtime_imports_final.json', verify_if_snapshot(
                Path(__file__).resolve().parents[2], required_modules=(
                    'scripts.validation.mechanical_function_worker', 'bench.runtime.worker',
                    'bench.simulation.isaacsim_backend.env'), entrypoint_file=Path(__file__)))
            dump(self.directory / 'metadata_final.json', self.env.metadata())
            fixture = self.fixture.result(self.samples)
            coverage = contact_exchange_check(exchange, self.samples, self.samples)
            result = {'passed': bool(coverage['passed'] and fixture['passed']
                                    and not self.interference_samples and self.maximum_penetration <= .002),
                      'sampling_passed': coverage['passed'],
                      'fixture_passed': fixture['passed'],
                      'contact_exchange': exchange, 'coverage': coverage, 'fixture': fixture,
                      'native_samples': self.samples,
                      'maximum_robot_penetration_m': self.maximum_penetration,
                      'interference_samples': self.interference_samples,
                      'task_success_claimed': False}
            dump(self.directory / 'seal.json', result)
            self.close_result = result
            if not result['passed']:
                raise RuntimeError('Fastener observer failed its native evidence seal')
            return result
        finally:
            self.env._read_rigid_contacts = self.original_contact_reader


def main():
    from bench.runtime.worker import main as run_worker
    run_worker(observer_factory=FunctionObserver)


if __name__ == '__main__':
    main()
