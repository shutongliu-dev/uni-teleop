"""Actual native practice/Agent interface probe with zero external model requests.

Only public queries, owned wrist micro-motion/return and acknowledged-drive
holds; never manipulation evidence or a supplied piece-handling strategy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from agent.model_client.function_runner import RunConfig, run
from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_runtime_imports
from scripts.validation.card_game_agent_smoke import validate_action


class PublicProbe:
    def __init__(self, side):
        self.side, self.index = side, 0
        self.target = self.inactive = None
        self.moved_distance_m = self.return_error_m = None

    def public_metadata(self):
        return {'model': 'scripted_public_interface_probe', 'external_model_requests': 0}

    def generate(self, prompt, turns, **kwargs):
        context = json.loads(next(row['text'] for row in turns[-1]['content'] if row['type'] == 'text'))
        observation = context['state']['observation']
        proprio = observation['proprioception']
        drive = proprio['commanded_joint_positions_rad']
        inactive = {n: v for n, v in drive.items() if not n.startswith((self.side+'_', 'fr3_'+self.side+'_'))}
        if self.inactive is None:
            self.inactive = inactive
        if inactive != self.inactive:
            raise RuntimeError('Practice probe observed an inactive drive change')
        geometry = proprio['hand_geometries'][self.side]
        index = self.index
        self.index += 1
        if index in (2, 4, 5, 6):
            feedback = context.get('last_feedback')
            if feedback is None:
                feedback = json.loads(turns[-2]['content'])['tool_feedback']
            validate_action(feedback, 'hold' if index in (2, 6) else 'move_geometry',
                duration=.5 if index == 2 else .12 if index == 6 else None)
        if index in (0, 2, 6):
            decision = {'function': 'get_task', 'arguments': {}}
        elif index in (1, 5):
            decision = {'function': 'hold', 'arguments': {
                'duration_s': .5 if index == 1 else .12, 'mode': 'acknowledged_drive'}}
        elif index == 3:
            self.target = {'wrist': deepcopy(geometry['wrist']),
                'fingertip_frame_id': self.side+'_hand_flange',
                'position_tolerance_m': .0005, 'orientation_tolerance_rad': .05}
            moved = deepcopy(self.target)
            moved['wrist']['position_m'][2] += .002
            decision = {'function': 'move_geometry', 'arguments': {
                'hand_geometries': {self.side: moved}, 'timeout_sim_s': 2.}}
        elif index == 4:
            self.moved_distance_m = geometry['wrist']['position_m'][2]-self.target['wrist']['position_m'][2]
            if self.moved_distance_m < .0004:
                raise RuntimeError('Owned practice wrist did not move')
            decision = {'function': 'move_geometry', 'arguments': {
                'hand_geometries': {self.side: self.target}, 'timeout_sim_s': 2.}}
        elif index == 7:
            self.return_error_m = math.dist(geometry['wrist']['position_m'], self.target['wrist']['position_m'])
            if self.return_error_m >= .002:
                raise RuntimeError('Owned practice wrist did not return')
            decision = {'stop': True}
        else:
            raise RuntimeError('Unexpected practice probe decision')
        return SimpleNamespace(text=json.dumps(decision),
            public_metadata=lambda: {**self.public_metadata(), 'attempts': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--side', choices=('left', 'right'), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    verification = verify_runtime_imports(root, required_modules=('agent.model_client.function_runner',),
                                          entrypoint_file=Path(__file__))
    verification['manifest_sha256'] = hashlib.sha256((root / 'source_snapshot_manifest.json').read_bytes()).hexdigest()
    probe = PublicProbe(args.side)
    report = run(RunConfig(bench_url=args.bench_url, task='polyomino_territory_practice', family='card_room',
        output=args.output, practice={'active_side': args.side, 'case_id': 'mono_translate_00', 'split': 'train'},
        max_decisions=8, max_actions=5, max_function_calls=8, sim_budget=5., wall_budget=None,
        model_timeout=None, http_timeout=None), models={'single': probe}, source_verification=verification)
    calls = [load_json(p) for p in sorted((args.output / 'single/calls').glob('*.json'))]
    passed = (report['run_error'] is None and report['finalization_error'] is None
        and (report.get('result') or {}).get('physics_valid') is True
        and len(calls) == 7 and all(call.get('ok') for call in calls)
        and all(calls[index]['result']['status'] == 'completed' for index in (1, 3, 4, 5))
        and probe.return_error_m is not None)
    (args.output / 'probe_assessment.json').write_text(json.dumps({
        'passed': passed, 'external_model_requests': 0, 'native_manipulation_qualified': False,
        'side': args.side, 'episode_id': report.get('episode_id'),
        'wrist_upward_displacement_m': probe.moved_distance_m, 'wrist_return_error_m': probe.return_error_m,
        'scope': 'Public Agent/Bench/native creation, task feedback, owned wrist 2mm upward micro-motion/return and holds; no piece handling'}, indent=2)+'\n')
    if not passed:
        raise SystemExit('Practice interface probe failed; inspect retained artifacts')


if __name__ == '__main__':
    main()
