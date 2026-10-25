"""Read immutable gear rig evidence without retaining native point arrays."""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

from scripts.validation.gear_transmission_checks import assess_transmission, compact_native_row


class JsonlReader:
    def __init__(self, path):
        self.path = Path(path)
        self.before = self.path.stat()
        self.sha256 = hashlib.sha256()
        self.count = 0

    def __iter__(self):
        with self.path.open('rb') as stream:
            for line in stream:
                self.sha256.update(line)
                if not line.endswith(b'\n'):
                    raise ValueError('Truncated native JSONL: '+str(self.path))
                self.count += 1
                yield json.loads(line)
        after = self.path.stat()
        if (after.st_size, after.st_mtime_ns) != (self.before.st_size, self.before.st_mtime_ns):
            raise ValueError('Evidence changed during streaming read: '+str(self.path))

    def receipt(self):
        return {'path': str(self.path), 'sha256': self.sha256.hexdigest(),
                'bytes': self.before.st_size, 'rows': self.count}


def scan_native_frames(path, *, wrench_path=None):
    """Keep body states and exact signed contact sums; join receipts by step."""
    native = JsonlReader(path)
    wrenches = JsonlReader(wrench_path) if wrench_path is not None else None
    rows = []
    joined = zip_longest(native, wrenches) if wrenches is not None else ((r, None) for r in native)
    for row, wrench in joined:
        if row is None or wrenches is not None and wrench is None:
            raise ValueError('Native frames and applied-wrench receipts have different lengths')
        if wrench is not None and (wrench['native_step'] != row['native_step']
                or abs(wrench['time_s']-row['time_s']) > 1.e-9):
            raise ValueError('Applied wrench receipt does not match the native integration step')
        compact = compact_native_row(row)
        if wrench is not None:
            compact['wrench'] = wrench
        rows.append(compact)
    return rows, {'native': native.receipt(), 'wrenches': wrenches.receipt() if wrenches else None,
                  'retained_representation': 'Body states and signed_raw_point_reduction.v2; original raw files unchanged.'}


def _bound_json(path):
    path = Path(path)
    contents = path.read_bytes()
    return json.loads(contents), {'path': str(path), 'sha256': hashlib.sha256(contents).hexdigest()}


def reassess(run):
    from scripts.validation.validate_gear_transmission import assess_supply
    from scripts.validation.gear_passive_checks import canonical_passive_window
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    import scripts.validation.gear_transmission_checks as checks
    import scripts.validation.validate_gear_transmission as rig
    run = Path(run).resolve()
    source_root = Path(__file__).resolve().parents[2]
    imports = verify_if_snapshot(source_root, required_modules=(
        'scripts.validation.reassess_gear_transmission', 'scripts.validation.gear_transmission_checks',
        'scripts.validation.validate_gear_transmission', 'scripts.validation.gear_passive_checks'), entrypoint_file=Path(__file__))
    if imports is None:
        raise ValueError('Independent reassessment requires a complete source snapshot')
    # The supervisor writes this only after the native child has exited.
    process, process_identity = _bound_json(run/'process_receipt.json')
    original, original_identity = _bound_json(run/'result/report.json')
    inputs, input_identity = _bound_json(run/'input.json')
    metadata, metadata_identity = _bound_json(run/'result/metadata.json')
    info, scene = metadata['scene_info'], inputs['scene']
    dt = scene['isaacsim']['physics_dt_s']
    report = {'schema': 'gear_signed_raw_reassessment.v1', 'task_success_claimed': False,
        'full_scene_physics_accepted': False, 'quantitative_contact_force_qualified': False,
        'scope': 'Independent complete raw replay of contact signs/support/transmission only; preserves original report and all original limitations.',
        'original_process_receipt': process_identity, 'original_report': original_identity,
        'original_passed': original.get('passed'), 'original_exit_code': process.get('exit_code'),
        'runtime_imports': imports,
        'assessment_source_manifest_sha256': hashlib.sha256((source_root/'source_snapshot_manifest.json').read_bytes()).hexdigest(),
        'inputs': input_identity, 'metadata': metadata_identity,
        'assessment_source_sha256': {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, checks.__file__, rig.__file__)}}
    rows, identity = scan_native_frames(run/'result/nominal_supply/native_frames.jsonl')
    report['nominal_supply'] = assess_supply(rows, scene, info, dt)
    report['nominal_supply_stream'] = identity
    del rows
    root = run/'result/contact_load'
    if root.exists():
        rows, identity = scan_native_frames(root/'native_frames.jsonl', wrench_path=root/'applied_wrenches.jsonl')
        report['contact_load'] = assess_transmission(rows, info, dt, inputs['probe_contract'],
            geometry=inputs.get('transmission_geometry_contract', inputs['task']['evaluation']))
        report['contact_load_stream'] = identity
        elapsed = 0.
        report['canonical_passive_windows'] = {}
        for phase in inputs['probe_contract']['phases']:
            elapsed += phase['duration_s']
            if phase['id'] in ('settle', 'final_hold'):
                report['canonical_passive_windows'][phase['id']] = canonical_passive_window(
                    rows, info, end_time_s=elapsed, phase=phase['id'], dt=dt,
                    geometry=inputs.get('transmission_geometry_contract', inputs['task']['evaluation']))
        if rows:
            report['other_supply_tools'] = assess_supply(rows, scene, info, dt,
                names=['open_end_wrench', 'straight_hex_screwdriver'])
    report['recomputed_contact_checks_passed'] = bool(report['nominal_supply']['passed']
        and report.get('contact_load', {}).get('passed') and report.get('other_supply_tools', {}).get('passed')
        and len(report.get('canonical_passive_windows', {})) == 2
        and all(w['passed'] for w in report['canonical_passive_windows'].values()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = reassess(args.run)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
