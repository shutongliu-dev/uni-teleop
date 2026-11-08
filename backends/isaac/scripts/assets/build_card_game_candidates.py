"""Build immutable card-game assets from a verified complete source snapshot.

This CPU USD command never launches Isaac or alters the existing room. It runs
structural checks before writing, but those outcomes are not asset dependencies
or task admission gates. Native development evidence must be stored separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

from bench.paths import DATA_ROOT

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scenes.workrooms.card_room.game_designs import TASK_IDS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_candidate(task_id, output, *, source_snapshot, version='scene_v1'):
    """Build one fresh folder; source verification is mandatory at this boundary."""
    from pxr import Usd
    from scenes.workrooms.card_room.game_expansion_assets import author_task, audit_stage
    if not source_snapshot:
        raise ValueError('A verified complete source snapshot is required')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Candidate assets are immutable; choose a fresh version: '+str(output))
    stage = Usd.Stage.CreateInMemory()
    metadata = author_task(stage, task_id)
    metadata['version'] = version
    audit = audit_stage(stage, metadata)
    # These are physical inputs and original source identity, never prior test
    # results. The audit is returned to the caller, not embedded as admission.
    metadata['source_root'] = str(PROJECT)
    metadata['source_snapshot'] = source_snapshot
    output.mkdir(parents=True, exist_ok=False)
    usd_path = output/'initial.usda'
    stage.GetRootLayer().Export(str(usd_path))
    metadata_path = output/'metadata.json'
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    hashes = {name: digest(output/name) for name in ('initial.usda', 'metadata.json')}
    manifest = {
        'schema': 'card_game_candidate_asset_manifest.v1', 'task_id': task_id, 'version': version,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'initial_usd': str(usd_path), 'initial_usd_sha256': hashes['initial.usda'],
        'metadata': str(metadata_path), 'metadata_sha256': hashes['metadata.json'],
        'files_sha256': hashes, 'source_root': str(PROJECT), 'source_snapshot': source_snapshot,
        'root_prim': metadata['root_prim'], 'parent_environment': metadata['parent_environment'],
        'actor_count': len(metadata['actors']),
        'scope': 'Immutable physical scene inputs; no task admission, native qualification or manipulation result.',
    }
    manifest_path = output/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return {'task_id': task_id, 'output': str(output), 'manifest': str(manifest_path),
            'manifest_sha256': digest(manifest_path), 'structural_check': audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument('--task', choices=TASK_IDS)
    tasks.add_argument('--all', action='store_true')
    parser.add_argument('--output-root', type=Path,
                        default=Path(f'{DATA_ROOT}/assets/task_parts/board_games'))
    parser.add_argument('--version', default='scene_v1')
    parser.add_argument('--audit-output', type=Path,
                        help='Optional new development JSON outside the asset root; never an asset dependency')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]*', args.version):
        parser.error('Version must be a simple lowercase immutable directory name')
    # The verifier checks actual loaded origins; it intentionally does not
    # import the asset author on behalf of the build process.
    from scenes.workrooms.card_room import game_expansion_assets
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    verified = verify_if_snapshot(PROJECT,
        required_modules=('scenes.workrooms.card_room.game_designs',
                          'scenes.workrooms.card_room.game_expansion_assets'),
        entrypoint_file=Path(__file__))
    if verified is None:
        parser.error('Build from a complete frozen source snapshot, not an editable checkout')
    output_root = args.output_root.resolve()
    if args.audit_output:
        audit_path = args.audit_output.resolve()
        if audit_path == output_root or audit_path.is_relative_to(output_root):
            parser.error('Development results must stay outside the asset root')
        if audit_path.exists():
            raise FileExistsError('Preserve existing development evidence: '+str(audit_path))
    results = [build_candidate(task_id, output_root/task_id/args.version,
                               source_snapshot=verified, version=args.version)
               for task_id in (TASK_IDS if args.all else (args.task,))]
    if args.audit_output:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps({'source_snapshot': verified, 'results': results},
                                         ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(results, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
