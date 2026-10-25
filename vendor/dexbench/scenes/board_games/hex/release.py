"""Freeze a new task definition and evaluator; never rewrite old releases."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .geometry import HexGeometry
from .rules import HexRules
from .settlement import SettlementThresholds
from bench.paths import OUTPUT_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def task_specification(geometry: HexGeometry | None = None,
                       thresholds: SettlementThresholds | None = None) -> dict:
    geometry, thresholds = geometry or HexGeometry(), thresholds or SettlementThresholds()
    return {'schema': 'robot_hex_task.v1', 'task_id': 'hex7x7',
            'release_policy': 'new_directory_for_every_scene_geometry_physics_or_evaluator_change',
            'rules': HexRules.specification(), 'geometry': geometry.specification(),
            'settlement': {**asdict(thresholds), 'position_origin': 'piece_base_centre',
                           'evidence_source': 'complete_consecutive_native_solver_substeps',
                           'support': 'positive_impulse_from_named_board_collider',
                           'release': 'no_positive_contact_with_either_robot',
                           'integrity': 'committed_pieces_remain_in_their_cells',
                           'outcome': 'turn_advances_only_after_verified_physical_settlement'},
            'initialization': 'all_piece_poses_set_at_reset_only',
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'sources_sha256': {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))}}


def prepare_task_release(output: str | Path, *, scene: dict, geometry: HexGeometry,
                         asset_files: tuple[str | Path, ...] = (),
                         thresholds: SettlementThresholds | None = None,
                         protocol: dict | None = None,
                         initial_state: dict | None = None) -> dict:
    """Write immutable preparation files under shared datasets, with asset hashes.

    The caller supplies the fully resolved scene and protocol. Preparation is
    deliberately not physics acceptance; attach native evidence separately.
    """
    output = Path(output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError(f'Task releases are prepared under {OUTPUT_ROOT}')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        task = task_specification(geometry, thresholds)
        if protocol is not None:
            task['player_protocol'] = protocol
        for name, data in (('scene.json', scene), ('task.json', task)):
            (staging/name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')
        if initial_state is not None:
            (staging/'initial_state.json').write_text(json.dumps(initial_state, indent=2, sort_keys=True, allow_nan=False)+'\n')
        evaluator = staging/'evaluator'/'hex'
        evaluator.mkdir(parents=True)
        for source in sorted(Path(__file__).parent.glob('*.py')):
            shutil.copy2(source, evaluator/source.name)
            if sha256(evaluator/source.name) != task['sources_sha256'][source.name]:
                raise ValueError('Evaluator source changed while freezing')
        manifest = {'schema': 'robot_hex_task_release.v1', 'task_id': 'hex7x7',
                    'status': 'prepared_unvalidated', 'physics_accepted': False,
                    'task_policy_success_claimed': False, 'scene': 'scene.json', 'task': 'task.json',
                    'frozen_evaluator': 'evaluator/hex',
                    'asset_files_sha256': {str(Path(path).resolve()): sha256(Path(path)) for path in asset_files},
                    'files_sha256': {str(path.relative_to(staging)): sha256(path)
                                     for path in sorted(staging.rglob('*')) if path.is_file()}}
        (staging/'release_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
