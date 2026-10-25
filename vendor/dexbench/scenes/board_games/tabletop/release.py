"""Freeze a new task definition and evaluator; never rewrite old releases."""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile

from .geometry import BoardGameGeometry, geometry_for
from .rules import BoardGameRules, contract_sha256, public_rule_contract
from .settlement import SettlementThresholds
from bench.paths import OUTPUT_ROOT, load_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_public_rule_contract(task: dict, *, release: str | Path | None = None) -> dict:
    """Validate the public contract against actual geometry/rules/thresholds.

    Catalog and native loaders call this before accepting a new release. When
    a release path is supplied, also require the separately frozen rules.json
    and its immutable manifest file hash. This function starts no simulator.
    """
    if not isinstance(task, dict):
        raise ValueError('Expected a task object with a public rule contract')
    try:
        geometry = BoardGameGeometry(**{f.name: task['geometry'][f.name] for f in fields(BoardGameGeometry)})
        thresholds = SettlementThresholds(**{f.name: task['settlement'][f.name] for f in fields(SettlementThresholds)})
        contract = task['public_rules']
        claimed = task['public_rules_sha256']
        rules = task['rules']
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing or malformed public rule contract') from exc
    expected = public_rule_contract(geometry, asdict(thresholds))
    normalize = lambda value: json.loads(json.dumps(value, allow_nan=False))
    if (task.get('schema') != 'robot_board_game_task.v1' or task.get('task_id') != geometry.game_id
            or normalize(task['geometry']) != normalize(geometry.specification())
            or normalize(rules) != normalize(BoardGameRules(geometry).specification())
            or normalize(contract) != normalize(expected) or claimed != contract_sha256(expected)
            or task.get('public_rules_file') != 'rules.json'):
        raise ValueError('Public rule contract differs from the evaluated task')
    if release is not None:
        root = Path(release).resolve()
        path = root/'rules.json'
        try:
            manifest = load_json(root/'release_manifest.json')
            frozen_rules = load_json(path)
        except (OSError, ValueError) as exc:
            raise ValueError('Missing or malformed frozen public rule files') from exc
        if (manifest.get('public_rules') != 'rules.json'
                or manifest.get('files_sha256', {}).get('rules.json') != sha256(path)
                or normalize(frozen_rules) != normalize(expected)):
            raise ValueError('Frozen public rules file differs from the task contract')
    return {'schema': expected['schema'], 'revision': expected['revision'],
            'public_rules_sha256': claimed, 'verified_against_evaluator': True,
            'frozen_rules_file_verified': release is not None}


def verify_evaluator_sources(release: str | Path, task: dict | None = None) -> dict:
    """Verify frozen files and every actual imported evaluator module together.

    All project dependencies of this evaluator are inside this package; the
    native observer imports only NumPy and the Python standard library outside
    it. The runtime's complete source snapshot separately pins the simulator.
    """
    release = Path(release).resolve()
    task = task if task is not None else load_json(release/'task.json')
    expected = task['sources_sha256']
    source_root = Path(__file__).resolve().parent
    frozen_root = release/'evaluator'/'tabletop'
    actual = {path.name: sha256(path) for path in sorted(source_root.glob('*.py'))}
    frozen = {path.name: sha256(path) for path in sorted(frozen_root.glob('*.py'))}
    if actual != expected or frozen != expected:
        raise ValueError('Imported or frozen tabletop evaluator differs from task source inventory')
    for name in expected:
        if Path(name).name != name or not name.endswith('.py'):
            raise ValueError('Evaluator inventory must contain local Python source filenames')
        module_name = __package__+('' if name == '__init__.py' else '.'+Path(name).stem)
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != source_root/name:
            raise ValueError('Evaluator imported outside the verified package: '+module_name)
    return {'source_root': str(source_root), 'sources_sha256': actual,
            'frozen_root': str(frozen_root), 'all_import_paths_verified': True}


def task_specification(geometry: BoardGameGeometry | None = None,
                       thresholds: SettlementThresholds | None = None) -> dict:
    geometry, thresholds = geometry or geometry_for('tic_tac_toe_3x3'), thresholds or SettlementThresholds()
    contract = public_rule_contract(geometry, asdict(thresholds))
    return {'schema': 'robot_board_game_task.v1', 'task_id': geometry.game_id,
            'release_policy': 'new_directory_for_every_scene_geometry_physics_or_evaluator_change',
            'rules': BoardGameRules(geometry).specification(), 'geometry': geometry.specification(),
            'public_rules': contract, 'public_rules_sha256': contract_sha256(contract),
            'public_rules_file': 'rules.json',
            'settlement': {**asdict(thresholds), 'position_origin': 'piece_base_centre',
                           'evidence_source': 'complete_consecutive_native_solver_substeps',
                           'support': 'positive_impulse_from_named_board_collider',
                           'release': 'no_positive_contact_with_either_robot',
                           'integrity': 'recoverable_displacement_blocks_commit_until_original_cells_are_stable',
                           'outcome': 'turn_advances_only_after_verified_physical_settlement'},
            'initialization': 'all_piece_poses_set_at_reset_only',
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'sources_sha256': {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))}}


def prepare_task_release(output: str | Path, *, scene: dict, geometry: BoardGameGeometry,
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
        verify_public_rule_contract(task)
        for name, data in (('scene.json', scene), ('task.json', task), ('rules.json', task['public_rules'])):
            (staging/name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')
        if initial_state is not None:
            (staging/'initial_state.json').write_text(json.dumps(initial_state, indent=2, sort_keys=True, allow_nan=False)+'\n')
        evaluator = staging/'evaluator'/'tabletop'
        evaluator.mkdir(parents=True)
        for source in sorted(Path(__file__).parent.glob('*.py')):
            shutil.copy2(source, evaluator/source.name)
            if sha256(evaluator/source.name) != task['sources_sha256'][source.name]:
                raise ValueError('Evaluator source changed while freezing')
        manifest = {'schema': 'robot_board_game_task_release.v1', 'task_id': geometry.game_id,
                    'status': 'prepared_unvalidated', 'physics_accepted': False,
                    'task_policy_success_claimed': False, 'scene': 'scene.json', 'task': 'task.json',
                    'public_rules': 'rules.json',
                    'frozen_evaluator': 'evaluator/tabletop',
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
