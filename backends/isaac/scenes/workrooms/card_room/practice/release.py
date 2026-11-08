"""Immutable ordinary-episode derivatives of the four unchanged formal games."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile

from bench.paths import load_json, OUTPUT_ROOT
from scenes.board_games.expansion_release import (SOURCES as FORMAL_SOURCES,
    canonical_hash, checked, dump, local_path, sha256)
from ..expansion_tasks import RELEASES
from .cases import GYM_IDS, build_case, parent_id, rules_for, selections, verify_case

SCHEMA = 'card_rsi_practice_release.v1'
TASK_SCHEMA = 'card_rsi_practice_task.v1'
SOURCE_ROOT = Path(__file__).resolve().parents[4]
SOURCES = tuple(dict.fromkeys((*FORMAL_SOURCES,
    'scenes/workrooms/card_room/expansion_tasks.py',
    'scenes/workrooms/card_room/practice/__init__.py',
    'scenes/workrooms/card_room/practice/cases.py',
    'scenes/workrooms/card_room/practice/referee.py',
    'scenes/workrooms/card_room/practice/runtime.py',
    'scenes/workrooms/card_room/practice/release.py',
    'bench/protocol/practice.py', 'bench/protocol/task_referee.py')))


def source_hashes():
    return {relative: sha256(SOURCE_ROOT/relative) for relative in SOURCES}


def case_key(selection):
    return '/'.join(selection[key] for key in ('split', 'active_side', 'case_id'))


def reset_layer(source_usd, case, layer_dir=None):
    """Only transform opinions over original colliders/materials/mass/inertia.

    With ``layer_dir`` (the directory the layer is written to) the sublayer is
    layer-relative, so the frozen layer never names this machine's data root.
    """
    if layer_dir is not None:
        source_usd = os.path.relpath(source_usd, layer_dir)
        source_usd = source_usd if source_usd.startswith('.') else './'+source_usd
    lines = ['#usda 1.0', '(', '    defaultPrim = "World"', '    metersPerUnit = 1',
             '    upAxis = "Z"', f'    subLayers = [@{source_usd}@]', ')',
             'over "World" {', '    over "CardGame" {', '        over "Actors" {']
    for name, pose in sorted(case['reset'].items()):
        vector = ', '.join(repr(float(v)) for v in pose['position_m'])
        quat = ', '.join(repr(float(v)) for v in pose['quaternion_wxyz'])
        lines.extend([f'            over "{name}" {{', f'                double3 xformOp:translate = ({vector})',
            f'                quatf xformOp:orient = ({quat})', '            }'])
    return '\n'.join(lines+['        }', '    }', '}', ''])


def parent_inputs(gym_id, *, assets=True):
    from scenes.board_games.expansion_release import validate_release
    parent = RELEASES[parent_id(gym_id)]
    root = Path(parent['path'])
    checked(root/'release_manifest.json', parent['manifest_sha256'])
    manifest = validate_release(root, load_json(root/'release_manifest.json'), assets=assets)
    scene, task, initial = [load_json(root/name) for name in ('scene.json', 'task.json', 'initial_state.json')]
    return manifest, scene, task, initial


def _scene(root, source_scene, case):
    scene = deepcopy(source_scene)
    scene.pop('board_game_task')
    scene['description'] = 'Single learner card-room practice; the other original robot retains acknowledged targets.'
    public = case['public']
    scene['card_practice_task'] = {'schema': 'card_rsi_practice_native.v1',
        'task_id': public['gym_id'], 'active_side': public['active_side'],
        'private_case_sha256': case['private_case_sha256']}
    relative = 'cases/'+case_key(public)+'/initial.usda'
    scene['native_task_bundle']['usd_path'] = str(Path(root)/relative)
    scene['native_task_bundle']['usd_sha256'] = sha256(Path(root)/relative)
    return scene


def _task(gym_id, parent_task, case):
    rules = rules_for(gym_id)
    title = parent_id(gym_id)+' 单侧自主练习'
    return {'schema': TASK_SCHEMA, 'id': gym_id, 'task_id': gym_id, 'title_zh': title,
        'steps': [{'instruction_zh': case['public']['goal']['description']}],
        'public_rules': rules, 'public_rules_sha256': canonical_hash(rules),
        'public_referee_contract_sha256': canonical_hash(rules['referee']), 'public_rules_file': 'rules.json',
        'capabilities': {'robot_episodes': True}, 'asset_metadata': deepcopy(parent_task['asset_metadata']),
        'practice_case': deepcopy(case), 'sources_sha256': source_hashes(),
        'physics_accepted': False, 'task_policy_success_claimed': False,
        'initialization': 'finite_immutable_reset_only_transform_layer', 'formal_game_result': False}


def prepare_release(output, gym_id):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Prepare a fresh practice release under shared datasets')
    parent_manifest, scene, parent_task, initial = parent_inputs(gym_id)
    source_usd = scene['native_task_bundle']['usd_path']
    cases = {case_key(selection): build_case(gym_id, selection, parent_task['asset_metadata'])
             for selection in selections(gym_id)}
    if len({c['case_fingerprint'] for c in cases.values()}) != len(cases):
        raise ValueError('Curriculum repeats physical reset and goal under another split or label')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'-', dir=output.parent))
    try:
        for key, case in cases.items():
            target = staging/'cases'/key
            target.mkdir(parents=True)
            dump(target/'case.json', case)
            (target/'initial.usda').write_text(reset_layer(source_usd, case, target))
        default = cases[next(iter(cases))]
        base_scene = _scene(staging, scene, default)
        base_scene['native_task_bundle']['usd_path'] = str(output/'cases'/case_key(default['public'])/'initial.usda')
        for name, value in (('scene.json', base_scene), ('task.json', _task(gym_id, parent_task, default)),
                            ('initial_state.json', initial), ('rules.json', rules_for(gym_id)),
                            ('cases.json', cases)):
            dump(staging/name, value)
        for relative in SOURCES:
            target = staging/'evaluator'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE_ROOT/relative, target)
        # Standard EpisodeSpec hashes this exact evaluator file. The loader
        # selects the verified registered native adapter, never its update(obs).
        shutil.copyfile(SOURCE_ROOT/'scenes/workrooms/card_room/practice/referee.py', staging/'evaluator/sequence_evaluator.py')
        files = {str(p.relative_to(staging)): sha256(p) for p in staging.rglob('*') if p.is_file()}
        physics = {'parent_profile_sha256': parent_manifest['runtime_identity']['physics_profile_sha256'],
            'physics': scene['physics'], 'isaacsim': scene['isaacsim'],
            'rigid_body_sampling': scene['native_task_bundle']['rigid_body_sampling']}
        manifest = {'schema': SCHEMA, 'task_id': gym_id, 'status': 'prepared_unvalidated',
            'scene': 'scene.json', 'task': 'task.json', 'initial_state': 'initial_state.json',
            'public_rules': 'rules.json', 'cases': 'cases.json', 'files_sha256': files,
            'parent_release': deepcopy(RELEASES[parent_id(gym_id)]),
            'parent_environment': deepcopy(parent_manifest['parent_environment']),
            'asset_files_sha256': deepcopy(parent_manifest['asset_files_sha256']),
            'sources_sha256': source_hashes(), 'physics_profile_sha256': canonical_hash(physics),
            'physics_accepted': False, 'task_policy_success_claimed': False, 'formal_game_result': False}
        dump(staging/'release_manifest.json', manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_release(output, manifest)


def validate_release(root, manifest, *, assets=True):
    root = Path(root).resolve()
    gym_id = manifest.get('task_id')
    if (manifest.get('schema') != SCHEMA or gym_id not in GYM_IDS
            or manifest.get('status') != 'prepared_unvalidated'
            or any(manifest.get(k) is not False for k in ('physics_accepted', 'task_policy_success_claimed', 'formal_game_result'))
            or manifest.get('parent_release') != RELEASES[parent_id(gym_id)]
            or manifest.get('sources_sha256') != source_hashes()):
        raise ValueError('Practice release identity or evaluator sources changed')
    files = manifest.get('files_sha256', {})
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'release_manifest.json'}
    if set(files) != actual:
        raise ValueError('Practice release file inventory is not exact')
    for relative, digest in files.items():
        checked(local_path(root, relative), digest)
    for relative, digest in source_hashes().items():
        if files.get('evaluator/'+relative) != digest:
            raise ValueError('Practice release omitted an evaluator dependency')
    if files.get('evaluator/sequence_evaluator.py') != source_hashes()['scenes/workrooms/card_room/practice/referee.py']:
        raise ValueError('Practice evaluator identity differs from its native scorer')
    parent_manifest, source_scene, parent_task, initial = parent_inputs(gym_id, assets=assets)
    if (manifest.get('parent_environment') != parent_manifest['parent_environment']
            or manifest.get('asset_files_sha256') != parent_manifest['asset_files_sha256']):
        raise ValueError('Practice room/asset dependencies differ from original formal release')
    cases = load_json(root/'cases.json')
    if set(cases) != {case_key(s) for s in selections(gym_id)}:
        raise ValueError('Practice release must contain exactly the finite case inventory')
    fingerprints = set()
    for key, case in cases.items():
        verify_case(case, parent_task['asset_metadata'])
        if case_key(case['public']) != key or case['case_fingerprint'] in fingerprints:
            raise ValueError('Practice case identity/split repeats another physical case')
        fingerprints.add(case['case_fingerprint'])
        if load_json(root/'cases'/key/'case.json') != case:
            raise ValueError('Practice case index differs from its frozen payload')
        if (root/'cases'/key/'initial.usda').read_text() != reset_layer(
                source_scene['native_task_bundle']['usd_path'], case, root/'cases'/key):
            raise ValueError('Practice layer changed more than declared reset transforms')
    default = cases[case_key(selections(gym_id)[0])]
    for name, expected in (('scene.json', _scene(root, source_scene, default)),
        ('task.json', _task(gym_id, parent_task, default)), ('rules.json', rules_for(gym_id)), ('initial_state.json', initial)):
        if load_json(root/name) != expected:
            raise ValueError('Practice changed frozen robot/room/rules/reset inputs: '+name)
    physics = {'parent_profile_sha256': parent_manifest['runtime_identity']['physics_profile_sha256'],
        'physics': source_scene['physics'], 'isaacsim': source_scene['isaacsim'],
        'rigid_body_sampling': source_scene['native_task_bundle']['rigid_body_sampling']}
    if manifest.get('physics_profile_sha256') != canonical_hash(physics):
        raise ValueError('Practice physics profile changed')
    entry = {'id': gym_id, 'title_zh': parent_id(gym_id)+' 单侧自主练习',
        'difficulty': 8+GYM_IDS.index(gym_id), 'cloth': False, 'scene': 'scene.json', 'task': 'task.json',
        'initial_state': 'initial_state.json', 'task_kind': 'manipulation'}
    return {**deepcopy(manifest), 'tasks': [entry], 'task_kind': 'manipulation',
        'files': {k: {'sha256': v} for k, v in files.items()}}


def task_inputs(root, gym_id, manifest, *, seed=0, practice=None):
    if gym_id != manifest['task_id']:
        raise ValueError('Wrong practice task selected')
    root = Path(root)
    # Default is for static registry discovery only. HTTP explicitly requires
    # practice selection; seed never relabels an immutable physical case.
    selection = selections(gym_id)[0] if practice is None else practice
    key = case_key(selection)
    cases = load_json(root/'cases.json')
    if key not in cases or {k: cases[key]['public'][k] for k in ('active_side', 'case_id', 'split')} != selection:
        raise ValueError('Unknown frozen practice selection')
    case = cases[key]
    _, source_scene, parent_task, initial = parent_inputs(gym_id, assets=False)
    task = _task(gym_id, parent_task, case)
    task['runtime_release_manifest_sha256'] = sha256(root/'release_manifest.json')
    return deepcopy(manifest['tasks'][0]), _scene(root, source_scene, case), task, initial


def load_practice_inputs(root):
    """Trusted host input; private fingerprints/reset maps must not be prompted."""
    root = Path(root).resolve()
    manifest = validate_release(root, load_json(root/'release_manifest.json'))
    return {'manifest': manifest, 'cases': load_json(root/'cases.json'),
        'parent_release': deepcopy(manifest['parent_release']),
        'source_metadata': {'release_root': str(root), 'release_manifest_sha256': sha256(root/'release_manifest.json'),
            'sources_sha256': deepcopy(manifest['sources_sha256'])}}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=GYM_IDS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare_release(args.output, args.task)
    print(json.dumps({'task_id': result['task_id'], 'output': str(args.output.resolve()),
        'manifest_sha256': sha256(args.output/'release_manifest.json')}))


if __name__ == '__main__':
    main()
