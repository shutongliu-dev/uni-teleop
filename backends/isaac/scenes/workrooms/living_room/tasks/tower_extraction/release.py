"""Derive a frozen host-termination contract without rebuilding tower physics."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil

from bench.isaacsim.release import checked, digest, dump, validate
from .task import task_configuration
from bench.paths import DATA_ROOT, load_json

DATASETS = Path(f'{DATA_ROOT}/datasets')


def prepare_termination_release(source, output, *, source_manifest_sha256):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not output.is_relative_to(DATASETS) or output.exists():
        raise ValueError('Choose a fresh directory under shared datasets')
    checked(source/'release_manifest.json', source_manifest_sha256)
    original = validate(source, assets=True)
    if original.get('task_id') != 'tower_extraction' or original.get('schema') != 'living_room_task_release.v1':
        raise ValueError('Expected a frozen tower-extraction release')
    output.mkdir(parents=True)
    for relative in original['files']:
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative, target)
    authoring = Path(__file__).parent
    for name in ('rules.py', 'evaluator.py'):
        text = (authoring/name).read_text().replace('from scenes.workrooms.living_room.', 'from ...')
        (output/'evaluator/tasks/tower_extraction'/name).write_text(text)

    task = load_json(output/'task.json')
    previous = task['evaluation']['reference_configuration']
    rules = previous['rules']
    current = task_configuration(previous['geometry'], seed=previous['seed'],
        max_turns_per_player=rules['max_turns_per_player'],
        protected_bottom_layers=rules['protected_bottom_layers'], thresholds=rules['thresholds'],
        collection_trays=previous.get('collection_trays'))
    protected = lambda value: {k: v for k, v in value.items() if k not in ('rules', 'configuration_sha256')}
    if protected(previous) != protected(current):
        raise ValueError('Termination-only derivation changed game geometry or reset')
    changed_rule_keys = {k for k in rules.keys() | current['rules'].keys() if rules.get(k) != current['rules'].get(k)}
    if changed_rule_keys - {'schema', 'host_termination', 'host_termination_scope', 'incomplete_without_winner'}:
        raise ValueError('Termination-only derivation changed physical or gameplay rules')
    if any(task['public_rules'].get(k) != v for k, v in rules.items()):
        raise ValueError('Published rules differ from the frozen game configuration')
    extras = {k: v for k, v in task['public_rules'].items() if k not in rules}
    task['evaluation']['reference_configuration'] = current
    task['public_rules'] = {**current['rules'], **extras}
    dump(output/'task.json', task)
    dump(output/'rules.json', task['public_rules'])

    scene = load_json(output/'scene.json')
    inventory = load_json(output/'asset_file_hashes.json')
    # Relocate the byte-identical stand file; do not add an old-release runtime
    # dependency solely because it happened to hold this physical source file.
    for layer in scene['task_layers']:
        if layer['usd_path'] == str(source/'robot_stands.usda'):
            previous_path = layer['usd_path']
            layer['usd_path'] = str(output/'robot_stands.usda')
            inventory[layer['usd_path']] = inventory.pop(previous_path)
    dump(output/'scene.json', scene)
    dump(output/'asset_file_hashes.json', inventory)
    derivation = {
        'source_release': str(source), 'source_manifest_sha256': source_manifest_sha256,
        'changed': 'Host interruption vocabulary and its frozen referee; no gameplay or physical changes.',
        'authoring_sources_sha256': {str(authoring/name): digest(authoring/name)
                                    for name in ('release.py', 'rules.py', 'evaluator.py', 'task.py')},
        'new_native_execution': False,
    }
    provenance = load_json(output/'source_provenance.json')
    provenance['host_termination_derivation'] = derivation
    dump(output/'source_provenance.json', provenance)
    manifest = deepcopy(original)
    manifest.update(release_id='tower_extraction_'+output.name,
                    prepared_at_utc=datetime.now(timezone.utc).isoformat(),
                    host_termination_derivation=derivation,
                    files={name: {'sha256': digest(output/name)} for name in original['files']},
                    asset_files_sha256={name: row['sha256'] for name, row in inventory.items()})
    manifest['files_sha256'] = {name: row['sha256'] for name, row in manifest['files'].items()}
    for entry in manifest['tasks']:
        for name in ('scene', 'task', 'initial_state'):
            entry[name+'_sha256'] = manifest['files_sha256'][entry[name]]
    for name in ('initial_state.json', 'physics_profile.json', 'robot_stands.usda'):
        if manifest['files_sha256'][name] != original['files_sha256'][name]:
            raise ValueError('Termination derivation changed a protected physical input')
    dump(output/'release_manifest.json', manifest)
    validate(output, assets=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    prepare_termination_release(args.source, args.output,
                                source_manifest_sha256=args.source_manifest_sha256)


if __name__ == '__main__':
    main()
