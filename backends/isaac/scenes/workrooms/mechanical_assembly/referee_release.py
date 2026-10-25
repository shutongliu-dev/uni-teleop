"""Version a rule/evaluator revision while preserving the exact physical scene."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump, validate
from .referee import bind_task_rules
from bench.paths import DATA_ROOT, load_json

DATASETS = Path(f'{DATA_ROOT}/datasets')
UNCHANGED = ('scene.json', 'initial_state.json', 'asset_file_hashes.json', 'physics_profile.json')


def revise_rules(source, output, *, source_manifest_sha256):
    """Never rebuild reset or geometry from current authoring defaults."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not output.is_relative_to(DATASETS) or output.exists():
        raise ValueError('Choose a fresh referee release under shared datasets')
    checked(source/'release_manifest.json', source_manifest_sha256)
    parent = validate(source)
    if parent.get('schema') not in ('mechanical_assembly_task_release.v1', 'mechanical_fastener_task_release.v1'):
        raise ValueError('Expected an independent frozen mechanical task')
    original = load_json(source/'task.json')
    task_id = original['id']
    base = Path(__file__).parent
    individual = task_id in ('fastener_assembly', 'gear_train_assembly')
    scoring = base/'tasks'/task_id if individual else base
    module_name = ('scenes.workrooms.mechanical_assembly.tasks.'+task_id if individual
                   else 'scenes.workrooms.mechanical_assembly')+'.rules'
    rules_module = importlib.import_module(module_name)
    args = ((original['public_rules']['success']['nominal_turn_degrees'],) if task_id == 'fastener_assembly'
            else () if task_id == 'gear_train_assembly' else (task_id,))
    rule_options = ({'reset_variant': original['public_rules'].get('reset_variant', 'preinserted')}
                    if task_id == 'fastener_assembly' else {})
    expected = rules_module.thresholds(*args)
    if any(original['evaluation'].get(key) != value for key, value in expected.items()):
        raise ValueError('A rule-only revision cannot change frozen physical or task thresholds')
    task = deepcopy(original)
    task['public_rules'] = rules_module.public_rules(*args, **rule_options)
    bind_task_rules(task)
    if task['evaluation'] != original['evaluation']:
        raise ValueError('A rule-only revision cannot change private task geometry or reset')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for relative in parent['files']:
            destination = staging/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/relative, destination)
            checked(destination, parent['files'][relative]['sha256'])
        dump(staging/'task.json', task)
        dump(staging/'rules.json', task['public_rules'])
        evaluator_name = 'evaluator.py' if individual else 'evaluation.py'
        replacements = {scoring/evaluator_name: 'evaluator/sequence_evaluator.py',
                        scoring/'rules.py': 'evaluator/rules.py', base/'referee.py': 'evaluator/referee.py'}
        if task_id != 'fastener_assembly':
            replacements[base/'native_evidence.py'] = 'evaluator/native_evidence.py'
        for path, relative in replacements.items():
            shutil.copyfile(path, staging/relative)
            checked(staging/relative, digest(path))
        provenance = load_json(source/'source_provenance.json')
        provenance.update(rule_revision={
            'parent_release': str(source), 'parent_manifest_sha256': source_manifest_sha256,
            'sources_sha256': {str(path): digest(path) for path in replacements},
            'physical_inputs_byte_identical': {name: parent['files'][name]['sha256'] for name in UNCHANGED},
            'evaluation_parameters_unchanged': True, 'new_native_qualification_claimed': False})
        dump(staging/'source_provenance.json', provenance)
        for name in UNCHANGED:
            checked(staging/name, parent['files'][name]['sha256'])
        manifest = deepcopy(parent)
        manifest.update(release_id=task_id+'_'+output.name,
            prepared_at_utc=datetime.now(timezone.utc).isoformat(),
            rule_parent_release={'path': str(source), 'manifest_sha256': source_manifest_sha256},
            qualification_scope='Rule/evaluator revision with byte-identical physical scene/reset/assets/profile. CPU contract checks are not native manipulation qualification.')
        files = {str(path.relative_to(staging)): {'sha256': digest(path)}
                 for path in sorted(staging.rglob('*')) if path.is_file()}
        manifest['files'] = files
        manifest['files_sha256'] = {name: row['sha256'] for name, row in files.items()}
        for name in ('scene', 'task', 'initial_state'):
            manifest['tasks'][0][name+'_sha256'] = files[name+'.json']['sha256']
        dump(staging/'release_manifest.json', manifest)
        validate(staging)
        checked(source/'release_manifest.json', source_manifest_sha256)
        for relative, row in parent['files'].items():
            checked(source/relative, row['sha256'])
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
