"""Derive the reviewed budget repair without migrating any native observer logic.

Only the exact historical SequenceEvaluator class is replaced. Everything
outside that class, all helpers, task rules, assets and physics stay frozen.
Development outcomes are deliberately absent from the resulting task inputs.
"""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump, validate
from . import evaluation
from .release import DATASETS, SCHEMA, validate_release
from bench.paths import load_json

OLD_SCORER_SHA256 = 'cad825ce4ab1a96b578faa3034196302ac6bcdbc623cfc5c41249395c6d979c2'


def _class_span(source):
    classes = [node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == 'SequenceEvaluator']
    if len(classes) != 1 or classes[0].decorator_list:
        raise ValueError('Expected one undecorated frozen SequenceEvaluator')
    node = classes[0]
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1]), ast.get_source_segment(source, node), ''.join(lines[node.end_lineno:])


def repaired_evaluator_source(source):
    before, old, after = _class_span(source)
    if hashlib.sha256(old.encode()).hexdigest() != OLD_SCORER_SHA256:
        raise ValueError('Unreviewed historical laboratory scorer')
    _, current, _ = _class_span(Path(evaluation.__file__).read_text())
    # Keep the complete pre-class observer and any subsequent helpers verbatim.
    return before + current + '\n' + after


def prepare_budget_release(source, output):
    """Create a new single-arm base; dual options derive through their publisher."""
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    project = Path(__file__).resolve().parents[3]
    snapshot = verify_if_snapshot(project, required_modules=(__name__, evaluation.__name__),
                                  entrypoint_file=Path(__file__))
    if snapshot is None:
        raise ValueError('Prepare the repair from complete frozen project source')
    source, output = Path(source).resolve(), Path(output).resolve()
    original = validate(source)
    if original.get('schema') != SCHEMA or output.exists() or not output.is_relative_to(DATASETS):
        raise ValueError('Choose a fresh laboratory base release under datasets')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for relative, row in original['files'].items():
            checked(source/relative, row['sha256'])
            target = staging/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/relative, target)
            target.chmod(0o644)
        scorer = staging/'evaluator/sequence_evaluator.py'
        scorer.write_text(repaired_evaluator_source(scorer.read_text()))
        scene = load_json(source/'scene.json')
        for name in ('observer', 'reduced_dynamics'):
            hook = scene['native_task_bundle'].get(name)
            if hook is not None:
                hook['files']['sequence_evaluator.py'] = digest(scorer)
                hook['sha256'] = digest(staging/hook['path'])
        dump(staging/'scene.json', scene)
        dump(staging/'source_provenance.json', {
            'source_snapshot': snapshot, 'source_root': str(project),
            'sources_sha256': {str(p): digest(p) for p in (Path(__file__), Path(evaluation.__file__))},
            'budget_evaluator_derivation': {
                'base_release': {'path': str(source), 'manifest_sha256': digest(source/'release_manifest.json')},
                'old_scorer_class_sha256': OLD_SCORER_SHA256,
                'scorer_file_sha256': digest(scorer),
                'scope': 'Terminal budget envelope, deadline-before-success and incomplete latching only.',
                'native_observer_unchanged': True, 'task_rules_unchanged': True,
            },
            'parent_environment': scene['workroom'], 'original_assets_modified': False,
            'original_releases_modified': False, 'shared_controller_unchanged': True,
        })
        files = {str(p.relative_to(staging)): {'sha256': digest(p)}
                 for p in sorted(staging.rglob('*')) if p.is_file()}
        manifest = deepcopy(original)
        manifest.update(release_id='laboratory_'+manifest['task_id']+'_'+output.name,
                        prepared_at_utc=datetime.now(timezone.utc).isoformat(), files=files,
                        files_sha256={name: row['sha256'] for name, row in files.items()})
        manifest['tasks'][0]['scene_sha256'] = files['scene.json']['sha256']
        changed = {name for name, row in files.items() if row != original['files'][name]}
        if changed != {'scene.json', 'source_provenance.json', 'evaluator/sequence_evaluator.py'}:
            raise ValueError('Budget repair changed an unrelated release input')
        dump(staging/'release_manifest.json', manifest)
        validate_release(staging, manifest)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
