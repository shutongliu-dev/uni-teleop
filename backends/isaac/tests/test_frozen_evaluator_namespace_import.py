"""Historical evaluator namespaces retain the source-only import boundary."""
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
from types import ModuleType

import pytest

from bench.isaacsim.frozen_import import _FrozenFinder
from bench.isaacsim.release import _frozen_evaluator_module


def release(root):
    evaluator = root / 'evaluator'
    evaluator.mkdir(parents=True)
    (root / 'release_manifest.json').write_text(json.dumps({'test_identity': str(root)}))
    return evaluator


def test_nested_namespace_helpers_load_source_without_writing_bytecode(tmp_path):
    root = tmp_path / 'release'
    evaluator = release(root)
    task = evaluator / 'tasks' / 'tower_building'
    task.mkdir(parents=True)
    (evaluator / 'sequence_evaluator.py').write_text(
        'from .tasks.tower_building.evaluator import SequenceEvaluator\n')
    (task / 'evaluator.py').write_text(
        'from .rules import VALUE\nclass SequenceEvaluator:\n    value = VALUE\n')
    rules = task / 'rules.py'
    rules.write_text('VALUE = 99\n')
    stale = py_compile.compile(str(rules), doraise=True)
    # Keep size and timestamp identical: ordinary bytecode loading would use 99.
    original_stat = rules.stat()
    rules.write_text('VALUE = 42\n')
    os.utime(rules, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert Path(stale).is_file()
    ordinary = tmp_path / 'ordinary'
    ordinary.mkdir()
    (ordinary / 'ordinary_module.py').write_text('VALUE = 17\n')
    environment = os.environ.copy()
    environment.pop('PYTHONDONTWRITEBYTECODE', None)
    environment.pop('PYTHONPYCACHEPREFIX', None)
    script = '''
import sys
from pathlib import Path
from bench.isaacsim.release import _frozen_evaluator_module
sys.dont_write_bytecode = False
root = Path(sys.argv[1])
module = _frozen_evaluator_module(root)
assert module.SequenceEvaluator.value == 42
rules = sys.modules[module.__package__ + '.tasks.tower_building.rules']
assert type(rules.__loader__).__name__ == '_SourceOnlyLoader'
sys.path.insert(0, sys.argv[2])
import ordinary_module
assert ordinary_module.VALUE == 17
assert list(Path(sys.argv[2]).rglob('*.pyc'))
'''
    result = subprocess.run([sys.executable, '-c', script, str(root), str(ordinary)],
        cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert after == before
    assert not list(root.rglob('__init__.py'))


@pytest.mark.parametrize('mixed', [False, True])
def test_namespace_locations_reject_escaped_or_mixed_symlinks(tmp_path, monkeypatch, mixed):
    root, outside = tmp_path / 'frozen', tmp_path / 'outside'
    inside, linked = root / 'inside', root / 'linked'
    for path in (inside / 'tasks', linked, outside):
        path.mkdir(parents=True)
    (linked / 'tasks').symlink_to(outside, target_is_directory=True)
    name = '_isaac_frozen_evaluator_namespace_boundary'
    package = ModuleType(name)
    package.__path__ = [str(inside), str(linked)] if mixed else [str(linked)]
    monkeypatch.setitem(sys.modules, name, package)
    finder = _FrozenFinder()
    finder.roots[name] = root.resolve()
    with pytest.raises(ImportError, match='namespace escaped'):
        finder.find_spec(name + '.tasks', package.__path__)


def test_namespace_child_still_rejects_sourceless_bytecode(tmp_path):
    root = tmp_path / 'release'
    evaluator = release(root)
    task = evaluator / 'tasks' / 'tower_building'
    task.mkdir(parents=True)
    (evaluator / 'sequence_evaluator.py').write_text(
        'from .tasks.tower_building.rules import VALUE\nclass SequenceEvaluator:\n    value = VALUE\n')
    rules = task / 'rules.py'
    rules.write_text('VALUE = 99\n')
    py_compile.compile(str(rules), cfile=str(task / 'rules.pyc'), doraise=True)
    rules.unlink()
    with pytest.raises(ImportError, match='validated Python source'):
        _frozen_evaluator_module(root)
