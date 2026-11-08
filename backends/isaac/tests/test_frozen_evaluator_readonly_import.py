import json
import os
from pathlib import Path
import subprocess
import sys


def test_frozen_evaluator_and_relative_helpers_never_use_bytecode(tmp_path):
    root = tmp_path/'release'
    evaluator = root/'evaluator'
    evaluator.mkdir(parents=True)
    (root/'release_manifest.json').write_text(json.dumps({'test_identity': str(root)}))
    (evaluator/'sequence_evaluator.py').write_text('from .rules import VALUE\nclass SequenceEvaluator:\n    value = VALUE\n')
    (evaluator/'rules.py').write_text('from .nested import EXTRA\nVALUE = 41 + EXTRA\n')
    (evaluator/'nested.py').write_text('EXTRA = 1\n')
    ordinary = tmp_path/'ordinary'
    ordinary.mkdir()
    (ordinary/'ordinary_module.py').write_text('VALUE = 17\n')
    environment = os.environ.copy()
    environment.pop('PYTHONDONTWRITEBYTECODE', None)
    environment.pop('PYTHONPYCACHEPREFIX', None)
    script = '''
import sys
from pathlib import Path
from bench.isaacsim.release import _frozen_evaluator_module
sys.dont_write_bytecode = False
root = Path(sys.argv[1])
assert _frozen_evaluator_module(root).SequenceEvaluator.value == 42
assert not list(root.rglob('*.pyc'))
sys.path.insert(0, sys.argv[2])
import ordinary_module
assert ordinary_module.VALUE == 17
assert list(Path(sys.argv[2]).rglob('*.pyc'))
'''
    # Explicitly start without -B; this is the publication failure being fixed.
    result = subprocess.run([sys.executable, '-c', script, str(root), str(ordinary)],
        cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(root.rglob('__pycache__'))
