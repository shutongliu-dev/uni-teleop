"""Frozen evaluator identity checks, with temporary CPU fixtures only."""
import json
from pathlib import Path
import shutil

import pytest

from scenes.board_games import tabletop
from scenes.board_games.tabletop import geometry_for, prepare_task_release, task_specification, verify_evaluator_sources


def frozen_fixture(tmp_path):
    source = Path(tabletop.__file__).parent
    frozen = tmp_path/'evaluator'/'tabletop'
    frozen.mkdir(parents=True)
    for path in source.glob('*.py'):
        shutil.copy2(path, frozen/path.name)
    task = task_specification(geometry_for('connect_four_5x5'))
    (tmp_path/'task.json').write_text(json.dumps(task))
    return task, frozen


def test_exact_frozen_and_imported_evaluator_inventory(tmp_path):
    task, _ = frozen_fixture(tmp_path)
    report = verify_evaluator_sources(tmp_path)
    assert report['all_import_paths_verified']
    assert report['sources_sha256'] == task['sources_sha256']


@pytest.mark.parametrize('mutation', ['changed', 'missing', 'extra'])
def test_frozen_source_mutations_rejected(tmp_path, mutation):
    task, frozen = frozen_fixture(tmp_path)
    if mutation == 'changed':
        with (frozen/'settlement.py').open('a') as stream:
            stream.write('\n# mutation\n')
    elif mutation == 'missing':
        (frozen/'settlement.py').unlink()
    else:
        (frozen/'unexpected.py').write_text('')
    with pytest.raises(ValueError, match='source inventory'):
        verify_evaluator_sources(tmp_path, task)


def test_module_from_unverified_path_rejected(tmp_path, monkeypatch):
    task, _ = frozen_fixture(tmp_path)
    monkeypatch.setattr(tabletop.rules, '__file__', str(tmp_path/'rules.py'))
    with pytest.raises(ValueError, match='outside the verified package'):
        verify_evaluator_sources(tmp_path, task)


def test_release_requires_shared_dataset_location(tmp_path):
    with pytest.raises(ValueError, match='prepared under'):
        prepare_task_release(tmp_path/'release', scene={}, geometry=geometry_for('tic_tac_toe_3x3'))
