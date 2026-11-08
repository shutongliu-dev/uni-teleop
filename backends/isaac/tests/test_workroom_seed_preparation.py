"""Regeneration must keep the original physical parents after publication."""
from pathlib import Path

import pytest

from scenes.workrooms.initial_randomization import TASK_FAMILIES, configurations
from scenes.workrooms.initial_randomization.catalog import PINS
from scenes.workrooms.initial_randomization.common import read_json
from scripts.assets.prepare_workroom_randomization import original_task_definition


@pytest.mark.parametrize('task_id,family', [(task_id, TASK_FAMILIES[task_id]) for task_id in read_json(PINS)])
def test_regeneration_uses_exact_original_published_parents(task_id, family):
    pin = read_json(PINS)[task_id]
    definition = original_task_definition(family, task_id)
    assert definition['source_release'] == pin['parent_default_release']
    options = definition.get('robot_configurations') or {configurations(task_id)[0]: definition}
    assert set(options) == set(pin['releases'])
    for configuration, option in options.items():
        reference = option['source_release']
        assert reference == pin['parents'][configuration]
        published = read_json(Path(pin['releases'][configuration]['path'])/'release_manifest.json')
        assert reference['path'] == published['base_release']['path']
        assert reference['manifest_sha256'] == published['base_release']['manifest_sha256']
        assert reference != pin['releases'][configuration]
